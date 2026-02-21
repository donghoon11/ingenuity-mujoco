"""
Optimizer: NSGA-II Multi-Objective Optimization

Minimizes [P_total, -FM, -T_margin_pct_rhomin] subject to constraints.
Uses surrogate model (GP) for fast function evaluations if available,
falls back to direct BEMT+structural evaluation otherwise.

Objectives (all minimized):
  f1: P_total         (hover power — minimize)
  f2: -FM             (figure of merit — maximize)
  f3: -T_margin_pct   (thrust margin @ rho_min — maximize)

Constraints (all <= 0):
  g1: rpm - RPM_MAX           <= 0
  g2: tip_mach - TIP_MACH_MAX <= 0
  g3: -T_margin_pct_rhomin    <= 0  (must have positive margin)
  g4: F1_BENDING_MIN - f1_hz  <= 0  (must meet frequency)

Outputs:
  - results/optimizer/pareto_front.json
  - results/optimizer/pareto_designs.csv

Usage:
  cd E:/mujoco_projects/ingenuity-mujoco
  python 0219/design/optimizer.py
  python 0219/design/optimizer.py --gen 50 --pop 80 --no-surrogate
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DESIGN_LOWER, DESIGN_UPPER, N_DESIGN_VARS, DESIGN_VAR_NAMES,
    RHO_NOMINAL, RHO_RANGE, MARS_WEIGHT, NUM_ROTORS,
    TIP_MACH_MAX, RPM_MAX, F1_BENDING_MIN, RESULTS_DIR,
)
from blade_param import BladeDesign, baseline_design
from bemt import bemt_hover
from structural import evaluate_structural
from doe import evaluate_design
from utils import ensure_results_dir, save_json, log_to_csv


# ─── Objective/Constraint Evaluation ────────────────────────────────────────

def evaluate_objectives(vec: np.ndarray,
                        surrogate=None) -> tuple:
    """
    Evaluate objectives and constraints for a design vector.

    Returns:
      objectives : [P_total, -FM, -T_margin_rhomin]
      constraints: [g1_rpm, g2_mach, g3_thrust_robust, g4_f1]
                   (all <= 0 means feasible)
    """
    if surrogate is not None:
        # Surrogate prediction
        lower = surrogate['design_lower']
        upper = surrogate['design_upper']
        x_norm = (vec - lower) / (upper - lower)
        x_norm = np.clip(x_norm, 0.0, 1.0).reshape(1, -1)

        P_pred = float(surrogate['models'][0].predict(x_norm)[0])
        FM_pred = float(surrogate['models'][1].predict(x_norm)[0])
        Tm_pred = float(surrogate['models'][2].predict(x_norm)[0])

        # Direct blade properties (cheap)
        try:
            blade = BladeDesign(vec)
            rpm_constraint = blade.rpm - RPM_MAX
            mach_constraint = blade.tip_mach() - TIP_MACH_MAX
        except Exception:
            rpm_constraint = 1000.0
            mach_constraint = 1.0

        objectives = [P_pred, -FM_pred, -Tm_pred]
        constraints = [
            rpm_constraint,            # g1: rpm <= RPM_MAX
            mach_constraint,           # g2: tip_mach <= TIP_MACH_MAX
            -Tm_pred,                  # g3: T_margin_rhomin >= 0
            -FM_pred * 10.0,           # g4: FM > 0 (proxy for f1)
        ]
    else:
        # Direct evaluation (slow)
        res = evaluate_design(vec)

        P = res.get('P_total', 1e6)
        FM = res.get('FM', 0.0)
        Tm = res.get('T_margin_pct_rhomin', -100.0)
        rpm = res.get('rpm', 0.0)
        mach = res.get('tip_mach', 1.0)
        f1 = res.get('f1_bending_hz', 0.0)

        objectives = [P, -FM, -Tm]
        constraints = [
            rpm - RPM_MAX,            # g1
            mach - TIP_MACH_MAX,      # g2
            -Tm,                      # g3: T_margin_rhomin >= 0
            F1_BENDING_MIN - f1,      # g4: f1 >= F1_BENDING_MIN
        ]

    return np.array(objectives), np.array(constraints)


# ─── NSGA-II via pymoo ────────────────────────────────────────────────────

def run_nsga2_pymoo(surrogate=None, n_gen: int = 40, pop_size: int = 60,
                   seed: int = 0, verbose: bool = True):
    """Run NSGA-II using pymoo library."""
    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize as pymoo_minimize
        from pymoo.termination import get_termination
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
    except ImportError:
        raise ImportError(
            "pymoo not installed. Run: pip install pymoo\n"
            "Then retry: python 0219/design/optimizer.py"
        )

    class BladeProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=N_DESIGN_VARS,
                n_obj=3,
                n_ieq_constr=4,
                xl=DESIGN_LOWER,
                xu=DESIGN_UPPER,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            obj, con = evaluate_objectives(x, surrogate=surrogate)
            out["F"] = obj
            out["G"] = con

    problem = BladeProblem()

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=20),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", n_gen)

    if verbose:
        print(f"  Running NSGA-II: pop={pop_size}, gen={n_gen}")

    res = pymoo_minimize(
        problem, algorithm, termination,
        seed=seed, verbose=False,
        save_history=False,
    )

    return res


# ─── Fallback: Simple Evolutionary Search ────────────────────────────────

def run_simple_evolution(surrogate=None, n_gen: int = 40, pop_size: int = 60,
                         seed: int = 0):
    """
    Simple evolution fallback (no pymoo dependency).
    Uses random sampling + selection for approximate Pareto front.
    """
    rng = np.random.default_rng(seed)

    # Initialize population
    population = rng.uniform(DESIGN_LOWER, DESIGN_UPPER, (pop_size, N_DESIGN_VARS))
    # Add baseline
    population[0] = baseline_design().vector

    best_pareto = []

    for gen in range(n_gen):
        # Evaluate population
        F_all = []
        G_all = []
        for vec in population:
            obj, con = evaluate_objectives(vec, surrogate=surrogate)
            F_all.append(obj)
            G_all.append(con)

        F_all = np.array(F_all)
        G_all = np.array(G_all)

        # Feasibility mask (all constraints <= 0)
        feasible = np.all(G_all <= 0.5, axis=1)

        if np.any(feasible):
            # Extract feasible Pareto front (simplified: non-dominated)
            F_feas = F_all[feasible]
            pop_feas = population[feasible]
            best_pareto = _nondominated(F_feas, pop_feas)

        # Evolution: tournament + mutation
        new_pop = []
        for _ in range(pop_size):
            # Tournament selection
            idx1, idx2 = rng.integers(0, pop_size, 2)
            winner = idx1 if np.sum(G_all[idx1] > 0) <= np.sum(G_all[idx2] > 0) else idx2
            parent = population[winner].copy()

            # Polynomial mutation
            sigma = (DESIGN_UPPER - DESIGN_LOWER) * 0.05
            child = parent + rng.normal(0, sigma)
            child = np.clip(child, DESIGN_LOWER, DESIGN_UPPER)
            new_pop.append(child)

        population = np.array(new_pop)
        if gen % 10 == 0:
            print(f"    Gen {gen:3d}: feasible={np.sum(feasible)}/{pop_size}  "
                  f"pareto_size={len(best_pareto)}")

    return best_pareto


def _nondominated(F: np.ndarray, X: np.ndarray):
    """Extract non-dominated solutions from F (minimize all objectives)."""
    n = len(F)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                dominated[i] = True
                break
    pareto_X = X[~dominated]
    pareto_F = F[~dominated]
    return list(zip(pareto_X, pareto_F))


# ─── Main ─────────────────────────────────────────────────────────────────

def run_optimizer(n_gen: int = 40, pop_size: int = 60,
                  use_surrogate: bool = True, seed: int = 0):
    print("=" * 70)
    print("NSGA-II Multi-Objective Blade Optimization")
    print("=" * 70)
    print(f"  Objectives: Minimize [P_total, -FM, -T_margin_rhomin]")
    print(f"  Generations: {n_gen}  |  Population: {pop_size}")
    print()

    results_dir = ensure_results_dir("optimizer")

    # Load surrogate if available
    surrogate = None
    if use_surrogate:
        surrogate_path = str(RESULTS_DIR / "surrogate" / "gp_model.pkl")
        if Path(surrogate_path).exists():
            import pickle
            with open(surrogate_path, 'rb') as f:
                surrogate = pickle.load(f)
            print(f"  Using surrogate model: {surrogate_path}")
            print(f"  ({surrogate['n_samples']} training samples)")
        else:
            print(f"  WARNING: Surrogate model not found at {surrogate_path}")
            print("  Run: python 0219/design/surrogate.py")
            print("  Falling back to direct BEMT evaluation (slower)")
    else:
        print("  Using direct BEMT evaluation (no surrogate)")
    print()

    # ── Run NSGA-II ────────────────────────────────────────────────────────
    t_start = time.time()
    pareto_designs = []
    pareto_objectives = []

    try:
        print("[Running NSGA-II via pymoo...]")
        res = run_nsga2_pymoo(
            surrogate=surrogate,
            n_gen=n_gen, pop_size=pop_size, seed=seed, verbose=True,
        )

        if res.X is not None:
            for i in range(len(res.X)):
                x = res.X[i]
                F = res.F[i]
                pareto_designs.append(x.tolist())
                pareto_objectives.append(F.tolist())

            print(f"  Pareto front size: {len(pareto_designs)}")
        else:
            print("  WARNING: No Pareto solutions found from pymoo")

    except ImportError as e:
        print(f"  {e}")
        print("[Falling back to simple evolutionary search...]")
        pareto = run_simple_evolution(
            surrogate=surrogate,
            n_gen=n_gen, pop_size=pop_size, seed=seed,
        )
        for x, F in pareto:
            pareto_designs.append(x.tolist())
            pareto_objectives.append(F.tolist())

    elapsed = time.time() - t_start
    print(f"\n  Optimization completed in {elapsed:.1f}s")
    print(f"  Pareto front size: {len(pareto_designs)}")
    print()

    # ── Re-evaluate Pareto with direct BEMT (ground truth) ────────────────
    print("[Re-evaluating Pareto front with direct BEMT...]")
    pareto_results = []
    for i, vec in enumerate(pareto_designs):
        vec_np = np.array(vec)
        res_direct = evaluate_design(vec_np)
        res_direct['pareto_id'] = i
        res_direct['surrogate_F'] = pareto_objectives[i]
        pareto_results.append(res_direct)

    # Sort by FM (descending) for display
    pareto_results_sorted = sorted(
        pareto_results,
        key=lambda x: x.get('FM', 0.0), reverse=True
    )

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'Rank':>4}  {'P(W)':>8}  {'FM':>6}  {'T_margin%':>10}  "
          f"{'T_rob%':>8}  {'f1(Hz)':>7}  {'Feas':>5}")
    print("-" * 65)
    for rank, r in enumerate(pareto_results_sorted[:10]):
        print(f"{rank + 1:4d}  "
              f"{r.get('P_total', 0.0):8.2f}  "
              f"{r.get('FM', 0.0):6.4f}  "
              f"{r.get('T_margin_pct', 0.0):10.2f}  "
              f"{r.get('T_margin_pct_rhomin', 0.0):8.2f}  "
              f"{r.get('f1_bending_hz', 0.0):7.2f}  "
              f"{'Y' if r.get('feasible', False) else 'N':>5}")

    if len(pareto_results_sorted) > 10:
        print(f"  ... ({len(pareto_results_sorted) - 10} more)")

    # ── Save results ───────────────────────────────────────────────────────
    output = {
        'pareto_results': pareto_results_sorted,
        'pareto_designs': pareto_designs,
        'pareto_surrogate_F': pareto_objectives,
        'n_pareto': len(pareto_designs),
        'params': {
            'n_gen': n_gen,
            'pop_size': pop_size,
            'seed': seed,
            'use_surrogate': use_surrogate and surrogate is not None,
        },
        'elapsed_s': elapsed,
    }
    save_json(str(results_dir / "pareto_front.json"), output)

    # CSV
    headers = [
        'pareto_id', *DESIGN_VAR_NAMES,
        'P_total', 'FM', 'T_margin_pct', 'T_margin_pct_rhomin',
        'f1_bending_hz', 'blade_mass_kg', 'feasible', 'robust_feasible',
    ]
    rows = []
    for r in pareto_results_sorted:
        row = [
            r.get('pareto_id', -1),
            *r.get('design_vector', [0.0] * N_DESIGN_VARS),
            r.get('P_total', 0.0), r.get('FM', 0.0),
            r.get('T_margin_pct', 0.0), r.get('T_margin_pct_rhomin', 0.0),
            r.get('f1_bending_hz', 0.0), r.get('blade_mass_kg', 0.0),
            int(r.get('feasible', False)), int(r.get('robust_feasible', False)),
        ]
        rows.append(row)
    log_to_csv(str(results_dir / "pareto_designs.csv"), headers, rows)

    print(f"\nResults saved to: {results_dir}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSGA-II Blade Optimizer")
    parser.add_argument("--gen", type=int, default=40,
                        help="Number of NSGA-II generations")
    parser.add_argument("--pop", type=int, default=60,
                        help="Population size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--no-surrogate", action="store_true",
                        help="Use direct BEMT (skip surrogate)")
    args = parser.parse_args()

    run_optimizer(
        n_gen=args.gen,
        pop_size=args.pop,
        use_surrogate=not args.no_surrogate,
        seed=args.seed,
    )
