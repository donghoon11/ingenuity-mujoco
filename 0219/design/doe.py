"""
DOE: Latin Hypercube Sampling + BEMT/Structural Evaluation

Generates N design samples using Latin Hypercube Sampling (LHS)
and evaluates each with BEMT aerodynamics and structural constraints.
MuJoCo simulation is NOT used here (fast batch evaluation).

Outputs:
  - results/doe/doe_samples.csv      (12D design vectors)
  - results/doe/doe_results.json     (all KPIs per sample)
  - results/doe/doe_summary.csv      (samples + KPIs in one file)
  - Console progress

Usage:
  cd E:/mujoco_projects/ingenuity-mujoco
  python 0219/design/doe.py
  python 0219/design/doe.py --n 200 --seed 42
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
    TIP_MACH_MAX, RPM_MAX, F1_BENDING_MIN, RESONANCE_MARGIN,
)
from blade_param import BladeDesign, baseline_design
from bemt import bemt_hover
from structural import evaluate_structural
from utils import ensure_results_dir, save_json, log_to_csv


def lhs_sample(n: int, n_vars: int, lower: np.ndarray, upper: np.ndarray,
               seed: int = 0) -> np.ndarray:
    """
    Latin Hypercube Sampling via scipy.stats.qmc.LatinHypercube.

    Returns array of shape (n, n_vars) in physical units.
    """
    try:
        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=n_vars, seed=seed)
        unit_samples = sampler.random(n)
    except ImportError:
        # Fallback: simple random with stratification
        rng = np.random.default_rng(seed)
        unit_samples = np.zeros((n, n_vars))
        for j in range(n_vars):
            perm = rng.permutation(n)
            unit_samples[:, j] = (perm + rng.uniform(size=n)) / n

    # Scale to physical bounds
    samples = lower + unit_samples * (upper - lower)
    return samples


def evaluate_design(vec: np.ndarray, rho: float = RHO_NOMINAL) -> dict:
    """
    Evaluate a single blade design vector.

    Returns dict with:
      - aerodynamic KPIs (BEMT)
      - structural KPIs
      - feasibility flags
    """
    try:
        blade = BladeDesign(vec)
    except Exception as e:
        return {'error': str(e), 'feasible': False}

    result = {
        'design_vector': vec.tolist(),
        'tip_mach': blade.tip_mach(),
        'solidity': blade.solidity(),
        'rpm': blade.rpm,
    }

    # ── BEMT @ nominal rho ────────────────────────────────────────────────
    try:
        bemt_nom = bemt_hover(blade, rho=rho)
        T_req = MARS_WEIGHT / NUM_ROTORS
        T_margin = (bemt_nom['T_per_rotor'] / T_req - 1.0) * 100.0

        result.update({
            'T_per_rotor': bemt_nom['T_per_rotor'],
            'P_total': bemt_nom['P'],
            'FM': bemt_nom['FM'],
            'CT': bemt_nom['CT'],
            'CP': bemt_nom['CP'],
            'T_margin_pct': T_margin,
            'ctrl_thrust': bemt_nom['ctrl_thrust'],
            'bemt_converged': bemt_nom['converged'],
        })
    except Exception as e:
        result.update({
            'T_per_rotor': 0.0, 'P_total': float('inf'), 'FM': 0.0,
            'CT': 0.0, 'CP': 0.0, 'T_margin_pct': -100.0,
            'ctrl_thrust': 0.0, 'bemt_converged': False,
            'bemt_error': str(e),
        })

    # ── BEMT @ rho_min (worst case) ───────────────────────────────────────
    try:
        rho_min = RHO_RANGE[0]
        bemt_min = bemt_hover(blade, rho=rho_min)
        T_req_min = MARS_WEIGHT / NUM_ROTORS
        T_margin_min = (bemt_min['T_per_rotor'] / T_req_min - 1.0) * 100.0
        result.update({
            'T_per_rotor_rhomin': bemt_min['T_per_rotor'],
            'P_total_rhomin': bemt_min['P'],
            'FM_rhomin': bemt_min['FM'],
            'T_margin_pct_rhomin': T_margin_min,
        })
    except Exception:
        result.update({
            'T_per_rotor_rhomin': 0.0,
            'P_total_rhomin': float('inf'),
            'FM_rhomin': 0.0,
            'T_margin_pct_rhomin': -100.0,
        })

    # ── Structural evaluation ─────────────────────────────────────────────
    try:
        struct = evaluate_structural(blade)
        result.update({
            'f1_bending_hz': struct['f1_bending'],
            'f_rev_hz': struct['f_rev'],
            'blade_mass_kg': struct['blade_mass'],
            'total_rotor_mass_kg': struct['total_rotor_mass'],
            'f1_feasible': struct['f1_feasible'],
            'resonance_feasible': struct['resonance_feasible'],
            'structural_feasible': struct['feasible'],
        })
    except Exception as e:
        result.update({
            'f1_bending_hz': 0.0, 'f_rev_hz': 0.0,
            'blade_mass_kg': 0.0, 'total_rotor_mass_kg': 0.0,
            'f1_feasible': False, 'resonance_feasible': False,
            'structural_feasible': False,
            'struct_error': str(e),
        })

    # ── Aerodynamic constraints ───────────────────────────────────────────
    rpm_ok = blade.rpm <= RPM_MAX
    mach_ok = blade.tip_mach() <= TIP_MACH_MAX
    hover_ok = result.get('T_margin_pct', -100.0) >= 0.0
    hover_robust = result.get('T_margin_pct_rhomin', -100.0) >= 0.0

    result['rpm_feasible'] = rpm_ok
    result['mach_feasible'] = mach_ok
    result['hover_feasible'] = hover_ok
    result['hover_robust_feasible'] = hover_robust
    result['feasible'] = (
        rpm_ok and mach_ok and hover_ok
        and result.get('structural_feasible', False)
    )
    result['robust_feasible'] = (
        rpm_ok and mach_ok and hover_robust
        and result.get('structural_feasible', False)
    )

    return result


def run_doe(n_samples: int = 100, seed: int = 0,
            rho: float = RHO_NOMINAL) -> dict:
    print("=" * 70)
    print(f"DOE: Latin Hypercube Sampling (N={n_samples}, seed={seed})")
    print("=" * 70)
    print(f"  Design space: {N_DESIGN_VARS}D")
    print(f"  Variables: {DESIGN_VAR_NAMES}")
    print(f"  Evaluation: BEMT (rho={rho}) + Structural")
    print()

    results_dir = ensure_results_dir("doe")

    # Add baseline as first sample
    baseline = baseline_design()
    baseline_vec = baseline.vector

    # LHS samples
    print("[Generating LHS samples...]")
    lhs_vectors = lhs_sample(n_samples - 1, N_DESIGN_VARS,
                              DESIGN_LOWER, DESIGN_UPPER, seed=seed)

    # Prepend baseline
    all_vectors = np.vstack([baseline_vec.reshape(1, -1), lhs_vectors])
    n_total = len(all_vectors)

    print(f"  Total samples: {n_total} (1 baseline + {n_total - 1} LHS)")
    print()

    # ── Evaluate all samples ───────────────────────────────────────────────
    print("[Evaluating designs (BEMT + Structural)...]")
    print(f"{'#':>5}  {'T/rot(N)':>9}  {'P(W)':>8}  {'FM':>6}  "
          f"{'f1(Hz)':>7}  {'Feas':>5}  {'Robust':>7}")
    print("-" * 65)

    all_results = []
    n_feasible = 0
    n_robust = 0
    t_start = time.time()

    for i, vec in enumerate(all_vectors):
        is_baseline = (i == 0)
        res = evaluate_design(vec, rho=rho)
        res['sample_id'] = i
        res['is_baseline'] = is_baseline
        all_results.append(res)

        feas = res.get('feasible', False)
        robust = res.get('robust_feasible', False)
        if feas:
            n_feasible += 1
        if robust:
            n_robust += 1

        if i % 10 == 0 or is_baseline or i == n_total - 1:
            T_r = res.get('T_per_rotor', 0.0)
            P_t = res.get('P_total', float('inf'))
            FM = res.get('FM', 0.0)
            f1 = res.get('f1_bending_hz', 0.0)
            tag = " [BL]" if is_baseline else ""
            print(f"{i:5d}  {T_r:9.3f}  {P_t:8.2f}  {FM:6.4f}  "
                  f"{f1:7.2f}  {'Y' if feas else 'N':>5}  "
                  f"{'Y' if robust else 'N':>7}{tag}")

    elapsed = time.time() - t_start
    print("-" * 65)
    print(f"  Evaluated {n_total} samples in {elapsed:.1f}s")
    print(f"  Feasible: {n_feasible}/{n_total} ({n_feasible / n_total * 100:.1f}%)")
    print(f"  Robust feasible: {n_robust}/{n_total} ({n_robust / n_total * 100:.1f}%)")
    print()

    # ── Best designs ────────────────────────────────────────────────────
    feasible_results = [r for r in all_results if r.get('feasible', False)]
    if feasible_results:
        # Best FM
        best_fm = max(feasible_results, key=lambda x: x.get('FM', 0.0))
        # Best power (min)
        best_pow = min(feasible_results, key=lambda x: x.get('P_total', float('inf')))
        # Best T margin
        best_tm = max(feasible_results, key=lambda x: x.get('T_margin_pct', -999.0))

        print("[Best feasible designs]")
        print(f"  Best FM:      #{best_fm['sample_id']}  "
              f"FM={best_fm['FM']:.4f}  P={best_fm['P_total']:.2f}W  "
              f"T_margin={best_fm['T_margin_pct']:+.1f}%")
        print(f"  Min power:    #{best_pow['sample_id']}  "
              f"P={best_pow['P_total']:.2f}W  FM={best_pow['FM']:.4f}  "
              f"T_margin={best_pow['T_margin_pct']:+.1f}%")
        print(f"  Best T margin:#{best_tm['sample_id']}  "
              f"T_margin={best_tm['T_margin_pct']:+.1f}%  "
              f"FM={best_tm['FM']:.4f}")
        print()

    # ── Save ────────────────────────────────────────────────────────────
    doe_output = {
        'n_samples': n_total,
        'n_feasible': n_feasible,
        'n_robust': n_robust,
        'seed': seed,
        'rho': rho,
        'results': all_results,
        'design_var_names': DESIGN_VAR_NAMES,
        'design_lower': DESIGN_LOWER.tolist(),
        'design_upper': DESIGN_UPPER.tolist(),
    }
    save_json(str(results_dir / "doe_results.json"), doe_output)

    # CSV: samples
    sample_headers = DESIGN_VAR_NAMES
    sample_rows = [r['design_vector'] for r in all_results]
    log_to_csv(str(results_dir / "doe_samples.csv"), sample_headers, sample_rows)

    # CSV: summary (design + KPIs)
    kpi_headers = [
        'sample_id', 'is_baseline',
        *DESIGN_VAR_NAMES,
        'T_per_rotor', 'P_total', 'FM', 'CT', 'CP', 'T_margin_pct',
        'T_per_rotor_rhomin', 'P_total_rhomin', 'T_margin_pct_rhomin',
        'f1_bending_hz', 'blade_mass_kg',
        'feasible', 'robust_feasible',
        'rpm_feasible', 'mach_feasible', 'hover_feasible', 'structural_feasible',
    ]
    kpi_rows = []
    for r in all_results:
        row = [
            r['sample_id'], int(r.get('is_baseline', False)),
            *r.get('design_vector', [0.0] * N_DESIGN_VARS),
            r.get('T_per_rotor', 0.0), r.get('P_total', 0.0),
            r.get('FM', 0.0), r.get('CT', 0.0), r.get('CP', 0.0),
            r.get('T_margin_pct', 0.0),
            r.get('T_per_rotor_rhomin', 0.0), r.get('P_total_rhomin', 0.0),
            r.get('T_margin_pct_rhomin', 0.0),
            r.get('f1_bending_hz', 0.0), r.get('blade_mass_kg', 0.0),
            int(r.get('feasible', False)), int(r.get('robust_feasible', False)),
            int(r.get('rpm_feasible', False)), int(r.get('mach_feasible', False)),
            int(r.get('hover_feasible', False)), int(r.get('structural_feasible', False)),
        ]
        kpi_rows.append(row)
    log_to_csv(str(results_dir / "doe_summary.csv"), kpi_headers, kpi_rows)

    print(f"Results saved to: {results_dir}")
    return doe_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DOE: LHS + BEMT/Structural")
    parser.add_argument("--n", type=int, default=100, help="Number of samples")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    run_doe(n_samples=args.n, seed=args.seed)
