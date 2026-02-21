"""
Master Pipeline: DOE → Surrogate → Optimizer → MuJoCo Validation

Orchestrates the full blade optimization workflow:
  Phase 1: DOE (LHS sampling + BEMT/structural evaluation)
  Phase 2: Surrogate model (GP regression)
  Phase 3: NSGA-II optimization
  Phase 4: Top-N candidate validation with MuJoCo hover simulation

Outputs:
  - results/pipeline/final_candidates.json  (top candidates + MuJoCo KPIs)
  - results/pipeline/pipeline_summary.json

Usage:
  cd E:/mujoco_projects/ingenuity-mujoco
  python 0219/design/pipeline.py
  python 0219/design/pipeline.py --doe-n 100 --gen 30 --top 5
"""

import argparse
import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RHO_NOMINAL, DESIGN_VAR_NAMES, N_DESIGN_VARS,
    RESULTS_DIR, Z_REF, SIM_DURATION,
)
from blade_param import BladeDesign, baseline_design
from bemt import bemt_hover
from doe import run_doe
from surrogate import run_surrogate
from optimizer import run_optimizer
from sim_interface import MarsSimulator
from utils import ensure_results_dir, save_json, log_to_csv


def validate_with_mujoco(candidates: list, n_top: int = 5,
                          hover_duration: float = 10.0) -> list:
    """
    Run MuJoCo hover simulation for top-N candidates.

    Parameters
    ----------
    candidates : list of dicts from optimizer pareto_results
    n_top      : number of candidates to validate
    hover_duration : seconds of hover simulation

    Returns
    -------
    list of dicts with added 'mujoco_kpis' field
    """
    # Select top-N by FM (feasible first, then by FM)
    feasible = [c for c in candidates if c.get('feasible', False)]
    if not feasible:
        feasible = candidates  # Use all if none feasible

    # Sort by FM descending
    top_candidates = sorted(feasible, key=lambda x: x.get('FM', 0.0), reverse=True)[:n_top]

    print(f"  Validating top {len(top_candidates)} candidates with MuJoCo...")
    sim = MarsSimulator(headless=True)

    for i, cand in enumerate(top_candidates):
        vec = np.array(cand.get('design_vector', []))
        if len(vec) != N_DESIGN_VARS:
            cand['mujoco_kpis'] = {'error': 'invalid design vector'}
            continue

        ctrl_thrust = cand.get('ctrl_thrust', 0.495)

        print(f"  [{i + 1}/{len(top_candidates)}] "
              f"FM={cand.get('FM', 0):.4f}  "
              f"P={cand.get('P_total', 0):.2f}W  "
              f"ctrl={ctrl_thrust:.4f}")

        hover_result = sim.run_hover(
            ctrl_thrust=ctrl_thrust,
            z_ref=Z_REF,
            duration=hover_duration,
            use_controller=True,
            viewer=False,
        )
        cand['mujoco_kpis'] = hover_result.kpis
        cand['mujoco_validated'] = True

    return top_candidates


def run_pipeline(doe_n: int = 100, doe_seed: int = 0,
                 n_gen: int = 40, pop_size: int = 60,
                 n_top: int = 5,
                 skip_doe: bool = False,
                 skip_surrogate: bool = False,
                 skip_optimizer: bool = False,
                 skip_mujoco: bool = False,
                 hover_duration: float = 10.0):

    print("=" * 70)
    print("MASTER PIPELINE: Blade Optimization (DOE → Surrogate → NSGA-II → MuJoCo)")
    print("=" * 70)
    print(f"  DOE samples: {doe_n}  |  NSGA-II: gen={n_gen}, pop={pop_size}")
    print(f"  Top-N for MuJoCo validation: {n_top}")
    print()

    results_dir = ensure_results_dir("pipeline")
    t_pipeline_start = time.time()
    pipeline_log = {}

    # ── Phase 1: DOE ────────────────────────────────────────────────────────
    doe_file = str(RESULTS_DIR / "doe" / "doe_results.json")
    if skip_doe and Path(doe_file).exists():
        print("[Phase 1] DOE: Skipping (using existing results)")
        with open(doe_file) as f:
            doe_output = json.load(f)
        pipeline_log['doe'] = {
            'skipped': True,
            'n_samples': doe_output.get('n_samples', 0),
            'n_feasible': doe_output.get('n_feasible', 0),
        }
    else:
        print("[Phase 1] DOE: Latin Hypercube Sampling + BEMT/Structural")
        t0 = time.time()
        doe_output = run_doe(n_samples=doe_n, seed=doe_seed)
        elapsed = time.time() - t0
        pipeline_log['doe'] = {
            'n_samples': doe_output.get('n_samples', 0),
            'n_feasible': doe_output.get('n_feasible', 0),
            'elapsed_s': elapsed,
        }
    print()

    # ── Phase 2: Surrogate ─────────────────────────────────────────────────
    surrogate_path = str(RESULTS_DIR / "surrogate" / "gp_model.pkl")
    if skip_surrogate and Path(surrogate_path).exists():
        print("[Phase 2] Surrogate: Skipping (using existing model)")
        pipeline_log['surrogate'] = {'skipped': True}
    else:
        print("[Phase 2] Surrogate: Gaussian Process Regression")
        t0 = time.time()
        surrogate_output = run_surrogate(doe_file=doe_file)
        elapsed = time.time() - t0
        pipeline_log['surrogate'] = {
            'elapsed_s': elapsed,
            'success': surrogate_output is not None,
        }
        if surrogate_output:
            pipeline_log['surrogate']['cv_errors'] = surrogate_output.get('cv_errors', {})
    print()

    # ── Phase 3: NSGA-II Optimization ─────────────────────────────────────
    pareto_file = str(RESULTS_DIR / "optimizer" / "pareto_front.json")
    if skip_optimizer and Path(pareto_file).exists():
        print("[Phase 3] NSGA-II: Skipping (using existing Pareto front)")
        with open(pareto_file) as f:
            optim_output = json.load(f)
        pipeline_log['optimizer'] = {'skipped': True, 'n_pareto': optim_output.get('n_pareto', 0)}
    else:
        print("[Phase 3] NSGA-II: Multi-Objective Optimization")
        t0 = time.time()
        optim_output = run_optimizer(
            n_gen=n_gen, pop_size=pop_size,
            use_surrogate=True, seed=doe_seed,
        )
        elapsed = time.time() - t0
        pipeline_log['optimizer'] = {
            'n_pareto': optim_output.get('n_pareto', 0),
            'elapsed_s': elapsed,
        }
    print()

    # ── Phase 4: MuJoCo Validation ─────────────────────────────────────────
    if skip_mujoco:
        print("[Phase 4] MuJoCo Validation: Skipped")
        validated_candidates = []
        pipeline_log['mujoco'] = {'skipped': True}
    else:
        print("[Phase 4] MuJoCo Validation: Top candidates hover simulation")
        t0 = time.time()
        pareto_results = optim_output.get('pareto_results', [])

        if not pareto_results:
            print("  WARNING: No Pareto results to validate")
            validated_candidates = []
        else:
            validated_candidates = validate_with_mujoco(
                pareto_results, n_top=n_top, hover_duration=hover_duration
            )

        elapsed = time.time() - t0
        pipeline_log['mujoco'] = {
            'n_validated': len(validated_candidates),
            'elapsed_s': elapsed,
        }
    print()

    # ── Final Summary ──────────────────────────────────────────────────────
    t_total = time.time() - t_pipeline_start

    print("=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)

    # Baseline comparison
    baseline = baseline_design()
    bemt_bl = bemt_hover(baseline, rho=RHO_NOMINAL)
    print(f"  Baseline: P={bemt_bl['P']:.2f}W  FM={bemt_bl['FM']:.4f}")

    if validated_candidates:
        best = validated_candidates[0]
        print(f"  Best (FM): P={best.get('P_total', 0):.2f}W  "
              f"FM={best.get('FM', 0):.4f}  "
              f"stable={best.get('mujoco_kpis', {}).get('stable', 'N/A')}")

        # Improvement
        if bemt_bl['FM'] > 0:
            fm_improve = (best.get('FM', 0) / bemt_bl['FM'] - 1.0) * 100.0
            print(f"  FM improvement over baseline: {fm_improve:+.1f}%")
        if bemt_bl['P'] > 0:
            p_reduce = (1.0 - best.get('P_total', bemt_bl['P']) / bemt_bl['P']) * 100.0
            print(f"  Power reduction: {p_reduce:+.1f}%")
    else:
        print("  (No MuJoCo-validated candidates)")

    print(f"  Total pipeline time: {t_total:.1f}s")
    print("=" * 70)

    # ── Save ──────────────────────────────────────────────────────────────
    final_output = {
        'final_candidates': validated_candidates,
        'n_candidates': len(validated_candidates),
        'baseline': {
            'P_total': bemt_bl['P'],
            'FM': bemt_bl['FM'],
            'CT': bemt_bl['CT'],
            'design_vector': baseline.vector.tolist(),
        },
        'pipeline_log': pipeline_log,
        'total_elapsed_s': t_total,
        'params': {
            'doe_n': doe_n,
            'n_gen': n_gen,
            'pop_size': pop_size,
            'n_top': n_top,
        },
    }
    save_json(str(results_dir / "final_candidates.json"), final_output)
    save_json(str(results_dir / "pipeline_summary.json"), pipeline_log)

    print(f"\nResults saved to: {results_dir}")
    return final_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Optimization Pipeline")
    parser.add_argument("--doe-n", type=int, default=100, help="DOE sample count")
    parser.add_argument("--gen", type=int, default=40, help="NSGA-II generations")
    parser.add_argument("--pop", type=int, default=60, help="Population size")
    parser.add_argument("--top", type=int, default=5, help="Top candidates for MuJoCo")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--skip-doe", action="store_true", help="Reuse existing DOE")
    parser.add_argument("--skip-surrogate", action="store_true", help="Reuse existing surrogate")
    parser.add_argument("--skip-optimizer", action="store_true", help="Reuse existing Pareto front")
    parser.add_argument("--skip-mujoco", action="store_true", help="Skip MuJoCo validation")
    args = parser.parse_args()

    run_pipeline(
        doe_n=args.doe_n,
        doe_seed=args.seed,
        n_gen=args.gen,
        pop_size=args.pop,
        n_top=args.top,
        skip_doe=args.skip_doe,
        skip_surrogate=args.skip_surrogate,
        skip_optimizer=args.skip_optimizer,
        skip_mujoco=args.skip_mujoco,
    )
