"""
E2: Robust Hover — Atmospheric Density Sweep

Evaluates hover performance across Mars atmospheric density range:
  rho = [0.012, 0.014, 0.015, 0.017, 0.021] kg/m^3

For each density:
  1) BEMT → thrust/power/FM
  2) MuJoCo closed-loop hover (10s)
  3) Record KPIs

Pareto analysis: identify density range where design can still hover.

Outputs:
  - results/e2/robust_hover_map.json
  - results/e2/sweep_log.csv

Usage:
  cd E:/mujoco_projects/ingenuity-mujoco
  python 0219/design/e2_robust_hover.py
"""

import argparse
import sys
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RHO_NOMINAL, RHO_RANGE, MARS_WEIGHT, Z_REF, SIM_DURATION,
)
from blade_param import baseline_design
from bemt import bemt_hover, hover_thrust_required
from sim_interface import MarsSimulator
from utils import ensure_results_dir, save_json, log_to_csv


# Density sweep points (Mars range)
RHO_VALUES = [0.012, 0.014, 0.015, 0.017, 0.021]


def run_e2(use_viewer: bool = False, hover_duration: float = 10.0):
    print("=" * 70)
    print("E2: Robust Hover — Atmospheric Density Sweep")
    print("=" * 70)
    print(f"  Densities: {RHO_VALUES} kg/m^3")
    print(f"  Hover duration per point: {hover_duration}s")
    print()

    results_dir = ensure_results_dir("e2")
    blade = baseline_design()

    print(f"  Baseline blade: {blade}")
    print()

    sweep_results = []
    csv_rows = []

    # Print header
    print(f"{'rho':>8}  {'T/rot(N)':>10}  {'P(W)':>8}  {'FM':>6}  "
          f"{'Hover?':>6}  {'AltErr(m)':>10}  {'AltErrSS':>10}  {'CtrlSat':>8}")
    print("-" * 80)

    for rho in RHO_VALUES:
        # ── BEMT 해석 ──────────────────────────────────────────────────────
        bemt_result = bemt_hover(blade, rho=rho)
        T_per_rotor = bemt_result['T_per_rotor']
        P_total = bemt_result['P']
        FM = bemt_result['FM']
        ctrl_thrust = bemt_result['ctrl_thrust']
        ctrl_yaw = bemt_result['ctrl_yaw']

        T_required = MARS_WEIGHT / 2.0
        T_margin_pct = (T_per_rotor / T_required - 1.0) * 100.0
        bemt_can_hover = T_per_rotor >= T_required

        # ── MuJoCo 시뮬레이션 ──────────────────────────────────────────────
        sim = MarsSimulator(headless=not use_viewer)
        sim.set_density(rho)

        hover_result = sim.run_hover(
            ctrl_thrust=ctrl_thrust,
            ctrl_yaw=ctrl_yaw,
            z_ref=Z_REF,
            duration=hover_duration,
            use_controller=True,
            viewer=False,  # no viewer in sweep
        )
        kpis = hover_result.kpis

        # ── 결과 집계 ──────────────────────────────────────────────────────
        entry = {
            'rho': rho,
            'bemt': {
                'T_per_rotor': T_per_rotor,
                'P_total': P_total,
                'FM': FM,
                'CT': bemt_result['CT'],
                'CP': bemt_result['CP'],
                'ctrl_thrust': ctrl_thrust,
                'T_margin_pct': T_margin_pct,
                'can_hover_bemt': bemt_can_hover,
            },
            'mujoco_kpis': kpis,
        }
        sweep_results.append(entry)

        alt_rms = kpis.get('alt_error_rms', float('nan'))
        alt_ss  = kpis.get('alt_error_ss_rms', float('nan'))
        ctrl_sat = kpis.get('ctrl_saturation_rate', float('nan'))
        stable = kpis.get('stable', False)

        print(f"{rho:8.4f}  {T_per_rotor:10.3f}  {P_total:8.2f}  {FM:6.4f}  "
              f"{'YES' if stable else 'NO ':>6}  {alt_rms:10.4f}  {alt_ss:10.4f}  "
              f"{ctrl_sat:8.2%}")

        csv_rows.append([
            rho, T_per_rotor, P_total, FM,
            bemt_result['CT'], bemt_result['CP'],
            T_margin_pct,
            int(stable),
            alt_rms, alt_ss,
            kpis.get('alt_error_max', float('nan')),
            ctrl_sat,
            kpis.get('settling_time', float('nan')),
        ])

    print("-" * 80)
    print()

    # ── Pareto / 요약 분석 ─────────────────────────────────────────────────
    print("=" * 70)
    print("E2 SUMMARY — Density Sweep Analysis")
    print("=" * 70)

    stable_densities = [
        r['rho'] for r in sweep_results
        if r['mujoco_kpis'].get('stable', False)
    ]
    if stable_densities:
        print(f"  Stable hover range: {min(stable_densities):.3f} ~ "
              f"{max(stable_densities):.3f} kg/m^3")
    else:
        print("  WARNING: No stable hover in any density!")

    # Find best FM
    best_fm_entry = max(sweep_results, key=lambda x: x['bemt']['FM'])
    print(f"  Best FM: {best_fm_entry['bemt']['FM']:.4f} @ rho={best_fm_entry['rho']:.4f}")

    # Worst (min density) performance
    worst = sweep_results[0]  # rho_min
    print(f"  Min density ({worst['rho']:.3f}) T_margin: "
          f"{worst['bemt']['T_margin_pct']:+.1f}%")
    print(f"  Min density power: {worst['bemt']['P_total']:.2f} W")

    # Best (max density) performance
    best = sweep_results[-1]  # rho_max
    print(f"  Max density ({best['rho']:.3f}) T_margin: "
          f"{best['bemt']['T_margin_pct']:+.1f}%")
    print("=" * 70)

    # ── 결과 저장 ──────────────────────────────────────────────────────────
    summary = {
        'sweep_results': sweep_results,
        'stable_rho_range': [min(stable_densities), max(stable_densities)]
        if stable_densities else [],
        'design_vector': blade.vector.tolist(),
        'params': {
            'rho_values': RHO_VALUES,
            'hover_duration': hover_duration,
            'z_ref': Z_REF,
        },
    }
    save_json(str(results_dir / "robust_hover_map.json"), summary)

    headers = [
        'rho', 'T_per_rotor', 'P_total', 'FM', 'CT', 'CP',
        'T_margin_pct', 'stable',
        'alt_error_rms', 'alt_error_ss_rms', 'alt_error_max',
        'ctrl_saturation_rate', 'settling_time',
    ]
    log_to_csv(str(results_dir / "sweep_log.csv"), headers, csv_rows)

    print(f"\nResults saved to: {results_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2: Robust Hover Density Sweep")
    parser.add_argument("--viewer", action="store_true", help="Show MuJoCo viewer")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Hover duration per density point (s)")
    args = parser.parse_args()
    run_e2(use_viewer=args.viewer, hover_duration=args.duration)
