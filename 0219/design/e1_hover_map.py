"""
E1: Hover Performance Map

Evaluates baseline blade design hover performance:
  1) BEMT aerodynamic analysis → CT, CP, FM, T_per_rotor, P_total
  2) Convert BEMT thrust to MuJoCo ctrl_thrust
  3) Run 10s closed-loop hover simulation
  4) Report hover KPIs: alt_error, ctrl_saturation, power estimate

Outputs:
  - results/e1/baseline_hover.json
  - results/e1/hover_log.csv
  - results/e1/bemt_baseline.json

Usage:
  cd E:/mujoco_projects/ingenuity-mujoco
  python 0219/design/e1_hover_map.py
  python 0219/design/e1_hover_map.py --viewer
"""

import argparse
import sys
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RHO_NOMINAL, MARS_WEIGHT, THRUST_GEAR, Z_REF, SIM_DURATION,
    DESIGN_VAR_NAMES,
)
from blade_param import baseline_design
from bemt import bemt_hover
from sim_interface import MarsSimulator
from utils import ensure_results_dir, save_json, log_to_csv


def run_e1(use_viewer: bool = False):
    print("=" * 70)
    print("E1: Hover Performance Map (Baseline Design)")
    print("=" * 70)
    print()

    results_dir = ensure_results_dir("e1")

    # ─── Step 1: BEMT 공력 해석 ────────────────────────────────────────────
    print("[Step 1] BEMT aerodynamic analysis (baseline, rho=0.015)")
    blade = baseline_design()
    print(f"  Blade: {blade}")
    print(f"  Solidity: {blade.solidity():.4f}")
    print(f"  Tip speed: {blade.tip_speed():.1f} m/s  |  Tip Mach: {blade.tip_mach():.3f}")
    print()

    bemt_result = bemt_hover(blade, rho=RHO_NOMINAL, verbose=True)
    print()

    T_req_per_rotor = MARS_WEIGHT / 2.0
    T_margin = (bemt_result['T_per_rotor'] / T_req_per_rotor - 1.0) * 100.0
    can_hover = bemt_result['T_per_rotor'] >= T_req_per_rotor

    print(f"  Required T/rotor: {T_req_per_rotor:.3f} N")
    print(f"  Achieved T/rotor: {bemt_result['T_per_rotor']:.3f} N  (margin: {T_margin:+.1f}%)")
    print(f"  Can hover: {can_hover}")
    print()

    # Save BEMT results
    bemt_save = {k: v for k, v in bemt_result.items() if k != 'radial'}
    bemt_save['radial'] = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in bemt_result['radial'].items()
    }
    bemt_save['design_vector'] = blade.vector.tolist()
    bemt_save['design_var_names'] = DESIGN_VAR_NAMES
    bemt_save['rho'] = RHO_NOMINAL
    bemt_save['T_required_per_rotor'] = T_req_per_rotor
    bemt_save['T_margin_pct'] = T_margin
    bemt_save['can_hover'] = can_hover
    save_json(str(results_dir / "bemt_baseline.json"), bemt_save)

    # ─── Step 2: MuJoCo 호버 시뮬레이션 ──────────────────────────────────
    print("[Step 2] MuJoCo closed-loop hover (10s, z_ref=1.0m)")
    ctrl_thrust = bemt_result['ctrl_thrust']
    ctrl_yaw = bemt_result['ctrl_yaw']
    print(f"  ctrl_thrust (BEMT): {ctrl_thrust:.4f}  ctrl_yaw: {ctrl_yaw:.6f}")
    print()

    sim = MarsSimulator(headless=not use_viewer)
    hover_result = sim.run_hover(
        ctrl_thrust=ctrl_thrust,
        ctrl_yaw=ctrl_yaw,
        z_ref=Z_REF,
        duration=SIM_DURATION,
        use_controller=True,
        viewer=use_viewer,
    )

    kpis = hover_result.kpis

    # ─── Step 3: 결과 출력 및 저장 ────────────────────────────────────────
    print()
    print("=" * 70)
    print("E1 SUMMARY")
    print("=" * 70)
    print(f"  BEMT T/rotor:    {bemt_result['T_per_rotor']:.3f} N  (required: {T_req_per_rotor:.3f} N)")
    print(f"  BEMT P_total:    {bemt_result['P']:.2f} W")
    print(f"  BEMT FM:         {bemt_result['FM']:.4f}")
    print(f"  BEMT CT:         {bemt_result['CT']:.6f}")
    print(f"  BEMT CP:         {bemt_result['CP']:.8f}")
    print(f"  ctrl_thrust:     {ctrl_thrust:.4f}")
    print()
    print(f"  Hover stable:    {kpis.get('stable', 'N/A')}")
    print(f"  Alt error RMS:   {kpis.get('alt_error_rms', float('nan')):.4f} m")
    print(f"  Alt error SS:    {kpis.get('alt_error_ss_rms', float('nan')):.4f} m")
    print(f"  Alt error max:   {kpis.get('alt_error_max', float('nan')):.4f} m")
    print(f"  Roll RMS:        {np.degrees(kpis.get('roll_rms', 0.0)):.3f} deg")
    print(f"  Pitch RMS:       {np.degrees(kpis.get('pitch_rms', 0.0)):.3f} deg")
    print(f"  Ctrl saturation: {kpis.get('ctrl_saturation_rate', float('nan')):.2%}")
    print(f"  Settling time:   {kpis.get('settling_time', float('nan')):.2f} s")
    print("=" * 70)

    # Save hover result JSON
    hover_summary = {
        'bemt': {
            'T_per_rotor': bemt_result['T_per_rotor'],
            'T_total': bemt_result['T'],
            'Q_per_rotor': bemt_result['Q'],
            'P_total': bemt_result['P'],
            'CT': bemt_result['CT'],
            'CP': bemt_result['CP'],
            'FM': bemt_result['FM'],
            'ctrl_thrust': ctrl_thrust,
            'ctrl_yaw': ctrl_yaw,
            'converged': bemt_result['converged'],
            'T_margin_pct': T_margin,
        },
        'mujoco_kpis': kpis,
        'params': {
            'rho': RHO_NOMINAL,
            'z_ref': Z_REF,
            'duration': SIM_DURATION,
            'design_vector': blade.vector.tolist(),
        },
    }
    save_json(str(results_dir / "baseline_hover.json"), hover_summary)

    # Save time-series CSV
    if len(hover_result.time) > 0:
        headers = ['time', 'x', 'y', 'z', 'vx', 'vy', 'vz',
                   'roll_deg', 'pitch_deg', 'yaw_deg', 'p', 'q', 'r',
                   'ctrl_thrust', 'z_ref']
        rows = []
        for i in range(len(hover_result.time)):
            row = [hover_result.time[i]]
            row.extend(hover_result.pos[i].tolist())
            row.extend(hover_result.vel[i].tolist())
            row.extend(np.degrees(hover_result.euler[i]).tolist())
            row.extend(hover_result.omega[i].tolist())
            row.append(hover_result.ctrl[i][0])   # thrust ctrl
            row.append(hover_result.z_ref[i])
            rows.append(row)
        log_to_csv(str(results_dir / "hover_log.csv"), headers, rows)

    print(f"\nResults saved to: {results_dir}")
    return hover_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E1: Hover Performance Map")
    parser.add_argument("--viewer", action="store_true", help="Show MuJoCo viewer")
    args = parser.parse_args()
    run_e1(use_viewer=args.viewer)
