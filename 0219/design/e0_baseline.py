"""
E0: Baseline Dynamics Calibration

Verifies the Mars-patched MuJoCo model works correctly:
  1) Freefall: ctrl=0 -> verify gravity ~3.71 m/s^2
  2) Thrust sign: thrust1=thrust2=+0.2 -> ascend or descend?
  3) Roll: x_movement=+0.2 -> which direction?
  4) Pitch: y_movement=+0.2 -> which direction?
  5) Yaw: z_rotation=+0.2 -> which direction?
  6) Hover test: PID closed-loop at z_ref=1.0m

Outputs:
  - results/e0/sign_conventions.json
  - results/e0/e0_log.csv
  - Console summary

Usage:
  cd E:/mujoco_projects/ingenuity-mujoco
  python 0219/design/e0_baseline.py
  python 0219/design/e0_baseline.py --viewer
"""

import argparse
import sys
import numpy as np

# Ensure design/ is in path for imports
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MARS_GRAVITY, HOVER_CTRL, THRUST_GEAR, DT, Z_REF,
    IDX_THRUST1, IDX_THRUST2, IDX_ROLL, IDX_PITCH, IDX_YAW,
)
from sim_interface import MarsSimulator, SimResult
from utils import (
    extract_state, ensure_results_dir, save_json, log_to_csv,
)


def run_e0(use_viewer: bool = False):
    """Run E0 baseline calibration."""

    print("=" * 70)
    print("E0: Baseline Dynamics Calibration (Mars-patched model)")
    print("=" * 70)
    print()

    results_dir = ensure_results_dir("e0")
    sim = MarsSimulator(headless=not use_viewer)

    conventions = {}
    all_logs = []

    # ─── Test 1: Freefall (0~1s) ──────────────────────────────────────────
    print("[Test 1] Freefall (ctrl=0, 1s)")
    sim.reset()
    z_positions = []
    z_velocities = []
    times = []

    n_steps_1s = int(1.0 / DT)
    for step in range(n_steps_1s):
        t = step * DT
        sim.data.ctrl[:] = 0.0
        state = extract_state(sim.data)
        z_positions.append(state['pos'][2])
        z_velocities.append(state['vel'][2])
        times.append(t)
        import mujoco
        mujoco.mj_step(sim.model, sim.data)

    z_arr = np.array(z_positions)
    vz_arr = np.array(z_velocities)

    # Estimate gravity from velocity change
    if len(vz_arr) > 10:
        # dv/dt at middle of freefall
        i_mid = len(vz_arr) // 2
        accel_z = (vz_arr[i_mid + 5] - vz_arr[i_mid - 5]) / (10 * DT)
    else:
        accel_z = 0.0

    conventions['freefall_accel_z'] = float(accel_z)
    conventions['gravity_matches_mars'] = abs(accel_z + MARS_GRAVITY) < 0.5

    print(f"  z acceleration: {accel_z:.3f} m/s^2 (expected: -{MARS_GRAVITY:.2f})")
    print(f"  Final z:        {z_arr[-1]:.4f} m (started at {z_arr[0]:.4f})")
    print(f"  Gravity OK:     {conventions['gravity_matches_mars']}")
    print()

    # ─── Test 2: Thrust sign (1~2s) ──────────────────────────────────────
    print("[Test 2] Thrust sign check (ctrl[0]=ctrl[1]=+0.5, 1s)")
    sim.reset()
    z_before = sim.data.qpos[2]

    for step in range(n_steps_1s):
        sim.data.ctrl[:] = 0.0
        sim.data.ctrl[IDX_THRUST1] = 0.5
        sim.data.ctrl[IDX_THRUST2] = 0.5
        mujoco.mj_step(sim.model, sim.data)

    z_after = sim.data.qpos[2]
    dz = z_after - z_before
    thrust_up = dz > 0

    conventions['thrust_positive_is_up'] = bool(thrust_up)
    conventions['thrust_test_dz'] = float(dz)

    direction = "UP (correct)" if thrust_up else "DOWN (sign flip needed!)"
    print(f"  Delta z: {dz:.4f} m -> Positive ctrl = {direction}")
    print()

    # ─── Test 3: Roll (x_movement) ───────────────────────────────────────
    print("[Test 3] Roll check (ctrl[2]=+0.5, 0.5s)")
    sim.reset()
    # Apply some thrust to keep airborne
    n_steps_half = int(0.5 / DT)
    for step in range(n_steps_half):
        sim.data.ctrl[:] = 0.0
        sim.data.ctrl[IDX_THRUST1] = HOVER_CTRL
        sim.data.ctrl[IDX_THRUST2] = HOVER_CTRL
        sim.data.ctrl[IDX_ROLL] = 0.5
        mujoco.mj_step(sim.model, sim.data)

    state = extract_state(sim.data)
    roll_deg = np.degrees(state['euler'][0])

    if roll_deg > 1.0:
        roll_dir = "positive_roll"
    elif roll_deg < -1.0:
        roll_dir = "negative_roll"
    else:
        roll_dir = "negligible"

    conventions['x_movement_positive_result'] = roll_dir
    conventions['x_movement_roll_deg'] = float(roll_deg)

    print(f"  Roll angle: {roll_deg:.2f} deg -> {roll_dir}")
    print()

    # ─── Test 4: Pitch (y_movement) ──────────────────────────────────────
    print("[Test 4] Pitch check (ctrl[3]=+0.5, 0.5s)")
    sim.reset()
    for step in range(n_steps_half):
        sim.data.ctrl[:] = 0.0
        sim.data.ctrl[IDX_THRUST1] = HOVER_CTRL
        sim.data.ctrl[IDX_THRUST2] = HOVER_CTRL
        sim.data.ctrl[IDX_PITCH] = 0.5
        mujoco.mj_step(sim.model, sim.data)

    state = extract_state(sim.data)
    pitch_deg = np.degrees(state['euler'][1])

    if pitch_deg > 1.0:
        pitch_dir = "positive_pitch"
    elif pitch_deg < -1.0:
        pitch_dir = "negative_pitch"
    else:
        pitch_dir = "negligible"

    conventions['y_movement_positive_result'] = pitch_dir
    conventions['y_movement_pitch_deg'] = float(pitch_deg)

    print(f"  Pitch angle: {pitch_deg:.2f} deg -> {pitch_dir}")
    print()

    # ─── Test 5: Yaw (z_rotation) ────────────────────────────────────────
    print("[Test 5] Yaw check (ctrl[4]=+0.5, 0.5s)")
    sim.reset()
    has_yaw = sim.model.nu > IDX_YAW

    if has_yaw:
        for step in range(n_steps_half):
            sim.data.ctrl[:] = 0.0
            sim.data.ctrl[IDX_THRUST1] = HOVER_CTRL
            sim.data.ctrl[IDX_THRUST2] = HOVER_CTRL
            sim.data.ctrl[IDX_YAW] = 0.5
            mujoco.mj_step(sim.model, sim.data)

        state = extract_state(sim.data)
        yaw_deg = np.degrees(state['euler'][2])

        if yaw_deg > 1.0:
            yaw_dir = "positive_yaw (CCW from top)"
        elif yaw_deg < -1.0:
            yaw_dir = "negative_yaw (CW from top)"
        else:
            yaw_dir = "negligible"

        conventions['z_rotation_positive_result'] = yaw_dir
        conventions['z_rotation_yaw_deg'] = float(yaw_deg)
        print(f"  Yaw angle: {yaw_deg:.2f} deg -> {yaw_dir}")
    else:
        conventions['z_rotation_positive_result'] = "NO_YAW_ACTUATOR"
        print("  No yaw actuator found!")
    print()

    # ─── Test 6: Closed-loop hover (10s) ─────────────────────────────────
    print("[Test 6] Closed-loop hover test (z_ref=1.0m, 10s)")
    hover_result = sim.run_hover(
        ctrl_thrust=HOVER_CTRL,
        z_ref=Z_REF,
        duration=10.0,
        use_controller=True,
        viewer=use_viewer,
    )

    kpis = hover_result.kpis
    conventions['hover_stable'] = kpis.get('stable', False)
    conventions['hover_alt_error_rms'] = kpis.get('alt_error_rms', float('inf'))
    conventions['hover_alt_error_ss_rms'] = kpis.get('alt_error_ss_rms', float('inf'))
    conventions['hover_settling_time'] = kpis.get('settling_time', float('inf'))
    conventions['hover_ctrl_saturation'] = kpis.get('ctrl_saturation_rate', 1.0)

    print(f"  Stable:          {kpis.get('stable', 'N/A')}")
    print(f"  Alt error RMS:   {kpis.get('alt_error_rms', 'N/A'):.4f} m")
    print(f"  Alt error SS:    {kpis.get('alt_error_ss_rms', 'N/A'):.4f} m")
    print(f"  Settling time:   {kpis.get('settling_time', 'N/A'):.2f} s")
    print(f"  Ctrl saturation: {kpis.get('ctrl_saturation_rate', 'N/A'):.2%}")
    print()

    # ─── Save results ────────────────────────────────────────────────────
    conventions['model_info'] = {
        'n_actuators': int(sim.model.nu),
        'timestep': float(sim.dt),
        'density': float(sim.model.opt.density),
        'gravity': sim.model.opt.gravity.tolist(),
        'thrust_gear': THRUST_GEAR,
        'hover_ctrl': HOVER_CTRL,
    }

    save_json(str(results_dir / "sign_conventions.json"), conventions)
    print(f"Results saved to: {results_dir}")

    # Save hover log as CSV
    if len(hover_result.time) > 0:
        hover_result.to_arrays()
        headers = ['time', 'x', 'y', 'z', 'vx', 'vy', 'vz',
                   'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'z_ref']
        rows = []
        for i in range(len(hover_result.time)):
            row = [hover_result.time[i]]
            row.extend(hover_result.pos[i].tolist())
            row.extend(hover_result.vel[i].tolist())
            row.extend(hover_result.euler[i].tolist())
            row.extend(hover_result.omega[i].tolist())
            row.append(hover_result.z_ref[i])
            rows.append(row)
        log_to_csv(str(results_dir / "hover_log.csv"), headers, rows)

    # ─── Summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("E0 SUMMARY")
    print("=" * 70)
    print(f"  Gravity:         {'OK' if conventions['gravity_matches_mars'] else 'MISMATCH'} "
          f"({conventions['freefall_accel_z']:.3f} m/s^2)")
    print(f"  Thrust sign:     {'UP (OK)' if conventions['thrust_positive_is_up'] else 'INVERTED'}")
    print(f"  Roll (x_ctrl):   {conventions['x_movement_positive_result']} "
          f"({conventions['x_movement_roll_deg']:.1f} deg)")
    print(f"  Pitch (y_ctrl):  {conventions['y_movement_positive_result']} "
          f"({conventions['y_movement_pitch_deg']:.1f} deg)")
    print(f"  Yaw (z_ctrl):    {conventions.get('z_rotation_positive_result', 'N/A')}")
    print(f"  Hover stable:    {conventions['hover_stable']}")
    print(f"  Alt error (SS):  {conventions['hover_alt_error_ss_rms']:.4f} m")
    print("=" * 70)

    return conventions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E0: Baseline Dynamics Calibration")
    parser.add_argument("--viewer", action="store_true", help="Show MuJoCo viewer")
    args = parser.parse_args()

    run_e0(use_viewer=args.viewer)
