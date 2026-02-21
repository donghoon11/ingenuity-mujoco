"""
E3: Forward Flight — Waypoint Tracking + Gust Response

Evaluates forward flight capability:
  Phase 1 (0~3s):   Hover at z=1.0m, x=y=0
  Phase 2 (3~8s):   Fly to x=3.0m (pitch-based forward flight)
  Phase 3 (8~13s):  Hover at x=3.0m

Gust disturbance: Fx=0.2N at t=6s, duration=1s

Uses ForwardFlightController + AltitudePID + AttitudePD from sim_interface.

Outputs:
  - results/e3/forward_flight.json
  - results/e3/trajectory_log.csv

Usage:
  cd E:/mujoco_projects/ingenuity-mujoco
  python 0219/design/e3_forward_flight.py
  python 0219/design/e3_forward_flight.py --viewer --gust
"""

import argparse
import sys
import numpy as np
import mujoco

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RHO_NOMINAL, MARS_WEIGHT, Z_REF, DT,
    IDX_THRUST1, IDX_THRUST2, IDX_ROLL, IDX_PITCH, IDX_YAW,
    HOVER_CTRL,
)
from blade_param import baseline_design
from bemt import bemt_hover
from sim_interface import MarsSimulator, SimResult
from controller import AltitudePID, AttitudePD, ForwardFlightController
from utils import (
    ensure_results_dir, save_json, log_to_csv,
    extract_state, get_sensor_data, get_body_id,
    update_rotor_visual, update_tracking_camera,
)


# ─── Trajectory Definition ──────────────────────────────────────────────────

WAYPOINTS = [
    # (t_start, x_ref, y_ref, z_ref)
    (0.0,  0.0, 0.0, 1.0),   # Hover
    (3.0,  3.0, 0.0, 1.0),   # Forward to x=3m
    (8.0,  3.0, 0.0, 1.0),   # Hover at destination
]
TOTAL_DURATION = 13.0

# Gust parameters
GUST_T0   = 6.0    # s
GUST_DT   = 1.0    # s
GUST_FX   = 0.2    # N (significant disturbance)
GUST_FY   = 0.0
GUST_FZ   = 0.0


def trajectory_fn(t: float) -> np.ndarray:
    """Returns (x_ref, y_ref, z_ref) for time t."""
    # Find active waypoint segment
    x_ref, y_ref, z_ref = WAYPOINTS[-1][1:]
    for i in range(len(WAYPOINTS) - 1):
        t0, x0, y0, z0 = WAYPOINTS[i]
        t1, x1, y1, z1 = WAYPOINTS[i + 1]
        if t0 <= t < t1:
            alpha = (t - t0) / (t1 - t0)
            x_ref = x0 + alpha * (x1 - x0)
            y_ref = y0 + alpha * (y1 - y0)
            z_ref = z0 + alpha * (z1 - z0)
            break
    return np.array([x_ref, y_ref, z_ref])


def run_e3(use_viewer: bool = False, enable_gust: bool = False):
    print("=" * 70)
    print("E3: Forward Flight — Waypoint Tracking + Gust Response")
    print("=" * 70)
    print(f"  Duration: {TOTAL_DURATION}s")
    print(f"  Waypoints: hover(0~3s) → forward x=3m(3~8s) → hover(8~13s)")
    if enable_gust:
        print(f"  Gust: Fx={GUST_FX}N @ t={GUST_T0}~{GUST_T0+GUST_DT}s")
    print()

    results_dir = ensure_results_dir("e3")

    # ── BEMT로 hover ctrl 계산 ─────────────────────────────────────────────
    blade = baseline_design()
    bemt_result = bemt_hover(blade, rho=RHO_NOMINAL)
    ctrl_thrust_ff = bemt_result['ctrl_thrust']
    ctrl_yaw_ff = bemt_result['ctrl_yaw']
    print(f"  BEMT ctrl_thrust: {ctrl_thrust_ff:.4f}")
    print()

    # ── 시뮬레이션 실행 ────────────────────────────────────────────────────
    sim = MarsSimulator(headless=not use_viewer)
    sim.reset(z_init=0.5)

    result = SimResult()

    alt_pid = AltitudePID(ff_thrust=ctrl_thrust_ff)
    att_pd = AttitudePD()
    fwd_ctrl = ForwardFlightController(
        kp_pos=0.4, kd_pos=0.6,
        max_tilt=np.radians(12),   # 12° max tilt on Mars
    )

    n_steps = int(TOTAL_DURATION / DT)

    # Cache body ID for gust
    body_id = get_body_id(sim.model, "ingenuity")

    def _run_sim(viewer_ctx=None):
        gust_active_log = []

        for step in range(n_steps):
            t = step * DT
            state = extract_state(sim.data)
            pos = state['pos']
            vel = state['vel']
            roll, pitch, yaw = state['euler']
            p, q, r = state['omega']

            # Trajectory reference
            ref = trajectory_fn(t)
            x_ref, y_ref, z_ref = ref[0], ref[1], ref[2]

            # Get altitude measurement
            try:
                laser, gyro, acc, quat_sensor = get_sensor_data(sim.model, sim.data)
                z_meas = laser if laser > 0 else pos[2]
            except Exception:
                z_meas = pos[2]

            # Outer loop: position → tilt reference
            roll_ref, pitch_ref = fwd_ctrl.compute(ref, pos, vel)

            # Altitude PID
            thrust_cmd = alt_pid.compute(z_ref, z_meas, DT)

            # Tilt compensation (cos correction)
            tilt_mag = np.sqrt(roll ** 2 + pitch ** 2)
            cos_comp = 1.0 / max(np.cos(tilt_mag), 0.85)
            thrust_cmd = float(np.clip(thrust_cmd * cos_comp, 0.0, 1.0))

            # Attitude PD
            x_cmd, y_cmd, z_cmd = att_pd.compute(
                roll_ref, pitch_ref, 0.0,
                roll, pitch, yaw,
                p, q, r,
            )

            # Apply ctrl
            sim.data.ctrl[IDX_THRUST1] = thrust_cmd
            sim.data.ctrl[IDX_THRUST2] = thrust_cmd
            sim.data.ctrl[IDX_ROLL] = x_cmd
            sim.data.ctrl[IDX_PITCH] = y_cmd
            if sim.model.nu > IDX_YAW:
                sim.data.ctrl[IDX_YAW] = ctrl_yaw_ff + z_cmd * 0.1

            # Gust disturbance
            gust_on = False
            if enable_gust and GUST_T0 <= t < GUST_T0 + GUST_DT:
                force = np.array([GUST_FX, GUST_FY, GUST_FZ])
                torque = np.zeros(3)
                mujoco.mj_applyFT(
                    sim.model, sim.data,
                    force, torque,
                    sim.data.xpos[body_id], body_id,
                    sim.data.qfrc_applied,
                )
                gust_on = True

            # Rotor visual
            update_rotor_visual(
                sim.data, sim.top_idx, sim.bot_idx,
                thrust_cmd, DT,
            )

            mujoco.mj_step(sim.model, sim.data)

            ctrl_record = sim.data.ctrl[:sim.model.nu].copy()
            result.append(t, state, ctrl_record, z_ref)
            gust_active_log.append(int(gust_on))

            if viewer_ctx is not None:
                update_tracking_camera(viewer_ctx, sim.data, body_id)
                viewer_ctx.sync()

            # Safety exit
            if abs(pos[2]) > 25.0:
                print(f"  WARNING: Altitude diverged at t={t:.2f}s")
                break

        return gust_active_log

    print("[Running simulation...]")
    if use_viewer:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as v:
            gust_log = _run_sim(v)
    else:
        gust_log = _run_sim(None)

    result.compute_kpis()
    kpis = result.kpis

    # ── 추가 KPI: 경로추종 오차 ────────────────────────────────────────────
    if len(result.time) > 0:
        x_actual = result.pos[:, 0]
        y_actual = result.pos[:, 1]
        z_actual = result.pos[:, 2]

        x_refs = np.array([trajectory_fn(t)[0] for t in result.time])
        y_refs = np.array([trajectory_fn(t)[1] for t in result.time])
        z_refs = result.z_ref

        pos_err_xy = np.sqrt((x_refs - x_actual) ** 2 + (y_refs - y_actual) ** 2)
        pos_err_3d = np.sqrt(pos_err_xy ** 2 + (z_refs - z_actual) ** 2)

        kpis['xy_tracking_rms'] = float(np.sqrt(np.mean(pos_err_xy ** 2)))
        kpis['xy_tracking_max'] = float(np.max(pos_err_xy))
        kpis['pos3d_tracking_rms'] = float(np.sqrt(np.mean(pos_err_3d ** 2)))

        # Phase-specific errors (flight phase: 3~8s)
        flight_mask = (result.time >= 3.0) & (result.time <= 8.0)
        if np.sum(flight_mask) > 0:
            kpis['xy_tracking_rms_flight'] = float(
                np.sqrt(np.mean(pos_err_xy[flight_mask] ** 2))
            )
        else:
            kpis['xy_tracking_rms_flight'] = float('nan')

    # ── 출력 ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("E3 SUMMARY")
    print("=" * 70)
    print(f"  Stable:            {kpis.get('stable', 'N/A')}")
    print(f"  Alt error RMS:     {kpis.get('alt_error_rms', float('nan')):.4f} m")
    print(f"  XY tracking RMS:   {kpis.get('xy_tracking_rms', float('nan')):.4f} m")
    print(f"  XY flight RMS:     {kpis.get('xy_tracking_rms_flight', float('nan')):.4f} m")
    print(f"  XY tracking max:   {kpis.get('xy_tracking_max', float('nan')):.4f} m")
    print(f"  3D tracking RMS:   {kpis.get('pos3d_tracking_rms', float('nan')):.4f} m")
    print(f"  Roll RMS:          {np.degrees(kpis.get('roll_rms', 0.0)):.3f} deg")
    print(f"  Pitch RMS:         {np.degrees(kpis.get('pitch_rms', 0.0)):.3f} deg")
    print(f"  Ctrl saturation:   {kpis.get('ctrl_saturation_rate', float('nan')):.2%}")
    if enable_gust:
        print(f"  Gust applied: Yes  (t={GUST_T0}~{GUST_T0+GUST_DT}s, Fx={GUST_FX}N)")
    print("=" * 70)

    # ── 저장 ──────────────────────────────────────────────────────────────
    summary = {
        'kpis': kpis,
        'params': {
            'duration': TOTAL_DURATION,
            'waypoints': [[t, x, y, z] for t, x, y, z in WAYPOINTS],
            'gust_enabled': enable_gust,
            'gust': {'t0': GUST_T0, 'dt': GUST_DT, 'fx': GUST_FX, 'fy': GUST_FY, 'fz': GUST_FZ},
            'rho': RHO_NOMINAL,
            'bemt_ctrl_thrust': ctrl_thrust_ff,
        },
    }
    save_json(str(results_dir / "forward_flight.json"), summary)

    if len(result.time) > 0:
        headers = [
            'time', 'x', 'y', 'z', 'vx', 'vy', 'vz',
            'roll_deg', 'pitch_deg', 'yaw_deg',
            'p', 'q', 'r',
            'x_ref', 'y_ref', 'z_ref',
            'ctrl_thrust', 'ctrl_roll', 'ctrl_pitch',
            'gust_active',
        ]
        rows = []
        n = len(result.time)
        for i in range(n):
            ref_i = trajectory_fn(result.time[i])
            row = [
                result.time[i],
                *result.pos[i].tolist(),
                *result.vel[i].tolist(),
                *np.degrees(result.euler[i]).tolist(),
                *result.omega[i].tolist(),
                ref_i[0], ref_i[1], ref_i[2],
                result.ctrl[i][0],   # thrust
                result.ctrl[i][2] if result.ctrl.shape[1] > 2 else 0.0,
                result.ctrl[i][3] if result.ctrl.shape[1] > 3 else 0.0,
                gust_log[i] if i < len(gust_log) else 0,
            ]
            rows.append(row)
        log_to_csv(str(results_dir / "trajectory_log.csv"), headers, rows)

    print(f"\nResults saved to: {results_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E3: Forward Flight")
    parser.add_argument("--viewer", action="store_true", help="Show MuJoCo viewer")
    parser.add_argument("--gust", action="store_true", help="Enable gust disturbance")
    args = parser.parse_args()
    run_e3(use_viewer=args.viewer, enable_gust=args.gust)
