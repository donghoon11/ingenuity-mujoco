"""
Viewer: Optimized Blade Design — MuJoCo 3D Visualization

Loads the optimized blade STL + Mars scene and runs an interactive
hover simulation with 3D viewer.

Steps:
  1. Generate STL if not present
  2. Load scene_optimized.xml
  3. Run closed-loop hover with rotor spin visualization
  4. Display in MuJoCo viewer

Usage:
  cd E:/mujoco_projects/ingenuity-mujoco
  python 0219/design/view_optimized.py
  python 0219/design/view_optimized.py --baseline   # View baseline instead
"""

import argparse
import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MARS_GRAVITY, RHO_NOMINAL, MARS_WEIGHT, NUM_ROTORS,
    THRUST_GEAR, ATTITUDE_GEAR, YAW_GEAR, HOVER_CTRL,
    IDX_THRUST1, IDX_THRUST2, IDX_ROLL, IDX_PITCH, IDX_YAW,
    ALT_KP, ALT_KI, ALT_KD, ALT_I_LIMIT, ATT_KP, ATT_KD,
    RESULTS_DIR, PROJECT_ROOT,
)
from blade_param import BladeDesign, baseline_design
from bemt import bemt_hover


def ensure_stl_exists(force_regen: bool = False):
    """Generate optimized STL if not present (or force regeneration)."""
    assets_dir = PROJECT_ROOT / "assets"
    top_stl = assets_dir / "optimized_topblades.stl"
    bot_stl = assets_dir / "optimized_bottomblades.stl"

    if top_stl.exists() and bot_stl.exists() and not force_regen:
        print(f"  STL files found: {top_stl.name}, {bot_stl.name}")
        return True

    if force_regen:
        print("  Regenerating STL files (hub-grafted)...")
    else:
        print("  STL files not found. Generating (hub-grafted)...")
    from generate_blade_stl import main as gen_main
    gen_main()

    return top_stl.exists() and bot_stl.exists()


def run_viewer(use_baseline: bool = False, regenerate: bool = False):
    import mujoco
    import mujoco.viewer

    print("=" * 70)
    if use_baseline:
        print("Viewer: Baseline Blade Design (Mars Scene)")
        scene_xml = str(Path(__file__).parent / "models" / "scene_mars.xml")
    else:
        print("Viewer: Optimized Blade Design (Mars Scene, Hub-Grafted)")
        print()
        if not ensure_stl_exists(force_regen=regenerate):
            print("ERROR: Could not generate STL files.")
            return
        scene_xml = str(Path(__file__).parent / "models" / "scene_optimized.xml")

    print("=" * 70)

    # Load design info and compute hover feedforward
    # Note: In MuJoCo, thrust = ctrl * THRUST_GEAR, independent of blade design.
    # The feedforward ctrl for hover is always MARS_WEIGHT / 2 / THRUST_GEAR.
    # BEMT ctrl_thrust reflects aerodynamic capability, not MuJoCo control.
    ctrl_ff = HOVER_CTRL  # ~0.495

    if not use_baseline:
        candidates_file = RESULTS_DIR / "pipeline" / "final_candidates.json"
        if candidates_file.exists():
            with open(candidates_file) as f:
                pipeline = json.load(f)
            candidates = pipeline.get('final_candidates', [])
            if candidates:
                best = candidates[0]
                vec = np.array(best['design_vector'])
                blade = BladeDesign(vec)
                print(f"  Design: {blade}")
                print(f"  FM={best.get('FM', 0):.4f}  P={best.get('P_total', 0):.1f}W")

                bemt_res = bemt_hover(blade, rho=RHO_NOMINAL)
                print(f"  BEMT T/rotor={bemt_res['T_per_rotor']:.3f}N  "
                      f"FM={bemt_res['FM']:.4f}")
        else:
            blade = baseline_design()
    else:
        blade = baseline_design()
        bemt_res = bemt_hover(blade, rho=RHO_NOMINAL)
        print(f"  Baseline: {blade}")
        print(f"  BEMT T/rotor={bemt_res['T_per_rotor']:.3f}N")

    print(f"  Hover feedforward ctrl={ctrl_ff:.4f} "
          f"(={MARS_WEIGHT/NUM_ROTORS:.3f}N / {THRUST_GEAR}N/ctrl)")
    print(f"\n  Loading: {scene_xml}")

    # Load model
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)

    # Controller state
    z_ref = 1.0
    alt_integral = 0.0
    dt = model.opt.timestep

    # Find rotor joints for visual spinning
    top_rotor_jnt = None
    bot_rotor_jnt = None
    for i in range(model.njnt):
        name = model.jnt(i).name
        if name == "top_rotor_joint":
            top_rotor_jnt = i
        elif name == "bottom_rotor_joint":
            bot_rotor_jnt = i

    rpm = blade.rpm if not use_baseline else 2537.0
    omega = rpm * 2.0 * np.pi / 60.0  # rad/s

    print(f"  Rotor visual spin: {rpm:.0f} RPM ({omega:.1f} rad/s)")
    print(f"\n  Starting viewer... (close window to exit)")
    print("=" * 70)

    def controller(model, data):
        nonlocal alt_integral

        # Get current state
        z = data.qpos[2]
        vz = data.qvel[2]

        # Quaternion → Euler (simplified)
        qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        # Roll
        sinr = 2.0 * (qw * qx + qy * qz)
        cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr, cosr)
        # Pitch
        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
        # Yaw rate
        yaw_rate = data.qvel[5]

        # Altitude PID
        alt_err = z_ref - z
        alt_integral += alt_err * dt
        alt_integral = np.clip(alt_integral, -ALT_I_LIMIT, ALT_I_LIMIT)
        alt_cmd = ctrl_ff + ALT_KP * alt_err + ALT_KI * alt_integral + ALT_KD * (-vz)
        alt_cmd = np.clip(alt_cmd, 0.0, 1.0)

        # Attitude PD
        roll_cmd = -(ATT_KP * roll + ATT_KD * data.qvel[3])
        pitch_cmd = -(ATT_KP * pitch + ATT_KD * data.qvel[4])
        yaw_cmd = -(2.0 * yaw_rate)

        roll_cmd = np.clip(roll_cmd, -1.0, 1.0)
        pitch_cmd = np.clip(pitch_cmd, -1.0, 1.0)
        yaw_cmd = np.clip(yaw_cmd, -1.0, 1.0)

        # Apply controls
        data.ctrl[IDX_THRUST1] = alt_cmd
        data.ctrl[IDX_THRUST2] = alt_cmd
        data.ctrl[IDX_ROLL] = roll_cmd
        data.ctrl[IDX_PITCH] = pitch_cmd
        if model.nu > IDX_YAW:
            data.ctrl[IDX_YAW] = yaw_cmd

        # Spin rotors visually
        if top_rotor_jnt is not None:
            # Find qpos index for this joint
            jnt_qposadr = model.jnt_qposadr[top_rotor_jnt]
            data.qpos[jnt_qposadr] += omega * dt  # CW
        if bot_rotor_jnt is not None:
            jnt_qposadr = model.jnt_qposadr[bot_rotor_jnt]
            data.qpos[jnt_qposadr] -= omega * dt  # CCW

    # Launch viewer
    mujoco.viewer.launch(model, data, show_left_ui=True, show_right_ui=True)


def main():
    parser = argparse.ArgumentParser(description="MuJoCo Viewer for Optimized Blade Design")
    parser.add_argument("--baseline", action="store_true",
                        help="View baseline design instead of optimized")
    parser.add_argument("--regenerate", action="store_true",
                        help="Force regenerate STL files (hub-grafted)")
    args = parser.parse_args()

    run_viewer(use_baseline=args.baseline, regenerate=args.regenerate)


if __name__ == "__main__":
    main()
