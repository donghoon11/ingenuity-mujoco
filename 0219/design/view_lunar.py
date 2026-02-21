"""
Viewer: Lunar Environment — Keyboard Flight Control

Ingenuity helicopter hovering over Apollo 16 lunar terrain.
Physics use Mars gravity/density. Keyboard for dynamic flight control.

Controls:
  W / Up     Forward  (+X, pitch tilt)
  S / Down   Backward (-X)
  A / Left   Strafe left  (-Y, roll tilt)
  D / Right  Strafe right (+Y)
  Space      Ascend
  Shift      Descend
  E          Level reset (cancel tilt, hold altitude)
  ESC        Quit

Usage:
  cd E:/mujoco_projects/ingenuity-mujoco
  python 0219/design/view_lunar.py
  python 0219/design/view_lunar.py --regenerate
"""

import argparse
import sys
import time
import json
import threading
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MARS_GRAVITY, MARS_WEIGHT, TOTAL_MASS, NUM_ROTORS,
    THRUST_GEAR, ATTITUDE_GEAR, YAW_GEAR, HOVER_CTRL,
    IDX_THRUST1, IDX_THRUST2, IDX_ROLL, IDX_PITCH, IDX_YAW,
    RESULTS_DIR, PROJECT_ROOT,
)
from blade_param import BladeDesign, baseline_design

# ─── Flight parameters ──────────────────────────────────────────────────────
Z_REF_INIT = 1.0
Z_MIN      = 0.3
Z_MAX      = 5.0
ALT_RATE   = 0.005        # m per step for space/shift

# Tilt control
MAX_TILT   = np.radians(12.0)
RAMP_RATE  = np.radians(60.0)   # rad/s ramp toward target
DECAY_RATE = np.radians(90.0)   # rad/s decay to zero

# Velocity damping (when no directional key)
VEL_DAMP_GAIN  = 0.02
MAX_DAMP_TILT  = np.radians(5.0)
MAX_SPEED      = 5.0

# Altitude PID (Mars-tuned)
KP_ALT  = 2.0
KI_ALT  = 0.3
KD_ALT  = 1.0
I_LIMIT = 0.3

# Attitude PD
KP_ATT = 5.0
KD_ATT = 1.5

# Rotor visual RPM
MIN_RPM   = 2400
MAX_RPM   = 3200
HOVER_RPM = 2537

# Tracking camera
CAM_DISTANCE  = 3.0
CAM_AZIMUTH   = 135.0
CAM_ELEVATION = -25.0
CAM_LOOKAT_OFFSET = np.array([0.0, 0.0, -0.108])

PRINT_INTERVAL = 1.0


# ─── Utilities ───────────────────────────────────────────────────────────────

def quat_to_euler(w, x, y, z):
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return roll, pitch, yaw


def get_rotor_joint_indices(model):
    import mujoco
    top = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "top_rotor_joint")
    bot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottom_rotor_joint")
    return model.jnt_qposadr[top], model.jnt_qposadr[bot]


def thrust_to_rpm(thrust_cmd):
    ff = HOVER_CTRL
    ratio = max(thrust_cmd, 0.01) / max(ff, 0.01)
    rpm = HOVER_RPM * np.sqrt(ratio)
    return np.clip(rpm, MIN_RPM, MAX_RPM)


def update_rotor_visual(data, top_idx, bot_idx, thrust_cmd, dt):
    rpm = thrust_to_rpm(thrust_cmd)
    rad_s = rpm * 2.0 * np.pi / 60.0
    data.qpos[top_idx] += rad_s * dt
    data.qpos[bot_idx] -= rad_s * 1.05 * dt


# ─── Keyboard input (pynput) ────────────────────────────────────────────────

class KeyStateTracker:
    def __init__(self):
        from pynput import keyboard
        self._keyboard = keyboard
        self._pressed = set()
        self._lock = threading.Lock()

        self.SPECIAL_KEYS = {
            keyboard.Key.up:      'up',
            keyboard.Key.down:    'down',
            keyboard.Key.left:    'left',
            keyboard.Key.right:   'right',
            keyboard.Key.space:   'space',
            keyboard.Key.shift:   'shift',
            keyboard.Key.shift_l: 'shift',
            keyboard.Key.shift_r: 'shift',
            keyboard.Key.esc:     'esc',
        }

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()

    def _key_name(self, key):
        if key in self.SPECIAL_KEYS:
            return self.SPECIAL_KEYS[key]
        try:
            return key.char.lower() if key.char else None
        except AttributeError:
            return None

    def _on_press(self, key):
        name = self._key_name(key)
        if name:
            with self._lock:
                self._pressed.add(name)

    def _on_release(self, key):
        name = self._key_name(key)
        if name:
            with self._lock:
                self._pressed.discard(name)

    def get_pressed(self):
        with self._lock:
            return self._pressed.copy()

    def stop(self):
        self._listener.stop()


# ─── Reference generator ────────────────────────────────────────────────────

class ReferenceGenerator:
    def __init__(self, z_init=Z_REF_INIT):
        self.roll_ref  = 0.0
        self.pitch_ref = 0.0
        self.z_ref     = z_init

    def update(self, keys, dt):
        roll_target  = 0.0
        pitch_target = 0.0

        if 'w' in keys or 'up' in keys:
            pitch_target += MAX_TILT
        if 's' in keys or 'down' in keys:
            pitch_target -= MAX_TILT
        if 'a' in keys or 'left' in keys:
            roll_target += MAX_TILT
        if 'd' in keys or 'right' in keys:
            roll_target -= MAX_TILT

        self.roll_ref  = self._smooth(self.roll_ref, roll_target, dt)
        self.pitch_ref = self._smooth(self.pitch_ref, pitch_target, dt)

        if 'space' in keys:
            self.z_ref += ALT_RATE
        if 'shift' in keys:
            self.z_ref -= ALT_RATE
        self.z_ref = np.clip(self.z_ref, Z_MIN, Z_MAX)

        if 'e' in keys:
            self.roll_ref  = 0.0
            self.pitch_ref = 0.0

        return self.roll_ref, self.pitch_ref, self.z_ref

    def _smooth(self, current, target, dt):
        rate = RAMP_RATE * dt if abs(target) > 1e-6 else DECAY_RATE * dt
        diff = target - current
        if abs(diff) < rate:
            return target
        return current + np.sign(diff) * rate


# ─── STL check ───────────────────────────────────────────────────────────────

def ensure_stl_exists(force_regen=False):
    assets_dir = PROJECT_ROOT / "assets"
    top_stl = assets_dir / "optimized_topblades.stl"
    bot_stl = assets_dir / "optimized_bottomblades.stl"
    if top_stl.exists() and bot_stl.exists() and not force_regen:
        return True
    print("  Generating STL files...")
    from generate_blade_stl import main as gen_main
    gen_main()
    return top_stl.exists() and bot_stl.exists()


# ─── Main ────────────────────────────────────────────────────────────────────

def run(regenerate=False):
    import mujoco
    import mujoco.viewer

    print("=" * 70)
    print("  Lunar Environment — Keyboard Flight Control")
    print("=" * 70)
    print()
    print("  W/Up=Forward  S/Down=Backward  A/Left=Left  D/Right=Right")
    print("  Space=Ascend  Shift=Descend  E=Level reset  ESC=Quit")
    print()

    if not ensure_stl_exists(force_regen=regenerate):
        print("ERROR: Could not generate STL files.")
        return

    scene_xml = str(Path(__file__).parent / "models" / "scene_lunar.xml")

    # Load blade info for RPM
    candidates_file = RESULTS_DIR / "pipeline" / "final_candidates.json"
    if candidates_file.exists():
        with open(candidates_file) as f:
            pipeline = json.load(f)
        candidates = pipeline.get('final_candidates', [])
        if candidates:
            blade = BladeDesign(np.array(candidates[0]['design_vector']))
            print(f"  Design: RPM={blade.rpm:.0f}, tip_Mach={blade.tip_mach():.3f}")
    else:
        blade = baseline_design()

    ctrl_ff = HOVER_CTRL
    print(f"  Hover ctrl={ctrl_ff:.4f} ({MARS_WEIGHT/NUM_ROTORS:.3f}N / {THRUST_GEAR}N/ctrl)")
    print(f"  Loading: {scene_xml}")

    model = mujoco.MjModel.from_xml_path(scene_xml)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep

    top_idx, bot_idx = get_rotor_joint_indices(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ingenuity")

    keys = KeyStateTracker()
    ref  = ReferenceGenerator(z_init=Z_REF_INIT)

    z_integral = 0.0
    z_err_prev = 0.0
    step = 0
    next_print = 0.0

    print()
    print("-" * 85)
    print(f"{'Time':>6}  {'Alt':>6}  {'zRef':>5}  "
          f"{'Roll':>6}  {'Pitch':>7}  "
          f"{'rRef':>6}  {'pRef':>7}  "
          f"{'Thr':>5}  {'RPM':>5}  {'Vx':>6}  {'Vy':>6}")
    print("-" * 85)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        # ── Stabilization (0.5s, no keyboard) ──
        for _ in range(int(0.5 / dt)):
            z = data.qpos[2]
            vz = data.qvel[2]
            qw, qx, qy, qz_ = data.qpos[3:7]
            roll, pitch, _ = quat_to_euler(qw, qx, qy, qz_)

            z_err = Z_REF_INIT - z
            z_integral = np.clip(z_integral + z_err * dt, -I_LIMIT, I_LIMIT)
            z_deriv = (z_err - z_err_prev) / dt
            z_err_prev = z_err

            thrust_cmd = np.clip(
                ctrl_ff + KP_ALT * z_err + KI_ALT * z_integral + KD_ALT * z_deriv,
                0.0, 1.0)

            x_cmd = np.clip(-(KP_ATT * roll + KD_ATT * data.qvel[3]), -1.0, 1.0)
            y_cmd = np.clip(-(KP_ATT * pitch + KD_ATT * data.qvel[4]), -1.0, 1.0)
            yaw_cmd = np.clip(-(2.0 * data.qvel[5]), -1.0, 1.0)

            data.ctrl[IDX_THRUST1] = thrust_cmd
            data.ctrl[IDX_THRUST2] = thrust_cmd
            data.ctrl[IDX_ROLL]    = x_cmd
            data.ctrl[IDX_PITCH]   = y_cmd
            if model.nu > IDX_YAW:
                data.ctrl[IDX_YAW] = yaw_cmd

            update_rotor_visual(data, top_idx, bot_idx, thrust_cmd, dt)
            mujoco.mj_step(model, data)

            # Tracking camera
            drone_pos = data.xpos[body_id].copy()
            viewer.cam.lookat[:] = drone_pos + CAM_LOOKAT_OFFSET
            viewer.cam.distance  = CAM_DISTANCE
            viewer.cam.azimuth   = CAM_AZIMUTH
            viewer.cam.elevation = CAM_ELEVATION
            viewer.sync()
            time.sleep(dt)

        print("[INFO] Stabilized. Keyboard active.")
        print()

        # ── Main loop ──
        while viewer.is_running():
            step_start = time.time()

            pressed = keys.get_pressed()
            if 'esc' in pressed:
                print("\n[INFO] ESC — Exiting")
                break

            # Reference
            roll_ref, pitch_ref, z_ref = ref.update(pressed, dt)

            # State
            z = data.qpos[2]
            vz = data.qvel[2]
            qw, qx, qy, qz_ = data.qpos[3:7]
            roll, pitch, yaw = quat_to_euler(qw, qx, qy, qz_)
            vx = data.qvel[0]
            vy = data.qvel[1]

            # Velocity damping (no directional key)
            directional = {'w', 's', 'a', 'd', 'up', 'down', 'left', 'right'}
            if not pressed.intersection(directional):
                pitch_ref += np.clip(-VEL_DAMP_GAIN * vx, -MAX_DAMP_TILT, MAX_DAMP_TILT)
                roll_ref  += np.clip( VEL_DAMP_GAIN * vy, -MAX_DAMP_TILT, MAX_DAMP_TILT)

            # Speed limit
            speed = np.sqrt(vx**2 + vy**2)
            if speed > MAX_SPEED:
                pitch_ref = 0.5 * pitch_ref + 0.5 * np.clip(-0.05 * vx, -MAX_TILT, MAX_TILT)
                roll_ref  = 0.5 * roll_ref  + 0.5 * np.clip( 0.05 * vy, -MAX_TILT, MAX_TILT)

            # Altitude PID + tilt compensation
            z_err = z_ref - z
            z_integral = np.clip(z_integral + z_err * dt, -I_LIMIT, I_LIMIT)
            z_deriv = (z_err - z_err_prev) / dt
            z_err_prev = z_err

            thrust_cmd = ctrl_ff + KP_ALT * z_err + KI_ALT * z_integral + KD_ALT * z_deriv
            tilt_mag = np.sqrt(roll**2 + pitch**2)
            thrust_cmd *= 1.0 / max(np.cos(tilt_mag), 0.85)
            thrust_cmd = np.clip(thrust_cmd, 0.0, 1.0)

            # Attitude PD (reference tracking)
            roll_err  = roll_ref - roll
            pitch_err = pitch_ref - pitch

            x_cmd = np.clip( KP_ATT * roll_err  - KD_ATT * data.qvel[3], -1.0, 1.0)
            y_cmd = np.clip( KP_ATT * pitch_err - KD_ATT * data.qvel[4], -1.0, 1.0)
            yaw_cmd = np.clip(-(2.0 * data.qvel[5]), -1.0, 1.0)

            # Actuators
            data.ctrl[IDX_THRUST1] = thrust_cmd
            data.ctrl[IDX_THRUST2] = thrust_cmd
            data.ctrl[IDX_ROLL]    = x_cmd
            data.ctrl[IDX_PITCH]   = y_cmd
            if model.nu > IDX_YAW:
                data.ctrl[IDX_YAW] = yaw_cmd

            # Rotor visual
            update_rotor_visual(data, top_idx, bot_idx, thrust_cmd, dt)

            # Step
            mujoco.mj_step(model, data)
            step += 1

            # Camera
            drone_pos = data.xpos[body_id].copy()
            viewer.cam.lookat[:] = drone_pos + CAM_LOOKAT_OFFSET
            viewer.cam.distance  = CAM_DISTANCE
            viewer.cam.azimuth   = CAM_AZIMUTH
            viewer.cam.elevation = CAM_ELEVATION
            viewer.sync()

            # Console
            if data.time >= next_print:
                rpm_now = thrust_to_rpm(thrust_cmd)
                print(f"{data.time:6.1f}  {z:6.3f}  {z_ref:5.2f}  "
                      f"{np.degrees(roll):6.2f}  {np.degrees(pitch):7.2f}  "
                      f"{np.degrees(roll_ref):6.2f}  {np.degrees(pitch_ref):7.2f}  "
                      f"{thrust_cmd:5.3f}  {rpm_now:5.0f}  {vx:6.3f}  {vy:6.3f}")
                next_print += PRINT_INTERVAL

            # Realtime pacing
            elapsed = time.time() - step_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    keys.stop()
    print("-" * 85)
    print(f"[DONE] {step} steps | Final altitude: {data.qpos[2]:.4f} m")


def main():
    parser = argparse.ArgumentParser(description="Lunar Environment — Keyboard Flight")
    parser.add_argument("--regenerate", action="store_true",
                        help="Force regenerate blade STL files")
    args = parser.parse_args()
    run(regenerate=args.regenerate)


if __name__ == "__main__":
    main()
