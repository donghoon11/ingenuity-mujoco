"""
compare_dynamic_performance.py
-------------------------------
Baseline vs Optimized Blade — Dynamic Performance Comparison

scene.xml (mhs.xml) 의 기존 블레이드와
scene_optimized.xml (mhs_optimized.xml) 의 최적화 블레이드에
동일한 제어 레퍼런스 신호를 인가하여 동적 성능을 비교한다.

레퍼런스 프로파일:
  Phase 0  ( 0~2s)  : 호버 안정화  z_ref=1.0m
  Phase 1  ( 2~4s)  : Altitude step up     z_ref=1.0→2.0m  (+1m)
  Phase 2  ( 4~7s)  : Altitude hold        z_ref=2.0m
  Phase 3  ( 7~9s)  : Altitude step down   z_ref=2.0→0.5m  (-1.5m)
  Phase 4  ( 9~12s) : Altitude hold        z_ref=0.5m
  Phase 5  (12~14s) : Pitch step (forward) pitch_ref=+8°
  Phase 6  (14~17s) : Pitch hold → decay   pitch_ref→0°
  Phase 7  (17~20s) : Roll step (lateral)  roll_ref=+8°
  Phase 8  (20~23s) : Roll hold → decay    roll_ref→0°
  Phase 9  (23~25s) : Final hover          z_ref=1.0m

출력 메트릭:
  - 고도 추적 오차 (RMSE, 과도 응답 시간)
  - 자세각 (roll/pitch) 추적 정밀도
  - 추력 제어 사용량 (RMS, 포화 횟수)
  - 수평 속도 (drift, 안정성)
  - 에너지 소비 (추력 적분)

사용법:
  cd E:/mujoco_projects/ingenuity-mujoco
  python 0219/design/compare_dynamic_performance.py
  python 0219/design/compare_dynamic_performance.py --no-viewer
"""

import argparse
import sys
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
PROJECT_ROOT = SCRIPT_DIR.parent.parent   # ingenuity-mujoco/

# ── 플롯 출력 디렉토리 ─────────────────────────────────────────────────────────
RESULTS_DIR = SCRIPT_DIR / "results" / "compare"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 모델별 파라미터 정의
# ══════════════════════════════════════════════════════════════════════════════

MODELS = {
    "baseline": {
        "label":       "Baseline (Original)",
        "color":       "#E05C5C",      # 붉은 계열
        "scene_xml":   str(PROJECT_ROOT / "scene.xml"),
        # mhs.xml: density=1.225 (Earth), THRUST_GEAR=50, XY_GEAR=0.09
        "gravity":     9.81,
        "total_mass":  1.6,
        "thrust_gear": 50.0,
        "xy_gear":     0.09,
        "nu":          4,              # actuator 수 (thrust1, thrust2, x, y)
        "idx_thrust1": 0,
        "idx_thrust2": 1,
        "idx_roll":    2,
        "idx_pitch":   3,
        "idx_yaw":     None,
        # PID 게인 (hover_keyboard.py 기준)
        "KP_alt": 0.5,  "KI_alt": 0.1,  "KD_alt": 0.3,  "I_lim": 0.2,
        "KP_att": 3.0,  "KD_att": 0.8,
        "hover_rpm": 2537.0,   # BEMT baseline design RPM
    },
    "optimized": {
        "label":       "Optimized Blade",
        "color":       "#4A90D9",      # 파란 계열
        "scene_xml":   str(SCRIPT_DIR / "models" / "scene_optimized.xml"),
        # mhs_optimized.xml: Earth env, gear scaled by FM improvement
        # FM: baseline=0.3730 → optimized=0.4642 (+24.5%)
        # thrust_gear = 50 * 1.2445 = 62.2 N/ctrl  → hover ff=0.1261 (vs baseline 0.1570)
        "gravity":     9.81,
        "total_mass":  1.6,
        "thrust_gear": 62.2,
        "xy_gear":     0.112,          # 0.09 * 1.2445
        "nu":          4,
        "idx_thrust1": 0,
        "idx_thrust2": 1,
        "idx_roll":    2,
        "idx_pitch":   3,
        "idx_yaw":     None,
        # PID 게인 — baseline과 동일 (동일 환경, PID 튜닝 차이 배제)
        "KP_alt": 0.5,  "KI_alt": 0.1,  "KD_alt": 0.3,  "I_lim": 0.2,
        "KP_att": 3.0,  "KD_att": 0.8,
        # hover_rpm: optimized=3012.84 (baseline=2537 in run_simulation default)
        "hover_rpm": 3012.84,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 레퍼런스 프로파일 생성
# ══════════════════════════════════════════════════════════════════════════════

def build_reference_profile(t_arr: np.ndarray) -> dict:
    """시간 배열에 대해 z_ref, roll_ref, pitch_ref 생성"""
    z_ref     = np.ones_like(t_arr)
    roll_ref  = np.zeros_like(t_arr)
    pitch_ref = np.zeros_like(t_arr)

    for i, t in enumerate(t_arr):
        # ── 고도 프로파일 ──────────────────────────────
        if t < 2.0:
            z = 1.0
        elif t < 4.0:
            # 2→4s: smooth step 1.0→2.0m (S-curve)
            s = (t - 2.0) / 2.0
            s_smooth = 3*s**2 - 2*s**3
            z = 1.0 + 1.0 * s_smooth
        elif t < 7.0:
            z = 2.0
        elif t < 9.0:
            # 7→9s: smooth step 2.0→0.5m
            s = (t - 7.0) / 2.0
            s_smooth = 3*s**2 - 2*s**3
            z = 2.0 - 1.5 * s_smooth
        elif t < 12.0:
            z = 0.5
        elif t < 23.0:
            z = 1.0
        else:
            # 23→25s: return to 1.0 (already 1.0)
            z = 1.0
        z_ref[i] = z

        # ── Pitch step (전진 기동) ──────────────────────
        pitch_deg = 0.0
        if 12.0 <= t < 14.0:
            # step on at t=12s
            pitch_deg = 8.0
        elif 14.0 <= t < 16.0:
            # smooth decay to 0
            s = (t - 14.0) / 2.0
            pitch_deg = 8.0 * (1.0 - (3*s**2 - 2*s**3))
        pitch_ref[i] = np.radians(pitch_deg)

        # ── Roll step (횡 기동) ────────────────────────
        roll_deg = 0.0
        if 17.0 <= t < 19.0:
            roll_deg = 8.0
        elif 19.0 <= t < 21.0:
            s = (t - 19.0) / 2.0
            roll_deg = 8.0 * (1.0 - (3*s**2 - 2*s**3))
        roll_ref[i] = np.radians(roll_deg)

    return {"z_ref": z_ref, "roll_ref": roll_ref, "pitch_ref": pitch_ref}


# ══════════════════════════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════════════════════════

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


def get_sensor_data(model, data):
    import mujoco
    def sensor_val(name):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = model.sensor_adr[sid]
        dim = model.sensor_dim[sid]
        return data.sensordata[adr: adr + dim]
    laser = float(sensor_val("sensor_laser")[0])
    gyro  = sensor_val("sensor_gyro").copy()
    quat  = sensor_val("sensor_quat").copy()
    return laser, gyro, quat


def get_rotor_joint_indices(model):
    import mujoco
    top = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "top_rotor_joint")
    bot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottom_rotor_joint")
    return model.jnt_qposadr[top], model.jnt_qposadr[bot]


# ══════════════════════════════════════════════════════════════════════════════
# 단일 모델 시뮬레이션 실행
# ══════════════════════════════════════════════════════════════════════════════

SIM_DURATION = 25.0    # 총 시뮬레이션 시간 (초)


def run_simulation(cfg: dict, ref_profile: dict, use_viewer: bool = False) -> dict:
    """cfg 파라미터로 시뮬레이션 실행 → 시계열 로그 반환"""
    import mujoco
    import mujoco.viewer

    label = cfg["label"]
    print(f"\n{'='*65}")
    print(f"  시뮬레이션: {label}")
    print(f"  Model: {cfg['scene_xml']}")
    print(f"  gravity={cfg['gravity']:.2f}m/s²  "
          f"thrust_gear={cfg['thrust_gear']:.1f}N/ctrl")
    print(f"{'='*65}")

    model = mujoco.MjModel.from_xml_path(cfg["scene_xml"])
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep

    ff = cfg["total_mass"] * cfg["gravity"] / 2.0 / cfg["thrust_gear"]
    print(f"  Feedforward: {ff:.4f} ctrl/rotor  dt={dt:.4f}s")

    top_idx, bot_idx = get_rotor_joint_indices(model)

    # 로그 배열
    n_steps = int(SIM_DURATION / dt) + 1
    log = {
        "t":              np.zeros(n_steps),
        "z":              np.zeros(n_steps),
        "z_ref":          np.zeros(n_steps),
        "roll":           np.zeros(n_steps),
        "pitch":          np.zeros(n_steps),
        "roll_ref":       np.zeros(n_steps),
        "pitch_ref":      np.zeros(n_steps),
        "thrust_cmd":     np.zeros(n_steps),
        "thrust_excess":  np.zeros(n_steps),  # thrust_cmd - ff (normalized delta)
        "vx":             np.zeros(n_steps),
        "vy":             np.zeros(n_steps),
        "vz":             np.zeros(n_steps),
    }
    log["ff"] = ff  # scalar, for reference

    z_integral = 0.0
    z_err_prev = 0.0
    step = 0

    KP_alt = cfg["KP_alt"]
    KI_alt = cfg["KI_alt"]
    KD_alt = cfg["KD_alt"]
    I_lim  = cfg["I_lim"]
    KP_att = cfg["KP_att"]
    KD_att = cfg["KD_att"]

    # 레퍼런스 보간용 시간 배열
    ref_t = np.linspace(0.0, SIM_DURATION, len(ref_profile["z_ref"]))

    def interp_ref(t_now):
        z_r     = float(np.interp(t_now, ref_t, ref_profile["z_ref"]))
        roll_r  = float(np.interp(t_now, ref_t, ref_profile["roll_ref"]))
        pitch_r = float(np.interp(t_now, ref_t, ref_profile["pitch_ref"]))
        return z_r, roll_r, pitch_r

    def do_step(t_now):
        nonlocal z_integral, z_err_prev, step

        z_r, roll_r, pitch_r = interp_ref(t_now)

        # 센서
        laser, gyro, quat_s = get_sensor_data(model, data)
        w, qx, qy, qz_ = quat_s
        roll, pitch, _ = quat_to_euler(w, qx, qy, qz_)
        p, q, r = gyro

        z_meas = laser if laser > 0.01 else data.qpos[2]
        vx = data.qvel[0]
        vy = data.qvel[1]
        vz = data.qvel[2]

        # 고도 PID
        z_err = z_r - z_meas
        z_integral = np.clip(z_integral + z_err * dt, -I_lim, I_lim)
        z_deriv = (z_err - z_err_prev) / dt
        z_err_prev = z_err

        thrust_cmd = ff + KP_alt * z_err + KI_alt * z_integral + KD_alt * z_deriv

        # 틸트 보상
        tilt = np.sqrt(roll**2 + pitch**2)
        thrust_cmd *= 1.0 / max(np.cos(tilt), 0.85)
        thrust_cmd = np.clip(thrust_cmd, 0.0, 1.0)

        # 자세 PD
        roll_err  = roll_r  - roll
        pitch_err = pitch_r - pitch
        x_cmd = np.clip( KP_att * roll_err  - KD_att * p, -1.0, 1.0)
        y_cmd = np.clip( KP_att * pitch_err - KD_att * q, -1.0, 1.0)

        # 액추에이터 적용
        data.ctrl[cfg["idx_thrust1"]] = thrust_cmd
        data.ctrl[cfg["idx_thrust2"]] = thrust_cmd
        data.ctrl[cfg["idx_roll"]]    = x_cmd
        data.ctrl[cfg["idx_pitch"]]   = y_cmd
        if cfg["idx_yaw"] is not None and model.nu > cfg["idx_yaw"]:
            yaw_cmd = np.clip(-(2.0 * r), -1.0, 1.0)
            data.ctrl[cfg["idx_yaw"]] = yaw_cmd

        # 로터 시각화
        hover_rpm = cfg.get("hover_rpm", 2537)
        ratio = max(thrust_cmd, 0.01) / max(ff, 0.01)
        rpm = np.clip(hover_rpm * np.sqrt(ratio), 2000, 3200)
        rad_s = rpm * 2.0 * np.pi / 60.0
        data.qpos[top_idx] += rad_s * dt
        data.qpos[bot_idx] -= rad_s * 1.05 * dt

        # 로그
        if step < n_steps:
            log["t"][step]             = t_now
            log["z"][step]             = z_meas
            log["z_ref"][step]         = z_r
            log["roll"][step]          = roll
            log["pitch"][step]         = pitch
            log["roll_ref"][step]      = roll_r
            log["pitch_ref"][step]     = pitch_r
            log["thrust_cmd"][step]    = thrust_cmd
            log["thrust_excess"][step] = thrust_cmd - ff   # feedforward 제거한 제어 활동량
            log["vx"][step]            = vx
            log["vy"][step]            = vy
            log["vz"][step]            = vz

        mujoco.mj_step(model, data)
        step += 1

    print(f"  Running {SIM_DURATION:.0f}s simulation ({int(SIM_DURATION/dt)} steps)...")
    t0 = time.time()

    if use_viewer:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ingenuity")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while data.time < SIM_DURATION and viewer.is_running():
                do_step(data.time)
                # 카메라 추적
                drone_pos = data.xpos[body_id].copy()
                viewer.cam.lookat[:] = drone_pos + np.array([0, 0, -0.1])
                viewer.cam.distance = 4.0
                viewer.cam.azimuth = 135.0
                viewer.cam.elevation = -20.0
                viewer.sync()
                time.sleep(dt)
    else:
        while data.time < SIM_DURATION:
            do_step(data.time)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  ({step} steps)")

    # 실제 기록된 스텝 수만 반환 (scalar 항목 제외)
    n_rec = min(step, n_steps)
    scalar_keys = {"ff"}
    for k in log:
        if k not in scalar_keys:
            log[k] = log[k][:n_rec]

    return log


# ══════════════════════════════════════════════════════════════════════════════
# 성능 메트릭 계산
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(log: dict) -> dict:
    t          = log["t"]
    z          = log["z"]
    z_ref      = log["z_ref"]
    roll       = log["roll"]
    pitch      = log["pitch"]
    roll_ref   = log["roll_ref"]
    pitch_ref  = log["pitch_ref"]
    thrust_cmd = log["thrust_cmd"]
    vx         = log["vx"]
    vy         = log["vy"]

    z_err   = z - z_ref
    att_err = np.sqrt((roll - roll_ref)**2 + (pitch - pitch_ref)**2)

    # 고도 추적 RMSE (전체)
    z_rmse = float(np.sqrt(np.mean(z_err**2)))

    # 고도 step 구간 과도 응답 (2~7s: step up + hold)
    mask_step = (t >= 2.0) & (t <= 7.0)
    z_rmse_step = float(np.sqrt(np.mean(z_err[mask_step]**2))) if mask_step.any() else np.nan

    # 자세 RMSE (전체, 기동 구간 12~21s)
    att_rmse = float(np.sqrt(np.mean(att_err**2)))
    mask_att = (t >= 12.0) & (t <= 21.0)
    att_rmse_manuever = float(np.sqrt(np.mean(att_err[mask_att]**2))) if mask_att.any() else np.nan

    # 추력 사용량
    thrust_rms  = float(np.sqrt(np.mean(thrust_cmd**2)))
    thrust_sat  = int(np.sum(thrust_cmd >= 0.99))  # 포화 스텝 수
    # feedforward 제거 후 제어 활동량 (환경 무관 비교)
    thrust_excess_rms = float(np.sqrt(np.mean(log.get("thrust_excess", thrust_cmd - log.get("ff", 0))**2)))

    # 수평 드리프트 (호버 구간 4~7s)
    mask_hover = (t >= 4.0) & (t <= 7.0)
    h_speed = np.sqrt(vx**2 + vy**2)
    drift_rms = float(np.sqrt(np.mean(h_speed[mask_hover]**2))) if mask_hover.any() else np.nan

    # 에너지 (추력 적분 ≈ 비례)
    dt_arr = np.diff(t, prepend=t[0])
    energy = float(np.sum(thrust_cmd * dt_arr))

    # 안정화 시간 (step up, z가 1.9m에 최초 도달하는 시각 − 2.0s)
    mask_settle = (t >= 2.0) & (z >= 1.9)
    settle_time = float(t[mask_settle][0] - 2.0) if mask_settle.any() else np.nan

    return {
        "z_rmse_total":      z_rmse,
        "z_rmse_step":       z_rmse_step,
        "settle_time_s":     settle_time,
        "att_rmse_total":    att_rmse,
        "att_rmse_maneuver": att_rmse_manuever,
        "thrust_rms":        thrust_rms,
        "thrust_excess_rms": thrust_excess_rms,   # ff 제거 → 환경 무관 활동량
        "thrust_sat_steps":  thrust_sat,
        "drift_rms_ms":      drift_rms,
        "energy_integral":   energy,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 플롯 생성
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(logs: dict, metrics: dict):
    """overlay 비교 그래프 생성 및 저장"""
    fig = plt.figure(figsize=(18, 22))
    fig.suptitle("Dynamic Performance Comparison\nBaseline vs Optimized Blade",
                 fontsize=15, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(5, 2, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.08, right=0.97,
                           top=0.94, bottom=0.05)

    # ── 1. 고도 추적 ──────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for key, log in logs.items():
        cfg = MODELS[key]
        ax1.plot(log["t"], log["z"], color=cfg["color"],
                 lw=1.8, label=cfg["label"])
    # z_ref (공통)
    first_log = next(iter(logs.values()))
    ax1.plot(first_log["t"], first_log["z_ref"], 'k--', lw=1.2, alpha=0.7,
             label="z_ref")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Altitude (m)")
    ax1.set_title("Altitude Tracking")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    _shade_phases(ax1)

    # ── 2. 고도 오차 ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    for key, log in logs.items():
        cfg = MODELS[key]
        z_err = log["z"] - log["z_ref"]
        ax2.plot(log["t"], z_err, color=cfg["color"], lw=1.5, label=cfg["label"])
    ax2.axhline(0, color='k', lw=0.8, ls='--')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Altitude Error (m)")
    ax2.set_title("Altitude Tracking Error  (z − z_ref)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    _shade_phases(ax2)

    # ── 3. Roll 추적 ──────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    for key, log in logs.items():
        cfg = MODELS[key]
        ax3.plot(log["t"], np.degrees(log["roll"]),
                 color=cfg["color"], lw=1.5, label=cfg["label"])
    ax3.plot(first_log["t"], np.degrees(first_log["roll_ref"]),
             'k--', lw=1.2, alpha=0.7, label="roll_ref")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Roll (°)")
    ax3.set_title("Roll Tracking")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── 4. Pitch 추적 ─────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    for key, log in logs.items():
        cfg = MODELS[key]
        ax4.plot(log["t"], np.degrees(log["pitch"]),
                 color=cfg["color"], lw=1.5, label=cfg["label"])
    ax4.plot(first_log["t"], np.degrees(first_log["pitch_ref"]),
             'k--', lw=1.2, alpha=0.7, label="pitch_ref")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Pitch (°)")
    ax4.set_title("Pitch Tracking")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── 5. 추력 초과량 (feedforward 제거, 환경 무관 비교) ─────────────────────
    ax5 = fig.add_subplot(gs[3, 0])
    for key, log in logs.items():
        cfg = MODELS[key]
        ax5.plot(log["t"], log["thrust_excess"], color=cfg["color"],
                 lw=1.5, label=f"{cfg['label']} (ff={log['ff']:.3f})")
    ax5.axhline(0, color='k', lw=0.8, ls='--')
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Thrust − ff (ctrl)")
    ax5.set_title("Thrust Excess (ff removed)\nComparable across environments")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    _shade_phases(ax5)

    # ── 6. 수평 속도 ──────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, 1])
    for key, log in logs.items():
        cfg = MODELS[key]
        h_speed = np.sqrt(log["vx"]**2 + log["vy"]**2)
        ax6.plot(log["t"], h_speed, color=cfg["color"],
                 lw=1.5, label=cfg["label"])
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Horizontal Speed (m/s)")
    ax6.set_title("Horizontal Speed (Drift)")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # ── 7. 메트릭 테이블 (bar chart) ─────────────────────────────────────────
    ax7 = fig.add_subplot(gs[4, :])
    ax7.axis('off')

    metric_labels = [
        ("Alt RMSE (total)", "z_rmse_total", "m", 4),
        ("Alt RMSE (step)", "z_rmse_step", "m", 4),
        ("Settle Time", "settle_time_s", "s", 3),
        ("Att RMSE (total)", "att_rmse_total", "°", 3),
        ("Att RMSE (maneuver)", "att_rmse_maneuver", "°", 3),
        ("Thrust Excess RMS", "thrust_excess_rms", "", 4),
        ("Thrust Sat Steps", "thrust_sat_steps", "", 0),
        ("Drift RMS", "drift_rms_ms", "m/s", 4),
        ("Energy Integral", "energy_integral", "ctrl·s", 2),
    ]

    keys_list = list(logs.keys())
    col_headers = ["Metric"] + [MODELS[k]["label"] for k in keys_list] + ["Better"]
    rows = []
    for label, mkey, unit, dp in metric_labels:
        row = [f"{label} [{unit}]" if unit else label]
        vals = [metrics[k][mkey] for k in keys_list]
        for v in vals:
            if dp == 0:
                row.append(f"{int(v)}" if not np.isnan(v) else "N/A")
            else:
                row.append(f"{v:.{dp}f}" if not np.isnan(v) else "N/A")
        # 'Better' 표시 (낮을수록 좋음)
        if not any(np.isnan(v) for v in vals):
            better_idx = int(np.argmin(vals))
            row.append(MODELS[keys_list[better_idx]]["label"].split()[0])
        else:
            row.append("–")
        rows.append(row)

    tbl = ax7.table(
        cellText=rows, colLabels=col_headers,
        cellLoc='center', loc='center',
        bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    # 헤더 배경색
    for j in range(len(col_headers)):
        tbl[0, j].set_facecolor('#2C3E50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    # 데이터 행 색상
    for i, row in enumerate(rows):
        for j in range(len(col_headers)):
            if j == 0:
                tbl[i+1, j].set_facecolor('#ECF0F1')
            elif j == len(col_headers) - 1:
                tbl[i+1, j].set_facecolor('#EBF5FB')
    ax7.set_title("Performance Metrics Summary", fontweight='bold', pad=8)

    out_path = RESULTS_DIR / "comparison_overlay.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved: {out_path}")
    return fig


def _shade_phases(ax):
    """레퍼런스 프로파일 구간을 배경 음영으로 표시"""
    phases = [
        (0,  2,  0.92, "Stabilize"),
        (2,  7,  0.88, "Alt Step"),
        (7,  12, 0.92, "Alt Hold"),
        (12, 16, 0.88, "Pitch"),
        (17, 21, 0.88, "Roll"),
        (23, 25, 0.92, "Final"),
    ]
    colors = ['#D5E8D4', '#DAE8FC', '#D5E8D4', '#FFE6CC', '#FFE6CC', '#D5E8D4']
    for (ts, te, _, ph), c in zip(phases, colors):
        ax.axvspan(ts, te, alpha=0.25, color=c, zorder=0)
        ax.text((ts + te) / 2, ax.get_ylim()[1] * 0.97, ph,
                ha='center', va='top', fontsize=6.5, color='#555555')


# ══════════════════════════════════════════════════════════════════════════════
# 추가: 메트릭 bar chart (오버레이 보조 플롯)
# ══════════════════════════════════════════════════════════════════════════════

def plot_metric_bars(metrics: dict):
    """핵심 메트릭 bar chart 비교"""
    keys_list = list(metrics.keys())
    labels    = [MODELS[k]["label"] for k in keys_list]
    colors    = [MODELS[k]["color"] for k in keys_list]

    bar_metrics = [
        ("z_rmse_total",      "Alt RMSE (total) [m]"),
        ("z_rmse_step",       "Alt RMSE (step zone) [m]"),
        ("settle_time_s",     "Settle Time (step up) [s]"),
        ("att_rmse_maneuver", "Att RMSE (maneuver) [°]"),
        ("thrust_excess_rms", "Thrust Excess RMS\n[ctrl, ff removed]"),
        ("drift_rms_ms",      "Drift RMS [m/s]"),
        ("energy_integral",   "Energy Integral [ctrl·s]"),
    ]

    n = len(bar_metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5))
    fig.suptitle("Performance Metric Comparison\nBaseline vs Optimized Blade",
                 fontsize=13, fontweight='bold')

    for ax, (mkey, mlabel) in zip(axes, bar_metrics):
        vals = [metrics[k][mkey] for k in keys_list]
        bars = ax.bar(labels, vals, color=colors, edgecolor='white',
                      linewidth=1.2, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{v:.3f}" if v < 100 else f"{v:.1f}",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_title(mlabel, fontsize=9, pad=6)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_facecolor('#FAFAFA')
        # 더 낮은 값 강조
        best_idx = int(np.argmin(vals))
        bars[best_idx].set_edgecolor('#27AE60')
        bars[best_idx].set_linewidth(3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = RESULTS_DIR / "metric_bars.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {out_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Baseline vs Optimized — Dynamic Performance Comparison")
    parser.add_argument("--viewer", action="store_true",
                        help="Open MuJoCo viewer during simulation")
    parser.add_argument("--no-viewer", action="store_true",
                        help="Headless (default)")
    parser.add_argument("--only", choices=["baseline", "optimized"],
                        help="Run only one model")
    args = parser.parse_args()

    use_viewer = args.viewer and not args.no_viewer

    print("=" * 65)
    print("  Dynamic Performance Comparison")
    print("  Baseline (scene.xml) vs Optimized (scene_optimized.xml)")
    print("=" * 65)

    # 레퍼런스 프로파일 생성
    t_ref = np.linspace(0, SIM_DURATION, int(SIM_DURATION / 0.008) + 1)
    ref_profile = build_reference_profile(t_ref)

    print(f"\n  Reference profile ({SIM_DURATION:.0f}s):")
    print(f"    0-2s  : Hover stabilize (z=1.0m)")
    print(f"    2-7s  : Altitude step up (1.0→2.0m) + hold")
    print(f"    7-9s  : Altitude step down (2.0→0.5m)")
    print(f"    9-12s : Altitude hold (z=0.5m)")
    print(f"    12-16s: Pitch step (+8°) + decay")
    print(f"    17-21s: Roll step (+8°) + decay")
    print(f"    23-25s: Final hover (z=1.0m)")

    # 시뮬레이션 실행
    run_keys = list(MODELS.keys())
    if args.only:
        run_keys = [args.only]

    logs    = {}
    metrics = {}
    for key in run_keys:
        log = run_simulation(MODELS[key], ref_profile, use_viewer=use_viewer)
        logs[key]    = log
        metrics[key] = compute_metrics(log)

    # 메트릭 출력
    print("\n" + "=" * 65)
    print("  Performance Metrics Summary")
    print("=" * 65)
    metric_print = [
        ("z_rmse_total",      "Alt RMSE (total)",         "m"),
        ("z_rmse_step",       "Alt RMSE (step zone)",     "m"),
        ("settle_time_s",     "Settle Time",              "s"),
        ("att_rmse_maneuver", "Att RMSE (maneuver)",      "deg"),
        ("thrust_excess_rms", "Thrust Excess RMS (ff rem)","ctrl"),
        ("thrust_sat_steps",  "Thrust Saturation Steps",  "steps"),
        ("drift_rms_ms",      "Horizontal Drift RMS",     "m/s"),
        ("energy_integral",   "Energy Integral",          "ctrl·s"),
    ]
    hdr = f"  {'Metric':<30}"
    for k in run_keys:
        hdr += f"  {MODELS[k]['label']:>20}"
    print(hdr)
    print("  " + "-" * (30 + 22 * len(run_keys)))
    for mkey, mlabel, unit in metric_print:
        row = f"  {mlabel+' ['+unit+']':<30}"
        for k in run_keys:
            v = metrics[k][mkey]
            if mkey == "thrust_sat_steps":
                row += f"  {int(v):>20d}"
            else:
                row += f"  {v:>20.4f}"
        print(row)

    # 플롯 생성
    if len(logs) == 2:
        print("\n  Generating comparison plots...")
        plot_comparison(logs, metrics)
        plot_metric_bars(metrics)
    elif len(logs) == 1:
        key = run_keys[0]
        print(f"\n  Single model run: {MODELS[key]['label']}")
        print(f"  (Run both models for comparison plots)")

    # JSON 저장
    summary = {}
    for k in run_keys:
        summary[k] = {
            "label":   MODELS[k]["label"],
            "metrics": {mk: float(mv) for mk, mv in metrics[k].items()},
        }
    out_json = RESULTS_DIR / "comparison_metrics.json"
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Metrics saved: {out_json}")

    print("\n  Done.")
    print(f"  Results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
