"""
hover_basic.py
--------------
NASA Ingenuity 화성 헬리콥터 MuJoCo 시뮬레이션 - 기본 호버링

모델: ingenuity-mujoco/scene.xml (mhs.xml 포함)

액추에이터:
  [0] thrust1    - 상부 로터 Z 추력  (gear=50 N/ctrl, range [-1,1])
  [1] thrust2    - 하부 로터 Z 추력  (gear=50 N/ctrl, range [-1,1])
  [2] x_movement - X 횡이동력        (gear=0.09 N/ctrl, range [-1,1])
  [3] y_movement - Y 횡이동력        (gear=0.09 N/ctrl, range [-1,1])

센서:
  sensor_laser  - 레이저 거리계 (고도, m)
  sensor_gyro   - 자이로 (각속도, rad/s)
  sensor_acc    - 가속도계 (m/s²)
  sensor_quat   - 쿼터니언 자세 (w, x, y, z)

제어:
  - 고도: PID (feedforward + feedback)
  - 자세: PD (roll/pitch → x_movement/y_movement)

사용법:
  conda activate mujoco_air
  python 0219/hover_basic.py          # 헤드리스
  python 0219/hover_basic.py --viewer # MuJoCo 뷰어 시각화
"""

import argparse
import time
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path

# ─── 경로 설정 ─────────────────────────────────────────────────────────────────
SCENE_XML = Path(__file__).parent.parent / "scene.xml"

# ─── 시뮬레이션 파라미터 ──────────────────────────────────────────────────────
SIM_DURATION   = 20.0   # 총 시뮬레이션 시간 (초)
Z_REF          = 1.0    # 목표 고도 (m)
PRINT_INTERVAL = 0.5    # 콘솔 출력 주기 (초)

# ─── 물리 상수 ────────────────────────────────────────────────────────────────
GRAVITY        = 9.81   # m/s²
TOTAL_MASS     = 1.6    # kg (바디 1.0 + 다리 0.4 + 로터 0.2)
HOVER_THRUST   = TOTAL_MASS * GRAVITY  # 총 호버 추력 (N) ≈ 15.7 N
THRUST_GEAR    = 50.0   # N per ctrl unit (thrust1, thrust2)
XY_GEAR        = 0.09   # N per ctrl unit (x_movement, y_movement)
SIM_SPEED      = 0.5    # 시뮬레이션 속도 배율 (1.0=실시간, 0.5=절반)

# ─── 로터 시각화 ──────────────────────────────────────────────────────────────
BASE_RPM       = 2500   # 기본 로터 RPM
BASE_RAD_S     = BASE_RPM * 2.0 * np.pi / 60.0  # ≈ 261.8 rad/s

# ─── 추적 카메라 설정 ─────────────────────────────────────────────────────────
CAM_DISTANCE   = 3.0    # 드론으로부터 카메라 거리 (m)
CAM_AZIMUTH    = 135.0  # 수평 회전각 (도) — 정면 비스듬히
CAM_ELEVATION  = -25.0  # 앙각 (도, 음수=내려다봄)
# 무게중심 오프셋 (mhs.xml 분석값, 로컬 바디 프레임 Z축 기준)
CAM_LOOKAT_OFFSET = np.array([0.0, 0.0, -0.108])  # CoM Z ≈ -0.108 m

# ─── 제어 게인 ────────────────────────────────────────────────────────────────
# 고도 PID
KP_ALT = 0.5
KI_ALT = 0.1
KD_ALT = 0.3
I_LIMIT = 0.2  # 적분 와인드업 제한

# 자세 PD (roll → x_movement, pitch → y_movement)
KP_ATT = 2.0
KD_ATT = 0.5

# ─── 외란(Gust) 기본값 ──────────────────────────────────────────────────────
GUST_T0 = 6.0     # 시작 시각 (초)
GUST_DT = 2.0     # 지속 시간 (초)
GUST_FX = 0.005   # X방향 힘 (N)  — 극미세 외란
GUST_FY = 0.0     # Y방향 힘 (N)
GUST_FZ = 0.0     # Z방향 힘 (N)


def quat_to_euler(w, x, y, z):
    """쿼터니언(w,x,y,z) → 오일러각(roll, pitch, yaw) [rad]"""
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def get_sensor_data(model, data):
    """센서 인덱스를 이름으로 찾아 값을 반환"""
    def sensor_val(name):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = model.sensor_adr[sid]
        dim = model.sensor_dim[sid]
        return data.sensordata[adr: adr + dim]

    laser = float(sensor_val("sensor_laser")[0])
    gyro  = sensor_val("sensor_gyro").copy()   # [p, q, r] rad/s
    acc   = sensor_val("sensor_acc").copy()    # [ax, ay, az] m/s²
    quat  = sensor_val("sensor_quat").copy()   # [w, x, y, z]

    return laser, gyro, acc, quat


def apply_gust(model, data, t, gust_cfg):
    """시간 구간 내에서 ingenuity 바디에 외력 인가 (mj_applyFT)"""
    if not gust_cfg["enabled"]:
        return False
    t0 = gust_cfg["t0"]
    if t0 <= t < t0 + gust_cfg["dt"]:
        force  = np.array([gust_cfg["fx"], gust_cfg["fy"], gust_cfg["fz"]])
        torque = np.zeros(3)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ingenuity")
        mujoco.mj_applyFT(model, data, force, torque,
                          data.xpos[body_id], body_id, data.qfrc_applied)
        return True
    return False


def get_rotor_joint_indices(model):
    """상/하 로터 hinge joint의 qpos 인덱스 반환"""
    top_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "top_rotor_joint")
    bot_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottom_rotor_joint")
    return model.jnt_qposadr[top_jid], model.jnt_qposadr[bot_jid]


def update_rotor_visual(data, top_idx, bot_idx, thrust_cmd, dt):
    """로터 블레이드 회전 업데이트 (시각 전용, qpos 직접 조작)"""
    speed = BASE_RAD_S * max(thrust_cmd, 0.05)  # 최소 5% 유지
    data.qpos[top_idx] += speed * dt          # 상부: CW
    data.qpos[bot_idx] -= speed * 1.1 * dt    # 하부: CCW (10% 빠름, 토크 균형)


def control_step(model, data, z_integral, z_err_prev, dt, ff_each,
                 top_idx, bot_idx):
    """한 스텝 제어 + 로터 시각 업데이트. 업데이트된 상태 반환."""
    laser, gyro, _, quat = get_sensor_data(model, data)
    w, qx, qy, qz = quat
    roll, pitch, _ = quat_to_euler(w, qx, qy, qz)
    p, q, _ = gyro

    # 고도 PID
    z_meas     = laser if laser > 0 else data.qpos[2]
    z_err      = Z_REF - z_meas
    z_integral = np.clip(z_integral + z_err * dt, -I_LIMIT, I_LIMIT)
    z_deriv    = (z_err - z_err_prev) / dt

    thrust_cmd = np.clip(
        ff_each + KP_ALT * z_err + KI_ALT * z_integral + KD_ALT * z_deriv,
        0.0, 1.0
    )

    # 자세 PD
    x_cmd = np.clip(-(KP_ATT * roll  + KD_ATT * p), -1.0, 1.0)
    y_cmd = np.clip(-(KP_ATT * pitch + KD_ATT * q), -1.0, 1.0)

    data.ctrl[0] = thrust_cmd
    data.ctrl[1] = thrust_cmd
    data.ctrl[2] = x_cmd
    data.ctrl[3] = y_cmd

    # 로터 블레이드 회전 시각화
    update_rotor_visual(data, top_idx, bot_idx, thrust_cmd, dt)

    return z_integral, z_err, z_meas, thrust_cmd, x_cmd, y_cmd, roll, pitch


def print_header():
    print("-" * 65)
    print(f"{'Time':>6}  {'Alt(m)':>7}  {'Roll°':>6}  {'Pitch°':>7}  "
          f"{'T1':>5}  {'T2':>5}  {'Xm':>6}  {'Ym':>6}")
    print("-" * 65)


def update_tracking_camera(viewer, data, body_id):
    """드론 무게중심을 lookat으로 설정하는 추적 카메라 업데이트"""
    # 드론 월드 위치 (freejoint → qpos[0:3])
    drone_pos = data.xpos[body_id].copy()
    lookat = drone_pos + CAM_LOOKAT_OFFSET

    viewer.cam.lookat[:] = lookat
    viewer.cam.distance   = CAM_DISTANCE
    viewer.cam.azimuth    = CAM_AZIMUTH
    viewer.cam.elevation  = CAM_ELEVATION


def sim_loop(model, data, dt, ff_each, top_idx, bot_idx, gust_cfg, viewer=None,
             body_id=None):
    """공통 시뮬레이션 루프 (뷰어/헤드리스 겸용)"""
    z_integral = 0.0
    z_err_prev = 0.0
    next_print = 0.0
    step       = 0

    def running():
        if viewer is not None:
            return data.time < SIM_DURATION and viewer.is_running()
        return data.time < SIM_DURATION

    while running():
        # 제어
        z_integral, z_err_prev, z_meas, thrust_cmd, x_cmd, y_cmd, roll, pitch = \
            control_step(model, data, z_integral, z_err_prev, dt, ff_each, top_idx, bot_idx)

        # 외란 인가 (mj_step 전)
        gust_active = apply_gust(model, data, data.time, gust_cfg)

        mujoco.mj_step(model, data)
        step += 1

        if viewer is not None:
            # 추적 카메라 업데이트
            if body_id is not None:
                update_tracking_camera(viewer, data, body_id)
            viewer.sync()
            # 속도 제어: dt/SIM_SPEED 만큼 대기 (SIM_SPEED < 1 → 느리게)
            time.sleep(dt / SIM_SPEED)

        # 콘솔 출력
        if data.time >= next_print:
            mark = " *GUST*" if gust_active else ""
            print(f"{data.time:6.2f}  {z_meas:7.4f}  "
                  f"{np.degrees(roll):6.2f}  {np.degrees(pitch):7.2f}  "
                  f"{thrust_cmd:5.3f}  {thrust_cmd:5.3f}  "
                  f"{x_cmd:6.3f}  {y_cmd:6.3f}{mark}")
            next_print += PRINT_INTERVAL

    return step


def run(use_viewer=False, gust_cfg=None):
    # ─── 모델 로드 ──────────────────────────────────────────────────────────
    print(f"[INFO] Loading model: {SCENE_XML}")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data  = mujoco.MjData(model)

    dt      = model.opt.timestep
    ff_each = HOVER_THRUST / 2 / THRUST_GEAR
    top_idx, bot_idx = get_rotor_joint_indices(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ingenuity")

    print(f"[INFO] 총 질량: {TOTAL_MASS:.2f} kg  |  호버 추력: {HOVER_THRUST:.2f} N")
    print(f"[INFO] 각 로터 feedforward: {ff_each:.4f} ctrl")
    print(f"[INFO] 로터 RPM: {BASE_RPM} (상부 CW, 하부 CCW×1.1)")
    print(f"[INFO] 시뮬레이션: {SIM_DURATION:.1f}s  |  목표 고도: {Z_REF:.1f}m")
    print(f"[INFO] 뷰어: {'ON' if use_viewer else 'OFF (헤드리스)'}")
    if gust_cfg["enabled"]:
        print(f"[INFO] 외란: t={gust_cfg['t0']:.1f}~{gust_cfg['t0']+gust_cfg['dt']:.1f}s  "
              f"F=({gust_cfg['fx']:.2f}, {gust_cfg['fy']:.2f}, {gust_cfg['fz']:.2f}) N")
    print_header()

    if use_viewer:
        with mujoco.viewer.launch_passive(model, data) as v:
            step = sim_loop(model, data, dt, ff_each, top_idx, bot_idx, gust_cfg,
                            viewer=v, body_id=body_id)
    else:
        step = sim_loop(model, data, dt, ff_each, top_idx, bot_idx, gust_cfg)

    print("-" * 65)
    print(f"[DONE] {step} steps  |  최종 고도: {data.qpos[2]:.4f} m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingenuity 호버링 시뮬레이션")
    parser.add_argument("--viewer",  action="store_true", help="MuJoCo 3D 뷰어 활성화")
    parser.add_argument("--gust",    action="store_true", help="돌풍 외란 활성화")
    parser.add_argument("--gust_t0", type=float, default=GUST_T0, help="외란 시작 시각 (초)")
    parser.add_argument("--gust_dt", type=float, default=GUST_DT, help="외란 지속 시간 (초)")
    parser.add_argument("--gust_fx", type=float, default=GUST_FX, help="X방향 힘 (N)")
    parser.add_argument("--gust_fy", type=float, default=GUST_FY, help="Y방향 힘 (N)")
    parser.add_argument("--gust_fz", type=float, default=GUST_FZ, help="Z방향 힘 (N)")
    args = parser.parse_args()

    gust_cfg = {
        "enabled": args.gust,
        "t0": args.gust_t0,
        "dt": args.gust_dt,
        "fx": args.gust_fx,
        "fy": args.gust_fy,
        "fz": args.gust_fz,
    }
    run(use_viewer=args.viewer, gust_cfg=gust_cfg)
