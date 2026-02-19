"""
hover_keyboard.py
-----------------
NASA Ingenuity 화성 헬리콥터 MuJoCo 시뮬레이션 - 키보드 비행 제어

비행 메커니즘:
  실제 Ingenuity는 스와시플레이트 기반 사이클릭 피치 제어로
  로터 디스크를 기울여 기체를 틸트시키고, 추력 벡터의 수평 성분으로 이동한다.
  이 시뮬레이션에서는 키보드로 목표 roll/pitch 각도를 명령하여
  해당 물리적 메커니즘을 재현한다.

제어 아키텍처:
  키보드 → 목표 roll/pitch 각도 → 스무딩 → 자세 PD 추적 → 액추에이터
                                                ↕
                                     고도 PID (자동, 틸트 보상 포함)

키 매핑:
  W / ↑ : 전진 (+X)   pitch_ref = +max_tilt
  S / ↓ : 후진 (-X)   pitch_ref = -max_tilt
  A / ← : 좌이동 (-Y) roll_ref  = +max_tilt
  D / → : 우이동 (+Y) roll_ref  = -max_tilt
  Space  : 상승        z_ref += 0.005/step
  Shift  : 하강        z_ref -= 0.005/step
  E      : 리셋 (수평 복귀 + 고도 유지)
  ESC    : 종료

사용법:
  conda activate mujoco_air
  python 0219/hover_keyboard.py
"""

import time
import threading
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
from pynput import keyboard

# ─── 경로 설정 ─────────────────────────────────────────────────────────────────
SCENE_XML = Path(__file__).parent.parent / "scene.xml"

# ─── 시뮬레이션 파라미터 ──────────────────────────────────────────────────────
Z_REF_INIT     = 1.0    # 초기 목표 고도 (m)
Z_MIN          = 0.3    # 최소 고도 (m)
Z_MAX          = 5.0    # 최대 고도 (m)
SIM_SPEED      = 1.0    # 실시간 (키보드 반응을 위해)
PRINT_INTERVAL = 1.0    # 콘솔 출력 주기 (초)

# ─── 물리 상수 ────────────────────────────────────────────────────────────────
GRAVITY      = 9.81
TOTAL_MASS   = 1.6      # kg
HOVER_THRUST = TOTAL_MASS * GRAVITY
THRUST_GEAR  = 50.0     # N per ctrl unit
XY_GEAR      = 0.09     # N per ctrl unit

# ─── 로터 시각화 / RPM 동적 모델 ───────────────────────────────────────────────
#  실제 Ingenuity 운용 RPM: 2400~2800 (호버 ~2537, 기동 시 증가)
#  thrust_cmd에 비례하여 RPM을 동적으로 계산
MIN_RPM    = 2400
MAX_RPM    = 2800
HOVER_RPM  = 2537    # 호버 기준 RPM
MIN_RAD_S  = MIN_RPM * 2.0 * np.pi / 60.0   # ≈ 251.3 rad/s
MAX_RAD_S  = MAX_RPM * 2.0 * np.pi / 60.0   # ≈ 293.2 rad/s
HOVER_RAD_S = HOVER_RPM * 2.0 * np.pi / 60.0 # ≈ 265.6 rad/s
# 이전 호환용 (hover_basic.py와 동일)
BASE_RPM   = 2500
BASE_RAD_S = BASE_RPM * 2.0 * np.pi / 60.0

# ─── 추적 카메라 ──────────────────────────────────────────────────────────────
CAM_DISTANCE      = 3.0
CAM_AZIMUTH       = 135.0
CAM_ELEVATION     = -25.0
CAM_LOOKAT_OFFSET = np.array([0.0, 0.0, -0.108])

# ─── 제어 게인 ────────────────────────────────────────────────────────────────
# 고도 PID
KP_ALT  = 0.5
KI_ALT  = 0.1
KD_ALT  = 0.3
I_LIMIT = 0.2

# 자세 PD (강화: 레퍼런스 추적용)
KP_ATT = 3.0
KD_ATT = 0.8

# ─── 레퍼런스 생성 파라미터 ───────────────────────────────────────────────────
MAX_TILT    = np.radians(12.0)   # 최대 틸트 각도 (rad) — 논문 기반 ±10° + 여유
RAMP_RATE   = np.radians(60.0)   # 틸트 증가 속도 (rad/s)
DECAY_RATE  = np.radians(90.0)   # 틸트 감소 속도 (rad/s) — 빠른 복원
ALT_RATE    = 0.005              # 고도 변경 속도 (m/step)

# ─── 속도 감쇠 ────────────────────────────────────────────────────────────────
VEL_DAMP_GAIN = 0.02    # 속도 감쇠 계수
MAX_DAMP_TILT = np.radians(5.0)  # 감쇠 틸트 최대값

# ─── 속도 제한 ────────────────────────────────────────────────────────────────
MAX_SPEED = 5.0   # m/s


# ═══════════════════════════════════════════════════════════════════════════════
# 유틸리티 함수 (hover_basic.py와 동일)
# ═══════════════════════════════════════════════════════════════════════════════

def quat_to_euler(w, x, y, z):
    """쿼터니언(w,x,y,z) → 오일러각(roll, pitch, yaw) [rad]"""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

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
    gyro  = sensor_val("sensor_gyro").copy()
    acc   = sensor_val("sensor_acc").copy()
    quat  = sensor_val("sensor_quat").copy()
    return laser, gyro, acc, quat


def get_rotor_joint_indices(model):
    """상/하 로터 hinge joint의 qpos 인덱스 반환"""
    top_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "top_rotor_joint")
    bot_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottom_rotor_joint")
    return model.jnt_qposadr[top_jid], model.jnt_qposadr[bot_jid]


def thrust_to_rpm(thrust_cmd):
    """thrust_cmd(0~1) → RPM 변환 (실제 Ingenuity 운용 범위 매핑)

    물리적 근거: 추력 T ∝ Ω² (로터 추력은 RPM 제곱에 비례)
    따라서 RPM ∝ √(thrust_cmd)

    hover thrust_cmd ≈ 0.157 → HOVER_RPM(2537)
    최대 thrust_cmd = 1.0 → MAX_RPM(2800)
    최소 thrust_cmd = 0.0 → MIN_RPM(2400, 아이들)
    """
    # 호버 기준 정규화 후 RPM 매핑
    ff_each = HOVER_THRUST / 2 / THRUST_GEAR  # ≈ 0.157
    ratio = max(thrust_cmd, 0.01) / ff_each   # 호버 대비 비율 (1.0 = 호버)
    # T ∝ Ω² → Ω ∝ √T → rpm = hover_rpm * √(ratio)
    rpm = HOVER_RPM * np.sqrt(ratio)
    rpm = np.clip(rpm, MIN_RPM, MAX_RPM)
    return rpm


def update_rotor_visual(data, top_idx, bot_idx, thrust_cmd, dt):
    """로터 블레이드 회전 업데이트 (시각 전용, RPM 동적 반영)

    thrust_cmd가 높을수록 (상승/틸트 보상) RPM 증가 → 빠르게 회전
    thrust_cmd가 낮을수록 (하강 중) RPM 감소 → 느리게 회전
    """
    rpm = thrust_to_rpm(thrust_cmd)
    rad_s = rpm * 2.0 * np.pi / 60.0

    data.qpos[top_idx] += rad_s * dt           # 상부: CW
    data.qpos[bot_idx] -= rad_s * 1.05 * dt    # 하부: CCW (5% 빠름, 토크 균형)


def update_tracking_camera(viewer, data, body_id):
    """드론 무게중심을 lookat으로 설정하는 추적 카메라"""
    drone_pos = data.xpos[body_id].copy()
    lookat = drone_pos + CAM_LOOKAT_OFFSET
    viewer.cam.lookat[:] = lookat
    viewer.cam.distance  = CAM_DISTANCE
    viewer.cam.azimuth   = CAM_AZIMUTH
    viewer.cam.elevation = CAM_ELEVATION


# ═══════════════════════════════════════════════════════════════════════════════
# 키보드 입력 (pynput)
# ═══════════════════════════════════════════════════════════════════════════════

class KeyStateTracker:
    """pynput 기반 키 상태 추적기 (백그라운드 스레드)"""

    # pynput Key → 문자열 매핑
    SPECIAL_KEYS = {
        keyboard.Key.up:    'up',
        keyboard.Key.down:  'down',
        keyboard.Key.left:  'left',
        keyboard.Key.right: 'right',
        keyboard.Key.space: 'space',
        keyboard.Key.shift: 'shift',
        keyboard.Key.shift_l: 'shift',
        keyboard.Key.shift_r: 'shift',
        keyboard.Key.esc:   'esc',
    }

    def __init__(self):
        self._pressed = set()
        self._lock = threading.Lock()
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()

    def _key_name(self, key):
        """pynput key 객체 → 문자열"""
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

    def is_pressed(self, key_name):
        with self._lock:
            return key_name in self._pressed

    def get_pressed(self):
        with self._lock:
            return self._pressed.copy()

    def stop(self):
        self._listener.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# 레퍼런스 생성기
# ═══════════════════════════════════════════════════════════════════════════════

class ReferenceGenerator:
    """키보드 입력 → 스무딩된 roll/pitch/z 레퍼런스"""

    def __init__(self, z_init=Z_REF_INIT):
        self.roll_ref  = 0.0
        self.pitch_ref = 0.0
        self.z_ref     = z_init

    def update(self, keys, dt):
        """키 상태에 따라 레퍼런스 업데이트"""
        # ─── 방향 목표 ───
        roll_target  = 0.0
        pitch_target = 0.0

        # W/↑: 전진 (+X) → pitch 양수
        if 'w' in keys or 'up' in keys:
            pitch_target += MAX_TILT
        # S/↓: 후진 (-X) → pitch 음수
        if 's' in keys or 'down' in keys:
            pitch_target -= MAX_TILT
        # A/←: 좌이동 (-Y) → roll 양수
        if 'a' in keys or 'left' in keys:
            roll_target += MAX_TILT
        # D/→: 우이동 (+Y) → roll 음수
        if 'd' in keys or 'right' in keys:
            roll_target -= MAX_TILT

        # ─── 스무딩 (ramp toward target, decay toward zero) ───
        self.roll_ref  = self._smooth(self.roll_ref, roll_target, dt)
        self.pitch_ref = self._smooth(self.pitch_ref, pitch_target, dt)

        # ─── 고도 ───
        if 'space' in keys:
            self.z_ref += ALT_RATE
        if 'shift' in keys:
            self.z_ref -= ALT_RATE
        self.z_ref = np.clip(self.z_ref, Z_MIN, Z_MAX)

        # ─── 리셋 ───
        if 'e' in keys:
            self.roll_ref  = 0.0
            self.pitch_ref = 0.0

        return self.roll_ref, self.pitch_ref, self.z_ref

    def _smooth(self, current, target, dt):
        """현재값을 목표로 ramp/decay"""
        if abs(target) > 1e-6:
            # 키 입력 중: ramp toward target
            rate = RAMP_RATE * dt
        else:
            # 키 미입력: decay toward zero
            rate = DECAY_RATE * dt

        diff = target - current
        if abs(diff) < rate:
            return target
        return current + np.sign(diff) * rate


# ═══════════════════════════════════════════════════════════════════════════════
# 메인 시뮬레이션
# ═══════════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 65)
    print("  NASA Ingenuity 화성 헬리콥터 - 키보드 비행 제어")
    print("=" * 65)
    print()
    print("  조작법:")
    print("    W / ↑     전진 (+X)")
    print("    S / ↓     후진 (-X)")
    print("    A / ←     좌이동 (-Y)")
    print("    D / →     우이동 (+Y)")
    print("    Space     상승")
    print("    Shift     하강")
    print("    E         수평 리셋")
    print("    ESC       종료")
    print()

    # ─── 모델 로드 ──────────────────────────────────────────────────────────
    print(f"[INFO] Loading model: {SCENE_XML}")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data  = mujoco.MjData(model)

    dt      = model.opt.timestep
    ff_each = HOVER_THRUST / 2 / THRUST_GEAR   # 호버 feedforward
    top_idx, bot_idx = get_rotor_joint_indices(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ingenuity")

    print(f"[INFO] 총 질량: {TOTAL_MASS:.2f} kg  |  호버 추력: {HOVER_THRUST:.2f} N")
    print(f"[INFO] Feedforward: {ff_each:.4f} ctrl/rotor  |  dt: {dt:.4f}s")
    print(f"[INFO] 자세 PD: KP={KP_ATT}, KD={KD_ATT}  |  최대 틸트: {np.degrees(MAX_TILT):.1f}°")
    print(f"[INFO] RPM 범위: {MIN_RPM}~{MAX_RPM}  |  호버 RPM: {HOVER_RPM}")
    print()

    # ─── 키보드 & 레퍼런스 초기화 ───────────────────────────────────────────
    keys = KeyStateTracker()
    ref  = ReferenceGenerator(z_init=Z_REF_INIT)

    # ─── PID 상태 ───────────────────────────────────────────────────────────
    z_integral = 0.0
    z_err_prev = 0.0
    z_ref      = Z_REF_INIT

    next_print = 0.0
    step       = 0

    print("-" * 85)
    print(f"{'Time':>6}  {'Alt':>6}  {'zRef':>5}  "
          f"{'Roll°':>6}  {'Pitch°':>7}  "
          f"{'rRef°':>6}  {'pRef°':>7}  "
          f"{'Thr':>5}  {'RPM':>5}  {'Vx':>6}  {'Vy':>6}")
    print("-" * 85)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        # ─── 초기 안정화 (0.5초, 수평 호버) ─────────────────────────────────
        for _ in range(int(0.5 / dt)):
            # 기본 호버 제어만 (키보드 무시)
            laser, gyro, _, quat = get_sensor_data(model, data)
            w, qx, qy, qz = quat
            roll, pitch, _ = quat_to_euler(w, qx, qy, qz)
            p, q, _ = gyro

            z_meas = laser if laser > 0 else data.qpos[2]
            z_err  = Z_REF_INIT - z_meas
            z_integral = np.clip(z_integral + z_err * dt, -I_LIMIT, I_LIMIT)
            z_deriv = (z_err - z_err_prev) / dt

            thrust_cmd = np.clip(
                ff_each + KP_ALT * z_err + KI_ALT * z_integral + KD_ALT * z_deriv,
                0.0, 1.0)

            x_cmd = np.clip(-(KP_ATT * roll + KD_ATT * p), -1.0, 1.0)
            y_cmd = np.clip(-(KP_ATT * pitch + KD_ATT * q), -1.0, 1.0)

            data.ctrl[0] = thrust_cmd
            data.ctrl[1] = thrust_cmd
            data.ctrl[2] = x_cmd
            data.ctrl[3] = y_cmd

            update_rotor_visual(data, top_idx, bot_idx, thrust_cmd, dt)
            z_err_prev = z_err

            mujoco.mj_step(model, data)
            update_tracking_camera(viewer, data, body_id)
            viewer.sync()
            time.sleep(dt / SIM_SPEED)

        print("[INFO] 안정화 완료. 키보드 제어 활성화.")
        print()

        # ─── 메인 루프 ──────────────────────────────────────────────────────
        while viewer.is_running():
            step_start = time.time()

            # 1) 키보드 상태 읽기
            pressed = keys.get_pressed()

            # ESC 종료
            if 'esc' in pressed:
                print("\n[INFO] ESC — 시뮬레이션 종료")
                break

            # 2) 레퍼런스 업데이트
            roll_ref, pitch_ref, z_ref = ref.update(pressed, dt)

            # 3) 센서 읽기
            laser, gyro, _, quat = get_sensor_data(model, data)
            w, qx, qy, qz = quat
            roll, pitch, yaw = quat_to_euler(w, qx, qy, qz)
            p, q, r = gyro

            z_meas = laser if laser > 0 else data.qpos[2]

            # 4) 속도 읽기 (월드 프레임)
            vx = data.qvel[0]
            vy = data.qvel[1]

            # 5) 속도 감쇠 (방향키 미입력 시)
            directional = {'w', 's', 'a', 'd', 'up', 'down', 'left', 'right'}
            if not pressed.intersection(directional):
                # 현재 속도에 비례한 반대 방향 미세 틸트
                damp_pitch = np.clip(-VEL_DAMP_GAIN * vx, -MAX_DAMP_TILT, MAX_DAMP_TILT)
                damp_roll  = np.clip( VEL_DAMP_GAIN * vy, -MAX_DAMP_TILT, MAX_DAMP_TILT)
                pitch_ref += damp_pitch
                roll_ref  += damp_roll

            # 6) 속도 제한
            speed = np.sqrt(vx**2 + vy**2)
            if speed > MAX_SPEED:
                # 속도 초과 시 레퍼런스 틸트를 제한 (브레이크)
                brake_pitch = np.clip(-0.05 * vx, -MAX_TILT, MAX_TILT)
                brake_roll  = np.clip( 0.05 * vy, -MAX_TILT, MAX_TILT)
                pitch_ref = 0.5 * pitch_ref + 0.5 * brake_pitch
                roll_ref  = 0.5 * roll_ref  + 0.5 * brake_roll

            # 7) 고도 PID + 틸트 보상
            z_err      = z_ref - z_meas
            z_integral = np.clip(z_integral + z_err * dt, -I_LIMIT, I_LIMIT)
            z_deriv    = (z_err - z_err_prev) / dt
            z_err_prev = z_err

            thrust_cmd = ff_each + KP_ALT * z_err + KI_ALT * z_integral + KD_ALT * z_deriv

            # 틸트 보상: 기울면 수직 추력 감소 → feedforward 보상
            tilt_mag = np.sqrt(roll**2 + pitch**2)
            cos_comp = 1.0 / max(np.cos(tilt_mag), 0.85)
            thrust_cmd *= cos_comp

            thrust_cmd = np.clip(thrust_cmd, 0.0, 1.0)

            # 8) 자세 PD (레퍼런스 추적)
            roll_err  = roll_ref - roll
            pitch_err = pitch_ref - pitch

            x_cmd = np.clip( KP_ATT * roll_err  - KD_ATT * p, -1.0, 1.0)
            y_cmd = np.clip( KP_ATT * pitch_err - KD_ATT * q, -1.0, 1.0)

            # 9) 액추에이터 적용
            data.ctrl[0] = thrust_cmd
            data.ctrl[1] = thrust_cmd
            data.ctrl[2] = x_cmd
            data.ctrl[3] = y_cmd

            # 10) 로터 시각화
            update_rotor_visual(data, top_idx, bot_idx, thrust_cmd, dt)

            # 11) 물리 스텝
            mujoco.mj_step(model, data)
            step += 1

            # 12) 카메라 & 뷰어
            update_tracking_camera(viewer, data, body_id)
            viewer.sync()

            # 13) 콘솔 출력
            current_rpm = thrust_to_rpm(thrust_cmd)
            if data.time >= next_print:
                print(f"{data.time:6.1f}  {z_meas:6.3f}  {z_ref:5.2f}  "
                      f"{np.degrees(roll):6.2f}  {np.degrees(pitch):7.2f}  "
                      f"{np.degrees(roll_ref):6.2f}  {np.degrees(pitch_ref):7.2f}  "
                      f"{thrust_cmd:5.3f}  {current_rpm:5.0f}  {vx:6.3f}  {vy:6.3f}")
                next_print += PRINT_INTERVAL

            # 14) 실시간 속도 제어
            elapsed = time.time() - step_start
            sleep_time = dt / SIM_SPEED - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ─── 정리 ───────────────────────────────────────────────────────────────
    keys.stop()
    print("-" * 75)
    print(f"[DONE] {step} steps  |  최종 고도: {data.qpos[2]:.4f} m")


if __name__ == "__main__":
    run()
