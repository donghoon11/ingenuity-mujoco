# Ingenuity Blade Optimization Testbed — Report v3

**Date**: 2026-02-20
**Project**: `E:/mujoco_projects/ingenuity-mujoco/`
**Working directory**: `0219/design/`

---

## 1. 프로젝트 개요

NASA Ingenuity 화성 헬리콥터의 블레이드 설계를 MuJoCo 기반 시뮬레이션 환경에서 최적화하는 테스트베드.

- **물리 환경**: 화성 중력 (3.71 m/s²), 대기 밀도 (0.015 kg/m³)
- **차량**: 1.6 kg, 공축 이중 반전 로터 (블레이드 반경 0.606 m, 2-blade × 2)
- **최적화 목표**: 호버 소비전력 최소화, 추력 효율(FM) 최대화, 대기 밀도 변동 대응 마진 최대화
- **설계 변수**: 12차원 (chord 5점, twist 3점, t/c 비 2점, camber 1개, RPM 1개)

---

## 2. 이번 세션에서 수행한 작업

### 2-1. Bottom Blade STL 메쉬 찢김 수정 (완료)

#### 문제
- Bottom blade의 허브-블레이드 전환부에서 메쉬 찢김 발생
- 블레이드 날과 허브 연결부가 거의 선(single line)으로만 연결된 불안정 구조

#### 근본 원인 분석
`generate_blade_stl.py`의 `flip_camber` 구현이 X좌표 반전 방식을 사용하여 전환 구간에서 파괴적 간섭 발생:

- 허브 타원(ellipse) 중심: X ≈ 0.0
- X-반전된 에어포일 중심: X ≈ −0.027
- 블렌딩 중간(t=0.5): 두 좌표계가 상쇄 → chord 0.057m → **0.012m로 붕괴**

#### 수정 내용
**파일**: `0219/design/generate_blade_stl.py`
**위치**: `generate_blade_sections()` 함수, 허브 그래프팅 전환 구간

| 항목 | 수정 전 | 수정 후 |
|------|---------|---------|
| flip_camber 방식 | X좌표 반전 (`x_af_rot = -x_af_rot`) | camber 값 부호 반전 |
| 전환 구간 chord | 0.012 m (붕괴) | 0.042~0.116 m (단조 증가) |
| 메쉬 품질 | 찢김 발생 | 정상 |

```python
# generate_blade_stl.py (lines 205-208)
cam = blade.camber
# For counter-rotating bottom rotor: negate camber to flip lift direction
# while keeping the section centered (no X-coordinate offset).
effective_cam = -cam if flip_camber else cam
```

#### 검증 결과
- 전환 구간 chord: 0.042 → 0.049 → 0.060 → 0.074 → 0.102 → 0.116 m (단조 증가, 정상)
- 호버 테스트: z = 1.010 m (오차 0.010 m) — PASS

---

### 2-2. 달 환경 MuJoCo 장면 생성 (완료)

#### 목적
- Apollo 달 착륙 지점 실제 지형 STL을 활용한 달 표면 환경 구현
- 화성 물리 속성 (중력, 밀도) 그대로 유지
- 키보드 비행 제어 인터페이스 구현

#### 생성 파일

| 파일 | 역할 |
|------|------|
| `0219/design/models/scene_lunar.xml` | 달 환경 MuJoCo scene |
| `0219/design/view_lunar.py` | 키보드 비행 제어 뷰어 |

---

### 2-3. scene_lunar.xml 최종 구성

#### Apollo STL 메쉬 현황 (in `assets/lunar_landing_site/`)

| 메쉬 | 삼각형 수 | 파일 크기 | 사용 가능 |
|------|----------|----------|----------|
| Apollo 11 | 142,768 | 6.8 MB | ✓ |
| Apollo 12 | 528,374 | 25.2 MB | ✗ (200K 초과) |
| Apollo 14 | 139,270 | 6.6 MB | ✓ |
| Apollo 15 | 141,296 | 6.7 MB | ✓ |
| Apollo 16 | 138,810 | 6.6 MB | ✓ (채택) |
| Apollo 17 | 533,506 | 25.4 MB | ✗ (200K 초과) |

> MuJoCo 메쉬 face 제한: 200,000

#### 최종 타일 구성

- **메쉬**: Apollo 16 단일 메쉬 (138,810 faces, 1번 로드)
- **배치**: 2×2 그리드 = 4타일 (14m × 14m 커버)
- **타일 크기**: 7m × 7m (XY scale = 0.027397)
- **Z 과장**: 2.5배 (Z scale = 0.068493) → 기복 ~1.0m로 확대
- **반복감 제거**: 각 타일에 0°/90°/180°/270° Z축 회전 적용
- **로드 시간**: 약 3.4초

```xml
<!-- asset 정의 -->
<mesh name="terrain_a16"
      file="lunar_landing_site/Apollo 16 - Landing Site.stl"
      scale="0.027397 0.027397 0.068493"/>

<!-- 2x2 타일 배치 -->
<geom name="tile_00" ... pos="-3.5 -3.5 -0.523" />               <!-- 0° -->
<geom name="tile_10" ... pos="3.5 -3.5 -0.523" quat="0.707 0 0 0.707"/>   <!-- 90° -->
<geom name="tile_01" ... pos="-3.5 3.5 -0.523" quat="0 0 0 1"/>           <!-- 180° -->
<geom name="tile_11" ... pos="3.5 3.5 -0.523" quat="0.707 0 0 -0.707"/>   <!-- 270° -->
```

#### 조명 설정 (scene.xml 기반)

```xml
<visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-20" elevation="-20"/>
</visual>
<light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
```

- 기존 scene.xml 조명과 동일한 구조
- Skybox: 어두운 우주 공간 (`rgb1="0.03 0.05 0.08"`)
- 달 표면 색상: 밝은 회색 (`rgba="0.80 0.79 0.77 1"`)

---

### 2-4. view_lunar.py — 키보드 비행 제어

#### 키 매핑

| 키 | 동작 |
|----|------|
| W / ↑ | 전진 (+X, pitch tilt) |
| S / ↓ | 후진 (−X) |
| A / ← | 좌측 이동 (roll tilt) |
| D / → | 우측 이동 |
| Space | 상승 |
| Shift | 하강 |
| E | 수평 복귀 (tilt 해제) |
| ESC | 종료 |

#### 제어 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| 고도 PID KP/KI/KD | 2.0 / 0.3 / 1.0 | Mars-tuned |
| 자세 PD KP/KD | 5.0 / 1.5 | |
| 최대 tilt | 12° | 방향 입력 시 |
| 속도 감쇠 | 자동 | 방향 키 해제 시 역tilt |
| 최대 속도 | 5.0 m/s | 초과 시 자동 감속 |
| 호버 추력 | ~0.495 ctrl | MARS_WEIGHT/2/THRUST_GEAR |

#### 추가 기능
- **추적 카메라**: 헬리콥터를 항상 프레임 중앙에 유지
- **로터 RPM 시각화**: 추력에 비례한 회전 속도 표현
- **Tilt 보상**: 기울기에 따른 추력 증가 (1/cos(θ))
- **실시간 콘솔 출력**: 고도, 자세, 추력, RPM, 속도 1초 간격 표시

---

## 3. 전체 파일 구조 및 실행 명령

### 3-1. 디렉토리 구조

```
ingenuity-mujoco/
├── scene.xml                    # 원본 Mars 기본 장면 (조명 참조용)
├── assets/
│   ├── carbon.png               # 블레이드 재질 텍스처
│   ├── mhs_mainbody_v16.stl     # 본체 메쉬
│   ├── mhs_leg01~04_v16.stl     # 다리 메쉬
│   ├── optimized_topblades.stl  # 최적화 상부 블레이드 (생성됨)
│   ├── optimized_bottomblades.stl  # 최적화 하부 블레이드 (생성됨)
│   └── lunar_landing_site/
│       ├── Apollo 11 - Landing Site.stl
│       ├── Apollo 15 - Landing Site.stl
│       └── Apollo 16 - Landing Site.stl  # 메인 지형
└── 0219/
    ├── hover_basic.py           # 기본 호버 시뮬레이션
    ├── hover_keyboard.py        # 키보드 비행 제어 (Mars 장면)
    └── design/
        ├── config.py            # 물리 상수 / 설계 변수 / 경로
        ├── blade_param.py       # 블레이드 형상 파라미터화 (12D)
        ├── bemt.py              # 블레이드 요소 모멘텀 이론 (공력)
        ├── structural.py        # 구조 진동 주파수 / 공진 마진
        ├── controller.py        # PID/PD 제어기
        ├── sim_interface.py     # MuJoCo 시뮬레이션 인터페이스
        ├── utils.py             # 공용 유틸리티
        ├── e0_baseline.py       # 실험 E0: 역학 기준 교정
        ├── e1_hover_map.py      # 실험 E1: 호버 성능 평가
        ├── e2_robust_hover.py   # 실험 E2: 밀도 변동 강건성
        ├── e3_forward_flight.py # 실험 E3: 전진 비행 + 돌풍 대응
        ├── e4_structural.py     # 실험 E4: 구조 평가
        ├── doe.py               # 라틴 하이퍼큐브 샘플링
        ├── surrogate.py         # 가우시안 프로세스 대리 모델
        ├── optimizer.py         # NSGA-II 다목적 최적화
        ├── pipeline.py          # 전체 워크플로우 (DOE→Surrogate→NSGA-II→검증)
        ├── generate_blade_stl.py # 블레이드 STL 기하 생성
        ├── analyze_original_stl.py # 원본 STL 분석
        ├── run_all.py           # 마스터 CLI
        ├── view_optimized.py    # 최적화 블레이드 뷰어 (Mars 장면)
        ├── view_lunar.py        # 달 환경 키보드 비행 뷰어
        └── models/
            ├── mhs_mars.xml     # 헬리콥터 모델 (Mars baseline 블레이드)
            ├── mhs_optimized.xml # 헬리콥터 모델 (최적화 블레이드)
            ├── scene_mars.xml   # Mars 환경 장면
            ├── scene_optimized.xml # Mars 환경 + 최적화 블레이드 장면
            └── scene_lunar.xml  # 달 환경 장면 (신규)
```

### 3-2. 실행 명령 (working dir: `E:/mujoco_projects/ingenuity-mujoco`)

```bash
# ── 달 환경 키보드 비행 뷰어 ──────────────────────────────────────────────
python 0219/design/view_lunar.py

# STL 재생성 후 달 환경 실행
python 0219/design/view_lunar.py --regenerate

# ── 최적화 블레이드 뷰어 (Mars 장면) ──────────────────────────────────────
python 0219/design/view_optimized.py

# 최적화 STL 재생성 후 실행
python 0219/design/view_optimized.py --regenerate

# Baseline 블레이드로 실행
python 0219/design/view_optimized.py --baseline

# ── 기본 비행 시뮬레이션 ──────────────────────────────────────────────────
python 0219/hover_basic.py              # 헤드리스
python 0219/hover_basic.py --viewer     # 시각화 포함

# Mars 키보드 비행 (원본 장면)
python 0219/hover_keyboard.py

# ── 블레이드 STL 생성 ─────────────────────────────────────────────────────
python 0219/design/generate_blade_stl.py

# ── 실험 시리즈 ──────────────────────────────────────────────────────────
python 0219/design/e0_baseline.py              # E0: 동역학 기준 교정
python 0219/design/e0_baseline.py --viewer     # E0 + MuJoCo 뷰어

python 0219/design/e1_hover_map.py             # E1: 호버 성능 평가
python 0219/design/e2_robust_hover.py          # E2: 밀도 변동 강건성
python 0219/design/e3_forward_flight.py        # E3: 전진 비행
python 0219/design/e3_forward_flight.py --gust # E3 + 돌풍 테스트
python 0219/design/e4_structural.py            # E4: 구조 평가

# ── 최적화 파이프라인 ────────────────────────────────────────────────────
python 0219/design/doe.py                         # DOE (기본 150 샘플)
python 0219/design/doe.py --n 200 --seed 42

python 0219/design/surrogate.py                   # GP 대리 모델 학습

python 0219/design/optimizer.py                   # NSGA-II 최적화
python 0219/design/optimizer.py --gen 50 --pop 80

python 0219/design/pipeline.py                    # 전체 파이프라인
python 0219/design/pipeline.py --doe-n 100 --gen 30 --top 5

# ── 마스터 CLI ───────────────────────────────────────────────────────────
python 0219/design/run_all.py --phase 0              # 빠른 검증
python 0219/design/run_all.py --phase 1 --viewer     # E0 + 뷰어
python 0219/design/run_all.py --phase 2              # E1~E4 전체
python 0219/design/run_all.py --phase 3 --doe-n 100  # 최적화 파이프라인
python 0219/design/run_all.py --only e1              # E1만 실행
```

---

## 4. 주요 물리 상수 (config.py)

| 항목 | 값 | 단위 |
|------|-----|------|
| 화성 중력 | 3.71 | m/s² |
| 대기 밀도 (nominal) | 0.015 | kg/m³ |
| 차량 질량 | 1.6 | kg |
| 차량 중량 | 5.936 | N |
| 블레이드 반경 | 0.606 | m |
| 추력 actuator | 6.0 | N/ctrl |
| 호버 추력 제어값 | ~0.495 | ctrl |
| 타임스텝 | 0.008 | s |

---

## 5. MuJoCo 모델 구성 (mhs_optimized.xml)

| 항목 | 값 |
|------|-----|
| 중력 | (0, 0, −3.71) m/s² |
| 대기 밀도 | 0.015 kg/m³ |
| 타임스텝 | 0.008 s |
| Actuator: thrust1/2 | site thrust, gear 6 N/ctrl |
| Actuator: x/y_movement | site thrustxy, gear 0.5 Nm/ctrl |
| Actuator: z_rotation | site thrustxy, gear 0.3 Nm/ctrl |
| Top rotor joint | hinge, Z축, CW |
| Bottom rotor joint | hinge, Z축, CCW (quat="0.6 0 0 1" 회전) |
| Sensor | rangefinder, gyro, accelerometer, framequat |

---

## 6. 이슈 이력 및 해결책

| 이슈 | 원인 | 해결 |
|------|------|------|
| Bottom blade 메쉬 찢김 | flip_camber X좌표 반전으로 블렌딩 파괴적 간섭 | camber 값 부호 반전으로 변경 |
| 지형 충돌로 헬리콥터 밀려남 | terrain geom에 collision 활성화 | `contype="0" conaffinity="0"` 추가 |
| Apollo 17/12 로드 실패 | MuJoCo 200K faces 제한 초과 | Apollo 16 (138K) 만 사용 |
| 3×3 타일 로딩 느림 / 동결 | 9개 타일 × ~140K faces = ~1.26M faces | 2×2 + 단일 메쉬 인스턴싱으로 변경 |
| 울퉁불퉁함 안 보임 | Z scale 동일하여 255m→5m 축소 시 기복 ~0.3m | Z scale 2.5배 과장 (기복 ~1m) |

---

## 7. 의존성

```
python >= 3.8
mujoco >= 3.0
numpy
pynput          # 키보드 제어 (view_lunar.py, hover_keyboard.py)
scipy           # 스플라인 보간 (blade_param.py)
scikit-learn    # GP 대리 모델 (surrogate.py)
pymoo           # NSGA-II (optimizer.py)
```
