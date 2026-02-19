"""
Ingenuity Mars Helicopter - Blade Optimization Testbed
Central configuration: paths, physics constants, actuator mapping, design variable bounds.
"""

from pathlib import Path
import numpy as np

# ============================================================================
# Paths
# ============================================================================
DESIGN_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = DESIGN_DIR.parent.parent          # ingenuity-mujoco/
ASSETS_DIR   = PROJECT_ROOT / "assets"
MODELS_DIR   = DESIGN_DIR / "models"
RESULTS_DIR  = DESIGN_DIR / "results"
CODE_0219    = DESIGN_DIR.parent                  # 0219/

SCENE_MARS_XML = str(MODELS_DIR / "scene_mars.xml")
MHS_MARS_XML   = str(MODELS_DIR / "mhs_mars.xml")

# ============================================================================
# Mars Environment
# ============================================================================
MARS_GRAVITY       = 3.71          # m/s^2
RHO_NOMINAL        = 0.015         # kg/m^3  (Mars nominal)
RHO_RANGE          = (0.012, 0.021)  # Mars atmospheric density range
MARS_VISCOSITY     = 1.1e-5        # Pa·s  (CO2)
MARS_SPEED_OF_SOUND = 240.0        # m/s   (CO2 @ ~210 K)

# ============================================================================
# Vehicle Parameters
# ============================================================================
TOTAL_MASS    = 1.6       # kg
MARS_WEIGHT   = TOTAL_MASS * MARS_GRAVITY   # 5.936 N
BLADE_RADIUS  = 0.606     # m  (from STL bounding box analysis)
BLADE_AREA    = np.pi * BLADE_RADIUS ** 2   # Rotor disk area (m^2)
NUM_BLADES    = 2          # blades per rotor
NUM_ROTORS    = 2          # coaxial (top + bottom)
ROTOR_SPACING = 0.0830 + 0.0150   # Distance between thrust sites (m)

# ============================================================================
# Actuator Mapping (mhs_mars.xml)
# ============================================================================
IDX_THRUST1   = 0    # ctrl[0]: top rotor Z-force
IDX_THRUST2   = 1    # ctrl[1]: bottom rotor Z-force
IDX_ROLL      = 2    # ctrl[2]: x_movement (roll torque)
IDX_PITCH     = 3    # ctrl[3]: y_movement (pitch torque)
IDX_YAW       = 4    # ctrl[4]: z_rotation (yaw torque)
NUM_ACTUATORS = 5

# Gear ratios (force/torque per unit ctrl)
THRUST_GEAR   = 6.0     # N per ctrl  (patched from 50)
ATTITUDE_GEAR = 0.5     # Nm per ctrl (patched from 0.09)
YAW_GEAR      = 0.3     # Nm per ctrl (new)

# Hover feedforward
HOVER_THRUST_PER_ROTOR = MARS_WEIGHT / NUM_ROTORS   # 2.968 N
HOVER_CTRL = HOVER_THRUST_PER_ROTOR / THRUST_GEAR   # ~0.495

# ============================================================================
# Simulation Defaults
# ============================================================================
DT           = 0.008     # s  (timestep, from MJCF)
SIM_FREQ     = 1.0 / DT  # 125 Hz
Z_REF        = 1.0       # Default hover altitude (m)
SIM_DURATION = 10.0      # Default simulation duration (s)
INIT_POS_Z   = 0.5       # Initial spawn height (m)

# ============================================================================
# Baseline Blade Data (from STL analysis, report Section 3)
# ============================================================================
# Radial stations for STL-measured chord/thickness
BASELINE_R_OVER_R = np.array([0.20, 0.40, 0.60, 0.80, 0.95])

# Top blade chord (m) at each station
BASELINE_CHORD_TOP = np.array([0.0801, 0.1135, 0.0888, 0.0694, 0.0447])

# Top blade thickness (m) at each station
BASELINE_THICK_TOP = np.array([0.0322, 0.0289, 0.0158, 0.0082, 0.0043])

# Bottom blade (very similar)
BASELINE_CHORD_BOT = np.array([0.0784, 0.1125, 0.0880, 0.0681, 0.0447])
BASELINE_THICK_BOT = np.array([0.0329, 0.0314, 0.0176, 0.0089, 0.0050])

# Average for baseline design vector
BASELINE_CHORD = (BASELINE_CHORD_TOP + BASELINE_CHORD_BOT) / 2.0

# Baseline t/c at root and tip
BASELINE_TC_ROOT = 0.0326 / 0.0793   # ~0.41 (thick root)
BASELINE_TC_TIP  = 0.0047 / 0.0447   # ~0.10

# ============================================================================
# Design Variable Bounds (12 variables)
# ============================================================================
#
# Index  Variable        Lower    Upper    Units
# 0      c0 (r/R=0.15)  0.030    0.150    m
# 1      c1 (r/R=0.30)  0.030    0.150    m
# 2      c2 (r/R=0.50)  0.030    0.150    m
# 3      c3 (r/R=0.70)  0.020    0.120    m
# 4      c4 (r/R=0.95)  0.015    0.080    m
# 5      theta_root     5.0      40.0     deg
# 6      theta_mid      2.0      25.0     deg
# 7      theta_tip     -5.0      15.0     deg
# 8      tc_root        0.08     0.25     -
# 9      tc_tip         0.04     0.12     -
# 10     camber         0.00     0.06     -
# 11     RPM            2000     3200     rev/min

N_DESIGN_VARS = 12

DESIGN_VAR_NAMES = [
    "c0", "c1", "c2", "c3", "c4",
    "theta_root", "theta_mid", "theta_tip",
    "tc_root", "tc_tip", "camber", "RPM",
]

DESIGN_LOWER = np.array([
    0.030, 0.030, 0.030, 0.020, 0.015,    # chord (m)
    5.0,   2.0,  -5.0,                     # twist (deg)
    0.08,  0.04,  0.00,                    # tc, camber
    2000.0,                                 # RPM
])

DESIGN_UPPER = np.array([
    0.150, 0.150, 0.150, 0.120, 0.080,    # chord (m)
    40.0,  25.0,  15.0,                    # twist (deg)
    0.25,  0.12,  0.06,                    # tc, camber
    3200.0,                                 # RPM
])

# Chord control point radial locations (r/R)
CHORD_STATIONS = np.array([0.15, 0.30, 0.50, 0.70, 0.95])

# Twist control point radial locations (r/R)
TWIST_STATIONS = np.array([0.20, 0.55, 0.95])

# ============================================================================
# Optimization Constraints
# ============================================================================
RPM_MAX          = 3200.0
TIP_MACH_MAX     = 0.80
CTRL_SAT_MAX     = 0.05     # Max ctrl saturation fraction (5%)
F1_BENDING_MIN   = 40.0     # Minimum first bending frequency (Hz)
RESONANCE_MARGIN = 5.0      # Hz, min distance from harmonics

# ============================================================================
# Controller Defaults (Mars-tuned)
# ============================================================================
# Altitude PID
ALT_KP = 2.0
ALT_KI = 0.3
ALT_KD = 1.0
ALT_I_LIMIT = 0.3

# Attitude PD
ATT_KP = 5.0
ATT_KD = 1.5

# Structural material (carbon fiber composite)
E_MATERIAL   = 70.0e9      # Young's modulus (Pa)
RHO_MATERIAL = 1600.0      # Material density (kg/m^3)
SPAR_FRACTION = 0.30       # Spar width as fraction of chord
