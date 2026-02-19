"""
Ingenuity Mars Helicopter - Blade Parameterization

Defines a blade design as a 12-dimensional vector and provides interpolation
to continuous chord, twist, and thickness-to-chord distributions.

Design vector (12 variables):
  [c0, c1, c2, c3, c4,              chord at r/R = 0.15, 0.30, 0.50, 0.70, 0.95
   theta_root, theta_mid, theta_tip, twist at r/R = 0.20, 0.55, 0.95
   tc_root, tc_tip,                  thickness-to-chord ratio
   camber,                           max camber fraction
   RPM]                              operating RPM
"""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d

from config import (
    BLADE_RADIUS, MARS_SPEED_OF_SOUND, NUM_BLADES,
    CHORD_STATIONS, TWIST_STATIONS,
    DESIGN_LOWER, DESIGN_UPPER, N_DESIGN_VARS,
    BASELINE_CHORD, BASELINE_TC_ROOT, BASELINE_TC_TIP,
)


class BladeDesign:
    """Parameterized blade geometry from a 12D design vector."""

    def __init__(self, design_vector: np.ndarray):
        assert len(design_vector) == N_DESIGN_VARS, \
            f"Expected {N_DESIGN_VARS} variables, got {len(design_vector)}"

        self.vector = np.array(design_vector, dtype=float)

        # Parse variables
        self.chord_ctrl = self.vector[0:5]       # m
        self.theta_root = self.vector[5]          # deg
        self.theta_mid  = self.vector[6]          # deg
        self.theta_tip  = self.vector[7]          # deg
        self.tc_root    = self.vector[8]          # -
        self.tc_tip     = self.vector[9]          # -
        self.camber     = self.vector[10]         # -
        self.rpm        = self.vector[11]         # rev/min

        # Build interpolators
        self._chord_interp = CubicSpline(
            CHORD_STATIONS, self.chord_ctrl,
            bc_type='natural',
        )
        twist_pts = np.array([self.theta_root, self.theta_mid, self.theta_tip])
        self._twist_interp = interp1d(
            TWIST_STATIONS, twist_pts,
            kind='quadratic', fill_value='extrapolate',
        )

    def chord_at(self, r_over_R: np.ndarray) -> np.ndarray:
        """Chord distribution c(r/R) in meters via cubic spline."""
        r = np.atleast_1d(r_over_R)
        c = self._chord_interp(r)
        return np.maximum(c, 0.005)  # Floor to 5mm

    def twist_at(self, r_over_R: np.ndarray) -> np.ndarray:
        """Twist distribution theta(r/R) in degrees via quadratic interpolation."""
        r = np.atleast_1d(r_over_R)
        return self._twist_interp(r)

    def twist_rad_at(self, r_over_R: np.ndarray) -> np.ndarray:
        """Twist in radians."""
        return np.radians(self.twist_at(r_over_R))

    def tc_at(self, r_over_R: np.ndarray) -> np.ndarray:
        """Thickness-to-chord ratio, linear from root to tip."""
        r = np.atleast_1d(r_over_R)
        return self.tc_root + (self.tc_tip - self.tc_root) * r

    def omega(self) -> float:
        """RPM to angular velocity (rad/s)."""
        return self.rpm * 2.0 * np.pi / 60.0

    def tip_speed(self) -> float:
        """Blade tip speed (m/s)."""
        return self.omega() * BLADE_RADIUS

    def tip_mach(self, a_sound: float = MARS_SPEED_OF_SOUND) -> float:
        """Tip Mach number."""
        return self.tip_speed() / a_sound

    def solidity(self) -> float:
        """Average rotor solidity sigma = N_b * c_avg / (pi * R)."""
        r_stations = np.linspace(0.15, 0.95, 30)
        c_avg = np.mean(self.chord_at(r_stations))
        return NUM_BLADES * c_avg / (np.pi * BLADE_RADIUS)

    def __repr__(self):
        return (
            f"BladeDesign(RPM={self.rpm:.0f}, "
            f"chord=[{self.chord_ctrl[0]:.3f}..{self.chord_ctrl[-1]:.3f}]m, "
            f"twist=[{self.theta_root:.1f}..{self.theta_tip:.1f}]deg, "
            f"tc=[{self.tc_root:.3f}..{self.tc_tip:.3f}], "
            f"camber={self.camber:.3f}, "
            f"tip_Mach={self.tip_mach():.3f})"
        )


def baseline_design() -> BladeDesign:
    """
    Return the baseline design derived from STL analysis (report Section 3).
    Uses average of top/bottom blade measurements.
    """
    vector = np.array([
        # chord at 5 stations (average of top/bottom)
        BASELINE_CHORD[0],  # r/R=0.20 -> mapped to 0.15 (extrapolated slightly)
        BASELINE_CHORD[1],  # r/R=0.40 -> mapped to 0.30
        BASELINE_CHORD[2],  # r/R=0.60 -> mapped to 0.50
        BASELINE_CHORD[3],  # r/R=0.80 -> mapped to 0.70
        BASELINE_CHORD[4],  # r/R=0.95 -> mapped to 0.95
        # twist (estimated from Ingenuity literature: ~25 deg root, ~10 deg tip)
        25.0,   # theta_root (deg)
        15.0,   # theta_mid (deg)
        8.0,    # theta_tip (deg)
        # t/c
        min(BASELINE_TC_ROOT, 0.25),  # tc_root (capped at bounds)
        BASELINE_TC_TIP,              # tc_tip
        # camber (Ingenuity uses CLF 5605 airfoil, moderate camber)
        0.03,
        # RPM (nominal hover)
        2537.0,
    ])
    return BladeDesign(vector)


def get_bounds() -> tuple:
    """Return (lower_bounds, upper_bounds) as numpy arrays."""
    return DESIGN_LOWER.copy(), DESIGN_UPPER.copy()


def random_design(rng: np.random.Generator = None) -> BladeDesign:
    """Generate a random blade design within bounds."""
    if rng is None:
        rng = np.random.default_rng()
    vec = rng.uniform(DESIGN_LOWER, DESIGN_UPPER)
    return BladeDesign(vec)


if __name__ == "__main__":
    # Quick test
    bl = baseline_design()
    print(bl)
    print(f"  Solidity:   {bl.solidity():.4f}")
    print(f"  Tip speed:  {bl.tip_speed():.1f} m/s")
    print(f"  Tip Mach:   {bl.tip_mach():.3f}")
    print()

    r_stations = np.linspace(0.10, 0.98, 20)
    print(f"  {'r/R':>5}  {'chord(m)':>9}  {'twist(deg)':>10}  {'t/c':>6}")
    for rr in r_stations:
        c = bl.chord_at(rr)[0] if np.ndim(bl.chord_at(rr)) > 0 else bl.chord_at(rr)
        t = bl.twist_at(rr)
        tc = bl.tc_at(rr)
        print(f"  {rr:5.2f}  {float(c):9.4f}  {float(t):10.2f}  {float(tc):6.3f}")
