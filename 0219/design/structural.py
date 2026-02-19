"""
Ingenuity Mars Helicopter - Structural and Vibration Constraints (E4)

Simplified analytical model for blade bending frequencies using
Euler-Bernoulli beam theory with centrifugal stiffening.
Used as optimization constraints, not a full FEA.

Assumptions:
  - Blade modeled as tapered cantilever beam (root fixed)
  - Rectangular cross-section: width = spar_fraction * chord, height = tc * chord
  - Carbon fiber composite: E = 70 GPa, rho = 1600 kg/m^3
  - Southwell approximation for rotating frequency
"""

import numpy as np

from config import (
    BLADE_RADIUS, NUM_BLADES,
    E_MATERIAL, RHO_MATERIAL, SPAR_FRACTION,
    F1_BENDING_MIN, RESONANCE_MARGIN,
)
from blade_param import BladeDesign


def blade_section_properties(chord: float, tc: float) -> tuple:
    """
    Compute beam cross-section properties for a blade station.

    Parameters
    ----------
    chord : float  Chord length (m)
    tc    : float  Thickness-to-chord ratio

    Returns
    -------
    A  : float  Cross-sectional area (m^2)
    I  : float  Second moment of area, flapwise (m^4)
    """
    # Spar width and height
    b = SPAR_FRACTION * chord        # spar width
    h = tc * chord                   # section height

    A = b * h
    I = b * h ** 3 / 12.0           # Rectangular section, flapwise

    return A, I


def estimate_non_rotating_freq(blade: BladeDesign, n_stations: int = 20) -> float:
    """
    Estimate first flapwise bending frequency (non-rotating) using
    Rayleigh-Ritz method with assumed mode shape phi(x) = (x/L)^2.

    Parameters
    ----------
    blade      : BladeDesign
    n_stations : int  Number of integration points

    Returns
    -------
    f1_nr : float  First non-rotating bending frequency (Hz)
    """
    R = BLADE_RADIUS
    r_min = 0.10 * R
    r_max = 0.98 * R
    L = r_max - r_min

    r_stations = np.linspace(r_min, r_max, n_stations)
    dr = r_stations[1] - r_stations[0]
    r_over_R = r_stations / R

    chords = blade.chord_at(r_over_R)
    tcs = blade.tc_at(r_over_R)

    # Assumed mode: phi(x) = (x/L)^2, phi''(x) = 2/L^2
    # x = r - r_min  (distance from root)
    x_arr = r_stations - r_min

    # Rayleigh quotient: omega^2 = integral(EI * phi''^2 dx) / integral(m * phi^2 dx)
    numerator = 0.0   # Stiffness term
    denominator = 0.0  # Mass term

    for i in range(n_stations):
        A, I = blade_section_properties(chords[i], tcs[i])

        # phi''(x) = 2/L^2
        phi_pp = 2.0 / (L ** 2)

        # phi(x) = (x/L)^2
        phi = (x_arr[i] / L) ** 2

        # Mass per unit length
        m_per_length = RHO_MATERIAL * A

        numerator += E_MATERIAL * I * phi_pp ** 2 * dr
        denominator += m_per_length * phi ** 2 * dr

    if denominator > 1e-15:
        omega_sq = numerator / denominator
        omega = np.sqrt(omega_sq)
        f1_nr = omega / (2.0 * np.pi)
    else:
        f1_nr = 0.0

    return f1_nr


def estimate_first_bending_freq(blade: BladeDesign) -> float:
    """
    First flapwise bending frequency (rotating) using Southwell approximation:
      f1 = sqrt(f1_nr^2 + k * (Omega/(2*pi))^2)

    where k ~ 1.1 for typical blade (centrifugal stiffening coefficient)

    Parameters
    ----------
    blade : BladeDesign

    Returns
    -------
    f1 : float  First rotating bending frequency (Hz)
    """
    f1_nr = estimate_non_rotating_freq(blade)
    Omega = blade.omega()
    f_rev = Omega / (2.0 * np.pi)   # Hz

    k = 1.1  # Centrifugal stiffening coefficient

    f1 = np.sqrt(f1_nr ** 2 + k * f_rev ** 2)
    return f1


def check_resonance_margins(blade: BladeDesign,
                            harmonics: list = None) -> dict:
    """
    Check if blade natural frequency avoids multiples of the rotation frequency.

    |f1 - n * f_rev| >= RESONANCE_MARGIN  for each harmonic n

    Parameters
    ----------
    blade     : BladeDesign
    harmonics : list of int  Harmonics to check (default [1,2,3,4])

    Returns
    -------
    dict with keys:
        'f1'       : First bending frequency (Hz)
        'f_rev'    : Rotational frequency (Hz)
        'margins'  : dict {n: margin_Hz} for each harmonic
        'feasible' : bool (all margins met)
    """
    if harmonics is None:
        harmonics = [1, 2, 3, 4]

    f1 = estimate_first_bending_freq(blade)
    f_rev = blade.omega() / (2.0 * np.pi)

    margins = {}
    feasible = True
    for n in harmonics:
        f_harmonic = n * f_rev
        margin = abs(f1 - f_harmonic)
        margins[n] = margin
        if margin < RESONANCE_MARGIN:
            feasible = False

    return {
        'f1': f1,
        'f_rev': f_rev,
        'margins': margins,
        'feasible': feasible,
    }


def blade_mass_estimate(blade: BladeDesign, n_stations: int = 20) -> float:
    """
    Estimate total blade mass (single blade) from geometry and material properties.

    Parameters
    ----------
    blade : BladeDesign

    Returns
    -------
    mass : float  Blade mass (kg)
    """
    R = BLADE_RADIUS
    r_min = 0.10 * R
    r_max = 0.98 * R

    r_stations = np.linspace(r_min, r_max, n_stations)
    dr = r_stations[1] - r_stations[0]
    r_over_R = r_stations / R

    chords = blade.chord_at(r_over_R)
    tcs = blade.tc_at(r_over_R)

    mass = 0.0
    for i in range(n_stations):
        A, _ = blade_section_properties(chords[i], tcs[i])
        mass += RHO_MATERIAL * A * dr

    return mass


def evaluate_structural(blade: BladeDesign) -> dict:
    """
    Full structural evaluation for optimization constraints.

    Returns
    -------
    dict with keys:
        'f1_bending'        : First bending frequency (Hz)
        'f_rev'             : Rotation frequency (Hz)
        'resonance_margins' : dict of margin per harmonic
        'blade_mass'        : Single blade mass (kg)
        'total_rotor_mass'  : All blades mass (kg)
        'f1_feasible'       : f1 >= F1_BENDING_MIN
        'resonance_feasible': All margins met
        'feasible'          : Both constraints met
    """
    res = check_resonance_margins(blade)
    mass_single = blade_mass_estimate(blade)
    total_rotor_mass = mass_single * NUM_BLADES * 2  # 2 rotors, each N_b blades

    f1_ok = res['f1'] >= F1_BENDING_MIN

    return {
        'f1_bending': res['f1'],
        'f_rev': res['f_rev'],
        'resonance_margins': res['margins'],
        'blade_mass': mass_single,
        'total_rotor_mass': total_rotor_mass,
        'f1_feasible': f1_ok,
        'resonance_feasible': res['feasible'],
        'feasible': f1_ok and res['feasible'],
    }


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == "__main__":
    from blade_param import baseline_design

    bl = baseline_design()
    print(f"Blade: {bl}")
    print()

    result = evaluate_structural(bl)
    print(f"First bending freq:  {result['f1_bending']:.1f} Hz")
    print(f"Rotation freq:       {result['f_rev']:.1f} Hz")
    print(f"Single blade mass:   {result['blade_mass']*1000:.1f} g")
    print(f"Total rotor mass:    {result['total_rotor_mass']*1000:.1f} g")
    print(f"f1 >= {F1_BENDING_MIN} Hz:     {result['f1_feasible']}")
    print(f"Resonance margins:")
    for n, margin in result['resonance_margins'].items():
        status = "OK" if margin >= RESONANCE_MARGIN else "FAIL"
        print(f"  {n}P: {margin:.1f} Hz  ({status})")
    print(f"Overall feasible:    {result['feasible']}")
