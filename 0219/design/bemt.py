"""
Ingenuity Mars Helicopter - Blade Element Momentum Theory (BEMT)

Computes hover thrust, torque, and power for a given blade design
and atmospheric conditions. This is the core aerodynamic engine
that connects blade geometry to MuJoCo actuator commands.

Physical basis:
  - Rotor disk divided into N annular elements
  - Each element: combined blade-element forces + momentum inflow balance
  - Iterative solve for induced velocity at each station
  - Prandtl tip loss correction
  - Low-Reynolds, transonic airfoil model for Mars conditions
"""

import numpy as np

from config import (
    BLADE_RADIUS, BLADE_AREA, NUM_BLADES, NUM_ROTORS,
    MARS_SPEED_OF_SOUND, MARS_VISCOSITY,
    THRUST_GEAR, YAW_GEAR,
)
from blade_param import BladeDesign


# ============================================================================
# Airfoil Polar Model
# ============================================================================

def airfoil_polar(alpha_rad: float, tc: float, camber: float,
                  Re: float, Mach: float) -> tuple:
    """
    Semi-empirical airfoil Cl/Cd model for low-Re Mars conditions.

    Parameters
    ----------
    alpha_rad : float  Angle of attack (radians)
    tc        : float  Thickness-to-chord ratio
    camber    : float  Maximum camber (fraction of chord)
    Re        : float  Reynolds number
    Mach      : float  Local Mach number

    Returns
    -------
    Cl, Cd : float  Lift and drag coefficients
    """
    # Lift slope: thin airfoil = 2*pi, with Prandtl-Glauert compressibility
    Cl_alpha_0 = 2.0 * np.pi
    beta_pg = np.sqrt(max(1.0 - Mach ** 2, 0.01))  # Prandtl-Glauert
    Cl_alpha = Cl_alpha_0 / beta_pg

    # Low-Re correction: ~20% reduction at Re < 50000
    Re_eff = max(Re, 5000.0)
    Re_factor = min(1.0, 0.75 + 0.25 * (Re_eff / 50000.0))
    Cl_alpha *= Re_factor

    # Zero-lift angle from camber (thin airfoil: alpha_0 ~ -2 * camber)
    alpha_0 = -2.0 * camber

    # Linear Cl
    Cl = Cl_alpha * (alpha_rad - alpha_0)

    # Stall model: smooth clamp
    Cl_max = 1.1 + 0.5 * camber  # camber increases Cl_max slightly
    if Cl > Cl_max:
        Cl = Cl_max - 0.3 * (alpha_rad - (Cl_max / Cl_alpha + alpha_0)) ** 2
        Cl = max(Cl, 0.1)
    elif Cl < -Cl_max:
        Cl = -Cl_max + 0.3 * (alpha_rad + (Cl_max / Cl_alpha - alpha_0)) ** 2
        Cl = min(Cl, -0.1)

    # Drag polar: base + induced
    Cd_0 = 0.015 + 0.015 * (50000.0 / max(Re_eff, 10000.0))  # Higher at low Re
    Cd_0 += 0.005 * tc  # Thickness penalty
    Cd_i = Cl ** 2 / (np.pi * 5.0)  # Approximate 2D induced drag
    Cd = Cd_0 + Cd_i

    # Compressibility drag rise near Mach 0.7+
    if Mach > 0.65:
        Cd += 0.1 * (Mach - 0.65) ** 2

    return float(Cl), float(Cd)


# ============================================================================
# Prandtl Tip Loss
# ============================================================================

def prandtl_tip_loss(r: float, R: float, n_blades: int, phi: float) -> float:
    """
    Prandtl tip loss factor F.

    F = (2/pi) * arccos(exp(-f))
    f = (n_blades/2) * (R - r) / (r * sin(phi))
    """
    sin_phi = abs(np.sin(phi))
    if sin_phi < 1e-6 or r < 0.01 * R:
        return 1.0

    f = (n_blades / 2.0) * (R - r) / (r * sin_phi)
    f = min(f, 20.0)  # Prevent overflow
    exp_val = np.exp(-f)

    if exp_val >= 1.0:
        return 1.0

    F = (2.0 / np.pi) * np.arccos(exp_val)
    return max(F, 0.01)  # Floor to prevent division by zero


# ============================================================================
# BEMT Hover Solver
# ============================================================================

def bemt_hover(blade: BladeDesign, rho: float, n_elements: int = 30,
               max_iter: int = 100, tol: float = 1e-6,
               verbose: bool = False) -> dict:
    """
    Run BEMT analysis for hover condition (single rotor).

    Parameters
    ----------
    blade      : BladeDesign  Parameterized blade geometry
    rho        : float        Atmospheric density (kg/m^3)
    n_elements : int          Number of radial stations
    max_iter   : int          Max iterations per station
    tol        : float        Convergence tolerance for inflow
    verbose    : bool         Print per-station details

    Returns
    -------
    dict with keys:
        'T'          : Total thrust, both rotors (N)
        'Q'          : Torque per rotor (Nm)
        'P'          : Total power, both rotors (W)
        'T_per_rotor': Thrust per rotor (N)
        'CT'         : Thrust coefficient
        'CP'         : Power coefficient
        'FM'         : Figure of Merit
        'ctrl_thrust': Equivalent MuJoCo ctrl value (per rotor)
        'ctrl_yaw'   : Equivalent yaw ctrl (net torque)
        'converged'  : bool
        'radial'     : dict of per-element arrays
    """
    R = BLADE_RADIUS
    Omega = blade.omega()
    n_b = NUM_BLADES
    mu = rho * MARS_VISCOSITY / rho if rho > 0 else MARS_VISCOSITY  # kinematic -> dynamic

    # Dynamic viscosity for Reynolds number
    mu_dyn = MARS_VISCOSITY  # Pa·s (already dynamic viscosity)

    # Radial stations (avoid hub center and exact tip)
    r_min = 0.10 * R
    r_max = 0.98 * R
    r_stations = np.linspace(r_min, r_max, n_elements)
    dr = r_stations[1] - r_stations[0]

    # Get blade properties at each station
    r_over_R = r_stations / R
    chords = blade.chord_at(r_over_R)
    thetas = blade.twist_rad_at(r_over_R)
    tcs = blade.tc_at(r_over_R)

    # Output arrays
    dT_arr = np.zeros(n_elements)
    dQ_arr = np.zeros(n_elements)
    alpha_arr = np.zeros(n_elements)
    Cl_arr = np.zeros(n_elements)
    Cd_arr = np.zeros(n_elements)
    phi_arr = np.zeros(n_elements)
    F_arr = np.zeros(n_elements)
    converged_all = True

    for i, r in enumerate(r_stations):
        c = chords[i]
        theta = thetas[i]
        tc = tcs[i]
        sigma_local = n_b * c / (2.0 * np.pi * r)

        # Initial inflow guess (simple momentum: lambda = sqrt(sigma*Cl_alpha*theta/16))
        Cl_alpha_est = 2.0 * np.pi * 0.85  # Rough estimate with Re correction
        lam = np.sqrt(max(sigma_local * Cl_alpha_est * max(theta, 0.01) / 16.0, 1e-8))

        converged_station = False
        for iteration in range(max_iter):
            # Induced velocity
            v_i = lam * Omega * R

            # Inflow angle
            Omega_r = Omega * r
            if Omega_r < 1e-3:
                phi = np.pi / 4.0
            else:
                phi = np.arctan2(v_i, Omega_r)

            # Angle of attack
            alpha = theta - phi

            # Resultant velocity
            V_rel = np.sqrt(Omega_r ** 2 + v_i ** 2)

            # Reynolds and Mach
            Re = rho * V_rel * c / mu_dyn
            Mach = V_rel / MARS_SPEED_OF_SOUND

            # Airfoil polar
            Cl, Cd = airfoil_polar(alpha, tc, blade.camber, Re, Mach)

            # Tip loss
            F = prandtl_tip_loss(r, R, n_b, phi)

            # New inflow from BEM equation
            # sigma * Cl / (16 * F) * (sqrt(1 + 32*F*theta*r/(sigma*Cl_alpha_est*R)) - 1)
            # Simplified: lambda_new from momentum-BEM balance
            if F > 0.01 and abs(np.sin(phi)) > 1e-6 and abs(np.cos(phi)) > 1e-6:
                # BEM: dT = 0.5 * rho * V_rel^2 * c * Cl * cos(phi) * dr (N_b blades)
                # Momentum: dT = 4 * pi * r * rho * v_i^2 * F * dr (annular)
                # Equate and solve for v_i:
                dT_elem = 0.5 * rho * V_rel ** 2 * c * Cl * np.cos(phi)
                dT_mom_coeff = 4.0 * np.pi * r * rho * F

                if dT_mom_coeff > 1e-10:
                    v_i_new_sq = n_b * dT_elem / dT_mom_coeff
                    if v_i_new_sq > 0:
                        v_i_new = np.sqrt(v_i_new_sq)
                    else:
                        v_i_new = v_i * 0.5
                else:
                    v_i_new = v_i

                lam_new = v_i_new / (Omega * R) if Omega * R > 1e-3 else lam
            else:
                lam_new = lam

            # Relaxed update
            relax = 0.3
            lam_old = lam
            lam = relax * lam_new + (1.0 - relax) * lam

            # Convergence check
            if abs(lam - lam_old) < tol:
                converged_station = True
                break

        if not converged_station:
            converged_all = False

        # Compute final elemental forces
        V_rel = np.sqrt((Omega * r) ** 2 + (lam * Omega * R) ** 2)
        phi_final = np.arctan2(lam * Omega * R, Omega * r)
        alpha_final = theta - phi_final
        Re_final = rho * V_rel * c / mu_dyn
        Mach_final = V_rel / MARS_SPEED_OF_SOUND
        Cl_f, Cd_f = airfoil_polar(alpha_final, tc, blade.camber, Re_final, Mach_final)
        F_f = prandtl_tip_loss(r, R, n_b, phi_final)

        # dT per blade, per dr
        dT = 0.5 * rho * V_rel ** 2 * c * (
            Cl_f * np.cos(phi_final) - Cd_f * np.sin(phi_final)
        ) * F_f * dr

        # dQ per blade, per dr
        dQ = 0.5 * rho * V_rel ** 2 * c * (
            Cl_f * np.sin(phi_final) + Cd_f * np.cos(phi_final)
        ) * r * F_f * dr

        dT_arr[i] = dT
        dQ_arr[i] = dQ
        alpha_arr[i] = alpha_final
        Cl_arr[i] = Cl_f
        Cd_arr[i] = Cd_f
        phi_arr[i] = phi_final
        F_arr[i] = F_f

    # Integrate (sum over elements, multiply by N_blades)
    T_per_rotor = n_b * np.sum(dT_arr)
    Q_per_rotor = n_b * np.sum(dQ_arr)
    T_total = NUM_ROTORS * T_per_rotor
    P_per_rotor = Q_per_rotor * Omega
    P_total = NUM_ROTORS * P_per_rotor

    # Non-dimensional coefficients
    Omega_R = Omega * R
    if rho > 0 and Omega_R > 0:
        CT = T_per_rotor / (rho * BLADE_AREA * Omega_R ** 2)
        CP = P_per_rotor / (rho * BLADE_AREA * Omega_R ** 3)
    else:
        CT = 0.0
        CP = 0.0

    # Figure of Merit
    if CP > 1e-10:
        FM = CT ** 1.5 / (np.sqrt(2.0) * CP)
    else:
        FM = 0.0

    # MuJoCo ctrl values
    ctrl_thrust = T_per_rotor / THRUST_GEAR if THRUST_GEAR > 0 else 0.0
    # Net yaw torque: for identical coaxial rotors, net ≈ 0
    # But top CW and bottom CCW: small imbalance modeled as 2% differential
    ctrl_yaw = (Q_per_rotor * 0.02) / YAW_GEAR if YAW_GEAR > 0 else 0.0

    if verbose:
        print(f"BEMT Results (rho={rho:.4f}, RPM={blade.rpm:.0f}):")
        print(f"  T/rotor = {T_per_rotor:.4f} N  |  T_total = {T_total:.4f} N")
        print(f"  Q/rotor = {Q_per_rotor:.6f} Nm")
        print(f"  P_total = {P_total:.3f} W")
        print(f"  CT = {CT:.6f}  |  CP = {CP:.8f}  |  FM = {FM:.4f}")
        print(f"  ctrl_thrust = {ctrl_thrust:.4f}  |  ctrl_yaw = {ctrl_yaw:.6f}")
        print(f"  Converged: {converged_all}")

    return {
        'T': T_total,
        'T_per_rotor': T_per_rotor,
        'Q': Q_per_rotor,
        'P': P_total,
        'CT': CT,
        'CP': CP,
        'FM': FM,
        'ctrl_thrust': ctrl_thrust,
        'ctrl_yaw': ctrl_yaw,
        'converged': converged_all,
        'radial': {
            'r': r_stations,
            'r_over_R': r_over_R,
            'dT': dT_arr,
            'dQ': dQ_arr,
            'alpha': alpha_arr,
            'Cl': Cl_arr,
            'Cd': Cd_arr,
            'phi': phi_arr,
            'F': F_arr,
            'chord': chords,
            'twist': thetas,
        },
    }


def bemt_to_ctrl(bemt_result: dict) -> float:
    """Extract MuJoCo thrust ctrl value from BEMT result."""
    return bemt_result['ctrl_thrust']


def hover_thrust_required(rho: float, rpm: float) -> float:
    """
    Minimum thrust coefficient required for hover.
    T_req = mg / 2  (per rotor)
    CT_req = T_req / (rho * A * (Omega*R)^2)
    """
    from config import MARS_WEIGHT
    T_req = MARS_WEIGHT / NUM_ROTORS
    Omega = rpm * 2.0 * np.pi / 60.0
    Omega_R = Omega * BLADE_RADIUS
    if rho > 0 and Omega_R > 0:
        return T_req / (rho * BLADE_AREA * Omega_R ** 2)
    return float('inf')


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == "__main__":
    from blade_param import baseline_design
    from config import RHO_NOMINAL, MARS_WEIGHT

    bl = baseline_design()
    print(f"Baseline blade: {bl}")
    print(f"Mars weight (total): {MARS_WEIGHT:.3f} N")
    print(f"Required T/rotor:    {MARS_WEIGHT / 2:.3f} N")
    print()

    result = bemt_hover(bl, rho=RHO_NOMINAL, verbose=True)
    print()

    # Check if design can hover
    can_hover = result['T_per_rotor'] >= MARS_WEIGHT / 2
    print(f"Can hover: {can_hover}  "
          f"(T/rotor={result['T_per_rotor']:.3f} N vs required {MARS_WEIGHT/2:.3f} N)")
    print(f"Thrust margin: {(result['T_per_rotor'] / (MARS_WEIGHT/2) - 1) * 100:.1f}%")
