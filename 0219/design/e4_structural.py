"""
E4: Structural and Vibration Evaluation (No MuJoCo required)

Evaluates blade structural constraints for the baseline design:
  1) First flapwise bending frequency (non-rotating + rotating)
  2) Resonance margin check (1P/2P/3P/4P harmonics)
  3) Blade mass and total rotor mass
  4) Design feasibility summary

Outputs:
  - results/e4/structural_baseline.json
  - Console report

Usage:
  cd E:/mujoco_projects/ingenuity-mujoco
  python 0219/design/e4_structural.py
"""

import sys
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    F1_BENDING_MIN, RESONANCE_MARGIN, NUM_BLADES, NUM_ROTORS,
    BLADE_RADIUS, E_MATERIAL, RHO_MATERIAL, SPAR_FRACTION,
    DESIGN_VAR_NAMES, DESIGN_LOWER, DESIGN_UPPER,
)
from blade_param import baseline_design, BladeDesign
from structural import (
    evaluate_structural,
    estimate_non_rotating_freq,
    estimate_first_bending_freq,
    check_resonance_margins,
    blade_mass_estimate,
    blade_section_properties,
)
from bemt import bemt_hover
from config import RHO_NOMINAL, MARS_WEIGHT
from utils import ensure_results_dir, save_json


def format_margin_status(margin: float, threshold: float) -> str:
    """Return OK/FAIL string with margin value."""
    status = "OK  " if margin >= threshold else "FAIL"
    return f"{margin:6.2f} Hz  [{status}]"


def run_e4():
    print("=" * 70)
    print("E4: Structural and Vibration Evaluation (Baseline Design)")
    print("=" * 70)
    print()

    results_dir = ensure_results_dir("e4")
    blade = baseline_design()

    print(f"  Blade: {blade}")
    print(f"  Tip Mach: {blade.tip_mach():.4f}")
    print(f"  Solidity: {blade.solidity():.4f}")
    print()

    # ── 1. 단면 특성 프로파일 ──────────────────────────────────────────────
    print("[Section Properties at 5 radial stations]")
    print(f"  {'r/R':>5}  {'chord(m)':>9}  {'tc':>6}  {'A(m^2)':>10}  {'I(m^4)':>12}")
    r_stations = np.array([0.20, 0.40, 0.60, 0.80, 0.95])
    chords = blade.chord_at(r_stations)
    tcs = blade.tc_at(r_stations)
    section_data = []
    for i, rr in enumerate(r_stations):
        A, I = blade_section_properties(chords[i], tcs[i])
        print(f"  {rr:5.2f}  {chords[i]:9.4f}  {tcs[i]:6.3f}  {A:10.4e}  {I:12.4e}")
        section_data.append({'r_over_R': rr, 'chord': chords[i], 'tc': tcs[i], 'A': A, 'I': I})
    print()

    # ── 2. 고유진동수 ──────────────────────────────────────────────────────
    print("[Natural Frequency Analysis]")
    f1_nr = estimate_non_rotating_freq(blade)
    f1 = estimate_first_bending_freq(blade)
    f_rev = blade.omega() / (2.0 * np.pi)

    print(f"  Rotation frequency:      {f_rev:.2f} Hz  ({blade.rpm:.0f} RPM)")
    print(f"  1st flapwise (non-rot):  {f1_nr:.2f} Hz")
    print(f"  1st flapwise (rotating): {f1:.2f} Hz  (Southwell corrected)")
    print(f"  Constraint:              f1 >= {F1_BENDING_MIN:.0f} Hz  →  "
          f"{'PASS' if f1 >= F1_BENDING_MIN else 'FAIL'}")
    print()

    # ── 3. 공진 마진 ──────────────────────────────────────────────────────
    print("[Resonance Margin Analysis (must be >= 5 Hz from each harmonic)]")
    resonance_result = check_resonance_margins(blade, harmonics=[1, 2, 3, 4])
    all_ok = True
    for n, margin in resonance_result['margins'].items():
        status = format_margin_status(margin, RESONANCE_MARGIN)
        print(f"  {n}P ({n * f_rev:.2f} Hz):  {status}")
        if margin < RESONANCE_MARGIN:
            all_ok = False
    print(f"  Overall resonance: {'PASS' if all_ok else 'FAIL'}")
    print()

    # ── 4. 질량 추정 ──────────────────────────────────────────────────────
    print("[Mass Estimates]")
    mass_single = blade_mass_estimate(blade)
    total_rotor_mass = mass_single * NUM_BLADES * NUM_ROTORS

    print(f"  Single blade mass:    {mass_single * 1000:.1f} g")
    print(f"  Total rotor mass:     {total_rotor_mass * 1000:.1f} g "
          f"({NUM_BLADES} blades × {NUM_ROTORS} rotors)")
    print(f"  Vehicle total mass:   1600 g  (given)")
    print(f"  Rotor mass fraction:  {total_rotor_mass / 1.6 * 100:.1f}%")
    print()

    # ── 5. BEMT + 구조 통합 평가 ───────────────────────────────────────────
    print("[BEMT + Structural Integrated Evaluation]")
    bemt_result = bemt_hover(blade, rho=RHO_NOMINAL, verbose=False)
    T_req = MARS_WEIGHT / 2.0
    T_margin_pct = (bemt_result['T_per_rotor'] / T_req - 1.0) * 100.0

    print(f"  Hover thrust required: {T_req:.3f} N/rotor")
    print(f"  BEMT thrust achieved:  {bemt_result['T_per_rotor']:.3f} N/rotor  "
          f"(margin: {T_margin_pct:+.1f}%)")
    print(f"  Hover power:           {bemt_result['P']:.2f} W")
    print(f"  Figure of Merit:       {bemt_result['FM']:.4f}")
    print(f"  Tip Mach:              {blade.tip_mach():.4f}")
    print()

    # ── 6. 최적화 제약 체크리스트 ─────────────────────────────────────────
    print("[Optimization Constraint Checklist]")
    constraints = {}

    # Structural
    struct_result = evaluate_structural(blade)
    constraints['f1_bending_ok'] = struct_result['f1_feasible']
    constraints['resonance_ok'] = struct_result['resonance_feasible']
    constraints['structural_ok'] = struct_result['feasible']

    # Aerodynamic
    rpm_ok = blade.rpm <= 3200.0
    mach_ok = blade.tip_mach() <= 0.80
    hover_ok = bemt_result['T_per_rotor'] >= T_req

    constraints['rpm_ok'] = rpm_ok
    constraints['tip_mach_ok'] = mach_ok
    constraints['hover_feasible'] = hover_ok

    # Summary
    all_constraints_ok = all(constraints.values())

    print(f"  RPM ≤ 3200:          {'PASS' if rpm_ok else 'FAIL'}  ({blade.rpm:.0f} RPM)")
    print(f"  Tip Mach ≤ 0.80:     {'PASS' if mach_ok else 'FAIL'}  ({blade.tip_mach():.4f})")
    print(f"  Hover capable:       {'PASS' if hover_ok else 'FAIL'}  "
          f"(T_margin={T_margin_pct:+.1f}%)")
    print(f"  f1 ≥ {F1_BENDING_MIN:.0f} Hz:       "
          f"{'PASS' if constraints['f1_bending_ok'] else 'FAIL'}  "
          f"({struct_result['f1_bending']:.1f} Hz)")
    print(f"  Resonance margins:   {'PASS' if constraints['resonance_ok'] else 'FAIL'}")
    print()
    print(f"  ALL CONSTRAINTS:     {'PASS' if all_constraints_ok else 'FAIL'}")

    print("=" * 70)

    # ── 저장 ──────────────────────────────────────────────────────────────
    summary = {
        'design_vector': blade.vector.tolist(),
        'design_var_names': DESIGN_VAR_NAMES,
        'natural_frequency': {
            'f1_non_rotating_hz': f1_nr,
            'f1_rotating_hz': f1,
            'f_rev_hz': f_rev,
            'rpm': blade.rpm,
        },
        'resonance': {
            'margins_hz': {str(n): m for n, m in resonance_result['margins'].items()},
            'all_ok': all_ok,
        },
        'mass': {
            'single_blade_kg': mass_single,
            'total_rotor_kg': total_rotor_mass,
        },
        'bemt': {
            'T_per_rotor': bemt_result['T_per_rotor'],
            'P_total': bemt_result['P'],
            'FM': bemt_result['FM'],
            'CT': bemt_result['CT'],
            'CP': bemt_result['CP'],
            'T_margin_pct': T_margin_pct,
        },
        'constraints': constraints,
        'all_constraints_ok': all_constraints_ok,
        'section_profiles': section_data,
        'params': {
            'rho': RHO_NOMINAL,
            'f1_bending_min': F1_BENDING_MIN,
            'resonance_margin': RESONANCE_MARGIN,
            'E_material': E_MATERIAL,
            'rho_material': RHO_MATERIAL,
            'spar_fraction': SPAR_FRACTION,
        },
    }
    save_json(str(results_dir / "structural_baseline.json"), summary)

    print(f"\nResults saved to: {results_dir}")
    return summary


if __name__ == "__main__":
    run_e4()
