"""
run_all.py — CLI Entry Point for Ingenuity Blade Optimization Testbed

Phases:
  0: Verification only (BEMT + structural quick test)
  1: E0 baseline calibration
  2: Phase 2 experiments (E1~E4)
  3: Full optimization pipeline (DOE → Surrogate → NSGA-II → MuJoCo)

Usage:
  cd E:/mujoco_projects/ingenuity-mujoco
  conda activate mujoco_air

  # Quick sanity check (Phase 0)
  python 0219/design/run_all.py --phase 0

  # E0 calibration
  python 0219/design/run_all.py --phase 1

  # All experiments (E1~E4)
  python 0219/design/run_all.py --phase 2

  # Full pipeline
  python 0219/design/run_all.py --phase 3 --doe-n 100 --gen 40

  # Individual experiments
  python 0219/design/run_all.py --phase 2 --only e1
  python 0219/design/run_all.py --phase 2 --only e3 --gust

  # Viewer mode (for experiments that support it)
  python 0219/design/run_all.py --phase 1 --viewer
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def phase0_verify():
    """Phase 0: Quick sanity check (no MuJoCo)."""
    print("=" * 70)
    print("Phase 0: Sanity Verification (BEMT + Structural)")
    print("=" * 70)

    from blade_param import baseline_design
    from bemt import bemt_hover
    from structural import evaluate_structural
    from config import RHO_NOMINAL, MARS_WEIGHT, NUM_ROTORS

    blade = baseline_design()
    print(f"\nBaseline blade: {blade}")
    print(f"  Solidity:  {blade.solidity():.4f}")
    print(f"  Tip Mach:  {blade.tip_mach():.4f}")
    print(f"  Tip speed: {blade.tip_speed():.1f} m/s")

    bemt_result = bemt_hover(blade, rho=RHO_NOMINAL, verbose=True)
    T_req = MARS_WEIGHT / NUM_ROTORS
    print(f"\nHover check: {bemt_result['T_per_rotor']:.3f} N / {T_req:.3f} N required")
    print(f"Can hover: {bemt_result['T_per_rotor'] >= T_req}")

    struct = evaluate_structural(blade)
    print(f"\nStructural:")
    print(f"  f1_bending: {struct['f1_bending']:.1f} Hz  (min 40 Hz → {'OK' if struct['f1_feasible'] else 'FAIL'})")
    print(f"  Resonance:  {'OK' if struct['resonance_feasible'] else 'FAIL'}")
    print(f"  Blade mass: {struct['blade_mass'] * 1000:.1f} g")
    print(f"  Overall:    {'FEASIBLE' if struct['feasible'] else 'INFEASIBLE'}")
    print()
    print("Phase 0 complete.")


def phase1_e0(use_viewer: bool = False):
    """Phase 1: E0 baseline calibration."""
    from e0_baseline import run_e0
    run_e0(use_viewer=use_viewer)


def phase2_experiments(use_viewer: bool = False, only: str = None,
                       enable_gust: bool = False):
    """Phase 2: Run experiment scripts E1~E4."""
    experiments = ['e1', 'e2', 'e3', 'e4']
    if only:
        experiments = [only.lower()]

    for exp in experiments:
        print(f"\n{'=' * 70}")
        print(f"Running: {exp.upper()}")
        print('=' * 70)

        if exp == 'e1':
            from e1_hover_map import run_e1
            run_e1(use_viewer=use_viewer)

        elif exp == 'e2':
            from e2_robust_hover import run_e2
            run_e2(use_viewer=use_viewer)

        elif exp == 'e3':
            from e3_forward_flight import run_e3
            run_e3(use_viewer=use_viewer, enable_gust=enable_gust)

        elif exp == 'e4':
            from e4_structural import run_e4
            run_e4()

        else:
            print(f"  Unknown experiment: {exp}")


def phase3_pipeline(doe_n: int = 100, n_gen: int = 40, pop_size: int = 60,
                    n_top: int = 5, seed: int = 0,
                    skip_doe: bool = False, skip_surrogate: bool = False,
                    skip_optimizer: bool = False, skip_mujoco: bool = False):
    """Phase 3: Full optimization pipeline."""
    from pipeline import run_pipeline
    run_pipeline(
        doe_n=doe_n,
        doe_seed=seed,
        n_gen=n_gen,
        pop_size=pop_size,
        n_top=n_top,
        skip_doe=skip_doe,
        skip_surrogate=skip_surrogate,
        skip_optimizer=skip_optimizer,
        skip_mujoco=skip_mujoco,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Ingenuity Blade Optimization Testbed — CLI Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 0219/design/run_all.py --phase 0               # Quick sanity check
  python 0219/design/run_all.py --phase 1               # E0 calibration
  python 0219/design/run_all.py --phase 1 --viewer      # E0 with MuJoCo viewer
  python 0219/design/run_all.py --phase 2               # All experiments E1-E4
  python 0219/design/run_all.py --phase 2 --only e3 --gust  # E3 with gust
  python 0219/design/run_all.py --phase 3               # Full pipeline
  python 0219/design/run_all.py --phase 3 --doe-n 200 --gen 50
  python 0219/design/run_all.py --phase 3 --skip-doe --skip-surrogate  # Rerun optimizer only
        """,
    )

    # Phase selection
    parser.add_argument("--phase", type=int, choices=[0, 1, 2, 3], default=0,
                        help="Phase to run: 0=verify, 1=E0, 2=experiments, 3=pipeline")

    # Common options
    parser.add_argument("--viewer", action="store_true",
                        help="Enable MuJoCo 3D viewer (phases 1, 2)")
    parser.add_argument("--gust", action="store_true",
                        help="Enable gust disturbance (E3 only)")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only specific experiment: e1, e2, e3, e4 (phase 2 only)")

    # Phase 3 options
    parser.add_argument("--doe-n", type=int, default=100,
                        help="DOE sample count (phase 3, default: 100)")
    parser.add_argument("--gen", type=int, default=40,
                        help="NSGA-II generations (phase 3, default: 40)")
    parser.add_argument("--pop", type=int, default=60,
                        help="Population size (phase 3, default: 60)")
    parser.add_argument("--top", type=int, default=5,
                        help="Top N for MuJoCo validation (phase 3, default: 5)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--skip-doe", action="store_true")
    parser.add_argument("--skip-surrogate", action="store_true")
    parser.add_argument("--skip-optimizer", action="store_true")
    parser.add_argument("--skip-mujoco", action="store_true",
                        help="Skip MuJoCo validation in pipeline")

    args = parser.parse_args()

    t_start = time.time()
    print()

    if args.phase == 0:
        phase0_verify()

    elif args.phase == 1:
        phase1_e0(use_viewer=args.viewer)

    elif args.phase == 2:
        phase2_experiments(
            use_viewer=args.viewer,
            only=args.only,
            enable_gust=args.gust,
        )

    elif args.phase == 3:
        phase3_pipeline(
            doe_n=args.doe_n,
            n_gen=args.gen,
            pop_size=args.pop,
            n_top=args.top,
            seed=args.seed,
            skip_doe=args.skip_doe,
            skip_surrogate=args.skip_surrogate,
            skip_optimizer=args.skip_optimizer,
            skip_mujoco=args.skip_mujoco,
        )

    elapsed = time.time() - t_start
    print(f"\n[Total elapsed: {elapsed:.1f}s]")


if __name__ == "__main__":
    main()
