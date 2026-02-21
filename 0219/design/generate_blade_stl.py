"""
Generate STL files for optimized blade designs.

Creates 3D blade geometry from BladeDesign parameters:
  - Airfoil cross-sections at radial stations
  - NACA-style profile with camber and thickness
  - Lofted surface between stations
  - Two-bladed rotor (180° apart)

Outputs:
  - assets/optimized_topblades.stl
  - assets/optimized_bottomblades.stl

Usage:
  cd E:/mujoco_projects/ingenuity-mujoco
  python 0219/design/generate_blade_stl.py
"""

import struct
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import BLADE_RADIUS, RESULTS_DIR, PROJECT_ROOT
from blade_param import BladeDesign, baseline_design

# ─── Hub grafting parameters ────────────────────────────────────────────────

ORIGINAL_TOP_STL = "mhs_topblades_v16.stl"
ORIGINAL_BOT_STL = "mhs_bottomblades_v16.stl"

# Top rotor: small hub connector, blade starts close to hub
Y_CUT_TOP = 0.07        # Hub triangle filter threshold (m)
Y_START_TOP = 0.065      # Blade geometry start — overlaps hub by 5mm

# Bottom rotor: larger hub connector extends further along span
# Hub connector thick (0.06m) out to Y≈0.08, actual blade starts at Y≈0.085
Y_CUT_BOT = 0.09         # Captures the full bottom connector
Y_START_BOT = 0.085       # Blade starts where hub connector ends

TRANSITION_R_END = 0.20  # r/R where transition to pure airfoil completes


# ─── STL I/O ────────────────────────────────────────────────────────────────

def read_binary_stl(filepath: str) -> np.ndarray:
    """Read binary STL, return (N, 3, 3) triangle vertices."""
    with open(filepath, 'rb') as f:
        f.read(80)  # header
        n_tri = struct.unpack('<I', f.read(4))[0]
        tris = np.zeros((n_tri, 3, 3), dtype=np.float32)
        for i in range(n_tri):
            f.read(12)  # normal
            for j in range(3):
                tris[i, j] = struct.unpack('<3f', f.read(12))
            f.read(2)  # attribute
    return tris


def extract_hub_triangles(original_stl_path: str, y_cut: float = Y_CUT_TOP) -> np.ndarray:
    """Extract hub region: triangles where ALL 3 vertices have |Y| < y_cut."""
    tris = read_binary_stl(original_stl_path)
    hub_mask = np.all(np.abs(tris[:, :, 1]) < y_cut, axis=1)
    hub_tris = tris[hub_mask]
    print(f"    Hub extraction: {hub_mask.sum()}/{len(tris)} triangles (|Y| < {y_cut})")
    return hub_tris


def get_hub_neck_profile(original_stl_path: str, y_pos: float,
                         dy: float = 0.005) -> dict:
    """Measure the hub cross-section at Y=y_pos (positive side).

    Returns dict with x_center, z_center, chord, thickness, and bounds.
    """
    tris = read_binary_stl(original_stl_path)
    all_v = tris.reshape(-1, 3)
    near = all_v[(all_v[:, 1] > y_pos - dy) & (all_v[:, 1] < y_pos + dy)]

    x_min, x_max = float(near[:, 0].min()), float(near[:, 0].max())
    z_min, z_max = float(near[:, 2].min()), float(near[:, 2].max())

    return {
        'x_center': (x_min + x_max) / 2,
        'z_center': (z_min + z_max) / 2,
        'chord': x_max - x_min,
        'thickness': z_max - z_min,
        'x_min': x_min, 'x_max': x_max,
        'z_min': z_min, 'z_max': z_max,
    }


# ─── Airfoil generation ─────────────────────────────────────────────────────

def naca_airfoil(tc: float, camber: float, n_pts: int = 30) -> np.ndarray:
    """
    Generate a NACA-style airfoil profile.

    Parameters
    ----------
    tc     : thickness-to-chord ratio
    camber : max camber as fraction of chord
    n_pts  : number of points per surface

    Returns
    -------
    pts : (2*n_pts, 2) array of (x, z) coordinates, chord-normalized [0,1]
          Upper surface from TE→LE, then lower surface from LE→TE (closed loop)
    """
    # Cosine-spaced x for finer leading edge resolution
    beta = np.linspace(0, np.pi, n_pts)
    x = 0.5 * (1.0 - np.cos(beta))

    # Thickness distribution (NACA 4-digit style)
    t = tc  # max thickness ratio
    yt = (t / 0.2) * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    # Camber line (parabolic, max at 40% chord)
    p = 0.4  # position of max camber
    m = camber  # max camber value

    yc = np.where(
        x < p,
        m / p**2 * (2 * p * x - x**2),
        m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2),
    )

    # Upper and lower surfaces
    xu = x
    zu = yc + yt
    xl = x
    zl = yc - yt

    # Combine: upper TE→LE, lower LE→TE
    upper = np.column_stack([xu[::-1], zu[::-1]])
    lower = np.column_stack([xl[1:], zl[1:]])  # skip duplicate LE

    return np.vstack([upper, lower])


# ─── Blade geometry ─────────────────────────────────────────────────────────

def ellipse_section(cx: float, cz: float, semi_x: float, semi_z: float,
                    n_pts: int) -> np.ndarray:
    """Generate an elliptical cross-section centered at (cx, cz).

    Returns (2*n_pts-1, 2) array to match airfoil point count.
    """
    total_pts = 2 * n_pts - 1
    theta = np.linspace(0, 2 * np.pi, total_pts, endpoint=False)
    x = cx + semi_x * np.cos(theta)
    z = cz + semi_z * np.sin(theta)
    return np.column_stack([x, z])


def generate_blade_sections(blade: BladeDesign, n_stations: int = 25,
                            n_airfoil_pts: int = 30,
                            hub_neck: dict = None,
                            y_start: float = Y_START_TOP,
                            flip_camber: bool = False) -> list:
    """
    Generate 3D cross-sections along the blade span.

    If hub_neck is provided (graft mode), starts at y_start with an ellipse
    matching the hub neck profile and transitions to the optimized airfoil
    by r/R = TRANSITION_R_END.

    If hub_neck is None (fallback), uses old circular-to-airfoil transition
    from r/R=0.0.

    Returns list of (r_frac, points_3d) where points_3d is (N, 3) array.
    Each section is in the blade frame: x=chordwise, y=spanwise, z=thickness.
    """
    if hub_neck is not None:
        # ── Graft mode: start at y_start, transition to pure airfoil ──
        r_start = y_start / BLADE_RADIUS  # ~0.107
        transition_stations = np.linspace(r_start, TRANSITION_R_END, 8)
        blade_stations = np.linspace(TRANSITION_R_END + 0.01, 0.98, n_stations)
        all_stations = np.concatenate([transition_stations, blade_stations])

        neck_cx = hub_neck['x_center']
        neck_cz = hub_neck['z_center']
        neck_semi_x = hub_neck['chord'] / 2
        neck_semi_z = hub_neck['thickness'] / 2
    else:
        # ── Fallback: old circular hub transition ──
        hub_stations = np.array([0.0, 0.03, 0.06, 0.10, 0.13])
        blade_stations = np.linspace(0.15, 0.98, n_stations)
        all_stations = np.concatenate([hub_stations, blade_stations])

        r_start = 0.0
        neck_cx = 0.0
        neck_cz = 0.0
        neck_semi_x = 0.015
        neck_semi_z = 0.015

    cam = blade.camber
    # For counter-rotating bottom rotor: negate camber to flip lift direction
    # while keeping the section centered (no X-coordinate offset).
    effective_cam = -cam if flip_camber else cam
    sections = []

    for r_frac in all_stations:
        r_m = r_frac * BLADE_RADIUS

        if hub_neck is not None:
            # Graft mode transition
            is_transition = r_frac < TRANSITION_R_END
        else:
            # Fallback mode transition
            is_transition = r_frac < 0.15

        if is_transition:
            # ── Transition zone: blend ellipse → airfoil ──
            if hub_neck is not None:
                t_raw = (r_frac - r_start) / (TRANSITION_R_END - r_start)
            else:
                t_raw = r_frac / 0.15
            t_raw = np.clip(t_raw, 0.0, 1.0)
            # Cubic smoothstep for visual quality
            t_smooth = 3 * t_raw**2 - 2 * t_raw**3

            # Ellipse section — slightly grows with span
            grow = 1.0 + 0.5 * t_raw
            ell = ellipse_section(neck_cx, neck_cz,
                                  neck_semi_x * grow,
                                  neck_semi_z * grow,
                                  n_airfoil_pts)

            # Airfoil at this station (or at root if r_frac < 0.15)
            r_query = max(r_frac, 0.15)
            chord = float(blade.chord_at(r_query))
            twist_deg = float(blade.twist_at(r_query))
            tc = float(blade.tc_at(r_query))

            airfoil_2d = naca_airfoil(tc, effective_cam, n_pts=n_airfoil_pts)
            x_af = (airfoil_2d[:, 0] - 0.25) * chord
            z_af = airfoil_2d[:, 1] * chord

            twist_rad = np.radians(twist_deg)
            cos_t, sin_t = np.cos(twist_rad), np.sin(twist_rad)
            x_af_rot = x_af * cos_t - z_af * sin_t
            z_af_rot = x_af * sin_t + z_af * cos_t

            x_local = (1.0 - t_smooth) * ell[:, 0] + t_smooth * x_af_rot
            z_local = (1.0 - t_smooth) * ell[:, 1] + t_smooth * z_af_rot
        else:
            # ── Pure airfoil section ──
            chord = float(blade.chord_at(r_frac))
            twist_deg = float(blade.twist_at(r_frac))
            tc = float(blade.tc_at(r_frac))

            profile_2d = naca_airfoil(tc, effective_cam, n_pts=n_airfoil_pts)
            x_local = (profile_2d[:, 0] - 0.25) * chord
            z_local = profile_2d[:, 1] * chord

            twist_rad = np.radians(twist_deg)
            cos_t, sin_t = np.cos(twist_rad), np.sin(twist_rad)
            x_new = x_local * cos_t - z_local * sin_t
            z_new = x_local * sin_t + z_local * cos_t
            x_local = x_new
            z_local = z_new

        # 3D: x=chordwise, y=spanwise(radial), z=thickness
        pts_3d = np.zeros((len(x_local), 3))
        pts_3d[:, 0] = x_local
        pts_3d[:, 1] = r_m
        pts_3d[:, 2] = z_local

        sections.append((r_frac, pts_3d))

    return sections


def loft_blade_surface(sections: list) -> np.ndarray:
    """
    Loft between blade cross-sections to create triangulated surface.

    Returns (N_tri, 3, 3) array of triangle vertices.
    """
    triangles = []

    for i in range(len(sections) - 1):
        _, pts_a = sections[i]
        _, pts_b = sections[i + 1]

        n = len(pts_a)
        assert len(pts_b) == n, "Sections must have same point count"

        for j in range(n):
            j_next = (j + 1) % n

            # Two triangles per quad
            v0 = pts_a[j]
            v1 = pts_a[j_next]
            v2 = pts_b[j_next]
            v3 = pts_b[j]

            triangles.append([v0, v1, v2])
            triangles.append([v0, v2, v3])

    # Add tip cap
    _, pts_tip = sections[-1]
    tip_center = pts_tip.mean(axis=0)
    for j in range(len(pts_tip)):
        j_next = (j + 1) % len(pts_tip)
        triangles.append([tip_center, pts_tip[j], pts_tip[j_next]])

    # Add root cap
    _, pts_root = sections[0]
    root_center = pts_root.mean(axis=0)
    for j in range(len(pts_root)):
        j_next = (j + 1) % len(pts_root)
        triangles.append([root_center, pts_root[j_next], pts_root[j]])

    return np.array(triangles)


def make_two_blade_rotor(single_blade_tris: np.ndarray) -> np.ndarray:
    """
    Create a two-bladed rotor by rotating the blade 180 degrees.
    The rotor spins about the Z axis, blades extend along Y.
    """
    # Blade 1: as-is (already extends along +Y)
    blade1 = single_blade_tris.copy()

    # Blade 2: rotate 180° about Z axis
    blade2 = single_blade_tris.copy()
    blade2[:, :, 0] = -blade2[:, :, 0]  # x → -x
    blade2[:, :, 1] = -blade2[:, :, 1]  # y → -y
    # Flip triangle winding for correct normals
    blade2 = blade2[:, ::-1, :]

    return np.concatenate([blade1, blade2], axis=0)


# ─── STL Write ──────────────────────────────────────────────────────────────

def write_binary_stl(filepath: str, triangles: np.ndarray, name: str = "blade"):
    """Write binary STL file from (N, 3, 3) triangle array."""
    n_tri = len(triangles)

    with open(filepath, 'wb') as f:
        # 80-byte header
        header = f"{name}".encode('ascii')[:80].ljust(80, b'\x00')
        f.write(header)

        # Number of triangles
        f.write(struct.pack('<I', n_tri))

        for tri in triangles:
            v0, v1, v2 = tri
            # Compute normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-12:
                normal /= norm_len

            # Write: normal(3f), v0(3f), v1(3f), v2(3f), attr(H)
            f.write(struct.pack('<3f', *normal))
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))

    print(f"  Written: {filepath} ({n_tri} triangles, {Path(filepath).stat().st_size / 1024:.0f} KB)")


# ─── Main ────────────────────────────────────────────────────────────────────

def generate_stl_for_design(blade: BladeDesign, output_dir: str,
                            prefix: str = "optimized",
                            graft_hub: bool = True) -> dict:
    """
    Generate top and bottom rotor STL files for a blade design.

    If graft_hub=True and original STLs exist, extracts the real hub geometry
    from the original files and grafts optimized blade airfoils onto it.

    Returns dict with file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Blade: {blade}")
    print(f"  Output: {output_dir}")

    assets_dir = PROJECT_ROOT / "assets"
    original_top = assets_dir / ORIGINAL_TOP_STL
    original_bot = assets_dir / ORIGINAL_BOT_STL
    use_graft = graft_hub and original_top.exists() and original_bot.exists()

    if use_graft:
        print(f"  Mode: Hub grafting (overlap method)")

        # ── TOP ROTOR ──────────────────────────────────────────────────
        print(f"\n  [Top rotor] (Y_CUT={Y_CUT_TOP}, Y_START={Y_START_TOP})")
        hub_tris_top = extract_hub_triangles(str(original_top), Y_CUT_TOP)
        neck_top = get_hub_neck_profile(str(original_top), Y_START_TOP)
        print(f"    Neck at Y={Y_START_TOP}: chord={neck_top['chord']:.4f}m "
              f"thick={neck_top['thickness']:.4f}m "
              f"center=({neck_top['x_center']:.4f}, {neck_top['z_center']:.4f})")

        sections_top = generate_blade_sections(
            blade, hub_neck=neck_top, y_start=Y_START_TOP)
        blade_tris_single_top = loft_blade_surface(sections_top)
        blade_tris_top = make_two_blade_rotor(blade_tris_single_top)
        top_combined = np.concatenate([hub_tris_top, blade_tris_top], axis=0)
        print(f"    Hub: {len(hub_tris_top)}, Blade: {len(blade_tris_top)}, "
              f"Total: {len(top_combined)} triangles")

        top_path = str(output_dir / f"{prefix}_topblades.stl")
        write_binary_stl(top_path, top_combined, name=f"{prefix}_top_grafted")

        # ── BOTTOM ROTOR ───────────────────────────────────────────────
        # Bottom hub connector is larger, so use wider Y_CUT and Y_START
        print(f"\n  [Bottom rotor] (Y_CUT={Y_CUT_BOT}, Y_START={Y_START_BOT})")
        hub_tris_bot = extract_hub_triangles(str(original_bot), Y_CUT_BOT)
        neck_bot = get_hub_neck_profile(str(original_bot), Y_START_BOT)
        print(f"    Neck at Y={Y_START_BOT}: chord={neck_bot['chord']:.4f}m "
              f"thick={neck_bot['thickness']:.4f}m "
              f"center=({neck_bot['x_center']:.4f}, {neck_bot['z_center']:.4f})")

        sections_bot = generate_blade_sections(
            blade, hub_neck=neck_bot, y_start=Y_START_BOT, flip_camber=True)
        blade_tris_single_bot = loft_blade_surface(sections_bot)

        # Bottom rotor: NO X-mirror needed.
        # MuJoCo handles counter-rotation via quat="0.6 0 0 1" in XML.
        # Only camber direction is flipped (via flip_camber=True above).
        blade_tris_bot = make_two_blade_rotor(blade_tris_single_bot)
        bot_combined = np.concatenate([hub_tris_bot, blade_tris_bot], axis=0)
        print(f"    Hub: {len(hub_tris_bot)}, Blade: {len(blade_tris_bot)}, "
              f"Total: {len(bot_combined)} triangles")

        bot_path = str(output_dir / f"{prefix}_bottomblades.stl")
        write_binary_stl(bot_path, bot_combined, name=f"{prefix}_bot_grafted")

        n_top = len(top_combined)
        n_bot = len(bot_combined)
    else:
        # ── Fallback: simple generated hub (no original STL) ───────────
        if graft_hub:
            print("  WARNING: Original STLs not found, falling back to generated hub")
        print(f"  Mode: Generated hub (circular transition)")

        sections = generate_blade_sections(blade)
        single_tris = loft_blade_surface(sections)
        rotor_tris = make_two_blade_rotor(single_tris)
        print(f"  Two-blade rotor: {len(rotor_tris)} triangles")

        top_path = str(output_dir / f"{prefix}_topblades.stl")
        write_binary_stl(top_path, rotor_tris, name=f"{prefix}_top_rotor")

        bot_tris = rotor_tris.copy()
        bot_tris[:, :, 0] = -bot_tris[:, :, 0]
        bot_tris = bot_tris[:, ::-1, :]

        bot_path = str(output_dir / f"{prefix}_bottomblades.stl")
        write_binary_stl(bot_path, bot_tris, name=f"{prefix}_bottom_rotor")

        n_top = len(rotor_tris)
        n_bot = len(rotor_tris)

    return {
        'top_stl': top_path,
        'bottom_stl': bot_path,
        'n_triangles_top': n_top,
        'n_triangles_bottom': n_bot,
    }


def main():
    print("=" * 70)
    print("Blade STL Generator (Hub-Grafted)")
    print("=" * 70)

    assets_dir = PROJECT_ROOT / "assets"

    # ── 1) Baseline STL (no grafting — baseline uses original meshes) ────
    print("\n[1] Baseline blade STL")
    baseline = baseline_design()
    bl_info = generate_stl_for_design(
        baseline, str(assets_dir), prefix="baseline", graft_hub=False)

    # ── 2) Optimized blade STL (with hub grafting) ───────────────────────
    candidates_file = RESULTS_DIR / "pipeline" / "final_candidates.json"
    if candidates_file.exists():
        with open(candidates_file) as f:
            pipeline = json.load(f)

        candidates = pipeline.get('final_candidates', [])
        if candidates:
            best = candidates[0]
            vec = np.array(best['design_vector'])
            opt_blade = BladeDesign(vec)

            print(f"\n[2] Optimized blade STL (Pareto #{best.get('pareto_id', '?')})")
            print(f"  FM={best.get('FM', 0):.4f}  P={best.get('P_total', 0):.1f}W  "
                  f"RPM={opt_blade.rpm:.0f}")
            opt_info = generate_stl_for_design(
                opt_blade, str(assets_dir), prefix="optimized", graft_hub=True)

            # Save design info
            info = {
                'design_vector': vec.tolist(),
                'FM': best.get('FM', 0),
                'P_total': best.get('P_total', 0),
                'RPM': opt_blade.rpm,
                'tip_mach': opt_blade.tip_mach(),
                'stl_files': opt_info,
            }
            info_path = assets_dir / "optimized_blade_info.json"
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
            print(f"\n  Design info: {info_path}")
    else:
        print(f"\n  WARNING: No pipeline results found at {candidates_file}")
        print("  Run: python 0219/design/run_all.py --phase 3")

    print("\nDone.")


if __name__ == "__main__":
    main()
