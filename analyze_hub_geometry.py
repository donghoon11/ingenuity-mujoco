"""
Analyze Ingenuity Mars Helicopter blade STL files to understand hub/attachment geometry.
Focus on the central hub region and the transition to the blade airfoil.

Coordinate system: Y = radial (spanwise), X = chordwise, Z = thickness
BLADE_RADIUS = 0.606 m
"""

import struct
import numpy as np

BLADE_RADIUS = 0.606  # meters

def read_binary_stl(filepath):
    """Read a binary STL file and return triangle normals and vertices."""
    with open(filepath, 'rb') as f:
        header = f.read(80)
        num_triangles = struct.unpack('<I', f.read(4))[0]

        normals = np.zeros((num_triangles, 3), dtype=np.float32)
        vertices = np.zeros((num_triangles, 3, 3), dtype=np.float32)

        for i in range(num_triangles):
            data = struct.unpack('<12fH', f.read(50))
            normals[i] = data[0:3]
            vertices[i, 0] = data[3:6]
            vertices[i, 1] = data[6:9]
            vertices[i, 2] = data[9:12]
            # data[12] is attribute byte count (ignored)

    return normals, vertices


def get_cross_section_at_y(vertices, y_val, tolerance=0.003):
    """
    Find triangles that intersect a plane at Y=y_val.
    Return the X and Z ranges of those intersecting triangles.
    """
    y_min_per_tri = vertices[:, :, 1].min(axis=1)
    y_max_per_tri = vertices[:, :, 1].max(axis=1)

    mask = (y_min_per_tri <= y_val + tolerance) & (y_max_per_tri >= y_val - tolerance)
    selected = vertices[mask]

    if len(selected) == 0:
        return None

    all_verts = selected.reshape(-1, 3)
    x_min, x_max = all_verts[:, 0].min(), all_verts[:, 0].max()
    z_min, z_max = all_verts[:, 2].min(), all_verts[:, 2].max()
    chord = x_max - x_min
    thickness = z_max - z_min

    return {
        'n_triangles': len(selected),
        'x_min': x_min, 'x_max': x_max,
        'z_min': z_min, 'z_max': z_max,
        'chord': chord,
        'thickness': thickness,
        'thickness_to_chord': thickness / chord if chord > 1e-9 else 0.0
    }


def analyze_stl(filepath, label):
    print("=" * 80)
    print(f"  {label}")
    print(f"  File: {filepath}")
    print("=" * 80)

    normals, vertices = read_binary_stl(filepath)
    n_tri = len(vertices)

    # Flatten all vertices for global bounding box
    all_v = vertices.reshape(-1, 3)

    print(f"\n--- Global Statistics ---")
    print(f"  Total triangles: {n_tri}")
    print(f"  Bounding box:")
    print(f"    X (chord):     [{all_v[:,0].min():.6f}, {all_v[:,0].max():.6f}]  range={all_v[:,0].max()-all_v[:,0].min():.6f}")
    print(f"    Y (span/rad):  [{all_v[:,1].min():.6f}, {all_v[:,1].max():.6f}]  range={all_v[:,1].max()-all_v[:,1].min():.6f}")
    print(f"    Z (thickness): [{all_v[:,2].min():.6f}, {all_v[:,2].max():.6f}]  range={all_v[:,2].max()-all_v[:,2].min():.6f}")

    y_extent = all_v[:,1].max() - all_v[:,1].min()
    print(f"  Total Y extent as fraction of R: {y_extent/BLADE_RADIUS:.4f}")

    # ---- Hub region: all 3 vertices have |Y| <= 0.05 ----
    print(f"\n--- Hub Region (all vertices |Y| <= 0.05) ---")
    hub_mask = np.all(np.abs(vertices[:, :, 1]) <= 0.05, axis=1)
    hub_tris = vertices[hub_mask]
    hub_v = hub_tris.reshape(-1, 3) if len(hub_tris) > 0 else np.zeros((0, 3))

    print(f"  Triangles in hub region: {len(hub_tris)}")
    if len(hub_tris) > 0:
        print(f"  Hub bounding box:")
        print(f"    X: [{hub_v[:,0].min():.6f}, {hub_v[:,0].max():.6f}]  range={hub_v[:,0].max()-hub_v[:,0].min():.6f}")
        print(f"    Y: [{hub_v[:,1].min():.6f}, {hub_v[:,1].max():.6f}]  range={hub_v[:,1].max()-hub_v[:,1].min():.6f}")
        print(f"    Z: [{hub_v[:,2].min():.6f}, {hub_v[:,2].max():.6f}]  range={hub_v[:,2].max()-hub_v[:,2].min():.6f}")

    # ---- Hub region with wider Y: |Y| <= 0.03 (tighter) ----
    print(f"\n--- Tight Hub Core (all vertices |Y| <= 0.03) ---")
    core_mask = np.all(np.abs(vertices[:, :, 1]) <= 0.03, axis=1)
    core_tris = vertices[core_mask]
    core_v = core_tris.reshape(-1, 3) if len(core_tris) > 0 else np.zeros((0, 3))

    print(f"  Triangles in core: {len(core_tris)}")
    if len(core_tris) > 0:
        print(f"  Core bounding box:")
        print(f"    X: [{core_v[:,0].min():.6f}, {core_v[:,0].max():.6f}]  range={core_v[:,0].max()-core_v[:,0].min():.6f}")
        print(f"    Y: [{core_v[:,1].min():.6f}, {core_v[:,1].max():.6f}]  range={core_v[:,1].max()-core_v[:,1].min():.6f}")
        print(f"    Z: [{core_v[:,2].min():.6f}, {core_v[:,2].max():.6f}]  range={core_v[:,2].max()-core_v[:,2].min():.6f}")

    # ---- Transition zone: any vertex with Y in [0.05, 0.12] ----
    print(f"\n--- Transition Zone (any vertex Y in [0.05, 0.12]) ---")
    trans_mask = np.any((vertices[:, :, 1] >= 0.05) & (vertices[:, :, 1] <= 0.12), axis=1)
    trans_tris = vertices[trans_mask]
    trans_v = trans_tris.reshape(-1, 3) if len(trans_tris) > 0 else np.zeros((0, 3))

    print(f"  Triangles in transition zone: {len(trans_tris)}")
    if len(trans_tris) > 0:
        print(f"  Transition bounding box:")
        print(f"    X: [{trans_v[:,0].min():.6f}, {trans_v[:,0].max():.6f}]  range={trans_v[:,0].max()-trans_v[:,0].min():.6f}")
        print(f"    Y: [{trans_v[:,1].min():.6f}, {trans_v[:,1].max():.6f}]  range={trans_v[:,1].max()-trans_v[:,1].min():.6f}")
        print(f"    Z: [{trans_v[:,2].min():.6f}, {trans_v[:,2].max():.6f}]  range={trans_v[:,2].max()-trans_v[:,2].min():.6f}")

    # Also check negative Y transition (other blade)
    print(f"\n--- Transition Zone NEGATIVE (any vertex Y in [-0.12, -0.05]) ---")
    trans_neg_mask = np.any((vertices[:, :, 1] >= -0.12) & (vertices[:, :, 1] <= -0.05), axis=1)
    trans_neg_tris = vertices[trans_neg_mask]
    trans_neg_v = trans_neg_tris.reshape(-1, 3) if len(trans_neg_tris) > 0 else np.zeros((0, 3))
    print(f"  Triangles: {len(trans_neg_tris)}")
    if len(trans_neg_tris) > 0:
        print(f"  Bounding box:")
        print(f"    X: [{trans_neg_v[:,0].min():.6f}, {trans_neg_v[:,0].max():.6f}]")
        print(f"    Y: [{trans_neg_v[:,1].min():.6f}, {trans_neg_v[:,1].max():.6f}]")
        print(f"    Z: [{trans_neg_v[:,2].min():.6f}, {trans_neg_v[:,2].max():.6f}]")

    # ---- Cross sections at specific Y values ----
    print(f"\n--- Cross-Section Analysis (chord & thickness vs Y) ---")
    print(f"  {'Y':>8s} {'r/R':>6s} {'chord':>8s} {'thick':>8s} {'t/c':>6s} {'X_min':>9s} {'X_max':>9s} {'Z_min':>9s} {'Z_max':>9s} {'n_tri':>6s}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*6}")

    # Scan from Y=0 to Y=max in steps, to find where airfoil begins
    y_max_val = all_v[:, 1].max()
    y_min_val = all_v[:, 1].min()

    # Positive Y blade
    y_values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                0.11, 0.12, 0.13, 0.14, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50, 0.55, 0.60]

    for y_val in y_values:
        if y_val > y_max_val + 0.01:
            break
        cs = get_cross_section_at_y(vertices, y_val, tolerance=0.003)
        if cs is not None:
            rR = y_val / BLADE_RADIUS
            print(f"  {y_val:8.4f} {rR:6.3f} {cs['chord']:8.5f} {cs['thickness']:8.5f} {cs['thickness_to_chord']:6.3f} {cs['x_min']:9.5f} {cs['x_max']:9.5f} {cs['z_min']:9.5f} {cs['z_max']:9.5f} {cs['n_triangles']:6d}")
        else:
            print(f"  {y_val:8.4f} {y_val/BLADE_RADIUS:6.3f}  -- no triangles --")

    # Also check negative Y blade
    print(f"\n--- Cross-Section Analysis (NEGATIVE Y blade) ---")
    print(f"  {'Y':>8s} {'|r/R|':>6s} {'chord':>8s} {'thick':>8s} {'t/c':>6s} {'X_min':>9s} {'X_max':>9s} {'Z_min':>9s} {'Z_max':>9s} {'n_tri':>6s}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*6}")

    neg_y_values = [0.0, -0.01, -0.02, -0.03, -0.05, -0.08, -0.10, -0.12, -0.15, -0.20, -0.30, -0.40, -0.50, -0.55, -0.60]
    for y_val in neg_y_values:
        if y_val < y_min_val - 0.01:
            break
        cs = get_cross_section_at_y(vertices, y_val, tolerance=0.003)
        if cs is not None:
            rR = abs(y_val) / BLADE_RADIUS
            print(f"  {y_val:8.4f} {rR:6.3f} {cs['chord']:8.5f} {cs['thickness']:8.5f} {cs['thickness_to_chord']:6.3f} {cs['x_min']:9.5f} {cs['x_max']:9.5f} {cs['z_min']:9.5f} {cs['z_max']:9.5f} {cs['n_triangles']:6d}")
        else:
            print(f"  {y_val:8.4f} {abs(y_val)/BLADE_RADIUS:6.3f}  -- no triangles --")

    # ---- Determine where airfoil shape begins ----
    print(f"\n--- Detecting Hub-to-Airfoil Transition ---")
    # Scan finely through the transition region
    fine_y = np.arange(0.0, min(0.20, y_max_val), 0.005)
    chords = []
    thicknesses = []
    for y_val in fine_y:
        cs = get_cross_section_at_y(vertices, y_val, tolerance=0.003)
        if cs is not None:
            chords.append((y_val, cs['chord'], cs['thickness'], cs['thickness_to_chord']))

    if chords:
        print(f"  Fine scan (positive Y, step=0.005):")
        print(f"  {'Y':>7s} {'chord':>8s} {'thick':>8s} {'t/c':>6s}")
        for y_val, c, t, tc in chords:
            print(f"  {y_val:7.4f} {c:8.5f} {t:8.5f} {tc:6.3f}")

        # Find where chord starts changing significantly (hub is roughly constant cross-section)
        if len(chords) > 3:
            chord_vals = [c[1] for c in chords]
            for i in range(1, len(chord_vals)):
                if abs(chord_vals[i] - chord_vals[i-1]) / max(chord_vals[i-1], 1e-9) > 0.15:
                    y_transition = chords[i][0]
                    print(f"\n  >> Significant chord change detected at Y ~ {y_transition:.4f} (r/R = {y_transition/BLADE_RADIUS:.4f})")
                    print(f"     Chord went from {chord_vals[i-1]:.5f} to {chord_vals[i]:.5f}")
                    break

    # Also scan negative Y
    fine_y_neg = np.arange(0.0, max(-0.20, y_min_val), -0.005)
    chords_neg = []
    for y_val in fine_y_neg:
        cs = get_cross_section_at_y(vertices, y_val, tolerance=0.003)
        if cs is not None:
            chords_neg.append((y_val, cs['chord'], cs['thickness'], cs['thickness_to_chord']))

    if chords_neg:
        print(f"\n  Fine scan (negative Y, step=0.005):")
        print(f"  {'Y':>7s} {'chord':>8s} {'thick':>8s} {'t/c':>6s}")
        for y_val, c, t, tc in chords_neg:
            print(f"  {y_val:7.4f} {c:8.5f} {t:8.5f} {tc:6.3f}")

    print()
    return vertices


def main():
    print("\n" + "#" * 80)
    print("#  Ingenuity Mars Helicopter - Blade STL Hub Geometry Analysis")
    print(f"#  BLADE_RADIUS = {BLADE_RADIUS} m")
    print("#" * 80 + "\n")

    files = [
        ("E:/mujoco_projects/ingenuity-mujoco/assets/mhs_topblades_v16.stl", "TOP BLADES (mhs_topblades_v16.stl)"),
        ("E:/mujoco_projects/ingenuity-mujoco/assets/mhs_bottomblades_v16.stl", "BOTTOM BLADES (mhs_bottomblades_v16.stl)"),
    ]

    for filepath, label in files:
        analyze_stl(filepath, label)

    print("=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
