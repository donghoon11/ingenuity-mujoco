"""
Analyze the geometry of Ingenuity blade STL files.
Reads binary STL using struct, computes bounding box, centroid,
root/hub vertices, tip vertices, and radial extents.
"""

import struct
import math
import os

def read_binary_stl(filepath):
    vertices = []
    triangles = []
    with open(filepath, 'rb') as f:
        header = f.read(80)
        num_triangles = struct.unpack('<I', f.read(4))[0]
        for _ in range(num_triangles):
            nx, ny, nz = struct.unpack('<3f', f.read(12))
            v1 = struct.unpack('<3f', f.read(12))
            v2 = struct.unpack('<3f', f.read(12))
            v3 = struct.unpack('<3f', f.read(12))
            attr = struct.unpack('<H', f.read(2))[0]
            triangles.append((v1, v2, v3))
            vertices.extend([v1, v2, v3])
    return header, num_triangles, triangles, vertices

def analyze_stl(filepath):
    name = os.path.basename(filepath)
    print("=" * 70)
    print(f"  FILE: {name}")
    print(f"  PATH: {filepath}")
    print("=" * 70)

    header, num_tri, triangles, vertices = read_binary_stl(filepath)

    try:
        header_str = header.decode('ascii', errors='replace').strip('\x00').strip()
    except:
        header_str = "(binary)"
    print(f"\n  Header : {header_str[:60]}")
    print(f"  Triangles: {num_tri}")
    print(f"  Vertices (with duplicates): {len(vertices)}")

    unique_verts = list(set(vertices))
    print(f"  Unique vertices: {len(unique_verts)}")

    xs = [v[0] for v in unique_verts]
    ys = [v[1] for v in unique_verts]
    zs = [v[2] for v in unique_verts]

    # 1. Bounding box
    print(f"\n  --- 1. BOUNDING BOX ---")
    print(f"  X: min={min(xs):.4f}  max={max(xs):.4f}  span={max(xs)-min(xs):.4f}")
    print(f"  Y: min={min(ys):.4f}  max={max(ys):.4f}  span={max(ys)-min(ys):.4f}")
    print(f"  Z: min={min(zs):.4f}  max={max(zs):.4f}  span={max(zs)-min(zs):.4f}")

    # 2. Centroid
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    cz = sum(zs) / len(zs)
    print(f"\n  --- 2. CENTROID (mean of unique vertices) ---")
    print(f"  ({cx:.4f}, {cy:.4f}, {cz:.4f})")

    # 3. Vertices near y=0
    y_thresh = 0.01
    root_verts = [v for v in unique_verts if abs(v[1]) < y_thresh]
    print(f"\n  --- 3. VERTICES NEAR y=0 (|y| < {y_thresh}) --- root/hub ---")
    print(f"  Count: {len(root_verts)}")
    if root_verts:
        rxs = [v[0] for v in root_verts]
        rys = [v[1] for v in root_verts]
        rzs = [v[2] for v in root_verts]
        print(f"  X range: [{min(rxs):.4f}, {max(rxs):.4f}]")
        print(f"  Y range: [{min(rys):.4f}, {max(rys):.4f}]")
        print(f"  Z range: [{min(rzs):.4f}, {max(rzs):.4f}]")
        root_cx = sum(rxs) / len(rxs)
        root_cy = sum(rys) / len(rys)
        root_cz = sum(rzs) / len(rzs)
        print(f"  Root centroid: ({root_cx:.4f}, {root_cy:.4f}, {root_cz:.4f})")
        print(f"  Sample vertices (up to 10):")
        for v in sorted(root_verts, key=lambda v: v[0])[:10]:
            r_xy = math.sqrt(v[0]**2 + v[1]**2)
            print(f"    ({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})  r_xy={r_xy:.4f}")
    else:
        for wider in [0.02, 0.05, 0.1]:
            root_verts_w = [v for v in unique_verts if abs(v[1]) < wider]
            if root_verts_w:
                print(f"  (No verts within {y_thresh}, but {len(root_verts_w)} within |y|<{wider})")
                rxs = [v[0] for v in root_verts_w]
                rys = [v[1] for v in root_verts_w]
                rzs = [v[2] for v in root_verts_w]
                print(f"  X range: [{min(rxs):.4f}, {max(rxs):.4f}]")
                print(f"  Y range: [{min(rys):.4f}, {max(rys):.4f}]")
                print(f"  Z range: [{min(rzs):.4f}, {max(rzs):.4f}]")
                break

    # 4. Vertices at maximum |y|
    max_abs_y = max(abs(v[1]) for v in unique_verts)
    tip_thresh = 0.005
    tip_verts = [v for v in unique_verts if abs(abs(v[1]) - max_abs_y) < tip_thresh]
    print(f"\n  --- 4. VERTICES AT MAX |y| --- blade tip ---")
    print(f"  Max |y| = {max_abs_y:.4f}")
    print(f"  Tip vertices (within {tip_thresh} of max |y|): {len(tip_verts)}")
    if tip_verts:
        txs = [v[0] for v in tip_verts]
        tys = [v[1] for v in tip_verts]
        tzs = [v[2] for v in tip_verts]
        print(f"  X range: [{min(txs):.4f}, {max(txs):.4f}]")
        print(f"  Y range: [{min(tys):.4f}, {max(tys):.4f}]")
        print(f"  Z range: [{min(tzs):.4f}, {max(tzs):.4f}]")
        print(f"  Sample tip vertices (up to 10):")
        for v in sorted(tip_verts, key=lambda v: abs(v[1]), reverse=True)[:10]:
            r_xy = math.sqrt(v[0]**2 + v[1]**2)
            print(f"    ({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})  r_xy={r_xy:.4f}")

    min_y = min(ys)
    max_y = max(ys)
    if abs(min_y) > 0.01 and abs(max_y) > 0.01:
        print(f"\n  Blades extend in both +Y and -Y directions:")
        print(f"    +Y tip at y={max_y:.4f}")
        print(f"    -Y tip at y={min_y:.4f}")

    # 5. Radial extents
    radii = [math.sqrt(v[0]**2 + v[1]**2) for v in unique_verts]
    print(f"\n  --- 5. RADIAL EXTENTS (XY plane from origin) ---")
    print(f"  Min radius: {min(radii):.4f}")
    print(f"  Max radius: {max(radii):.4f}")
    print(f"  Mean radius: {sum(radii)/len(radii):.4f}")

    min_r_idx = radii.index(min(radii))
    max_r_idx = radii.index(max(radii))
    vmin = unique_verts[min_r_idx]
    vmax = unique_verts[max_r_idx]
    print(f"  Vertex at min radius: ({vmin[0]:.4f}, {vmin[1]:.4f}, {vmin[2]:.4f})")
    print(f"  Vertex at max radius: ({vmax[0]:.4f}, {vmax[1]:.4f}, {vmax[2]:.4f})")

    print(f"\n  --- Radial distribution (10 bins) ---")
    r_min, r_max = min(radii), max(radii)
    n_bins = 10
    bin_width = (r_max - r_min) / n_bins
    for i in range(n_bins):
        lo = r_min + i * bin_width
        hi = lo + bin_width
        count = sum(1 for r in radii if lo <= r < hi)
        bar = "#" * min(count // 2, 40)
        print(f"    [{lo:.4f}, {hi:.4f}): {count:5d}  {bar}")

    print()
    return unique_verts

if __name__ == "__main__":
    assets_dir = r"E:\mujoco_projects\ingenuity-mujoco\assets"
    files = [
        os.path.join(assets_dir, "mhs_topblades_v16.stl"),
        os.path.join(assets_dir, "mhs_bottomblades_v16.stl"),
    ]
    all_data = {}
    for f in files:
        if os.path.exists(f):
            verts = analyze_stl(f)
            all_data[os.path.basename(f)] = verts
        else:
            print(f"  FILE NOT FOUND: {f}")

    if len(all_data) == 2:
        print("=" * 70)
        print("  COMPARISON SUMMARY")
        print("=" * 70)
        for name, verts in all_data.items():
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            zs = [v[2] for v in verts]
            radii = [math.sqrt(v[0]**2 + v[1]**2) for v in verts]
            span_y = max(ys) - min(ys)
            print(f"\n  {name}:")
            print(f"    Y span (blade diameter): {span_y:.4f}")
            print(f"    Z range (thickness):     [{min(zs):.4f}, {max(zs):.4f}]")
            print(f"    Max XY radius:           {max(radii):.4f}")
            print(f"    Unique vertices:         {len(verts)}")
