"""Analyze original STL files to extract hub geometry info."""
import struct
import numpy as np
from pathlib import Path

ASSETS = Path(__file__).parent.parent.parent / "assets"
BLADE_RADIUS = 0.606


def read_binary_stl(filepath):
    """Read binary STL, return (N,3,3) triangle vertices."""
    with open(filepath, 'rb') as f:
        f.read(80)  # header
        n_tri = struct.unpack('<I', f.read(4))[0]
        tris = []
        for _ in range(n_tri):
            struct.unpack('<3f', f.read(12))  # normal
            v0 = struct.unpack('<3f', f.read(12))
            v1 = struct.unpack('<3f', f.read(12))
            v2 = struct.unpack('<3f', f.read(12))
            struct.unpack('<H', f.read(2))  # attr
            tris.append([v0, v1, v2])
    return np.array(tris)


def analyze(name, filepath):
    print(f"\n{'='*60}")
    print(f"  {name}: {filepath.name}")
    print(f"{'='*60}")
    tris = read_binary_stl(str(filepath))
    verts = tris.reshape(-1, 3)
    print(f"  Triangles: {len(tris)}")
    print(f"  Bounding box:")
    for ax, label in enumerate(['X(chord)', 'Y(span)', 'Z(thick)']):
        print(f"    {label}: [{verts[:,ax].min():.4f}, {verts[:,ax].max():.4f}]")

    # Hub region: |Y| < 0.05
    hub_mask = np.all(np.abs(tris[:,:,1]) < 0.05, axis=1)
    hub_tris = tris[hub_mask]
    print(f"\n  Hub region (|Y|<0.05): {len(hub_tris)} triangles")
    if len(hub_tris) > 0:
        hv = hub_tris.reshape(-1, 3)
        for ax, label in enumerate(['X', 'Y', 'Z']):
            print(f"    {label}: [{hv[:,ax].min():.4f}, {hv[:,ax].max():.4f}]")

    # Cross-sections at various Y values
    print(f"\n  Cross-section analysis (vertex Y ranges):")
    for y_cut in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        dy = 0.005
        mask = np.any((tris[:,:,1] > y_cut - dy) & (tris[:,:,1] < y_cut + dy), axis=1)
        cut_tris = tris[mask]
        if len(cut_tris) > 0:
            cv = cut_tris.reshape(-1, 3)
            y_sel = (cv[:,1] > y_cut - dy) & (cv[:,1] < y_cut + dy)
            pts = cv[y_sel]
            r_over_R = y_cut / BLADE_RADIUS
            print(f"    Y={y_cut:.3f} (r/R={r_over_R:.3f}): "
                  f"X=[{pts[:,0].min():.4f},{pts[:,0].max():.4f}] "
                  f"Z=[{pts[:,2].min():.4f},{pts[:,2].max():.4f}] "
                  f"chord~{pts[:,0].max()-pts[:,0].min():.4f} "
                  f"thick~{pts[:,2].max()-pts[:,2].min():.4f}")
        else:
            print(f"    Y={y_cut:.3f}: no triangles")

    # Find where hub structure ends (Y threshold where X-extent changes)
    print(f"\n  Radial extent per Y-slice (hub vs blade transition):")
    for y_cut in np.arange(0.01, 0.12, 0.01):
        dy = 0.003
        mask = np.any((tris[:,:,1] > y_cut - dy) & (tris[:,:,1] < y_cut + dy), axis=1)
        cut_tris = tris[mask]
        if len(cut_tris) > 0:
            cv = cut_tris.reshape(-1, 3)
            y_sel = (cv[:,1] > y_cut - dy) & (cv[:,1] < y_cut + dy)
            pts = cv[y_sel]
            x_range = pts[:,0].max() - pts[:,0].min()
            z_range = pts[:,2].max() - pts[:,2].min()
            print(f"    Y={y_cut:.3f}: X_range={x_range:.4f}  Z_range={z_range:.4f}  "
                  f"n_verts={len(pts)}")

    return tris


if __name__ == "__main__":
    top = ASSETS / "mhs_topblades_v16.stl"
    bot = ASSETS / "mhs_bottomblades_v16.stl"

    if top.exists():
        analyze("TOP BLADES", top)
    else:
        print(f"NOT FOUND: {top}")

    if bot.exists():
        analyze("BOTTOM BLADES", bot)
    else:
        print(f"NOT FOUND: {bot}")
