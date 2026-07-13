#!/usr/bin/env python
"""
Occupancy-grid voxelization of an existing views.npz (no GPU / re-inference needed).

Loads ~/MapAnything/outputs/<capture>/views.npz, reconstructs the merged world-frame
colored point cloud (all 3 views), applies the same optional pre-filters as
filter_export.py, then bins the points into a fixed-resolution sparse voxel grid:

  idx = floor((pts - origin) / voxel_size)

Per occupied voxel: point count, mean color, max confidence. Semantic fields
(label, label_score) are reserved as zeros for Task 2 (semantic lift) to fill in.

The numpy grid is cross-checked against Open3D's VoxelGrid built from the same
points with the same bounds (open3d is the reference implementation; the numpy
path is what downstream consumes). If open3d cannot be imported (e.g. headless
server without libGL), the cross-check is skipped with a warning — the outputs
are produced by the numpy path either way.

Portable: depends only on numpy, open3d, trimesh (no torch / mapanything / GPU),
so it runs on an Apple Silicon Mac or any Linux PC given a copied views.npz.

Outputs (next to the npz):
  voxels.npz — sparse grid: indices (N,3) int32, origin (3,), voxel_size, dims (3,),
               counts (N,), colors (N,3) uint8, conf (N,), labels (N,) int32,
               label_scores (N,)
  voxels.glb — one cube per occupied voxel, mean-color shaded, merged into a single
               mesh. Applies the same 180-deg X flip as predictions_to_glb so it
               overlays scene.glb / scene_filtered.glb in the same viewer pose.

Pre-filters (all optional, stackable, same semantics as filter_export.py):
  --max_radius <m>                       keep points within radius of world origin
  --min_conf <val>                       keep points with confidence >= val
  --bbox xmin xmax ymin ymax zmin zmax   workspace AABB: drops outside points AND
                                         fixes the grid origin/extent (default:
                                         tight bbox of the filtered points)

Example:
  python voxelize.py --captures g2_smoke_20260702_142817 --voxel_size 0.02 --max_radius 2.0
"""

import argparse
import json
import os

import numpy as np
import trimesh

try:
    import open3d as o3d
except (ImportError, OSError) as e:  # OSError: missing libGL on headless servers
    o3d = None
    _O3D_IMPORT_ERROR = e

OUT_ROOT = os.path.expanduser("~/MapAnything/outputs")

VIEW_NAMES = ["head", "hand_left", "hand_right"]

CAPTURES = [
    "g2_smoke_20260702_142817",
    "g2_smoke_20260702_144239",
    "g2_smoke_20260702_144354",
    "g2_smoke_20260702_144728",
]


def build_filter_mask(pts3d, conf, max_radius=None, bbox=None, min_conf=None):
    """Pointwise keep-mask over world-frame points; same semantics as
    filter_export.build_filter_mask, duplicated here so voxelize.py stays free
    of the torch/mapanything imports that module pulls in."""
    keep = np.ones(pts3d.shape[:-1], dtype=bool)
    if max_radius is not None:
        keep &= np.linalg.norm(pts3d, axis=-1) <= max_radius
    if bbox is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        keep &= (
            (pts3d[..., 0] >= xmin)
            & (pts3d[..., 0] <= xmax)
            & (pts3d[..., 1] >= ymin)
            & (pts3d[..., 1] <= ymax)
            & (pts3d[..., 2] >= zmin)
            & (pts3d[..., 2] <= zmax)
        )
    if min_conf is not None:
        if conf is None:
            raise ValueError("--min_conf requested but no confidence data available")
        keep &= conf >= min_conf
    return keep


def load_points(capture):
    """Merged filtered-input point cloud from views.npz.
    Returns (pts (N,3) f32, cols (N,3) f32 in [0,1], conf (N,) f32 or None)."""
    npz = np.load(os.path.join(OUT_ROOT, capture, "views.npz"))
    pts_list, col_list, conf_list = [], [], []
    have_conf = all(f"{n}_conf" in npz for n in VIEW_NAMES)
    for name in VIEW_NAMES:
        mask = npz[f"{name}_mask"].astype(bool)
        pts_list.append(npz[f"{name}_pts3d"][mask])
        col_list.append(npz[f"{name}_img"][mask].astype(np.float32) / 255.0)
        if have_conf:
            conf_list.append(npz[f"{name}_conf"][mask])
    pts = np.concatenate(pts_list, axis=0).astype(np.float32)
    cols = np.concatenate(col_list, axis=0)
    conf = np.concatenate(conf_list, axis=0) if have_conf else None
    return pts, cols, conf


def voxelize_points(pts, cols, conf, voxel_size, origin, dims):
    """Sparse occupancy grid by integer binning. pts must already lie inside
    [origin, origin + dims * voxel_size). Returns dict of per-voxel arrays,
    sorted by flat voxel index."""
    idx3 = np.floor((pts - origin) / voxel_size).astype(np.int64)
    np.clip(idx3, 0, np.asarray(dims) - 1, out=idx3)  # guard boundary rounding
    flat = np.ravel_multi_index((idx3[:, 0], idx3[:, 1], idx3[:, 2]), dims)

    uniq, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    n = uniq.shape[0]

    color_sum = np.zeros((n, 3), dtype=np.float64)
    for c in range(3):
        color_sum[:, c] = np.bincount(inverse, weights=cols[:, c], minlength=n)
    mean_colors = color_sum / counts[:, None]

    if conf is not None:
        conf_max = np.zeros(n, dtype=np.float32)
        np.maximum.at(conf_max, inverse, conf.astype(np.float32))
    else:
        conf_max = np.full(n, np.nan, dtype=np.float32)

    indices = np.stack(np.unravel_index(uniq, dims), axis=1).astype(np.int32)
    return {
        "indices": indices,
        "counts": counts.astype(np.int32),
        "colors": np.clip(mean_colors * 255.0, 0, 255).astype(np.uint8),
        "conf": conf_max,
    }


def crosscheck_open3d(pts, cols, voxel_size, origin, dims, np_indices):
    """Voxelize the same points with Open3D over identical bounds and compare
    the occupied-voxel index sets against the numpy grid."""
    if o3d is None:
        print(f"  open3d cross-check SKIPPED (import failed: {_O3D_IMPORT_ERROR})")
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    max_bound = origin + np.asarray(dims) * voxel_size
    vg = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd, voxel_size, origin.astype(np.float64), max_bound.astype(np.float64)
    )
    o3d_idx = np.array([v.grid_index for v in vg.get_voxels()], dtype=np.int64)
    o3d_set = set(map(tuple, o3d_idx))
    np_set = set(map(tuple, np_indices.astype(np.int64)))
    inter = len(np_set & o3d_set)
    union = max(len(np_set | o3d_set), 1)
    agree = inter / union
    print(
        f"  open3d cross-check: numpy {len(np_set)} vs open3d {len(o3d_set)} "
        f"occupied voxels, IoU {agree:.4f}"
    )
    # Boundary rounding can flip a handful of voxels between float64 (open3d)
    # and float32 (numpy) paths; anything beyond that is a real bug.
    assert agree > 0.999, f"numpy/open3d voxel grids disagree (IoU {agree})"
    return len(o3d_set)


def voxels_to_glb_mesh(indices, colors, voxel_size, origin):
    """Single merged mesh with one cube per occupied voxel, per-face mean colors.
    Cubes are shrunk 2% so adjacent voxels stay visually separable."""
    cube = trimesh.creation.box(extents=[voxel_size * 0.98] * 3)
    cv = np.asarray(cube.vertices)  # (8, 3), centered at 0
    cf = np.asarray(cube.faces)  # (12, 3)
    centers = origin + (indices.astype(np.float64) + 0.5) * voxel_size  # (N, 3)

    n = indices.shape[0]
    verts = (centers[:, None, :] + cv[None, :, :]).reshape(-1, 3)
    faces = (cf[None, :, :] + (np.arange(n) * cv.shape[0])[:, None, None]).reshape(-1, 3)
    face_colors = np.repeat(
        np.concatenate([colors, np.full((n, 1), 255, np.uint8)], axis=1),
        cf.shape[0],
        axis=0,
    )
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual.face_colors = face_colors
    # Same 180-deg X flip as predictions_to_glb, so voxels.glb overlays scene.glb.
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    return mesh


def process_capture(capture, voxel_size, max_radius=None, bbox=None, min_conf=None):
    out_dir = os.path.join(OUT_ROOT, capture)
    print(f"\n===== {capture} =====")

    pts, cols, conf = load_points(capture)
    n_raw = pts.shape[0]

    keep = build_filter_mask(pts, conf, max_radius=max_radius, bbox=bbox, min_conf=min_conf)
    pts, cols = pts[keep], cols[keep]
    conf = conf[keep] if conf is not None else None
    print(
        f"  pre-filter (max_radius={max_radius}, bbox={bbox}, min_conf={min_conf}): "
        f"{n_raw} -> {pts.shape[0]} points"
    )
    if pts.shape[0] == 0:
        raise RuntimeError("All points filtered out; nothing to voxelize")

    # Grid frame: --bbox fixes it, otherwise tight bounds of the filtered cloud.
    if bbox is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        origin = np.array([xmin, ymin, zmin], dtype=np.float32)
        extent = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=np.float32)
    else:
        origin = pts.min(axis=0)
        extent = pts.max(axis=0) - origin
    dims = tuple(
        int(v) for v in np.maximum(np.floor(extent / voxel_size).astype(np.int64) + 1, 1)
    )

    vox = voxelize_points(pts, cols, conf, voxel_size, origin, dims)
    n_vox = vox["indices"].shape[0]
    occ_pct = 100.0 * n_vox / float(np.prod(dims))
    print(
        f"  grid dims={dims} ({int(np.prod(dims))} cells) voxel_size={voxel_size} m | "
        f"occupied {n_vox} ({occ_pct:.2f}%) | pts/voxel min/med/max="
        f"{vox['counts'].min()}/{int(np.median(vox['counts']))}/{vox['counts'].max()}"
    )

    crosscheck_open3d(pts, cols, voxel_size, origin, dims, vox["indices"])

    # Sparse grid for downstream (robot / Task 2 semantic lift).
    npz_path = os.path.join(out_dir, "voxels.npz")
    np.savez_compressed(
        npz_path,
        indices=vox["indices"],
        origin=origin.astype(np.float32),
        voxel_size=np.float32(voxel_size),
        dims=np.asarray(dims, dtype=np.int32),
        counts=vox["counts"],
        colors=vox["colors"],
        conf=vox["conf"],
        # Reserved for Task 2 (semantic lift): 0 = background/unknown.
        labels=np.zeros(n_vox, dtype=np.int32),
        label_scores=np.zeros(n_vox, dtype=np.float32),
    )

    mesh = voxels_to_glb_mesh(vox["indices"], vox["colors"], voxel_size, origin)
    glb_path = os.path.join(out_dir, "voxels.glb")
    mesh.export(glb_path)

    print(
        f"  saved: {npz_path}, {glb_path} ({os.path.getsize(glb_path)} B, "
        f"{len(mesh.faces)} faces)"
    )
    return {
        "capture": capture,
        "points_in": int(pts.shape[0]),
        "voxel_size": voxel_size,
        "dims": [int(d) for d in dims],
        "occupied_voxels": int(n_vox),
        "occupancy_pct": round(occ_pct, 3),
        "origin": [round(float(v), 4) for v in origin],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sparse occupancy-grid voxelization from views.npz"
    )
    parser.add_argument("--captures", nargs="*", default=CAPTURES)
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.02,
        help="Voxel edge length in meters (0.02 manipulation-grade, 0.05 for viz)",
    )
    parser.add_argument(
        "--max_radius",
        type=float,
        default=None,
        help="Pre-filter: keep points within this distance (m) from world origin",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=6,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Workspace AABB: pre-filter points and fix the grid origin/extent",
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=None,
        help="Pre-filter: keep points with confidence >= this value",
    )
    args = parser.parse_args()

    results = []
    for capture in args.captures:
        results.append(
            process_capture(
                capture,
                voxel_size=args.voxel_size,
                max_radius=args.max_radius,
                bbox=args.bbox,
                min_conf=args.min_conf,
            )
        )
    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
