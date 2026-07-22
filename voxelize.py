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
from pathlib import Path

import numpy as np
import trimesh

from capture_contract import VIEW_NAMES, resolve_reconstruction_captures
from gripper_pose import (
    DEFAULT_G1_URDF,
    DEFAULT_GRIPPER_URDF,
    resolve_gripper_poses,
    write_gripper_poses,
)

try:
    import open3d as o3d
except (ImportError, OSError) as e:  # OSError: missing libGL on headless servers
    o3d = None
    _O3D_IMPORT_ERROR = e

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = os.path.expanduser(
    os.environ.get("G2_OUT_ROOT", str(PROJECT_ROOT / "outputs"))
)

GRIPPER_MARKER_COLORS = {
    "left": [255, 160, 0, 255],
    "right": [0, 220, 255, 255],
}


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


def read_export_world_frame(out_dir):
    """Read the frame the reconstruction was exported in, or say it is unknown.

    Guessing here would silently mislabel a grid that downstream avoidance code
    trusts, so an unreadable or absent declaration stays "unknown".
    """
    path = os.path.join(out_dir, "camera_poses_used_for_export.json")
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        return "unknown"
    frame = document.get("world_frame")
    return frame if isinstance(frame, str) and frame else "unknown"


def load_points(capture, output_root=DEFAULT_OUTPUT_ROOT):
    """Merged filtered-input point cloud from views.npz.
    Returns (pts (N,3) f32, cols (N,3) f32 in [0,1], conf (N,) f32 or None)."""
    npz = np.load(os.path.join(output_root, capture, "views.npz"))
    pts_list, col_list, conf_list, view_list = [], [], [], []
    have_conf = all(f"{n}_conf" in npz for n in VIEW_NAMES)
    for index, name in enumerate(VIEW_NAMES):
        mask = npz[f"{name}_mask"].astype(bool)
        pts_list.append(npz[f"{name}_pts3d"][mask])
        col_list.append(npz[f"{name}_img"][mask].astype(np.float32) / 255.0)
        # Remember which camera saw each point.  Which views agree on a voxel is
        # a strong self-filter cue: gripper surfaces are typically seen only by
        # the wrist camera on the same arm, while real scene geometry near the
        # hand is also seen from the head.
        view_list.append(np.full(int(mask.sum()), 1 << index, dtype=np.uint8))
        if have_conf:
            conf_list.append(npz[f"{name}_conf"][mask])
    pts = np.concatenate(pts_list, axis=0).astype(np.float32)
    cols = np.concatenate(col_list, axis=0)
    conf = np.concatenate(conf_list, axis=0) if have_conf else None
    view_bits = np.concatenate(view_list, axis=0)
    return pts, cols, conf, view_bits


def voxelize_points(pts, cols, conf, voxel_size, origin, dims, view_bits=None):
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
    if view_bits is not None:
        source_views = np.zeros(n, dtype=np.uint8)
        np.bitwise_or.at(source_views, inverse, view_bits.astype(np.uint8))
    else:
        source_views = np.zeros(n, dtype=np.uint8)
    return {
        "indices": indices,
        "counts": counts.astype(np.int32),
        "colors": np.clip(mean_colors * 255.0, 0, 255).astype(np.uint8),
        "conf": conf_max,
        "source_views": source_views,
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


def voxel_scene_with_grippers(voxel_mesh, gripper_poses):
    """Add tool-center spheres/frames using the same GLB X flip as voxels."""
    scene = trimesh.Scene()
    scene.add_geometry(voxel_mesh, geom_name="occupied_voxels")
    flip_x = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    for side in ("left", "right"):
        pose = np.asarray(
            gripper_poses["poses"][side]["pose_matrix"], dtype=np.float64
        )
        center = trimesh.creation.icosphere(subdivisions=2, radius=0.018)
        center.apply_translation(pose[:3, 3])
        center.visual.face_colors = np.tile(
            np.asarray(GRIPPER_MARKER_COLORS[side], dtype=np.uint8),
            (len(center.faces), 1),
        )
        frame = trimesh.creation.axis(
            origin_size=0.008,
            axis_radius=0.08 / 30.0,
            axis_length=0.08,
        )
        frame.apply_transform(pose)
        center.apply_transform(flip_x)
        frame.apply_transform(flip_x)
        scene.add_geometry(center, geom_name=f"gripper_{side}_center")
        scene.add_geometry(frame, geom_name=f"gripper_{side}_frame")
    return scene


def process_capture(
    capture,
    voxel_size,
    max_radius=None,
    bbox=None,
    min_conf=None,
    show_grippers=False,
    g1_urdf=DEFAULT_G1_URDF,
    gripper_urdf=DEFAULT_GRIPPER_URDF,
    output_root=DEFAULT_OUTPUT_ROOT,
):
    out_dir = os.path.join(output_root, capture)
    print(f"\n===== {capture} =====")

    pts, cols, conf, view_bits = load_points(capture, output_root)
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

    vox = voxelize_points(pts, cols, conf, voxel_size, origin, dims, view_bits)
    n_vox = vox["indices"].shape[0]
    occ_pct = 100.0 * n_vox / float(np.prod(dims))
    print(
        f"  grid dims={dims} ({int(np.prod(dims))} cells) voxel_size={voxel_size} m | "
        f"occupied {n_vox} ({occ_pct:.2f}%) | pts/voxel min/med/max="
        f"{vox['counts'].min()}/{int(np.median(vox['counts']))}/{vox['counts'].max()}"
    )

    crosscheck_open3d(pts, cols, voxel_size, origin, dims, vox["indices"])

    world_frame = read_export_world_frame(out_dir)

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
        # Bit i is set when VIEW_NAMES[i] contributed a point to this voxel.
        source_views=vox["source_views"],
        source_view_names=np.asarray(list(VIEW_NAMES)),
        # Declare the frame in the file itself so consumers never have to infer
        # it from a sibling document.
        world_frame=np.asarray(world_frame),
        translation_unit=np.asarray("meter"),
        # Reserved for Task 2 (semantic lift): 0 = background/unknown.
        labels=np.zeros(n_vox, dtype=np.int32),
        label_scores=np.zeros(n_vox, dtype=np.float32),
    )

    mesh = voxels_to_glb_mesh(vox["indices"], vox["colors"], voxel_size, origin)
    glb_path = os.path.join(out_dir, "voxels.glb")
    gripper_poses = None
    gripper_pose_path = None
    export_geometry = mesh
    if show_grippers:
        # The overlay resolves G1 link names against the G1 URDF and needs the
        # G1-only pose_conversion_manifest.json.  Check for it up front so a G2
        # capture gets told why rather than a bare missing-file error.
        if not os.path.isfile(os.path.join(out_dir, "pose_conversion_manifest.json")):
            raise ValueError(
                "--show_grippers is G1-specific: it needs pose_conversion_manifest.json, "
                f"which {capture} does not have. Omit the flag for non-G1 captures."
            )
        gripper_poses = resolve_gripper_poses(
            out_dir,
            g1_urdf=g1_urdf,
            gripper_urdf=gripper_urdf,
        )
        gripper_pose_path = os.path.join(out_dir, "gripper_poses_base_link.json")
        write_gripper_poses(gripper_pose_path, gripper_poses)
        export_geometry = voxel_scene_with_grippers(mesh, gripper_poses)
    export_geometry.export(glb_path)

    print(
        f"  saved: {npz_path}, {glb_path} ({os.path.getsize(glb_path)} B, "
        f"{len(mesh.faces)} faces)"
    )
    if gripper_poses is not None:
        print(
            "  added gripper centers to voxels.glb: "
            f"left={np.round(gripper_poses['poses']['left']['position_m'], 6).tolist()}, "
            f"right={np.round(gripper_poses['poses']['right']['position_m'], 6).tolist()}"
        )
    return {
        "capture": capture,
        "points_in": int(pts.shape[0]),
        "voxel_size": voxel_size,
        "dims": [int(d) for d in dims],
        "occupied_voxels": int(n_vox),
        "occupancy_pct": round(occ_pct, 3),
        "origin": [round(float(v), 4) for v in origin],
        "show_grippers": show_grippers,
        "gripper_pose_json": gripper_pose_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sparse occupancy-grid voxelization from views.npz"
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--captures",
        nargs="*",
        default=None,
        help="Capture folder names; omit to auto-discover folders with views.npz",
    )
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
    parser.add_argument(
        "--show_grippers",
        action="store_true",
        help="Add resolved G1 left/right gripper-center markers to voxels.glb",
    )
    parser.add_argument("--g1_urdf", default=str(DEFAULT_G1_URDF))
    parser.add_argument("--gripper_urdf", default=str(DEFAULT_GRIPPER_URDF))
    args = parser.parse_args()
    captures = resolve_reconstruction_captures(args.output_root, args.captures)

    results = []
    for capture in captures:
        results.append(
            process_capture(
                capture,
                voxel_size=args.voxel_size,
                max_radius=args.max_radius,
                bbox=args.bbox,
                min_conf=args.min_conf,
                show_grippers=args.show_grippers,
                g1_urdf=args.g1_urdf,
                gripper_urdf=args.gripper_urdf,
                output_root=args.output_root,
            )
        )
    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
