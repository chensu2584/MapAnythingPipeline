#!/usr/bin/env python
"""
Filtered point-cloud export from an existing views.npz (no GPU / re-inference needed).

Loads ~/MapAnything/outputs/<capture>/views.npz, reconstructs the world-frame colored
point cloud, applies stackable filters, and writes scene_filtered.glb / scene_filtered.ply
next to the npz.

Geometry source (in order of preference):
  1. <view>_pts3d stored in the npz (written by run_inference.py).
  2. Unprojection of <view>_depth_z with <view>_intrinsics + <view>_camera_pose
     (uses the same mapanything.utils.geometry.depthmap_to_world_frame as inference).
Either way the unprojection is verified against the stored data / original scene.ply.

Color source (in order of preference):
  1. <view>_img stored in the npz (model's img_no_norm, exact).
  2. Undistorted PNGs from ~/MapAnything/outputs/undistorted/<capture>/, center-cropped
     to the depth-map aspect ratio and resized (approximate fallback).

Filters (all optional, stackable):
  --max_radius <m>                       keep points within radius of world origin
  --bbox xmin xmax ymin ymax zmin zmax   keep points inside world-frame box
  --min_conf <val>                       keep points with confidence >= val

Options:
  --show_cameras     add a colored frustum marker per camera to the GLB
                     (head=red, hand_left=green, hand_right=blue; GLB only,
                     the PLY stays a pure point cloud)

Example:
  python filter_export.py --captures g2_smoke_20260702_142817 --max_radius 2.0 --show_cameras
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.viz import predictions_to_glb

from capture_contract import VIEW_NAMES, resolve_reconstruction_captures

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = os.path.expanduser(
    os.environ.get("G2_OUT_ROOT", str(PROJECT_ROOT / "outputs"))
)



CAM_MARKER_COLORS = {
    "head": [230, 40, 40, 255],  # red
    "hand_left": [40, 200, 40, 255],  # green
    "hand_right": [40, 90, 230, 255],  # blue
}

VIEW_DEBUG_COLORS = {
    "head": np.array([230, 40, 40], dtype=np.float32) / 255.0,
    "hand_left": np.array([40, 200, 40], dtype=np.float32) / 255.0,
    "hand_right": np.array([40, 90, 230], dtype=np.float32) / 255.0,
}


def camera_frustum_mesh(K, pose, img_hw, color, frustum_depth=0.15):
    """Solid frustum pyramid for one camera: apex at the camera center, base at
    the image-plane corners unprojected to `frustum_depth` meters, transformed
    to world frame by the cam2world pose."""
    H, W = img_hw
    corners_px = np.array(
        [[0.0, 0.0, 1.0], [W, 0.0, 1.0], [W, H, 1.0], [0.0, H, 1.0]]
    )
    cam_pts = (corners_px @ np.linalg.inv(K).T) * frustum_depth
    verts_cam = np.vstack([np.zeros(3), cam_pts])
    verts_w = verts_cam @ pose[:3, :3].T + pose[:3, 3]
    faces = np.array(
        [[0, 2, 1], [0, 3, 2], [0, 4, 3], [0, 1, 4], [1, 2, 3], [1, 3, 4]]
    )
    mesh = trimesh.Trimesh(vertices=verts_w, faces=faces, process=False)
    mesh.visual.face_colors = np.tile(np.array(color, dtype=np.uint8), (len(faces), 1))
    return mesh


def build_filter_mask(pts3d, conf, max_radius=None, bbox=None, min_conf=None):
    """Pointwise keep-mask over world-frame points. pts3d: (..., 3); conf: same
    leading shape or None. Filters are ANDed; None filters are skipped."""
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


def unproject_view(npz, name):
    """World-frame points from depth + intrinsics + cam2world pose (same util as
    inference). Returns (pts3d (H,W,3) float32, valid (H,W) bool)."""
    depth = torch.from_numpy(npz[f"{name}_depth_z"].astype(np.float32))
    K = torch.from_numpy(npz[f"{name}_intrinsics"].astype(np.float32))
    pose = torch.from_numpy(npz[f"{name}_camera_pose"].astype(np.float32))
    pts3d, valid = depthmap_to_world_frame(depth, K, pose)
    return pts3d.numpy(), valid.numpy()


def fallback_colors(capture, name, target_hw, undist_root):
    """Approximate colors: undistorted PNG center-cropped to the depth aspect ratio
    then resized. Returns (H, W, 3) float in [0, 1]."""
    path = os.path.join(undist_root, capture, f"{name}.png")
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read fallback image: {path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    th, tw = target_hw
    H, W = img.shape[:2]
    target_ar = tw / th
    if W / H > target_ar:  # too wide -> crop width
        new_w = int(round(H * target_ar))
        x0 = (W - new_w) // 2
        img = img[:, x0 : x0 + new_w]
    else:  # too tall -> crop height
        new_h = int(round(W / target_ar))
        y0 = (H - new_h) // 2
        img = img[y0 : y0 + new_h, :]
    img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def process_capture(
    capture,
    max_radius=None,
    bbox=None,
    min_conf=None,
    show_cameras=False,
    color_by_view=False,
    frustum_depth=0.30,
    output_root=DEFAULT_OUTPUT_ROOT,
):
    out_dir = os.path.join(output_root, capture)
    undist_root = os.path.join(output_root, "undistorted")
    npz_path = os.path.join(out_dir, "views.npz")
    npz = np.load(npz_path)
    print(f"\n===== {capture} =====")

    pts_list, img_list, mask_list, conf_list = [], [], [], []
    have_conf = all(f"{n}_conf" in npz for n in VIEW_NAMES)

    for name in VIEW_NAMES:
        mask = npz[f"{name}_mask"].astype(bool)

        # Geometry: unproject from depth, verify against stored pts3d if present.
        # Note: inference computed pts3d on GPU (TF32/bf16 matmul path), so CPU
        # float32 unprojection can differ by a few mm; 1 cm tolerance covers that.
        pts3d, _valid = unproject_view(npz, name)
        if f"{name}_pts3d" in npz:
            stored = npz[f"{name}_pts3d"]
            diff = float(np.abs(pts3d[mask] - stored[mask]).max()) if mask.any() else 0.0
            assert diff < 1e-2, (
                f"{name}: unprojection deviates from stored pts3d by {diff} m"
            )
            print(f"  [{name}] unprojection vs stored pts3d: max dev {diff:.2e} m (ok)")
            pts3d = stored  # use stored values verbatim (exactly match scene.ply)

        # Colors: exact stored img preferred, else resampled undistorted PNG.
        if f"{name}_img" in npz:
            img = npz[f"{name}_img"].astype(np.float32) / 255.0
        else:
            print(f"  [{name}] no stored img in npz; resampling undistorted PNG "
                  f"(colors approximate, geometry exact)")
            img = fallback_colors(capture, name, mask.shape, undist_root)

        pts_list.append(pts3d.astype(np.float32))
        img_list.append(img)
        mask_list.append(mask)
        if have_conf:
            conf_list.append(npz[f"{name}_conf"])

    world_points = np.stack(pts_list, axis=0)  # (V, H, W, 3)
    images = np.stack(img_list, axis=0)  # (V, H, W, 3) in [0, 1]
    masks = np.stack(mask_list, axis=0)  # (V, H, W)
    conf = np.stack(conf_list, axis=0) if have_conf else None

    # --- Verify reconstruction against the original scene.ply -------------------
    # views.npz deliberately keeps unfiltered masks/geometry, whereas scene.ply
    # may have been exported with filters during inference. Reapply those saved
    # filters before comparing point count/order.
    orig_ply_path = os.path.join(out_dir, "scene.ply")
    n_unfiltered = int(masks.sum())
    if os.path.exists(orig_ply_path):
        verification_masks = masks
        summary_path = os.path.join(out_dir, "summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path, encoding="utf-8") as f:
                original_summary = json.load(f)
            original_filter = original_summary.get("export_filter")
            if original_filter:
                verification_masks = masks & build_filter_mask(
                    world_points,
                    conf,
                    max_radius=original_filter.get("max_radius"),
                    bbox=original_filter.get("bbox"),
                    min_conf=original_filter.get("min_conf"),
                )
        orig = trimesh.load(orig_ply_path, process=False)
        orig_v = np.asarray(orig.vertices, dtype=np.float32)
        recon_v = world_points.reshape(-1, 3)[verification_masks.reshape(-1)]
        assert orig_v.shape[0] == recon_v.shape[0], (
            f"point count mismatch vs scene.ply after replaying saved export filter: "
            f"{orig_v.shape[0]} vs {recon_v.shape[0]}"
        )
        max_dev = float(np.abs(recon_v - orig_v).max())
        assert max_dev < 1e-3, f"coordinates deviate from scene.ply by {max_dev}"
        print(
            f"  verification vs scene.ply: {recon_v.shape[0]} points match, "
            f"max coordinate deviation {max_dev:.2e} m"
        )
    else:
        print("  scene.ply not found; skipping cross-check")

    # --- Apply filters -----------------------------------------------------------
    keep = build_filter_mask(
        world_points, conf, max_radius=max_radius, bbox=bbox, min_conf=min_conf
    )
    final_masks = masks & keep
    n_filtered = int(final_masks.sum())
    pct = 100.0 * n_filtered / max(n_unfiltered, 1)
    print(
        f"  filters (max_radius={max_radius}, bbox={bbox}, min_conf={min_conf}): "
        f"{n_unfiltered} -> {n_filtered} points ({pct:.1f}% kept)"
    )

    flat_keep = final_masks.reshape(-1)
    pts = world_points.reshape(-1, 3)[flat_keep]
    export_images = images
    if color_by_view:
        export_images = np.stack(
            [
                np.broadcast_to(VIEW_DEBUG_COLORS[name], images[view_idx].shape)
                for view_idx, name in enumerate(VIEW_NAMES)
            ],
            axis=0,
        )
    cols = np.clip(export_images.reshape(-1, 3)[flat_keep] * 255.0, 0, 255).astype(
        np.uint8
    )

    if pts.shape[0] == 0:
        raise RuntimeError("All points filtered out; nothing to export")

    bb_min, bb_max = pts.min(axis=0), pts.max(axis=0)
    ext = bb_max - bb_min
    print(
        f"  filtered bbox min={np.round(bb_min, 3)} max={np.round(bb_max, 3)} "
        f"extents={np.round(ext, 3)} m"
    )

    # --- Export ------------------------------------------------------------------
    # GLB via the repo utility (same 180-deg X flip as scene.glb).
    scene_glb = predictions_to_glb(
        {
            "world_points": world_points,
            "images": export_images,
            "final_masks": final_masks,
        },
        as_mesh=False,
    )
    if show_cameras:
        # predictions_to_glb already flipped the scene 180 deg around X; new
        # geometry added afterwards must carry the same transform itself.
        flip_x = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        for name in VIEW_NAMES:
            frustum = camera_frustum_mesh(
                npz[f"{name}_intrinsics"].astype(np.float64),
                npz[f"{name}_camera_pose"].astype(np.float64),
                npz[f"{name}_mask"].shape,
                CAM_MARKER_COLORS[name],
                frustum_depth=frustum_depth,
            )
            frustum.apply_transform(flip_x)
            scene_glb.add_geometry(frustum, geom_name=f"camera_{name}")
        print(f"  added camera markers: {', '.join(VIEW_NAMES)}")
    suffix = "_by_view" if color_by_view else ""
    glb_path = os.path.join(out_dir, f"scene_filtered{suffix}.glb")
    scene_glb.export(glb_path)

    # PLY in raw world coordinates (same convention as scene.ply).
    ply_path = os.path.join(out_dir, f"scene_filtered{suffix}.ply")
    trimesh.PointCloud(vertices=pts, colors=cols).export(ply_path)

    print(
        f"  saved: {glb_path} ({os.path.getsize(glb_path)} B), "
        f"{ply_path} ({os.path.getsize(ply_path)} B)"
    )
    return {
        "capture": capture,
        "points_before": n_unfiltered,
        "points_after": n_filtered,
        "pct_kept": round(pct, 2),
        "bbox_min": [round(float(v), 4) for v in bb_min],
        "bbox_max": [round(float(v), 4) for v in bb_max],
        "bbox_extents": [round(float(v), 4) for v in ext],
        "color_by_view": color_by_view,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filtered GLB/PLY export from views.npz"
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--captures",
        nargs="*",
        default=None,
        help="Capture folder names; omit to auto-discover folders with views.npz",
    )
    parser.add_argument(
        "--max_radius",
        type=float,
        default=None,
        help="Keep points within this distance (m) from world origin (head camera)",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=6,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Keep points inside this world-frame box",
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=None,
        help="Keep points with confidence >= this value",
    )
    parser.add_argument(
        "--show_cameras",
        action="store_true",
        help="Add colored camera frustum markers to the exported GLB",
    )
    parser.add_argument(
        "--color_by_view",
        action="store_true",
        help="Color head/left/right points red/green/blue for rig diagnostics",
    )
    parser.add_argument(
        "--frustum_depth",
        type=float,
        default=0.30,
        help="Camera frustum length in meters for --show_cameras",
    )
    args = parser.parse_args()
    captures = resolve_reconstruction_captures(args.output_root, args.captures)

    results = []
    for capture in captures:
        results.append(
            process_capture(
                capture,
                max_radius=args.max_radius,
                bbox=args.bbox,
                min_conf=args.min_conf,
                show_cameras=args.show_cameras,
                color_by_view=args.color_by_view,
                frustum_depth=args.frustum_depth,
                output_root=args.output_root,
            )
        )
    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
