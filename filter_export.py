#!/usr/bin/env python
"""
Filtered point-cloud export from an existing views.npz (no GPU / re-inference needed).

Loads ~/MapAnything/outputs/<capture>/views.npz, reconstructs the world-frame colored
point cloud, applies stackable filters, and writes scene_filtered.glb / scene_filtered.ply
next to the npz. With ``--color_by_view`` it additionally writes
scene_filtered_by_view.glb without replacing the normal RGB-colored export.

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
  --show_cameras     add a small colored frustum + center marker per camera and
                     a world-origin XYZ frame to the GLB (head=red,
                     hand_left=green, hand_right=blue; GLB only, the PLY stays
                     a pure point cloud)
  --per_camera_k_ab  additionally export experimental A/B GLBs using calibrated
                     K for head, model K for left, and model focal lengths with
                     calibrated principal point for right
  --show_grippers    resolve the captured G1 left/right gripper centers from the
                     robot pose manifest + robot_test URDFs and add markers

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
from mapanything.utils.cropping import crop_resize_if_necessary
from mapanything.utils.viz import predictions_to_glb

from capture_contract import VIEW_NAMES, present_views, resolve_reconstruction_captures
from gripper_pose import (
    DEFAULT_G1_URDF,
    DEFAULT_GRIPPER_URDF,
    resolve_gripper_poses,
    write_gripper_poses,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = os.path.expanduser(
    os.environ.get("G2_OUT_ROOT", str(PROJECT_ROOT / "outputs"))
)

DEFAULT_FRUSTUM_DEPTH_M = 0.06
DEFAULT_CAMERA_CENTER_RADIUS_M = 0.01
DEFAULT_ORIGIN_AXIS_LENGTH_M = 0.12
DEFAULT_ORIGIN_SIZE_M = 0.012
DEFAULT_GRIPPER_CENTER_RADIUS_M = 0.018
DEFAULT_GRIPPER_AXIS_LENGTH_M = 0.08

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

GRIPPER_MARKER_COLORS = {
    "left": [255, 160, 0, 255],  # orange
    "right": [0, 220, 255, 255],  # cyan
}


def camera_frustum_mesh(
    K,
    pose,
    img_hw,
    color,
    frustum_depth=DEFAULT_FRUSTUM_DEPTH_M,
):
    """Small solid frustum pyramid for one camera.

    The apex is the exact camera center and the base is formed by image-plane
    corners unprojected to ``frustum_depth`` meters.  The old diagnostic used a
    30 cm frustum which could obscure nearby geometry; the default marker is now
    6 cm long.
    """
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


def camera_center_mesh(pose, color, radius=DEFAULT_CAMERA_CENTER_RADIUS_M):
    """Sphere centered at the exact cam2world translation."""
    center = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    center.apply_translation(np.asarray(pose, dtype=np.float64)[:3, 3])
    center.visual.face_colors = np.tile(
        np.asarray(color, dtype=np.uint8), (len(center.faces), 1)
    )
    return center


def world_origin_frame_mesh(
    axis_length=DEFAULT_ORIGIN_AXIS_LENGTH_M,
    origin_size=DEFAULT_ORIGIN_SIZE_M,
):
    """RGB XYZ coordinate frame whose center is exactly world ``[0, 0, 0]``."""
    return trimesh.creation.axis(
        origin_size=origin_size,
        axis_radius=origin_size / 3.0,
        axis_length=axis_length,
    )


def gripper_center_mesh(
    pose,
    color,
    radius=DEFAULT_GRIPPER_CENTER_RADIUS_M,
):
    """Colored sphere centered at a resolved gripper tool-center point."""
    marker = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    marker.apply_translation(np.asarray(pose, dtype=np.float64)[:3, 3])
    marker.visual.face_colors = np.tile(
        np.asarray(color, dtype=np.uint8), (len(marker.faces), 1)
    )
    return marker


def gripper_frame_mesh(pose, axis_length=DEFAULT_GRIPPER_AXIS_LENGTH_M):
    """Small XYZ frame using the production hand-mount orientation at the tip."""
    frame = trimesh.creation.axis(
        origin_size=axis_length / 10.0,
        axis_radius=axis_length / 30.0,
        axis_length=axis_length,
    )
    frame.apply_transform(np.asarray(pose, dtype=np.float64))
    return frame


def view_debug_images(images, view_names=VIEW_NAMES):
    """Return per-view solid colors: head red, left green, right blue.

    ``view_names`` must match the views stacked into ``images`` in order, which
    is a subset of VIEW_NAMES when the reconstruction used fewer cameras.
    """
    return np.stack(
        [
            np.broadcast_to(VIEW_DEBUG_COLORS[name], images[view_idx].shape)
            for view_idx, name in enumerate(view_names)
        ],
        axis=0,
    )


def build_glb_scene(
    world_points,
    display_images,
    final_masks,
    npz,
    *,
    show_cameras,
    frustum_depth,
    origin_axis_length,
    marker_intrinsics=None,
    gripper_poses=None,
):
    """Build one GLB scene, including optional rig/origin reference markers."""
    scene_glb = predictions_to_glb(
        {
            "world_points": world_points,
            "images": display_images,
            "final_masks": final_masks,
        },
        as_mesh=False,
    )
    if not show_cameras and not gripper_poses:
        return scene_glb

    # predictions_to_glb already flipped the scene 180 deg around X; new
    # geometry added afterwards must carry the same transform itself.
    flip_x = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    if show_cameras:
        for name in present_views(npz):
            camera_pose = npz[f"{name}_camera_pose"].astype(np.float64)
            camera_intrinsics = (
                npz[f"{name}_intrinsics"].astype(np.float64)
                if marker_intrinsics is None
                else marker_intrinsics[name]
            )
            frustum = camera_frustum_mesh(
                camera_intrinsics,
                camera_pose,
                npz[f"{name}_mask"].shape,
                CAM_MARKER_COLORS[name],
                frustum_depth=frustum_depth,
            )
            center = camera_center_mesh(camera_pose, CAM_MARKER_COLORS[name])
            frustum.apply_transform(flip_x)
            center.apply_transform(flip_x)
            scene_glb.add_geometry(frustum, geom_name=f"camera_{name}")
            scene_glb.add_geometry(center, geom_name=f"camera_center_{name}")
        origin_frame = world_origin_frame_mesh(axis_length=origin_axis_length)
        origin_frame.apply_transform(flip_x)
        scene_glb.add_geometry(origin_frame, geom_name="world_origin_xyz")
    if gripper_poses:
        for side in ("left", "right"):
            pose = np.asarray(
                gripper_poses["poses"][side]["pose_matrix"], dtype=np.float64
            )
            center = gripper_center_mesh(pose, GRIPPER_MARKER_COLORS[side])
            frame = gripper_frame_mesh(pose)
            center.apply_transform(flip_x)
            frame.apply_transform(flip_x)
            scene_glb.add_geometry(center, geom_name=f"gripper_{side}_center")
            scene_glb.add_geometry(frame, geom_name=f"gripper_{side}_frame")
    return scene_glb


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
    K = npz[f"{name}_intrinsics"].astype(np.float32)
    return unproject_view_with_intrinsics(npz, name, K)


def unproject_view_with_intrinsics(npz, name, intrinsics):
    """Unproject one stored depth map with an explicitly selected K."""
    depth = torch.from_numpy(npz[f"{name}_depth_z"].astype(np.float32))
    K = torch.from_numpy(np.asarray(intrinsics, dtype=np.float32))
    pose = torch.from_numpy(npz[f"{name}_camera_pose"].astype(np.float32))
    pts3d, valid = depthmap_to_world_frame(depth, K, pose)
    return pts3d.numpy(), valid.numpy()


def load_preprocessed_calibrated_intrinsics(
    capture, name, target_hw, undist_root
):
    """Replay MapAnything's image resize/crop to obtain calibrated input K."""
    capture_dir = Path(undist_root) / capture
    image_path = capture_dir / f"{name}.png"
    k_path = capture_dir / f"{name}_K.json"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(
            f"Per-camera K A/B requires undistorted image: {image_path}"
        )
    try:
        with k_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Per-camera K A/B cannot read calibrated intrinsics {k_path}: {exc}"
        ) from exc
    source_k = np.asarray(data.get("K"), dtype=np.float64)
    if source_k.shape != (3, 3) or not np.isfinite(source_k).all():
        raise ValueError(f"{k_path} must contain a finite 3x3 K matrix")
    target_h, target_w = target_hw
    _, processed_k = crop_resize_if_necessary(
        image,
        (int(target_w), int(target_h)),
        intrinsics=source_k,
    )
    return np.asarray(processed_k, dtype=np.float32)


def select_per_camera_k(predicted_intrinsics, calibrated_intrinsics):
    """Experimental A/B policy validated on the five ChArUco captures.

    head: calibrated K; hand_left: model-predicted K; hand_right: model focal
    lengths combined with the calibrated principal point.
    """
    selected = {
        "head": np.asarray(calibrated_intrinsics["head"], dtype=np.float32).copy(),
        "hand_left": np.asarray(
            predicted_intrinsics["hand_left"], dtype=np.float32
        ).copy(),
        "hand_right": np.asarray(
            predicted_intrinsics["hand_right"], dtype=np.float32
        ).copy(),
    }
    selected["hand_right"][0, 2] = calibrated_intrinsics["hand_right"][0, 2]
    selected["hand_right"][1, 2] = calibrated_intrinsics["hand_right"][1, 2]
    return selected


def validate_unprojection_consistency(
    computed,
    stored,
    mask,
    *,
    absolute_tolerance_m=2e-3,
    relative_tolerance=4e-3,
):
    """Validate CPU re-unprojection against GPU-stored world points.

    GPU inference runs under bfloat16 autocast while this script replays the
    operation in CPU float32.  The resulting Cartesian error grows with range,
    so a fixed absolute threshold incorrectly rejects distant points.  Use a
    per-point L-infinity bound of ``atol + rtol * ||point||`` instead.

    ``relative_tolerance`` is set from the precision of the arithmetic being
    replayed, not from a wish: bfloat16 keeps an 8-bit mantissa, so its relative
    precision is about ``2**-8 = 3.9e-3``.  A tighter bound asks a bf16 result to
    be more accurate than bf16 can represent, and fails on distant points for
    purely numerical reasons.  At 2 m the limit is 10 mm and at 20 m it is 82 mm,
    which still catches a wrong K, pose, depth convention or matrix direction --
    those are wrong by a fraction of the point coordinates themselves, that is
    by meters, not millimeters.
    """

    computed = np.asarray(computed)
    stored = np.asarray(stored)
    mask = np.asarray(mask, dtype=bool)
    if computed.shape != stored.shape or computed.shape[:-1] != mask.shape:
        raise AssertionError(
            "Unprojection validation shape mismatch: "
            f"computed={computed.shape}, stored={stored.shape}, mask={mask.shape}"
        )
    if not mask.any():
        return {
            "max_abs_m": 0.0,
            "p99_9_abs_m": 0.0,
            "max_tolerance_ratio": 0.0,
            "checked_points": 0,
        }

    computed_valid = computed[mask]
    stored_valid = stored[mask]
    point_error = np.max(np.abs(computed_valid - stored_valid), axis=-1)
    point_scale = np.maximum(
        np.linalg.norm(computed_valid, axis=-1),
        np.linalg.norm(stored_valid, axis=-1),
    )
    allowed_error = absolute_tolerance_m + relative_tolerance * point_scale
    tolerance_ratio = point_error / allowed_error
    stats = {
        "max_abs_m": float(point_error.max()),
        "p99_9_abs_m": float(np.quantile(point_error, 0.999)),
        "max_tolerance_ratio": float(tolerance_ratio.max()),
        "checked_points": int(point_error.size),
    }
    if np.any(point_error > allowed_error):
        worst = int(np.argmax(tolerance_ratio))
        raise AssertionError(
            "CPU unprojection is inconsistent with stored GPU pts3d: "
            f"max_abs={stats['max_abs_m']:.9g} m, "
            f"p99.9={stats['p99_9_abs_m']:.9g} m, "
            f"worst_error={point_error[worst]:.9g} m, "
            f"allowed={allowed_error[worst]:.9g} m at "
            f"range={point_scale[worst]:.9g} m"
        )
    return stats


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
    per_camera_k_ab=False,
    show_grippers=False,
    frustum_depth=DEFAULT_FRUSTUM_DEPTH_M,
    origin_axis_length=DEFAULT_ORIGIN_AXIS_LENGTH_M,
    g1_urdf=DEFAULT_G1_URDF,
    gripper_urdf=DEFAULT_GRIPPER_URDF,
    output_root=DEFAULT_OUTPUT_ROOT,
):
    out_dir = os.path.join(output_root, capture)
    undist_root = os.path.join(output_root, "undistorted")
    npz_path = os.path.join(out_dir, "views.npz")
    npz = np.load(npz_path)
    print(f"\n===== {capture} =====")

    views_present = present_views(npz)
    pts_list, img_list, mask_list, conf_list = [], [], [], []
    have_conf = all(f"{n}_conf" in npz for n in views_present)

    for name in views_present:
        mask = npz[f"{name}_mask"].astype(bool)

        # Geometry: unproject from depth, verify against stored pts3d if present.
        # Inference computed pts3d on a GPU TF32/BF16 path, so validate with a
        # range-aware numerical tolerance before using the exact stored values.
        pts3d, _valid = unproject_view(npz, name)
        if f"{name}_pts3d" in npz:
            stored = npz[f"{name}_pts3d"]
            stats = validate_unprojection_consistency(pts3d, stored, mask)
            print(
                f"  [{name}] unprojection vs stored pts3d: "
                f"max={stats['max_abs_m']:.2e} m, "
                f"p99.9={stats['p99_9_abs_m']:.2e} m, "
                f"tolerance ratio={stats['max_tolerance_ratio']:.3f} (ok)"
            )
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
    cols = np.clip(images.reshape(-1, 3)[flat_keep] * 255.0, 0, 255).astype(
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

    gripper_poses = None
    gripper_pose_path = None
    if show_grippers:
        # G1-only: the overlay resolves G1 link names and needs the G1
        # pose_conversion_manifest.json.  Say so rather than failing on a bare
        # missing file when this runs against another robot's capture.
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
        left_position = gripper_poses["poses"]["left"]["position_m"]
        right_position = gripper_poses["poses"]["right"]["position_m"]
        print(
            "  resolved gripper centers in base_link (m): "
            f"left={np.round(left_position, 6).tolist()}, "
            f"right={np.round(right_position, 6).tolist()}"
        )

    # --- Export ------------------------------------------------------------------
    # Normal RGB GLB via the repo utility (same 180-deg X flip as scene.glb).
    scene_glb = build_glb_scene(
        world_points,
        images,
        final_masks,
        npz,
        show_cameras=show_cameras,
        frustum_depth=frustum_depth,
        origin_axis_length=origin_axis_length,
        gripper_poses=gripper_poses,
    )
    if show_cameras:
        print(
            f"  added small camera markers ({frustum_depth:.3f} m): "
            f"{', '.join(VIEW_NAMES)}; world origin XYZ axis "
            f"({origin_axis_length:.3f} m)"
        )
    glb_path = os.path.join(out_dir, "scene_filtered.glb")
    scene_glb.export(glb_path)

    # PLY in raw world coordinates (same convention as scene.ply).
    ply_path = os.path.join(out_dir, "scene_filtered.ply")
    trimesh.PointCloud(vertices=pts, colors=cols).export(ply_path)

    by_view_glb_path = None
    if color_by_view:
        by_view_glb = build_glb_scene(
            world_points,
            view_debug_images(images, views_present),
            final_masks,
            npz,
            show_cameras=show_cameras,
            frustum_depth=frustum_depth,
            origin_axis_length=origin_axis_length,
            gripper_poses=gripper_poses,
        )
        by_view_glb_path = os.path.join(out_dir, "scene_filtered_by_view.glb")
        by_view_glb.export(by_view_glb_path)

    per_camera_k_glb_path = None
    per_camera_k_by_view_glb_path = None
    if per_camera_k_ab:
        predicted_intrinsics = {
            name: npz[f"{name}_intrinsics"].astype(np.float32)
            for name in views_present
        }
        calibrated_intrinsics = {
            name: load_preprocessed_calibrated_intrinsics(
                capture,
                name,
                npz[f"{name}_mask"].shape,
                undist_root,
            )
            for name in views_present
        }
        selected_intrinsics = select_per_camera_k(
            predicted_intrinsics, calibrated_intrinsics
        )
        per_camera_points = np.stack(
            [
                (
                    world_points[view_idx]
                    if name == "hand_left"
                    else unproject_view_with_intrinsics(
                        npz, name, selected_intrinsics[name]
                    )[0].astype(np.float32)
                )
                for view_idx, name in enumerate(views_present)
            ],
            axis=0,
        )
        per_camera_keep = build_filter_mask(
            per_camera_points,
            conf,
            max_radius=max_radius,
            bbox=bbox,
            min_conf=min_conf,
        )
        per_camera_masks = masks & per_camera_keep
        if not per_camera_masks.any():
            raise RuntimeError(
                "All per-camera K A/B points filtered out; nothing to export"
            )
        per_camera_scene = build_glb_scene(
            per_camera_points,
            images,
            per_camera_masks,
            npz,
            show_cameras=show_cameras,
            frustum_depth=frustum_depth,
            origin_axis_length=origin_axis_length,
            marker_intrinsics=selected_intrinsics,
            gripper_poses=gripper_poses,
        )
        per_camera_k_glb_path = os.path.join(
            out_dir, "scene_filtered_per_camera_k.glb"
        )
        per_camera_scene.export(per_camera_k_glb_path)
        if color_by_view:
            per_camera_by_view_scene = build_glb_scene(
                per_camera_points,
                view_debug_images(images, views_present),
                per_camera_masks,
                npz,
                show_cameras=show_cameras,
                frustum_depth=frustum_depth,
                origin_axis_length=origin_axis_length,
                marker_intrinsics=selected_intrinsics,
                gripper_poses=gripper_poses,
            )
            per_camera_k_by_view_glb_path = os.path.join(
                out_dir, "scene_filtered_per_camera_k_by_view.glb"
            )
            per_camera_by_view_scene.export(per_camera_k_by_view_glb_path)

    print(
        f"  saved: {glb_path} ({os.path.getsize(glb_path)} B), "
        f"{ply_path} ({os.path.getsize(ply_path)} B)"
    )
    if by_view_glb_path is not None:
        print(
            f"  saved view-colored diagnostic: {by_view_glb_path} "
            f"({os.path.getsize(by_view_glb_path)} B; "
            "head=red, hand_left=green, hand_right=blue)"
        )
    if per_camera_k_glb_path is not None:
        print(
            "  saved experimental per-camera K A/B: "
            f"{per_camera_k_glb_path} ({os.path.getsize(per_camera_k_glb_path)} B); "
            "head=calibrated K, hand_left=model K, "
            "hand_right=model focal+calibrated principal point"
        )
    if per_camera_k_by_view_glb_path is not None:
        print(
            "  saved view-colored per-camera K A/B: "
            f"{per_camera_k_by_view_glb_path} "
            f"({os.path.getsize(per_camera_k_by_view_glb_path)} B)"
        )
    if gripper_pose_path is not None:
        print(
            "  added gripper markers: left=orange, right=cyan; "
            f"saved provenance: {gripper_pose_path}"
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
        "rgb_glb": glb_path,
        "view_colored_glb": by_view_glb_path,
        "per_camera_k_ab": per_camera_k_ab,
        "per_camera_k_glb": per_camera_k_glb_path,
        "per_camera_k_by_view_glb": per_camera_k_by_view_glb_path,
        "show_cameras_and_origin": show_cameras,
        "camera_frustum_depth_m": frustum_depth if show_cameras else None,
        "origin_axis_length_m": origin_axis_length if show_cameras else None,
        "show_grippers": show_grippers,
        "gripper_pose_json": gripper_pose_path,
        "gripper_positions_m": (
            {
                side: gripper_poses["poses"][side]["position_m"]
                for side in ("left", "right")
            }
            if gripper_poses is not None
            else None
        ),
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
        help=(
            "Add small colored camera frustum/center markers and a world-origin "
            "XYZ frame to the exported GLB"
        ),
    )
    parser.add_argument(
        "--color_by_view",
        action="store_true",
        help=(
            "Additionally export scene_filtered_by_view.glb with head/left/right "
            "points colored red/green/blue; keep the normal RGB GLB/PLY"
        ),
    )
    parser.add_argument(
        "--per_camera_k_ab",
        action="store_true",
        help=(
            "Additionally export experimental per-camera K A/B GLBs: head uses "
            "calibrated K, left uses model K, right uses model focal lengths "
            "with the calibrated principal point"
        ),
    )
    parser.add_argument(
        "--show_grippers",
        action="store_true",
        help=(
            "Resolve G1 left/right gripper centers in base_link from the capture "
            "robot poses and robot_test URDFs, then add orange/cyan markers"
        ),
    )
    parser.add_argument(
        "--g1_urdf",
        default=str(DEFAULT_G1_URDF),
        help="Current G1 URDF used to validate WBC Link7-to-hand mounts",
    )
    parser.add_argument(
        "--gripper_urdf",
        default=str(DEFAULT_GRIPPER_URDF),
        help="URDF providing fixed gripper-base-to-center displacement",
    )
    parser.add_argument(
        "--frustum_depth",
        type=float,
        default=DEFAULT_FRUSTUM_DEPTH_M,
        help=(
            "Camera frustum length in meters for --show_cameras "
            f"(default: {DEFAULT_FRUSTUM_DEPTH_M})"
        ),
    )
    parser.add_argument(
        "--origin_axis_length",
        type=float,
        default=DEFAULT_ORIGIN_AXIS_LENGTH_M,
        help=(
            "World-origin XYZ axis length in meters for --show_cameras "
            f"(default: {DEFAULT_ORIGIN_AXIS_LENGTH_M})"
        ),
    )
    args = parser.parse_args()
    if args.frustum_depth <= 0:
        parser.error("--frustum_depth must be positive")
    if args.origin_axis_length <= 0:
        parser.error("--origin_axis_length must be positive")
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
                per_camera_k_ab=args.per_camera_k_ab,
                show_grippers=args.show_grippers,
                frustum_depth=args.frustum_depth,
                origin_axis_length=args.origin_axis_length,
                g1_urdf=args.g1_urdf,
                gripper_urdf=args.gripper_urdf,
                output_root=args.output_root,
            )
        )
    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
