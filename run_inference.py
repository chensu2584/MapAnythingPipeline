#!/usr/bin/env python
"""
Step B: MapAnything 3D reconstruction inference on undistorted G1/G2 captures.

For each capture:
  - Load the 3 undistorted images + adjusted newK, plus validated metric OpenCV
    RDF cam2world poses used as model conditions and a world-frame head anchor.
  - Build views [{"img": HxWx3 uint8, "intrinsics": 3x3, "camera_poses": 4x4}],
    preprocess_inputs (unify to 518-set).
  - model.infer(..., memory_efficient_inference=True, use_amp, bf16, apply_mask, mask_edges).
  - By default fit one uniform scale from camera baselines, apply it to model depth
    and relative translations, then rigidly anchor the calibrated head pose.
  - Save scene.glb, scene.ply, views.npz, summary.json to outputs/<capture>/.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.environ.get("TMPDIR", "/tmp"), "mapanything-matplotlib-cache"),
)

import cv2
import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filter_export import build_filter_mask
from pose_export import (
    CALIBRATED_INPUT,
    DEFAULT_POSE_EXPORT_MODE,
    MODEL_PREDICTION_ARBITRARY_SCALE,
    MODEL_RELATIVE_HEAD_ANCHORED,
    MODEL_RELATIVE_HEAD_ANCHORED_BASELINE_SCALED,
    MODEL_RELATIVE_HEAD_ANCHORED_DEPTH_AFFINE,
    MODEL_RELATIVE_HEAD_ANCHORED_DEPTH_SCALED,
    POSE_EXPORT_MODES,
    estimate_depth_similarity_scale,
    pose_delta,
    select_export_poses,
)
from capture_contract import (
    POSES_FILE,
    PROVENANCE_FILES,
    VIEW_NAMES,
    resolve_captures,
    validate_pose_document,
)

from mapanything.models import MapAnything
from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.image import preprocess_inputs
from mapanything.utils.viz import predictions_to_glb

# Must match undistort.py's G2_OUT_ROOT (Step A writes into <G2_OUT_ROOT>/undistorted).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = os.path.expanduser(
    os.environ.get("G2_OUT_ROOT", str(PROJECT_ROOT / "outputs"))
)
DEFAULT_UNDIST_ROOT = os.path.join(DEFAULT_OUTPUT_ROOT, "undistorted")


def _timing_value(seconds):
    return round(float(seconds), 6)


def _print_timing(label, seconds):
    print(f"  [TIMING] {label}: {float(seconds):.3f} s")


def _synchronize_model_device(model):
    """Synchronize CUDA only at timing boundaries so GPU timings are meaningful."""

    device = getattr(model, "device", None)
    if device is None:
        return
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _load_adjusted_K(path, image_width, image_height):
    try:
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read adjusted intrinsics {path}: {exc}") from exc
    K = np.asarray(data.get("K"), dtype=np.float32)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError(f"{path} must contain a finite 3x3 K matrix")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError(f"{path} contains an invalid pinhole K matrix")
    if not (0 <= K[0, 2] < image_width and 0 <= K[1, 2] < image_height):
        raise ValueError(f"{path} principal point lies outside the image")
    declared_size = (data.get("width"), data.get("height"))
    if declared_size != (image_width, image_height):
        raise ValueError(
            f"{path} declares size {declared_size}, image is {(image_width, image_height)}"
        )
    return K


def load_registered_depth(capture, undist_root=DEFAULT_UNDIST_ROOT):
    """Load Step A's metric depth, already reprojected into each colour view.

    Returns ``{view: (depth_z, valid)}`` in meters, or an empty dict when the
    capture carries no depth.  Absent depth is normal (G1 has none) and must not
    be an error.
    """
    path = Path(undist_root) / capture / "registered_depth.npz"
    if not path.is_file():
        return {}
    result = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            if not key.endswith("_depth_z"):
                continue
            view = key[: -len("_depth_z")]
            valid_key = f"{view}_depth_valid"
            if valid_key not in data.files:
                continue
            depth = np.asarray(data[key], dtype=np.float64)
            valid = np.asarray(data[valid_key], dtype=bool) & np.isfinite(depth)
            result[view] = (depth, valid)
    return result


def load_self_occlusion_masks(capture, undist_root=DEFAULT_UNDIST_ROOT):
    """Load Step A's robot self-occlusion masks, or an empty dict if absent.

    Absent masks are normal: the capture may predate the mask stage, or the
    robot may have no geometry in view.  Missing masks must not be an error.
    """
    path = Path(undist_root) / capture / "self_occlusion_mask.npz"
    if not path.is_file():
        return {}
    result = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            if key.endswith("_self_mask"):
                result[key[: -len("_self_mask")]] = np.asarray(data[key], dtype=bool)
    return result


def resample_mask_to(mask, target_hw):
    """Nearest-neighbour resample a boolean mask to ``target_hw``."""
    target_h, target_w = (int(v) for v in target_hw)
    height, width = mask.shape
    if (height, width) == (target_h, target_w):
        return mask
    rows = np.clip((np.arange(target_h) + 0.5) * height / target_h, 0, height - 1).astype(int)
    cols = np.clip((np.arange(target_w) + 0.5) * width / target_w, 0, width - 1).astype(int)
    return mask[np.ix_(rows, cols)]


def resample_depth_to(depth, valid, target_hw):
    """Nearest-neighbour resample a depth map and its validity to ``target_hw``.

    Nearest neighbour rather than interpolation: averaging across a depth
    discontinuity invents a surface that exists in neither the sensor nor the
    model, and this map is used to fit a scale.
    """
    target_h, target_w = (int(v) for v in target_hw)
    height, width = depth.shape
    if (height, width) == (target_h, target_w):
        return depth, valid
    rows = np.clip((np.arange(target_h) + 0.5) * height / target_h, 0, height - 1).astype(int)
    cols = np.clip((np.arange(target_w) + 0.5) * width / target_w, 0, width - 1).astype(int)
    index = np.ix_(rows, cols)
    return depth[index], valid[index]


def load_views(
    capture,
    undist_root=DEFAULT_UNDIST_ROOT,
    allow_missing_poses=False,
    ignore_poses=False,
    depth_inputs=None,
    depth_holdout=0.0,
    self_masks=None,
):
    cap_dir = Path(undist_root) / capture

    poses = None
    pose_contract = None
    poses_path = cap_dir / POSES_FILE
    if ignore_poses:
        print("  metric pose file explicitly ignored; model will estimate pose/scale")
    elif poses_path.is_file():
        poses, pose_contract = validate_pose_document(poses_path)
        print(
            f"  using metric poses: {pose_contract['frame_convention']}, "
            f"world={pose_contract['world_frame']}, unit=m"
        )
    elif not allow_missing_poses:
        raise FileNotFoundError(
            f"{poses_path} is required for metric reconstruction; "
            "use --allow-missing-poses only for intentional pose-free inference"
        )
    else:
        print("  no camera pose file found; model will estimate poses (arbitrary scale)")

    views = []
    for name in VIEW_NAMES:
        image_path = cap_dir / f"{name}.png"
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # HxWx3 uint8
        height, width = img_rgb.shape[:2]
        K = _load_adjusted_K(cap_dir / f"{name}_K.json", width, height)
        if self_masks and name in self_masks:
            robot_pixels = resample_mask_to(self_masks[name], img_rgb.shape[:2])
            # Neutral grey rather than black: a black region reads as a dark
            # surface, while grey carries no gradient for the network to latch
            # onto.  The pixels are excluded from the output either way.
            img_rgb = img_rgb.copy()
            img_rgb[robot_pixels] = 128
            print(
                f"  self-mask input: {name} blanked {int(robot_pixels.sum())} px "
                f"({100 * robot_pixels.mean():.1f}% of frame)"
            )
        view = {
            "img": img_rgb.astype(np.uint8),
            "intrinsics": torch.from_numpy(K),
        }
        if poses is not None:
            pose = np.asarray(poses[name], dtype=np.float32)
            view["camera_poses"] = torch.from_numpy(pose)
            view["is_metric_scale"] = torch.ones(1, dtype=torch.bool)
        if depth_inputs and name in depth_inputs:
            depth, valid = depth_inputs[name]
            fed = np.where(valid, depth, 0.0).astype(np.float32)
            if depth_holdout > 0.0:
                # Withhold a random subset so the head view keeps pixels the model
                # never saw; evaluating a fed-in depth against itself is circular.
                rng = np.random.default_rng(abs(hash((capture, name))) % (2**32))
                held = rng.random(fed.shape) < float(depth_holdout)
                fed[held] = 0.0
            view["depth_z"] = torch.from_numpy(fed)
            view["is_metric_scale"] = torch.ones(1, dtype=torch.bool)
            print(
                f"  depth input: {name} feeding {int((fed > 0).sum())} metric pixels"
                + (f" (holdout {depth_holdout:.0%})" if depth_holdout > 0 else "")
            )
        views.append(view)
    return views, pose_contract


def validate_preprocessed_views(raw_views, processed_views):
    if len(processed_views) != len(VIEW_NAMES):
        raise ValueError(f"Expected {len(VIEW_NAMES)} processed views")
    report = {"view_order": list(VIEW_NAMES), "views": {}}
    for name, raw, processed in zip(VIEW_NAMES, raw_views, processed_views):
        image = processed["img"]
        K = processed["intrinsics"]
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError(f"Processed {name} image has invalid shape {tuple(image.shape)}")
        if K.shape != (1, 3, 3) or not torch.isfinite(K).all():
            raise ValueError(f"Processed {name} intrinsics have invalid shape/values")
        pose_diff = None
        if "camera_poses" in raw:
            processed_pose = processed.get("camera_poses")
            if processed_pose is None or processed_pose.shape != (1, 4, 4):
                raise ValueError(f"Processed {name} pose has invalid shape")
            pose_diff = float(
                torch.max(torch.abs(processed_pose[0] - raw["camera_poses"])).item()
            )
            if pose_diff > 1e-6:
                raise ValueError(f"Preprocessing changed {name} pose by {pose_diff}")
            metric = processed.get("is_metric_scale")
            if metric is None or metric.shape != (1,) or not bool(metric[0]):
                raise ValueError(f"Processed {name} is not explicitly marked metric")
        report["views"][name] = {
            "model_input_resolution_hw": [int(image.shape[2]), int(image.shape[3])],
            "pose_preprocess_max_abs_diff": pose_diff,
        }
    return report


def run_capture(
    model,
    capture,
    minibatch_size=None,
    max_radius=None,
    bbox=None,
    min_conf=None,
    undist_root=DEFAULT_UNDIST_ROOT,
    output_root=DEFAULT_OUTPUT_ROOT,
    allow_missing_poses=False,
    ignore_poses=False,
    pose_export_mode=DEFAULT_POSE_EXPORT_MODE,
    memory_efficient_inference=True,
    use_depth_input=False,
    depth_holdout=0.0,
    use_self_mask_input=False,
):
    capture_started_at = time.perf_counter()
    timings = {}
    out_dir = os.path.join(output_root, capture)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n===== {capture} =====")

    self_masks = load_self_occlusion_masks(capture, undist_root)
    self_mask_report = {}
    if self_masks:
        print("  robot self-occlusion masks available for: " + ", ".join(sorted(self_masks)))
    elif use_self_mask_input:
        raise FileNotFoundError(
            f"--self-mask-input was requested but {capture} has no self_occlusion_mask.npz; "
            "run self_occlusion_mask.py first"
        )

    registered_depth = load_registered_depth(capture, undist_root)
    if registered_depth:
        print(
            "  registered metric depth available for: "
            + ", ".join(sorted(registered_depth))
        )
    elif use_depth_input:
        raise FileNotFoundError(
            f"--depth-input was requested but {capture} has no registered_depth.npz; "
            "run Step A on a capture whose robot provides a metric depth camera"
        )

    section_started_at = time.perf_counter()
    views, pose_contract = load_views(
        capture,
        undist_root,
        allow_missing_poses,
        ignore_poses=ignore_poses,
        depth_inputs=registered_depth if use_depth_input else None,
        depth_holdout=depth_holdout,
        self_masks=self_masks if use_self_mask_input else None,
    )
    timings["input_load"] = time.perf_counter() - section_started_at
    _print_timing("input_load", timings["input_load"])

    section_started_at = time.perf_counter()
    processed = preprocess_inputs(views)
    preprocess_report = validate_preprocessed_views(views, processed)
    timings["preprocess_inputs"] = time.perf_counter() - section_started_at
    _print_timing("preprocess_inputs", timings["preprocess_inputs"])
    if pose_contract is None and any(v is not None for v in (max_radius, bbox)):
        raise ValueError("Metric radius/bbox filters require metric camera poses")

    infer_kwargs = dict(
        memory_efficient_inference=memory_efficient_inference,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )
    if minibatch_size is not None:
        infer_kwargs["minibatch_size"] = minibatch_size

    print(
        "  inference dense-head mode: "
        + ("memory-efficient" if memory_efficient_inference else "fast/all-views")
    )
    _synchronize_model_device(model)
    section_started_at = time.perf_counter()
    outputs = model.infer(processed, **infer_kwargs)
    _synchronize_model_device(model)
    timings["model_inference"] = time.perf_counter() - section_started_at
    _print_timing("model_inference", timings["model_inference"])
    if len(outputs) != len(VIEW_NAMES):
        raise RuntimeError(f"Model returned {len(outputs)} views, expected {len(VIEW_NAMES)}")

    section_started_at = time.perf_counter()
    model_pose_arrays = [
        pred["camera_poses"][0].detach().cpu().numpy().astype(np.float64)
        for pred in outputs
    ]
    calibrated_pose_arrays = (
        [
            view["camera_poses"][0].detach().cpu().numpy().astype(np.float64)
            for view in processed
        ]
        if pose_contract is not None
        else None
    )
    # Compare predicted head depth against the measured metric depth.  This runs
    # whatever the export mode is: as a standalone quality report by default, and
    # additionally as the scale estimator when the depth-scaled mode is asked for.
    depth_scale_inputs = None
    depth_diagnostic = None
    if registered_depth:
        depth_diagnostic = {}
        for view_index, name in enumerate(VIEW_NAMES):
            if name not in registered_depth:
                continue
            model_depth = (
                outputs[view_index]["depth_z"][0].squeeze(-1).detach().cpu().numpy()
                .astype(np.float64)
            )
            reference, valid = registered_depth[name]
            reference, valid = resample_depth_to(reference, valid, model_depth.shape)
            entry = estimate_depth_similarity_scale(model_depth, reference, valid)
            entry["was_fed_to_model"] = bool(use_depth_input)
            entry["depth_holdout_fraction"] = float(depth_holdout)
            if use_depth_input and depth_holdout <= 0.0:
                entry["evaluation_warning"] = (
                    "this depth was fed to the model, so agreement here is circular; "
                    "use --depth-holdout or judge the views that were not fed"
                )
            depth_diagnostic[name] = entry
            print(
                f"  depth check[{name}]: "
                + (
                    f"scale={entry['scale']:.6f} "
                    f"residual RMSE={entry['residual_rmse_m'] * 1000:.1f} mm "
                    f"median={entry['residual_median_abs_m'] * 1000:.1f} mm "
                    f"inliers={entry['inlier_ratio']:.1%} "
                    f"affine b={entry['affine_test']['b_m'] * 1000:+.1f} mm "
                    f"over {entry['pixel_count']} px"
                    if entry.get("converged")
                    else f"not usable ({entry.get('reason')})"
                )
            )
        head_view = VIEW_NAMES[0]
        if head_view in registered_depth:
            model_depth = (
                outputs[0]["depth_z"][0].squeeze(-1).detach().cpu().numpy().astype(np.float64)
            )
            reference, valid = resample_depth_to(
                *registered_depth[head_view], model_depth.shape
            )
            depth_scale_inputs = {
                "model_depth": model_depth,
                "reference_depth": reference,
                "valid": valid,
            }

    (
        export_pose_arrays,
        effective_pose_mode,
        world_t_model_reference,
        scale_report,
    ) = (
        select_export_poses(
            model_pose_arrays,
            calibrated_pose_arrays,
            requested_mode=pose_export_mode,
            depth_scale_inputs=depth_scale_inputs,
        )
    )
    pose_mode_labels = {
        MODEL_RELATIVE_HEAD_ANCHORED_BASELINE_SCALED: (
            "model_prediction_baseline_scaled_and_rigidly_anchored_to_calibrated_head"
        ),
        MODEL_RELATIVE_HEAD_ANCHORED: "model_prediction_rigidly_anchored_to_calibrated_head",
        CALIBRATED_INPUT: "calibrated_input_pose_legacy_hybrid",
        MODEL_PREDICTION_ARBITRARY_SCALE: "model_prediction_arbitrary_scale",
    }
    print(
        f"  pose export mode: requested={pose_export_mode}, "
        f"effective={effective_pose_mode}"
    )
    similarity_scale = scale_report["scale"] if scale_report is not None else 1.0
    # Only the affine diagnostic mode sets an offset; every other mode leaves
    # this at zero and stays a pure similarity transform.
    depth_offset_m = (
        float(scale_report.get("depth_offset_m", 0.0)) if scale_report else 0.0
    )
    if depth_offset_m:
        print(
            f"  DIAGNOSTIC affine depth correction: depth = {similarity_scale:.6f} * model "
            f"{depth_offset_m:+.4f} m -- this warps geometry and breaks cross-camera "
            "agreement; do not plan against this output"
        )
    scale_metadata = scale_report or {
        "applied": False,
        "scale": 1.0,
        "estimator": None,
    }
    if scale_report is not None:
        print(
            f"  baseline similarity scale: {similarity_scale:.6f} | "
            f"baseline RMSE {scale_report['baseline_rmse_before_m'] * 1000:.2f} -> "
            f"{scale_report['baseline_rmse_after_m'] * 1000:.2f} mm"
        )

    world_points_list = []
    images_list = []
    masks_list = []

    npz = {}
    summary = {
        "capture": capture,
        "metric_pose_input": pose_contract is not None,
        "pose_input_mode": "ignored" if ignore_poses else "metric_if_available",
        "pose_contract": pose_contract,
        "preprocess_validation": preprocess_report,
        "memory_efficient_inference": bool(memory_efficient_inference),
        "pose_export_mode_requested": pose_export_mode,
        "pose_export_mode_effective": effective_pose_mode,
        "camera_pose_used_for_export": pose_mode_labels[effective_pose_mode],
        "similarity_scale_correction": scale_metadata,
        "metric_depth_diagnostic": depth_diagnostic,
        "self_occlusion_mask": self_mask_report or None,
        "pose_anchor": (
            {
                "reference_view": VIEW_NAMES[0],
                "operation": (
                    "uniform_similarity_about_model_head_then_rigid_head_anchor"
                    if scale_report is not None
                    else "rigid_head_anchor"
                ),
                "formula": (
                    "model_head_T_camera = inverse(model_reference_T_head) @ "
                    "model_reference_T_camera; "
                    "model_head_T_camera.translation *= similarity_scale; "
                    "world_T_camera = calibrated_world_T_head @ model_head_T_camera"
                ),
                "world_T_model_reference": world_t_model_reference.tolist(),
            }
            if world_t_model_reference is not None
            else None
        ),
        "views": [],
    }
    cam_translations = []
    npz["pose_export_similarity_scale"] = np.asarray(
        similarity_scale, dtype=np.float32
    )
    npz["pose_export_depth_offset_m"] = np.asarray(depth_offset_m, dtype=np.float32)

    for view_idx, pred in enumerate(outputs):
        name = VIEW_NAMES[view_idx]
        model_depthmap = pred["depth_z"][0].squeeze(-1)  # (H, W), already metric-head scaled
        depthmap = model_depthmap * similarity_scale + depth_offset_m
        intrinsics = pred["intrinsics"][0]  # (3, 3)
        model_cam_pose = pred["camera_poses"][0]
        # Keep network depth and network relative camera geometry self-consistent.
        # In the default metric mode one uniform similarity correction is applied
        # to both depth and head-relative camera translations, followed by a rigid
        # head anchor. This preserves the model's multi-view alignment.
        cam_pose = torch.as_tensor(
            export_pose_arrays[view_idx],
            dtype=model_cam_pose.dtype,
            device=model_cam_pose.device,
        )

        pts3d, valid_mask = depthmap_to_world_frame(depthmap, intrinsics, cam_pose)

        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask = mask & valid_mask.cpu().numpy()
        # Drop the robot's own body from the output whether or not it was hidden
        # from the model.  Blanking the input is an experiment; removing the
        # robot from the exported cloud is always right, because those points
        # are the robot, not the scene it has to avoid.
        if self_masks and name in self_masks:
            robot_pixels = resample_mask_to(self_masks[name], mask.shape)
            removed = int((mask & robot_pixels).sum())
            mask = mask & ~robot_pixels
            self_mask_report[name] = {
                "removed_points": removed,
                "mask_coverage_fraction": float(robot_pixels.mean()),
                "blanked_before_inference": bool(use_self_mask_input),
            }
        pts3d_np = pts3d.cpu().numpy()
        image_np = pred["img_no_norm"][0].cpu().numpy()  # (H, W, 3) in [0, 1]
        depth_np = depthmap.cpu().numpy()
        conf_np = pred["conf"][0].squeeze(-1).cpu().numpy() if "conf" in pred else None
        K_np = intrinsics.cpu().numpy()
        pose_np = cam_pose.cpu().numpy()
        model_pose_np = model_cam_pose.cpu().numpy()
        model_metric_scaling_factor = None
        if "metric_scaling_factor" in pred:
            model_metric_scaling_factor = float(
                pred["metric_scaling_factor"].detach().cpu().reshape(-1)[0].item()
            )

        world_points_list.append(pts3d_np)
        images_list.append(image_np)
        masks_list.append(mask)

        # npz payload
        npz[f"{name}_depth_z"] = depth_np.astype(np.float32)
        npz[f"{name}_intrinsics"] = K_np.astype(np.float32)
        npz[f"{name}_camera_pose"] = pose_np.astype(np.float32)
        npz[f"{name}_model_camera_pose_head_reference"] = model_pose_np.astype(
            np.float32
        )
        if model_metric_scaling_factor is not None:
            npz[f"{name}_model_metric_scaling_factor"] = np.asarray(
                model_metric_scaling_factor, dtype=np.float32
            )
        if calibrated_pose_arrays is not None:
            npz[f"{name}_calibrated_input_camera_pose"] = calibrated_pose_arrays[
                view_idx
            ].astype(np.float32)
        npz[f"{name}_mask"] = mask
        npz[f"{name}_pts3d"] = pts3d_np.astype(np.float32)
        npz[f"{name}_img"] = np.clip(image_np * 255.0, 0.0, 255.0).astype(np.uint8)
        if conf_np is not None:
            npz[f"{name}_conf"] = conf_np.astype(np.float32)

        # summary stats over valid pixels
        valid_pct = 100.0 * float(mask.mean())
        d_valid = depth_np[mask]
        if d_valid.size > 0:
            d_min, d_med, d_max = (
                float(d_valid.min()),
                float(np.median(d_valid)),
                float(d_valid.max()),
            )
        else:
            d_min = d_med = d_max = float("nan")
        if conf_np is not None and mask.sum() > 0:
            c_valid = conf_np[mask]
            c_mean, c_med = float(c_valid.mean()), float(np.median(c_valid))
        else:
            c_mean = c_med = float("nan")

        trans = pose_np[:3, 3]
        cam_translations.append(trans)
        pose_prior_error = None
        export_vs_calibrated = None
        if pose_contract is not None:
            input_pose_head_reference = (
                np.linalg.inv(calibrated_pose_arrays[0])
                @ calibrated_pose_arrays[view_idx]
            )
            model_vs_input = pose_delta(input_pose_head_reference, model_pose_np)
            pose_prior_error = {
                "model_vs_input_translation_m": model_vs_input["translation_m"],
                "model_vs_input_rotation_deg": model_vs_input["rotation_deg"],
            }
            export_vs_calibrated = pose_delta(
                calibrated_pose_arrays[view_idx], pose_np
            )

        summary["views"].append(
            {
                "name": name,
                "resolution": [int(mask.shape[0]), int(mask.shape[1])],
                "valid_pixel_pct": round(valid_pct, 2),
                "conf_mean": round(c_mean, 4),
                "conf_median": round(c_med, 4),
                "camera_translation": [round(float(x), 4) for x in trans],
                "pose_prior_diagnostic": pose_prior_error,
                "export_pose_vs_calibrated_input": export_vs_calibrated,
                "model_metric_scaling_factor": model_metric_scaling_factor,
                "post_model_similarity_scale": similarity_scale,
                "depth_min": round(d_min, 4),
                "depth_median": round(d_med, 4),
                "depth_max": round(d_max, 4),
            }
        )
        print(
            f"  {name:11s} valid={valid_pct:5.1f}% conf(mean/med)="
            f"{c_mean:.3f}/{c_med:.3f} depth[min/med/max]="
            f"{d_min:.2f}/{d_med:.2f}/{d_max:.2f} t={np.round(trans,3)}"
        )

    # inter-camera baselines (meters)
    baselines = {}
    for i in range(len(cam_translations)):
        for j in range(i + 1, len(cam_translations)):
            d = float(np.linalg.norm(cam_translations[i] - cam_translations[j]))
            key = f"{VIEW_NAMES[i]}__{VIEW_NAMES[j]}"
            baselines[key] = round(d, 4)
    summary["baselines_m"] = baselines
    print(f"  baselines(m): {baselines}")

    # Stack for GLB (all views share the unified resolution after preprocess_inputs)
    world_points = np.stack(world_points_list, axis=0)
    images = np.stack(images_list, axis=0)
    final_masks = np.stack(masks_list, axis=0)

    # Optional export filters (stackable); geometry in views.npz stays unfiltered.
    if any(f is not None for f in (max_radius, bbox, min_conf)):
        conf_stack = (
            np.stack([npz[f"{n}_conf"] for n in VIEW_NAMES], axis=0)
            if f"{VIEW_NAMES[0]}_conf" in npz
            else None
        )
        keep = build_filter_mask(
            world_points, conf_stack, max_radius=max_radius, bbox=bbox, min_conf=min_conf
        )
        n_before = int(final_masks.sum())
        final_masks = final_masks & keep
        print(
            f"  export filter (max_radius={max_radius}, bbox={bbox}, min_conf={min_conf}): "
            f"{n_before} -> {int(final_masks.sum())} points"
        )
        summary["export_filter"] = {
            "max_radius": max_radius,
            "bbox": bbox,
            "min_conf": min_conf,
            "points_before": n_before,
            "points_after": int(final_masks.sum()),
        }

    predictions = {
        "world_points": world_points,
        "images": images,
        "final_masks": final_masks,
    }
    timings["reconstruction_postprocess"] = time.perf_counter() - section_started_at
    _print_timing(
        "reconstruction_postprocess", timings["reconstruction_postprocess"]
    )

    # GLB: colored point cloud of all 3 views merged
    section_started_at = time.perf_counter()
    scene_glb = predictions_to_glb(predictions, as_mesh=False)
    glb_path = os.path.join(out_dir, "scene.glb")
    scene_glb.export(glb_path)

    # PLY: same masked points
    all_pts = world_points.reshape(-1, 3)[final_masks.reshape(-1)]
    all_cols = (images.reshape(-1, 3)[final_masks.reshape(-1)] * 255.0).astype(np.uint8)
    pc = trimesh.PointCloud(vertices=all_pts, colors=all_cols)
    ply_path = os.path.join(out_dir, "scene.ply")
    pc.export(ply_path)
    timings["scene_glb_ply_export"] = time.perf_counter() - section_started_at
    _print_timing("scene_glb_ply_export", timings["scene_glb_ply_export"])

    # NPZ
    section_started_at = time.perf_counter()
    npz_path = os.path.join(out_dir, "views.npz")
    np.savez_compressed(npz_path, **npz)
    timings["views_npz_write"] = time.perf_counter() - section_started_at
    _print_timing("views_npz_write", timings["views_npz_write"])

    # summary
    section_started_at = time.perf_counter()
    summary["num_points"] = int(all_pts.shape[0])
    copied_provenance = []
    source_dir = Path(undist_root) / capture
    for filename in (*PROVENANCE_FILES, "pipeline_preprocess_manifest.json", POSES_FILE):
        source = source_dir / filename
        if source.is_file():
            shutil.copy2(source, Path(out_dir) / filename)
            copied_provenance.append(filename)
    summary["copied_provenance_files"] = copied_provenance
    export_pose_document = {
        "frame_convention": "opencv_rdf_cam2world",
        "matrix_direction": "camera_to_world",
        "world_frame": (
            pose_contract["world_frame"]
            if pose_contract is not None
            else "mapanything_model_reference_arbitrary_scale"
        ),
        "translation_unit": (
            "meter" if pose_contract is not None else "arbitrary_model_scale"
        ),
        "pose_export_mode_requested": pose_export_mode,
        "pose_export_mode_effective": effective_pose_mode,
        "similarity_scale_correction": scale_metadata,
        "metric_depth_diagnostic": depth_diagnostic,
        "self_occlusion_mask": self_mask_report or None,
        "anchor": summary["pose_anchor"],
        "poses": {
            name: export_pose_arrays[index].tolist()
            for index, name in enumerate(VIEW_NAMES)
        },
    }
    export_pose_filename = "camera_poses_used_for_export.json"
    with open(Path(out_dir) / export_pose_filename, "w") as f:
        json.dump(export_pose_document, f, indent=2)
    summary["export_pose_document"] = export_pose_filename
    timings["metadata_export"] = time.perf_counter() - section_started_at
    timings["capture_total"] = time.perf_counter() - capture_started_at
    summary["timings_seconds"] = {
        name: _timing_value(seconds) for name, seconds in timings.items()
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _print_timing("metadata_export", timings["metadata_export"])
    _print_timing("capture_total", timings["capture_total"])

    print(
        f"  saved: {glb_path} ({os.path.getsize(glb_path)} B), "
        f"{ply_path} ({os.path.getsize(ply_path)} B), {npz_path}, summary.json | "
        f"{all_pts.shape[0]} points"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default=DEFAULT_UNDIST_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--captures",
        nargs="*",
        default=None,
        help="Capture folder names; omit to auto-discover preprocessed captures",
    )
    parser.add_argument(
        "--allow-missing-poses",
        action="store_true",
        help="Allow intentional pose-free, arbitrary-scale inference",
    )
    parser.add_argument(
        "--ignore-poses",
        action="store_true",
        help=(
            "Ignore camera_poses_opencv_cam2world.json even when present; run with "
            "RGB/intrinsics only and let the model estimate arbitrary-scale poses"
        ),
    )
    parser.add_argument(
        "--pose-export-mode",
        choices=POSE_EXPORT_MODES,
        default=DEFAULT_POSE_EXPORT_MODE,
        help=(
            "Final geometry when metric poses are supplied. Default estimates one "
            "uniform scale from calibrated/model camera baselines, applies it to "
            "model depth and relative translations, then anchors the calibrated "
            "head. depth-scaled instead fits that scale per pixel against a metric "
            "depth camera and falls back to the baseline fit if the depth fit cannot "
            "be defended. Other choices preserve the unscaled model geometry or "
            "legacy hybrid."
        ),
    )
    parser.add_argument(
        "--depth-input",
        action="store_true",
        help=(
            "Feed the registered metric depth to the model as a per-view depth_z "
            "prior instead of only using it afterwards. Requires a robot with a "
            "metric depth camera (G2)."
        ),
    )
    parser.add_argument(
        "--self-mask-input",
        action="store_true",
        help=(
            "Blank the robot's own pixels before inference, to test whether the "
            "gripper filling the wrist view distorts the estimate. Robot points "
            "are removed from the exported cloud regardless of this flag."
        ),
    )
    parser.add_argument(
        "--depth-holdout",
        type=float,
        default=0.0,
        help=(
            "Fraction of depth pixels to withhold from --depth-input so the "
            "diagnostic still has pixels the model never saw. Without it, agreement "
            "on a fed-in view is circular and proves nothing."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate/load/preprocess inputs without loading the model or using CUDA",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minibatch_size", type=int, default=None)
    parser.add_argument(
        "--fast-inference",
        action="store_true",
        help=(
            "Run all dense prediction heads together for speed at higher peak VRAM; "
            "CUDA OOM automatically retries memory-efficient inference"
        ),
    )
    parser.add_argument(
        "--max_radius",
        type=float,
        default=None,
        help="Export filter: keep points within this distance (m) from world origin",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=6,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Export filter: keep points inside this world-frame box",
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=None,
        help="Export filter: keep points with confidence >= this value",
    )
    args = parser.parse_args()
    run_started_at = time.perf_counter()
    captures = resolve_captures(args.input_root, args.captures, preprocessed=True)
    filter_kwargs = dict(
        max_radius=args.max_radius, bbox=args.bbox, min_conf=args.min_conf
    )

    if args.validate_only:
        print(f"Validating captures: {', '.join(captures)}")
        for capture in captures:
            print(f"\n===== {capture} =====")
            views, contract = load_views(
                capture,
                args.input_root,
                args.allow_missing_poses,
                ignore_poses=args.ignore_poses,
            )
            report = validate_preprocessed_views(views, preprocess_inputs(views))
            print(
                json.dumps(
                    {
                        "pose_contract": contract,
                        "pose_export_mode_requested": args.pose_export_mode,
                        **report,
                    },
                    indent=2,
                )
            )
        print("\nValidation complete; model was not loaded.")
        return

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Fix the NVIDIA driver/runtime or run --validate-only."
        )
    print("Loading MapAnything model...")
    model_load_started_at = time.perf_counter()
    model = MapAnything.from_pretrained("facebook/map-anything").to(args.device)
    model.eval()
    model_load_seconds = time.perf_counter() - model_load_started_at
    print(f"[TIMING] model_load: {model_load_seconds:.3f} s")

    all_summaries = {}
    for capture in captures:
        try:
            all_summaries[capture] = run_capture(
                model,
                capture,
                minibatch_size=args.minibatch_size,
                undist_root=args.input_root,
                output_root=args.output_root,
                allow_missing_poses=args.allow_missing_poses,
                ignore_poses=args.ignore_poses,
                pose_export_mode=args.pose_export_mode,
                use_depth_input=args.depth_input,
                depth_holdout=args.depth_holdout,
                use_self_mask_input=args.self_mask_input,
                memory_efficient_inference=not args.fast_inference,
                **filter_kwargs,
            )
        except torch.cuda.OutOfMemoryError as e:
            print(
                f"  OOM on {capture}: {e}\n"
                "  retrying with memory-efficient inference and minibatch_size=1"
            )
            torch.cuda.empty_cache()
            all_summaries[capture] = run_capture(
                model,
                capture,
                minibatch_size=1,
                undist_root=args.input_root,
                output_root=args.output_root,
                allow_missing_poses=args.allow_missing_poses,
                ignore_poses=args.ignore_poses,
                pose_export_mode=args.pose_export_mode,
                use_depth_input=args.depth_input,
                depth_holdout=args.depth_holdout,
                use_self_mask_input=args.self_mask_input,
                memory_efficient_inference=True,
                **filter_kwargs,
            )

    total_seconds = time.perf_counter() - run_started_at
    print(f"\nAll done.\n[TIMING] run_inference_total: {total_seconds:.3f} s")


if __name__ == "__main__":
    main()
