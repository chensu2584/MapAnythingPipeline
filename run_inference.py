#!/usr/bin/env python
"""
Step B: MapAnything 3D reconstruction inference on undistorted G1/G2 captures.

For each capture:
  - Load the 3 undistorted images + adjusted newK, plus validated metric OpenCV
    RDF cam2world poses. The world frame is read from the pose document.
  - Build views [{"img": HxWx3 uint8, "intrinsics": 3x3, "camera_poses": 4x4}],
    preprocess_inputs (unify to 518-set).
  - model.infer(..., memory_efficient_inference=True, use_amp, bf16, apply_mask, mask_edges).
  - Save scene.glb, scene.ply, views.npz, summary.json to outputs/<capture>/.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filter_export import build_filter_mask
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


def load_views(capture, undist_root=DEFAULT_UNDIST_ROOT, allow_missing_poses=False):
    cap_dir = Path(undist_root) / capture

    poses = None
    pose_contract = None
    poses_path = cap_dir / POSES_FILE
    if poses_path.is_file():
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
        view = {
            "img": img_rgb.astype(np.uint8),
            "intrinsics": torch.from_numpy(K),
        }
        if poses is not None:
            pose = np.asarray(poses[name], dtype=np.float32)
            view["camera_poses"] = torch.from_numpy(pose)
            view["is_metric_scale"] = torch.ones(1, dtype=torch.bool)
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
):
    out_dir = os.path.join(output_root, capture)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n===== {capture} =====")

    views, pose_contract = load_views(capture, undist_root, allow_missing_poses)
    processed = preprocess_inputs(views)
    preprocess_report = validate_preprocessed_views(views, processed)
    if pose_contract is None and any(v is not None for v in (max_radius, bbox)):
        raise ValueError("Metric radius/bbox filters require metric camera poses")

    infer_kwargs = dict(
        memory_efficient_inference=True,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )
    if minibatch_size is not None:
        infer_kwargs["minibatch_size"] = minibatch_size

    outputs = model.infer(processed, **infer_kwargs)
    if len(outputs) != len(VIEW_NAMES):
        raise RuntimeError(f"Model returned {len(outputs)} views, expected {len(VIEW_NAMES)}")

    world_points_list = []
    images_list = []
    masks_list = []

    npz = {}
    summary = {
        "capture": capture,
        "metric_pose_input": pose_contract is not None,
        "pose_contract": pose_contract,
        "preprocess_validation": preprocess_report,
        "camera_pose_used_for_export": (
            "calibrated_input_pose" if pose_contract is not None else "model_prediction"
        ),
        "views": [],
    }
    cam_translations = []
    input_head_pose = processed[0].get("camera_poses")

    for view_idx, pred in enumerate(outputs):
        name = VIEW_NAMES[view_idx]
        depthmap = pred["depth_z"][0].squeeze(-1)  # (H, W)
        intrinsics = pred["intrinsics"][0]  # (3, 3)
        model_cam_pose = pred["camera_poses"][0]
        # Input poses are conditions, not hard constraints in MapAnything. When
        # calibrated poses exist, use them for final unprojection/export so the
        # result stays exactly in the capture's declared world frame.
        cam_pose = (
            processed[view_idx]["camera_poses"][0]
            if pose_contract is not None
            else model_cam_pose
        )

        pts3d, valid_mask = depthmap_to_world_frame(depthmap, intrinsics, cam_pose)

        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask = mask & valid_mask.cpu().numpy()
        pts3d_np = pts3d.cpu().numpy()
        image_np = pred["img_no_norm"][0].cpu().numpy()  # (H, W, 3) in [0, 1]
        depth_np = depthmap.cpu().numpy()
        conf_np = pred["conf"][0].squeeze(-1).cpu().numpy() if "conf" in pred else None
        K_np = intrinsics.cpu().numpy()
        pose_np = cam_pose.cpu().numpy()
        model_pose_np = model_cam_pose.cpu().numpy()

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
        if pose_contract is not None:
            head_T_world = torch.linalg.inv(input_head_pose[0])
            input_pose_head_reference = (
                head_T_world @ processed[view_idx]["camera_poses"][0]
            ).cpu().numpy()
            delta = np.linalg.inv(input_pose_head_reference) @ model_pose_np
            cos_angle = np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            pose_prior_error = {
                "model_vs_input_translation_m": float(np.linalg.norm(delta[:3, 3])),
                "model_vs_input_rotation_deg": float(np.degrees(np.arccos(cos_angle))),
            }

        summary["views"].append(
            {
                "name": name,
                "resolution": [int(mask.shape[0]), int(mask.shape[1])],
                "valid_pixel_pct": round(valid_pct, 2),
                "conf_mean": round(c_mean, 4),
                "conf_median": round(c_med, 4),
                "camera_translation": [round(float(x), 4) for x in trans],
                "pose_prior_diagnostic": pose_prior_error,
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

    # GLB: colored point cloud of all 3 views merged
    scene_glb = predictions_to_glb(predictions, as_mesh=False)
    glb_path = os.path.join(out_dir, "scene.glb")
    scene_glb.export(glb_path)

    # PLY: same masked points
    all_pts = world_points.reshape(-1, 3)[final_masks.reshape(-1)]
    all_cols = (images.reshape(-1, 3)[final_masks.reshape(-1)] * 255.0).astype(np.uint8)
    pc = trimesh.PointCloud(vertices=all_pts, colors=all_cols)
    ply_path = os.path.join(out_dir, "scene.ply")
    pc.export(ply_path)

    # NPZ
    npz_path = os.path.join(out_dir, "views.npz")
    np.savez_compressed(npz_path, **npz)

    # summary
    summary["num_points"] = int(all_pts.shape[0])
    copied_provenance = []
    source_dir = Path(undist_root) / capture
    for filename in (*PROVENANCE_FILES, "pipeline_preprocess_manifest.json", POSES_FILE):
        source = source_dir / filename
        if source.is_file():
            shutil.copy2(source, Path(out_dir) / filename)
            copied_provenance.append(filename)
    summary["copied_provenance_files"] = copied_provenance
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

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
        "--validate-only",
        action="store_true",
        help="Validate/load/preprocess inputs without loading the model or using CUDA",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minibatch_size", type=int, default=None)
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
    captures = resolve_captures(args.input_root, args.captures, preprocessed=True)
    filter_kwargs = dict(
        max_radius=args.max_radius, bbox=args.bbox, min_conf=args.min_conf
    )

    if args.validate_only:
        print(f"Validating captures: {', '.join(captures)}")
        for capture in captures:
            print(f"\n===== {capture} =====")
            views, contract = load_views(
                capture, args.input_root, args.allow_missing_poses
            )
            report = validate_preprocessed_views(views, preprocess_inputs(views))
            print(json.dumps({"pose_contract": contract, **report}, indent=2))
        print("\nValidation complete; model was not loaded.")
        return

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Fix the NVIDIA driver/runtime or run --validate-only."
        )
    print("Loading MapAnything model...")
    model = MapAnything.from_pretrained("facebook/map-anything").to(args.device)
    model.eval()

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
                **filter_kwargs,
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM on {capture}: {e}\n  retrying with minibatch_size=1")
            torch.cuda.empty_cache()
            all_summaries[capture] = run_capture(
                model,
                capture,
                minibatch_size=1,
                undist_root=args.input_root,
                output_root=args.output_root,
                allow_missing_poses=args.allow_missing_poses,
                **filter_kwargs,
            )

    print("\nAll done.")


if __name__ == "__main__":
    main()
