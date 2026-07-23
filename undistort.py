#!/usr/bin/env python
"""
Step A: Undistortion preprocessing for G1/G2 robot camera captures.

For each capture folder and each of the 3 RGB images (head, hand_left, hand_right):
  1. Build K from the matching intrinsics JSON, dist = [k1, k2, p1, p2, k3].
  2. getOptimalNewCameraMatrix(alpha=0) -> initUndistortRectifyMap -> remap -> crop to ROI.
  3. Shift the new K principal point by the ROI offset.
  4. Save undistorted PNG + <name>_K.json (adjusted newK) to outputs/undistorted/<capture>/.
Also copies camera_poses_opencv_cam2world.json (extrinsics) through unchanged.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from capture_contract import (
    IMAGE_TO_INTRINSIC,
    POSES_FILE,
    PROVENANCE_FILES,
    load_intrinsics,
    validate_pose_document,
)
from depth_tools import register_depth_to_camera
from robot_profiles import detect_profile, get_profile

DEPTH_FILE = "registered_depth.npz"
CAPTURE_STATE_FILE = "capture_state.json"

# Machine-specific roots; override via env vars instead of editing code:
#   G2_DATA_ROOT: dir containing the capture folders (g_1_Test_*)
#   G2_OUT_ROOT:  dir where all pipeline outputs are written
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = os.path.expanduser(
    os.environ.get("G2_DATA_ROOT", str(PROJECT_ROOT / "TestData"))
)
DEFAULT_OUTPUT_ROOT = os.path.expanduser(
    os.environ.get("G2_OUT_ROOT", str(PROJECT_ROOT / "outputs"))
)
PREPROCESS_CACHE_VERSION = 1


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_preprocess_cache_record(cap_in, ignore_poses=False, profile_name="g1"):
    """Fingerprint every source file that affects reusable preprocessing output.

    Robot layouts differ in which files exist, so every non-hidden regular file
    in the capture folder is fingerprinted rather than a per-robot allow-list.
    That is layout-agnostic and strictly more conservative: any source change at
    all invalidates the cache.
    """

    cap_in = Path(cap_in)
    if not cap_in.is_dir():
        raise FileNotFoundError(f"Capture folder does not exist: {cap_in}")
    sources = sorted(
        path
        for path in cap_in.rglob("*")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(cap_in).parts)
    )
    if not sources:
        raise FileNotFoundError(f"Capture folder has no readable inputs: {cap_in}")
    inputs = {}
    digest = hashlib.sha256()
    digest.update(f"undistort-cache-v{PREPROCESS_CACHE_VERSION}\n".encode())
    digest.update(f"profile={profile_name}\n".encode())
    digest.update(f"ignore_poses={bool(ignore_poses)}\n".encode())
    for path in sources:
        relative = path.relative_to(cap_in).as_posix()
        file_digest = _sha256(path)
        inputs[relative] = {"sha256": file_digest, "size": path.stat().st_size}
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\0")
    return {
        "version": PREPROCESS_CACHE_VERSION,
        "key": digest.hexdigest(),
        "profile": profile_name,
        "ignore_poses": bool(ignore_poses),
        "inputs": inputs,
    }


def reusable_preprocess_manifest(cap_out, cache_record):
    """Return a complete matching manifest, otherwise ``None``."""

    cap_out = Path(cap_out)
    manifest_path = cap_out / "pipeline_preprocess_manifest.json"
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or manifest.get("cache") != cache_record:
        return None

    # Trust the manifest's own record of what it wrote rather than re-deriving
    # the expected file set, which differs per robot layout.
    written = manifest.get("written_outputs")
    if not isinstance(written, list) or not written:
        return None
    if not all((cap_out / name).is_file() for name in written):
        return None
    pose_out = cap_out / POSES_FILE
    if cache_record["ignore_poses"] and pose_out.exists():
        return None
    return manifest


def load_K_dist(intrinsic_path, image_width=None, image_height=None):
    K, dist, _ = load_intrinsics(intrinsic_path, image_width, image_height)
    return K, dist


def undistort_image(img, K, dist):
    H, W = img.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), alpha=0)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, newK, (W, H), cv2.CV_32FC1
    )
    undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undist = undist[y : y + rh, x : x + rw]
    else:
        # Degenerate ROI (can happen if alpha=0 leaves no valid rect); keep full image.
        x, y, rw, rh = 0, 0, W, H
    # Shift principal point by the crop offset.
    adjK = newK.copy()
    adjK[0, 2] -= x
    adjK[1, 2] -= y
    return undist, adjK, (x, y, rw, rh)


def build_pose_document(raw, profile):
    """Synthesise the canonical pose document from an already-metric capture.

    G1 captures ship this file; G2 carries the same information inside its
    extrinsics document, so it is rewritten here in one shared format instead of
    teaching every downstream stage a second contract.
    """
    poses = {}
    for name in profile.view_names:
        matrix = raw.views[name].base_T_cam
        if matrix is None:
            return None
        poses[name] = np.asarray(matrix, dtype=np.float64).tolist()
    return {
        "frame_convention": profile.pose_frame_convention,
        "world_frame": profile.world_frame,
        "translation_unit": "meter",
        "extrinsic_direction": profile.extrinsic_direction,
        "camera_axes": "OpenCV RDF: +X right, +Y down, +Z forward",
        "matrix_direction": "camera_to_world",
        "poses": poses,
        "derived_from": raw.provenance.get("extrinsics_document"),
        "robot_profile": profile.name,
    }


def write_registered_depth(cap_out, raw, profile, view_results, splat_radius=0):
    """Reproject each metric depth map into its undistorted colour view."""
    if not raw.depths:
        return None
    payload = {}
    report = {}
    for view_name, depth in raw.depths.items():
        target = view_results.get(view_name)
        colour = raw.views.get(view_name)
        if target is None or colour is None or colour.base_T_cam is None:
            continue
        K_target = np.asarray(target["K"], dtype=np.float64)
        depth_z, valid, stats = register_depth_to_camera(
            depth.depth_m,
            depth.valid,
            K_source=depth.K,
            base_T_source=depth.base_T_cam,
            K_target=K_target,
            base_T_target=colour.base_T_cam,
            target_shape=(target["height"], target["width"]),
            splat_radius=splat_radius,
        )
        payload[f"{view_name}_depth_z"] = depth_z.astype(np.float32)
        payload[f"{view_name}_depth_valid"] = valid
        stats.update(
            {
                "source_camera": depth.name,
                "source_shape": [int(v) for v in depth.depth_m.shape],
                "target_shape": [int(target["height"]), int(target["width"])],
                "unit_scale_to_m": depth.unit_scale_to_m,
                "invalid_source_values": list(depth.invalid_values),
            }
        )
        report[view_name] = stats
    if not payload:
        return None
    payload["views"] = np.asarray(sorted(report))
    payload["world_frame"] = np.asarray(profile.world_frame)
    payload["translation_unit"] = np.asarray("meter")
    payload["depth_convention"] = np.asarray(
        "z_along_target_camera_optical_axis; NaN marks invalid"
    )
    np.savez_compressed(cap_out / DEPTH_FILE, **payload)
    return report


def process_capture(
    capture,
    data_root,
    output_root,
    allow_missing_poses=False,
    ignore_poses=False,
    reuse_existing=False,
    profile=None,
    depth_splat_radius=0,
):
    cap_in = Path(data_root) / capture
    cap_out = Path(output_root) / "undistorted" / capture
    profile = profile or detect_profile(data_root, capture)
    print(f"\n=== {capture} [{profile.name}] ===")

    cache_record = build_preprocess_cache_record(
        cap_in, ignore_poses=ignore_poses, profile_name=profile.name
    )
    if reuse_existing:
        cached_manifest = reusable_preprocess_manifest(cap_out, cache_record)
        if cached_manifest is not None:
            print("  reuse: undistorted images and metadata are unchanged; skipping remap")
            return cached_manifest

    # Load and validate the whole capture before creating any partial output.
    raw = profile.load(data_root, capture)

    poses_src = cap_in / POSES_FILE
    pose_contract = None
    pose_document = None
    if ignore_poses:
        print("  pose: explicitly ignored; preparing pose-free model input")
    elif poses_src.is_file():
        _, pose_contract = validate_pose_document(poses_src)
    else:
        pose_document = build_pose_document(raw, profile)
        if pose_document is None and not allow_missing_poses:
            raise FileNotFoundError(
                f"{poses_src} is required for metric reconstruction and {capture} carries "
                "no per-view extrinsics; use --allow-missing-poses only for intentional "
                "pose-free inference"
            )

    cap_out.mkdir(parents=True, exist_ok=True)
    written_outputs = []
    poses_out = cap_out / POSES_FILE
    if pose_contract is not None:
        shutil.copy2(poses_src, poses_out)
        written_outputs.append(POSES_FILE)
        print(
            f"  pose: {pose_contract['frame_convention']}, "
            f"world={pose_contract['world_frame']}, unit=m (copied unchanged)"
        )
    elif pose_document is not None:
        with poses_out.open("w", encoding="utf-8") as handle:
            json.dump(pose_document, handle, indent=2)
        # Re-read through the shared validator so a synthesised document has to
        # satisfy exactly the same contract as a shipped one.
        _, pose_contract = validate_pose_document(poses_out)
        written_outputs.append(POSES_FILE)
        print(
            f"  pose: {pose_contract['frame_convention']}, "
            f"world={pose_contract['world_frame']}, unit=m (derived from capture extrinsics)"
        )
    else:
        # A previous metric preprocessing run may have left a pose file in the
        # same output folder.  Explicit pose-free mode must not consume it.
        if ignore_poses and poses_out.is_file():
            poses_out.unlink()
        print("  WARNING: no pose file; output scale/frame will not be physically anchored")

    copied_metadata = []
    for filename in PROVENANCE_FILES:
        source = cap_in / filename
        if source.is_file():
            shutil.copy2(source, cap_out / filename)
            copied_metadata.append(filename)
            written_outputs.append(filename)

    view_results = {}
    for name in profile.view_names:
        view = raw.views[name]
        H, W = view.image_bgr.shape[:2]
        undist, adjK, roi = undistort_image(view.image_bgr, view.K, view.dist)
        oh, ow = undist.shape[:2]

        out_png = cap_out / f"{name}.png"
        if not cv2.imwrite(str(out_png), undist):
            raise OSError(f"OpenCV failed to write {out_png}")
        k_document = {
            "K": adjK.tolist(),
            "Fx": float(adjK[0, 0]),
            "Fy": float(adjK[1, 1]),
            "Cx": float(adjK[0, 2]),
            "Cy": float(adjK[1, 2]),
            "width": int(ow),
            "height": int(oh),
            "orig_width": int(W),
            "orig_height": int(H),
            "roi": [int(v) for v in roi],
            "source_intrinsic": view.intrinsic_source,
            "distortion_removed": True,
        }
        with (cap_out / f"{name}_K.json").open("w", encoding="utf-8") as f:
            json.dump(k_document, f, indent=2)
        view_results[name] = k_document
        written_outputs.extend([f"{name}.png", f"{name}_K.json"])
        cx_off = adjK[0, 2] - ow / 2.0
        cy_off = adjK[1, 2] - oh / 2.0
        print(
            f"  {name:11s} in {W}x{H} -> out {ow}x{oh} | "
            f"Fx={adjK[0,0]:.1f} Cx={adjK[0,2]:.1f}({cx_off:+.1f} from center) "
            f"Cy={adjK[1,2]:.1f}({cy_off:+.1f} from center) roi={roi}"
        )

    depth_report = write_registered_depth(
        cap_out, raw, profile, view_results, splat_radius=depth_splat_radius
    )
    if depth_report:
        written_outputs.append(DEPTH_FILE)
        for view_name, stats in depth_report.items():
            print(
                f"  depth[{view_name}] {stats['source_camera']} -> undistorted {view_name}: "
                f"{stats['target_filled_pixels']} px "
                f"({100 * stats['target_fill_ratio']:.1f}% of frame), "
                f"behind_camera={stats['dropped_behind_camera']}, "
                f"outside_frame={stats['dropped_outside_frame']}"
            )

    if raw.joint_positions:
        state = {
            "schema_version": 1,
            "robot_profile": profile.name,
            "world_frame": profile.world_frame,
            "joint_positions_rad": raw.joint_positions,
            "source": raw.provenance.get("extrinsics_document"),
            "kinematic_validation": raw.provenance.get("kinematic_validation"),
            "synchronisation": raw.provenance.get("synchronisation"),
        }
        with (cap_out / CAPTURE_STATE_FILE).open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
        if CAPTURE_STATE_FILE not in written_outputs:
            written_outputs.append(CAPTURE_STATE_FILE)

    preprocess_manifest = {
        "schema_version": 3,
        "capture": capture,
        "robot_profile": profile.name,
        "source_capture_dir": str(cap_in.resolve()),
        "source_layout": raw.provenance.get("layout"),
        "pose_contract": pose_contract,
        "pose_copied_unchanged": poses_src.is_file() and not ignore_poses,
        "pose_derived_from_capture": pose_document is not None,
        "pose_input_mode": "ignored" if ignore_poses else "metric_if_available",
        "source_pose_was_present": poses_src.is_file(),
        "copied_provenance_files": copied_metadata,
        "views": view_results,
        "depth": depth_report,
        "capture_provenance": raw.provenance,
        "written_outputs": sorted(set(written_outputs)),
        "cache": cache_record,
    }
    with (cap_out / "pipeline_preprocess_manifest.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(preprocess_manifest, f, indent=2)
    return preprocess_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--captures",
        nargs="*",
        default=None,
        help="Capture folder names; omit to auto-discover compatible captures",
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
            "Ignore an existing pose file and remove any stale copied pose from the "
            "preprocessed output; prepares RGB/intrinsics-only model input"
        ),
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "Skip undistortion when the content fingerprint and every expected output "
            "match the previous preprocessing manifest"
        ),
    )
    parser.add_argument(
        "--robot",
        default=None,
        help="Robot profile (g1/g2); omit to detect it from the capture layout",
    )
    parser.add_argument(
        "--depth-splat-radius",
        type=int,
        default=0,
        help=(
            "Widen each reprojected depth sample into a square of this radius to "
            "close forward-mapping holes; trades edge bleeding for coverage"
        ),
    )
    args = parser.parse_args()
    profile = (
        get_profile(args.robot) if args.robot else detect_profile(args.data_root)
    )
    captures = list(args.captures) if args.captures else profile.discover(args.data_root)
    if not captures:
        raise FileNotFoundError(
            f"No {profile.name} captures found under {Path(args.data_root).resolve()}"
        )
    for capture in captures:
        if not capture or Path(capture).name != capture:
            raise ValueError(f"Capture must be a folder name, not a path: {capture!r}")
    print(f"Input root: {Path(args.data_root).resolve()}")
    print(f"Robot profile: {profile.name}")
    print(f"Captures: {', '.join(captures)}")
    for capture in captures:
        process_capture(
            capture,
            args.data_root,
            args.output_root,
            allow_missing_poses=args.allow_missing_poses,
            ignore_poses=args.ignore_poses,
            reuse_existing=args.reuse_existing,
            profile=profile,
            depth_splat_radius=args.depth_splat_radius,
        )
    print("\nUndistortion complete.")


if __name__ == "__main__":
    main()


"""
Usage (any machine):
  conda activate MAP   # env with numpy/opencv/torch/mapanything

  # Point the pipeline at this machine's data/output dirs (defaults shown are
  # for yizhic3's machine; on ck's machine use ~/MapAnythingTest/TestData etc.):
  export G2_DATA_ROOT=~/MapAnything/MapAnythingTestData1
  export G2_OUT_ROOT=~/MapAnything/outputs

  python undistort.py --captures g1_capture_20260715_121059
  python run_inference.py --captures g_1_Test_1 --max_radius 2.0
  python voxelize.py --captures g_1_Test_1 --voxel_size 0.02 --max_radius 2.0
"""
