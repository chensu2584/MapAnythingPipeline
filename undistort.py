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
    resolve_captures,
    load_intrinsics,
    validate_pose_document,
)

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


def build_preprocess_cache_record(cap_in, ignore_poses=False):
    """Fingerprint every source file that affects reusable preprocessing output."""

    cap_in = Path(cap_in)
    required = [
        *(cap_in / f"{name}.png" for name in IMAGE_TO_INTRINSIC),
        *(cap_in / filename for filename in IMAGE_TO_INTRINSIC.values()),
    ]
    optional = [
        cap_in / POSES_FILE,
        *(cap_in / filename for filename in PROVENANCE_FILES),
    ]
    inputs = {}
    digest = hashlib.sha256()
    digest.update(f"undistort-cache-v{PREPROCESS_CACHE_VERSION}\n".encode())
    digest.update(f"ignore_poses={bool(ignore_poses)}\n".encode())
    for path in (*required, *(path for path in optional if path.is_file())):
        if not path.is_file():
            raise FileNotFoundError(f"Missing preprocessing input: {path}")
        relative = path.relative_to(cap_in).as_posix()
        file_digest = _sha256(path)
        size = path.stat().st_size
        inputs[relative] = {"sha256": file_digest, "size": size}
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\0")
    return {
        "version": PREPROCESS_CACHE_VERSION,
        "key": digest.hexdigest(),
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

    required_outputs = [
        *(cap_out / f"{name}.png" for name in IMAGE_TO_INTRINSIC),
        *(cap_out / f"{name}_K.json" for name in IMAGE_TO_INTRINSIC),
    ]
    input_names = cache_record["inputs"]
    required_outputs.extend(
        cap_out / filename for filename in PROVENANCE_FILES if filename in input_names
    )
    source_pose_present = POSES_FILE in input_names
    pose_out = cap_out / POSES_FILE
    if source_pose_present and not cache_record["ignore_poses"]:
        required_outputs.append(pose_out)
    elif pose_out.exists():
        return None
    if not all(path.is_file() for path in required_outputs):
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


def process_capture(
    capture,
    data_root,
    output_root,
    allow_missing_poses=False,
    ignore_poses=False,
    reuse_existing=False,
):
    cap_in = Path(data_root) / capture
    cap_out = Path(output_root) / "undistorted" / capture
    print(f"\n=== {capture} ===")

    cache_record = build_preprocess_cache_record(cap_in, ignore_poses=ignore_poses)
    if reuse_existing:
        cached_manifest = reusable_preprocess_manifest(cap_out, cache_record)
        if cached_manifest is not None:
            print("  reuse: undistorted images and metadata are unchanged; skipping remap")
            return cached_manifest

    poses_src = cap_in / POSES_FILE
    pose_contract = None
    if ignore_poses:
        print("  pose: explicitly ignored; preparing pose-free model input")
    elif poses_src.is_file():
        _, pose_contract = validate_pose_document(poses_src)
    elif not allow_missing_poses:
        raise FileNotFoundError(
            f"{poses_src} is required for metric reconstruction; "
            "use --allow-missing-poses only for intentional pose-free inference"
        )

    # Validate all image/intrinsic pairs before creating partial output.
    loaded = {}
    for name, intr_file in IMAGE_TO_INTRINSIC.items():
        img_path = cap_in / f"{name}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        H, W = img.shape[:2]
        K, dist = load_K_dist(cap_in / intr_file, W, H)
        loaded[name] = (img, K, dist, W, H)

    cap_out.mkdir(parents=True, exist_ok=True)
    poses_out = cap_out / POSES_FILE
    if pose_contract is not None:
        shutil.copy2(poses_src, poses_out)
        print(
            f"  pose: {pose_contract['frame_convention']}, "
            f"world={pose_contract['world_frame']}, unit=m"
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

    view_results = {}
    for name, (img, K, dist, W, H) in loaded.items():
        intr_file = IMAGE_TO_INTRINSIC[name]
        undist, adjK, roi = undistort_image(img, K, dist)
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
            "source_intrinsic": intr_file,
            "distortion_removed": True,
        }
        with (cap_out / f"{name}_K.json").open("w", encoding="utf-8") as f:
            json.dump(k_document, f, indent=2)
        view_results[name] = k_document
        cx_off = adjK[0, 2] - ow / 2.0
        cy_off = adjK[1, 2] - oh / 2.0
        print(
            f"  {name:11s} in {W}x{H} -> out {ow}x{oh} | "
            f"Fx={adjK[0,0]:.1f} Cx={adjK[0,2]:.1f}({cx_off:+.1f} from center) "
            f"Cy={adjK[1,2]:.1f}({cy_off:+.1f} from center) roi={roi}"
        )

    preprocess_manifest = {
        "schema_version": 2,
        "capture": capture,
        "source_capture_dir": str(cap_in.resolve()),
        "pose_contract": pose_contract,
        "pose_copied_unchanged": pose_contract is not None,
        "pose_input_mode": "ignored" if ignore_poses else "metric_if_available",
        "source_pose_was_present": poses_src.is_file(),
        "copied_provenance_files": copied_metadata,
        "views": view_results,
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
    args = parser.parse_args()
    captures = resolve_captures(args.data_root, args.captures, preprocessed=False)
    print(f"Input root: {Path(args.data_root).resolve()}")
    print(f"Captures: {', '.join(captures)}")
    for capture in captures:
        process_capture(
            capture,
            args.data_root,
            args.output_root,
            allow_missing_poses=args.allow_missing_poses,
            ignore_poses=args.ignore_poses,
            reuse_existing=args.reuse_existing,
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
