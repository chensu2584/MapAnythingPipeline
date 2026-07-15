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


def process_capture(capture, data_root, output_root, allow_missing_poses=False):
    cap_in = Path(data_root) / capture
    cap_out = Path(output_root) / "undistorted" / capture
    print(f"\n=== {capture} ===")

    poses_src = cap_in / POSES_FILE
    pose_contract = None
    if poses_src.is_file():
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
    if pose_contract is not None:
        shutil.copy2(poses_src, cap_out / POSES_FILE)
        print(
            f"  pose: {pose_contract['frame_convention']}, "
            f"world={pose_contract['world_frame']}, unit=m"
        )
    else:
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
        "schema_version": 1,
        "capture": capture,
        "source_capture_dir": str(cap_in.resolve()),
        "pose_contract": pose_contract,
        "pose_copied_unchanged": pose_contract is not None,
        "copied_provenance_files": copied_metadata,
        "views": view_results,
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
