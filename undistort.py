#!/usr/bin/env python
"""
Step A: Undistortion preprocessing for G2 robot camera captures.

For each capture folder and each of the 3 RGB images (head, hand_left, hand_right):
  1. Build K from the matching intrinsics JSON, dist = [k1, k2, p1, p2, k3].
  2. getOptimalNewCameraMatrix(alpha=0) -> initUndistortRectifyMap -> remap -> crop to ROI.
  3. Shift the new K principal point by the ROI offset.
  4. Save undistorted PNG + <name>_K.json (adjusted newK) to outputs/undistorted/<capture>/.
Also copies camera_poses_opencv_cam2world.json (extrinsics) through unchanged.
"""

import json
import os
import shutil

import cv2
import numpy as np

# Machine-specific roots; override via env vars instead of editing code:
#   G2_DATA_ROOT: dir containing the capture folders (g_1_Test_*)
#   G2_OUT_ROOT:  dir where all pipeline outputs are written
TEST_DATA = os.path.expanduser(
    os.environ.get("G2_DATA_ROOT", "~/MapAnything/MapAnythingTestData1")
)
OUT_ROOT = os.path.join(
    os.path.expanduser(os.environ.get("G2_OUT_ROOT", "~/MapAnything/outputs")),
    "undistorted",
)

CAPTURES = [
    "g_1_Test_1",
    "g_1_Test_2",
    "g_1_Test_3",
    "g_1_Test_4",
]

# image name -> intrinsics json file
IMAGE_TO_INTRINSIC = {
    "head": "intrinsic_head_front_rgb.json",
    "hand_left": "intrinsic_hand_left_rgb.json",
    "hand_right": "intrinsic_hand_right_rgb.json",
}

# Per-capture camera extrinsics (OpenCV cam2world 4x4, world frame = robot "end").
# Copied through unchanged: undistortion only modifies K, never the pose.
POSES_FILE = "camera_poses_opencv_cam2world.json"


def load_K_dist(intrinsic_path):
    with open(intrinsic_path) as f:
        c = json.load(f)
    K = np.array(
        [
            [c["Fx"], 0.0, c["Cx"]],
            [0.0, c["Fy"], c["Cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.array([c["k1"], c["k2"], c["p1"], c["p2"], c["k3"]], dtype=np.float64)
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


def main():
    for capture in CAPTURES:
        cap_in = os.path.join(TEST_DATA, capture)
        cap_out = os.path.join(OUT_ROOT, capture)
        os.makedirs(cap_out, exist_ok=True)
        print(f"\n=== {capture} ===")
        poses_src = os.path.join(cap_in, POSES_FILE)
        if os.path.isfile(poses_src):
            shutil.copy(poses_src, os.path.join(cap_out, POSES_FILE))
            print(f"  copied {POSES_FILE}")
        else:
            print(f"  WARNING: {POSES_FILE} not found; inference will run without poses")
        for name, intr_file in IMAGE_TO_INTRINSIC.items():
            img_path = os.path.join(cap_in, f"{name}.png")
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
            if img is None:
                raise FileNotFoundError(img_path)
            K, dist = load_K_dist(os.path.join(cap_in, intr_file))
            H, W = img.shape[:2]
            undist, adjK, roi = undistort_image(img, K, dist)
            oh, ow = undist.shape[:2]

            out_png = os.path.join(cap_out, f"{name}.png")
            cv2.imwrite(out_png, undist)
            with open(os.path.join(cap_out, f"{name}_K.json"), "w") as f:
                json.dump(
                    {
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
                    },
                    f,
                    indent=2,
                )
            cx_off = adjK[0, 2] - ow / 2.0
            cy_off = adjK[1, 2] - oh / 2.0
            print(
                f"  {name:11s} in {W}x{H} -> out {ow}x{oh} | "
                f"Fx={adjK[0,0]:.1f} Cx={adjK[0,2]:.1f}({cx_off:+.1f} from center) "
                f"Cy={adjK[1,2]:.1f}({cy_off:+.1f} from center) roi={roi}"
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

  python undistort.py
  python run_inference.py --captures g_1_Test_1 --max_radius 2.0
  python voxelize.py --captures g_1_Test_1 --voxel_size 0.02 --max_radius 2.0
"""