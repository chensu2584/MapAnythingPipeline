#!/usr/bin/env python
"""
Reconstruction quality diagnosis against a metric depth camera.

Answers three questions that ``summary.json`` alone cannot:

  1. Is each view's depth right?  Per-ray ``model_range / measured_range``.
     One number per view, comparable across views and captures.

  2. When a view is wrong, is it wrong *along* the camera ray or *across* it?
     Radial error is a depth problem and metric depth input can fix it.
     Lateral error is a pose problem and depth input cannot fix it.
     Separating them says which lever to pull.

  3. Is the error a pure scale?  Fits ``measured = a * model + b`` per view.
     A ``b`` well above the residual means no single global scalar can fix the
     reconstruction, which is invisible to a camera-baseline scale fit.

The depth camera is the reference, not ground truth: it has its own noise and
its own mounting error.  ``--planes`` characterises it on surfaces that are
known to be flat, so its own limits are on the table beside the numbers.

Only views whose reconstruction overlaps the depth camera can be judged.  Every
statistic reports how many rays it used, and a view with too few is skipped
rather than summarised from noise.

Example:
  python diagnose_reconstruction.py \
      --session ~/MapAnythingTest/TestData/session_20260721_232012 \
      --output-root ~/MapAnythingTest/outputs_g2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from capture_contract import VIEW_NAMES
from depth_tools import depth_to_base_points, fit_scale_robust, plane_fit_report
from robot_profiles import detect_profile, get_profile

MIN_RAYS = 2000
MAX_RANGE_M = 3.0


def _camera(output_root: Path, capture: str, view: str):
    """Return (K, width, height, base_T_cam) for one exported view."""
    undistorted = output_root / "undistorted" / capture
    with (undistorted / f"{view}_K.json").open(encoding="utf-8") as handle:
        intrinsics = json.load(handle)
    poses_path = output_root / capture / "camera_poses_opencv_cam2world.json"
    if not poses_path.is_file():
        poses_path = undistorted / "camera_poses_opencv_cam2world.json"
    with poses_path.open(encoding="utf-8") as handle:
        poses = json.load(handle)["poses"]
    return (
        np.asarray(intrinsics["K"], dtype=np.float64),
        int(intrinsics["width"]),
        int(intrinsics["height"]),
        np.asarray(poses[view], dtype=np.float64),
    )


def _range_buffer(points_base, K, width, height, base_T_cam):
    """Rasterise reference points into one camera as a nearest-surface range map."""
    centre = base_T_cam[:3, 3]
    local = (points_base - centre) @ base_T_cam[:3, :3]
    z = local[:, 2]
    ahead = z > 0.05
    u = np.full(len(local), -1.0)
    v = np.full(len(local), -1.0)
    u[ahead] = local[ahead, 0] / z[ahead] * K[0, 0] + K[0, 2]
    v[ahead] = local[ahead, 1] / z[ahead] * K[1, 1] + K[1, 2]
    inside = ahead & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    buffer = np.full((height, width), np.inf)
    if inside.any():
        np.minimum.at(
            buffer,
            (v[inside].astype(int), u[inside].astype(int)),
            np.linalg.norm(points_base[inside] - centre, axis=1),
        )
    return buffer


def diagnose_view(model_points, reference_points, K, width, height, base_T_cam):
    """Compare one view's reconstruction with the reference along shared rays."""
    centre = base_T_cam[:3, 3]
    buffer = _range_buffer(reference_points, K, width, height, base_T_cam)

    local = (model_points - centre) @ base_T_cam[:3, :3]
    z = local[:, 2]
    ahead = z > 0.05
    u = np.full(len(local), -1.0)
    v = np.full(len(local), -1.0)
    u[ahead] = local[ahead, 0] / z[ahead] * K[0, 0] + K[0, 2]
    v[ahead] = local[ahead, 1] / z[ahead] * K[1, 1] + K[1, 2]
    inside = ahead & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not inside.any():
        return {"usable": False, "reason": "reconstruction projects outside the reference camera"}

    reference_range = buffer[v[inside].astype(int), u[inside].astype(int)]
    model_range = np.linalg.norm(model_points[inside] - centre, axis=1)
    shared = (
        np.isfinite(reference_range)
        & (reference_range > 0.15)
        & (reference_range < MAX_RANGE_M)
        & (model_range < MAX_RANGE_M)
    )
    if shared.sum() < MIN_RAYS:
        return {
            "usable": False,
            "reason": f"only {int(shared.sum())} shared rays; need {MIN_RAYS}",
            "shared_rays": int(shared.sum()),
        }

    measured = reference_range[shared]
    predicted = model_range[shared]
    ratio = predicted / measured
    affine = fit_scale_robust(predicted, measured, np.ones(len(predicted), bool))

    # Split the displacement into "wrong distance" and "wrong direction".
    kept = model_points[inside][shared]
    ray = kept - centre
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    displacement = ray * (predicted - measured)[:, None]
    radial = float(np.median(predicted - measured))
    lateral_points = kept - displacement
    # Lateral error is what remains once the range is corrected: how far the
    # range-corrected point still sits from the reference surface along the ray
    # cannot be recovered here, so report the ray-perpendicular spread instead.
    perpendicular = np.linalg.norm(
        (kept - lateral_points) - ray * ((kept - lateral_points) * ray).sum(1)[:, None],
        axis=1,
    )

    return {
        "usable": True,
        "shared_rays": int(shared.sum()),
        "range_ratio_median": float(np.median(ratio)),
        "range_ratio_p25": float(np.percentile(ratio, 25)),
        "range_ratio_p75": float(np.percentile(ratio, 75)),
        "range_ratio_spread": float(np.percentile(ratio, 75) - np.percentile(ratio, 25)),
        "radial_error_median_m": radial,
        "radial_error_p95_m": float(np.percentile(np.abs(predicted - measured), 95)),
        "perpendicular_residual_median_m": float(np.median(perpendicular)),
        "affine_fit": affine.get("affine_test"),
        "affine_scale": affine.get("scale"),
        "affine_residual_rmse_m": affine.get("residual_rmse_m"),
        "affine_inlier_ratio": affine.get("inlier_ratio"),
    }


def decompose_against_surface(model_points, reference_points, base_T_cam, max_distance=0.30):
    """Split nearest-surface displacement into along-ray and across-ray parts.

    Radial error means the view put the surface at the wrong distance; a metric
    depth prior can correct that.  Lateral error means it put the surface in the
    wrong direction, which is a pose problem that no depth input will fix.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return {"usable": False, "reason": "SciPy is required for the surface decomposition"}
    if len(reference_points) < MIN_RAYS or len(model_points) < MIN_RAYS:
        return {"usable": False, "reason": "not enough points"}
    tree = cKDTree(reference_points)
    distance, index = tree.query(model_points, k=1)
    near = distance < max_distance
    if near.sum() < MIN_RAYS:
        return {"usable": False, "reason": f"only {int(near.sum())} matched points"}
    displacement = model_points[near] - reference_points[index[near]]
    ray = model_points[near] - base_T_cam[:3, 3]
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    radial = (displacement * ray).sum(1)
    lateral = np.linalg.norm(displacement - radial[:, None] * ray, axis=1)
    radial_scale = float(np.median(np.abs(radial)))
    lateral_scale = float(np.median(lateral))
    total = radial_scale + lateral_scale
    return {
        "usable": True,
        "matched_points": int(near.sum()),
        "radial_median_m": float(np.median(radial)),
        "radial_abs_median_m": radial_scale,
        "lateral_median_m": lateral_scale,
        "radial_fraction": float(radial_scale / total) if total > 0 else float("nan"),
        "note": (
            "radial_fraction near 1 means a depth problem that metric depth input can "
            "fix; near 0 means a pose problem that it cannot"
        ),
    }


def plane_report(points, low, high, label):
    """Fit the dominant plane in a height slab and report tilt against vertical."""
    try:
        from scipy.spatial import cKDTree  # noqa: F401  (kept for parity of deps)
    except ImportError:
        pass
    rng = np.random.default_rng(0)
    slab = points[(points[:, 2] > low) & (points[:, 2] < high)]
    if len(slab) < MIN_RAYS:
        return {"label": label, "fitted": False, "reason": f"only {len(slab)} points in slab"}
    if len(slab) > 60000:
        slab = slab[rng.choice(len(slab), 60000, replace=False)]
    best_inliers = None
    best_count = 0
    for _ in range(300):
        sample = slab[rng.choice(len(slab), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        inliers = np.abs((slab - sample[0]) @ normal) < 0.01
        if int(inliers.sum()) > best_count:
            best_count, best_inliers = int(inliers.sum()), inliers
    if best_inliers is None or best_count < MIN_RAYS:
        return {"label": label, "fitted": False, "reason": "no dominant plane"}
    report = plane_fit_report(slab[best_inliers])
    tilt = float(np.degrees(np.arccos(min(1.0, abs(report["normal"][2])))))
    report.update(
        {
            "label": label,
            "inlier_ratio": best_count / len(slab),
            "tilt_from_vertical_deg": tilt,
            "height_m": report["centroid_m"][2],
        }
    )
    return report


def diagnose_capture(session, output_root, capture, profile, planes=False):
    loaded = profile.load(session, capture)
    depth = loaded.depths.get("head")
    if depth is None:
        return {"capture": capture, "usable": False, "reason": "no metric depth in this capture"}
    reference = depth_to_base_points(depth.depth_m, depth.valid, depth.K, depth.base_T_cam)
    reference = reference[np.linalg.norm(reference, axis=1) < MAX_RANGE_M]

    views_path = output_root / capture / "views.npz"
    if not views_path.is_file():
        return {"capture": capture, "usable": False, "reason": f"missing {views_path}"}

    result = {"capture": capture, "usable": True, "reference_points": int(len(reference)), "views": {}}
    with np.load(views_path, allow_pickle=False) as data:
        for view in VIEW_NAMES:
            if f"{view}_pts3d" not in data.files:
                continue
            mask = np.asarray(data[f"{view}_mask"], dtype=bool)
            points = np.asarray(data[f"{view}_pts3d"], dtype=np.float64)[mask]
            points = points[np.linalg.norm(points, axis=1) < MAX_RANGE_M]
            K, width, height, base_T_cam = _camera(output_root, capture, view)
            entry = diagnose_view(points, reference, K, width, height, base_T_cam)
            entry["surface_decomposition"] = decompose_against_surface(
                points, reference, base_T_cam
            )
            result["views"][view] = entry

    if planes:
        result["reference_planes"] = [
            plane_report(reference, 0.55, 0.85, "table"),
            plane_report(reference, -0.15, 0.10, "floor"),
        ]
    return result


def print_capture(result):
    print(f"\n===== {result['capture']} =====")
    if not result.get("usable"):
        print(f"  skipped: {result.get('reason')}")
        return
    print(
        f"  {'view':11s} {'rays':>7s} {'range ratio (p25-p75)':>24s} "
        f"{'radial':>9s} {'lateral':>9s} {'rad%':>5s}  affine b"
    )
    for view, entry in result["views"].items():
        if not entry.get("usable"):
            print(f"  {view:11s} unusable: {entry.get('reason')}")
            continue
        decomposition = entry.get("surface_decomposition", {})
        radial = decomposition.get("radial_median_m")
        lateral = decomposition.get("lateral_median_m")
        fraction = decomposition.get("radial_fraction")
        affine = entry.get("affine_fit") or {}
        print(
            f"  {view:11s} {entry['shared_rays']:7d} "
            f"{entry['range_ratio_median']:8.3f} "
            f"({entry['range_ratio_p25']:.3f}-{entry['range_ratio_p75']:.3f})".ljust(45)
            + (f"{1000 * radial:+8.1f}mm " if radial is not None else "       -- ")
            + (f"{1000 * lateral:8.1f}mm " if lateral is not None else "       -- ")
            + (f"{100 * fraction:4.0f}% " if fraction is not None else "  --  ")
            + (f" {1000 * affine['b_m']:+7.1f}mm" if affine else "")
        )
    for plane in result.get("reference_planes", []):
        if plane.get("fitted"):
            print(
                f"  reference plane [{plane['label']}]: height {plane['height_m']:+.4f} m, "
                f"tilt {plane['tilt_from_vertical_deg']:.2f} deg, "
                f"flatness RMS {1000 * plane['residual_rms_m']:.2f} mm"
            )


def print_summary(results):
    print(f"\n{'=' * 72}\nAcross captures\n{'=' * 72}")
    print(
        f"  {'view':11s} {'n':>3s} {'range ratio':>18s} {'radial':>12s} "
        f"{'lateral':>12s}  interpretation"
    )
    for view in VIEW_NAMES:
        ratios, radial, lateral = [], [], []
        for result in results:
            entry = result.get("views", {}).get(view)
            if not entry or not entry.get("usable"):
                continue
            ratios.append(entry["range_ratio_median"])
            decomposition = entry.get("surface_decomposition", {})
            if decomposition.get("usable"):
                radial.append(decomposition["radial_median_m"])
                lateral.append(decomposition["lateral_median_m"])
        if not ratios:
            print(f"  {view:11s}  no usable captures")
            continue
        ratios = np.asarray(ratios)
        verdict = ""
        if radial and lateral:
            radial_scale = float(np.median(np.abs(radial)))
            lateral_scale = float(np.median(lateral))
            if lateral_scale > 2 * radial_scale:
                verdict = "mostly POSE error: depth input will not fix this"
            elif radial_scale > 2 * lateral_scale:
                verdict = "mostly DEPTH error: metric depth input should fix it"
            else:
                verdict = "mixed depth and pose error"
        line = (
            f"  {view:11s} {len(ratios):3d} "
            f"{ratios.mean():7.3f} +- {ratios.std():.3f}   "
        )
        if radial:
            line += f"{1000 * np.mean(radial):+8.1f}mm  "
            line += f"{1000 * np.mean(lateral):8.1f}mm  "
        print(line + verdict)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--session", type=Path, required=True, help="Raw capture session root")
    parser.add_argument("--output-root", type=Path, required=True, help="Pipeline output root")
    parser.add_argument("--captures", nargs="*", default=None)
    parser.add_argument("--robot", default=None)
    parser.add_argument(
        "--planes",
        action="store_true",
        help="Also characterise the reference depth camera on surfaces known to be flat",
    )
    parser.add_argument("--json-out", type=Path, help="Write the full report as JSON")
    args = parser.parse_args()

    profile = get_profile(args.robot) if args.robot else detect_profile(args.session)
    captures = args.captures or profile.discover(args.session)
    if not captures:
        print(f"No captures found under {args.session}", file=sys.stderr)
        return 2

    results = []
    for capture in captures:
        try:
            result = diagnose_capture(
                args.session, args.output_root, capture, profile, planes=args.planes
            )
        except (OSError, ValueError, KeyError) as exc:
            result = {"capture": capture, "usable": False, "reason": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        print_capture(result)

    usable = [r for r in results if r.get("usable")]
    if usable:
        print_summary(usable)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0 if usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
