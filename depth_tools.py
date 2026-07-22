"""Metric depth registration and robust scale fitting.

Depth and colour are separate physical cameras, so a depth map only becomes
usable as a per-pixel prior for a colour view after being reprojected into that
view.  Nothing here invents data: pixels with no source sample stay invalid, and
every function reports how much of the target it actually filled.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# What counts as a pixel agreeing with the reference depth.  The measured head
# depth sensor sits around 3-5 mm RMS on flat surfaces, so a 2 cm floor is well
# outside its own noise while the relative term keeps the gate fair at range.
ABSOLUTE_INLIER_TOLERANCE_M = 0.02
RELATIVE_INLIER_TOLERANCE = 0.05


def unproject_depth(
    depth_m: np.ndarray, valid: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project valid Z-depth pixels into their own camera frame.

    Returns the ``(N, 3)`` camera-frame points and the flat pixel indices they
    came from, so callers can map results back onto the image grid.
    """
    depth_m = np.asarray(depth_m, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(depth_m) & (depth_m > 0.0)
    height, width = depth_m.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    flat = np.flatnonzero(valid)
    z = depth_m.reshape(-1)[flat]
    x = (u.reshape(-1)[flat] - K[0, 2]) / K[0, 0] * z
    y = (v.reshape(-1)[flat] - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=1), flat


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to ``(N, 3)`` points."""
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def depth_to_base_points(
    depth_m: np.ndarray, valid: np.ndarray, K: np.ndarray, base_T_cam: np.ndarray
) -> np.ndarray:
    """Back-project a depth map straight into the robot base frame."""
    points, _ = unproject_depth(depth_m, valid, K)
    return transform_points(points, base_T_cam)


def register_depth_to_camera(
    depth_m: np.ndarray,
    valid: np.ndarray,
    *,
    K_source: np.ndarray,
    base_T_source: np.ndarray,
    K_target: np.ndarray,
    base_T_target: np.ndarray,
    target_shape: tuple[int, int],
    splat_radius: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reproject a depth map into another camera as that camera's Z-depth.

    Uses forward mapping with a Z-buffer so an occluding surface wins whenever
    several source pixels land on one target pixel.  ``splat_radius`` widens each
    sample into a square to close the sampling holes that forward mapping leaves;
    it trades a little edge bleeding for coverage and defaults to off.

    Target pixels that receive no sample stay ``NaN`` and invalid.
    """
    target_height, target_width = (int(v) for v in target_shape)
    points_source, _ = unproject_depth(depth_m, valid, K_source)
    source_count = len(points_source)
    if not source_count:
        empty = np.full((target_height, target_width), np.nan)
        return empty, np.zeros_like(empty, dtype=bool), {
            "source_valid_pixels": 0,
            "target_filled_pixels": 0,
            "target_fill_ratio": 0.0,
            "dropped_behind_camera": 0,
            "dropped_outside_frame": 0,
        }

    target_T_source = np.linalg.inv(base_T_target) @ base_T_source
    points_target = transform_points(points_source, target_T_source)
    z = points_target[:, 2]
    in_front = z > 0.0
    dropped_behind = int((~in_front).sum())
    points_target, z = points_target[in_front], z[in_front]

    u = points_target[:, 0] / z * K_target[0, 0] + K_target[0, 2]
    v = points_target[:, 1] / z * K_target[1, 1] + K_target[1, 2]
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)

    buffer = np.full(target_height * target_width, np.inf)
    offsets = range(-int(splat_radius), int(splat_radius) + 1)
    dropped_outside = 0
    for dv in offsets:
        for du in offsets:
            uu, vv = ui + du, vi + dv
            inside = (uu >= 0) & (uu < target_width) & (vv >= 0) & (vv < target_height)
            if dv == 0 and du == 0:
                dropped_outside = int((~inside).sum())
            flat = vv[inside] * target_width + uu[inside]
            # Keep the nearest surface per target pixel.
            np.minimum.at(buffer, flat, z[inside])

    filled = np.isfinite(buffer)
    depth_target = np.where(filled, buffer, np.nan).reshape(target_height, target_width)
    valid_target = filled.reshape(target_height, target_width)
    return depth_target, valid_target, {
        "source_valid_pixels": int(source_count),
        "target_filled_pixels": int(filled.sum()),
        "target_fill_ratio": float(filled.mean()),
        "dropped_behind_camera": dropped_behind,
        "dropped_outside_frame": dropped_outside,
        "splat_radius": int(splat_radius),
        "method": "forward_projection_with_z_buffer",
    }


def fit_scale_robust(
    model_depth: np.ndarray,
    reference_depth: np.ndarray,
    valid: np.ndarray,
    *,
    max_iterations: int = 20,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Fit ``reference ~= scale * model`` with an iteratively reweighted Huber loss.

    Also fits the unconstrained affine ``reference ~= a * model + b`` so callers
    can test whether the single-scale assumption holds at all: a ``b`` far from
    zero means the residual is not a pure scale error and no global scalar will
    fix the reconstruction.
    """
    model = np.asarray(model_depth, dtype=np.float64)
    reference = np.asarray(reference_depth, dtype=np.float64)
    mask = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(model)
        & np.isfinite(reference)
        & (model > 0.0)
        & (reference > 0.0)
    )
    count = int(mask.sum())
    if count < 100:
        return {
            "converged": False,
            "reason": f"only {count} co-visible metric pixels; need at least 100",
            "pixel_count": count,
        }
    x = model[mask]
    y = reference[mask]

    scale = float(np.median(y / x))
    weights = np.ones_like(x)
    for _ in range(max_iterations):
        previous = scale
        scale = float(np.sum(weights * x * y) / np.sum(weights * x * x))
        residual = y - scale * x
        spread = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        delta = max(spread, 1e-6)
        absolute = np.abs(residual)
        weights = np.where(absolute <= delta, 1.0, delta / absolute)
        if abs(scale - previous) <= tolerance:
            break

    residual = y - scale * x
    # Inliers are measured against an absolute agreement tolerance, not against
    # the residuals' own spread: a self-scaled threshold calls ~99 percent of
    # pixels inliers even when the two depth maps are unrelated, which makes it
    # worthless as a quality gate.
    tolerance = np.maximum(ABSOLUTE_INLIER_TOLERANCE_M, RELATIVE_INLIER_TOLERANCE * y)
    inliers = np.abs(residual) <= tolerance
    design = np.stack([x, np.ones_like(x)], axis=1)
    affine, *_ = np.linalg.lstsq(design, y, rcond=None)

    return {
        "converged": True,
        "pixel_count": count,
        "scale": scale,
        "residual_rmse_m": float(np.sqrt(np.mean(residual**2))),
        "residual_median_abs_m": float(np.median(np.abs(residual))),
        "residual_p95_abs_m": float(np.percentile(np.abs(residual), 95)),
        "inlier_ratio": float(inliers.mean()),
        "inlier_tolerance": {
            "absolute_m": ABSOLUTE_INLIER_TOLERANCE_M,
            "relative": RELATIVE_INLIER_TOLERANCE,
            "note": "|residual| <= max(absolute_m, relative * reference_depth)",
        },
        "reference_median_m": float(np.median(y)),
        "model_median_m": float(np.median(x)),
        "affine_test": {
            "a": float(affine[0]),
            "b_m": float(affine[1]),
            "note": (
                "reference ~= a * model + b; |b| well above the residual RMSE means "
                "the error is not a pure scale error and one global scalar cannot fix it"
            ),
        },
        "loss": "iteratively_reweighted_huber",
    }


def plane_fit_report(points: np.ndarray) -> dict[str, Any]:
    """Fit a plane by total least squares and report the residual spread.

    Used to characterise a depth sensor's own noise on a surface that is known
    to be flat, which is what makes it usable as a scale reference at all.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return {"fitted": False, "reason": "need at least three points"}
    centroid = points.mean(axis=0)
    _, singular, right = np.linalg.svd(points - centroid, full_matrices=False)
    normal = right[-1]
    distance = (points - centroid) @ normal
    return {
        "fitted": True,
        "point_count": int(len(points)),
        "centroid_m": centroid.tolist(),
        "normal": normal.tolist(),
        "residual_rms_m": float(np.sqrt(np.mean(distance**2))),
        "residual_p95_abs_m": float(np.percentile(np.abs(distance), 95)),
        "singular_values": singular.tolist(),
    }
