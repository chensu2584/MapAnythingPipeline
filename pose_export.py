"""Pure-numpy camera-pose selection for MapAnything reconstruction export."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


CALIBRATED_INPUT = "calibrated-input"
MODEL_RELATIVE_HEAD_ANCHORED = "model-relative-head-anchored"
MODEL_RELATIVE_HEAD_ANCHORED_BASELINE_SCALED = (
    "model-relative-head-anchored-baseline-scaled"
)
MODEL_RELATIVE_HEAD_ANCHORED_DEPTH_SCALED = (
    "model-relative-head-anchored-depth-scaled"
)
MODEL_RELATIVE_HEAD_ANCHORED_DEPTH_AFFINE = (
    "model-relative-head-anchored-depth-affine"
)
MODEL_PREDICTION_ARBITRARY_SCALE = "model-prediction-arbitrary-scale"
POSE_EXPORT_MODES = (
    MODEL_RELATIVE_HEAD_ANCHORED_BASELINE_SCALED,
    MODEL_RELATIVE_HEAD_ANCHORED_DEPTH_SCALED,
    MODEL_RELATIVE_HEAD_ANCHORED_DEPTH_AFFINE,
    MODEL_RELATIVE_HEAD_ANCHORED,
    CALIBRATED_INPUT,
)

# Modes that warp geometry rather than apply one similarity transform.  They
# exist to measure the shape of the depth error, not to produce a map anything
# downstream should plan against.
DIAGNOSTIC_ONLY_MODES = frozenset({MODEL_RELATIVE_HEAD_ANCHORED_DEPTH_AFFINE})
DEFAULT_POSE_EXPORT_MODE = MODEL_RELATIVE_HEAD_ANCHORED_BASELINE_SCALED

# A depth-anchored scale is only trusted when the robust fit actually had
# something to work with.  Below these the mode falls back to the baseline fit
# and records why, rather than silently exporting a scale nobody can defend.
MIN_DEPTH_SCALE_PIXELS = 5000
MIN_DEPTH_SCALE_INLIER_RATIO = 0.5


def _pose(value: np.ndarray, label: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"{label} has shape {pose.shape}, expected (4, 4)")
    if not np.isfinite(pose).all():
        raise ValueError(f"{label} contains non-finite values")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError(f"{label} has an invalid homogeneous bottom row")
    return pose


def pose_delta(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Return candidate-vs-reference rigid-pose difference."""

    reference = _pose(reference, "reference pose")
    candidate = _pose(candidate, "candidate pose")
    delta = np.linalg.inv(reference) @ candidate
    cos_angle = np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return {
        "translation_m": float(np.linalg.norm(delta[:3, 3])),
        "rotation_deg": float(np.degrees(np.arccos(cos_angle))),
    }


def align_model_poses_to_calibrated_head(
    model_poses: Sequence[np.ndarray],
    calibrated_head_pose: np.ndarray,
    similarity_scale: float = 1.0,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Place model-relative poses in calibrated world, optionally scaling baselines.

    MapAnything returns all model poses in one internally consistent reference
    frame.  View 0 is the head by the pipeline contract.  A single left-side
    transform maps that reference frame to the calibrated world.  A common positive
    similarity scale changes only head-relative translations; rotations and the
    relative directions remain untouched.  Depth must be multiplied by the same
    scale by the caller to keep the reconstructed geometry self-consistent.
    """

    if not model_poses:
        raise ValueError("At least one model pose is required")
    checked = [_pose(value, f"model pose {index}") for index, value in enumerate(model_poses)]
    calibrated_head = _pose(calibrated_head_pose, "calibrated head pose")
    if not np.isfinite(similarity_scale) or similarity_scale <= 0:
        raise ValueError("Similarity scale must be finite and positive")
    model_head_t_world = np.linalg.inv(checked[0])
    world_t_model_reference = calibrated_head @ np.linalg.inv(checked[0])
    aligned = []
    for pose in checked:
        head_t_camera = model_head_t_world @ pose
        head_t_camera = head_t_camera.copy()
        head_t_camera[:3, 3] *= similarity_scale
        aligned.append(calibrated_head @ head_t_camera)
    if not np.allclose(aligned[0], calibrated_head, atol=1e-7):
        raise AssertionError("Head anchoring did not reproduce the calibrated head pose")
    return aligned, world_t_model_reference


def estimate_baseline_similarity_scale(
    model_poses: Sequence[np.ndarray], calibrated_poses: Sequence[np.ndarray]
) -> dict:
    """Fit one scale from all pairwise calibrated/model camera baselines."""

    model = [_pose(value, f"model pose {index}") for index, value in enumerate(model_poses)]
    calibrated = [
        _pose(value, f"calibrated pose {index}")
        for index, value in enumerate(calibrated_poses)
    ]
    if len(model) != len(calibrated):
        raise ValueError("Model and calibrated pose counts differ")
    if len(model) < 2:
        raise ValueError("At least two camera poses are required to estimate scale")

    labels = (
        ("head", "hand_left", "hand_right")
        if len(model) == 3
        else tuple(f"view_{index}" for index in range(len(model)))
    )
    pairs = []
    model_baselines = []
    calibrated_baselines = []
    for first in range(len(model)):
        for second in range(first + 1, len(model)):
            model_baseline = float(
                np.linalg.norm(model[first][:3, 3] - model[second][:3, 3])
            )
            calibrated_baseline = float(
                np.linalg.norm(
                    calibrated[first][:3, 3] - calibrated[second][:3, 3]
                )
            )
            if model_baseline <= 1e-6 or calibrated_baseline <= 1e-6:
                raise ValueError("Cannot estimate scale from a degenerate camera baseline")
            model_baselines.append(model_baseline)
            calibrated_baselines.append(calibrated_baseline)
            pairs.append((labels[first], labels[second]))

    model_vector = np.asarray(model_baselines)
    calibrated_vector = np.asarray(calibrated_baselines)
    scale = float(
        np.dot(model_vector, calibrated_vector) / np.dot(model_vector, model_vector)
    )
    if not 0.25 <= scale <= 4.0:
        raise ValueError(f"Estimated similarity scale {scale:.6g} is implausible")
    corrected = scale * model_vector
    ratios = calibrated_vector / model_vector
    ratio_median = float(np.median(ratios))
    pairwise = {}
    for (first, second), model_value, calibrated_value, ratio, corrected_value in zip(
        pairs, model_vector, calibrated_vector, ratios, corrected
    ):
        pairwise[f"{first}__{second}"] = {
            "model_baseline_before_m": float(model_value),
            "calibrated_baseline_m": float(calibrated_value),
            "calibrated_over_model_ratio": float(ratio),
            "model_baseline_after_m": float(corrected_value),
            "residual_after_m": float(corrected_value - calibrated_value),
        }
    return {
        "applied": True,
        "scale": scale,
        "estimator": "least_squares_all_pairwise_camera_baseline_lengths",
        "pairwise": pairwise,
        "baseline_rmse_before_m": float(
            np.sqrt(np.mean((model_vector - calibrated_vector) ** 2))
        ),
        "baseline_rmse_after_m": float(
            np.sqrt(np.mean((corrected - calibrated_vector) ** 2))
        ),
        "pair_ratio_median": ratio_median,
        "pair_ratio_relative_spread": float(
            (ratios.max() - ratios.min()) / ratio_median
        ),
    }


def estimate_depth_similarity_scale(
    model_depth: np.ndarray,
    reference_depth: np.ndarray,
    valid: np.ndarray,
) -> dict:
    """Fit one scale by comparing predicted and measured depth pixel by pixel.

    Where the baseline estimator fits one scalar to three camera-centre
    distances, this fits it to every co-visible metric pixel and constrains
    scene depth directly.  The returned report carries the affine test so the
    caller can see whether a single global scale is even the right model.
    """
    from depth_tools import fit_scale_robust

    result = fit_scale_robust(model_depth, reference_depth, valid)
    if not result.get("converged"):
        return {"applied": False, **result, "estimator": "robust_per_pixel_metric_depth"}
    scale = result["scale"]
    if not 0.25 <= scale <= 4.0:
        return {
            "applied": False,
            "reason": f"estimated depth scale {scale:.6g} is implausible",
            **result,
            "estimator": "robust_per_pixel_metric_depth",
        }
    if result["pixel_count"] < MIN_DEPTH_SCALE_PIXELS:
        return {
            "applied": False,
            "reason": (
                f"only {result['pixel_count']} co-visible metric pixels; "
                f"need at least {MIN_DEPTH_SCALE_PIXELS}"
            ),
            **result,
            "estimator": "robust_per_pixel_metric_depth",
        }
    if result["inlier_ratio"] < MIN_DEPTH_SCALE_INLIER_RATIO:
        return {
            "applied": False,
            "reason": (
                f"inlier ratio {result['inlier_ratio']:.3f} below "
                f"{MIN_DEPTH_SCALE_INLIER_RATIO}; the model and the depth sensor "
                "do not agree well enough for a single scale"
            ),
            **result,
            "estimator": "robust_per_pixel_metric_depth",
        }
    return {"applied": True, **result, "estimator": "robust_per_pixel_metric_depth"}


def select_export_poses(
    model_poses: Sequence[np.ndarray],
    calibrated_poses: Sequence[np.ndarray] | None,
    requested_mode: str = DEFAULT_POSE_EXPORT_MODE,
    depth_scale_inputs: dict | None = None,
) -> tuple[list[np.ndarray], str, np.ndarray | None, dict | None]:
    """Return final poses, effective mode, rigid anchor and scale report.

    ``depth_scale_inputs`` optionally supplies ``model_depth`` / ``reference_depth``
    / ``valid`` arrays for the head view; it is required by the depth-scaled mode
    and ignored by every other mode.
    """

    if requested_mode not in POSE_EXPORT_MODES:
        raise ValueError(
            f"Unknown pose export mode {requested_mode!r}; expected one of {POSE_EXPORT_MODES}"
        )
    checked_model = [
        _pose(value, f"model pose {index}") for index, value in enumerate(model_poses)
    ]
    if calibrated_poses is None:
        return checked_model, MODEL_PREDICTION_ARBITRARY_SCALE, None, None

    checked_calibrated = [
        _pose(value, f"calibrated pose {index}")
        for index, value in enumerate(calibrated_poses)
    ]
    if len(checked_calibrated) != len(checked_model):
        raise ValueError("Model and calibrated pose counts differ")
    if requested_mode == CALIBRATED_INPUT:
        return checked_calibrated, CALIBRATED_INPUT, None, None
    scale_report = None
    similarity_scale = 1.0
    effective_mode = requested_mode
    if requested_mode == MODEL_RELATIVE_HEAD_ANCHORED_BASELINE_SCALED:
        scale_report = estimate_baseline_similarity_scale(
            checked_model, checked_calibrated
        )
        similarity_scale = scale_report["scale"]
    elif requested_mode in (
        MODEL_RELATIVE_HEAD_ANCHORED_DEPTH_SCALED,
        MODEL_RELATIVE_HEAD_ANCHORED_DEPTH_AFFINE,
    ):
        affine_requested = requested_mode == MODEL_RELATIVE_HEAD_ANCHORED_DEPTH_AFFINE
        depth_report = None
        if depth_scale_inputs:
            depth_report = estimate_depth_similarity_scale(
                depth_scale_inputs["model_depth"],
                depth_scale_inputs["reference_depth"],
                depth_scale_inputs["valid"],
            )
        # The affine mode exists precisely for the case where a single scale
        # does not fit, so gating it on the single-scale fit's quality would
        # reject it exactly when it is needed.  Judge it on its own fit.
        usable = bool(depth_report and depth_report.get("applied"))
        if affine_requested and depth_report and depth_report.get("converged"):
            fit = depth_report.get("affine_test") or {}
            usable = (
                depth_report["pixel_count"] >= MIN_DEPTH_SCALE_PIXELS
                and 0.25 <= fit.get("a", 0.0) <= 4.0
                and fit.get("inlier_ratio", 0.0) >= MIN_DEPTH_SCALE_INLIER_RATIO
            )
            if not usable:
                depth_report = {
                    **depth_report,
                    "applied": False,
                    "reason": (
                        f"affine fit a={fit.get('a')} inlier_ratio="
                        f"{fit.get('inlier_ratio')} does not meet the gate"
                    ),
                }
        if usable:
            scale_report = depth_report
            if affine_requested:
                # Correct depth as measured = a * model + b instead of a single
                # scalar.  Camera translations still take the multiplicative
                # part only, because an offset has no meaning for a baseline.
                fit = depth_report["affine_test"]
                similarity_scale = float(fit["a"])
                scale_report = {
                    **depth_report,
                    "scale": similarity_scale,
                    "depth_offset_m": float(fit["b_m"]),
                    "estimator": "robust_per_pixel_metric_depth_affine",
                    "geometry_warning": (
                        "an affine depth correction is not a similarity transform: it "
                        "stretches the scene by a different amount at each range and "
                        "breaks agreement between the three cameras. Diagnostic only; "
                        "do not plan against this map."
                    ),
                }
            else:
                similarity_scale = depth_report["scale"]
        else:
            # Fall back rather than export an undefendable scale, but keep the
            # rejected depth fit visible beside the baseline one.
            scale_report = estimate_baseline_similarity_scale(
                checked_model, checked_calibrated
            )
            similarity_scale = scale_report["scale"]
            effective_mode = MODEL_RELATIVE_HEAD_ANCHORED_BASELINE_SCALED
            scale_report = {
                **scale_report,
                "requested_estimator": (
                    "robust_per_pixel_metric_depth_affine"
                    if affine_requested
                    else "robust_per_pixel_metric_depth"
                ),
                "fell_back_to_baseline": True,
                "depth_scale_rejected": depth_report
                or {
                    "applied": False,
                    "reason": "no registered metric depth was available for the head view",
                },
            }
    aligned, anchor = align_model_poses_to_calibrated_head(
        checked_model,
        checked_calibrated[0],
        similarity_scale=similarity_scale,
    )
    return aligned, effective_mode, anchor, scale_report
