"""Shared input-contract validation for G1 MapAnything captures."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


VIEW_NAMES = ("head", "hand_left", "hand_right")
IMAGE_TO_INTRINSIC = {
    "head": "intrinsic_head_front_rgb.json",
    "hand_left": "intrinsic_hand_left_rgb.json",
    "hand_right": "intrinsic_hand_right_rgb.json",
}
POSES_FILE = "camera_poses_opencv_cam2world.json"
PROVENANCE_FILES = (
    "manifest.json",
    "capture_state.json",
    "pose_conversion_manifest.json",
    "pose_validation_report.json",
)
SUPPORTED_POSE_CONVENTIONS = {
    "opencv_cam2world",  # Legacy captures; OpenCV implies RDF optical axes.
    "opencv_rdf_cam2world",
}


def present_views(npz) -> list[str]:
    """Return the views actually stored in a views.npz, in canonical order.

    A reconstruction from a subset of cameras (``--view-order head,hand_left``)
    writes only those views' arrays, so consumers must ask which views are
    present rather than assuming all of VIEW_NAMES.  Order follows VIEW_NAMES so
    stacked arrays stay in the declared, reproducible sequence.
    """
    views = [name for name in VIEW_NAMES if f"{name}_pts3d" in npz]
    if not views:
        raise ValueError("views.npz contains no recognised view arrays")
    return views


def _read_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as f:
            value = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def discover_raw_captures(data_root: os.PathLike | str) -> list[str]:
    """Return capture folders having all three raw images and intrinsics."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Capture root does not exist: {root}")
    required = {
        *(f"{name}.png" for name in VIEW_NAMES),
        *IMAGE_TO_INTRINSIC.values(),
    }
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and all((p / filename).is_file() for filename in required)
    )


def discover_preprocessed_captures(undist_root: os.PathLike | str) -> list[str]:
    """Return folders having all images/K files required by inference."""
    root = Path(undist_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Undistorted root does not exist: {root}")
    required = {
        *(f"{name}.png" for name in VIEW_NAMES),
        *(f"{name}_K.json" for name in VIEW_NAMES),
    }
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and all((p / filename).is_file() for filename in required)
    )


def resolve_reconstruction_captures(
    output_root: os.PathLike | str, requested: Iterable[str] | None
) -> list[str]:
    """Resolve outputs that contain a completed views.npz reconstruction."""
    root = Path(output_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Output root does not exist: {root}")
    captures = list(requested) if requested else sorted(
        p.name for p in root.iterdir() if p.is_dir() and (p / "views.npz").is_file()
    )
    if not captures:
        raise FileNotFoundError(f"No reconstruction folders with views.npz under {root}")
    for capture in captures:
        if not capture or Path(capture).name != capture:
            raise ValueError(f"Capture must be a folder name, not a path: {capture!r}")
        path = root / capture / "views.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Reconstruction does not exist: {path}")
    return captures


def resolve_captures(
    root: os.PathLike | str,
    requested: Iterable[str] | None,
    *,
    preprocessed: bool,
) -> list[str]:
    """Resolve explicit capture names or auto-discover compatible folders."""
    root = Path(root)
    captures = list(requested) if requested else (
        discover_preprocessed_captures(root)
        if preprocessed
        else discover_raw_captures(root)
    )
    if not captures:
        kind = "preprocessed" if preprocessed else "raw"
        raise FileNotFoundError(f"No compatible {kind} capture folders found under {root}")
    for capture in captures:
        if not capture or Path(capture).name != capture:
            raise ValueError(f"Capture must be a folder name, not a path: {capture!r}")
        if not (root / capture).is_dir():
            raise FileNotFoundError(f"Capture folder does not exist: {root / capture}")
    return captures


def load_intrinsics(
    path: os.PathLike | str,
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read and validate an OpenCV pinhole + plumb-bob calibration."""
    path = Path(path)
    data = _read_json(path)
    fields = ("Fx", "Fy", "Cx", "Cy", "k1", "k2", "p1", "p2", "k3")
    missing = [key for key in fields if key not in data]
    if missing:
        raise KeyError(f"{path} missing intrinsic fields: {missing}")
    values = np.asarray([data[key] for key in fields], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite intrinsic/distortion values")
    fx, fy, cx, cy = values[:4]
    if fx <= 0 or fy <= 0:
        raise ValueError(f"{path} has non-positive focal length: Fx={fx}, Fy={fy}")
    model = data.get("distortion_model")
    if model not in (None, "plumb bob", "plumb_bob"):
        raise ValueError(f"{path} distortion_model={model!r}; expected OpenCV plumb bob")
    if image_width is not None and not (0 <= cx < image_width):
        raise ValueError(f"{path} Cx={cx} is outside image width {image_width}")
    if image_height is not None and not (0 <= cy < image_height):
        raise ValueError(f"{path} Cy={cy} is outside image height {image_height}")
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    dist = values[4:]
    return K, dist, data


def validate_pose_document(
    path: os.PathLike | str,
) -> tuple[dict[str, np.ndarray], dict]:
    """Validate OpenCV RDF cam2world poses and return matrices + provenance."""
    path = Path(path)
    data = _read_json(path)
    convention = data.get("frame_convention")
    if convention not in SUPPORTED_POSE_CONVENTIONS:
        raise ValueError(
            f"{path} frame_convention={convention!r}; expected one of "
            f"{sorted(SUPPORTED_POSE_CONVENTIONS)}"
        )
    world_frame = data.get("world_frame")
    if not isinstance(world_frame, str) or not world_frame:
        raise ValueError(f"{path} must declare a non-empty world_frame")
    unit = data.get("translation_unit")
    legacy_unit_assumption = unit is None
    if unit not in (None, "m", "meter", "meters"):
        raise ValueError(f"{path} translation_unit={unit!r}; expected meters")
    if convention == "opencv_rdf_cam2world" and legacy_unit_assumption:
        raise ValueError(f"{path} must explicitly declare translation_unit='meter'")
    extrinsic_direction = data.get("extrinsic_direction")
    if extrinsic_direction is not None and not str(extrinsic_direction).endswith(
        "_T_camera"
    ):
        raise ValueError(
            f"{path} extrinsic_direction={extrinsic_direction!r}; expected world_T_camera"
        )

    raw_poses = data.get("poses")
    if not isinstance(raw_poses, dict):
        raise ValueError(f"{path} must contain a poses object")
    missing = [name for name in VIEW_NAMES if name not in raw_poses]
    if missing:
        raise KeyError(f"{path} missing poses for views: {missing}")

    poses: dict[str, np.ndarray] = {}
    checks = {}
    for name in VIEW_NAMES:
        pose = np.asarray(raw_poses[name], dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(f"{path} pose {name!r} has shape {pose.shape}, expected (4, 4)")
        if not np.isfinite(pose).all():
            raise ValueError(f"{path} pose {name!r} contains non-finite values")
        R = pose[:3, :3]
        det = float(np.linalg.det(R))
        orth_error = float(np.max(np.abs(R.T @ R - np.eye(3))))
        bottom_error = float(np.max(np.abs(pose[3] - np.array([0, 0, 0, 1]))))
        translation_norm = float(np.linalg.norm(pose[:3, 3]))
        if abs(det - 1.0) > 1e-3 or orth_error > 1e-3:
            raise ValueError(
                f"{path} pose {name!r} rotation is not in SO(3): "
                f"det={det:.6g}, orthogonality_error={orth_error:.3g}"
            )
        if bottom_error > 1e-6:
            raise ValueError(f"{path} pose {name!r} has invalid homogeneous bottom row")
        if translation_norm > 10.0:
            raise ValueError(
                f"{path} pose {name!r} translation norm {translation_norm:.3f} exceeds 10 m"
            )
        poses[name] = pose
        checks[name] = {
            "determinant": det,
            "orthogonality_max_abs": orth_error,
            "translation_norm_m": translation_norm,
        }

    if world_frame == "head_rgb_opencv_at_capture" and not np.allclose(
        poses["head"], np.eye(4), atol=1e-6
    ):
        raise ValueError(
            f"{path} declares head_rgb_opencv_at_capture world but head pose is not identity"
        )

    metadata = {
        "source_file": str(path.resolve()),
        "frame_convention": convention,
        "camera_axes": "OpenCV RDF: +X right, +Y down, +Z forward",
        "matrix_direction": "camera_to_world",
        "world_frame": world_frame,
        "translation_unit": "meter",
        "legacy_translation_unit_assumption": legacy_unit_assumption,
        "extrinsic_direction": extrinsic_direction,
        "rigid_transform_checks": checks,
    }
    return poses, metadata
