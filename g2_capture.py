"""G2 snapshot reader and contract validation.

A G2 snapshot folder holds three colour PNGs, one uint16 head depth PNG and a
single ``camera_extrinsics.json`` carrying intrinsics, ``base_T_camera``
extrinsics, joint state and the capture script's own FK-vs-SDK validation.

Unlike the G1 captures this contract is already metric ``base_link``, so no
forward kinematics has to be re-derived here.  What this module does is refuse
anything it cannot prove: non-rigid matrices, disagreeing quaternion/inverse
copies, unsynchronised cameras, or a failed kinematic self-check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_profiles import VIEW_NAMES, RawCapture, RawDepth, RawView


EXTRINSICS_FILE = "camera_extrinsics.json"

# Snapshot capture keys -> canonical pipeline view names.
CAPTURE_TO_VIEW = {
    "head_rgb": "head",
    "hand_left_rgb": "hand_left",
    "hand_right_rgb": "hand_right",
}
VIEW_TO_CAPTURE = {view: key for key, view in CAPTURE_TO_VIEW.items()}
DEPTH_CAPTURE_TO_VIEW = {"head_depth": "head"}

DEPTH_UNIT_SCALE_TO_M = 1e-3  # uint16 millimetres
DEPTH_INVALID_VALUES = (0, 65535)
DEPTH_MAX_PLAUSIBLE_M = 20.0

MAX_FK_TRANSLATION_ERROR_M = 1e-3
MAX_FK_ROTATION_ERROR_DEG = 0.1
MAX_CAMERA_DESYNC_NS = 50_000_000  # 50 ms

# Tolerances for the redundant copies (quaternion / inverse / translation) that
# the capture script writes beside each matrix.  These catch a corrupted or
# mismatched document, not calibration error, so they only have to sit above
# JSON serialisation and quaternion-conversion noise while staying far below
# anything physically meaningful.
MAX_COPY_TRANSLATION_ERROR_M = 1e-6
MAX_COPY_ROTATION_ERROR_DEG = 1e-3
MAX_INVERSE_RESIDUAL = 1e-7


def _read_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def discover_g2_snapshots(root: Path | str) -> list[str]:
    """Return snapshot folder names that carry a G2 extrinsics document."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / EXTRINSICS_FILE).is_file()
    )


def _quaternion_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    norm = float(np.linalg.norm([x, y, z, w]))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Quaternion must be finite and non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle between two 3x3 rotations, in degrees."""
    cosine = np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def validate_transform(entry: dict, label: str) -> tuple[np.ndarray, dict[str, float]]:
    """Validate one extrinsic entry against its own quaternion and inverse copies."""
    matrix = np.asarray(entry.get("matrix"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} matrix must be a finite 4x4")
    rotation = matrix[:3, :3]
    determinant = float(np.linalg.det(rotation))
    orthogonality = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    bottom = float(np.max(np.abs(matrix[3] - np.array([0.0, 0.0, 0.0, 1.0]))))
    if abs(determinant - 1.0) > 1e-6 or orthogonality > 1e-6:
        raise ValueError(
            f"{label} rotation is not in SO(3): det={determinant:.9g}, "
            f"orthogonality_error={orthogonality:.3g}"
        )
    if bottom > 1e-9:
        raise ValueError(f"{label} has an invalid homogeneous bottom row")

    checks = {
        "determinant": determinant,
        "orthogonality_max_abs": orthogonality,
        "translation_norm_m": float(np.linalg.norm(matrix[:3, 3])),
    }

    translation = entry.get("translation_xyz_m")
    if translation is not None:
        error = float(np.max(np.abs(np.asarray(translation, dtype=np.float64) - matrix[:3, 3])))
        if error > MAX_COPY_TRANSLATION_ERROR_M:
            raise ValueError(f"{label} translation_xyz_m disagrees with matrix by {error:.3g} m")
        checks["translation_copy_max_abs_m"] = error

    quaternion = entry.get("quaternion_xyzw")
    if quaternion is not None:
        from_quaternion = _quaternion_xyzw_to_matrix(
            np.asarray(quaternion, dtype=np.float64)
        )
        error = _rotation_error_deg(rotation, from_quaternion)
        if error > MAX_COPY_ROTATION_ERROR_DEG:
            raise ValueError(f"{label} quaternion_xyzw disagrees with matrix by {error:.3g} deg")
        checks["quaternion_copy_error_deg"] = error

    inverse = entry.get("inverse_matrix")
    if inverse is not None:
        residual = np.asarray(inverse, dtype=np.float64) @ matrix
        error = float(np.max(np.abs(residual - np.eye(4))))
        if error > MAX_INVERSE_RESIDUAL:
            raise ValueError(f"{label} inverse_matrix is not the inverse (residual {error:.3g})")
        checks["inverse_residual_max_abs"] = error

    return matrix, checks


def _intrinsics_from_entry(entry: dict, label: str, width: int, height: int):
    fields = ("Fx", "Fy", "Cx", "Cy", "k1", "k2", "p1", "p2", "k3")
    missing = [key for key in fields if key not in entry]
    if missing:
        raise KeyError(f"{label} is missing intrinsic fields: {missing}")
    values = np.asarray([entry[key] for key in fields], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite intrinsic values")
    fx, fy, cx, cy = values[:4]
    if fx <= 0 or fy <= 0:
        raise ValueError(f"{label} has non-positive focal length: Fx={fx}, Fy={fy}")
    if not (0 <= cx < width) or not (0 <= cy < height):
        raise ValueError(
            f"{label} principal point ({cx:.2f}, {cy:.2f}) falls outside {width}x{height}"
        )
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K, values[4:]


def _check_kinematic_validation(document: dict, snapshot_dir: Path) -> dict[str, Any]:
    """Fail closed when the capture script's own FK/SDK cross-check did not pass."""
    validation = document.get("validation")
    if not isinstance(validation, dict):
        raise ValueError(f"{snapshot_dir} has no validation block")
    fk = validation.get("fk_vs_sdk_tf")
    if not isinstance(fk, dict) or not fk:
        raise ValueError(f"{snapshot_dir} has no fk_vs_sdk_tf validation")
    worst_translation = 0.0
    worst_rotation = 0.0
    for link, entry in fk.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{snapshot_dir} fk_vs_sdk_tf[{link}] is malformed")
        translation = float(entry.get("translation_error_m", np.inf))
        rotation = float(entry.get("rotation_error_deg", np.inf))
        if not np.isfinite(translation) or not np.isfinite(rotation):
            raise ValueError(f"{snapshot_dir} fk_vs_sdk_tf[{link}] has non-finite errors")
        worst_translation = max(worst_translation, translation)
        worst_rotation = max(worst_rotation, rotation)
    if (
        worst_translation > MAX_FK_TRANSLATION_ERROR_M
        or worst_rotation > MAX_FK_ROTATION_ERROR_DEG
    ):
        raise ValueError(
            f"{snapshot_dir} failed its FK-vs-SDK kinematic check: "
            f"max translation {worst_translation:.6f} m (limit {MAX_FK_TRANSLATION_ERROR_M}), "
            f"max rotation {worst_rotation:.4f} deg (limit {MAX_FK_ROTATION_ERROR_DEG})"
        )
    return {
        "links_checked": sorted(fk),
        "max_translation_error_m": worst_translation,
        "max_rotation_error_deg": worst_rotation,
    }


def _check_synchronisation(captures: dict, snapshot_dir: Path) -> dict[str, Any]:
    stamps = {}
    for key, entry in captures.items():
        value = entry.get("timestamp_ns")
        if value is None:
            continue
        stamps[key] = int(value)
    if not stamps:
        raise ValueError(f"{snapshot_dir} captures carry no timestamps")
    spread = max(stamps.values()) - min(stamps.values())
    if spread > MAX_CAMERA_DESYNC_NS:
        raise ValueError(
            f"{snapshot_dir} cameras are desynchronised by {spread / 1e6:.1f} ms "
            f"(limit {MAX_CAMERA_DESYNC_NS / 1e6:.0f} ms)"
        )
    return {"timestamp_ns": stamps, "max_spread_ns": int(spread)}


def load_depth_png(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode a uint16 millimetre depth PNG into metres plus a validity mask."""
    import cv2

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Cannot read depth image: {path}")
    if raw.dtype != np.uint16 or raw.ndim != 2:
        raise ValueError(
            f"{path} must be a single-channel uint16 depth image, got "
            f"{raw.dtype} with shape {raw.shape}"
        )
    valid = np.ones(raw.shape, dtype=bool)
    for sentinel in DEPTH_INVALID_VALUES:
        valid &= raw != sentinel
    depth = raw.astype(np.float64) * DEPTH_UNIT_SCALE_TO_M
    valid &= depth <= DEPTH_MAX_PLAUSIBLE_M
    depth[~valid] = np.nan
    return depth, valid


def load_g2_snapshot(snapshot_dir: Path | str, *, profile, capture: str) -> RawCapture:
    """Read and validate one G2 snapshot into the shared raw representation."""
    import cv2

    snapshot_dir = Path(snapshot_dir)
    document_path = snapshot_dir / EXTRINSICS_FILE
    if not document_path.is_file():
        raise FileNotFoundError(f"Not a G2 snapshot (no {EXTRINSICS_FILE}): {snapshot_dir}")
    document = _read_json(document_path)

    base_frame = document.get("base_link")
    if base_frame != "base_link":
        raise ValueError(f"{document_path} declares base_link={base_frame!r}; expected 'base_link'")
    convention = document.get("convention", {})
    declared = str(convention.get("base_T_camera", ""))
    if "camera-frame points into base_link" not in declared:
        raise ValueError(
            f"{document_path} does not declare the expected base_T_camera direction; "
            f"got {declared!r}"
        )

    captures = document.get("captures")
    extrinsics = document.get("extrinsics")
    if not isinstance(captures, dict) or not isinstance(extrinsics, dict):
        raise ValueError(f"{document_path} must contain captures and extrinsics objects")

    kinematics = _check_kinematic_validation(document, snapshot_dir)
    synchronisation = _check_synchronisation(captures, snapshot_dir)

    transform_checks: dict[str, Any] = {}
    views: dict[str, RawView] = {}
    for view_name in VIEW_NAMES:
        key = VIEW_TO_CAPTURE[view_name]
        entry = captures.get(key)
        if not isinstance(entry, dict):
            raise KeyError(f"{document_path} is missing capture {key!r}")
        error = str(entry.get("error", ""))
        if error:
            raise ValueError(f"{document_path} capture {key!r} reported an error: {error}")
        image_path = snapshot_dir / str(entry.get("saved_path", f"{key}.png"))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        height, width = image.shape[:2]
        shape = entry.get("shape")
        if shape is not None and tuple(int(v) for v in shape[:2]) != (height, width):
            raise ValueError(
                f"{image_path} is {height}x{width} but {key} declares shape {shape}"
            )
        K, dist = _intrinsics_from_entry(
            entry.get("intrinsic", {}), f"{document_path}:{key}.intrinsic", width, height
        )
        if key not in extrinsics:
            raise KeyError(f"{document_path} is missing extrinsics for {key!r}")
        matrix, checks = validate_transform(extrinsics[key], f"{document_path}:extrinsics.{key}")
        transform_checks[key] = checks
        views[view_name] = RawView(
            name=view_name,
            image_bgr=image,
            K=K,
            dist=dist,
            base_T_cam=matrix,
            intrinsic_source=f"{document_path.resolve()}#captures.{key}.intrinsic",
            intrinsic_raw=dict(entry.get("intrinsic", {})),
        )

    depths: dict[str, RawDepth] = {}
    for key, view_name in DEPTH_CAPTURE_TO_VIEW.items():
        entry = captures.get(key)
        if not isinstance(entry, dict) or str(entry.get("error", "")):
            continue
        depth_path = snapshot_dir / str(entry.get("saved_path", f"{key}.png"))
        if not depth_path.is_file():
            continue
        depth_m, valid = load_depth_png(depth_path)
        height, width = depth_m.shape
        shape = entry.get("shape")
        if shape is not None and tuple(int(v) for v in shape[:2]) != (height, width):
            raise ValueError(
                f"{depth_path} is {height}x{width} but {key} declares shape {shape}"
            )
        K, dist = _intrinsics_from_entry(
            entry.get("intrinsic", {}), f"{document_path}:{key}.intrinsic", width, height
        )
        if key not in extrinsics:
            raise KeyError(f"{document_path} is missing extrinsics for {key!r}")
        matrix, checks = validate_transform(extrinsics[key], f"{document_path}:extrinsics.{key}")
        transform_checks[key] = checks
        depths[view_name] = RawDepth(
            name=key,
            view=view_name,
            depth_m=depth_m,
            valid=valid,
            K=K,
            dist=dist,
            base_T_cam=matrix,
            unit_scale_to_m=DEPTH_UNIT_SCALE_TO_M,
            invalid_values=DEPTH_INVALID_VALUES,
        )

    joint_positions = document.get("joint_positions_rad")
    if not isinstance(joint_positions, dict) or not joint_positions:
        raise ValueError(f"{document_path} carries no joint_positions_rad")
    joint_positions = {str(k): float(v) for k, v in joint_positions.items()}
    if not all(np.isfinite(list(joint_positions.values()))):
        raise ValueError(f"{document_path} has non-finite joint positions")

    return RawCapture(
        profile=profile,
        capture=capture,
        capture_dir=snapshot_dir,
        views=views,
        depths=depths,
        world_frame="base_link",
        joint_positions=joint_positions,
        provenance={
            "layout": "g2_snapshot",
            "extrinsics_document": str(document_path.resolve()),
            "created_at": document.get("created_at"),
            "capture_script": document.get("script"),
            "pose_source": document.get("pose_source"),
            "urdf": document.get("urdf"),
            "camera_axes": "OpenCV RDF: +X right, +Y down, +Z forward",
            "matrix_direction": "camera_to_world",
            "world_frame": "base_link",
            "translation_unit": "meter",
            "rigid_transform_checks": transform_checks,
            "kinematic_validation": kinematics,
            "synchronisation": synchronisation,
            "missing_joints_using_zero": document.get("missing_joints_using_zero", []),
            "sensor_vs_urdf_camera_link_info": document.get("validation", {}).get(
                "sensor_calibration_vs_urdf_visual_camera_link_info"
            ),
        },
    )
