#!/usr/bin/env python3
"""Resolve G1 gripper-center poses in a reconstructed ``base_link`` frame.

The capture converter already records ``base_T_Link_hand_l/r`` after applying
the saved robot state and the production A2D FK chain.  This module appends the
fixed gripper-center displacement from robot_test's Omnipicker URDF.  It also
reconstructs the hand mount independently from the captured WBC Link7 pose and
``G1.urdf`` and rejects a mismatched frame chain.

Only the Omnipicker center displacement is reused.  Its complete arm/base model
is intentionally not mixed into the production G1/A2D chain.
"""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


DEFAULT_G1_URDF = Path("/home/ck/robot_test/G1.urdf")
DEFAULT_GRIPPER_URDF = Path(
    "/home/ck/robot_test/G1_URDF_Omnipicker/urdf/G1/"
    "G1_omnipicker_omnipicker.urdf"
)

SIDE_CONFIG = {
    "left": {
        "view": "hand_left",
        "link7": "arm_left_link7",
        "g1_hand": "hand_left_base_link",
        "a2d_hand": "Link_hand_l",
        "center_joint": "idx52_gripper_l_center_joint",
    },
    "right": {
        "view": "hand_right",
        "link7": "arm_right_link7",
        "g1_hand": "hand_right_base_link",
        "a2d_hand": "Link_hand_r",
        "center_joint": "idx92_gripper_r_center_joint",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numbers(text: str | None, *, count: int, default: tuple[float, ...]) -> np.ndarray:
    values = np.asarray(
        default if text is None else [float(value) for value in text.split()],
        dtype=np.float64,
    )
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError(f"Expected {count} finite values, got {text!r}")
    return values


def origin_matrix(origin: ET.Element | None) -> np.ndarray:
    """Return the URDF parent-to-child fixed transform for an ``origin``."""
    if origin is None:
        xyz = np.zeros(3, dtype=np.float64)
        rpy = np.zeros(3, dtype=np.float64)
    else:
        xyz = _numbers(origin.get("xyz"), count=3, default=(0.0, 0.0, 0.0))
        rpy = _numbers(origin.get("rpy"), count=3, default=(0.0, 0.0, 0.0))
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rz @ ry @ rx
    transform[:3, 3] = xyz
    return transform


def _load_urdf(path: Path) -> ET.Element:
    try:
        return ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"Cannot read URDF {path}: {exc}") from exc


def fixed_joint_transform(
    urdf_root: ET.Element,
    *,
    parent: str | None = None,
    child: str | None = None,
    joint_name: str | None = None,
) -> np.ndarray:
    """Find one fixed URDF joint and return its parent-to-child transform."""
    matches = []
    for joint in urdf_root.findall("joint"):
        parent_node = joint.find("parent")
        child_node = joint.find("child")
        if joint_name is not None and joint.get("name") != joint_name:
            continue
        if parent is not None and (
            parent_node is None or parent_node.get("link") != parent
        ):
            continue
        if child is not None and (child_node is None or child_node.get("link") != child):
            continue
        matches.append(joint)
    if len(matches) != 1:
        selector = f"joint={joint_name!r}, parent={parent!r}, child={child!r}"
        raise ValueError(f"Expected one URDF joint for {selector}; found {len(matches)}")
    joint = matches[0]
    if joint.get("type") != "fixed":
        raise ValueError(f"URDF joint {joint.get('name')} must be fixed")
    return origin_matrix(joint.find("origin"))


def _matrix(value: object, *, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{label} has an invalid homogeneous bottom row")
    if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError(f"{label} rotation is not orthonormal")
    return matrix


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _read_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def resolve_gripper_poses(
    capture_dir: str | Path,
    *,
    g1_urdf: str | Path = DEFAULT_G1_URDF,
    gripper_urdf: str | Path = DEFAULT_GRIPPER_URDF,
    max_mount_translation_error_m: float = 5e-4,
    max_mount_rotation_error_deg: float = 0.05,
) -> dict:
    """Return left/right gripper-center poses and full provenance.

    The result is suitable for JSON serialization.  ``pose_matrix`` uses the
    production hand-mount orientation and places its origin at the gripper
    center.  The center position is independent of the legacy URDF connector's
    differing yaw because the fixed displacement lies on local +Z.
    """
    capture_dir = Path(capture_dir)
    g1_urdf = Path(g1_urdf)
    gripper_urdf = Path(gripper_urdf)
    state_path = capture_dir / "capture_state.json"
    manifest_path = capture_dir / "pose_conversion_manifest.json"
    export_pose_path = capture_dir / "camera_poses_used_for_export.json"

    state = _read_json(state_path)
    manifest = _read_json(manifest_path)
    if manifest.get("output_contract", {}).get("world_frame") != "base_link":
        raise ValueError("Gripper overlay requires pose manifest world_frame=base_link")
    if export_pose_path.is_file():
        export_poses = _read_json(export_pose_path)
        if export_poses.get("world_frame") != "base_link":
            raise ValueError("Gripper overlay requires reconstruction world_frame=base_link")

    wbc = state.get("wbc_link7_capture", {})
    if wbc.get("world_frame") != "base_link" or wbc.get("pose_direction") != "base_T_frame":
        raise ValueError("capture_state WBC poses must be base_T_frame in base_link")
    if not wbc.get("complete"):
        raise ValueError("capture_state does not contain a complete WBC Link7 snapshot")

    base_t_parent = manifest.get("intermediate_transforms", {}).get("base_T_parent", {})
    g1_root = _load_urdf(g1_urdf)
    gripper_root = _load_urdf(gripper_urdf)
    poses = {}
    for side, config in SIDE_CONFIG.items():
        entry = base_t_parent.get(config["view"], {})
        if entry.get("parent") != "base_link" or entry.get("child") != config["a2d_hand"]:
            raise ValueError(
                f"Pose manifest is missing base_link_T_{config['a2d_hand']} for {side}"
            )
        base_t_hand = _matrix(entry.get("matrix"), label=f"base_T_{config['a2d_hand']}")

        view_state = wbc.get("views", {}).get(config["view"], {})
        frame = view_state.get("frames", {}).get(config["link7"], {})
        base_t_link7 = _matrix(
            frame.get("base_T_frame"), label=f"WBC base_T_{config['link7']}"
        )
        link7_t_hand = fixed_joint_transform(
            g1_root, parent=config["link7"], child=config["g1_hand"]
        )
        base_t_hand_wbc = base_t_link7 @ link7_t_hand
        translation_error_m = float(
            np.linalg.norm(base_t_hand_wbc[:3, 3] - base_t_hand[:3, 3])
        )
        rotation_error_deg = _rotation_error_deg(base_t_hand_wbc, base_t_hand)
        if translation_error_m > max_mount_translation_error_m:
            raise ValueError(
                f"{side} WBC/G1 hand mount differs from converter by "
                f"{translation_error_m * 1000.0:.3f} mm"
            )
        if rotation_error_deg > max_mount_rotation_error_deg:
            raise ValueError(
                f"{side} WBC/G1 hand mount differs from converter by "
                f"{rotation_error_deg:.4f} deg"
            )

        base_t_center_legacy = fixed_joint_transform(
            gripper_root, joint_name=config["center_joint"]
        )
        center_offset = base_t_center_legacy[:3, 3]
        if abs(center_offset[0]) > 1e-9 or abs(center_offset[1]) > 1e-9:
            raise ValueError(
                f"{config['center_joint']} is not a pure local-Z center displacement"
            )
        base_t_center = base_t_hand.copy()
        base_t_center[:3, 3] = (
            base_t_hand @ np.r_[center_offset, 1.0]
        )[:3]
        poses[side] = {
            "frame": f"gripper_{side}_center",
            "position_m": base_t_center[:3, 3].tolist(),
            "pose_matrix": base_t_center.tolist(),
            "mount_frame": config["a2d_hand"],
            "mount_pose_matrix": base_t_hand.tolist(),
            "mount_to_center_translation_m": center_offset.tolist(),
            "mount_crosscheck": {
                "wbc_frame": config["link7"],
                "g1_hand_frame": config["g1_hand"],
                "translation_error_m": translation_error_m,
                "rotation_error_deg": rotation_error_deg,
            },
        }

    return {
        "schema_version": 1,
        "world_frame": "base_link",
        "matrix_direction": "world_T_frame",
        "translation_unit": "meter",
        "semantic": "approximate center between gripper fingers / tool-center marker",
        "poses": poses,
        "assumptions": [
            "Production base_T_Link_hand_l/r comes from the capture pose manifest.",
            "G1.urdf plus captured WBC Link7 poses independently validates each hand mount.",
            "Only the Omnipicker URDF's 0.14308 m local-Z gripper-center displacement is reused.",
            "The legacy Omnipicker arm/base chain and connector yaw are not used.",
        ],
        "sources": {
            "capture_state": str(state_path.resolve()),
            "pose_conversion_manifest": str(manifest_path.resolve()),
            "camera_poses_used_for_export": (
                str(export_pose_path.resolve()) if export_pose_path.is_file() else None
            ),
            "g1_urdf": {"path": str(g1_urdf.resolve()), "sha256": _sha256(g1_urdf)},
            "gripper_urdf": {
                "path": str(gripper_urdf.resolve()),
                "sha256": _sha256(gripper_urdf),
            },
        },
    }


def write_gripper_poses(path: str | Path, result: dict) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
