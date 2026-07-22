"""Robot-specific capture layouts behind one in-memory representation.

Only the raw-capture reader differs between robots.  ``undistort.py`` writes the
same canonical preprocessed layout for every profile, so inference, filtering
and voxelization stay robot-agnostic.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable

import numpy as np


VIEW_NAMES = ("head", "hand_left", "hand_right")


@dataclasses.dataclass(frozen=True)
class RawView:
    """One raw colour view with its distorted calibration."""

    name: str
    image_bgr: np.ndarray
    K: np.ndarray
    dist: np.ndarray
    base_T_cam: np.ndarray | None
    intrinsic_source: str
    intrinsic_raw: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class RawDepth:
    """A metric depth map expressed along its own camera's optical axis.

    ``depth_m`` is Z-depth in meters with non-finite entries marking invalid
    pixels.  It carries its own intrinsics/extrinsics because depth and colour
    are usually different physical cameras.
    """

    name: str
    view: str
    depth_m: np.ndarray
    valid: np.ndarray
    K: np.ndarray
    dist: np.ndarray
    base_T_cam: np.ndarray
    unit_scale_to_m: float
    invalid_values: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class RawCapture:
    """Robot-agnostic view of one capture before undistortion."""

    profile: "RobotProfile"
    capture: str
    capture_dir: Path
    views: dict[str, RawView]
    depths: dict[str, RawDepth]
    world_frame: str
    joint_positions: dict[str, float] | None
    provenance: dict[str, Any]

    def ordered_views(self) -> list[RawView]:
        return [self.views[name] for name in self.profile.view_names]


@dataclasses.dataclass(frozen=True)
class RobotProfile:
    """Everything that differs between robot models at capture time."""

    name: str
    view_names: tuple[str, ...]
    world_frame: str
    pose_frame_convention: str
    extrinsic_direction: str
    depth_views: tuple[str, ...]
    urdf_hint: str
    loader: Callable[[Path, str], RawCapture]
    discover: Callable[[Path], list[str]]

    @property
    def has_depth(self) -> bool:
        return bool(self.depth_views)

    def load(self, root: Path | str, capture: str) -> RawCapture:
        return self.loader(Path(root), capture)


def _g1_loader(root: Path, capture: str) -> RawCapture:
    """Read a legacy G1 capture folder (flat PNG + per-camera intrinsic JSON)."""
    import cv2

    from capture_contract import (
        IMAGE_TO_INTRINSIC,
        POSES_FILE,
        load_intrinsics,
        validate_pose_document,
    )

    capture_dir = root / capture
    poses: dict[str, np.ndarray] | None = None
    pose_meta: dict[str, Any] = {}
    poses_path = capture_dir / POSES_FILE
    if poses_path.is_file():
        poses, pose_meta = validate_pose_document(poses_path)

    views: dict[str, RawView] = {}
    for name in VIEW_NAMES:
        image_path = capture_dir / f"{name}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        height, width = image.shape[:2]
        intrinsic_path = capture_dir / IMAGE_TO_INTRINSIC[name]
        K, dist, raw = load_intrinsics(intrinsic_path, width, height)
        views[name] = RawView(
            name=name,
            image_bgr=image,
            K=K,
            dist=dist,
            base_T_cam=None if poses is None else poses[name],
            intrinsic_source=str(intrinsic_path.resolve()),
            intrinsic_raw=raw,
        )
    return RawCapture(
        profile=G1_PROFILE,
        capture=capture,
        capture_dir=capture_dir,
        views=views,
        depths={},
        world_frame=pose_meta.get("world_frame", "unknown"),
        joint_positions=None,
        provenance={"pose_contract": pose_meta, "layout": "g1_flat_capture_dir"},
    )


def _g1_discover(root: Path) -> list[str]:
    from capture_contract import discover_raw_captures

    return discover_raw_captures(root)


def _g2_loader(root: Path, capture: str) -> RawCapture:
    from g2_capture import load_g2_snapshot

    return load_g2_snapshot(root / capture, profile=G2_PROFILE, capture=capture)


def _g2_discover(root: Path) -> list[str]:
    from g2_capture import discover_g2_snapshots

    return discover_g2_snapshots(root)


G1_PROFILE = RobotProfile(
    name="g1",
    view_names=VIEW_NAMES,
    world_frame="base_link",
    pose_frame_convention="opencv_rdf_cam2world",
    extrinsic_direction="base_link_T_camera",
    depth_views=(),
    urdf_hint="/home/ck/robot_test/G1.urdf",
    loader=_g1_loader,
    discover=_g1_discover,
)

G2_PROFILE = RobotProfile(
    name="g2",
    view_names=VIEW_NAMES,
    world_frame="base_link",
    pose_frame_convention="opencv_rdf_cam2world",
    extrinsic_direction="base_link_T_camera",
    depth_views=("head",),
    urdf_hint="G2_parameters/G2_t2_crs_omnipicker/urdf/G2_t2_crs_omnipicker.urdf",
    loader=_g2_loader,
    discover=_g2_discover,
)

PROFILES = {profile.name: profile for profile in (G1_PROFILE, G2_PROFILE)}
DEFAULT_PROFILE = "g1"


def get_profile(name: str) -> RobotProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(
            f"Unknown robot profile {name!r}; expected one of {sorted(PROFILES)}"
        ) from None


def detect_profile(root: Path | str, capture: str | None = None) -> RobotProfile:
    """Identify the profile of a capture root by its on-disk layout.

    A G2 session holds ``snapshot_*`` folders carrying ``camera_extrinsics.json``;
    a G1 root holds capture folders with three flat PNGs.  Ambiguous or empty
    roots are an error rather than a silent default.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Capture root does not exist: {root}")
    if capture is not None:
        if (root / capture / "camera_extrinsics.json").is_file():
            return G2_PROFILE
        return G1_PROFILE
    matches = [profile for profile in PROFILES.values() if profile.discover(root)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No G1 capture folders and no G2 snapshots found under {root}"
        )
    raise ValueError(
        f"{root} matches multiple robot layouts ({[m.name for m in matches]}); "
        "pass --robot explicitly"
    )
