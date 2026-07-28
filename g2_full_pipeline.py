#!/usr/bin/env python3
"""Command orchestration for the G2 capture-to-three-version pipeline."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from pipeline_gui import PipelineConfig, build_pipeline_commands
from pose_export import MODEL_RELATIVE_HEAD_ANCHORED_BASELINE_SCALED


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
DEFAULT_CAPTURE_SCRIPT = WORKSPACE_ROOT / "g2_four_camera_extrinsic_capture.py"
DEFAULT_G2_ROOT = WORKSPACE_ROOT / "G2"
DEFAULT_AVOID_ROOT = WORKSPACE_ROOT / "Avoid"
DEFAULT_TABLE_XY_BOUNDS = (0.239, 1.019, -0.694, 0.706)

CAPTURE_REFERENCE_TOKENS = (
    "get_nearest_image",
    "get_joint_states",
    "compute_camera_extrinsics",
    "extrinsic_end_T_head_front_rgbd.json",
    "extrinsic_end_T_hand_left_rgbd.json",
    "extrinsic_end_T_hand_right_rgbd.json",
    "camera_extrinsics.json",
    "base_T_camera",
)
REQUIRED_CAPTURE_FILES = (
    "camera_extrinsics.json",
    "head_rgb.png",
    "hand_left_rgb.png",
    "hand_right_rgb.png",
    "head_depth_raw16.png",
)
FINAL_VARIANT_FILENAMES = {
    "depth_only": ("depth_only_voxels.npz", "depth_only_voxels.glb"),
    "fused": ("fused_voxels.npz", "fused_voxels.glb"),
    "mapanything_only": (
        "mapanything_only_voxels.npz",
        "mapanything_only_voxels.glb",
    ),
}


@dataclasses.dataclass(frozen=True)
class TableBounds:
    x_min: float = DEFAULT_TABLE_XY_BOUNDS[0]
    x_max: float = DEFAULT_TABLE_XY_BOUNDS[1]
    y_min: float = DEFAULT_TABLE_XY_BOUNDS[2]
    y_max: float = DEFAULT_TABLE_XY_BOUNDS[3]

    def validate(self) -> None:
        values = self.as_tuple()
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Table bounds must be finite")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Table bounds require X_MIN < X_MAX and Y_MIN < Y_MAX")

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x_min, self.x_max, self.y_min, self.y_max


@dataclasses.dataclass(frozen=True)
class G2FullPipelineConfig:
    run_root: Path
    captures: tuple[str, ...] = ()
    table_bounds: TableBounds = dataclasses.field(default_factory=TableBounds)
    capture_script: Path = DEFAULT_CAPTURE_SCRIPT
    g2_root: Path = DEFAULT_G2_ROOT
    avoid_root: Path = DEFAULT_AVOID_ROOT
    pipeline_root: Path = SCRIPT_DIR
    device: str = "cuda"
    max_radius_m: float = 2.3
    voxel_size_m: float = 0.01
    hand_max_depth_m: float = 1.0
    minimum_depth_m: float = 0.15
    maximum_depth_m: float = 3.0
    cluster_eps_m: float = 0.03
    minimum_cluster_voxels: int = 24
    table_thickness_m: float = 0.06
    snap_distance_m: float = 0.03
    surface_tolerance_m: float = 0.04
    reuse_preprocessed: bool = False

    @property
    def input_root(self) -> Path:
        return self.run_root / "in"

    @property
    def map_root(self) -> Path:
        return self.run_root / "map"

    @property
    def depth_root(self) -> Path:
        return self.run_root / "depth"

    @property
    def versions_root(self) -> Path:
        return self.run_root / "versions"

    @property
    def sensor_dir(self) -> Path:
        return self.g2_root / "G2_parameters" / "sensor"

    @property
    def urdf(self) -> Path:
        return (
            self.g2_root
            / "G2_parameters"
            / "G2_t2_crs_omnipicker"
            / "urdf"
            / "G2_t2_crs_omnipicker.urdf"
        )

    def validate(self, *, require_captures: bool = False) -> None:
        self.table_bounds.validate()
        positive = {
            "max_radius_m": self.max_radius_m,
            "voxel_size_m": self.voxel_size_m,
            "hand_max_depth_m": self.hand_max_depth_m,
            "minimum_depth_m": self.minimum_depth_m,
            "maximum_depth_m": self.maximum_depth_m,
            "cluster_eps_m": self.cluster_eps_m,
            "table_thickness_m": self.table_thickness_m,
            "snap_distance_m": self.snap_distance_m,
            "surface_tolerance_m": self.surface_tolerance_m,
        }
        for label, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be a positive finite value")
        if self.minimum_depth_m >= self.maximum_depth_m:
            raise ValueError("minimum_depth_m must be smaller than maximum_depth_m")
        if self.minimum_cluster_voxels <= 0:
            raise ValueError("minimum_cluster_voxels must be positive")
        for label, path in (
            ("Pipeline checkout", self.pipeline_root),
            ("Avoid checkout", self.avoid_root),
            ("G2 parameter root", self.g2_root),
            ("sensor directory", self.sensor_dir),
        ):
            if not path.expanduser().is_dir():
                raise ValueError(f"{label} does not exist: {path}")
        for label, path in (("G2 URDF", self.urdf),):
            if not path.expanduser().is_file():
                raise ValueError(f"{label} does not exist: {path}")
        if require_captures and not self.captures:
            raise ValueError("Select at least one capture")
        for capture in self.captures:
            capture_dir = self.input_root / capture
            missing = [
                filename
                for filename in REQUIRED_CAPTURE_FILES
                if not (capture_dir / filename).is_file()
            ]
            if missing:
                raise ValueError(
                    f"Capture {capture} is incomplete; missing: {', '.join(missing)}"
                )

    def final_outputs(self, capture: str) -> dict[str, Path]:
        root = self.versions_root / capture
        return {
            key: root / filenames[1]
            for key, filenames in FINAL_VARIANT_FILENAMES.items()
        }

    def final_voxel_outputs(self, capture: str) -> dict[str, Path]:
        root = self.versions_root / capture
        return {
            key: root / filenames[0]
            for key, filenames in FINAL_VARIANT_FILENAMES.items()
        }

    def manifest(self) -> dict[str, Any]:
        capture_reference_sha256 = None
        if self.capture_script.is_file():
            capture_reference_sha256 = sha256_file(self.capture_script)
        return {
            "schema_version": 1,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "workflow": "g2_capture_mapanything_depth_fusion_three_cropped_versions",
            "paths": {
                "run_root": str(self.run_root.expanduser().resolve()),
                "input_root": str(self.input_root.expanduser().resolve()),
                "map_root": str(self.map_root.expanduser().resolve()),
                "depth_root": str(self.depth_root.expanduser().resolve()),
                "versions_root": str(self.versions_root.expanduser().resolve()),
                "capture_reference_script": str(
                    self.capture_script.expanduser().resolve()
                ),
                "capture_reference_sha256": capture_reference_sha256,
                "g2_root": str(self.g2_root.expanduser().resolve()),
                "avoid_root": str(self.avoid_root.expanduser().resolve()),
                "pipeline_root": str(self.pipeline_root.expanduser().resolve()),
            },
            "captures": list(self.captures),
            "capture_contract": {
                "pose_source": "fk",
                "timestamp_anchor": "head_depth_nearest_head_rgb",
                "extrinsic_direction": "base_T_camera",
                "joint_source": "live_gdk_joint_states",
            },
            "inference": {
                "robot": "g2",
                "feed_metric_depth": True,
                "depth_holdout": 0.0,
                "pose_export_mode": (
                    MODEL_RELATIVE_HEAD_ANCHORED_BASELINE_SCALED
                ),
                "view_order": ["head", "hand_left", "hand_right"],
                "hand_max_depth_m": self.hand_max_depth_m,
                "roll_normalize": False,
                "self_mask_input": False,
                "reuse_preprocessed": self.reuse_preprocessed,
                "max_radius_m": self.max_radius_m,
                "memory_efficient_inference": True,
                "device": self.device,
            },
            "postprocess": {
                "voxel_size_m": self.voxel_size_m,
                "table_xy_bounds_base_link_m": list(
                    self.table_bounds.as_tuple()
                ),
                "cluster_eps_m": self.cluster_eps_m,
                "minimum_cluster_voxels": self.minimum_cluster_voxels,
                "table_thickness_m": self.table_thickness_m,
                "gripper_removal": "operator_proxy_boxes_from_wrist_camera_poses",
                "fusion_method": "occupancy",
                "snap_distance_m": self.snap_distance_m,
                "surface_tolerance_m": self.surface_tolerance_m,
                "viewer_tint_strength": 0.0,
                "show_gripper_removal_shell": False,
            },
            "final_outputs": {
                capture: {
                    key: {
                        "npz": str(self.final_voxel_outputs(capture)[key].resolve()),
                        "glb": str(glb_path.resolve()),
                    }
                    for key, glb_path in self.final_outputs(capture).items()
                }
                for capture in self.captures
            },
            "final_output_contract": (
                "all three versions use the same measured table XY crop, "
                "gripper removal, DBSCAN denoise and below-table crop"
            ),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_g2_captures(input_root: Path) -> list[str]:
    root = input_root.expanduser()
    if not root.is_dir():
        return []
    return sorted(
        child.name
        for child in root.iterdir()
        if child.is_dir()
        and all((child / filename).is_file() for filename in REQUIRED_CAPTURE_FILES)
    )


def validate_capture_reference(path: Path) -> str:
    """Fail closed if the selected capture script is not the requested design."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Capture reference script does not exist: {resolved}")
    source = resolved.read_text(encoding="utf-8")
    missing = [token for token in CAPTURE_REFERENCE_TOKENS if token not in source]
    if missing:
        raise ValueError(
            "Capture reference does not implement the required G2 contract; "
            f"missing tokens: {', '.join(missing)}"
        )
    return sha256_file(resolved)


def build_capture_command(
    config: G2FullPipelineConfig,
    *,
    python_executable: str | Path = sys.executable,
) -> tuple[str, list[str]]:
    config.validate(require_captures=False)
    validate_capture_reference(config.capture_script)
    return (
        "four_camera_capture",
        [
            str(python_executable),
            "-u",
            str(config.capture_script.expanduser().resolve()),
            "--save-dir",
            str(config.input_root.expanduser().resolve()),
            "--sensor-dir",
            str(config.sensor_dir.expanduser().resolve()),
            "--urdf",
            str(config.urdf.expanduser().resolve()),
            "--base-link",
            "base_link",
            "--pose-source",
            "fk",
        ],
    )


def build_processing_commands(
    config: G2FullPipelineConfig,
    *,
    python_executable: str | Path = sys.executable,
) -> list[tuple[str, list[str]]]:
    config.validate(require_captures=True)
    py = str(python_executable)
    pipeline_config = PipelineConfig(
        data_root=config.input_root,
        output_root=config.map_root,
        captures=config.captures,
        stages=("undistort", "run_inference", "filter_export", "voxelize"),
        use_metric_poses=True,
        pose_export_mode=MODEL_RELATIVE_HEAD_ANCHORED_BASELINE_SCALED,
        max_radius=config.max_radius_m,
        voxel_size=config.voxel_size_m,
        device=config.device,
        show_scene_markers=True,
        show_gripper_markers=False,
        export_view_colored_glb=True,
        export_per_camera_k_ab_glb=False,
        reuse_preprocessed=config.reuse_preprocessed,
        fast_inference=False,
        robot="g2",
        depth_input=True,
        depth_holdout=0.0,
        self_mask_input=False,
        swap_wrist_views=False,
        hand_max_depth_m=config.hand_max_depth_m,
        view_subset="",
        roll_normalize=False,
    )
    commands = [
        (f"mapanything_{stage}", argv)
        for stage, argv in build_pipeline_commands(
            pipeline_config,
            python_executable=py,
            script_dir=config.pipeline_root,
        )
    ]
    reconstruct_script = config.avoid_root / "scripts" / "reconstruct_depth_voxels.py"
    for capture in config.captures:
        commands.append(
            (
                f"depth_{capture}",
                [
                    py,
                    "-u",
                    str(reconstruct_script),
                    "--input",
                    str(config.map_root / "undistorted" / capture),
                    "--out-root",
                    str(config.depth_root / capture),
                    "--voxel-size",
                    str(config.voxel_size_m),
                    "--min-depth",
                    str(config.minimum_depth_m),
                    "--max-depth",
                    str(config.maximum_depth_m),
                    "--pixel-stride",
                    "1",
                    "--min-points-per-voxel",
                    "1",
                ],
            )
        )

    fuse_command = [
        py,
        "-u",
        str(config.avoid_root / "scripts" / "fuse_depth_and_map.py"),
        "--root",
        str(config.run_root),
        "--out-root",
        str(config.versions_root),
        "--method",
        "occupancy",
        "--sensor-dir",
        str(config.sensor_dir),
        "--pipeline",
        str(config.pipeline_root),
        "--urdf",
        str(config.urdf),
        "--max-depth",
        str(config.maximum_depth_m),
        "--snap-distance",
        str(config.snap_distance_m),
        "--surface-tolerance",
        str(config.surface_tolerance_m),
        "--cluster-eps",
        str(config.cluster_eps_m),
        "--min-cluster",
        str(config.minimum_cluster_voxels),
        "--table-thickness",
        str(config.table_thickness_m),
        "--table-xy-bounds",
        *(str(value) for value in config.table_bounds.as_tuple()),
        "--no-gripper-shell",
        "--tint-strength",
        "0.0",
    ]
    for capture in config.captures:
        fuse_command.extend(("--snapshot", capture))
    commands.append(("export_three_cropped_versions", fuse_command))
    return commands


def prepare_run(config: G2FullPipelineConfig) -> Path:
    config.validate(require_captures=False)
    for path in (
        config.run_root,
        config.input_root,
        config.map_root,
        config.depth_root,
        config.versions_root,
    ):
        path.expanduser().mkdir(parents=True, exist_ok=True)
    manifest_path = config.run_root / "g2_full_pipeline_config.json"
    manifest_path.write_text(
        json.dumps(config.manifest(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def final_outputs_exist(config: G2FullPipelineConfig) -> bool:
    return all(
        path.is_file()
        for capture in config.captures
        for path in (
            *config.final_outputs(capture).values(),
            *config.final_voxel_outputs(capture).values(),
            config.versions_root / capture / "fusion_report.json",
        )
    )


def validate_final_outputs(config: G2FullPipelineConfig) -> dict[str, Any]:
    """Verify every published voxel centre is inside the selected table crop."""

    import numpy as np

    config.validate(require_captures=True)
    if not final_outputs_exist(config):
        raise ValueError("One or more final NPZ, GLB, or fusion reports are missing")
    bounds = config.table_bounds.as_tuple()
    x_min, x_max, y_min, y_max = bounds
    captures: dict[str, Any] = {}
    for capture in config.captures:
        variants: dict[str, Any] = {}
        fusion_report_path = config.versions_root / capture / "fusion_report.json"
        fusion_report = json.loads(fusion_report_path.read_text(encoding="utf-8"))
        for variant, path in config.final_voxel_outputs(capture).items():
            try:
                with np.load(path, allow_pickle=False) as data:
                    indices = np.asarray(data["indices"], dtype=np.float64)
                    origin = np.asarray(data["origin"], dtype=np.float64)
                    voxel_size = float(data["voxel_size"])
                    world_frame = str(data["world_frame"].item())
            except (OSError, ValueError, KeyError) as exc:
                raise ValueError(f"Cannot validate final voxel file {path}: {exc}") from exc
            if (
                indices.ndim != 2
                or indices.shape[1] != 3
                or origin.shape != (3,)
                or not np.isfinite(indices).all()
                or not np.isfinite(origin).all()
                or not math.isfinite(voxel_size)
                or voxel_size <= 0
            ):
                raise ValueError(f"Invalid sparse voxel metadata in {path}")
            if world_frame != "base_link":
                raise ValueError(f"{path} is in {world_frame!r}, expected base_link")
            points = origin + (indices + 0.5) * voxel_size
            if not len(points):
                raise ValueError(f"Refusing to publish empty output: {path}")
            inside = (
                (points[:, 0] >= x_min - 1e-7)
                & (points[:, 0] <= x_max + 1e-7)
                & (points[:, 1] >= y_min - 1e-7)
                & (points[:, 1] <= y_max + 1e-7)
            )
            if not inside.all():
                bad = points[~inside]
                raise ValueError(
                    f"{variant}/{capture} has {len(bad)} voxels outside the table "
                    f"crop; first={bad[0].round(6).tolist()}"
                )
            variants[variant] = {
                "npz": str(path.resolve()),
                "npz_sha256": sha256_file(path),
                "glb": str(config.final_outputs(capture)[variant].resolve()),
                "glb_sha256": sha256_file(config.final_outputs(capture)[variant]),
                "voxel_count": int(len(points)),
                "outside_table_xy_voxels": 0,
                "bounds_base_link_m": np.stack((points.min(0), points.max(0)))
                .round(6)
                .tolist(),
            }
        report_bounds = (
            fusion_report.get("cleanup", {})
            .get("depth", {})
            .get("parameters", {})
            .get("table_xy_bounds")
        )
        if report_bounds != list(bounds):
            raise ValueError(
                f"{capture} fusion report table crop {report_bounds} does not match "
                f"the selected bounds {list(bounds)}"
            )
        captures[capture] = variants
    document = {
        "schema_version": 1,
        "status": "complete",
        "world_frame": "base_link",
        "unit": "meter",
        "table_xy_bounds_base_link_m": list(bounds),
        "all_three_versions_workspace_cropped_and_validated": True,
        "captures": captures,
    }
    path = config.run_root / "three_version_validation.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return document


def command_preview(commands: Iterable[tuple[str, list[str]]]) -> str:
    """Return a readable argv preview without shell-specific quoting semantics."""

    return "\n".join(
        f"[{stage}] " + " ".join(json.dumps(arg) for arg in argv)
        for stage, argv in commands
    )
