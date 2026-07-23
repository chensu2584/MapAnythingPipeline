#!/usr/bin/env python3
"""Tk GUI for running selected MapAnythingPipeline stages in order."""

from __future__ import annotations

import dataclasses
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

from pose_export import (
    CALIBRATED_INPUT,
    DEFAULT_POSE_EXPORT_MODE,
    MODEL_RELATIVE_HEAD_ANCHORED,
    MODEL_RELATIVE_HEAD_ANCHORED_BASELINE_SCALED,
    MODEL_RELATIVE_HEAD_ANCHORED_DEPTH_AFFINE,
    MODEL_RELATIVE_HEAD_ANCHORED_DEPTH_SCALED,
    POSE_EXPORT_MODES,
)


SCRIPT_DIR = Path(__file__).resolve().parent
# The core pipeline every capture goes through.
STAGES = ("undistort", "run_inference", "filter_export", "voxelize")
# Opt-in extras.  They stay out of STAGES so selecting the whole pipeline does
# not silently require a URDF, and so existing callers keep working.
OPTIONAL_STAGES = ("self_mask", "diagnose")
ALL_STAGES = STAGES + OPTIONAL_STAGES
# "auto" is a UI choice only: it is resolved to a concrete robot when the
# config is read, so command construction never has to guess.
ROBOT_CHOICES = ("auto", "g1", "g2")
ROBOT_PROFILES = ("g1", "g2")
RAW_REQUIRED_FILES = {
    "head.png",
    "hand_left.png",
    "hand_right.png",
    "intrinsic_head_front_rgb.json",
    "intrinsic_hand_left_rgb.json",
    "intrinsic_hand_right_rgb.json",
}
# A G2 snapshot is identified by its single extrinsics document instead.
G2_REQUIRED_FILE = "camera_extrinsics.json"


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    data_root: Path
    output_root: Path
    captures: tuple[str, ...]
    stages: tuple[str, ...]
    use_metric_poses: bool = True
    pose_export_mode: str = DEFAULT_POSE_EXPORT_MODE
    max_radius: float | None = None
    voxel_size: float = 0.02
    device: str = "cuda"
    show_scene_markers: bool = True
    show_gripper_markers: bool = True
    export_view_colored_glb: bool = True
    export_per_camera_k_ab_glb: bool = False
    reuse_preprocessed: bool = True
    fast_inference: bool = False
    robot: str = "g1"
    depth_input: bool = False
    depth_holdout: float = 0.0
    self_mask_input: bool = False
    urdf: str = ""
    mask_dilate_px: int = 12
    mask_gripper_shrink: float = 0.7
    swap_wrist_views: bool = False
    hand_max_depth_m: float | None = 1.5
    view_subset: str = ""          # experiment A: e.g. "head,hand_right"; empty = all
    roll_normalize: bool = False   # experiment B


def format_duration(seconds: float) -> str:
    """Format a monotonic elapsed duration for GUI status and logs."""

    total_seconds = max(int(float(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def capture_layout(child: Path) -> str | None:
    """Return the robot layout of one folder, or None when it is not a capture."""

    if all((child / filename).is_file() for filename in RAW_REQUIRED_FILES):
        return "g1"
    if (child / G2_REQUIRED_FILE).is_file():
        return "g2"
    return None


def discover_captures(data_root: Path, robot: str = "auto") -> list[str]:
    """Find raw capture folders without importing heavy pipeline packages."""

    root = data_root.expanduser()
    if not root.is_dir():
        return []
    found = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        layout = capture_layout(child)
        if layout is not None and robot in ("auto", layout):
            found.append(child.name)
    return found


def detect_root_layout(data_root: Path) -> str | None:
    """Return the single layout a folder holds, or None if empty or mixed."""

    root = data_root.expanduser()
    if not root.is_dir():
        return None
    layouts = {
        layout
        for child in root.iterdir()
        if child.is_dir() and (layout := capture_layout(child)) is not None
    }
    return layouts.pop() if len(layouts) == 1 else None


def _append_captures(command: list[str], captures: Iterable[str]) -> None:
    names = tuple(captures)
    if names:
        command.extend(("--captures", *names))


def build_pipeline_commands(
    config: PipelineConfig,
    *,
    python_executable: str | Path = sys.executable,
    script_dir: Path = SCRIPT_DIR,
) -> list[tuple[str, list[str]]]:
    """Build argv-only commands; subprocesses never use a shell."""

    unknown = set(config.stages) - set(ALL_STAGES)
    if unknown:
        raise ValueError(f"Unknown pipeline stages: {sorted(unknown)}")
    if config.voxel_size <= 0:
        raise ValueError("Voxel size must be positive")
    if config.max_radius is not None and config.max_radius <= 0:
        raise ValueError("Max radius must be positive when supplied")
    if config.pose_export_mode not in POSE_EXPORT_MODES:
        raise ValueError(f"Unknown pose export mode: {config.pose_export_mode}")
    if config.robot not in ROBOT_PROFILES:
        raise ValueError(
            f"Robot must be resolved to one of {ROBOT_PROFILES} before building "
            f"commands, got {config.robot!r}"
        )
    if not 0.0 <= config.depth_holdout < 1.0:
        raise ValueError("Depth holdout must be in [0, 1)")
    if not 0.0 < config.mask_gripper_shrink <= 1.0:
        raise ValueError("Gripper shrink must be in (0, 1]")
    if config.hand_max_depth_m is not None and config.hand_max_depth_m <= 0.0:
        raise ValueError("Hand-camera max depth must be positive when set")
    if config.roll_normalize and (config.depth_input or config.self_mask_input):
        raise ValueError(
            "Roll-normalize cannot be combined with feed-depth or hide-robot: "
            "those align pixels to the unrotated image."
        )

    py = str(python_executable)
    data_root = str(config.data_root.expanduser().resolve())
    output_root = str(config.output_root.expanduser().resolve())
    undistorted_root = str((config.output_root.expanduser().resolve() / "undistorted"))
    result: list[tuple[str, list[str]]] = []

    if "undistort" in config.stages:
        command = [
            py,
            "-u",
            str(script_dir / "undistort.py"),
            "--data-root",
            data_root,
            "--output-root",
            output_root,
        ]
        command.extend(("--robot", config.robot))
        _append_captures(command, config.captures)
        if not config.use_metric_poses:
            command.append("--ignore-poses")
        if config.reuse_preprocessed:
            command.append("--reuse-existing")
        result.append(("undistort", command))

    if "self_mask" in config.stages:
        if not config.urdf:
            raise ValueError("The self-occlusion mask stage needs a URDF path")
        command = [
            py,
            "-u",
            str(script_dir / "self_occlusion_mask.py"),
            "--session",
            data_root,
            "--output-root",
            output_root,
            "--urdf",
            str(Path(config.urdf).expanduser()),
            "--robot",
            config.robot,
            "--dilate-px",
            str(config.mask_dilate_px),
        ]
        if config.mask_gripper_shrink != 1.0:
            command.extend(("--shrink-links", f"gripper={config.mask_gripper_shrink}"))
        _append_captures(command, config.captures)
        result.append(("self_mask", command))

    if "run_inference" in config.stages:
        command = [
            py,
            "-u",
            str(script_dir / "run_inference.py"),
            "--input-root",
            undistorted_root,
            "--output-root",
            output_root,
            "--device",
            config.device,
        ]
        _append_captures(command, config.captures)
        if not config.use_metric_poses:
            command.append("--ignore-poses")
        else:
            command.extend(("--pose-export-mode", config.pose_export_mode))
        if config.max_radius is not None and config.use_metric_poses:
            command.extend(("--max_radius", str(config.max_radius)))
        if config.fast_inference:
            command.append("--fast-inference")
        if config.depth_input:
            command.append("--depth-input")
            if config.depth_holdout > 0.0:
                command.extend(("--depth-holdout", str(config.depth_holdout)))
        if config.self_mask_input:
            command.append("--self-mask-input")
        if config.view_subset:
            # Experiment A: reconstruct from a subset (e.g. head,hand_right).
            command.extend(("--view-order", config.view_subset))
        elif config.swap_wrist_views:
            # Feed the wrist views in the opposite order. Outputs stay keyed by
            # name; only the index each camera occupies changes.
            command.extend(("--view-order", "head,hand_right,hand_left"))
        if config.roll_normalize:
            command.append("--roll-normalize")
        if config.hand_max_depth_m is not None:
            # The hand cameras share a short baseline with the head, so their
            # geometry is only trustworthy up close.
            command.extend(("--view-max-depth", f"hand_left={config.hand_max_depth_m}"))
            command.extend(("--view-max-depth", f"hand_right={config.hand_max_depth_m}"))
        result.append(("run_inference", command))

    if "filter_export" in config.stages:
        command = [
            py,
            "-u",
            str(script_dir / "filter_export.py"),
            "--output-root",
            output_root,
        ]
        _append_captures(command, config.captures)
        if config.max_radius is not None:
            command.extend(("--max_radius", str(config.max_radius)))
        if config.show_scene_markers:
            command.append("--show_cameras")
        if config.show_gripper_markers and config.robot == "g1":
            command.append("--show_grippers")
        if config.export_view_colored_glb:
            command.append("--color_by_view")
        if config.export_per_camera_k_ab_glb:
            command.append("--per_camera_k_ab")
        result.append(("filter_export", command))

    if "voxelize" in config.stages:
        command = [
            py,
            "-u",
            str(script_dir / "voxelize.py"),
            "--output-root",
            output_root,
            "--voxel_size",
            str(config.voxel_size),
        ]
        _append_captures(command, config.captures)
        if config.max_radius is not None:
            command.extend(("--max_radius", str(config.max_radius)))
        if config.show_gripper_markers and config.robot == "g1":
            command.append("--show_grippers")
        result.append(("voxelize", command))

    if "diagnose" in config.stages:
        command = [
            py,
            "-u",
            str(script_dir / "diagnose_reconstruction.py"),
            "--session",
            data_root,
            "--output-root",
            output_root,
            "--robot",
            config.robot,
            "--planes",
        ]
        _append_captures(command, config.captures)
        result.append(("diagnose", command))

    return result


def _ui_scale() -> float:
    """Overall interface scale, from PIPELINE_GUI_SCALE (default 1.0).

    One knob for fonts, widget padding, the tk scaling factor and the window
    size, so a smaller display can shrink the whole GUI without editing code:
    ``PIPELINE_GUI_SCALE=0.7 python pipeline_gui.py``.
    """
    try:
        value = float(os.environ.get("PIPELINE_GUI_SCALE", "1.0"))
    except ValueError:
        return 1.0
    return min(max(value, 0.4), 2.5)


class PipelineGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.scale = _ui_scale()
        self.root.title("MapAnything Pipeline")
        self.root.geometry(f"{int(1360 * self.scale)}x{int(920 * self.scale)}")
        self.root.minsize(int(1100 * self.scale), int(760 * self.scale))
        self._configure_style()

        project_root = SCRIPT_DIR.parent
        self.data_root_var = tk.StringVar(value=str(project_root / "TestData"))
        self.output_root_var = tk.StringVar(value=str(project_root / "outputs"))
        self.use_poses_var = tk.BooleanVar(value=True)
        self.pose_export_mode_var = tk.StringVar(value=DEFAULT_POSE_EXPORT_MODE)
        self.pose_mode_var = tk.StringVar()
        self.max_radius_var = tk.StringVar(value="")
        self.voxel_size_var = tk.StringVar(value="0.02")
        self.device_var = tk.StringVar(value="cuda")
        self.show_scene_markers_var = tk.BooleanVar(value=True)
        self.show_gripper_markers_var = tk.BooleanVar(value=True)
        self.export_view_colored_glb_var = tk.BooleanVar(value=True)
        self.export_per_camera_k_ab_glb_var = tk.BooleanVar(value=False)
        self.reuse_preprocessed_var = tk.BooleanVar(value=True)
        self.fast_inference_var = tk.BooleanVar(value=False)
        self.robot_var = tk.StringVar(value="auto")
        self.depth_input_var = tk.BooleanVar(value=False)
        self.depth_holdout_var = tk.StringVar(value="0.3")
        self.self_mask_input_var = tk.BooleanVar(value=False)
        self.urdf_var = tk.StringVar(value="")
        self.mask_dilate_var = tk.StringVar(value="12")
        self.mask_shrink_var = tk.StringVar(value="0.7")
        self.swap_wrist_var = tk.BooleanVar(value=False)
        self.hand_depth_cap_var = tk.BooleanVar(value=True)
        self.hand_depth_m_var = tk.StringVar(value="1.5")
        self.view_subset_var = tk.StringVar(value="")
        self.roll_normalize_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")
        self.elapsed_var = tk.StringVar(value="Elapsed: --:--:--")
        self.stage_vars = {
            # Extras default off: they cost time and the mask stage needs a URDF.
            name: tk.BooleanVar(value=name in STAGES)
            for name in ALL_STAGES
        }

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.process: subprocess.Popen[str] | None = None
        self.stop_requested = False
        self.capture_names: list[str] = []
        self.run_started_at: float | None = None
        self.stage_started_at: float | None = None
        self.active_stage: str | None = None

        self._build_ui()
        self._update_pose_mode_text()
        self.refresh_captures()
        self.root.after(80, self._poll_events)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_style(self) -> None:
        """Use explicit large fonts for both ttk and classic Tk widgets."""

        self.root.tk.call("tk", "scaling", 1.45 * self.scale)
        available = set(tkfont.families(self.root))
        family = next(
            (
                name
                for name in (
                    "Noto Sans CJK SC",
                    "Source Han Sans SC",
                    "WenQuanYi Micro Hei",
                    "DejaVu Sans",
                )
                if name in available
            ),
            str(tkfont.nametofont("TkDefaultFont").actual("family")),
        )
        mono_family = "DejaVu Sans Mono" if "DejaVu Sans Mono" in available else family
        ui_size = max(int(round(16 * self.scale)), 7)
        heading_size = max(int(round(17 * self.scale)), 8)
        log_size = max(int(round(14 * self.scale)), 6)
        self.ui_font = (family, ui_size)
        self.heading_font = (family, heading_size, "bold")
        self.log_font = (mono_family, log_size)

        for font_name in (
            "TkDefaultFont",
            "TkTextFont",
            "TkMenuFont",
            "TkHeadingFont",
            "TkCaptionFont",
            "TkSmallCaptionFont",
            "TkIconFont",
            "TkTooltipFont",
        ):
            try:
                named_font = tkfont.nametofont(font_name)
                named_font.configure(family=family, size=ui_size)
            except tk.TclError:
                pass
        self.root.option_add("*Font", self.ui_font)

        def pad(x, y):
            return (int(round(x * self.scale)), int(round(y * self.scale)))

        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=self.ui_font)
        style.configure("TLabelframe.Label", font=self.heading_font)
        style.configure("TButton", font=self.ui_font, padding=pad(14, 10))
        style.configure("TCheckbutton", font=self.ui_font, padding=pad(5, 5))
        style.configure("TEntry", font=self.ui_font, padding=pad(6, 7))
        style.configure("TCombobox", font=self.ui_font, padding=pad(6, 7))

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)

        folders = ttk.LabelFrame(self.root, text="Folders", padding=12)
        folders.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
        folders.columnconfigure(1, weight=1)
        ttk.Label(folders, text="TestData folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(folders, textvariable=self.data_root_var).grid(
            row=0, column=1, sticky="ew", padx=8
        )
        ttk.Button(folders, text="Browse…", command=self.choose_data_root).grid(
            row=0, column=2
        )
        ttk.Label(folders, text="Output folder").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(folders, textvariable=self.output_root_var).grid(
            row=1, column=1, sticky="ew", padx=8, pady=(8, 0)
        )
        ttk.Button(folders, text="Browse…", command=self.choose_output_root).grid(
            row=1, column=2, pady=(8, 0)
        )

        middle = ttk.Frame(self.root)
        middle.grid(row=1, column=0, sticky="nsew", padx=14, pady=8)
        middle.columnconfigure(0, weight=1)
        middle.columnconfigure(1, weight=1)
        middle.rowconfigure(0, weight=1)

        captures_frame = ttk.LabelFrame(middle, text="Captures", padding=10)
        captures_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        captures_frame.columnconfigure(0, weight=1)
        captures_frame.rowconfigure(0, weight=1)
        self.capture_list = tk.Listbox(
            captures_frame,
            selectmode=tk.EXTENDED,
            exportselection=False,
            height=10,
            font=self.ui_font,
            activestyle="none",
        )
        capture_scroll = ttk.Scrollbar(
            captures_frame, orient="vertical", command=self.capture_list.yview
        )
        self.capture_list.configure(yscrollcommand=capture_scroll.set)
        self.capture_list.grid(row=0, column=0, sticky="nsew")
        capture_scroll.grid(row=0, column=1, sticky="ns")
        capture_buttons = ttk.Frame(captures_frame)
        capture_buttons.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for column in range(3):
            capture_buttons.columnconfigure(column, weight=1)
        ttk.Button(capture_buttons, text="Refresh", command=self.refresh_captures).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(capture_buttons, text="Select all", command=self.select_all_captures).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(capture_buttons, text="Clear", command=lambda: self.capture_list.selection_clear(0, tk.END)).grid(
            row=0, column=2, sticky="ew", padx=(4, 0)
        )

        settings = ttk.LabelFrame(middle, text="Pipeline", padding=12)
        settings.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        settings.columnconfigure(0, weight=1)
        ttk.Label(settings, text="Steps run in this fixed order:").grid(
            row=0, column=0, sticky="w"
        )
        labels = {
            "undistort": "1. Undistort",
            "run_inference": "2. Run inference",
            "filter_export": "3. Filter / export",
            "voxelize": "4. Voxelize",
            "self_mask": "+ Robot self-occlusion mask (needs URDF; runs after 1)",
            "diagnose": "+ Diagnose against the depth camera (runs last)",
        }
        for row, name in enumerate(ALL_STAGES, start=1):
            ttk.Checkbutton(settings, text=labels[name], variable=self.stage_vars[name]).grid(
                row=row, column=0, sticky="w", pady=2
            )
        # A running counter rather than literal row numbers: hand-maintained
        # indices silently stack two widgets in one cell when the list above
        # changes length, and the later one simply hides the earlier.
        next_row = iter(range(len(ALL_STAGES) + 1, 100)).__next__
        ttk.Separator(settings).grid(row=next_row(), column=0, sticky="ew", pady=8)
        ttk.Checkbutton(
            settings,
            text="Use calibrated camera extrinsics as model input",
            variable=self.use_poses_var,
            command=self._update_pose_mode_text,
        ).grid(row=next_row(), column=0, sticky="w")
        pose_selector = ttk.Frame(settings)
        pose_selector.grid(row=next_row(), column=0, sticky="ew", pady=(6, 0))
        pose_selector.columnconfigure(1, weight=1)
        ttk.Label(pose_selector, text="Output geometry").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.pose_export_combo = ttk.Combobox(
            pose_selector,
            textvariable=self.pose_export_mode_var,
            values=POSE_EXPORT_MODES,
            state="readonly",
            width=34,
        )
        self.pose_export_combo.grid(row=0, column=1, sticky="ew")
        self.pose_export_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._update_pose_mode_text()
        )
        ttk.Label(
            settings, textvariable=self.pose_mode_var, wraplength=520, foreground="#7c2d12"
        ).grid(row=next_row(), column=0, sticky="ew", pady=(4, 10))

        options = ttk.Frame(settings)
        options.grid(row=next_row(), column=0, sticky="ew")
        options.columnconfigure(1, weight=1)
        ttk.Label(options, text="Max radius (blank = none)").grid(row=0, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.max_radius_var, width=10).grid(
            row=0, column=1, sticky="w", padx=8
        )
        ttk.Label(options, text="Voxel size").grid(row=1, column=0, sticky="w", pady=(7, 0))
        ttk.Entry(options, textvariable=self.voxel_size_var, width=10).grid(
            row=1, column=1, sticky="w", padx=8, pady=(7, 0)
        )
        ttk.Label(options, text="Inference device").grid(row=2, column=0, sticky="w", pady=(7, 0))
        ttk.Combobox(
            options,
            textvariable=self.device_var,
            values=("cuda", "cuda:0", "cpu"),
            width=10,
        ).grid(row=2, column=1, sticky="w", padx=8, pady=(7, 0))
        ttk.Checkbutton(
            options,
            text="Show small cameras + world origin in GLB",
            variable=self.show_scene_markers_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(9, 0))
        ttk.Checkbutton(
            options,
            text="Show G1 gripper centers in GLB (left orange, right cyan)",
            variable=self.show_gripper_markers_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(7, 0))
        ttk.Checkbutton(
            options,
            text="Export extra red/green/blue view GLB",
            variable=self.export_view_colored_glb_var,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(7, 0))
        ttk.Checkbutton(
            options,
            text="Experimental per-camera K A/B GLBs",
            variable=self.export_per_camera_k_ab_glb_var,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(7, 0))
        ttk.Checkbutton(
            options,
            text="Reuse unchanged undistorted inputs",
            variable=self.reuse_preprocessed_var,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(7, 0))
        ttk.Checkbutton(
            options,
            text="Fast inference dense head (more VRAM; OOM falls back safely)",
            variable=self.fast_inference_var,
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(7, 0))

        robot_row = ttk.Frame(options)
        robot_row.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        ttk.Label(robot_row, text="Robot").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.robot_combo = ttk.Combobox(
            robot_row,
            textvariable=self.robot_var,
            values=ROBOT_CHOICES,
            state="readonly",
            width=8,
        )
        self.robot_combo.grid(row=0, column=1, sticky="w")
        self.robot_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self.refresh_captures()
        )
        ttk.Checkbutton(
            options,
            text="Feed metric depth to the model (G2 only)",
            variable=self.depth_input_var,
            command=self._update_pose_mode_text,
        ).grid(row=10, column=0, columnspan=2, sticky="w", pady=(7, 0))
        ttk.Checkbutton(
            options,
            text="Hide the robot from the model (experiment: does the gripper distort it?)",
            variable=self.self_mask_input_var,
        ).grid(row=12, column=0, columnspan=2, sticky="w", pady=(7, 0))
        urdf_row = ttk.Frame(options)
        urdf_row.grid(row=13, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        urdf_row.columnconfigure(1, weight=1)
        ttk.Label(urdf_row, text="URDF (self-mask stage)").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(urdf_row, textvariable=self.urdf_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(urdf_row, text="Browse", command=self.choose_urdf).grid(row=0, column=2, padx=(6, 0))
        ttk.Label(urdf_row, text="dilate px").grid(row=0, column=3, sticky="w", padx=(10, 6))
        ttk.Entry(urdf_row, textvariable=self.mask_dilate_var, width=6).grid(row=0, column=4)
        ttk.Label(urdf_row, text="gripper shrink").grid(row=0, column=5, sticky="w", padx=(10, 6))
        ttk.Entry(urdf_row, textvariable=self.mask_shrink_var, width=6).grid(row=0, column=6)

        ttk.Checkbutton(
            options,
            text="Swap the two wrist views (is the worst view worst by camera, or by index?)",
            variable=self.swap_wrist_var,
        ).grid(row=14, column=0, columnspan=2, sticky="w", pady=(7, 0))
        hand_depth_row = ttk.Frame(options)
        hand_depth_row.grid(row=15, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        ttk.Checkbutton(
            hand_depth_row,
            text="Cap hand-camera usable depth (m)",
            variable=self.hand_depth_cap_var,
        ).grid(row=0, column=0, sticky="w")
        ttk.Entry(hand_depth_row, textvariable=self.hand_depth_m_var, width=6).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )
        subset_row = ttk.Frame(options)
        subset_row.grid(row=16, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        ttk.Label(subset_row, text="Views subset (exp A, e.g. head,hand_right)").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(subset_row, textvariable=self.view_subset_var, width=22).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )
        ttk.Checkbutton(
            options,
            text="Roll-normalize wrist views to upright (exp B)",
            variable=self.roll_normalize_var,
        ).grid(row=17, column=0, columnspan=2, sticky="w", pady=(7, 0))

        holdout_row = ttk.Frame(options)
        holdout_row.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Label(holdout_row, text="Depth holdout fraction").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        ttk.Entry(holdout_row, textvariable=self.depth_holdout_var, width=8).grid(
            row=0, column=1, sticky="w"
        )

        actions = ttk.Frame(self.root)
        actions.grid(row=2, column=0, sticky="ew", padx=14, pady=8)
        actions.columnconfigure(0, weight=1)
        self.run_button = ttk.Button(actions, text="Run selected pipeline", command=self.start)
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.stop_button = ttk.Button(actions, text="Stop", command=self.stop, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=(6, 0))
        ttk.Label(actions, textvariable=self.status_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(actions, textvariable=self.elapsed_var).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        log_frame = ttk.LabelFrame(self.root, text="Live log", padding=8)
        log_frame.grid(row=3, column=0, sticky="nsew", padx=14, pady=(8, 14))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(
            log_frame,
            wrap="word",
            state="disabled",
            bg="#101216",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            font=self.log_font,
            padx=10,
            pady=10,
        )
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        log_scroll.grid(row=0, column=1, sticky="ns")

    def choose_data_root(self) -> None:
        selected = filedialog.askdirectory(
            title="Choose TestData folder", initialdir=self.data_root_var.get(), parent=self.root
        )
        if selected:
            self.data_root_var.set(selected)
            self.refresh_captures()

    def choose_urdf(self) -> None:
        path = filedialog.askopenfilename(
            title="Select the robot URDF", filetypes=[("URDF", "*.urdf"), ("All files", "*")]
        )
        if path:
            self.urdf_var.set(path)

    def choose_output_root(self) -> None:
        selected = filedialog.askdirectory(
            title="Choose output folder", initialdir=self.output_root_var.get(), parent=self.root
        )
        if selected:
            self.output_root_var.set(selected)

    def refresh_captures(self) -> None:
        self.capture_names = discover_captures(
            Path(self.data_root_var.get()), self.robot_var.get()
        )
        self.capture_list.delete(0, tk.END)
        for name in self.capture_names:
            self.capture_list.insert(tk.END, name)
        self.select_all_captures()
        self.status_var.set(f"Found {len(self.capture_names)} compatible capture(s)")

    def select_all_captures(self) -> None:
        if self.capture_names:
            self.capture_list.selection_set(0, tk.END)

    def _update_pose_mode_text(self) -> None:
        if self.use_poses_var.get():
            self.pose_export_combo.configure(state="readonly")
            if (
                self.pose_export_mode_var.get()
                == MODEL_RELATIVE_HEAD_ANCHORED_BASELINE_SCALED
            ):
                self.pose_mode_var.set(
                    "Recommended: fit one scale from the three calibrated/model camera "
                    "baselines, scale model depth and relative translations together, "
                    "then anchor the calibrated head."
                )
            elif (
                self.pose_export_mode_var.get()
                == MODEL_RELATIVE_HEAD_ANCHORED_DEPTH_SCALED
            ):
                note = (
                    "G2 only: fit that scale per pixel against the metric head depth "
                    "camera instead of three camera baselines, which constrains scene "
                    "depth directly. Falls back to the baseline fit and records why if "
                    "the depth fit cannot be defended."
                )
                if self.depth_input_var.get():
                    note += (
                        "  Depth is also being fed to the model, so the head-view "
                        "agreement is circular: set a holdout fraction above 0."
                    )
                self.pose_mode_var.set(note)
            elif (
                self.pose_export_mode_var.get()
                == MODEL_RELATIVE_HEAD_ANCHORED_DEPTH_AFFINE
            ):
                self.pose_mode_var.set(
                    "DIAGNOSTIC ONLY. Corrects depth as measured = a * model + b, which "
                    "fits the measured error far better than any single scale. But an "
                    "affine correction is not a similarity transform: it stretches the "
                    "scene differently at each range and breaks agreement between the "
                    "three cameras. Use it to measure the error, never to plan against."
                )
            elif self.pose_export_mode_var.get() == MODEL_RELATIVE_HEAD_ANCHORED:
                self.pose_mode_var.set(
                    "Unscaled diagnostic: preserve MapAnything geometry and rigidly anchor "
                    "the head; known captures can remain about 11% too large."
                )
            else:
                self.pose_mode_var.set(
                    "Legacy diagnostic: model depth is reprojected with calibrated poses. "
                    "This hybrid caused the confirmed 170603/170700 separation."
                )
        else:
            self.pose_export_combo.configure(state="disabled")
            self.pose_mode_var.set(
                "RGB-only mode: existing pose files are explicitly ignored; model pose and scale are arbitrary."
            )

    def _read_config(self) -> PipelineConfig:
        data_root = Path(self.data_root_var.get()).expanduser()
        output_root = Path(self.output_root_var.get()).expanduser()
        if not data_root.is_dir():
            raise ValueError(f"TestData folder does not exist: {data_root}")
        output_root.mkdir(parents=True, exist_ok=True)
        selected = tuple(self.capture_names[i] for i in self.capture_list.curselection())
        if not selected:
            raise ValueError("Select at least one compatible capture")
        stages = tuple(name for name in ALL_STAGES if self.stage_vars[name].get())
        if not stages:
            raise ValueError("Select at least one pipeline step")
        radius_text = self.max_radius_var.get().strip()
        max_radius = float(radius_text) if radius_text else None
        voxel_size = float(self.voxel_size_var.get())
        device = self.device_var.get().strip()
        if not device:
            raise ValueError("Inference device cannot be empty")
        holdout_text = self.depth_holdout_var.get().strip()
        depth_holdout = float(holdout_text) if holdout_text else 0.0
        # Resolve "auto" here so command construction never has to guess which
        # robot-specific flags apply.
        robot = self.robot_var.get()
        if robot == "auto":
            robot = detect_root_layout(data_root) or "g1"
        return PipelineConfig(
            data_root=data_root,
            output_root=output_root,
            captures=selected,
            stages=stages,
            use_metric_poses=self.use_poses_var.get(),
            pose_export_mode=self.pose_export_mode_var.get(),
            max_radius=max_radius,
            voxel_size=voxel_size,
            device=device,
            show_scene_markers=self.show_scene_markers_var.get(),
            show_gripper_markers=self.show_gripper_markers_var.get(),
            export_view_colored_glb=self.export_view_colored_glb_var.get(),
            export_per_camera_k_ab_glb=(
                self.export_per_camera_k_ab_glb_var.get()
            ),
            reuse_preprocessed=self.reuse_preprocessed_var.get(),
            fast_inference=self.fast_inference_var.get(),
            robot=robot,
            depth_input=self.depth_input_var.get(),
            depth_holdout=depth_holdout,
            self_mask_input=self.self_mask_input_var.get(),
            urdf=self.urdf_var.get().strip(),
            mask_dilate_px=int(self.mask_dilate_var.get() or 12),
            mask_gripper_shrink=float(self.mask_shrink_var.get() or 1.0),
            swap_wrist_views=self.swap_wrist_var.get(),
            view_subset=self.view_subset_var.get().strip(),
            roll_normalize=self.roll_normalize_var.get(),
            hand_max_depth_m=(
                float(self.hand_depth_m_var.get() or 1.5)
                if self.hand_depth_cap_var.get()
                else None
            ),
        )

    def start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            config = self._read_config()
            commands = build_pipeline_commands(config)
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("Invalid pipeline configuration", str(exc), parent=self.root)
            return
        self.stop_requested = False
        self.run_started_at = time.perf_counter()
        self.stage_started_at = None
        self.active_stage = None
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Running…")
        self._append_log("\n=== New pipeline run ===\n")
        self.worker = threading.Thread(
            target=self._run_commands, args=(commands,), name="mapanything-pipeline", daemon=True
        )
        self.worker.start()

    def _run_commands(self, commands: list[tuple[str, list[str]]]) -> None:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        pipeline_started_at = time.perf_counter()
        try:
            for stage, command in commands:
                if self.stop_requested:
                    raise InterruptedError("Stopped before next stage")
                stage_started_at = time.perf_counter()
                self.events.put(("stage", (stage, stage_started_at)))
                self.events.put(("log", "$ " + shlex.join(command) + "\n"))
                self.process = subprocess.Popen(
                    command,
                    cwd=SCRIPT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                assert self.process.stdout is not None
                for line in self.process.stdout:
                    self.events.put(("log", line))
                return_code = self.process.wait()
                self.process = None
                if self.stop_requested:
                    raise InterruptedError(f"Stopped during {stage}")
                if return_code != 0:
                    raise RuntimeError(f"{stage} failed with exit code {return_code}")
                self.events.put(
                    ("stage_done", (stage, time.perf_counter() - stage_started_at))
                )
            self.events.put(("done", time.perf_counter() - pipeline_started_at))
        except Exception as exc:
            self.events.put(
                ("failed", (str(exc), time.perf_counter() - pipeline_started_at))
            )
        finally:
            self.process = None

    def stop(self) -> None:
        self.stop_requested = True
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
        self.status_var.set("Stopping…")

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "stage":
                    stage, started_at = payload
                    self.active_stage = str(stage)
                    self.stage_started_at = float(started_at)
                    self.status_var.set(f"Running: {stage}")
                elif kind == "stage_done":
                    stage, elapsed = payload
                    self._append_log(
                        f"\n[TIMING] {stage}: {format_duration(float(elapsed))} "
                        f"({float(elapsed):.3f} s)\n"
                    )
                elif kind == "done":
                    total = float(payload)
                    self.status_var.set(
                        f"Pipeline completed successfully in {format_duration(total)}"
                    )
                    self.elapsed_var.set(
                        f"Total: {format_duration(total)} ({total:.3f} s)"
                    )
                    self.run_started_at = None
                    self.stage_started_at = None
                    self.active_stage = None
                    self.run_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                elif kind == "failed":
                    message, total = payload
                    self.status_var.set(f"Pipeline stopped/failed: {message}")
                    self.elapsed_var.set(
                        f"Stopped after: {format_duration(float(total))} "
                        f"({float(total):.3f} s)"
                    )
                    self._append_log(f"\nERROR: {message}\n")
                    self.run_started_at = None
                    self.stage_started_at = None
                    self.active_stage = None
                    self.run_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
        except queue.Empty:
            pass
        self._refresh_elapsed_display()
        self.root.after(80, self._poll_events)

    def _refresh_elapsed_display(self) -> None:
        if self.run_started_at is None:
            return
        now = time.perf_counter()
        total = now - self.run_started_at
        if self.active_stage is not None and self.stage_started_at is not None:
            stage_elapsed = now - self.stage_started_at
            self.elapsed_var.set(
                f"{self.active_stage}: {format_duration(stage_elapsed)} | "
                f"pipeline total: {format_duration(total)}"
            )
        else:
            self.elapsed_var.set(f"Pipeline total: {format_duration(total)}")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def on_close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno(
                "Pipeline is running", "Stop the active process and close?", parent=self.root
            ):
                return
            self.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    PipelineGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
