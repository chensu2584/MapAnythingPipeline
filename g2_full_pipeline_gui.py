#!/usr/bin/env python3
"""Responsive Tk GUI for G2 capture, reconstruction and cropped exports."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

from g2_full_pipeline import (
    DEFAULT_AVOID_ROOT,
    DEFAULT_CAPTURE_SCRIPT,
    DEFAULT_G2_ROOT,
    G2FullPipelineConfig,
    TableBounds,
    WORKSPACE_ROOT,
    build_capture_command,
    build_processing_commands,
    command_preview,
    discover_g2_captures,
    final_outputs_exist,
    prepare_run,
    validate_final_outputs,
)


def _ui_scale() -> float:
    try:
        value = float(os.environ.get("G2_FULL_PIPELINE_GUI_SCALE", "1.0"))
    except ValueError:
        value = 1.0
    return min(max(value, 0.65), 1.8)


def default_run_root() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return DEFAULT_G2_ROOT / "7.24Exp" / "gui_pipeline_runs" / f"run_{stamp}"


class G2FullPipelineGui:
    def __init__(self, root: tk.Tk, *, run_root: Path | None = None):
        self.root = root
        self.scale = _ui_scale()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.process: subprocess.Popen[str] | None = None
        self.stop_requested = False
        self.active_job = ""
        self.job_started_at: float | None = None
        self.capture_names: list[str] = []
        self.stage_items: dict[str, str] = {}

        self.root.title("G2 三版本重建 Pipeline")
        self._configure_window()
        self._configure_style()

        self.run_root_var = tk.StringVar(value=str(run_root or default_run_root()))
        self.capture_script_var = tk.StringVar(value=str(DEFAULT_CAPTURE_SCRIPT))
        self.g2_root_var = tk.StringVar(value=str(DEFAULT_G2_ROOT))
        self.avoid_root_var = tk.StringVar(value=str(DEFAULT_AVOID_ROOT))
        self.capture_python_var = tk.StringVar(
            value=os.environ.get("G2_CAPTURE_PYTHON", sys.executable)
        )
        self.processing_python_var = tk.StringVar(value=sys.executable)
        self.device_var = tk.StringVar(value="cuda")
        self.x_min_var = tk.StringVar(value="0.239")
        self.x_max_var = tk.StringVar(value="1.019")
        self.y_min_var = tk.StringVar(value="-0.694")
        self.y_max_var = tk.StringVar(value="0.706")
        self.reuse_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="就绪")
        self.elapsed_var = tk.StringVar(value="00:00:00")

        self._build_ui()
        self.refresh_captures()
        self.root.after(100, self._poll_events)
        self.root.after(1500, self._periodic_refresh)
        self.root.after(500, self._update_elapsed)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_window(self) -> None:
        self.root.update_idletasks()
        screen_w = max(self.root.winfo_screenwidth(), 800)
        screen_h = max(self.root.winfo_screenheight(), 600)
        width = min(int(1220 * self.scale), screen_w - 60)
        height = min(int(820 * self.scale), screen_h - 90)
        width = max(width, min(900, screen_w - 30))
        height = max(height, min(620, screen_h - 50))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(min(860, width), min(560, height))

    def _configure_style(self) -> None:
        self.root.tk.call("tk", "scaling", 1.15 * self.scale)
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
            "TkDefaultFont",
        )
        mono = "DejaVu Sans Mono" if "DejaVu Sans Mono" in available else family
        size = max(9, int(round(10 * self.scale)))
        self.ui_font = (family, size)
        self.heading_font = (family, size + 1, "bold")
        self.log_font = (mono, max(8, size - 1))
        self.root.option_add("*Font", self.ui_font)

        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=self.ui_font)
        style.configure("TLabelframe.Label", font=self.heading_font)
        style.configure("Treeview", rowheight=max(24, int(27 * self.scale)))
        style.configure("Treeview.Heading", font=self.heading_font)
        style.configure(
            "TButton",
            padding=(max(7, int(9 * self.scale)), max(4, int(6 * self.scale))),
        )

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        paths = ttk.LabelFrame(self.root, text="运行路径", padding=8)
        paths.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        paths.columnconfigure(1, weight=1)
        rows = (
            ("运行目录", self.run_root_var, self.choose_run_root, True),
            ("采集参考脚本", self.capture_script_var, self.choose_capture_script, False),
            ("采集 Python", self.capture_python_var, self.choose_capture_python, False),
            ("处理 Python", self.processing_python_var, self.choose_processing_python, False),
            ("G2 参数目录", self.g2_root_var, self.choose_g2_root, True),
            ("Avoid 仓库", self.avoid_root_var, self.choose_avoid_root, True),
        )
        for row, (label, variable, callback, directory) in enumerate(rows):
            ttk.Label(paths, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(paths, textvariable=variable).grid(
                row=row, column=1, sticky="ew", padx=7, pady=2
            )
            ttk.Button(paths, text="选择", command=callback).grid(
                row=row, column=2, pady=2
            )
            del directory

        toolbar = ttk.Frame(self.root, padding=(10, 3))
        toolbar.grid(row=1, column=0, sticky="ew")
        self.capture_button = ttk.Button(
            toolbar, text="启动四相机采集", command=self.start_capture
        )
        self.capture_button.pack(side="left")
        self.run_button = ttk.Button(
            toolbar, text="运行选中 Snapshot", command=self.start_pipeline
        )
        self.run_button.pack(side="left", padx=(7, 0))
        self.stop_button = ttk.Button(
            toolbar, text="停止", command=self.stop_job, state="disabled"
        )
        self.stop_button.pack(side="left", padx=(7, 0))
        ttk.Button(toolbar, text="查看命令", command=self.show_command_preview).pack(
            side="left", padx=(7, 0)
        )
        ttk.Button(toolbar, text="打开结果目录", command=self.open_versions_root).pack(
            side="left", padx=(7, 0)
        )
        ttk.Label(toolbar, textvariable=self.elapsed_var).pack(side="right")
        ttk.Label(toolbar, textvariable=self.status_var).pack(
            side="right", padx=(0, 12)
        )

        main = ttk.Panedwindow(self.root, orient="horizontal")
        main.grid(row=2, column=0, sticky="nsew", padx=10, pady=(5, 10))

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=2)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        captures = ttk.LabelFrame(left, text="Snapshot", padding=7)
        captures.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        captures.columnconfigure(0, weight=1)
        captures.rowconfigure(0, weight=1)
        self.capture_list = tk.Listbox(
            captures,
            selectmode=tk.EXTENDED,
            exportselection=False,
            activestyle="none",
            font=self.ui_font,
        )
        capture_scroll = ttk.Scrollbar(
            captures, orient="vertical", command=self.capture_list.yview
        )
        self.capture_list.configure(yscrollcommand=capture_scroll.set)
        self.capture_list.grid(row=0, column=0, sticky="nsew")
        capture_scroll.grid(row=0, column=1, sticky="ns")
        capture_actions = ttk.Frame(captures)
        capture_actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        for column in range(3):
            capture_actions.columnconfigure(column, weight=1)
        ttk.Button(
            capture_actions, text="刷新", command=self.refresh_captures
        ).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(
            capture_actions, text="全选", command=self.select_all
        ).grid(row=0, column=1, sticky="ew", padx=3)
        ttk.Button(
            capture_actions, text="清空", command=self.clear_selection
        ).grid(row=0, column=2, sticky="ew", padx=(3, 0))

        settings = ttk.LabelFrame(left, text="桌面裁剪与生产参数", padding=8)
        settings.grid(row=1, column=0, sticky="ew", padx=(0, 5), pady=(8, 0))
        bounds = (
            ("X min", self.x_min_var),
            ("X max", self.x_max_var),
            ("Y min", self.y_min_var),
            ("Y max", self.y_max_var),
        )
        for index, (label, variable) in enumerate(bounds):
            row, column = divmod(index, 2)
            base = column * 2
            ttk.Label(settings, text=label).grid(
                row=row, column=base, sticky="w", pady=2
            )
            ttk.Entry(settings, textvariable=variable, width=9).grid(
                row=row, column=base + 1, sticky="ew", padx=(4, 8), pady=2
            )
        ttk.Label(settings, text="设备").grid(row=2, column=0, sticky="w", pady=(6, 2))
        ttk.Combobox(
            settings,
            textvariable=self.device_var,
            values=("cuda", "cpu"),
            state="readonly",
            width=8,
        ).grid(row=2, column=1, sticky="w", padx=(4, 8), pady=(6, 2))
        ttk.Checkbutton(
            settings, text="复用预处理", variable=self.reuse_var
        ).grid(row=2, column=2, columnspan=2, sticky="w", pady=(6, 2))
        profile_values = (
            ("深度输入", "开"),
            ("Depth holdout", "0.0"),
            ("手部深度上限", "1.0 m"),
            ("max radius", "2.3 m"),
            ("voxel", "0.01 m"),
            ("融合", "occupancy"),
        )
        for row, (label, value) in enumerate(profile_values, start=3):
            ttk.Label(settings, text=label).grid(row=row, column=0, columnspan=2, sticky="w")
            ttk.Label(settings, text=value).grid(row=row, column=2, columnspan=2, sticky="e")

        stages = ttk.LabelFrame(right, text="阶段", padding=7)
        stages.grid(row=0, column=0, sticky="ew", padx=(5, 0))
        stages.columnconfigure(0, weight=1)
        self.stage_tree = ttk.Treeview(
            stages,
            columns=("status",),
            show="tree headings",
            height=6,
        )
        self.stage_tree.heading("#0", text="步骤")
        self.stage_tree.heading("status", text="状态")
        self.stage_tree.column("#0", width=420, stretch=True)
        self.stage_tree.column("status", width=100, anchor="center", stretch=False)
        self.stage_tree.grid(row=0, column=0, sticky="ew")

        notebook = ttk.Notebook(right)
        notebook.grid(row=1, column=0, sticky="nsew", padx=(5, 0), pady=(8, 0))
        log_tab = ttk.Frame(notebook)
        output_tab = ttk.Frame(notebook)
        notebook.add(log_tab, text="运行日志")
        notebook.add(output_tab, text="最终三版本")
        log_tab.columnconfigure(0, weight=1)
        log_tab.rowconfigure(0, weight=1)
        self.log_text = tk.Text(
            log_tab,
            wrap="none",
            font=self.log_font,
            padx=8,
            pady=7,
            state="disabled",
        )
        log_y = ttk.Scrollbar(log_tab, orient="vertical", command=self.log_text.yview)
        log_x = ttk.Scrollbar(log_tab, orient="horizontal", command=self.log_text.xview)
        self.log_text.configure(yscrollcommand=log_y.set, xscrollcommand=log_x.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_y.grid(row=0, column=1, sticky="ns")
        log_x.grid(row=1, column=0, sticky="ew")

        output_tab.columnconfigure(0, weight=1)
        output_tab.rowconfigure(0, weight=1)
        self.output_tree = ttk.Treeview(
            output_tab,
            columns=("version", "path", "state"),
            show="headings",
        )
        self.output_tree.heading("version", text="版本")
        self.output_tree.heading("path", text="GLB")
        self.output_tree.heading("state", text="状态")
        self.output_tree.column("version", width=180, stretch=False)
        self.output_tree.column("path", width=520, stretch=True)
        self.output_tree.column("state", width=80, anchor="center", stretch=False)
        self.output_tree.grid(row=0, column=0, sticky="nsew")
        output_scroll = ttk.Scrollbar(
            output_tab, orient="vertical", command=self.output_tree.yview
        )
        self.output_tree.configure(yscrollcommand=output_scroll.set)
        output_scroll.grid(row=0, column=1, sticky="ns")
        self.output_tree.bind("<Double-1>", self.open_selected_output)

    def choose_run_root(self) -> None:
        value = filedialog.askdirectory(initialdir=self.run_root_var.get())
        if value:
            self.run_root_var.set(value)
            self.refresh_captures()

    def choose_capture_script(self) -> None:
        value = filedialog.askopenfilename(
            initialdir=str(Path(self.capture_script_var.get()).parent),
            filetypes=(("Python", "*.py"), ("All files", "*")),
        )
        if value:
            self.capture_script_var.set(value)

    def choose_capture_python(self) -> None:
        value = filedialog.askopenfilename(
            initialdir=str(Path(self.capture_python_var.get()).expanduser().parent),
            filetypes=(("Executable", "*"),),
        )
        if value:
            self.capture_python_var.set(value)

    def choose_processing_python(self) -> None:
        value = filedialog.askopenfilename(
            initialdir=str(Path(self.processing_python_var.get()).expanduser().parent),
            filetypes=(("Executable", "*"),),
        )
        if value:
            self.processing_python_var.set(value)

    def choose_g2_root(self) -> None:
        value = filedialog.askdirectory(initialdir=self.g2_root_var.get())
        if value:
            self.g2_root_var.set(value)

    def choose_avoid_root(self) -> None:
        value = filedialog.askdirectory(initialdir=self.avoid_root_var.get())
        if value:
            self.avoid_root_var.set(value)

    def selected_captures(self) -> tuple[str, ...]:
        return tuple(
            self.capture_names[index]
            for index in self.capture_list.curselection()
        )

    def _config(self, captures: tuple[str, ...] = ()) -> G2FullPipelineConfig:
        try:
            bounds = TableBounds(
                float(self.x_min_var.get()),
                float(self.x_max_var.get()),
                float(self.y_min_var.get()),
                float(self.y_max_var.get()),
            )
        except ValueError as exc:
            raise ValueError("桌面边界必须是数字") from exc
        return G2FullPipelineConfig(
            run_root=Path(self.run_root_var.get()).expanduser(),
            captures=captures,
            table_bounds=bounds,
            capture_script=Path(self.capture_script_var.get()).expanduser(),
            g2_root=Path(self.g2_root_var.get()).expanduser(),
            avoid_root=Path(self.avoid_root_var.get()).expanduser(),
            pipeline_root=Path(__file__).resolve().parent,
            capture_python=self.capture_python_var.get().strip(),
            processing_python=self.processing_python_var.get().strip(),
            device=self.device_var.get(),
            reuse_preprocessed=bool(self.reuse_var.get()),
        )

    def refresh_captures(self) -> None:
        selected = set(self.selected_captures()) if self.capture_names else set()
        names = discover_g2_captures(
            Path(self.run_root_var.get()).expanduser() / "in"
        )
        if names == self.capture_names:
            return
        self.capture_names = names
        self.capture_list.delete(0, tk.END)
        for index, name in enumerate(names):
            self.capture_list.insert(tk.END, name)
            if name in selected:
                self.capture_list.selection_set(index)

    def _periodic_refresh(self) -> None:
        try:
            self.refresh_captures()
        finally:
            self.root.after(1500, self._periodic_refresh)

    def select_all(self) -> None:
        self.capture_list.selection_set(0, tk.END)

    def clear_selection(self) -> None:
        self.capture_list.selection_clear(0, tk.END)

    def start_capture(self) -> None:
        try:
            config = self._config()
            prepare_run(config)
            command = build_capture_command(config)
        except (OSError, ValueError) as exc:
            messagebox.showerror("无法启动采集", str(exc))
            return
        self._start_job("capture", config, [command])

    def start_pipeline(self) -> None:
        captures = self.selected_captures()
        try:
            config = self._config(captures)
            config.validate(require_captures=True)
            prepare_run(config)
            commands = build_processing_commands(config)
        except (OSError, ValueError) as exc:
            messagebox.showerror("无法运行 Pipeline", str(exc))
            return
        self._start_job("pipeline", config, commands)

    def show_command_preview(self) -> None:
        captures = self.selected_captures()
        try:
            config = self._config(captures)
            if captures:
                commands = build_processing_commands(config)
            else:
                commands = [build_capture_command(config)]
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self._append_log("\n" + command_preview(commands) + "\n")

    def _start_job(
        self,
        kind: str,
        config: G2FullPipelineConfig,
        commands: list[tuple[str, list[str]]],
    ) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("任务正在运行", "请先停止当前任务")
            return
        self.stop_requested = False
        self.active_job = kind
        self.job_started_at = time.monotonic()
        self._set_running(True)
        self._populate_stages(commands)
        self._append_log(
            f"\n[{time.strftime('%H:%M:%S')}] {kind} started\n"
        )
        self.worker = threading.Thread(
            target=self._run_commands,
            args=(kind, config, commands),
            daemon=True,
        )
        self.worker.start()

    def _run_commands(
        self,
        kind: str,
        config: G2FullPipelineConfig,
        commands: list[tuple[str, list[str]]],
    ) -> None:
        try:
            env = os.environ.copy()
            avoid_pythonpath = str(config.avoid_root)
            old_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                avoid_pythonpath
                if not old_pythonpath
                else avoid_pythonpath + os.pathsep + old_pythonpath
            )
            for stage, command in commands:
                if self.stop_requested:
                    raise InterruptedError("任务已停止")
                self.events.put(("stage_started", stage))
                self.events.put(("log", f"\n[{stage}] {' '.join(command)}\n"))
                self.process = subprocess.Popen(
                    command,
                    cwd=str(WORKSPACE_ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert self.process.stdout is not None
                for line in self.process.stdout:
                    self.events.put(("log", line))
                return_code = self.process.wait()
                self.process = None
                if return_code != 0:
                    if self.stop_requested:
                        raise InterruptedError("任务已停止")
                    raise RuntimeError(f"{stage} exited with status {return_code}")
                self.events.put(("stage_done", stage))
            if kind == "pipeline" and not final_outputs_exist(config):
                raise RuntimeError("Pipeline completed but one or more final GLBs are missing")
            if kind == "pipeline":
                validation = validate_final_outputs(config)
                self.events.put(
                    (
                        "log",
                        "\n[crop_validation] "
                        f"{len(validation['captures'])} snapshot(s), "
                        "three versions each, outside_table_xy_voxels=0\n",
                    )
                )
            self.events.put(("job_done", (kind, config)))
        except Exception as exc:
            self.process = None
            self.events.put(("job_failed", (kind, str(exc))))

    def _populate_stages(self, commands: list[tuple[str, list[str]]]) -> None:
        for item in self.stage_tree.get_children():
            self.stage_tree.delete(item)
        self.stage_items.clear()
        for stage, _ in commands:
            item = self.stage_tree.insert("", "end", text=stage, values=("等待",))
            self.stage_items[stage] = item

    def _poll_events(self) -> None:
        while True:
            try:
                event, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "log":
                self._append_log(str(payload))
            elif event == "stage_started":
                stage = str(payload)
                self.status_var.set(f"运行中: {stage}")
                self.stage_tree.set(self.stage_items[stage], "status", "运行中")
            elif event == "stage_done":
                stage = str(payload)
                self.stage_tree.set(self.stage_items[stage], "status", "完成")
            elif event == "job_done":
                kind, config = payload
                self._set_running(False)
                self.status_var.set("采集窗口已关闭" if kind == "capture" else "全部完成")
                self.refresh_captures()
                if kind == "pipeline":
                    self._show_outputs(config)
            elif event == "job_failed":
                kind, error = payload
                self._set_running(False)
                self.status_var.set("已停止" if self.stop_requested else "失败")
                self._append_log(f"\n[{kind}] {error}\n")
                if not self.stop_requested:
                    messagebox.showerror("任务失败", error)
        self.root.after(100, self._poll_events)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _show_outputs(self, config: G2FullPipelineConfig) -> None:
        for item in self.output_tree.get_children():
            self.output_tree.delete(item)
        labels = {
            "depth_only": "Only 深度",
            "fused": "Fuse",
            "mapanything_only": "Only MapAnything",
        }
        validation_path = config.run_root / "three_version_validation.json"
        validation = {}
        if validation_path.is_file():
            try:
                validation = json.loads(
                    validation_path.read_text(encoding="utf-8")
                ).get("captures", {})
            except (OSError, json.JSONDecodeError):
                validation = {}
        for capture in config.captures:
            for key, path in config.final_outputs(capture).items():
                report = validation.get(capture, {}).get(key, {})
                validated = report.get("outside_table_xy_voxels") == 0
                self.output_tree.insert(
                    "",
                    "end",
                    values=(
                        f"{capture} / {labels[key]}",
                        str(path),
                        "已裁剪" if path.is_file() and validated else "未验证",
                    ),
                )

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        self.capture_button.configure(state=state)
        self.run_button.configure(state=state)
        self.stop_button.configure(state="normal" if running else "disabled")
        if not running:
            self.active_job = ""

    def _update_elapsed(self) -> None:
        if self.job_started_at is not None and self.active_job:
            elapsed = max(0, int(time.monotonic() - self.job_started_at))
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.elapsed_var.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        self.root.after(500, self._update_elapsed)

    def stop_job(self) -> None:
        self.stop_requested = True
        self.status_var.set("正在停止")
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()

    def open_versions_root(self) -> None:
        path = Path(self.run_root_var.get()).expanduser() / "versions"
        path.mkdir(parents=True, exist_ok=True)
        self._open_path(path)

    def open_selected_output(self, event=None) -> None:
        del event
        selected = self.output_tree.selection()
        if not selected:
            return
        values = self.output_tree.item(selected[0], "values")
        if len(values) >= 2:
            path = Path(str(values[1]))
            self._open_path(path if path.exists() else path.parent)

    def _open_path(self, path: Path) -> None:
        try:
            subprocess.Popen(
                ["xdg-open", str(path.expanduser().resolve())],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            messagebox.showerror("无法打开", str(exc))

    def on_close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.stop_requested = True
            self.process.terminate()
        self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Build and update the GUI once, then exit without showing it",
    )
    args = parser.parse_args()
    root = tk.Tk()
    app = G2FullPipelineGui(root, run_root=args.run_root)
    if args.smoke_test:
        root.withdraw()
        root.update_idletasks()
        root.update()
        app.on_close()
        return 0
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
