import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "pipeline_gui.py"
SPEC = importlib.util.spec_from_file_location("pipeline_gui_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PipelineGuiCommandTest(unittest.TestCase):
    def config(self, root, use_poses=True, pose_export_mode=None):
        return MODULE.PipelineConfig(
            data_root=root / "data",
            output_root=root / "out",
            captures=("capture_a", "capture_b"),
            stages=MODULE.STAGES,
            use_metric_poses=use_poses,
            pose_export_mode=(
                pose_export_mode or MODULE.DEFAULT_POSE_EXPORT_MODE
            ),
            max_radius=2.3,
            voxel_size=0.02,
            device="cuda:0",
        )

    def test_metric_mode_passes_roots_captures_and_radius(self):
        with tempfile.TemporaryDirectory() as tmp:
            commands = dict(
                MODULE.build_pipeline_commands(
                    self.config(Path(tmp)),
                    python_executable="python-test",
                    script_dir=Path("/pipeline"),
                )
            )
        self.assertEqual(list(commands), list(MODULE.STAGES))
        self.assertIn("--data-root", commands["undistort"])
        self.assertIn("--reuse-existing", commands["undistort"])
        self.assertIn("--input-root", commands["run_inference"])
        self.assertNotIn("--fast-inference", commands["run_inference"])
        self.assertIn("capture_a", commands["voxelize"])
        self.assertIn("2.3", commands["filter_export"])
        self.assertIn("--show_cameras", commands["filter_export"])
        self.assertIn("--show_grippers", commands["filter_export"])
        self.assertIn("--show_grippers", commands["voxelize"])
        self.assertIn("--color_by_view", commands["filter_export"])
        self.assertNotIn("--per_camera_k_ab", commands["filter_export"])
        self.assertNotIn("--ignore-poses", commands["undistort"])
        self.assertNotIn("--ignore-poses", commands["run_inference"])
        self.assertIn("--pose-export-mode", commands["run_inference"])
        mode_index = commands["run_inference"].index("--pose-export-mode")
        self.assertEqual(
            commands["run_inference"][mode_index + 1],
            MODULE.MODEL_RELATIVE_HEAD_ANCHORED_BASELINE_SCALED,
        )

    def test_unscaled_model_relative_mode_remains_selectable(self):
        with tempfile.TemporaryDirectory() as tmp:
            commands = dict(
                MODULE.build_pipeline_commands(
                    self.config(
                        Path(tmp),
                        pose_export_mode=MODULE.MODEL_RELATIVE_HEAD_ANCHORED,
                    ),
                    python_executable="python-test",
                    script_dir=Path("/pipeline"),
                )
            )
        mode_index = commands["run_inference"].index("--pose-export-mode")
        self.assertEqual(
            commands["run_inference"][mode_index + 1],
            MODULE.MODEL_RELATIVE_HEAD_ANCHORED,
        )

    def test_legacy_calibrated_hybrid_is_still_explicitly_selectable(self):
        with tempfile.TemporaryDirectory() as tmp:
            commands = dict(
                MODULE.build_pipeline_commands(
                    self.config(
                        Path(tmp), pose_export_mode=MODULE.CALIBRATED_INPUT
                    ),
                    python_executable="python-test",
                    script_dir=Path("/pipeline"),
                )
            )
        mode_index = commands["run_inference"].index("--pose-export-mode")
        self.assertEqual(
            commands["run_inference"][mode_index + 1], MODULE.CALIBRATED_INPUT
        )

    def test_rgb_only_mode_explicitly_ignores_pose_in_both_input_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            commands = dict(
                MODULE.build_pipeline_commands(
                    self.config(Path(tmp), use_poses=False),
                    python_executable="python-test",
                    script_dir=Path("/pipeline"),
                )
            )
        self.assertIn("--ignore-poses", commands["undistort"])
        self.assertIn("--ignore-poses", commands["run_inference"])
        self.assertNotIn("--pose-export-mode", commands["run_inference"])
        self.assertNotIn("--max_radius", commands["run_inference"])
        self.assertIn("--max_radius", commands["filter_export"])

    def test_scene_markers_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(Path(tmp))
            config = MODULE.dataclasses.replace(config, show_scene_markers=False)
            commands = dict(
                MODULE.build_pipeline_commands(
                    config,
                    python_executable="python-test",
                    script_dir=Path("/pipeline"),
                )
            )
        self.assertNotIn("--show_cameras", commands["filter_export"])

    def test_gripper_markers_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(Path(tmp))
            config = MODULE.dataclasses.replace(config, show_gripper_markers=False)
            commands = dict(
                MODULE.build_pipeline_commands(
                    config,
                    python_executable="python-test",
                    script_dir=Path("/pipeline"),
                )
            )
        self.assertNotIn("--show_grippers", commands["filter_export"])
        self.assertNotIn("--show_grippers", commands["voxelize"])

    def test_view_colored_glb_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(Path(tmp))
            config = MODULE.dataclasses.replace(
                config, export_view_colored_glb=False
            )
            commands = dict(
                MODULE.build_pipeline_commands(
                    config,
                    python_executable="python-test",
                    script_dir=Path("/pipeline"),
                )
            )
        self.assertNotIn("--color_by_view", commands["filter_export"])

    def test_per_camera_k_ab_glbs_can_be_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(Path(tmp))
            config = MODULE.dataclasses.replace(
                config, export_per_camera_k_ab_glb=True
            )
            commands = dict(
                MODULE.build_pipeline_commands(
                    config,
                    python_executable="python-test",
                    script_dir=Path("/pipeline"),
                )
            )
        self.assertIn("--per_camera_k_ab", commands["filter_export"])

    def test_fast_inference_and_forced_preprocess_can_be_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = MODULE.dataclasses.replace(
                self.config(Path(tmp)),
                fast_inference=True,
                reuse_preprocessed=False,
            )
            commands = dict(
                MODULE.build_pipeline_commands(
                    config,
                    python_executable="python-test",
                    script_dir=Path("/pipeline"),
                )
            )
        self.assertIn("--fast-inference", commands["run_inference"])
        self.assertNotIn("--reuse-existing", commands["undistort"])

    def test_duration_format_is_stable(self):
        self.assertEqual(MODULE.format_duration(0.0), "00:00:00")
        self.assertEqual(MODULE.format_duration(65.0), "00:01:05")
        self.assertEqual(MODULE.format_duration(3661.0), "01:01:01")



class PipelineGuiG2Test(unittest.TestCase):
    def make_roots(self, tmp):
        g1 = Path(tmp) / "g1"
        (g1 / "cap").mkdir(parents=True)
        for name in MODULE.RAW_REQUIRED_FILES:
            (g1 / "cap" / name).write_text("x", encoding="utf-8")
        g2 = Path(tmp) / "g2"
        (g2 / "snapshot_0001").mkdir(parents=True)
        (g2 / "snapshot_0001" / MODULE.G2_REQUIRED_FILE).write_text("{}", encoding="utf-8")
        return g1, g2

    def test_discovery_separates_the_two_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            g1, g2 = self.make_roots(tmp)
            self.assertEqual(MODULE.discover_captures(g1), ["cap"])
            self.assertEqual(MODULE.discover_captures(g2), ["snapshot_0001"])
            self.assertEqual(MODULE.discover_captures(g2, "g1"), [])
            self.assertEqual(MODULE.discover_captures(g1, "g2"), [])
            self.assertEqual(MODULE.detect_root_layout(g1), "g1")
            self.assertEqual(MODULE.detect_root_layout(g2), "g2")

    def base_config(self, tmp, **overrides):
        options = dict(
            data_root=Path(tmp),
            output_root=Path(tmp) / "out",
            captures=("snapshot_0001",),
            stages=MODULE.STAGES,
            robot="g2",
        )
        options.update(overrides)
        return MODULE.PipelineConfig(**options)

    def test_g2_passes_robot_and_never_asks_for_gripper_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            commands = dict(
                MODULE.build_pipeline_commands(
                    self.base_config(tmp, show_gripper_markers=True),
                    python_executable="py",
                    script_dir=Path("/pipeline"),
                )
            )
            self.assertIn("--robot", commands["undistort"])
            self.assertIn("g2", commands["undistort"])
            # The gripper overlay is G1-only and would abort Steps C/D on G2.
            self.assertNotIn("--show_grippers", commands["filter_export"])
            self.assertNotIn("--show_grippers", commands["voxelize"])

    def test_g1_still_gets_gripper_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            commands = dict(
                MODULE.build_pipeline_commands(
                    self.base_config(tmp, robot="g1", show_gripper_markers=True),
                    python_executable="py",
                    script_dir=Path("/pipeline"),
                )
            )
            self.assertIn("--show_grippers", commands["filter_export"])
            # The resolved robot is always stated explicitly rather than left to
            # undistort.py's layout detection.
            self.assertIn("g1", commands["undistort"])

    def test_depth_options_reach_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            commands = dict(
                MODULE.build_pipeline_commands(
                    self.base_config(tmp, depth_input=True, depth_holdout=0.3),
                    python_executable="py",
                    script_dir=Path("/pipeline"),
                )
            )
            command = commands["run_inference"]
            self.assertIn("--depth-input", command)
            self.assertIn("--depth-holdout", command)
            self.assertIn("0.3", command)

    def test_unresolved_robot_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "resolved"):
                MODULE.build_pipeline_commands(self.base_config(tmp, robot="auto"))

    def test_optional_stages_are_not_part_of_the_core_pipeline(self):
        """Selecting the whole core pipeline must not demand a URDF."""
        self.assertNotIn("self_mask", MODULE.STAGES)
        self.assertNotIn("diagnose", MODULE.STAGES)
        with tempfile.TemporaryDirectory() as tmp:
            commands = dict(
                MODULE.build_pipeline_commands(
                    self.base_config(tmp, stages=MODULE.STAGES),
                    python_executable="py",
                    script_dir=Path("/pipeline"),
                )
            )
            self.assertEqual(list(commands), list(MODULE.STAGES))

    def test_self_mask_runs_between_undistort_and_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            order = [
                name
                for name, _ in MODULE.build_pipeline_commands(
                    self.base_config(
                        tmp, stages=MODULE.ALL_STAGES, urdf="/data/robot.urdf",
                        self_mask_input=True,
                    ),
                    python_executable="py",
                    script_dir=Path("/pipeline"),
                )
            ]
            self.assertLess(order.index("undistort"), order.index("self_mask"))
            self.assertLess(order.index("self_mask"), order.index("run_inference"))
            self.assertEqual(order[-1], "diagnose")

    def test_self_mask_stage_without_urdf_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "URDF"):
                MODULE.build_pipeline_commands(
                    self.base_config(tmp, stages=("self_mask",), urdf="")
                )

    def test_hiding_the_robot_reaches_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            commands = dict(
                MODULE.build_pipeline_commands(
                    self.base_config(tmp, self_mask_input=True),
                    python_executable="py",
                    script_dir=Path("/pipeline"),
                )
            )
            self.assertIn("--self-mask-input", commands["run_inference"])

    def test_invalid_holdout_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "holdout"):
                MODULE.build_pipeline_commands(self.base_config(tmp, depth_holdout=1.5))


if __name__ == "__main__":
    unittest.main()
