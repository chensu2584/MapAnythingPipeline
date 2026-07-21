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


if __name__ == "__main__":
    unittest.main()
