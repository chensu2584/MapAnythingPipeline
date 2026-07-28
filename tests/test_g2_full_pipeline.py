"""Tests for the G2 capture-to-three-version command orchestration."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from g2_full_pipeline import (
    CAPTURE_REFERENCE_TOKENS,
    G2FullPipelineConfig,
    TableBounds,
    build_capture_command,
    build_processing_commands,
    discover_g2_captures,
    final_outputs_exist,
    prepare_run,
    validate_final_outputs,
    validate_capture_reference,
)


def fixture_layout(tmp_path: Path) -> G2FullPipelineConfig:
    pipeline = tmp_path / "MapAnythingPipeline"
    avoid = tmp_path / "Avoid"
    g2 = tmp_path / "G2"
    sensor = g2 / "G2_parameters" / "sensor"
    urdf = (
        g2
        / "G2_parameters"
        / "G2_t2_crs_omnipicker"
        / "urdf"
        / "G2_t2_crs_omnipicker.urdf"
    )
    capture_script = tmp_path / "g2_four_camera_extrinsic_capture.py"
    pipeline.mkdir()
    (avoid / "scripts").mkdir(parents=True)
    sensor.mkdir(parents=True)
    urdf.parent.mkdir(parents=True)
    urdf.write_text("<robot name='g2'/>", encoding="utf-8")
    capture_script.write_text(
        "\n".join(CAPTURE_REFERENCE_TOKENS),
        encoding="utf-8",
    )
    run_root = tmp_path / "run"
    snapshot = run_root / "in" / "snapshot_0001"
    snapshot.mkdir(parents=True)
    for filename in (
        "camera_extrinsics.json",
        "head_rgb.png",
        "hand_left_rgb.png",
        "hand_right_rgb.png",
        "head_depth_raw16.png",
    ):
        (snapshot / filename).write_text("{}", encoding="utf-8")
    return G2FullPipelineConfig(
        run_root=run_root,
        captures=("snapshot_0001",),
        capture_script=capture_script,
        g2_root=g2,
        avoid_root=avoid,
        pipeline_root=pipeline,
    )


def command_for(commands, stage):
    return next(command for name, command in commands if name == stage)


class G2FullPipelineTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temporary = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_capture_command_uses_the_canonical_capture_contract(self):
        config = fixture_layout(self.tmp_path)
        stage, command = build_capture_command(config, python_executable="/python")
        self.assertEqual(stage, "four_camera_capture")
        self.assertEqual(
            command[:3], ["/python", "-u", str(config.capture_script.resolve())]
        )
        self.assertEqual(
            command[command.index("--save-dir") + 1],
            str(config.input_root.resolve()),
        )
        self.assertEqual(
            command[command.index("--sensor-dir") + 1],
            str(config.sensor_dir.resolve()),
        )
        self.assertEqual(
            command[command.index("--urdf") + 1], str(config.urdf.resolve())
        )
        self.assertEqual(command[command.index("--pose-source") + 1], "fk")
        self.assertNotIn("--oneshot", command)

    def test_capture_and_processing_python_can_be_selected_independently(self):
        config = fixture_layout(self.tmp_path)
        config = G2FullPipelineConfig(
            **{
                **config.__dict__,
                "capture_python": "/robot/python",
                "processing_python": "/map/python",
            }
        )
        _, capture = build_capture_command(config)
        processing = build_processing_commands(config)
        self.assertEqual(capture[0], "/robot/python")
        self.assertTrue(all(command[0] == "/map/python" for _, command in processing))

    def test_processing_commands_use_best_production_inference_profile(self):
        config = fixture_layout(self.tmp_path)
        commands = build_processing_commands(config, python_executable="/python")
        self.assertEqual(
            [name for name, _ in commands],
            [
                "mapanything_undistort",
                "mapanything_run_inference",
                "mapanything_filter_export",
                "mapanything_voxelize",
                "depth_snapshot_0001",
                "export_three_cropped_versions",
            ],
        )
        inference = command_for(commands, "mapanything_run_inference")
        self.assertIn("--depth-input", inference)
        self.assertNotIn("--depth-holdout", inference)
        self.assertNotIn("--ignore-poses", inference)
        self.assertNotIn("--roll-normalize", inference)
        self.assertNotIn("--self-mask-input", inference)
        self.assertNotIn("--fast-inference", inference)
        self.assertEqual(inference[inference.index("--max_radius") + 1], "2.3")
        self.assertEqual(inference.count("--view-max-depth"), 2)
        self.assertIn("hand_left=1.0", inference)
        self.assertIn("hand_right=1.0", inference)

        undistort = command_for(commands, "mapanything_undistort")
        self.assertNotIn("--reuse-existing", undistort)

    def test_final_export_applies_one_measured_table_crop_to_all_versions(self):
        config = fixture_layout(self.tmp_path)
        commands = build_processing_commands(config, python_executable="/python")
        fusion = command_for(commands, "export_three_cropped_versions")
        table_index = fusion.index("--table-xy-bounds")
        self.assertEqual(
            fusion[table_index + 1 : table_index + 5],
            ["0.239", "1.019", "-0.694", "0.706"],
        )
        self.assertNotIn("--no-workspace-crop", fusion)
        self.assertIn("--no-gripper-shell", fusion)
        self.assertEqual(fusion[fusion.index("--tint-strength") + 1], "0.0")
        self.assertEqual(fusion[fusion.index("--method") + 1], "occupancy")

        outputs = config.final_outputs("snapshot_0001")
        self.assertEqual(outputs["depth_only"].name, "depth_only_voxels.glb")
        self.assertEqual(outputs["fused"].name, "fused_voxels.glb")
        self.assertEqual(
            outputs["mapanything_only"].name,
            "mapanything_only_voxels.glb",
        )
        self.assertEqual(len({path.parent for path in outputs.values()}), 1)

    def test_prepare_run_records_reproducible_profile_and_outputs(self):
        config = fixture_layout(self.tmp_path)
        manifest_path = prepare_run(config)
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(document["inference"]["feed_metric_depth"])
        self.assertEqual(document["inference"]["depth_holdout"], 0.0)
        self.assertEqual(
            document["postprocess"]["table_xy_bounds_base_link_m"],
            [0.239, 1.019, -0.694, 0.706],
        )
        self.assertEqual(document["postprocess"]["viewer_tint_strength"], 0.0)
        self.assertFalse(
            document["postprocess"]["show_gripper_removal_shell"]
        )
        self.assertEqual(
            set(document["final_outputs"]["snapshot_0001"]),
            {"depth_only", "fused", "mapanything_only"},
        )
        self.assertEqual(
            set(document["final_outputs"]["snapshot_0001"]["fused"]),
            {"npz", "glb"},
        )

    def test_discovery_only_accepts_g2_snapshot_contract(self):
        root = self.tmp_path / "in"
        (root / "snapshot_good").mkdir(parents=True)
        for filename in (
            "camera_extrinsics.json",
            "head_rgb.png",
            "hand_left_rgb.png",
            "hand_right_rgb.png",
            "head_depth_raw16.png",
        ):
            (root / "snapshot_good" / filename).write_text("{}", encoding="utf-8")
        (root / "snapshot_incomplete").mkdir()
        self.assertEqual(discover_g2_captures(root), ["snapshot_good"])

    def test_capture_reference_validation_fails_closed(self):
        path = self.tmp_path / "capture.py"
        path.write_text("print('not the G2 capture contract')", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing tokens"):
            validate_capture_reference(path)

    def test_table_bounds_reject_inverted_rectangle(self):
        with self.assertRaisesRegex(ValueError, "X_MIN"):
            TableBounds(1.0, 0.0, -0.5, 0.5).validate()

    def test_final_validation_rejects_any_voxel_outside_table_crop(self):
        import numpy as np

        config = fixture_layout(self.tmp_path)
        output = config.versions_root / "snapshot_0001"
        output.mkdir(parents=True)
        bounds = config.table_bounds.as_tuple()
        for key, path in config.final_voxel_outputs("snapshot_0001").items():
            point = np.array([0.5, 0.0, 0.7])
            if key == "mapanything_only":
                point[0] = bounds[1] + 0.05
            voxel_size = 0.01
            origin = np.zeros(3)
            index = np.rint(point / voxel_size - 0.5).astype(np.int32)
            np.savez_compressed(
                path,
                indices=index[None],
                origin=origin,
                voxel_size=np.float32(voxel_size),
                world_frame=np.asarray("base_link"),
            )
        for path in config.final_outputs("snapshot_0001").values():
            path.write_bytes(b"glTF")
        (output / "fusion_report.json").write_text(
            json.dumps(
                {
                    "cleanup": {
                        "depth": {
                            "parameters": {
                                "table_xy_bounds": list(bounds),
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        self.assertTrue(final_outputs_exist(config))
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_final_outputs(config)


if __name__ == "__main__":
    unittest.main()
