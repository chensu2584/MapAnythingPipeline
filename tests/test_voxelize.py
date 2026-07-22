import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

import voxelize
from capture_contract import VIEW_NAMES
from voxelize import voxel_scene_with_grippers


class VoxelGripperMarkerTests(unittest.TestCase):
    def test_gripper_markers_share_voxel_glb_flip(self):
        poses = {
            "poses": {
                "left": {"pose_matrix": np.eye(4).tolist()},
                "right": {"pose_matrix": np.eye(4).tolist()},
            }
        }
        poses["poses"]["left"]["pose_matrix"][0][3] = 0.9
        poses["poses"]["left"]["pose_matrix"][1][3] = 0.3
        poses["poses"]["left"]["pose_matrix"][2][3] = 0.7
        poses["poses"]["right"]["pose_matrix"][0][3] = 0.8
        poses["poses"]["right"]["pose_matrix"][1][3] = -0.2
        poses["poses"]["right"]["pose_matrix"][2][3] = 0.6

        scene = voxel_scene_with_grippers(trimesh.creation.box(), poses)

        self.assertIn("gripper_left_center", scene.geometry)
        self.assertIn("gripper_right_center", scene.geometry)
        np.testing.assert_allclose(
            scene.geometry["gripper_left_center"].centroid,
            [0.9, -0.3, -0.7],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            scene.geometry["gripper_right_center"].centroid,
            [0.8, 0.2, -0.6],
            atol=1e-12,
        )



class VoxelSourceViewTests(unittest.TestCase):
    """Cover the pre-filter path, where per-point arrays must stay in step."""

    @staticmethod
    def make_capture(root, regions):
        capture = root / "cap"
        capture.mkdir(parents=True)
        rng = np.random.default_rng(0)
        payload = {}
        for name, (low, high) in regions.items():
            points = np.stack(
                [
                    rng.uniform(low, high, 400),
                    rng.uniform(0.0, 0.2, 400),
                    rng.uniform(0.0, 0.2, 400),
                ],
                axis=1,
            ).astype(np.float32)
            payload[f"{name}_pts3d"] = points.reshape(20, 20, 3)
            payload[f"{name}_mask"] = np.ones((20, 20), bool)
            payload[f"{name}_img"] = np.full((20, 20, 3), 128, np.uint8)
            payload[f"{name}_conf"] = np.ones((20, 20), np.float32)
        np.savez(capture / "views.npz", **payload)
        return capture

    def test_filtered_capture_keeps_per_point_arrays_aligned(self):
        """Regression: view_bits was not filtered with pts, so a --max_radius
        run died inside the aggregation with an opaque broadcasting error."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_capture(
                root,
                {"head": (0.0, 0.5), "hand_left": (0.3, 0.8), "hand_right": (2.0, 4.0)},
            )
            stats = voxelize.process_capture(
                "cap", voxel_size=0.1, max_radius=1.5, output_root=str(root)
            )
            self.assertIsNotNone(stats)
            with np.load(root / "cap" / "voxels.npz") as data:
                self.assertEqual(len(data["source_views"]), len(data["indices"]))
                # hand_right lay entirely outside the radius.
                self.assertFalse((data["source_views"] & 0b100).any())
                self.assertTrue((data["source_views"] & 0b001).any())

    def test_mismatched_per_point_array_is_refused_clearly(self):
        with self.assertRaisesRegex(ValueError, "must be filtered together"):
            voxelize.voxelize_points(
                np.zeros((10, 3), np.float32),
                np.zeros((10, 3), np.float32),
                None,
                0.1,
                np.zeros(3),
                (2, 2, 2),
                np.zeros(7, np.uint8),
            )

    def test_frame_is_declared_only_when_known(self):
        """An "unknown" placeholder would stop consumers falling back to the
        pose document that does know the frame."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = self.make_capture(root, {n: (0.0, 0.5) for n in VIEW_NAMES})
            voxelize.process_capture("cap", voxel_size=0.1, output_root=str(root))
            with np.load(capture / "voxels.npz") as data:
                self.assertNotIn("world_frame", data.files)

            (capture / "camera_poses_used_for_export.json").write_text(
                '{"world_frame": "base_link"}', encoding="utf-8"
            )
            voxelize.process_capture("cap", voxel_size=0.1, output_root=str(root))
            with np.load(capture / "voxels.npz") as data:
                self.assertEqual(str(data["world_frame"]), "base_link")
                self.assertEqual(str(data["translation_unit"]), "meter")


if __name__ == "__main__":
    unittest.main()
