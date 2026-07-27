import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scene_simplify import (
    axis_aligned_boxes,
    find_support_surface,
    fit_support_box,
    primitives_to_glb,
    read_scene_markers,
)


class SupportSurfaceTests(unittest.TestCase):
    def setUp(self):
        xy = np.stack(
            np.meshgrid(
                np.arange(0.0, 0.61, 0.01),
                np.arange(-0.4, 0.41, 0.01),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 2)
        table = np.column_stack([xy, np.full(len(xy), 0.60)])

        object_xy = np.stack(
            np.meshgrid(
                np.arange(0.20, 0.31, 0.01),
                np.arange(0.10, 0.21, 0.01),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 2)
        objects = np.concatenate(
            [
                np.column_stack([object_xy, np.full(len(object_xy), z)])
                for z in (0.62, 0.64, 0.66, 0.68)
            ]
        )
        self.points = np.concatenate([table, objects])
        self.colors = np.full((len(self.points), 3), 180, np.uint8)

    def test_objects_do_not_pull_support_top_upward(self):
        top, mode, peak = find_support_surface(
            self.points, z_band=(0.5, 0.75), bin_m=0.02
        )

        self.assertGreater(peak, 4000)
        self.assertLessEqual(top, 0.62)
        self.assertLess(mode, top)

    def test_largest_support_component_becomes_table_box(self):
        top, mode, _ = find_support_surface(
            self.points, z_band=(0.5, 0.75), bin_m=0.02
        )
        box, count = fit_support_box(
            self.points,
            self.colors,
            mode,
            top,
            eps=0.03,
            voxel_size=0.01,
            thickness_m=0.06,
        )

        self.assertIsNotNone(box)
        self.assertEqual(box["role"], "support")
        self.assertEqual(box["label"], "table")
        self.assertGreater(count, 4000)
        self.assertAlmostEqual(box["size_m"][2], 0.06)


class ObjectBoxTests(unittest.TestCase):
    def test_xy_cluster_keeps_full_body_down_to_table(self):
        xy = np.stack(
            np.meshgrid(
                np.arange(0.20, 0.31, 0.01),
                np.arange(0.10, 0.21, 0.01),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 2)
        points = np.concatenate(
            [
                np.column_stack([xy, np.full(len(xy), z)])
                for z in (0.62, 0.64, 0.68)
            ]
        )
        colors = np.tile([40, 60, 190], (len(points), 1)).astype(np.uint8)

        boxes, labels = axis_aligned_boxes(
            points,
            colors,
            eps=0.03,
            min_samples=2,
            min_cluster=12,
            support_top_z=0.61,
            min_height_m=0.02,
            voxel_size=0.01,
        )

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["role"], "object")
        self.assertEqual(boxes[0]["primitive"], "box")
        self.assertAlmostEqual(boxes[0]["min_m"][2], 0.61)
        self.assertGreater(boxes[0]["size_m"][2], 0.07)
        self.assertTrue((labels >= 0).all())

    def test_circular_footprint_becomes_cylinder(self):
        angles = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
        ring = np.column_stack([0.3 + 0.05 * np.cos(angles), 0.2 + 0.05 * np.sin(angles)])
        points = np.concatenate(
            [
                np.column_stack([ring, np.full(len(ring), z)])
                for z in (0.64, 0.68, 0.72)
            ]
        )
        colors = np.tile([220, 180, 40], (len(points), 1)).astype(np.uint8)

        shapes, _ = axis_aligned_boxes(
            points,
            colors,
            eps=0.03,
            min_samples=2,
            min_cluster=24,
            support_top_z=0.61,
            min_height_m=0.03,
            voxel_size=0.01,
        )

        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0]["primitive"], "cylinder")
        self.assertGreater(shapes[0]["radius_m"], 0.05)
        self.assertAlmostEqual(shapes[0]["min_m"][2], 0.61)

    def test_thin_high_aspect_fragment_is_suppressed(self):
        points = np.column_stack(
            [
                np.linspace(0.2, 0.5, 40),
                np.full(40, 0.1),
                np.full(40, 0.68),
            ]
        )
        colors = np.full((len(points), 3), 150, np.uint8)

        shapes, _ = axis_aligned_boxes(
            points,
            colors,
            eps=0.03,
            min_samples=2,
            min_cluster=24,
            support_top_z=0.61,
            min_height_m=0.03,
            voxel_size=0.01,
        )

        self.assertEqual(shapes, [])


class SceneMarkerTests(unittest.TestCase):
    def test_camera_gripper_head_and_origin_markers_are_rendered(self):
        def pose(x, y, z):
            value = np.eye(4)
            value[:3, 3] = [x, y, z]
            return {"matrix": value.tolist()}

        payload = {
            "extrinsics": {
                "head_rgb": pose(0.4, 0.0, 1.4),
                "hand_left_rgb": pose(0.8, 0.3, 1.1),
                "hand_right_rgb": pose(0.8, -0.3, 1.1),
            },
            "fk_base_T_link": {
                "arm_l_end_link": pose(0.75, 0.37, 1.0),
                "arm_r_end_link": pose(0.75, -0.37, 1.0),
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera_extrinsics.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            markers = read_scene_markers(str(path))

        self.assertEqual(
            {marker["id"] for marker in markers},
            {
                "origin",
                "head_camera",
                "left_camera",
                "right_camera",
                "left_gripper",
                "right_gripper",
            },
        )
        scene = primitives_to_glb([], markers=markers)
        self.assertIn("marker_origin_axes", scene.geometry)
        self.assertIn("marker_head_camera_center", scene.geometry)
        self.assertIn("marker_left_camera_to_gripper", scene.geometry)
        self.assertIn("marker_right_camera_to_gripper", scene.geometry)
        np.testing.assert_allclose(
            scene.geometry["marker_left_camera_center"].centroid,
            [0.8, -0.3, -1.1],
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
