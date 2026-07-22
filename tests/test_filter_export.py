import unittest

import numpy as np

from filter_export import (
    DEFAULT_FRUSTUM_DEPTH_M,
    camera_center_mesh,
    camera_frustum_mesh,
    gripper_center_mesh,
    gripper_frame_mesh,
    select_per_camera_k,
    validate_unprojection_consistency,
    view_debug_images,
    world_origin_frame_mesh,
)


class UnprojectionConsistencyTests(unittest.TestCase):
    def test_allows_small_range_scaled_gpu_replay_error(self):
        stored = np.array([[[0.0, 0.0, 20.0]]], dtype=np.float32)
        computed = stored.copy()
        computed[0, 0, 0] += 0.011
        stats = validate_unprojection_consistency(
            computed, stored, np.array([[True]])
        )
        self.assertLess(stats["max_tolerance_ratio"], 1.0)

    def test_rejects_centimetre_pose_error_at_working_range(self):
        stored = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
        computed = stored.copy()
        computed[0, 0, 0] += 0.01
        with self.assertRaisesRegex(AssertionError, "inconsistent"):
            validate_unprojection_consistency(
                computed, stored, np.array([[True]])
            )

    def test_allows_the_bf16_replay_error_seen_on_real_g2_data(self):
        """Regression: a real G2 capture failed here by 4.6 percent.

        The point sat 12.61 m from the origin with an 8.69 mm L-infinity error,
        a relative error of 6.9e-4.  That is comfortably inside bfloat16's own
        3.9e-3 relative precision, so rejecting it was a tolerance bug rather
        than a geometry error.
        """
        stored = np.array([[[0.0, 0.0, 12.6145515]]], dtype=np.float32)
        computed = stored.copy()
        computed[0, 0, 0] += 0.00869369507
        stats = validate_unprojection_consistency(
            computed, stored, np.array([[True]])
        )
        self.assertLess(stats["max_tolerance_ratio"], 1.0)

    def test_still_rejects_a_wrong_frame_convention(self):
        """A real convention error is wrong by a fraction of the coordinates."""
        stored = np.array([[[0.3, 0.4, 12.0]]], dtype=np.float32)
        computed = stored.copy()
        computed[0, 0, 1] *= -1.0  # e.g. RDF read as RUB
        with self.assertRaisesRegex(AssertionError, "inconsistent"):
            validate_unprojection_consistency(
                computed, stored, np.array([[True]])
            )

    def test_empty_mask_is_well_defined(self):
        points = np.zeros((1, 1, 3), dtype=np.float32)
        stats = validate_unprojection_consistency(
            points, points, np.array([[False]])
        )
        self.assertEqual(stats["checked_points"], 0)
        self.assertEqual(stats["max_abs_m"], 0.0)


class SceneMarkerTests(unittest.TestCase):
    def test_per_camera_k_ab_policy(self):
        predicted = {
            name: np.array(
                [[100.0 + i, 0.0, 10.0 + i], [0.0, 200.0 + i, 20.0 + i], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            for i, name in enumerate(("head", "hand_left", "hand_right"))
        }
        calibrated = {
            name: np.array(
                [[300.0 + i, 0.0, 30.0 + i], [0.0, 400.0 + i, 40.0 + i], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            for i, name in enumerate(("head", "hand_left", "hand_right"))
        }

        selected = select_per_camera_k(predicted, calibrated)

        np.testing.assert_array_equal(selected["head"], calibrated["head"])
        np.testing.assert_array_equal(
            selected["hand_left"], predicted["hand_left"]
        )
        self.assertEqual(
            selected["hand_right"][0, 0], predicted["hand_right"][0, 0]
        )
        self.assertEqual(
            selected["hand_right"][1, 1], predicted["hand_right"][1, 1]
        )
        self.assertEqual(
            selected["hand_right"][0, 2], calibrated["hand_right"][0, 2]
        )
        self.assertEqual(
            selected["hand_right"][1, 2], calibrated["hand_right"][1, 2]
        )

    def test_view_debug_colors_are_head_red_left_green_right_blue(self):
        images = np.zeros((3, 2, 3, 3), dtype=np.float32)
        colored = view_debug_images(images)
        expected = np.array(
            [[230, 40, 40], [40, 200, 40], [40, 90, 230]],
            dtype=np.float32,
        ) / 255.0
        for view_idx in range(3):
            expected_image = np.broadcast_to(
                expected[view_idx], colored[view_idx].shape
            )
            np.testing.assert_allclose(colored[view_idx], expected_image)

    def test_camera_markers_use_exact_camera_center_and_small_default(self):
        K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
        pose = np.eye(4)
        pose[:3, 3] = [1.0, 2.0, 3.0]
        color = [230, 40, 40, 255]

        frustum = camera_frustum_mesh(K, pose, (80, 100), color)
        center = camera_center_mesh(pose, color)

        np.testing.assert_allclose(frustum.vertices[0], pose[:3, 3])
        np.testing.assert_allclose(center.centroid, pose[:3, 3], atol=1e-12)
        self.assertEqual(DEFAULT_FRUSTUM_DEPTH_M, 0.06)
        self.assertLess(np.linalg.norm(frustum.vertices - pose[:3, 3], axis=1).max(), 0.1)

    def test_world_origin_frame_is_centered_at_zero(self):
        frame = world_origin_frame_mesh(axis_length=0.12, origin_size=0.012)
        self.assertTrue(np.all(frame.bounds[0] <= 0.0))
        self.assertTrue(np.all(frame.bounds[1] >= 0.0))
        self.assertTrue(np.all(frame.bounds[1] > 0.1))

    def test_gripper_markers_are_centered_on_pose(self):
        pose = np.eye(4)
        pose[:3, 3] = [0.9, -0.2, 0.7]
        center = gripper_center_mesh(pose, [255, 160, 0, 255])
        frame = gripper_frame_mesh(pose, axis_length=0.08)

        np.testing.assert_allclose(center.centroid, pose[:3, 3], atol=1e-12)
        self.assertTrue(np.all(frame.bounds[0] <= pose[:3, 3]))
        self.assertTrue(np.all(frame.bounds[1] >= pose[:3, 3]))


if __name__ == "__main__":
    unittest.main()
