import unittest

import numpy as np

from pose_export import (
    CALIBRATED_INPUT,
    MODEL_PREDICTION_ARBITRARY_SCALE,
    MODEL_RELATIVE_HEAD_ANCHORED,
    MODEL_RELATIVE_HEAD_ANCHORED_BASELINE_SCALED,
    estimate_baseline_similarity_scale,
    pose_delta,
    select_export_poses,
)


def transform(translation, yaw_deg=0.0):
    angle = np.radians(yaw_deg)
    c, s = np.cos(angle), np.sin(angle)
    value = np.eye(4)
    value[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    value[:3, 3] = translation
    return value


class PoseExportTests(unittest.TestCase):
    def setUp(self):
        self.model = [
            transform([0.1, -0.2, 0.3], 12.0),
            transform([0.7, 0.1, -0.1], -8.0),
            transform([-0.5, 0.2, 0.4], 33.0),
        ]
        self.calibrated = [
            transform([1.0, 2.0, 3.0], -25.0),
            transform([1.4, 2.2, 2.8], 4.0),
            transform([0.6, 1.8, 3.1], 15.0),
        ]

    def test_model_relative_mode_anchors_head_and_preserves_relative_geometry(self):
        poses, mode, anchor, scale_report = select_export_poses(
            self.model, self.calibrated, MODEL_RELATIVE_HEAD_ANCHORED
        )
        self.assertEqual(mode, MODEL_RELATIVE_HEAD_ANCHORED)
        self.assertIsNotNone(anchor)
        self.assertIsNone(scale_report)
        np.testing.assert_allclose(poses[0], self.calibrated[0], atol=1e-12)
        for index in (1, 2):
            before = np.linalg.inv(self.model[0]) @ self.model[index]
            after = np.linalg.inv(poses[0]) @ poses[index]
            np.testing.assert_allclose(after, before, atol=1e-12)

    def test_calibrated_mode_preserves_legacy_hybrid_pose(self):
        poses, mode, anchor, scale_report = select_export_poses(
            self.model, self.calibrated, CALIBRATED_INPUT
        )
        self.assertEqual(mode, CALIBRATED_INPUT)
        self.assertIsNone(anchor)
        self.assertIsNone(scale_report)
        for actual, expected in zip(poses, self.calibrated):
            np.testing.assert_allclose(actual, expected)

    def test_pose_free_mode_uses_unanchored_model_prediction(self):
        poses, mode, anchor, scale_report = select_export_poses(
            self.model, None, MODEL_RELATIVE_HEAD_ANCHORED
        )
        self.assertEqual(mode, MODEL_PREDICTION_ARBITRARY_SCALE)
        self.assertIsNone(anchor)
        self.assertIsNone(scale_report)
        for actual, expected in zip(poses, self.model):
            np.testing.assert_allclose(actual, expected)

    def test_pose_delta_reports_translation_and_rotation(self):
        delta = pose_delta(np.eye(4), transform([0.003, 0.004, 0.0], 10.0))
        self.assertAlmostEqual(delta["translation_m"], 0.005)
        self.assertAlmostEqual(delta["rotation_deg"], 10.0)

    def test_baseline_scaled_mode_corrects_one_uniform_scale(self):
        calibrated = [
            transform([1.0, 2.0, 3.0], -25.0),
            transform([1.4, 2.0, 3.0], 5.0),
            transform([1.0, 2.6, 3.0], 15.0),
        ]
        model = [
            transform([0.2, -0.1, 0.3], 7.0),
            transform([1.0, -0.1, 0.3], 37.0),
            transform([0.2, 1.1, 0.3], 47.0),
        ]
        report = estimate_baseline_similarity_scale(model, calibrated)
        self.assertAlmostEqual(report["scale"], 0.5)
        self.assertAlmostEqual(report["baseline_rmse_after_m"], 0.0, places=12)

        poses, mode, anchor, scale_report = select_export_poses(
            model, calibrated, MODEL_RELATIVE_HEAD_ANCHORED_BASELINE_SCALED
        )
        self.assertEqual(mode, MODEL_RELATIVE_HEAD_ANCHORED_BASELINE_SCALED)
        self.assertIsNotNone(anchor)
        self.assertAlmostEqual(scale_report["scale"], 0.5)
        np.testing.assert_allclose(poses[0], calibrated[0], atol=1e-12)
        for first, second in ((0, 1), (0, 2), (1, 2)):
            calibrated_baseline = np.linalg.norm(
                calibrated[first][:3, 3] - calibrated[second][:3, 3]
            )
            final_baseline = np.linalg.norm(
                poses[first][:3, 3] - poses[second][:3, 3]
            )
            self.assertAlmostEqual(final_baseline, calibrated_baseline)


if __name__ == "__main__":
    unittest.main()
