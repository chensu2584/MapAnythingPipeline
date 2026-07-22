"""G2 snapshot contract tests: every check must fail closed."""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from g2_capture import (
    EXTRINSICS_FILE,
    discover_g2_snapshots,
    load_depth_png,
    load_g2_snapshot,
    validate_transform,
)
from depth_tools import fit_scale_robust, register_depth_to_camera
from pose_export import (
    MODEL_RELATIVE_HEAD_ANCHORED_BASELINE_SCALED,
    MODEL_RELATIVE_HEAD_ANCHORED_DEPTH_SCALED,
    select_export_poses,
)
from robot_profiles import G2_PROFILE, detect_profile


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def transform_entry(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    rotation = matrix[:3, :3]
    trace = np.trace(rotation)
    w = np.sqrt(max(0.0, 1.0 + trace)) / 2.0
    x = (rotation[2, 1] - rotation[1, 2]) / (4 * w)
    y = (rotation[0, 2] - rotation[2, 0]) / (4 * w)
    z = (rotation[1, 0] - rotation[0, 1]) / (4 * w)
    return {
        "matrix": matrix.tolist(),
        "translation_xyz_m": matrix[:3, 3].tolist(),
        "quaternion_xyzw": [x, y, z, w],
        "inverse_matrix": np.linalg.inv(matrix).tolist(),
    }


INTRINSIC = {
    "Fx": 300.0, "Fy": 300.0, "Cx": 160.0, "Cy": 120.0,
    "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0,
}


class G2SnapshotTests(unittest.TestCase):
    def make_snapshot(self, root: Path, **overrides) -> Path:
        snapshot = root / "snapshot_0001"
        snapshot.mkdir(parents=True)
        keys = {
            "head_rgb": "head_rgb.png",
            "hand_left_rgb": "hand_left_rgb.png",
            "hand_right_rgb": "hand_right_rgb.png",
        }
        for filename in keys.values():
            cv2.imwrite(str(snapshot / filename), np.zeros((240, 320, 3), np.uint8))
        depth = np.full((240, 320), 1000, np.uint16)
        depth[0, 0] = 0
        cv2.imwrite(str(snapshot / "head_depth.png"), depth)

        captures = {}
        for index, (key, filename) in enumerate(keys.items()):
            captures[key] = {
                "kind": "color",
                "timestamp_ns": 1_000_000_000,
                "saved_path": filename,
                "error": "",
                "shape": [240, 320, 3],
                "intrinsic": dict(INTRINSIC),
            }
        captures["head_depth"] = {
            "kind": "depth",
            "timestamp_ns": 1_000_000_000,
            "saved_path": "head_depth.png",
            "error": "",
            "shape": [240, 320],
            "intrinsic": dict(INTRINSIC),
        }
        extrinsics = {}
        for index, key in enumerate([*keys, "head_depth"]):
            matrix = np.eye(4)
            matrix[:3, :3] = rotation_z(0.1 * index)
            matrix[:3, 3] = [0.1 * index, 0.0, 1.0]
            extrinsics[key] = transform_entry(matrix)

        document = {
            "base_link": "base_link",
            "convention": {
                "base_T_camera": (
                    "4x4 transform mapping homogeneous camera-frame points into base_link"
                )
            },
            "captures": captures,
            "extrinsics": extrinsics,
            "joint_positions_rad": {"idx01_body_joint1": 0.5},
            "validation": {
                "fk_vs_sdk_tf": {
                    "head_link3": {"translation_error_m": 0.0, "rotation_error_deg": 0.0}
                }
            },
        }
        for key, value in overrides.items():
            document[key] = value
        (snapshot / EXTRINSICS_FILE).write_text(json.dumps(document), encoding="utf-8")
        return snapshot

    def test_valid_snapshot_loads_with_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_snapshot(root)
            self.assertEqual(discover_g2_snapshots(root), ["snapshot_0001"])
            self.assertIs(detect_profile(root), G2_PROFILE)
            capture = load_g2_snapshot(
                root / "snapshot_0001", profile=G2_PROFILE, capture="snapshot_0001"
            )
            self.assertEqual(sorted(capture.views), ["hand_left", "hand_right", "head"])
            self.assertIn("head", capture.depths)
            self.assertEqual(capture.world_frame, "base_link")
            self.assertAlmostEqual(capture.joint_positions["idx01_body_joint1"], 0.5)

    def test_failed_kinematic_check_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_snapshot(
                root,
                validation={
                    "fk_vs_sdk_tf": {
                        "head_link3": {
                            "translation_error_m": 0.05,
                            "rotation_error_deg": 0.0,
                        }
                    }
                },
            )
            with self.assertRaisesRegex(ValueError, "kinematic check"):
                load_g2_snapshot(
                    root / "snapshot_0001", profile=G2_PROFILE, capture="snapshot_0001"
                )

    def test_desynchronised_cameras_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = self.make_snapshot(root)
            document = json.loads((snapshot / EXTRINSICS_FILE).read_text())
            document["captures"]["hand_left_rgb"]["timestamp_ns"] = 2_000_000_000
            (snapshot / EXTRINSICS_FILE).write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "desynchronised"):
                load_g2_snapshot(snapshot, profile=G2_PROFILE, capture="snapshot_0001")

    def test_camera_reported_error_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = self.make_snapshot(root)
            document = json.loads((snapshot / EXTRINSICS_FILE).read_text())
            document["captures"]["head_rgb"]["error"] = "timeout"
            (snapshot / EXTRINSICS_FILE).write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reported an error"):
                load_g2_snapshot(snapshot, profile=G2_PROFILE, capture="snapshot_0001")

    def test_non_rigid_extrinsic_is_rejected(self):
        entry = transform_entry(np.eye(4))
        entry["matrix"][0][0] = 2.0
        with self.assertRaisesRegex(ValueError, "SO\\(3\\)"):
            validate_transform(entry, "test")

    def test_inconsistent_inverse_copy_is_rejected(self):
        entry = transform_entry(np.eye(4))
        entry["inverse_matrix"][0][3] = 5.0
        with self.assertRaisesRegex(ValueError, "inverse_matrix"):
            validate_transform(entry, "test")

    def test_depth_sentinels_become_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.png"
            raw = np.array([[0, 1000], [65535, 2500]], np.uint16)
            cv2.imwrite(str(path), raw)
            depth, valid = load_depth_png(path)
            self.assertTrue(np.array_equal(valid, [[False, True], [False, True]]))
            self.assertAlmostEqual(depth[0, 1], 1.0)
            self.assertAlmostEqual(depth[1, 1], 2.5)
            self.assertTrue(np.isnan(depth[0, 0]))


class DepthToolsTests(unittest.TestCase):
    def test_identity_registration_is_a_round_trip(self):
        K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])
        depth = np.full((240, 320), 2.0)
        valid = np.ones((240, 320), bool)
        out, out_valid, report = register_depth_to_camera(
            depth, valid,
            K_source=K, base_T_source=np.eye(4),
            K_target=K, base_T_target=np.eye(4),
            target_shape=(240, 320),
        )
        self.assertEqual(report["dropped_behind_camera"], 0)
        self.assertEqual(report["dropped_outside_frame"], 0)
        self.assertTrue(out_valid.all())
        np.testing.assert_allclose(out, depth)

    def test_translated_camera_shifts_depth(self):
        K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])
        depth = np.full((240, 320), 2.0)
        valid = np.ones((240, 320), bool)
        target = np.eye(4)
        target[2, 3] = 0.5  # move the target camera 0.5 m along its own axis
        out, out_valid, _ = register_depth_to_camera(
            depth, valid,
            K_source=K, base_T_source=np.eye(4),
            K_target=K, base_T_target=target,
            target_shape=(240, 320),
        )
        self.assertAlmostEqual(float(np.nanmedian(out)), 1.5, places=6)

    def test_scale_fit_recovers_a_known_scale(self):
        rng = np.random.default_rng(0)
        model = rng.uniform(0.5, 3.0, 5000)
        reference = 1.37 * model + rng.normal(0, 0.004, 5000)
        result = fit_scale_robust(model, reference, np.ones(5000, bool))
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["scale"], 1.37, places=3)
        self.assertLess(abs(result["affine_test"]["b_m"]), 0.01)

    def test_scale_fit_survives_outliers(self):
        rng = np.random.default_rng(1)
        model = rng.uniform(0.5, 3.0, 5000)
        reference = 1.37 * model
        reference[:750] = rng.uniform(0.1, 10.0, 750)  # 15 percent gross outliers
        result = fit_scale_robust(model, reference, np.ones(5000, bool))
        self.assertAlmostEqual(result["scale"], 1.37, places=2)

    def test_scale_fit_refuses_too_few_pixels(self):
        result = fit_scale_robust(np.ones(10), np.ones(10), np.ones(10, bool))
        self.assertFalse(result["converged"])

    def test_affine_test_exposes_a_constant_offset(self):
        """A pure-scale fit must not quietly absorb an additive depth bias."""
        rng = np.random.default_rng(2)
        reference = rng.uniform(0.5, 3.0, 20000)
        model = (reference - 0.08) / 1.15  # scale error *and* an 80 mm offset
        result = fit_scale_robust(model, reference, np.ones(20000, bool))
        self.assertAlmostEqual(result["affine_test"]["b_m"], 0.08, places=3)
        self.assertAlmostEqual(result["affine_test"]["a"], 1.15, places=3)
        # With no noise added, a single scale that actually fit would leave a
        # near-zero residual.  A large one is the signal that it does not fit.
        self.assertGreater(result["residual_rmse_m"], 0.01)


class DepthScaledModeTests(unittest.TestCase):
    @staticmethod
    def poses(spacing):
        result = []
        for index in range(3):
            pose = np.eye(4)
            pose[:3, 3] = [spacing * index, 0.0, 0.0]
            result.append(pose)
        return result

    def test_depth_scale_is_used_when_it_is_defendable(self):
        rng = np.random.default_rng(3)
        reference = rng.uniform(0.5, 3.0, (200, 200))
        model = reference / 1.15
        _, mode, _, report = select_export_poses(
            self.poses(0.5), self.poses(0.45),
            requested_mode=MODEL_RELATIVE_HEAD_ANCHORED_DEPTH_SCALED,
            depth_scale_inputs={
                "model_depth": model,
                "reference_depth": reference,
                "valid": np.ones((200, 200), bool),
            },
        )
        self.assertEqual(mode, MODEL_RELATIVE_HEAD_ANCHORED_DEPTH_SCALED)
        self.assertTrue(report["applied"])
        self.assertAlmostEqual(report["scale"], 1.15, places=4)

    def test_missing_depth_falls_back_and_records_why(self):
        _, mode, _, report = select_export_poses(
            self.poses(0.5), self.poses(0.45),
            requested_mode=MODEL_RELATIVE_HEAD_ANCHORED_DEPTH_SCALED,
            depth_scale_inputs=None,
        )
        self.assertEqual(mode, MODEL_RELATIVE_HEAD_ANCHORED_BASELINE_SCALED)
        self.assertTrue(report["fell_back_to_baseline"])
        self.assertIn("no registered metric depth", report["depth_scale_rejected"]["reason"])

    def test_implausible_depth_scale_falls_back(self):
        rng = np.random.default_rng(4)
        reference = rng.uniform(0.5, 3.0, (200, 200))
        junk = rng.uniform(0.2, 5.0, (200, 200))
        _, mode, _, report = select_export_poses(
            self.poses(0.5), self.poses(0.45),
            requested_mode=MODEL_RELATIVE_HEAD_ANCHORED_DEPTH_SCALED,
            depth_scale_inputs={
                "model_depth": junk,
                "reference_depth": reference,
                "valid": np.ones((200, 200), bool),
            },
        )
        self.assertEqual(mode, MODEL_RELATIVE_HEAD_ANCHORED_BASELINE_SCALED)
        self.assertFalse(report["depth_scale_rejected"]["applied"])


class MaskReprojectionTests(unittest.TestCase):
    """Preprocessing crops as well as resizes, so size-only mapping is wrong."""

    @staticmethod
    def reproject(*args):
        import importlib.util

        source = (Path(__file__).resolve().parents[1] / "run_inference.py").read_text()
        start = source.index("def reproject_mask_with_intrinsics")
        end = source.index("def resample_depth_to")
        namespace = {"np": np}
        exec(source[start:end], namespace)  # noqa: S102 - isolated helper
        return namespace["reproject_mask_with_intrinsics"](*args)

    def test_identity_intrinsics_round_trip(self):
        mask = np.zeros((40, 50), bool)
        mask[10:20, 15:30] = True
        K = np.array([[100.0, 0, 25.0], [0, 100.0, 20.0], [0, 0, 1.0]])
        out = self.reproject(mask, K, K, (40, 50))
        np.testing.assert_array_equal(out, mask)

    def test_a_cropped_target_maps_by_intrinsics_not_by_size(self):
        """A centre crop must sample the centre, not the whole frame."""
        mask = np.zeros((100, 100), bool)
        mask[:, :50] = True  # left half is robot
        K_source = np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
        # Target sees the central 50x50 of the source at the same resolution.
        K_target = np.array([[100.0, 0, 25.0], [0, 100.0, 25.0], [0, 0, 1.0]])
        out = self.reproject(mask, K_source, K_target, (50, 50))
        # Source columns 25..75 -> the left 25 target columns are robot.
        self.assertTrue(out[:, :24].all())
        self.assertFalse(out[:, 26:].any())
        # Size-only mapping would have called half the frame robot instead.
        self.assertNotAlmostEqual(out.mean(), 0.5, places=2)

    def test_target_pixels_outside_the_source_are_not_robot(self):
        mask = np.ones((20, 20), bool)
        K_source = np.array([[10.0, 0, 10.0], [0, 10.0, 10.0], [0, 0, 1.0]])
        # Target is twice as wide a field, so its edges fall outside the source.
        K_target = np.array([[5.0, 0, 20.0], [0, 5.0, 20.0], [0, 0, 1.0]])
        out = self.reproject(mask, K_source, K_target, (40, 40))
        self.assertFalse(out[0, 0])
        self.assertTrue(out[20, 20])


if __name__ == "__main__":
    unittest.main()
