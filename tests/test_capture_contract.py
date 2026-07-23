import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from capture_contract import (
    IMAGE_TO_INTRINSIC,
    VIEW_NAMES,
    discover_raw_captures,
    validate_pose_document,
)


def valid_pose_document():
    identity = np.eye(4).tolist()
    return {
        "frame_convention": "opencv_rdf_cam2world",
        "world_frame": "head_rgb_opencv_at_capture",
        "translation_unit": "meter",
        "poses": {name: identity for name in VIEW_NAMES},
    }


class PoseContractTests(unittest.TestCase):
    def validate(self, document):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "poses.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            return validate_pose_document(path)

    def test_accepts_explicit_rdf_metric_cam2world(self):
        poses, metadata = self.validate(valid_pose_document())
        self.assertEqual(metadata["world_frame"], "head_rgb_opencv_at_capture")
        self.assertEqual(metadata["translation_unit"], "meter")
        np.testing.assert_array_equal(poses["head"], np.eye(4))

    def test_rejects_camera_to_world_direction_mismatch(self):
        document = valid_pose_document()
        document["extrinsic_direction"] = "camera_T_head"
        with self.assertRaisesRegex(ValueError, "world_T_camera"):
            self.validate(document)

    def test_rejects_non_rigid_rotation(self):
        document = valid_pose_document()
        document["poses"]["hand_left"][0][0] = 2.0
        with self.assertRaisesRegex(ValueError, "not in SO\\(3\\)"):
            self.validate(document)

    def test_head_centered_world_requires_identity_head(self):
        document = valid_pose_document()
        document["poses"]["head"][0][3] = 0.1
        with self.assertRaisesRegex(ValueError, "head pose is not identity"):
            self.validate(document)

    def test_legacy_contract_is_accepted_but_unit_assumption_is_recorded(self):
        document = valid_pose_document()
        document["frame_convention"] = "opencv_cam2world"
        document["world_frame"] = "end"
        del document["translation_unit"]
        _, metadata = self.validate(document)
        self.assertTrue(metadata["legacy_translation_unit_assumption"])


class DiscoveryTests(unittest.TestCase):
    def test_discovers_only_complete_capture_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            complete = root / "complete"
            complete.mkdir()
            for name in VIEW_NAMES:
                (complete / f"{name}.png").touch()
            for filename in IMAGE_TO_INTRINSIC.values():
                (complete / filename).touch()
            (root / "incomplete").mkdir()
            self.assertEqual(discover_raw_captures(root), ["complete"])



class PresentViewsTests(unittest.TestCase):
    def test_full_and_subset_and_empty(self):
        from capture_contract import present_views, VIEW_NAMES

        full = {f"{n}_pts3d": 1 for n in VIEW_NAMES}
        self.assertEqual(present_views(full), list(VIEW_NAMES))
        subset = {"head_pts3d": 1, "hand_left_pts3d": 1}
        self.assertEqual(present_views(subset), ["head", "hand_left"])
        with self.assertRaises(ValueError):
            present_views({"nothing": 1})

if __name__ == "__main__":
    unittest.main()
