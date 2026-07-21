import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gripper_pose import fixed_joint_transform, resolve_gripper_poses


def rz(angle):
    transform = np.eye(4)
    transform[:3, :3] = [
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    return transform


class GripperPoseTests(unittest.TestCase):
    def make_fixture(self, root: Path):
        g1_urdf = root / "G1.urdf"
        g1_urdf.write_text(
            """<robot name="g1">
              <joint name="left_hand_joint" type="fixed">
                <parent link="arm_left_link7"/><child link="hand_left_base_link"/>
                <origin xyz="0 0 0" rpy="0 0 -0.5236"/>
              </joint>
              <joint name="right_hand_joint" type="fixed">
                <parent link="arm_right_link7"/><child link="hand_right_base_link"/>
                <origin xyz="0 0 0" rpy="0 0 0.5236"/>
              </joint>
            </robot>""",
            encoding="utf-8",
        )
        gripper_urdf = root / "gripper.urdf"
        gripper_urdf.write_text(
            """<robot name="gripper">
              <joint name="idx52_gripper_l_center_joint" type="fixed">
                <parent link="gripper_l_base_link"/><child link="gripper_l_center_link"/>
                <origin xyz="0 0 0.14308" rpy="0 0 -1.5707963267948966"/>
              </joint>
              <joint name="idx92_gripper_r_center_joint" type="fixed">
                <parent link="gripper_r_base_link"/><child link="gripper_r_center_link"/>
                <origin xyz="0 0 0.14308" rpy="0 0 -1.5707963267948966"/>
              </joint>
            </robot>""",
            encoding="utf-8",
        )

        left_link7 = np.eye(4)
        left_link7[:3, 3] = [1.0, 2.0, 3.0]
        right_link7 = rz(math.pi / 2.0)
        right_link7[:3, 3] = [-1.0, -2.0, 0.5]
        left_hand = left_link7 @ rz(-0.5236)
        right_hand = right_link7 @ rz(0.5236)

        state = {
            "wbc_link7_capture": {
                "complete": True,
                "world_frame": "base_link",
                "pose_direction": "base_T_frame",
                "views": {
                    "hand_left": {
                        "frames": {
                            "arm_left_link7": {"base_T_frame": left_link7.tolist()}
                        }
                    },
                    "hand_right": {
                        "frames": {
                            "arm_right_link7": {"base_T_frame": right_link7.tolist()}
                        }
                    },
                },
            }
        }
        manifest = {
            "output_contract": {"world_frame": "base_link"},
            "intermediate_transforms": {
                "base_T_parent": {
                    "hand_left": {
                        "parent": "base_link",
                        "child": "Link_hand_l",
                        "matrix": left_hand.tolist(),
                    },
                    "hand_right": {
                        "parent": "base_link",
                        "child": "Link_hand_r",
                        "matrix": right_hand.tolist(),
                    },
                }
            },
        }
        (root / "capture_state.json").write_text(json.dumps(state), encoding="utf-8")
        (root / "pose_conversion_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (root / "camera_poses_used_for_export.json").write_text(
            json.dumps({"world_frame": "base_link"}), encoding="utf-8"
        )
        return g1_urdf, gripper_urdf

    def test_resolves_center_in_base_link_and_crosschecks_mount(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            g1_urdf, gripper_urdf = self.make_fixture(root)
            result = resolve_gripper_poses(
                root, g1_urdf=g1_urdf, gripper_urdf=gripper_urdf
            )

        self.assertEqual(result["world_frame"], "base_link")
        np.testing.assert_allclose(
            result["poses"]["left"]["position_m"], [1.0, 2.0, 3.14308]
        )
        np.testing.assert_allclose(
            result["poses"]["right"]["position_m"], [-1.0, -2.0, 0.64308]
        )
        for side in ("left", "right"):
            check = result["poses"][side]["mount_crosscheck"]
            self.assertLess(check["translation_error_m"], 1e-12)
            self.assertLess(check["rotation_error_deg"], 1e-6)

    def test_fixed_joint_lookup_rejects_nonfixed_joint(self):
        import xml.etree.ElementTree as ET

        root = ET.fromstring(
            """<robot><joint name="moving" type="revolute">
            <parent link="a"/><child link="b"/></joint></robot>"""
        )
        with self.assertRaisesRegex(ValueError, "must be fixed"):
            fixed_joint_transform(root, parent="a", child="b")

    def test_rejects_non_base_link_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            g1_urdf, gripper_urdf = self.make_fixture(root)
            (root / "camera_poses_used_for_export.json").write_text(
                json.dumps({"world_frame": "head_rgb_opencv_at_capture"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "reconstruction world_frame"):
                resolve_gripper_poses(
                    root, g1_urdf=g1_urdf, gripper_urdf=gripper_urdf
                )


if __name__ == "__main__":
    unittest.main()
