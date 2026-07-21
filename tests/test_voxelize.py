import unittest

import numpy as np
import trimesh

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


if __name__ == "__main__":
    unittest.main()
