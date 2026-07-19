import unittest

import numpy as np

from src.model_wrapper.edge_traj_dnn import build_trajcorr_target


class TrajCorrGeometryTest(unittest.TestCase):
    def test_preserves_original_coarse_length_and_goal_bearing(self):
        result = build_trajcorr_target(
            current_pose=[2.0, 0.0, 0.0],
            current_rotation=np.eye(3),
            coarse_goal_world=[12.0, 0.0, 0.0],
            coarse_local=[5.0, 0.0, 0.0],
            observation_pose=[0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(result["trajcorr_goal_world"], [7.0, 0.0, 0.0])
        self.assertAlmostEqual(result["original_coarse_norm_m"], 5.0)
        self.assertAlmostEqual(result["trajcorr_coarse_norm_m"], 5.0)

    def test_current_goal_overlap_uses_observation_bearing(self):
        result = build_trajcorr_target(
            current_pose=[2.0, 0.0, 0.0],
            current_rotation=np.eye(3),
            coarse_goal_world=[2.0, 0.0, 0.0],
            coarse_local=[5.0, 0.0, 0.0],
            observation_pose=[0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(result["trajcorr_goal_world"], [7.0, 0.0, 0.0])
        self.assertAlmostEqual(np.linalg.norm(result["local_target"]), 5.0)

    def test_body_frame_target_keeps_length(self):
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        result = build_trajcorr_target(
            current_pose=[0.0, 0.0, 0.0],
            current_rotation=rotation,
            coarse_goal_world=[0.0, 10.0, 0.0],
            coarse_local=[3.0, 4.0, 0.0],
            observation_pose=[-1.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(result["local_target"], [5.0, 0.0, 0.0])
        self.assertAlmostEqual(result["trajcorr_coarse_norm_m"], 5.0)


if __name__ == "__main__":
    unittest.main()
