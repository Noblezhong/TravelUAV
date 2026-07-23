import unittest

import numpy as np

from src.model_wrapper.edge_traj_dnn import EdgeDNNModelWrapper, build_trajcorr_target


class TrajCorrGeometryTest(unittest.TestCase):
    def test_preserves_original_coarse_length_and_goal_bearing(self):
        result = build_trajcorr_target(
            current_pose=[2.0, 0.0, 0.0],
            current_rotation=np.eye(3),
            coarse_goal_world=[12.0, 0.0, 0.0],
            coarse_local=[5.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(result["virtual_goal_world"], [7.0, 0.0, 0.0])
        self.assertAlmostEqual(result["original_coarse_norm_m"], 5.0)
        self.assertAlmostEqual(result["trajcorr_coarse_norm_m"], 5.0)

    def test_near_goal_keeps_training_scale(self):
        result = build_trajcorr_target(
            current_pose=[2.0, 0.0, 0.0],
            current_rotation=np.eye(3),
            coarse_goal_world=[4.0, 0.0, 0.0],
            coarse_local=[5.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(result["virtual_goal_world"], [7.0, 0.0, 0.0])
        self.assertAlmostEqual(np.linalg.norm(result["local_target"]), 5.0)

    def test_body_frame_target_keeps_length(self):
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        result = build_trajcorr_target(
            current_pose=[0.0, 0.0, 0.0],
            current_rotation=rotation,
            coarse_goal_world=[0.0, 10.0, 0.0],
            coarse_local=[3.0, 4.0, 0.0],
        )
        np.testing.assert_allclose(result["local_target"], [5.0, 0.0, 0.0])
        self.assertAlmostEqual(result["trajcorr_coarse_norm_m"], 5.0)

    def test_goal_overlap_uses_original_world_bearing(self):
        result = build_trajcorr_target(
            current_pose=[5.0, 0.0, 0.0],
            current_rotation=np.eye(3),
            coarse_goal_world=[5.0, 0.0, 0.0],
            coarse_local=[5.0, 0.0, 0.0],
            observation_pose=[0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(result["virtual_goal_world"], [10.0, 0.0, 0.0])
        self.assertAlmostEqual(np.linalg.norm(result["local_target"]), 5.0)

    def test_passed_goal_continues_forward_instead_of_reversing(self):
        result = build_trajcorr_target(
            current_pose=[6.0, 0.0, 0.0],
            current_rotation=np.eye(3),
            coarse_goal_world=[5.0, 0.0, 0.0],
            coarse_local=[5.0, 0.0, 0.0],
            observation_pose=[0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(result["virtual_goal_world"], [11.0, 0.0, 0.0])
        self.assertAlmostEqual(np.linalg.norm(result["local_target"]), 5.0)

    def test_dnn_output_is_not_rescaled_to_stale_goal(self):
        wrapper = EdgeDNNModelWrapper.__new__(EdgeDNNModelWrapper)
        local_waypoints = np.asarray(
            [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
              [4.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0],
              [7.0, 0.0, 0.0]]],
            dtype=np.float64,
        )
        wrapper._infer_local_waypoints = lambda episodes, targets: (
            local_waypoints,
            {"traj_latency_ms": 1.0, "reprojected_coarse": targets.tolist()},
        )
        episode = [[{
            "sensors": {
                "state": {"position": [0.0, 0.0, 0.0]},
                "imu": {"rotation": np.eye(3).tolist()},
            }
        }]]
        refined, profile = wrapper.run_trajcorr_from_world_goal(
            episode,
            coarse_goals_world=[[2.0, 0.0, 0.0]],
            coarse_locals=[[5.0, 0.0, 0.0]],
            observation_poses=[[0.0, 0.0, 0.0]],
        )
        np.testing.assert_allclose(refined[0][4], [5.0, 0.0, 0.0])
        self.assertAlmostEqual(profile["trajectory_scale"][0], 1.0)
        self.assertAlmostEqual(profile["p5_to_virtual_goal_m"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
