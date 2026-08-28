import os
import sys
import unittest

import numpy as np

sys.argv = [sys.argv[0]]

from src.vlnce_src.env_uav import AirVLNENV


class _SimulatorTool:
    def __init__(self, policy_response, record_response):
        self.policy_response = policy_response
        self.record_response = record_response
        self.policy_cameras = None

    def getImageResponses(self, cameras):
        self.policy_cameras = cameras
        return self.policy_response

    def getImageResponsesForRecord(self, cameras):
        return self.record_response


class AeroDPORecordFramesTest(unittest.TestCase):
    def test_aerodpo_requests_only_policy_views_and_skips_record_stream(self):
        rgb = [np.full((2, 2, 3), value, np.uint8) for value in range(5)]
        depth = [np.full((2, 2), value, np.uint8) for value in range(5)]
        record_rgb = [np.full((2, 2, 3), 99, np.uint8)]
        record_depth = [np.full((2, 2), 99, np.uint8)]
        env = AirVLNENV.__new__(AirVLNENV)
        env.batch_size = 1
        env.machines_info = [{"open_scenes": ["Carla_Town04"]}]
        simulator_tool = _SimulatorTool(
            [[(rgb, depth)]], [[(record_rgb, record_depth)]]
        )
        env.simulator_tool = simulator_tool
        env.sim_states = [object()]
        env._last_frames = None
        env.aerodpo_eval_mode = True

        state = env._getStates()[0]

        self.assertEqual(simulator_tool.policy_cameras, ["FrontCamera", "DownCamera"])
        self.assertEqual(state[3], [])
        self.assertEqual(state[4], [])


if __name__ == "__main__":
    unittest.main()
