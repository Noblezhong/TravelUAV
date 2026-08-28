import sys
import unittest
from types import SimpleNamespace

sys.argv = [sys.argv[0]]

from src.vlnce_src.env_uav import AirVLNENV


class _SimulatorTool:
    def __init__(self):
        self.actions = None

    def move_path_by_aerodpo_actions(self, actions_list, start_states):
        self.actions = actions_list
        return [[{
            "states": [{
                "sensors": {
                    "state": {"position": [2.0, 0.0, 0.0]},
                    "imu": {},
                }
            }] * 5,
            "collision": True,
            "wall_time_ms": 4.0,
            "sim_time_ms": 50.0,
        }]]


class AeroDPOEnvAdapterTest(unittest.TestCase):
    def test_flattens_actions_and_uses_real_action_result_for_metrics(self):
        env = AirVLNENV.__new__(AirVLNENV)
        env.machines_info = [{"open_scenes": ["Carla_Town01"]}]
        env.simulator_tool = _SimulatorTool()
        env.sim_states = [SimpleNamespace(
            state={
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "linear_velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
            },
            trajectory=[],
            SUCCESS_DISTANCE=20.0,
            oracle_success=False,
            is_collisioned=False,
            step=0,
            pre_waypoints=None,
        )]
        env.batch = [{"object_position": [2.0, 0.0, 0.0]}]
        env.update_measurements = lambda: None

        outputs = env.makeAeroDPOActions([{"fwd": 2.0, "down": 0.0, "yaw": 0.0}])

        self.assertEqual(len(outputs[0]), 5)
        self.assertEqual(env.simulator_tool.actions, [[{"fwd": 2.0, "down": 0.0, "yaw": 0.0}]])
        self.assertTrue(env.sim_states[0].oracle_success)
        self.assertTrue(env.sim_states[0].is_collisioned)
        self.assertEqual(env.sim_states[0].step, 1)
        self.assertEqual(env.last_action_timings[0]["sim_time_ms"], 50.0)
