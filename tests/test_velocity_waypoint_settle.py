import unittest
from unittest import mock

from airsim_plugin import AirVLNSimulatorClientTool as client_module


class _AsyncResult:
    def join(self):
        return None


class _FakeClient:
    def enableApiControl(self, *_args, **_kwargs):
        return None

    def armDisarm(self, *_args, **_kwargs):
        return None

    def simPause(self, *_args, **_kwargs):
        return None

    def simSetKinematics(self, *_args, **_kwargs):
        return None

    def moveByVelocityAsync(self, *_args, **_kwargs):
        return _AsyncResult()


class _FakeState:
    def __init__(self, *_args, **_kwargs):
        self.timestamp = 0

    def retrieve(self):
        self.timestamp += 100_000_000
        return {
            "timestamp": self.timestamp,
            "position": [0.0, 0.0, 0.0],
            "collision": {"has_collided": False},
        }


class _FakeImu:
    def __init__(self, *_args, **_kwargs):
        pass

    def retrieve(self):
        return {"rotation": [[1.0, 0.0, 0.0]] * 3}


class VelocityWaypointSettleTest(unittest.TestCase):
    def test_fast_controller_accepts_stalled_close_waypoint(self):
        with mock.patch.object(client_module, "State", _FakeState), mock.patch.object(
            client_module, "Imu", _FakeImu
        ):
            tool = client_module.AirVLNSimulatorClientTool.__new__(
                client_module.AirVLNSimulatorClientTool
            )
            tool.fast_eval = True
            tool.fast_eval_speedup = 10.0
            tool.airsim_clients = [[_FakeClient()]]

            result = tool.move_path_by_velocity_waypoints(
                waypoints_list=[[[[0.25, 0.0, 0.0]]]],
                start_states=[[object()]],
                target_idx=1,
            )

        action = result[0][0]
        self.assertFalse(action["collision"])
        self.assertEqual(len(action["states"]), 1)
        self.assertGreaterEqual(action["sim_time_ms"], 1000.0)
        self.assertLess(action["sim_time_ms"], 2000.0)
