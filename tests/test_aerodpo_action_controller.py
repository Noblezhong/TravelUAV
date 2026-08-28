import sys
import time
import unittest
from unittest import mock

sys.argv = [sys.argv[0]]

from airsim_plugin import AirVLNSimulatorClientTool as client_module


class _AsyncResult:
    def join(self):
        return None


class _SlowAsyncResult:
    def join(self):
        time.sleep(0.05)


class _FakeClient:
    def __init__(self):
        self.rotate_commands = []
        self.velocity_commands = []
        self.z_commands = []

    def enableApiControl(self, *_args, **_kwargs):
        return None

    def armDisarm(self, *_args, **_kwargs):
        return None

    def simPause(self, *_args, **_kwargs):
        return None

    def rotateToYawAsync(self, yaw_deg, *_args, **_kwargs):
        self.rotate_commands.append(yaw_deg)
        return _AsyncResult()

    def moveByVelocityAsync(self, vx, vy, vz, duration, *_args, **_kwargs):
        self.velocity_commands.append((vx, vy, vz, duration))
        return _AsyncResult()

    def moveToZAsync(self, z, *_args, **_kwargs):
        self.z_commands.append(z)
        return _AsyncResult()


class _SlowClient(_FakeClient):
    def rotateToYawAsync(self, *_args, **_kwargs):
        return _SlowAsyncResult()


class _FakeState:
    def __init__(self, *_args, **_kwargs):
        self.timestamp = 0

    def retrieve(self):
        self.timestamp += 500_000_000
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


class AeroDPOActionControllerTest(unittest.TestCase):
    def test_preserves_native_rotate_move_hover_and_measured_timing(self):
        with mock.patch.object(client_module, "State", _FakeState), mock.patch.object(
            client_module, "Imu", _FakeImu
        ):
            tool = client_module.AirVLNSimulatorClientTool.__new__(
                client_module.AirVLNSimulatorClientTool
            )
            tool.fast_eval = True
            tool.fast_eval_speedup = 10.0
            fake_client = _FakeClient()
            tool.airsim_clients = [[fake_client]]
            start_state = type(
                "StartState",
                (),
                {
                    "position": type("Position", (), {"x_val": 0.0, "y_val": 0.0, "z_val": 0.0})(),
                    "orientation": type("Orientation", (), {"x_val": 0.0, "y_val": 0.0, "z_val": 0.0, "w_val": 1.0})(),
                },
            )()

            result = tool.move_path_by_aerodpo_actions(
                actions_list=[[{"fwd": 1.0, "down": 0.2, "yaw": 0.1}]],
                start_states=[[start_state]],
            )

        action = result[0][0]
        self.assertFalse(action["collision"])
        self.assertEqual(len(action["states"]), 5)
        self.assertGreater(action["sim_time_ms"], 0.0)
        self.assertTrue(fake_client.rotate_commands)
        self.assertGreater(fake_client.velocity_commands[0][0], 0.0)
        self.assertEqual(fake_client.velocity_commands[-1][0:3], (0, 0, 0))

    def test_returns_failure_when_native_action_thread_exceeds_timeout(self):
        with mock.patch.object(client_module, "State", _FakeState), mock.patch.object(
            client_module, "Imu", _FakeImu
        ):
            tool = client_module.AirVLNSimulatorClientTool.__new__(
                client_module.AirVLNSimulatorClientTool
            )
            tool.AERODPO_ACTION_TIMEOUT_S = 0.01
            tool.airsim_clients = [[_SlowClient()]]
            start_state = type(
                "StartState",
                (),
                {
                    "position": type("Position", (), {"x_val": 0.0, "y_val": 0.0, "z_val": 0.0})(),
                    "orientation": type("Orientation", (), {"x_val": 0.0, "y_val": 0.0, "z_val": 0.0, "w_val": 1.0})(),
                },
            )()

            result = tool.move_path_by_aerodpo_actions(
                actions_list=[[{"fwd": 1.0, "down": 0.0, "yaw": 0.1}]],
                start_states=[[start_state]],
            )

        self.assertIsNone(result)
