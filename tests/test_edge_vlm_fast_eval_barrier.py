import threading
import time
import unittest

from src.vlnce_src.fast_eval_time import (
    FastEvalClock,
    fast_result_is_ready,
    wait_for_fast_edge_worker_if_due,
)


class EdgeVLMFastEvalBarrierTest(unittest.TestCase):
    def test_early_result_is_held_until_modeled_ready_time(self):
        clock = FastEvalClock(True, 10.0)
        clock.advance_to(100.0)
        self.assertFalse(fast_result_is_ready(clock, True, 200.0))
        clock.advance_to(200.0)
        self.assertTrue(fast_result_is_ready(clock, True, 200.0))

    def test_modeled_edge_arrival_waits_for_real_worker_completion(self):
        clock = FastEvalClock(True, 10.0)
        clock.advance_to(100.0)
        condition = threading.Condition()
        state = {
            "has_result": False,
            "inflight": True,
            "edge_arrival_logical_ms": 100.0,
            "error": None,
        }
        returned = []

        def poll_barrier():
            with condition:
                waited_ms = wait_for_fast_edge_worker_if_due(
                    condition=condition,
                    clock=clock,
                    fast_eval=True,
                    has_result=lambda: state["has_result"],
                    is_inflight=lambda: state["inflight"],
                    edge_arrival_logical_ms=lambda: state[
                        "edge_arrival_logical_ms"
                    ],
                    get_error=lambda: state["error"],
                )
                returned.append(waited_ms)

        poll_thread = threading.Thread(
            target=poll_barrier,
            daemon=True,
        )
        poll_thread.start()
        time.sleep(0.06)
        self.assertTrue(poll_thread.is_alive())

        with condition:
            state["has_result"] = True
            state["inflight"] = False
            condition.notify_all()

        poll_thread.join(timeout=1.0)
        self.assertFalse(poll_thread.is_alive())
        self.assertEqual(len(returned), 1)
        self.assertGreaterEqual(returned[0], 50.0)
        self.assertEqual(clock.now_ms, 100.0)


if __name__ == "__main__":
    unittest.main()
