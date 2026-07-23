import tempfile
import unittest
from pathlib import Path

from src.vlnce_src.comm_delay import BandwidthTrace


class BandwidthTraceTest(unittest.TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        handle.write("bandwidth_mbps\n")
        for value in range(1, 11):
            handle.write(f"{value}\n")
        handle.close()
        self.path = Path(handle.name)

    def tearDown(self):
        self.path.unlink(missing_ok=True)

    def test_same_episode_always_uses_same_sequence(self):
        trace = BandwidthTrace(self.path)
        trace.reset_for_episode("episode-a")
        first = [trace.next_bandwidth_bps() for _ in range(4)]
        trace.reset_for_episode("episode-a")
        second = [trace.next_bandwidth_bps() for _ in range(4)]
        self.assertEqual(first, second)

    def test_previous_episode_length_does_not_change_next_episode(self):
        trace = BandwidthTrace(self.path)
        trace.reset_for_episode("episode-a")
        for _ in range(7):
            trace.next_bandwidth_bps()
        trace.reset_for_episode("episode-b")
        after_long_episode = trace.next_bandwidth_bps()

        trace.reset_for_episode("episode-a")
        trace.next_bandwidth_bps()
        trace.reset_for_episode("episode-b")
        after_short_episode = trace.next_bandwidth_bps()
        self.assertEqual(after_long_episode, after_short_episode)


if __name__ == "__main__":
    unittest.main()
