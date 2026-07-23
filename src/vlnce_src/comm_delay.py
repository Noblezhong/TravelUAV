import csv
import hashlib
from pathlib import Path

import numpy as np


def default_trace_path() -> Path:
    return Path(__file__).resolve().parents[2] / "bandwidth" / "ucc4g_bandwidth_trace.csv"


class BandwidthTrace:
    def __init__(self, csv_path, cycle=True):
        self.csv_path = Path(csv_path)
        self.cycle = bool(cycle)
        self.samples_bps = self._load_samples(self.csv_path)
        if not self.samples_bps:
            raise ValueError(f"No bandwidth samples found in {self.csv_path}")
        self.sample_count = len(self.samples_bps)
        self._index = 0

    def _load_samples(self, csv_path):
        samples = []
        with open(csv_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                value = None
                for key in ("bandwidth_mbps", "Bandwidth_Mbps", "mbps", "Mbps"):
                    if key in row and row[key] not in (None, ""):
                        value = float(row[key]) * 1e6
                        break
                if value is None:
                    for cell in row.values():
                        if cell not in (None, ""):
                            value = float(cell) * 1e6
                            break
                if value is not None and value > 0:
                    samples.append(value)
        return samples

    def next_bandwidth_bps(self):
        value = self.samples_bps[self._index]
        if self._index + 1 < self.sample_count:
            self._index += 1
        elif self.cycle:
            self._index = 0
        return value

    def reset(self, index=0):
        self._index = int(index) % self.sample_count

    def reset_for_episode(self, seq_name):
        digest = hashlib.sha256(str(seq_name).encode("utf-8")).hexdigest()
        self.reset(int(digest[:8], 16))


def calculate_latency_ms(payload_bits, bandwidth_bps):
    if bandwidth_bps <= 0:
        raise ValueError("bandwidth_bps must be positive")
    return float(payload_bits) / float(bandwidth_bps) * 1000.0


def estimate_uplink_payload_bits_from_outputs(outputs):
    payload_bytes = 0
    observations = []
    for item in outputs:
        if isinstance(item, (list, tuple)) and len(item) > 0:
            env_obs = item[0]
        else:
            env_obs = item
        if isinstance(env_obs, (list, tuple)) and len(env_obs) > 0:
            observations.append(env_obs)

    for env_obs in observations:
        latest = env_obs[-1]
        for key in ("rgb", "depth", "rgb_record", "depth_record"):
            if key not in latest:
                continue
            for item in latest[key]:
                payload_bytes += int(np.asarray(item).nbytes)
    payload_bits = payload_bytes * 8
    payload_mb = payload_bytes / (1024.0 * 1024.0)
    return payload_bytes, payload_bits, payload_mb


def estimate_uplink_payload_bits_from_episodes(episodes):
    payload_bytes = 0
    for episode in episodes:
        if not episode:
            continue
        latest = episode[-1]
        for key in ("rgb", "depth", "rgb_record", "depth_record"):
            if key not in latest:
                continue
            for item in latest[key]:
                payload_bytes += int(np.asarray(item).nbytes)
    payload_bits = payload_bytes * 8
    payload_mb = payload_bytes / (1024.0 * 1024.0)
    return payload_bytes, payload_bits, payload_mb
