"""Small, dependency-free parts of the AeroDPO Stop-and-go contract."""

import re

import numpy as np


AERODPO_SUCCESS_DISTANCE_M = 20.0
NUM_BINS = 99
NORM_STATS = {
    "fwd": (0.0, 5.0),
    "down": (-5.0, 5.0),
    "yaw": (-1.1, 1.1),
}


def _dequantize(bin_value, axis):
    lower, upper = NORM_STATS[axis]
    clipped = max(0, min(NUM_BINS - 1, int(bin_value)))
    return (clipped / (NUM_BINS - 1)) * (upper - lower) + lower


def parse_aerodpo_action_text(text):
    """Match AeroDPO's native first-three-integers action decoder."""
    numbers = re.findall(r"\d+", text)
    action = {"fwd": 0.0, "down": 0.0, "yaw": 0.0}
    if len(numbers) >= 3:
        action = {
            "fwd": _dequantize(numbers[0], "fwd"),
            "down": _dequantize(numbers[1], "down"),
            "yaw": _dequantize(numbers[2], "yaw"),
        }
    model_stop = (
        "LAND" in text
        or "<LAND>" in text
        or (action["fwd"] < 0.01 and abs(action["down"]) < 0.01 and abs(action["yaw"]) < 0.01)
    )
    return action, model_stop


def resolve_aerodpo_stop(model_stop, distance_m):
    if not model_stop:
        return "continue"
    if float(distance_m) <= AERODPO_SUCCESS_DISTANCE_M:
        return "success"
    return "early_end"


def summarize_latencies(episode_latencies_ms):
    """Return decision-weighted and episode-balanced latency summaries."""
    nonempty = [list(map(float, values)) for values in episode_latencies_ms if values]
    flat = [value for values in nonempty for value in values]
    if not flat:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "first_call_ms": None,
            "episode_balanced_mean": 0.0,
        }
    values = np.asarray(flat, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "first_call_ms": float(values[0]),
        "episode_balanced_mean": float(np.mean([np.mean(values) for values in nonempty])),
    }
