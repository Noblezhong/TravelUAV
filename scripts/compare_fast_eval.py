#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "decision_latency_ms",
    "decision_total_latency_ms",
    "airsim_action_latency_ms",
    "action_age_ms",
    "time_drift_ms",
    "state_drift_m",
    "episode_latency_ms",
    "final_ne_m",
    "control_steps",
)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return records


def seq_name(record):
    values = record.get("seq_names") or []
    return str(values[0]) if values else str(record.get("episode_idx", "unknown"))


def means(records):
    output = {}
    for metric in METRICS:
        values = [float(item[metric]) for item in records if item.get(metric) is not None]
        if values:
            output[metric] = float(np.mean(values))
    return output


def episode_outcomes(records):
    output = {}
    for item in records:
        if item.get("record_type") != "episode_end":
            continue
        output[seq_name(item)] = (
            bool(item.get("success")),
            bool(item.get("oracle_success")),
            bool(item.get("collision")),
        )
    return output


def apply_steps(records):
    output = {}
    for item in records:
        request_id = item.get("request_id")
        applied_step = item.get("result_applied_exec_step")
        if request_id is None or applied_step is None:
            continue
        output[(seq_name(item), int(request_id))] = int(applied_step)
    return output


def relative_error(normal, fast):
    if normal == 0:
        return 0.0 if fast == 0 else float("inf")
    return abs(fast - normal) / abs(normal)


def main():
    parser = argparse.ArgumentParser(description="Compare normal and Fast Eval JSONL profiles")
    parser.add_argument("--normal_log", required=True)
    parser.add_argument("--fast_log", required=True)
    parser.add_argument("--metric_tolerance", type=float, default=0.05)
    parser.add_argument("--max_outcome_mismatches", type=int, default=1)
    parser.add_argument("--min_apply_step_match", type=float, default=0.95)
    parser.add_argument("--min_wall_speedup", type=float, default=2.0)
    parser.add_argument("--output_json")
    args = parser.parse_args()

    normal_records = load_jsonl(args.normal_log)
    fast_records = load_jsonl(args.fast_log)
    normal_means = means(normal_records)
    fast_means = means(fast_records)
    metric_comparison = {}
    metric_failures = []
    for metric in sorted(set(normal_means) & set(fast_means)):
        error = relative_error(normal_means[metric], fast_means[metric])
        metric_comparison[metric] = {
            "normal": normal_means[metric],
            "fast": fast_means[metric],
            "relative_error": error,
        }
        if error > args.metric_tolerance:
            metric_failures.append(metric)

    normal_outcomes = episode_outcomes(normal_records)
    fast_outcomes = episode_outcomes(fast_records)
    common_episodes = sorted(set(normal_outcomes) & set(fast_outcomes))
    outcome_mismatches = [
        episode for episode in common_episodes if normal_outcomes[episode] != fast_outcomes[episode]
    ]

    normal_steps = apply_steps(normal_records)
    fast_steps = apply_steps(fast_records)
    common_requests = sorted(set(normal_steps) & set(fast_steps))
    step_matches = sum(normal_steps[key] == fast_steps[key] for key in common_requests)
    apply_step_match_ratio = step_matches / len(common_requests) if common_requests else None

    normal_wall = [
        float(item["wall_elapsed_ms"])
        for item in normal_records
        if item.get("record_type") == "episode_end" and item.get("wall_elapsed_ms") is not None
    ]
    fast_wall = [
        float(item["wall_elapsed_ms"])
        for item in fast_records
        if item.get("record_type") == "episode_end" and item.get("wall_elapsed_ms") is not None
    ]
    wall_speedup = None
    if normal_wall and fast_wall and np.mean(fast_wall) > 0:
        wall_speedup = float(np.mean(normal_wall) / np.mean(fast_wall))

    failures = []
    if metric_failures:
        failures.append(f"metric tolerance exceeded: {metric_failures}")
    if len(outcome_mismatches) > args.max_outcome_mismatches:
        failures.append(f"outcome mismatches={len(outcome_mismatches)}")
    if apply_step_match_ratio is not None and apply_step_match_ratio < args.min_apply_step_match:
        failures.append(f"apply-step match={apply_step_match_ratio:.3f}")
    if wall_speedup is not None and wall_speedup < args.min_wall_speedup:
        failures.append(f"wall speedup={wall_speedup:.3f}")

    report = {
        "passed": not failures,
        "failures": failures,
        "normal_records": len(normal_records),
        "fast_records": len(fast_records),
        "common_episodes": len(common_episodes),
        "outcome_mismatch_count": len(outcome_mismatches),
        "outcome_mismatches": outcome_mismatches[:20],
        "common_requests": len(common_requests),
        "apply_step_match_ratio": apply_step_match_ratio,
        "wall_speedup": wall_speedup,
        "metrics": metric_comparison,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json:
        Path(args.output_json).write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
