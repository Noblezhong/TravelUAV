#!/usr/bin/env python3
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def _seq_name(record):
    values = record.get("seq_names") or []
    return str(values[0]) if values else None


def _strip_result_prefix(name):
    for prefix in ("success_", "oracle_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _physical_collision(log_paths):
    for path in log_paths:
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        collision = (
            record.get("sensors", {})
            .get("state", {})
            .get("collision", {})
            .get("has_collided", False)
        )
        if collision:
            return True
    return False


def load_result_outcomes(root):
    outcomes = {}
    for episode_dir in root.iterdir():
        if not episode_dir.is_dir() or episode_dir.name == "profile_logs":
            continue
        seq = _strip_result_prefix(episode_dir.name)
        log_paths = sorted((episode_dir / "log").glob("*.json"))
        outcomes[seq] = {
            "success": episode_dir.name.startswith("success_"),
            "oracle_success": episode_dir.name.startswith(("success_", "oracle_")),
            "collision": _physical_collision(log_paths),
            "waypoints": max(0, len(log_paths) - 1),
        }
    return outcomes


def load_profile(root):
    trajectory_results = {}
    executions = {}
    episode_ends = {}
    mode_counts = Counter()
    for path in sorted((root / "profile_logs").glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                seq = _seq_name(record)
                if seq is None:
                    continue
                record_type = record.get("record_type")
                if record_type == "trajectory_result":
                    key = (seq, int(record["request_id"]))
                    trajectory_results[key] = record
                elif record_type == "execution":
                    key = (seq, int(record["exec_step"]))
                    executions[key] = record
                elif record_type == "episode_end":
                    episode_ends[seq] = record
    for record in trajectory_results.values():
        mode_counts[str(record.get("trajectory_mode"))] += 1
    return {
        "trajectory_results": trajectory_results,
        "executions": executions,
        "episode_ends": episode_ends,
        "mode_counts": dict(mode_counts),
    }


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def local_ne_progress(profile, mode, seqs=None):
    seqs = None if seqs is None else set(seqs)
    executions_by_request = {}
    for (seq, _), record in profile["executions"].items():
        if seqs is not None and seq not in seqs:
            continue
        key = (seq, int(record["request_id"]))
        executions_by_request.setdefault(key, []).append(record)
    values = []
    for key, decision in profile["trajectory_results"].items():
        if seqs is not None and key[0] not in seqs:
            continue
        if decision.get("trajectory_mode") != mode:
            continue
        start_ne = decision.get("ne_at_apply_m")
        records = sorted(
            executions_by_request.get(key, []),
            key=lambda item: int(item.get("exec_step", 0)),
        )
        if start_ne is None or not records:
            continue
        end_ne = records[min(4, len(records) - 1)].get("navigation_error_m")
        if end_ne is not None:
            values.append(float(start_ne) - float(end_ne))
    return _mean(values)


def summarize(outcomes, profile, common=None):
    seqs = sorted(set(outcomes) if common is None else set(outcomes) & set(common))
    episode_ends = profile["episode_ends"]
    trajectory_results = [
        record
        for (seq, _), record in profile["trajectory_results"].items()
        if seq in seqs
    ]
    executions = [
        record
        for (seq, _), record in profile["executions"].items()
        if seq in seqs
    ]
    return {
        "episodes": len(seqs),
        "SR": _mean(outcomes[seq]["success"] for seq in seqs),
        "OSR": _mean(outcomes[seq]["oracle_success"] for seq in seqs),
        "CR": _mean(outcomes[seq]["collision"] for seq in seqs),
        "avg_waypoints": _mean(outcomes[seq]["waypoints"] for seq in seqs),
        "avg_NE_m": _mean(
            episode_ends[seq].get("final_ne_m")
            for seq in seqs
            if seq in episode_ends
        ),
        "avg_T_dec_ms": _mean(
            float(record.get("uplink_latency_ms", 0.0))
            + float(record.get("edge_llm_latency_ms", 0.0))
            for record in trajectory_results
        ),
        "avg_T_action_ms": _mean(
            record.get("airsim_action_latency_ms") for record in executions
        ),
        "avg_coarse_time_shift_ms": _mean(
            record.get("coarse_time_shift_ms") for record in trajectory_results
        ),
        "avg_coarse_state_shift_m": _mean(
            record.get("coarse_state_shift_m") for record in trajectory_results
        ),
        "avg_traj_time_shift_ms": _mean(
            record.get("traj_time_shift_ms") for record in trajectory_results
        ),
        "avg_traj_state_shift_m": _mean(
            record.get("traj_state_shift_m") for record in trajectory_results
        ),
        "avg_episode_latency_s": (
            _mean(
                episode_ends[seq].get("episode_latency_ms")
                for seq in seqs
                if seq in episode_ends
            )
            or 0.0
        )
        / 1000.0,
        "trajectory_mode_counts": profile["mode_counts"],
        "original_NE_progress_5_steps_m": local_ne_progress(
            profile, "original", seqs
        ),
        "corrected_NE_progress_5_steps_m": local_ne_progress(
            profile, "corrected", seqs
        ),
    }


def mcnemar_exact(off, on, common):
    off_only = sum(off[seq]["success"] and not on[seq]["success"] for seq in common)
    on_only = sum(on[seq]["success"] and not off[seq]["success"] for seq in common)
    discordant = off_only + on_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(off_only, on_only) + 1)
        )
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "off_success_on_failure": off_only,
        "off_failure_on_success": on_only,
        "discordant_pairs": discordant,
        "two_sided_p": p_value,
    }


def paired_bootstrap(off_values, on_values, samples, seed=0):
    common = sorted(set(off_values) & set(on_values))
    if not common:
        return {"count": 0, "mean_difference_on_minus_off": None, "ci95": [None, None]}
    differences = np.asarray(
        [float(on_values[seq]) - float(off_values[seq]) for seq in common],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(int(samples), len(differences)))
    sampled_means = differences[indices].mean(axis=1)
    return {
        "count": len(common),
        "mean_difference_on_minus_off": float(differences.mean()),
        "ci95": [
            float(np.percentile(sampled_means, 2.5)),
            float(np.percentile(sampled_means, 97.5)),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Compare paired TrajCorr OFF/ON runs")
    parser.add_argument("--off_dir", type=Path, required=True)
    parser.add_argument("--on_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--output_json", type=Path)
    args = parser.parse_args()

    off_outcomes = load_result_outcomes(args.off_dir)
    on_outcomes = load_result_outcomes(args.on_dir)
    common = sorted(set(off_outcomes) & set(on_outcomes))
    off_profile = load_profile(args.off_dir)
    on_profile = load_profile(args.on_dir)

    off_ne = {
        seq: record["final_ne_m"]
        for seq, record in off_profile["episode_ends"].items()
        if seq in common and record.get("final_ne_m") is not None
    }
    on_ne = {
        seq: record["final_ne_m"]
        for seq, record in on_profile["episode_ends"].items()
        if seq in common and record.get("final_ne_m") is not None
    }
    off_latency = {
        seq: record["episode_latency_ms"]
        for seq, record in off_profile["episode_ends"].items()
        if seq in common and record.get("episode_latency_ms") is not None
    }
    on_latency = {
        seq: record["episode_latency_ms"]
        for seq, record in on_profile["episode_ends"].items()
        if seq in common and record.get("episode_latency_ms") is not None
    }

    report = {
        "common_episodes": len(common),
        "off": summarize(off_outcomes, off_profile, common),
        "on": summarize(on_outcomes, on_profile, common),
        "SR_mcnemar": mcnemar_exact(off_outcomes, on_outcomes, common),
        "NE_paired_bootstrap": paired_bootstrap(
            off_ne,
            on_ne,
            args.bootstrap_samples,
        ),
        "episode_latency_paired_bootstrap_ms": paired_bootstrap(
            off_latency,
            on_latency,
            args.bootstrap_samples,
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json:
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
