#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GT_PREFIXES = ("success_", "oracle_")


@dataclass(frozen=True)
class EpisodeRecord:
    seq_name: str
    episode_dir: Path
    status: str
    dataset_dir: Path
    target: List[float]
    gt_path: List[List[float]]
    actual_path: List[List[float]]
    final_ne_m: float


@dataclass(frozen=True)
class PairRecord:
    seq_name: str
    stopgo: EpisodeRecord
    continuous: EpisodeRecord
    ne_gap_m: float
    profile_jsonl: Path


def continuous_label(pair: PairRecord) -> str:
    name = pair.continuous.episode_dir.parent.name
    if name.startswith("eval_pro_con_"):
        return "continuous_" + name[len("eval_pro_con_") :]
    return "continuous"


def strip_prefix(name: str) -> str:
    for prefix in GT_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def episode_status(name: str) -> str:
    if name.startswith("success_"):
        return "success"
    if name.startswith("oracle_"):
        return "oracle"
    return "failure"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_positions_from_log_dir(log_dir: Path) -> List[List[float]]:
    positions: List[List[float]] = []
    for path in sorted(log_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        positions.append(payload["sensors"]["state"]["position"][0:3])
    return positions


def point_dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def extract_action_delay_ms(payload: dict) -> Optional[float]:
    value = payload.get("action_delay_ms")
    if value is None:
        value = payload.get("action_age_ms")
    if value is None:
        return None
    return float(value)


def find_episode_dirs(root: Path) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if not (child / "ori_info.json").exists():
            continue
        if not (child / "log").exists():
            continue
        result[strip_prefix(child.name)] = child
    return result


def load_episode_record(episode_dir: Path) -> EpisodeRecord:
    seq_name = strip_prefix(episode_dir.name)
    status = episode_status(episode_dir.name)
    ori_info = load_json(episode_dir / "ori_info.json")
    dataset_dir = Path(ori_info["ori_traj_dir"])
    actual_path = load_positions_from_log_dir(episode_dir / "log")
    gt_path = load_positions_from_log_dir(dataset_dir / "log")

    mark_path = dataset_dir / "mark.json"
    target = None
    if mark_path.exists():
        mark = load_json(mark_path)
        target = mark.get("target", {}).get("position")
    if target is None:
        target = gt_path[-1] if gt_path else actual_path[-1]

    final_actual = actual_path[-1] if actual_path else target
    final_gt = gt_path[-1] if gt_path else target
    final_ne_m = point_dist(final_actual, final_gt)

    return EpisodeRecord(
        seq_name=seq_name,
        episode_dir=episode_dir,
        status=status,
        dataset_dir=dataset_dir,
        target=target,
        gt_path=gt_path,
        actual_path=actual_path,
        final_ne_m=final_ne_m,
    )


def build_profile_index(profile_root: Path, only_file: Optional[Path] = None) -> Dict[str, Path]:
    if only_file is not None:
        candidates = [only_file]
    else:
        candidates = sorted(profile_root.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no jsonl found in {profile_root}")

    best: Dict[str, Tuple[Path, int, float]] = {}
    for candidate in candidates:
        counts: Dict[str, int] = {}
        with candidate.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("record_type") != "decision":
                    continue
                if extract_action_delay_ms(payload) is None or payload.get("state_drift_m") is None:
                    continue
                seq_name = payload.get("seq_names", [None])[0]
                if seq_name is None:
                    continue
                counts[seq_name] = counts.get(seq_name, 0) + 1
        candidate_mtime = candidate.stat().st_mtime
        for seq_name, count in counts.items():
            prev = best.get(seq_name)
            if prev is None or count > prev[1] or (count == prev[1] and candidate_mtime > prev[2]):
                best[seq_name] = (candidate, count, candidate_mtime)
    return {seq_name: item[0] for seq_name, item in best.items()}


def load_pairs(
    stopgo_root: Path,
    continuous_root: Path,
    profile_index: Dict[str, Path],
    stopgo_success_only: bool = False,
) -> List[PairRecord]:
    stopgo_dirs = find_episode_dirs(stopgo_root)
    continuous_dirs = find_episode_dirs(continuous_root)

    pairs: List[PairRecord] = []
    for seq_name, stopgo_dir in stopgo_dirs.items():
        allowed_stopgo_status = {"success"} if stopgo_success_only else {"success", "oracle"}
        if episode_status(stopgo_dir.name) not in allowed_stopgo_status:
            continue
        continuous_dir = continuous_dirs.get(seq_name)
        if continuous_dir is None:
            continue
        if episode_status(continuous_dir.name) != "failure":
            continue
        profile_jsonl = profile_index.get(seq_name)
        if profile_jsonl is None:
            continue

        stopgo = load_episode_record(stopgo_dir)
        continuous = load_episode_record(continuous_dir)
        pairs.append(
            PairRecord(
                seq_name=seq_name,
                stopgo=stopgo,
                continuous=continuous,
                ne_gap_m=continuous.final_ne_m - stopgo.final_ne_m,
                profile_jsonl=profile_jsonl,
            )
        )
    pairs.sort(key=lambda item: item.ne_gap_m, reverse=True)
    return pairs


def _scatter_point(ax, point, label, color, marker, size=80, edgecolor="black"):
    if point is None:
        return
    pt = np.asarray(point, dtype=np.float64)
    ax.scatter(
        [pt[0]],
        [pt[1]],
        [pt[2]],
        color=color,
        marker=marker,
        s=size,
        edgecolors=edgecolor,
        linewidths=0.8,
        label=label,
        zorder=5,
    )


def plot_trajectory_compare_3d(output_path: Path, pair: PairRecord) -> None:
    fig = plt.figure(figsize=(15.5, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15])
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    zoom_ax = fig.add_subplot(gs[0, 1], projection="3d")

    gt = np.asarray(pair.stopgo.gt_path, dtype=np.float64)
    stopgo = np.asarray(pair.stopgo.actual_path, dtype=np.float64)
    continuous = np.asarray(pair.continuous.actual_path, dtype=np.float64)
    cont_label = continuous_label(pair)

    def draw_paths(axis, with_endpoint_labels: bool):
        if gt.size:
            axis.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="tab:purple", linewidth=1.8, alpha=0.65, label="gt_path")
        if stopgo.size:
            axis.plot(stopgo[:, 0], stopgo[:, 1], stopgo[:, 2], color="tab:orange", linewidth=2.0, label="stop_and_go")
            if with_endpoint_labels:
                _scatter_point(axis, stopgo[-1], "stop_and_go_end", "tab:orange", "X", size=100)
        if continuous.size:
            axis.plot(continuous[:, 0], continuous[:, 1], continuous[:, 2], color="tab:blue", linewidth=2.2, label=cont_label)
            if with_endpoint_labels:
                _scatter_point(axis, continuous[-1], "continuous_end", "tab:blue", "X", size=100)
        if gt.size:
            _scatter_point(axis, gt[0], "start", "tab:green", "o", size=90)
        if with_endpoint_labels:
            _scatter_point(axis, pair.stopgo.target, "target", "tab:red", "^", size=100)

    draw_paths(ax, with_endpoint_labels=True)

    local_n = min(len(stopgo), max(len(continuous), 1))
    stopgo_local = stopgo[:local_n] if stopgo.size else stopgo
    if stopgo_local.size:
        zoom_ax.plot(
            stopgo_local[:, 0],
            stopgo_local[:, 1],
            stopgo_local[:, 2],
            color="tab:orange",
            linewidth=2.4,
            label=f"stop_and_go first {local_n}",
        )
        _scatter_point(zoom_ax, stopgo_local[-1], f"stop_and_go pt {local_n - 1}", "tab:orange", "X", size=95)
    if continuous.size:
        zoom_ax.plot(
            continuous[:, 0],
            continuous[:, 1],
            continuous[:, 2],
            color="tab:blue",
            linewidth=2.8,
            label=f"{cont_label} all {len(continuous)}",
        )
        _scatter_point(zoom_ax, continuous[-1], "continuous_end", "tab:blue", "X", size=95)
    if gt.size:
        _scatter_point(zoom_ax, gt[0], "start", "tab:green", "o", size=90)

    def set_axis_box(axis, points, pad_ratio=0.12, min_span=8.0):
        valid = [pts for pts in points if pts.size]
        if not valid:
            return
        merged = np.vstack(valid)
        mins = merged.min(axis=0)
        maxs = merged.max(axis=0)
        spans = np.maximum(maxs - mins, min_span)
        pads = spans * pad_ratio
        axis.set_xlim(mins[0] - pads[0], maxs[0] + pads[0])
        axis.set_ylim(mins[1] - pads[1], maxs[1] + pads[1])
        axis.set_zlim(mins[2] - pads[2], maxs[2] + pads[2])
        axis.set_box_aspect(spans)

    ax.set_title("Full trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=18, azim=-92)
    ax.legend(loc="best")
    zoom_ax.set_title("Zoom: same executed-point window")
    zoom_ax.set_xlabel("X")
    zoom_ax.set_ylabel("Y")
    zoom_ax.set_zlabel("Z")
    zoom_ax.view_init(elev=22, azim=-55)
    set_axis_box(zoom_ax, [continuous, stopgo_local], pad_ratio=0.1, min_span=6.0)
    zoom_ax.legend(loc="best")
    fig.suptitle(f"Trajectory Compare 3D: {pair.seq_name}", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def plot_distance_to_goal(output_path: Path, pair: PairRecord) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    stopgo = np.asarray(pair.stopgo.actual_path, dtype=np.float64)
    continuous = np.asarray(pair.continuous.actual_path, dtype=np.float64)
    goal = np.asarray(pair.stopgo.gt_path[-1], dtype=np.float64)
    cont_label = continuous_label(pair)

    if stopgo.size:
        stopgo_dist = np.linalg.norm(stopgo - goal[None, :], axis=1)
        stopgo_x = np.arange(len(stopgo_dist))
        ax.plot(stopgo_x, stopgo_dist, color="tab:orange", linewidth=2.0, label=f"stop_and_go (n={len(stopgo_dist)})")
        ax.scatter([stopgo_x[-1]], [stopgo_dist[-1]], color="tab:orange", marker="X", s=55, zorder=5)
    if continuous.size:
        cont_dist = np.linalg.norm(continuous - goal[None, :], axis=1)
        cont_x = np.arange(len(cont_dist))
        ax.plot(cont_x, cont_dist, color="tab:blue", linewidth=2.2, label=f"{cont_label} (n={len(cont_dist)})")
        ax.scatter([cont_x[-1]], [cont_dist[-1]], color="tab:blue", marker="X", s=55, zorder=5)

    ax.set_title(f"Distance to GT End by Executed Waypoint: {pair.seq_name}")
    ax.set_xlabel("executed waypoint index")
    ax.set_ylabel("distance to GT end (m)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def plot_temporal_mismatch(output_path: Path, pair: PairRecord, profile_jsonl: Path) -> Tuple[float, float]:
    decision_records = []
    with profile_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("seq_names", [None])[0] != pair.seq_name:
                continue
            if payload.get("record_type") != "decision":
                continue
            if extract_action_delay_ms(payload) is None or payload.get("state_drift_m") is None:
                continue
            decision_records.append(payload)

    decision_records.sort(key=lambda item: int(item.get("decision_step", 0)))
    steps = [int(item["decision_step"]) for item in decision_records]
    action_delay = [float(extract_action_delay_ms(item)) for item in decision_records]
    state_drift = [float(item["state_drift_m"]) for item in decision_records]

    fig, ax1 = plt.subplots(figsize=(10.5, 6.5))
    ax2 = ax1.twinx()
    ax1.plot(steps, state_drift, color="tab:blue", linewidth=2.0, marker="o", markersize=3.5, label="state_drift_m")
    ax2.plot(steps, action_delay, color="tab:red", linewidth=1.8, marker="s", markersize=3.0, alpha=0.9, label="action_age_ms")

    ax1.set_title(f"Temporal Mismatch: {pair.seq_name}")
    ax1.set_xlabel("decision step")
    ax1.set_ylabel("state drift (m)", color="tab:blue")
    ax2.set_ylabel("action age (ms)", color="tab:red")
    ax1.grid(True, alpha=0.25)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)

    return (float(np.mean(action_delay)) if action_delay else 0.0, float(np.mean(state_drift)) if state_drift else 0.0)


def write_summary(
    output_dir: Path,
    pair: PairRecord,
    mean_action_delay_ms: float,
    mean_state_drift_m: float,
) -> None:
    lines = [
        f"seq_name: {pair.seq_name}",
        f"stopgo_dir: {pair.stopgo.episode_dir}",
        f"continuous_dir: {pair.continuous.episode_dir}",
        f"stopgo_status: {pair.stopgo.status}",
        f"continuous_status: {pair.continuous.status}",
        f"profile_jsonl: {pair.profile_jsonl}",
        f"stopgo_final_ne_m: {pair.stopgo.final_ne_m:.3f}",
        f"continuous_final_ne_m: {pair.continuous.final_ne_m:.3f}",
        f"ne_gap_m: {pair.ne_gap_m:.3f}",
        f"mean_action_age_ms: {mean_action_delay_ms:.3f}",
        f"mean_state_drift_m: {mean_state_drift_m:.3f}",
        f"stopgo_points: {len(pair.stopgo.actual_path)}",
        f"continuous_points: {len(pair.continuous.actual_path)}",
        f"target: {pair.stopgo.target}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_pair(output_root: Path, pair: PairRecord, profile_jsonl: Path) -> None:
    output_dir = output_root / pair.seq_name
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_trajectory_compare_3d(output_dir / "trajectory_compare_3d.png", pair)
    plot_distance_to_goal(output_dir / "distance_to_goal.png", pair)
    mean_action_delay_ms, mean_state_drift_m = plot_temporal_mismatch(output_dir / "temporal_mismatch.png", pair, profile_jsonl)
    write_summary(output_dir, pair, mean_action_delay_ms, mean_state_drift_m)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze continuous-vs-stopgo pairs and visualize drift.")
    parser.add_argument("--stopgo_root", default="/HDD1/code/TravelUAV/eval_output")
    parser.add_argument("--continuous_root", default="/HDD1/code/TravelUAV/eval_pro_con_w3")
    parser.add_argument("--profile_jsonl", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--stopgo_success_only", action="store_true")
    args = parser.parse_args()

    stopgo_root = Path(args.stopgo_root)
    continuous_root = Path(args.continuous_root)
    output_root = Path(args.output_root) if args.output_root else (continuous_root / "analysis_pairs")
    output_root.mkdir(parents=True, exist_ok=True)

    if args.profile_jsonl:
        fixed_profile = Path(args.profile_jsonl)
        if not fixed_profile.exists():
            raise FileNotFoundError(fixed_profile)
        profile_index = build_profile_index(fixed_profile.parent, only_file=fixed_profile)
    else:
        profile_index = build_profile_index(continuous_root / "profile_logs")

    pairs = load_pairs(stopgo_root, continuous_root, profile_index, stopgo_success_only=args.stopgo_success_only)
    if not pairs:
        raise SystemExit("no matching stop-and-go success/oracle vs continuous failure pairs found")

    selected = pairs if args.top_k <= 0 else pairs[: args.top_k]

    index_lines = [
        "seq_name,stopgo_status,continuous_status,stopgo_final_ne_m,continuous_final_ne_m,ne_gap_m,profile_jsonl,output_dir",
    ]
    for pair in selected:
        analyze_pair(output_root, pair, pair.profile_jsonl)
        index_lines.append(
            ",".join(
                [
                    pair.seq_name,
                    pair.stopgo.status,
                    pair.continuous.status,
                    f"{pair.stopgo.final_ne_m:.3f}",
                    f"{pair.continuous.final_ne_m:.3f}",
                    f"{pair.ne_gap_m:.3f}",
                    str(pair.profile_jsonl),
                    str(output_root / pair.seq_name),
                ]
            )
        )

    (output_root / "selected_pairs.csv").write_text("\n".join(index_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
