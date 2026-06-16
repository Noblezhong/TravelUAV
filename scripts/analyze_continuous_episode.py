#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_positions_from_log_dir(log_dir: Path):
    positions = []
    for path in sorted(log_dir.glob("*.json")):
        with path.open() as handle:
            payload = json.load(handle)
        positions.append(payload["sensors"]["state"]["position"][0:3])
    return positions


def load_profile_records(profile_jsonl: Path, seq_name: str):
    records = []
    with profile_jsonl.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("seq_names", [None])[0] == seq_name:
                records.append(payload)
    return records


def point_dist(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def plot_3d_lines(path: Path, series, title: str):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for item in series:
        pts = np.asarray(item["points"], dtype=np.float64)
        if pts.size == 0:
            continue
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], label=item["label"], linewidth=item.get("linewidth", 1.5), alpha=item.get("alpha", 1.0))
        if item.get("scatter", False):
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=item.get("size", 10), alpha=item.get("alpha", 1.0))
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if any(item.get("label") for item in series):
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


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


def plot_actual_path(path: Path, actual_path, title: str, target=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    pts = np.asarray(actual_path, dtype=np.float64)
    if pts.size:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="tab:blue", linewidth=2.2, label="actual_uav")
        _scatter_point(ax, pts[0], "start", "tab:green", "o", size=90)
        _scatter_point(ax, pts[-1], "end", "tab:red", "X", size=110)
    _scatter_point(ax, target, "target", "tab:orange", "^", size=100)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_actual_vs_gt(path: Path, actual_path, gt_path, title: str, target=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    actual = np.asarray(actual_path, dtype=np.float64)
    gt = np.asarray(gt_path, dtype=np.float64)
    if actual.size:
        ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], color="tab:blue", linewidth=2.2, label="actual_uav")
        _scatter_point(ax, actual[0], "actual_start", "tab:green", "o", size=90)
        _scatter_point(ax, actual[-1], "actual_end", "tab:red", "X", size=110)
    if gt.size:
        ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="tab:purple", linewidth=1.8, alpha=0.7, label="gt_path")
        _scatter_point(ax, gt[0], "gt_start", "limegreen", "s", size=75)
        _scatter_point(ax, gt[-1], "gt_end", "magenta", "D", size=85)
    _scatter_point(ax, target, "target", "tab:orange", "^", size=100)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_continuous_stop_and_go_vs_gt(path: Path, continuous_path, stop_and_go_path, gt_path, title: str, target=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    continuous = np.asarray(continuous_path, dtype=np.float64)
    stop_and_go = np.asarray(stop_and_go_path, dtype=np.float64)
    gt = np.asarray(gt_path, dtype=np.float64)

    if gt.size:
        ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="tab:purple", linewidth=1.8, alpha=0.65, label="gt_path")
    if stop_and_go.size:
        ax.plot(
            stop_and_go[:, 0],
            stop_and_go[:, 1],
            stop_and_go[:, 2],
            color="tab:orange",
            linewidth=2.0,
            label="stop_and_go",
        )
    if continuous.size:
        ax.plot(
            continuous[:, 0],
            continuous[:, 1],
            continuous[:, 2],
            color="tab:blue",
            linewidth=2.2,
            label="continuous_w3",
        )

    if continuous.size:
        _scatter_point(ax, continuous[0], "start", "tab:green", "o", size=85)
        _scatter_point(ax, continuous[-1], "continuous_end", "tab:blue", "X", size=95)
    if stop_and_go.size:
        _scatter_point(ax, stop_and_go[-1], "stop_and_go_end", "tab:orange", "X", size=95)
    _scatter_point(ax, target, "target", "tab:red", "^", size=100)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_decision_waypoints(path: Path, decision_rows, title: str, actual_path=None, target=None, markers=True):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("viridis", max(len(decision_rows), 2))

    if actual_path:
        actual = np.asarray(actual_path, dtype=np.float64)
        if actual.size:
            ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], color="lightgray", linewidth=1.8, alpha=0.9, label="actual_uav")
            if markers:
                _scatter_point(ax, actual[0], "actual_start", "tab:green", "o", size=80)
                _scatter_point(ax, actual[-1], "actual_end", "tab:red", "X", size=95)

    for idx, row in enumerate(decision_rows):
        pts = np.asarray(row["waypoints_world"], dtype=np.float64)
        if pts.size == 0:
            continue
        color = cmap(idx)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=1.3, alpha=0.95)
        if markers:
            ax.scatter(pts[0:1, 0], pts[0:1, 1], pts[0:1, 2], color=color, marker="o", s=22)
            ax.scatter(pts[-1:, 0], pts[-1:, 1], pts[-1:, 2], color=color, marker=".", s=18)

    if markers and decision_rows:
        first = np.asarray(decision_rows[0]["waypoints_world"][0], dtype=np.float64)
        last = np.asarray(decision_rows[-1]["waypoints_world"][-1], dtype=np.float64)
        _scatter_point(ax, first, "first_plan_start", "cyan", "o", size=85)
        _scatter_point(ax, last, "last_plan_end", "navy", "P", size=95)
    if markers:
        _scatter_point(ax, target, "target", "tab:orange", "^", size=100)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_decision_waypoints_md(path: Path, decisions):
    lines = [
        "# Decision waypoint sequences",
        "",
        f"source_jsonl: `{decisions['source_jsonl']}`",
        "",
        f"episode: `{decisions['seq_name']}`",
        "",
    ]
    for row in decisions["rows"]:
        lines.append(f"## decision_step={row['decision_step']}")
        lines.append("")
        lines.append(f"uav_world={row['uav_world']}")
        lines.append("")
        lines.append(
            f"uav_to_first={row['uav_to_first_m']:.3f}m, "
            f"uav_to_nearest={row['uav_to_nearest_m']:.3f}m(idx={row['nearest_idx']})"
        )
        lines.append("")
        for idx, point in enumerate(row["waypoints_world"]):
            lines.append(f"- wp{idx}: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_jsonl", required=True)
    parser.add_argument("--episode_dir", required=True)
    parser.add_argument("--dataset_episode_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stop_and_go_episode_dir")
    args = parser.parse_args()

    profile_jsonl = Path(args.profile_jsonl)
    episode_dir = Path(args.episode_dir)
    dataset_episode_dir = Path(args.dataset_episode_dir)
    output_dir = Path(args.output_dir)
    stop_and_go_episode_dir = Path(args.stop_and_go_episode_dir) if args.stop_and_go_episode_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_name = episode_dir.name
    for prefix in ("success_", "oracle_"):
        if seq_name.startswith(prefix):
            seq_name = seq_name[len(prefix) :]
            break

    records = load_profile_records(profile_jsonl, seq_name)
    decision_records = [r for r in records if r.get("record_type") == "decision"]
    execution_records = [r for r in records if r.get("record_type") == "execution"]
    decision_stop_records = [r for r in records if r.get("record_type") == "decision_stop"]

    actual_path = load_positions_from_log_dir(episode_dir / "log")
    stop_and_go_path = load_positions_from_log_dir(stop_and_go_episode_dir / "log") if stop_and_go_episode_dir else []
    gt_path = load_positions_from_log_dir(dataset_episode_dir / "log")
    with (dataset_episode_dir / "mark.json").open() as handle:
        mark = json.load(handle)
    target = mark["target"]["position"]

    decision_rows = []
    continuity_rows = []
    prev_last = None
    exec_by_step = {int(r["exec_step"]): r for r in execution_records if "exec_step" in r}
    for rec in decision_records:
        step = int(rec["decision_step"])
        wps = rec["refined_waypoints"]
        uav_world = rec["current_pose"]
        dists = [point_dist(uav_world, wp) for wp in wps]
        nearest_idx = int(np.argmin(dists))
        row = {
            "decision_step": step,
            "uav_world": uav_world,
            "waypoints_world": wps,
            "uav_to_first_m": dists[0],
            "uav_to_nearest_m": dists[nearest_idx],
            "nearest_idx": nearest_idx,
        }
        decision_rows.append(row)
        continuity_rows.append(
            {
                "decision_step": step,
                "uav_to_first_m": dists[0],
                "uav_to_nearest_m": dists[nearest_idx],
                "nearest_idx": nearest_idx,
                "prev_last_to_this_first_m": "" if prev_last is None else point_dist(prev_last, wps[0]),
                "exec_dist_same_step_m": exec_by_step.get(step, {}).get("exec_target_distance_m", ""),
                "state_drift_same_step_m": rec.get("state_drift_m", ""),
            }
        )
        prev_last = wps[-1]

    write_decision_waypoints_md(
        output_dir / "decision_waypoints.md",
        {
            "source_jsonl": str(profile_jsonl),
            "seq_name": seq_name,
            "rows": decision_rows,
        },
    )

    with (output_dir / "decision_waypoints.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["decision_step", "uav_world", "waypoints_world", "uav_to_first_m", "uav_to_nearest_m", "nearest_idx"])
        for row in decision_rows:
            writer.writerow(
                [
                    row["decision_step"],
                    json.dumps(row["uav_world"]),
                    json.dumps(row["waypoints_world"]),
                    row["uav_to_first_m"],
                    row["uav_to_nearest_m"],
                    row["nearest_idx"],
                ]
            )

    with (output_dir / "continuity_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "decision_step",
                "uav_to_first_m",
                "uav_to_nearest_m",
                "nearest_idx",
                "prev_last_to_this_first_m",
                "exec_dist_same_step_m",
                "state_drift_same_step_m",
            ],
        )
        writer.writeheader()
        writer.writerows(continuity_rows)

    status_lines = [
        f"seq={seq_name}",
        f"success={episode_dir.name.startswith('success_')}",
        f"termination={'decision_stop' if decision_stop_records else 'saved_without_decision_stop'}",
        f"actual_points={len(actual_path)}",
        f"gt_points={len(gt_path)}",
        f"final_pose={actual_path[-1] if actual_path else None}",
        f"target={target}",
        f"gt_end={gt_path[-1] if gt_path else None}",
        f"stop_and_go_dir={stop_and_go_episode_dir}" if stop_and_go_episode_dir else "stop_and_go_dir=None",
        f"stop_and_go_success={stop_and_go_episode_dir.name.startswith('success_')}" if stop_and_go_episode_dir else "stop_and_go_success=None",
        f"stop_and_go_points={len(stop_and_go_path)}",
        f"final_to_target_m={point_dist(actual_path[-1], target):.3f}" if actual_path else "final_to_target_m=None",
        f"final_to_gt_end_m={point_dist(actual_path[-1], gt_path[-1]):.3f}" if actual_path and gt_path else "final_to_gt_end_m=None",
        f"stop_and_go_final_to_target_m={point_dist(stop_and_go_path[-1], target):.3f}" if stop_and_go_path else "stop_and_go_final_to_target_m=None",
        f"decision_count={len(decision_records)}",
        f"execution_count={len(execution_records)}",
    ]
    (output_dir / "episode_status.txt").write_text("\n".join(status_lines) + "\n", encoding="utf-8")

    plot_actual_path(output_dir / "actual_uav_path_3d.png", actual_path, f"Actual UAV Path: {seq_name}", target=target)
    plot_actual_vs_gt(output_dir / "actual_vs_gt_path_3d.png", actual_path, gt_path, f"Actual vs GT Path: {seq_name}", target=target)
    if stop_and_go_path:
        plot_continuous_stop_and_go_vs_gt(
            output_dir / "continuous_vs_stop_and_go_vs_gt_3d.png",
            actual_path,
            stop_and_go_path,
            gt_path,
            f"Continuous W3 vs Stop-and-Go vs GT: {seq_name}",
            target=target,
        )

    decision_series = []
    for row in decision_rows:
        decision_series.append(
            {
                "label": f"d{row['decision_step']}",
                "points": row["waypoints_world"],
                "linewidth": 1.0,
                "alpha": 0.75,
            }
        )
    plot_decision_waypoints(
        output_dir / "decision_waypoints_3d.png",
        decision_rows,
        f"Decision Waypoints: {seq_name}",
        actual_path=None,
        target=target,
        markers=True,
    )
    plot_decision_waypoints(
        output_dir / "combined_3d.png",
        decision_rows,
        f"Actual Path + Decision Waypoints: {seq_name}",
        actual_path=actual_path,
        target=None,
        markers=False,
    )


if __name__ == "__main__":
    main()
