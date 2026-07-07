#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SR = "sr"
OSR = "osr"
FAILURE = "failure"


def parse_args():
    parser = argparse.ArgumentParser(description="Build a DRL scheduler curriculum dataset from paired eval outputs.")
    parser.add_argument("--source_json", required=True, help="Frame-level TravelUAV source json.")
    parser.add_argument("--stop_eval_dir", required=True, help="Stop-and-go eval output directory.")
    parser.add_argument("--continuous_eval_dir", required=True, help="Continuous eval output directory.")
    parser.add_argument("--output_json", required=True, help="Filtered frame-level output json.")
    parser.add_argument("--manifest_json", required=True, help="Episode-level manifest output json.")
    return parser.parse_args()


def normalize_episode_dir_name(name: str) -> Tuple[str, str]:
    if name.startswith("success_"):
        return name[len("success_") :], SR
    if name.startswith("oracle_"):
        return name[len("oracle_") :], OSR
    return name, FAILURE


def load_eval_status(eval_dir: Path) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    for child in sorted(eval_dir.iterdir()):
        if not child.is_dir() or child.name == "profile_logs":
            continue
        episode_id, status = normalize_episode_dir_name(child.name)
        if episode_id in statuses:
            raise ValueError(f"Duplicate episode id {episode_id} in {eval_dir}")
        statuses[episode_id] = status
    return statuses


def source_episode_id(item: dict) -> str:
    json_path = item.get("json")
    if not json_path:
        raise ValueError(f"Source item has no json field: {item}")
    parts = Path(json_path).parts
    if len(parts) < 2:
        raise ValueError(f"Unexpected source json path: {json_path}")
    return parts[-2]


def classify(stop_status: str, continuous_status: str) -> str:
    stop_ok = stop_status in (SR, OSR)
    continuous_ok = continuous_status in (SR, OSR)
    if stop_ok and not continuous_ok:
        return "recoverable"
    if stop_ok and continuous_ok:
        return "easy"
    if not stop_ok and continuous_ok:
        return "continuous_only"
    return "unsolved"


def selected_category(category: str) -> bool:
    return category in ("recoverable", "easy")


def load_source_items(source_json: Path) -> List[dict]:
    with source_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a frame-level list in {source_json}")
    return data


def unique_episode_ids(items: Iterable[dict]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        episode_id = source_episode_id(item)
        if episode_id not in seen:
            seen.add(episode_id)
            ordered.append(episode_id)
    return ordered


def main() -> None:
    args = parse_args()
    source_json = Path(args.source_json)
    stop_eval_dir = Path(args.stop_eval_dir)
    continuous_eval_dir = Path(args.continuous_eval_dir)
    output_json = Path(args.output_json)
    manifest_json = Path(args.manifest_json)

    source_items = load_source_items(source_json)
    source_ids = unique_episode_ids(source_items)
    stop_statuses = load_eval_status(stop_eval_dir)
    continuous_statuses = load_eval_status(continuous_eval_dir)

    missing_stop = sorted(set(source_ids) - set(stop_statuses))
    missing_continuous = sorted(set(source_ids) - set(continuous_statuses))
    if missing_stop or missing_continuous:
        raise ValueError(
            "Eval outputs do not cover the source json episodes: "
            f"missing_stop={len(missing_stop)}, missing_continuous={len(missing_continuous)}"
        )

    episode_records = []
    selected_ids = set()
    category_counts = Counter()
    status_pair_counts = Counter()
    for episode_id in source_ids:
        stop_status = stop_statuses[episode_id]
        continuous_status = continuous_statuses[episode_id]
        category = classify(stop_status, continuous_status)
        selected = selected_category(category)
        category_counts[category] += 1
        status_pair_counts[f"{stop_status}->{continuous_status}"] += 1
        if selected:
            selected_ids.add(episode_id)
        episode_records.append(
            {
                "episode_id": episode_id,
                "stop_status": stop_status,
                "continuous_status": continuous_status,
                "category": category,
                "selected": selected,
            }
        )

    selected_items = [item for item in source_items if source_episode_id(item) in selected_ids]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(selected_items, handle, ensure_ascii=False, indent=2)

    manifest = {
        "source_json": str(source_json),
        "stop_eval_dir": str(stop_eval_dir),
        "continuous_eval_dir": str(continuous_eval_dir),
        "output_json": str(output_json),
        "total_episodes": len(source_ids),
        "total_frames": len(source_items),
        "selected_episodes": len(selected_ids),
        "selected_frames": len(selected_items),
        "category_counts": dict(category_counts),
        "status_pair_counts": dict(status_pair_counts),
        "selected_categories": ["recoverable", "easy"],
        "episodes": episode_records,
    }
    with manifest_json.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(json.dumps({k: manifest[k] for k in (
        "total_episodes",
        "total_frames",
        "selected_episodes",
        "selected_frames",
        "category_counts",
        "status_pair_counts",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
