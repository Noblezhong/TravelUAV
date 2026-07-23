#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def episode_id_from_source(item):
    path = item.get("json")
    if not path:
        raise ValueError("source item has no json field")
    return Path(path).parts[-2]


def load_statuses(root):
    statuses = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name == "profile_logs":
            continue
        if child.name.startswith("success_"):
            seq, status = child.name[len("success_") :], "sr"
        elif child.name.startswith("oracle_"):
            seq, status = child.name[len("oracle_") :], "osr"
        else:
            seq, status = child.name, "failure"
        statuses[seq] = status
    return statuses


def sample_without_overlap(rng, population, count, selected):
    candidates = sorted(set(population) - selected)
    if len(candidates) < count:
        raise ValueError(f"requested {count} episodes but only {len(candidates)} are available")
    chosen = set(rng.sample(candidates, count))
    selected.update(chosen)
    return chosen


def main():
    parser = argparse.ArgumentParser(description="Build a paired TrajCorr evaluation subset")
    parser.add_argument("--source_json", type=Path, required=True)
    parser.add_argument("--stop_eval_dir", type=Path, required=True)
    parser.add_argument("--continuous_eval_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--manifest_json", type=Path, required=True)
    parser.add_argument("--recoverable", type=int, default=60)
    parser.add_argument("--continuous_success", type=int, default=20)
    parser.add_argument("--random_failure", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source_items = json.loads(args.source_json.read_text(encoding="utf-8"))
    source_ids = []
    seen = set()
    for item in source_items:
        seq = episode_id_from_source(item)
        if seq not in seen:
            seen.add(seq)
            source_ids.append(seq)

    stop = load_statuses(args.stop_eval_dir)
    continuous = load_statuses(args.continuous_eval_dir)
    missing = [seq for seq in source_ids if seq not in stop or seq not in continuous]
    if missing:
        raise ValueError(f"evaluation directories are missing {len(missing)} source episodes")

    successful = {"sr", "osr"}
    recoverable_pool = [
        seq
        for seq in source_ids
        if stop[seq] in successful and continuous[seq] == "failure"
    ]
    continuous_success_pool = [
        seq for seq in source_ids if continuous[seq] in successful
    ]
    random_failure_pool = [
        seq
        for seq in source_ids
        if stop[seq] == "failure" and continuous[seq] == "failure"
    ]

    rng = random.Random(args.seed)
    selected = set()
    groups = {
        "recoverable": sample_without_overlap(
            rng, recoverable_pool, args.recoverable, selected
        ),
        "continuous_success": sample_without_overlap(
            rng, continuous_success_pool, args.continuous_success, selected
        ),
        "random_failure": sample_without_overlap(
            rng, random_failure_pool, args.random_failure, selected
        ),
    }
    output_items = [
        item for item in source_items if episode_id_from_source(item) in selected
    ]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output_items, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "source_json": str(args.source_json),
        "seed": args.seed,
        "episode_count": len(selected),
        "frame_count": len(output_items),
        "groups": {name: sorted(values) for name, values in groups.items()},
    }
    args.manifest_json.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
