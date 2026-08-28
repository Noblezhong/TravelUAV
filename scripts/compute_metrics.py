#!/usr/bin/env python3
"""Unified MATCH evaluation metrics from evaluator JSONL logs.

Single entry point for the main-table 9 metrics across all evaluation
paradigms (TC-ON / TC-OFF, PPO, Continuous, Stop-go, Rule-based), with a
single, verified aggregation protocol:

  SR / OSR / NE / control steps / logical E2E
      from episode_end records (identical for every paradigm)
  CR   AirSim judgment: episode_end.collision AND any frame has_collided
       (scans <eval_dir>/<episode>/log/*.json; the episode_end collision
       flag alone over-counts depth-sensor near-misses)
  time_shift / state_shift   paradigm-specific application-time sampling
  buffer_wait                paradigm-specific stalled-time accounting

Usage:
  python compute_metrics.py --input <eval.jsonl> [--eval-dir <dir>]
      [--output <out.json>] [--method auto|tc|ppo|continuous|stopgo|rule]
      [--wait-denominator all|waiting]

--eval-dir defaults to the JSONL's parent's parent (evaluations/<run>/).
--wait-denominator: 'all' (default) averages buffer_wait_s over every
episode — matches the main-table PPO / rule / continuous / stop-go rows;
'waiting' averages only over episodes that experienced any wait — matches
the two TC rows (TC-OFF 19.48, TC-ON 92.81).

Paradigm auto-detection (first 500 valid records):
  record_type trajectory_result  -> tc
  record_type scheduler_step     -> ppo
  record_type decision_stop      -> rule
  record_type execution          -> continuous
  otherwise                      -> stopgo
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

EPSILON_MS = 1e-6

# ---------------------------------------------------------------- utils


def _fmean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _number(record: dict, key: str, default=None) -> float | None:
    value = record.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _episode_id(record: dict) -> str | None:
    names = record.get("seq_names") or []
    return str(names[0]) if names else None


def _jsonl_records(path: Path):
    """Yield (line_number, record_or_None) for each non-empty line."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError:
                yield line_number, None


# ------------------------------------------------------------- detection


def detect_method(path: Path, invalid: list[int]) -> str:
    kinds: dict[str, int] = defaultdict(int)
    sample = 0
    for line_number, record in _jsonl_records(path):
        if record is None:
            invalid.append(line_number)
            continue
        sample += 1
        if sample > 500:
            break
        kind = record.get("record_type")
        if kind:
            kinds[kind] += 1
        if record.get("scheduler_action") is not None:
            kinds["<scheduler_action>"] += 1
    if kinds.get("trajectory_result"):
        return "tc"
    if kinds.get("scheduler_step"):
        return "ppo"
    if kinds.get("decision_stop") or kinds.get("<scheduler_action>"):
        return "rule"
    if kinds.get("execution"):
        return "continuous"
    return "stopgo"


# --------------------------------------------------- paradigm extractors
# Each returns dict[episode_id, record] of terminals plus per-episode
# application shift events and wait accumulation, in a uniform dict form.


def _collect_common(path: Path, invalid: list[int]):
    """One pass: terminals, plus a raw per-episode event bag.

    Returns (terminals, events) where events[ep] is a dict with
    'shifts' (list of (time_s, state_m) tuples) and 'wait_ms'.
    The wait_ms accumulation rule is paradigm-specific and supplied by the
    caller through 'wait_hooks'.
    """
    raise NotImplementedError


# We keep the per-paradigm passes explicit instead of over-abstracting:
# each evaluation line writes slightly different fields, and the protocol
# differences (which records sample the shift, what counts as waiting)
# must stay visible.

# ------------------------------------------------------------ TC line


def extract_tc(path: Path, invalid: list[int]) -> tuple[dict, dict]:
    terminals: dict[str, dict] = {}
    shifts: dict[str, list[tuple[float, float]]] = defaultdict(list)
    buffer_wait_ms: dict[str, float] = defaultdict(float)
    pending_waits: dict[str, list[float]] = {}
    records = 0

    for line_number, record in _jsonl_records(path):
        if record is None:
            invalid.append(line_number)
            continue
        records += 1
        ep = _episode_id(record)
        if ep is None:
            continue
        kind = record.get("record_type")
        if kind == "episode_end":
            if ep in terminals:
                raise ValueError(f"duplicate terminal episode {ep}")
            terminals[ep] = record
            pending_waits.pop(ep, None)
        elif kind == "trajectory_result":
            time_ms = _number(record, "time_shift_at_apply_ms")
            state_m = _number(record, "state_shift_at_apply_m")
            if time_ms is not None and state_m is not None:
                if time_ms < 0.0 or state_m < 0.0:
                    raise ValueError(f"negative shift in episode {ep}: {time_ms}, {state_m}")
                shifts[ep].append((time_ms / 1000.0, state_m))
            # Corrected definition (2026-08-07): pure stop-and-wait duration.
            # buffer_exhausted records mark the start of a wait (t0); the
            # matching trajectory_result's result_ready_logical_ms marks the
            # instant the awaited result became ready (t1). Wait time =
            # t1 - t0. The earlier definition accumulated the logical
            # interval between consecutive buffer_wait records, which also
            # contained the flight time between two waits and inflated the
            # metric ~26x.
            pending = pending_waits.get(ep)
            if pending:
                t0 = pending.pop(0)
                ready = _number(
                    record, "result_ready_logical_ms",
                    _number(record, "logical_elapsed_ms", 0.0),
                )
                if ready >= t0:
                    buffer_wait_ms[ep] += ready - t0
        elif kind == "buffer_wait":
            if record.get("buffer_exhausted"):
                now = _number(record, "logical_elapsed_ms", 0.0)
                pending_waits.setdefault(ep, []).append(now)

    events = {
        ep: {"shifts": shifts[ep], "wait_ms": buffer_wait_ms[ep]}
        for ep in terminals
    }
    return terminals, events, records


# --------------------------------------------------------- PPO line


def extract_ppo(path: Path, invalid: list[int]) -> tuple[dict, dict, int]:
    terminals: dict[str, dict] = {}
    waiting_ms: dict[str, float] = defaultdict(float)
    shifts: dict[str, list[tuple[float, float]]] = defaultdict(list)
    elapsed_ms: dict[str, float] = defaultdict(float)
    records = 0

    for line_number, record in _jsonl_records(path):
        if record is None:
            invalid.append(line_number)
            continue
        records += 1
        kind = record.get("record_type")
        if kind not in {"decision", "scheduler_step", "episode_end"}:
            continue
        ep = _episode_id(record)
        if ep is None:
            continue
        if kind == "episode_end":
            if ep in terminals:
                raise ValueError(f"duplicate terminal episode {ep}")
            terminals[ep] = record
            continue

        elapsed = _number(record, "elapsed_ms")
        if elapsed is None:
            continue
        action_ms = _number(record, "airsim_action_latency_ms", 0.0)
        if action_ms > elapsed + EPSILON_MS:
            raise ValueError(f"action exceeds elapsed time at line {line_number}")
        elapsed_ms[ep] += elapsed
        waiting_ms[ep] += max(0.0, elapsed - action_ms)

        if kind == "decision":
            time_ms = _number(record, "time_drift_ms")
            state_m = _number(record, "state_drift_m")
            if time_ms is not None and state_m is not None:
                shifts[ep].append((time_ms / 1000.0, state_m))
            continue

        # scheduler_step: sample shift at actual application points
        if record.get("trajectory_switch_applied_before_action"):
            time_ms = _number(record, "time_drift_before_ms")
            state_m = _number(record, "state_drift_before_m")
            if time_ms is not None and state_m is not None:
                shifts[ep].append((time_ms / 1000.0, state_m))
        applied_ms = record.get("result_applied_logical_ms")
        if isinstance(applied_ms, (int, float)) and math.isclose(
            float(applied_ms), float(record.get("logical_elapsed_ms", 0.0)), abs_tol=EPSILON_MS
        ):
            time_ms = _number(record, "time_drift_ms")
            state_m = _number(record, "state_drift_m")
            if time_ms is not None and state_m is not None:
                shifts[ep].append((time_ms / 1000.0, state_m))

    events = {ep: {"shifts": shifts[ep], "wait_ms": waiting_ms[ep]} for ep in terminals}
    return terminals, events, records


# ----------------------------------------------------- Continuous / Stop-go


def extract_action_line(path: Path, invalid: list[int], method: str, wait_mode: str) -> tuple[dict, dict, int]:
    """Continuous and Stop-go.

    Wait definitions (verified against the main table):
      e2e-action (stop-go, ppo): terminal E2E minus executed action time.
      approved (continuous): sum of non-moving local observation, non-moving
        DINO, and edge-result waiting components, per episode.
    """
    terminals: dict[str, dict] = {}
    shifts: dict[str, list[tuple[float, float]]] = defaultdict(list)
    action_ms: dict[str, float] = defaultdict(float)
    wait_parts: dict[str, list[float]] = defaultdict(list)  # approved components
    records = 0

    for line_number, record in _jsonl_records(path):
        if record is None:
            invalid.append(line_number)
            continue
        records += 1
        ep = _episode_id(record)
        if ep is None:
            continue
        kind = record.get("record_type")

        if kind == "episode_end":
            if ep in terminals:
                raise ValueError(f"duplicate terminal episode {ep}")
            terminals[ep] = record
            continue

        if method == "continuous":
            if kind == "decision":
                time_ms = _number(record, "action_age_ms")
                state_m = _number(record, "state_drift_m")
                if time_ms is not None and state_m is not None:
                    shifts[ep].append((time_ms / 1000.0, state_m))
            elif kind == "execution":
                action_ms[ep] += _number(record, "action_sim_time_ms", 0.0)
                if wait_mode == "approved":
                    wait_parts[ep].append(_number(record, "execution_post_obs_latency_ms", 0.0))
                    wait_parts[ep].append(_number(record, "execution_post_dino_latency_ms", 0.0))
            if wait_mode == "approved":
                # hover_wait_ms appears on decision and execution records alike
                wait_parts[ep].append(_number(record, "hover_wait_ms", 0.0))
        else:  # stopgo: every non-terminal record is an executed step
            time_ms = _number(record, "action_age_ms")
            state_m = _number(record, "state_drift_m")
            if time_ms is not None and state_m is not None:
                shifts[ep].append((time_ms / 1000.0, state_m))
            action_ms[ep] += _number(record, "action_sim_time_ms", 0.0)

    events: dict[str, dict] = {}
    for ep in terminals:
        if method == "continuous" and wait_mode == "approved":
            wait = sum(wait_parts[ep])
        else:
            wait = terminals[ep].get("logical_elapsed_ms", 0.0) - action_ms[ep]
        if wait < -1e-6:
            raise ValueError(f"negative waiting time in episode {ep}: {wait}ms")
        events[ep] = {"shifts": shifts[ep], "wait_ms": max(0.0, wait)}
    return terminals, events, records


# ------------------------------------------------------------ Rule-based


def extract_rule(path: Path, invalid: list[int]) -> tuple[dict, dict, int]:
    terminals: dict[str, dict] = {}
    shifts: dict[str, list[tuple[float, float]]] = defaultdict(list)
    hover_wait_ms: dict[str, float] = defaultdict(float)
    records = 0

    for line_number, record in _jsonl_records(path):
        if record is None:
            invalid.append(line_number)
            continue
        records += 1
        ep = _episode_id(record)
        if ep is None:
            continue
        kind = record.get("record_type")
        if kind == "episode_end":
            if ep in terminals:
                raise ValueError(f"duplicate terminal episode {ep}")
            terminals[ep] = record
            continue
        # hover_wait_ms is emitted on both decision and execution records
        # for the rule-based paradigm; sum both.
        hover = _number(record, "hover_wait_ms", 0.0)
        if hover > 0:
            hover_wait_ms[ep] += hover
        if kind == "decision":
            time_ms = _number(record, "action_age_ms")
            state_m = _number(record, "state_drift_m")
            if time_ms is not None and state_m is not None:
                shifts[ep].append((time_ms / 1000.0, state_m))

    events = {
        ep: {"shifts": shifts[ep], "wait_ms": hover_wait_ms[ep]}
        for ep in terminals
    }
    return terminals, events, records


# ------------------------------------------------------------ collision


def _strip_prefix(name: str) -> str:
    for prefix in ("success_", "oracle_success_", "oracle_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def scan_airsim_collision(eval_dir: Path, episode_ids: set[str]) -> dict[str, bool]:
    """AirSim judgment: has_collided=True in any frame of the episode's log.

    Scans <eval_dir>/<episode-id>/log/*.json (episode dirs may carry a
    success_/oracle_ prefix). Missing dirs/logs -> episode counts as no
    collision and is reported in warnings by the caller.
    """
    result: dict[str, bool] = {}
    if not eval_dir.is_dir():
        return result
    for entry in eval_dir.iterdir():
        if not entry.is_dir():
            continue
        name = _strip_prefix(entry.name)
        if name not in episode_ids or name in result:
            continue
        log_dir = entry / "log"
        if not log_dir.is_dir():
            result[name] = False
            continue
        collided = False
        for frame in sorted(log_dir.glob("*.json")):
            try:
                data = json.loads(frame.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                continue
            if data.get("sensors", {}).get("state", {}).get("collision", {}).get("has_collided", False):
                collided = True
                break
        result[name] = collided
    return result


# ----------------------------------------------------------- aggregation


def build_result(
    path: Path,
    method: str,
    terminals: dict,
    events: dict,
    records: int,
    invalid: list[int],
    eval_dir: Path,
    wait_denominator: str = "all",
) -> dict:
    episodes = []
    airsim = scan_airsim_collision(eval_dir, set(terminals))
    for ep in sorted(terminals):
        terminal = terminals[ep]
        ev = events.get(ep, {"shifts": [], "wait_ms": 0.0})
        shifts = ev["shifts"]
        # CR: AirSim judgment — episode_end.collision AND any frame has_collided
        has_airsim = bool(airsim.get(ep, False))
        episodes.append(
            {
                "episode_id": ep,
                "success": int(bool(terminal.get("success"))),
                "oracle_success": int(bool(terminal.get("oracle_success"))),
                "collision": int(bool(terminal.get("collision")) and has_airsim),
                "episode_end_collision": int(bool(terminal.get("collision"))),
                "airsim_collision": int(has_airsim),
                "navigation_error": _number(terminal, "final_ne_m", float("nan")),
                "e2e_latency_s": _number(terminal, "logical_elapsed_ms", 0.0) / 1000.0,
                "executed_waypoints": int(terminal.get("control_steps") or 0),
                "time_shift_s": _fmean([t for t, _ in shifts]),
                "state_shift_m": _fmean([s for _, s in shifts]),
                "applied_guidance": len(shifts),
                "buffer_wait_s": ev["wait_ms"] / 1000.0,
            }
        )

    n = len(episodes)
    if n == 0:
        raise ValueError("no terminal episodes found")
    summary = {
        "SR": sum(e["success"] for e in episodes) / n,
        "OSR": sum(e["oracle_success"] for e in episodes) / n,
        "CR": sum(e["collision"] for e in episodes) / n,
        "NE_m": statistics.fmean(e["navigation_error"] for e in episodes),
        "avg_control_steps": statistics.fmean(e["executed_waypoints"] for e in episodes),
        "mean_logical_E2E_s": statistics.fmean(e["e2e_latency_s"] for e in episodes),
        "time_shift_s": _fmean([e["time_shift_s"] for e in episodes if e["time_shift_s"] is not None]),
        "state_shift_m": _fmean([e["state_shift_m"] for e in episodes if e["state_shift_m"] is not None]),
        # Main table rows PPO/rule/continuous/stop-go average over all episodes;
        # the two TC rows (TC-OFF 19.48, TC-ON 92.81) average only over episodes
        # that experienced any wait. --wait-denominator reproduces either.
        "buffer_wait_s": statistics.fmean(
            [e["buffer_wait_s"] for e in episodes if wait_denominator == "all" or e["buffer_wait_s"] > 0.0]
        ),
    }
    warnings: list[str] = []
    if n != 1418:
        warnings.append(f"expected 1418 terminal episodes, got {n}")
    if invalid:
        warnings.append(f"{len(invalid)} invalid JSON lines: {invalid[:10]}")
    missing_shifts = [e["episode_id"] for e in episodes if e["time_shift_s"] is None]
    if missing_shifts and method != "stopgo":
        warnings.append(f"{len(missing_shifts)} episodes without shift samples")
    unscanned = [e["episode_id"] for e in episodes if e["episode_id"] not in airsim]
    if unscanned:
        warnings.append(f"{len(unscanned)} episodes without AirSim frame scan (CR counted as no collision)")

    return {
        "method": method,
        "input": str(path),
        "records": records,
        "episodes": n,
        "warnings": warnings,
        "summary": summary,
        "episode_details": episodes,
    }


# ------------------------------------------------------------------ main


EXTRACTORS = {
    "tc": extract_tc,
    "ppo": extract_ppo,
    "continuous": extract_action_line,
    "stopgo": extract_action_line,
    "rule": extract_rule,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--eval-dir", type=Path,
                        help="evaluation run directory containing episode log dirs "
                             "(default: parent of the JSONL's parent)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", default="auto",
                        choices=("auto",) + tuple(EXTRACTORS))
    parser.add_argument("--wait-mode", default="auto", choices=("auto", "e2e-action", "approved"),
                        help="buffer-wait definition: 'auto' uses the main-table "
                             "definition per paradigm (stop-go/ppo: E2E minus action "
                             "time; continuous: approved component sum; rule: hover "
                             "sum; tc: buffer_wait records)")
    parser.add_argument("--wait-denominator", default="all", choices=("all", "waiting"),
                        help="buffer_wait_s averaging denominator: 'all' (default, "
                             "matches PPO/rule/continuous/stop-go rows) averages over "
                             "every episode; 'waiting' averages only over episodes "
                             "with any wait (matches the two TC main-table rows)")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input file not found: {args.input}")
    eval_dir = args.eval_dir or args.input.parent.parent

    invalid: list[int] = []
    method = args.method if args.method != "auto" else detect_method(args.input, invalid)
    if args.method == "auto" and invalid:
        # detection consumed some invalid lines; re-run with a fresh list
        invalid = []
    wait_mode = args.wait_mode
    if wait_mode == "auto":
        # Main-table definitions per paradigm
        wait_mode = "approved" if method == "continuous" else "e2e-action"

    if method == "continuous":
        terminals, events, records = extract_action_line(args.input, invalid, "continuous", wait_mode)
    elif method == "stopgo":
        terminals, events, records = extract_action_line(args.input, invalid, "stopgo", wait_mode)
    else:
        terminals, events, records = EXTRACTORS[method](args.input, invalid)

    result = build_result(args.input, method, terminals, events, records, invalid, eval_dir,
                          wait_denominator=args.wait_denominator)
    output = args.output or args.input.with_name(args.input.stem + "_metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(
        {"method": result["method"], "episodes": result["episodes"],
         "records": result["records"], "warnings": result["warnings"],
         "summary": result["summary"]},
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
