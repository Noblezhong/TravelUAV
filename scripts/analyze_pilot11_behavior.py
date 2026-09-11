#!/usr/bin/env python3
"""pilot11 行为验证门（行为优先，再谈指标）。

读 eval_save_path 下全部 profile_logs/*.jsonl 的 scheduler_step / episode_end，
沿用 2026-09 /tmp 脚本的既有判定口径：
  - 决策时 buffer = 该 episode 上一步动作后的 buffer_remaining（首步用自身）。
  - 请求 = request_decision=='REQUEST' 的动作（含异步 CONTINUE_REQUEST）。
  - STOP_NO_REQUEST 判定: 动作后 buffer − 决策时 buffer ≥ 2 → wait(等在途结果落地)，
    否则 → dry_hover(真干悬停)。planner_has_inflight 是时序假象，不用于分类。

输出:
  1. 冻结占比(dry_hover 步 / 总步)，另给 dry_hover 且决策时 buffer>0 的占比(冻结的
     "有货不飞"语义)。
  2. 请求率条件表: 行=决策时 buffer 档，列=带宽档(Mbps) → req%(n)。期望形态:
     低 buffer&高 bw 高、其余低。
  3. 请求/ep、带宽决策级四分位请求率、STOP/飞行占比、动作分布。
  4. SR/CR/OSR/终局分布(metric 上下文)。

用法: python3 scripts/analyze_pilot11_behavior.py <eval_save_path>
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

def load(eval_dir):
    rows, ep_ends = [], []
    logs_dir = os.path.join(eval_dir, "profile_logs")
    if not os.path.isdir(logs_dir):
        sys.exit(f"no profile_logs under {eval_dir}")
    files = sorted(f for f in os.listdir(logs_dir) if f.endswith(".jsonl"))
    if not files:
        sys.exit(f"no jsonl under {logs_dir}")
    for fi, f in enumerate(files):
        with open(os.path.join(logs_dir, f), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                d["_src"] = fi  # 跨段/跨进程 episode_idx 会重置，必须带上文件源
                rt = d.get("record_type")
                if rt == "scheduler_step":
                    rows.append(d)
                elif rt == "episode_end":
                    ep_ends.append(d)
    return rows, ep_ends

def seq_by_episode(rows):
    seq = defaultdict(list)
    for r in rows:
        seq[(r["_src"], r["episode_idx"])].append(r)
    for rs in seq.values():
        for i, r in enumerate(rs):
            r["_prev_buf"] = rs[i - 1]["buffer_remaining"] if i > 0 else r["buffer_remaining"]
    return seq

def quartile_edges(vals):
    vals = sorted(vals)
    n = len(vals)
    if n == 0:
        return None
    def pct(p):
        return vals[min(n - 1, int(round(p * (n - 1))))]
    return [pct(0.25), pct(0.5), pct(0.75)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    args = ap.parse_args()
    rows, ep_ends = load(args.eval_dir)
    if not rows:
        sys.exit(f"no scheduler_step rows under {args.eval_dir}")
    seq = seq_by_episode(rows)
    total = len(rows)
    n_ep = len(seq)
    acts = Counter(r["action_name"] for r in rows)
    stop = [r for r in rows if r["motion_decision"] == "STOP"]
    req = [r for r in rows if r["request_decision"] == "REQUEST"]

    print("=== 1. 动作分布 / 冻结占比 ===")
    for an, c in acts.most_common():
        print(f"  {an:>22}: {c:>7} ({100.0 * c / total:.1f}%)")
    print(f"  总 scheduler_step: {total}, n_ep={n_ep}")
    print(f"  STOP 占比: {100.0 * len(stop) / total:.1f}%")
    print(f"  请求/ep: {len(req) / n_ep:.2f}")

    dry, dry_buf_pos, snr = 0, 0, 0
    for rs in seq.values():
        for i, r in enumerate(rs):
            if r["action_name"] != "STOP_NO_REQUEST":
                continue
            snr += 1
            prev_buf = rs[i - 1]["buffer_remaining"] if i > 0 else r["buffer_remaining"]
            if r["buffer_remaining"] - prev_buf >= 2:
                continue  # wait(结果落地)
            dry += 1
            if prev_buf > 0:
                dry_buf_pos += 1
    print(f"\n  冻结占比(dry_hover STOP_NO_REQUEST / 总步): {100.0 * dry / total:.1f}%")
    print(f"  冻结且有货不飞(dry 且决策时 buffer>0) / 总步: {100.0 * dry_buf_pos / total:.1f}%")
    print(f"  (STOP_NO_REQUEST {snr} 步: dry {100.0 * dry / snr:.1f}% / wait {100.0 * (snr - dry) / snr:.1f}%)")

    print("\n=== 2. 请求率条件表 req%(n)  行=决策时buffer  列=带宽档(Mbps) ===")
    bw_bands = [(0, 5, "<5"), (5, 10, "5-10"), (10, 25, "10-25"),
                (25, 50, "25-50"), (50, 100, "50-100"), (100, 1e9, ">=100")]
    header = "  buf\\bw |" + "".join(f"{t:>10}" for _, _, t in bw_bands)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for b in range(0, 8):
        cell = []
        for lo, hi, _ in bw_bands:
            g = [r for r in rows if r["_prev_buf"] == b and lo <= r["observed_bandwidth_mbps"] < hi]
            cell.append(f"{100.0 * sum(1 for r in g if r['request_decision'] == 'REQUEST') / len(g):>6.1f}%" if g else "      --")
        print(f"      {b:>3}  |" + "".join(f"{c:>10}" for c in cell))
    cell = []
    for lo, hi, _ in bw_bands:
        g = [r for r in rows if r["_prev_buf"] >= 8 and lo <= r["observed_bandwidth_mbps"] < hi]
        cell.append(f"{100.0 * sum(1 for r in g if r['request_decision'] == 'REQUEST') / len(g):>6.1f}%" if g else "      --")
    print(f"     >=8  |" + "".join(f"{c:>10}" for c in cell))

    print("\n=== 3. 带宽决策级四分位请求率(整 run 带宽分布分位) ===")
    edges = quartile_edges([r["observed_bandwidth_mbps"] for r in rows])
    qlabs = [(None, edges[0], "Q1"), (edges[0], edges[1], "Q2"),
             (edges[1], edges[2], "Q3"), (edges[2], None, "Q4")]
    for lo, hi, tag in qlabs:
        g = [r for r in rows if (lo is None or r["observed_bandwidth_mbps"] >= lo) and (hi is None or r["observed_bandwidth_mbps"] < hi)]
        n = len(g)
        print(f"  {tag} n={n:>6}: req={100.0 * sum(1 for r in g if r['request_decision'] == 'REQUEST') / n:6.2f}%")

    print("\n=== 4. SR / CR / OSR ===")
    if ep_ends:
        n_succ = sum(1 for e in ep_ends if e.get("success"))
        n_osr = sum(1 for e in ep_ends if e.get("oracle_success"))
        n_coll = sum(1 for e in ep_ends if e.get("collision"))
        print(f"  SR={100.0 * n_succ / len(ep_ends):.2f}% ({n_succ}/{len(ep_ends)})")
        print(f"  OSR={100.0 * n_osr / len(ep_ends):.2f}%")
        print(f"  CR={100.0 * n_coll / len(ep_ends):.2f}%")
        print(f"  终局: {dict(Counter(e.get('terminal_reason') for e in ep_ends))}")

if __name__ == "__main__":
    main()
