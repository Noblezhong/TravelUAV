# Continuous w=5 正式评估监管交接

更新时间：2026-07-29 21:58（Asia/Shanghai）

本文档交接给 5090 上的 Claude Code。当前任务是后台监管已经启动的
Continuous w=5 正式评估，不修改方法、不重启训练，也不并行启动其他
GPU 评估。

## 1. 当前正式任务

```text
机器：5090
tmux：continuous
冻结工作区：/code/TravelUAV_eval_frozen
冻结 commit：9eb63fd
输出目录：/code/TravelUAV/eval_continuous_w5_0729-215750_fast_x10
主日志：/tmp/continuous.log
输出路径指针：/tmp/continuous_output_path
目标：1418 episodes
```

关键配置：

```text
paradigm = Continuous
chunk_waypoints = 5
max_control_steps = 1000
max_episodes_per_scene = 80
enable_comm_delay = True
comm_trace = bandwidth/ucc4g_bandwidth_trace.csv
fast_eval = True
fast_eval_speedup = 10
eval_json = seen_valset.json
```

该任务与 5090 上已经完成的 Stop-and-go 和 PPO 使用相同硬件、相同
1418-episode seen split、相同通信 trace 和相同 Fast Eval 逻辑时间口径。
Stop-and-go 是同步基线，Continuous w=5 是固定异步基线，PPO 是异步
调度优化方法。

## 2. Claude Code 的监管职责

每次接管时先执行只读检查：

```bash
tmux has-session -t continuous
tmux capture-pane -pt continuous -S -40
ps -eo pid,etime,cmd | grep '[c]ontinue_eval.py'
nvidia-smi
tail -n 80 /tmp/continuous.log
cat /tmp/continuous_output_path
```

统计当前完成进度：

```bash
OUT=$(cat /tmp/continuous_output_path)
PROF=$(find "$OUT/profile_logs" -name '*.jsonl' | head -1)
python3 - "$PROF" <<'PY'
import json
import sys

records = [json.loads(line) for line in open(sys.argv[1])]
ends = [r for r in records if r.get("record_type") == "episode_end"]
ids = [r["seq_names"][0] for r in ends]

print("completed", len(ends), "/ 1418")
print("unique_ids", len(set(ids)))
print("duplicates", len(ids) - len(set(ids)))
print("success", sum(bool(r["success"]) for r in ends))
print("oracle_success", sum(bool(r["oracle_success"]) for r in ends))
print("collision", sum(bool(r["collision"]) for r in ends))
PY
```

至少检查：

- `tmux continuous` 和 `continue_eval.py` 仍然存在；
- profile JSONL 持续增长；
- `episode_end` 数量与唯一 ID 数量一致；
- 日志中没有 `Traceback`、CUDA OOM、持续的 simulator connection failure；
- 冻结工作区的 HEAD 仍然是 `9eb63fd`；
- 不要把主工作区正在修改的 `continue_eval.py` 混入本次运行。

## 3. 异常处理边界

以下情况应立即记录证据并通知用户：

- tmux 在未完成 1418 个 episode 时退出；
- profile 15 分钟以上没有增长，且日志也没有正常 waypoint 输出；
- 出现 Traceback、CUDA OOM 或无法恢复的 AirSim 连接错误；
- episode ID 重复；
- 输出目录或 commit 与本文档不一致。

没有用户授权时，不要自动删除结果、覆盖输出目录或从头重跑。先保存：

```text
最后完成 episode 数
最后 100 行日志
tmux 状态
GPU/进程状态
profile 文件大小和更新时间
错误发生时间
```

## 4. 实时计划更新

监管期间同步更新：

```text
/code/TravelUAV/PPO_SCHEDULER_EXPERIMENT_PLAN.md
```

更新规则：

1. 启动时记录 tmux、输出目录、commit 和配置。
2. 每次人工接管或出现阶段性变化时更新完成数和最近检查时间。
3. 异常时立刻把状态改为 BLOCKED，并记录原始错误，不先写推测结论。
4. 完成 1418/1418 后，将 Continuous 标为待完整性审查，而不是直接标为
   可用于论文。
5. 完整性审查通过后，再启动 PPO–Continuous 的逐 episode 配对分析；
   Stop-and-go 只作为两者共享的同步参考。

## 5. 完成后的验收

必须确认：

- 1418 条 `episode_end`；
- 1418 个唯一 episode ID，无缺失、无重复；
- 配置、commit、trace 和 Fast Eval 字段完整；
- 碰撞 episode 在 terminal marker 后没有继续执行或发起请求；
- 生成 SR、OSR、CR、NE，以及 mean/median/p95 logical E2E；
- 与 PPO 按 episode ID 配对，报告 paired transition 和 bootstrap CI。

完成这些检查之前，不根据中途前缀结果判断 Continuous、PPO 或论文主张。
