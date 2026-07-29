# PPO Scheduler 正式实验计划

> 最近更新：2026-07-29（Stop-and-go 正式评估完成；下一步为 Continuous 终止同步预检）
>
> PPO Scheduler 代码提交：`20fd509`
>
> **架构说明**：最终系统 = PPO Scheduler（调度层）+ TrajCorr（轨迹层）。当前分开评估，消融清晰。
> 主对比四组基线全部在本地 5090 上运行（统一硬件）。TC OFF 作为 Jetson 部署的补充验证点，不进入主对比。
>
> 本文档是 PPO 调度器实验的唯一工作台账。每完成一个阶段，都必须同步更新"当前状态、下一步任务、实验结果和 reviewer 结论"。

## 0. 如何根据本文档继续工作

当用户说"根据文档继续完成 PPO Scheduler 实验"时，执行者必须：

1. 先读取本文档，不依赖聊天记忆猜测实验状态。
2. 检查"当前状态"和"下一步唯一任务"，只推进第一个未完成任务。
3. 执行前确认不会中断正在运行的实验，也不会覆盖已有结果目录。
4. 不修改第 2 节的统一评估口径；如确需修改，先停止执行并交给 reviewer 决定哪些基线需要重跑。
5. 每完成一个实验或代码里程碑，更新本文档并提交 Git。
6. 到达标记为"Reviewer checkpoint"的位置时，先汇报证据、问题和建议，等待用户 rebuttal 后再进入下一阶段。

职责分工：

- **执行者（Claude Code）**：检查运行状态、审查代码、执行或指导实验、统计指标、定位异常、更新本文档。
- **Reviewer（用户）**：质疑实验公平性、判断结果是否支持研究假设、批准关键代码设计和下一阶段实验。

工作环境：

- 工作站仓库：`/code/TravelUAV`
- 训练脚本：`scripts/drl_scheduler_train.sh`
- 评估脚本：`scripts/drl_scheduler_eval.sh`
- 训练结果目录格式：`drl_train_MMDD-HHMM/`
- 评估结果目录格式：`eval_drl_MMDD-HHMM_fast_x10/`

运行约束：

- 每次只跑一个训练或评估任务，不并行启动。
- 结果目录必须使用独立时间戳，禁止覆盖或混合续跑不同代码版本。
- 训练/评估进程运行期间不修改代码或切换分支。
- 评估使用确定性推理（`deterministic=True`），CR 会比训练时低，这是预期行为。

## 当前任务看板

### 已完成

- [x] PPO Scheduler MDP 建模与 SplitACPolicy
- [x] 三项关键修复（request_age、gamma 0.995、DINO 时间减免）
- [x] 训练 0722-1254：CR 5-8%，无坍缩
- [x] **PPO Scheduler 正式评估 `0723-1845`**：SR=47.9%, OSR=58.5%, CR=1.4% 非零, e2e=407.3s
- [x] 控制预算修复：四个范式统一 1000 waypoints
- [x] 四个评估脚本全部对齐（eval.sh, continue_eval.sh, rule_eval.sh, drl_scheduler_eval.sh）
- [x] 重启后 AirSim Server 启动方法已记录
- [x] fast_eval 逻辑屏障验证：PPO Scheduler 环境 `poll_result` 有 `ready_logical_ms` 屏障，x10 与 x1 逻辑等价
- [x] **Stop-and-go 正式评估 `0728-1740` 完成 1418/1418**：SR=50.21%，OSR=64.81%，collision=42.03%，mean E2E=410.67s
- [x] Stop 与 PPO 完成 1418-episode 配对边界检查：PPO 仅快 0.82%，bootstrap 95% CI 跨 0，且 SR/OSR/CR/NE 均更差；当前不能宣称 C1 已成立

### 正在进行

- [ ] 无 scheduler 正式评估进程。另一个会话正在修改 `hybrid_eval.py`，不得覆盖或混入 Continuous 修复。

### 下一步

- [ ] 在 `continue_eval.py` 每次 `makeActionsChunk` 后同步 simulator collision/done/success 状态，使终止语义与 PPO evaluator 一致
- [ ] 使用独立输出目录运行 5–10 episode smoke test，确认 trace 按 `seq_name` reset、碰撞后不继续飞行、Fast Eval manifest 和 episode-end 完整
- [ ] smoke 通过后启动 **Continuous w=5 正式评估**（`scripts/continue_eval.sh`，1418 episodes）
- [ ] 三组数据齐后（Stop-and-go + PPO + Continuous），分析 PPO 的精度-时延 Pareto 位置
- [ ] 之后按需跑 Rule-based

### 暂时不要做

- [ ] 不启动新训练、不修改 reward/网络/观测空间
- [ ] 不跑 Rule-based（等三组主对比完成）

## 1. 研究目标

验证 PPO 调度器能否在 SMDP 时间尺度（1 waypoint 决策步）上学习何时切换到 Continuous 范式（管道化 VLM 请求），从而在维持与 Stop-and-go 接近的 SR/OSR 的前提下，显著降低 UAV-VLN 端到端时延。

正式实验主对比以下四组（全部在本地 5090，统一硬件）：

1. **Stop-and-go**：请求期间停止飞行，精度上限，时延最高。脚本：`scripts/eval.sh`，代码：`src/vlnce_src/eval.py`。
2. **Continuous `w=5`**：异步逐批请求（每 5 控制点提交一次），UAV 持续飞行。全管道化，状态漂移最大，精度下限，时延最低。脚本：`scripts/continue_eval.sh`，代码：`src/vlnce_src/continue_eval.py`。
3. **Rule-based Hybrid**：基于 buffer 水位和 NE 阈值的确定性规则调度器，工程基线。脚本：`scripts/rule_eval.sh`，代码：`src/vlnce_src/hybrid_eval.py`。（暂缓）
4. **PPO Scheduler（ours）**：DRL 学习的调度策略。脚本：`scripts/drl_scheduler_eval.sh`，代码：`src/vlnce_src/drl_scheduler_eval.py`。✅ 已完成。

四组使用相同数据集、模型权重、通信 trace、Fast Eval x10、控制预算（1000 waypoints）、DNN 在 5090。

TC OFF（TrajCorr Jetson 部署）作为补充验证点，不进入主对比。

历史 Continuous `w=5`（旧未修复版本）和早期坍缩版 PPO（0718 等）只用于研究动机，不进入最终统计。

## 2. 统一评估口径

四组实验必须固定以下条件：

- 数据集：相同的 1418 个 episode（`seen_valset.json`）。
- 模型：相同的 LLaMA-UAV、GroundingDINO 和 trajectory DNN 权重。
- 通信：按 `seq_name` 固定带宽 trace 起点，使用相同动态带宽与上行时延。
- Fast Eval：统一使用 `x10`，黑盒模型推理时间仍按真实测量值计入逻辑时间。
- 碰撞：使用 AirSim 物理碰撞。
- 终止：统一 `predict_done`、SR、OSR、collision 和 NE regression 规则。
- 控制预算：统一 `max_control_steps=1000`（Stop-and-go 为 `maxWaypoints=200` 决策步 × P1-P5 ≈ 1000 waypoints）。所有 async 范式使用 `args.max_control_steps` 参数（TrajCorr 已定义，default=1000）。

### 2.1 四种范式的实际执行语义

- **Stop-and-go**：每个决策步 STOP + REQUEST VLM → 等待结果 → 执行 DNN 输出的 waypoints（P1-P5，最多 5 点）。无管道化，请求期间 UAV 悬停。DNN 在 5090。
- **Continuous `w=5`**：异步逐批请求（每 5 控制点提交一次），结果返回后替换 buffer。全管道化，UAV 在 VLM 推理期间继续飞行。DNN 在 5090。
- **Rule-based Hybrid**：基于 buffer 水位和 NE 阈值的确定性规则切换 STOP/CONTINUE 和 REQUEST/NO_REQUEST。无学习。DNN 在 5090。（暂缓）
- **PPO Scheduler**：PPO 在 SMDP 尺度上输出 4 个离散动作，每步根据 8 维观测决策。安全网 DINO 在 NE 阈值处自动触发。DNN 在 5090。✅ 已完成。

DNN 硬件：主对比四组 DNN 全部在本地 5090。TC OFF（Jetson DNN）作为 TrajCorr 模块的补充验证点，不进入主对比表。

### 2.2 PPO Scheduler 设计摘要

| 组件 | 规格 |
|------|------|
| 观测空间 | 8 维：`[buf, wp_dist, bw, inflight, drift, td, NE, request_age]` |
| 动作空间 | 4 离散：`STOP_REQUEST, STOP_NO_REQUEST, CONTINUE_REQUEST, CONTINUE_NO_REQUEST` |
| 策略网络 | SplitACPolicy：Actor 看 7 维（NE 置零），Critic 看全 8 维 |
| 硬边界 | buf=0→CONTINUE 非法；无 inflight→STOP_NO_REQUEST 非法 |
| 安全网 | NE>25/20/15/10m 且无 inflight→自动 DINO |
| 训练算法 | PPO (SB3), gamma=0.995, ent_coef=0.01, n_steps=128 |
| 训练数据 | 639 集（recoverable + easy），场景课程排序 |
| Reward | time + drift + time_drift + NE_progress + 终端 + illegal |

### 2.3 指标解释

导航精度：

- `SR`：Success Rate，NE ≤ 20m 的 episode 比例。
- `OSR`：Oracle Success Rate，SR + NE ≤ 20m 但未触发 predict_done 的比例。
- `CR`：Collision Rate。
- `Avg NE`：最终导航误差均值。
- `Avg waypoints`：消耗的控制 waypoint 数量。

系统性能：

- `Avg e2e latency`：episode 端到端逻辑时间。
- `Avg T_action`：单步动作执行时间（含 UAV 飞行 + 等待）。
- `request count`：VLM 请求总次数。
- `Avg time drift`：请求提交到结果应用的时间延迟。
- `Avg state drift`：请求提交到结果应用的位置偏差。

调度器特有：

- `STOP_REQUEST rate`：停止并请求的比例。
- `CONTINUE_REQUEST rate`（CR rate）：管道化请求的比例——**核心指标**。CR=0% 意味着策略坍缩为 stop-and-go。
- `CONTINUE_NO_REQUEST rate`：纯飞行消费 waypoints 的比例。
- `DINO safety-net trigger count`：安全网触发的次数。

## 3. 当前状态

### 3.1 训练已完成：0722-1254

- 训练目录：`/code/TravelUAV/drl_train_0722-1254`
- 模型路径：`drl_train_0722-1254/scheduler_models/ppo_scheduler_20260722-125409-242489.zip`
- 训练参数：gamma=0.995, time_weight=0.5, drift_weight=0.5, request_weight=0.0
- 训练结果：
  - 639 个 episode，约 90k 步
  - CR 最后 10 个 slice 稳定在 5-8%，无坍缩
  - MA20 reward 始终在 −4 到 +4 之间窄幅波动，已收敛
  - CONT_REQ 行为具备策略智能：42% 在 buf=1、33% 在 buf=2、96% 有 inflight 管道化

### 3.2 PPO Scheduler 正式评估：✅ 已完成

- Run ID：`0723-1845`
- 结果目录：`/code/TravelUAV/eval_drl_0723-1845_fast_x10`
- 配置：确定性推理、Fast Eval `x10`、`max_control_steps=1000`
- 结果：见下方实验记录

### 3.3 Stop-and-go 正式评估：✅ 已完成

- Run ID：`0728-1740`
- 结果目录：`/code/TravelUAV/eval_stop_go_0728-1740_fast_x10`
- 配置：Fast Eval `x10`、`maxWaypoints=200`（→ 1000 waypoints）
- 完整性：1418/1418，episode ID 全唯一，无缺失或重复
- SR：50.21%（712/1418）
- OSR：64.81%（919/1418）
- Collision：42.03%（596/1418）
- Mean final NE：66.22m
- Mean E2E：410.67s（logical time；median 346.55s，p95 986.68s）
- Mean executed waypoints：208.43
- Reviewer：通过共享同步参照验收；现有 PPO 相对 Stop 未证明有效 trade-off，C1 仍需 PPO vs Continuous 决定

### 3.4 基线状态

- **Continuous w=5**：trace reset 和控制预算已对齐；正式运行前仍需补齐每次 waypoint 后的 simulator 终止状态同步，并通过 5–10 episode smoke。
- **Rule-based Hybrid**：`scripts/rule_eval.sh` 已就绪，暂缓。
- **TC OFF / TC ON**：TrajCorr 实验独立进行，作为 Jetson 部署补充验证。

### 3.4 PPO 训练历史（不进入正式对比）

以下版本因 CONT_REQ 坍缩到 0% 而废弃：

- 0718、0720、0721-1124、0722-0542 等 5+ 版训练

根因已定位并修复，这些结果仅记录在训练日志中，不参与任何正式统计。

## 4. 后续执行顺序

### 阶段一：完成 Stop-and-go 评估 ✅

- [x] Stop-and-go `0728-1740` 完成 1418/1418。
- [x] 冻结汇总、原始 JSONL、输入和 evaluator 哈希。
- [x] 完成与现有 PPO 的配对边界检查；该检查不替代 PPO–Continuous 主比较。

### 阶段二：运行 Continuous w=5 评估

- [ ] 补齐 Continuous 每次 waypoint 后的 simulator collision/done/success 同步。
- [ ] 运行 5–10 episode smoke 并验收 trace、终止、Fast Eval 和输出完整性。
- [ ] smoke 通过后使用 `scripts/continue_eval.sh` 运行 1418 集。
- [ ] 统计全部指标。

### 阶段三：三组统一对比

- [ ] Stop-and-go vs PPO vs Continuous w=5。
- [ ] 分析 PPO 在精度-时延 Pareto 前沿上的位置。

**Reviewer checkpoint A：**

- PPO 是否在精度-时延 trade-off 上优于纯 Continuous？
- CONT_REQ 是否具备可解释的规律？
- 证据是否足以支持论文表述？

### 阶段四（后续）：Rule-based + 消融

（P1 优先级，暂不安排）

## 5. 最终比较

导航精度：

- `SR`
- `OSR`
- `CR`
- `Avg waypoints`
- `Avg NE`

系统性能：

- `Avg e2e latency`
- `Avg T_action`
- `request count`
- `Avg time drift`
- `Avg state drift`

调度器行为：

- `STOP_REQUEST rate`
- `CONTINUE_REQUEST rate`（核心）
- `CONTINUE_NO_REQUEST rate`
- `STOP_NO_REQUEST rate`
- DINO safety-net trigger count
- Buffer-level 动作分布
- inflight 条件下的动作分布

PPO Scheduler 有效性的最低判断标准：

1. CONT_REQ rate > 0%（不坍缩为 stop-and-go）。
2. SR ≥ Continuous `w=1` 基线。
3. e2e latency < Stop-and-go 基线。
4. 调度行为具备可解释的规律（CR 集中在低 buffer、高 inflight）。
5. 在精度-时延 trade-off 上至少不劣于 Rule-based Hybrid。

## 6. 后续修改与重跑规则

### 6.1 可只重跑 PPO 评估

以下改动属于 PPO 调度器内部，不改变基线边界：

- reward 权重（time_weight、drift_weight 等）
- ent_coef
- 训练 episode 数量和课程排序
- 评估模型 checkpoint 选择

这些改动只需重新训练和/或重新评估 PPO Scheduler。改动后仍应先做 10 个 episode smoke test。

### 6.2 需要重跑全部基线

以下评估边界一旦修改，所有四组必须重新运行：

- Fast Eval speedup
- 碰撞检测逻辑
- SR/OSR 判定规则
- NE regression 判定
- **`max_control_steps`（控制预算）** — 已修复为 1000
- 通信 trace 及按 `seq_name` 的映射
- 数据集 episode 列表或模型权重

### 6.3 DNN 硬件差异

- PPO Scheduler / Rule-based 的 DNN 在 5090，Stop-and-go / TC OFF 的 DNN 在 Jetson。
- **导航精度直接可比**（DNN 权重相同）。
- 系统时延绝对值受硬件影响，但调度策略论证不依赖 DNN 硬件。论文 footnote 说明即可。

### 6.3 代码改动提交规则

- 每次改动前确认没有正在运行的实验。
- 改动后先用 `--max_episodes 5` 做 smoke test。
- 提交时在 commit message 中注明改动的实验假设。
- 推送后更新本文档的"最近更新"日期和 commit hash。

## 7. 实验记录模板

每完成一次正式或小规模实验，在本文档末尾追加以下记录：

```text
Experiment:
Run ID:
Date:
Code commit:
Mode: [train | eval]
Dataset / episode subset:
Output directory:
Key parameters:

Training (if applicable):
  Episodes:
  Timesteps:
  Final CR rate:

Evaluation:
  SR:
  OSR:
  CR:
  Avg waypoints:
  Avg NE:
  CONT_REQ rate:
  STOP_REQUEST rate:
  Avg e2e latency:
  Avg T_action:
  request count:

Finding:
Known issue:
Reviewer decision:
Next action:
```

## 8. 关键文件索引

| 文件 | 用途 |
|------|------|
| `src/vlnce_src/drl_scheduler_env.py` | SMDP 环境：观测、动作、reward、硬边界、安全网 |
| `src/vlnce_src/drl_ac_policy.py` | SplitACPolicy：Actor/Critic 输入分离 |
| `src/vlnce_src/drl_scheduler_train.py` | PPO 训练入口 |
| `src/vlnce_src/drl_scheduler_eval.py` | PPO 评估入口 |
| `src/vlnce_src/eval.py` | Stop-and-go 评估 |
| `src/vlnce_src/continue_eval.py` | Continuous w=5 评估 |
| `src/vlnce_src/hybrid_eval.py` | Rule-based 评估（暂缓） |
| `scripts/drl_scheduler_train.sh` | 训练超参 |
| `scripts/drl_scheduler_eval.sh` | PPO 评估配置 |
| `scripts/eval.sh` | Stop-and-go 评估脚本 |
| `scripts/continue_eval.sh` | Continuous w=5 评估脚本 |
| `scripts/rule_eval.sh` | Rule-based 评估脚本（暂缓） |
| `src/common/param.py` | 所有 CLI 参数定义与默认值 |
| `airsim_plugin/AirVLNSimulatorServerTool.py` | AirSim 管理服务 |

### 重启后环境恢复

```bash
# 1. SSHFS 挂载
sshfs -o reconnect,ServerAliveInterval=15 zt@192.168.105.17:/HDD2/TravelUAV_dataset /HDD2/TravelUAV_dataset
sshfs -o reconnect,ServerAliveInterval=15 zt@192.168.105.17:/HDD2/AeroDuo_envs /HDD2/AeroDuo_envs
sshfs -o reconnect,ServerAliveInterval=15 zt@192.168.105.17:/HDD1/code/TravelUAV/Model /code/TravelUAV/Model

# 2. AirSim Server
tmux new-session -d -s srv 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate llamauav && cd /code/TravelUAV && python -u airsim_plugin/AirVLNSimulatorServerTool.py --port 25000 --root_path /code/TravelUAV/'
```

## 9. 当前下一步

```text
修复 Continuous 每次 waypoint 后的 simulator 终止状态同步
→ 5–10 episode smoke：trace reset / collision termination / Fast Eval / episode_end
→ smoke 通过后启动 Continuous w=5 正式 1418（continue_eval.sh）
→ Continuous 完成后，三组数据齐
→ 统一对比分析：Stop-and-go vs PPO vs Continuous w=5
→ Reviewer checkpoint A
→ （后续）Rule-based + 消融
```

## 10. 实验记录

### Experiment: Stop-and-go 正式评估
- Run ID: 0728-1740
- Date: 2026-07-28/29
- Run snapshot: `20fd509`（进程启动后 7 分钟提交；关键 evaluator 文件之后未变化）
- Mode: eval
- Dataset: seen_valset.json (1418 unique episodes)
- Output: `/code/TravelUAV/eval_stop_go_0728-1740_fast_x10`
- Key params: maxWaypoints=200（约 1000 waypoints）, fast_eval x10, trace reset by seq_name

**Evaluation:**
- SR: 50.21%
- OSR: 64.81%
- Collision: 42.03%
- Avg waypoints: 208.43
- Avg NE: 66.22m
- Avg e2e latency: 410.67s (median 346.55s, p95 986.68s)

**Finding:** Stop 通过共享同步参照验收。与 PPO 的 1418-episode 配对边界检查中，PPO mean E2E 仅低 3.36s（0.82%），bootstrap 95% CI 跨 0，同时 SR/OSR/CR/NE 更差。该结果不能证明 C1，也不能替代 PPO–Continuous 正式比较。
**Reviewer decision:** Stop 结果冻结；PPO 重训 gate 保持关闭；下一步先修复并 smoke Continuous，再运行正式 1418。

### Experiment: PPO Scheduler 正式评估
- Run ID: 0723-1845
- Date: 2026-07-23/24
- Code commit: 6a73390
- Mode: eval
- Dataset: seen_valset.json (1418 episodes)
- Output: `/code/TravelUAV/eval_drl_0723-1845_fast_x10`
- Key params: max_control_steps=1000, fast_eval x10, deterministic=True

**Evaluation:**
- SR: 47.9%
- OSR: 58.5%
- CR (碰撞): 47.3%
- Avg waypoints: 205.6
- Avg NE: 74.6m
- CONT_REQ rate: 1.4%
- STOP_REQUEST rate: 14.7%
- Avg e2e latency: 407.3s (med 343.9s, p95 1007.6s)
- Avg T_action: 924ms
- Avg T_dec: 4782ms
- Avg time drift: 7074ms
- Avg state drift: 2.45m

**Finding:** CR=1.4% 非零，策略没有坍缩回 stop-and-go。大部分步是 stop-and-go 行为，PPO 在安全时使用 CONT_REQ 管道化。精度接近 stop-and-go（快照 57% vs PPO 48%）。
**Reviewer decision:** 待 Stop-and-go 完成后统一对比。
