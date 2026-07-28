# PPO Scheduler 正式实验计划

> 最近更新：2026-07-23（控制预算修复）
>
> PPO Scheduler 代码提交：待提交
>
> **架构说明**：最终系统 = PPO Scheduler（调度层）+ TrajCorr（轨迹层）。当前分开评估，消融清晰。
> TC OFF ≡ Continuous w=5 基线（DNN 在 Jetson vs 5090 的差异仅影响系统时延绝对值，不影响调度策略论证）。
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

- [x] PPO 调度器 MDP 建模：8 维观测、4 离散动作、SMDP 时间尺度。
- [x] SplitACPolicy：Actor 看不到 NE（index 7 置零），Critic 看全 8 维。
- [x] 三项关键修复：`request_age` 第 8 维、gamma 0.999→0.995、DINO 时间减免。
- [x] 训练 0722-1254 首次避免 CONT_REQ 坍缩，CR 稳定 5-8%。
- [x] 代码提交，与远程 TrajCorr 模块合并无冲突。
- [x] 控制预算修复：所有 async 范式统一使用 `max_control_steps=1000`（对齐 stop-and-go）。
- [x] `scripts/rule_eval.sh` 完善：+fast_eval、+时间戳、+max_control_steps。

### 正在进行

- [ ] 正式 DRL 评估 `0723-0012` 运行中（使用旧预算 200，仅用于 CR 坍缩判断）。

### 下一步唯一任务

- [ ] `0723-0012` 完成后，确认 CONT_REQ 没有坍缩到 0%。
- [ ] 用新预算 1000 重新评估 PPO Scheduler。

### 暂时不要做

- [ ] 不启动新训练。
- [ ] 不修改 reward 权重、网络结构或观测空间。
- [ ] 不跑 Rule-based 评估（等 DRL 评估有阶段性结论后）。
- [ ] 不尝试将 PPO/Rule-based 部署到 Jetson（DNN 硬件差异已记录，不影响导航精度比较）。

## 1. 研究目标

验证 PPO 调度器能否在 SMDP 时间尺度（1 waypoint 决策步）上学习何时切换到 Continuous 范式（管道化 VLM 请求），从而在维持与 Stop-and-go 接近的 SR/OSR 的前提下，显著降低 UAV-VLN 端到端时延。

正式实验比较以下四组：

1. **Stop-and-go**：请求期间停止飞行，精度上限，时延最高。代码路径：Jetson Edge DNN evaluator（TrajCorr 实验统一口径）。
2. **Continuous `w=5`（≡ TC OFF）**：异步逐批请求（每 5 waypoints 提交一次），UAV 持续飞行。全管道化，状态漂移最大，精度下限，时延最低。代码路径：TrajCorr TC OFF evaluator（Jetson 架构）。
3. **Rule-based Hybrid**：基于 buffer 水位和 NE 阈值的确定性规则调度器，工程基线。代码路径：本地 `hybrid_eval.py`（DNN 在 5090）。
4. **PPO Scheduler（ours）**：DRL 学习的调度策略，目标是在精度-时延 Pareto 前沿上取得优于 Rule-based 的 trade-off。代码路径：本地 `drl_scheduler_eval.py`（DNN 在 5090）。

四组使用相同数据集、模型权重、通信 trace、Fast Eval 配置和控制预算（1000 waypoints）。

历史 Continuous `w=5`（旧 `continue_eval.py`）和早期坍缩版 PPO（0718 等）只用于研究动机和问题排查，不进入最终严格对比。

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

- **Stop-and-go**：每个决策步 STOP + REQUEST VLM → 等待结果 → 执行 DNN 输出的 waypoints（实际执行 P1-P5，最多 5 点）。无管道化，请求期间 UAV 悬停。DNN 在 Jetson 运行。
- **Continuous `w=5`（≡ TC OFF）**：异步逐批请求（每 5 控制点提交一次），结果返回后替换 buffer。全管道化，状态漂移最大。DNN 在 Jetson 运行。与 TC OFF 是同一份代码、同一套指标。
- **Rule-based Hybrid**：基于 buffer 水位和 NE 阈值的确定性规则切换 STOP/CONTINUE 和 REQUEST/NO_REQUEST。无学习。DNN 在 5090 运行。
- **PPO Scheduler**：PPO 在 SMDP 时间尺度上输出 4 个离散动作之一，每步根据 8 维观测决策。安全网 DINO 在 NE 阈值处自动触发。DNN 在 5090 运行。

架构差异：TC OFF 的 DNN 在 Jetson 推理，PPO/Rule-based 的 DNN 在 5090 推理。DNN 模型权重相同，输出相同，**导航精度直接可比**。系统时延绝对值不可直接比（Jetson DNN 慢于 5090），但论文的核心论点是调度决策减少 VLM 等待时间，与 DNN 硬件无关。论文中以 footnote 说明即可，不作为需要修复的实验问题。

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

### 3.2 正式 DRL 评估：0723-0012（旧预算 200，仅用于坍缩检测）

- Run ID：`0723-0012`
- 结果目录：`/code/TravelUAV/eval_drl_0723-0012_fast_x10`
- 配置：确定性推理、Fast Eval `x10`、**旧预算 max_control_steps=200**
- 状态：即将完成

> 此评估的控制预算仅为 stop-and-go 的 1/5，SR/OSR 不可直接对比。仅用于确认 CONT_REQ 没有坍缩。
> 预算修复后需用 `max_control_steps=1000` 重跑。

### 3.3 基线状态

- Stop-and-go 基线：TrajCorr 实验阶段二产出（Jetson，`maxWaypoints=200` 决策步）。
- Continuous `w=5`（≡ TC OFF）：TrajCorr 实验阶段一产出（Jetson，`max_control_steps=1000`）。当前 TC OFF `0723-1551` 运行中。
- Rule-based Hybrid 基线：`scripts/rule_eval.sh` 已就绪，尚未运行。
- PPO Scheduler：待 0723-0012 完成后用新预算重跑。

### 3.4 PPO 训练历史（不进入正式对比）

以下版本因 CONT_REQ 坍缩到 0% 而废弃：

- 0718、0720、0721-1124、0722-0542 等 5+ 版训练

根因已定位并修复，这些结果仅记录在训练日志中，不参与任何正式统计。

## 4. 后续执行顺序

### 阶段一：完成并验收旧预算评估 0723-0012

- [ ] 等待评估完成全部 1418 个 episode。
- [ ] 确认 CONT_REQ > 0%（核心坍缩检测）。
- [ ] 统计动作分布、buffer-level 行为。
- [ ] 记录结果但注明预算不公平，不进入正式对比。

### 阶段二：新预算 PPO Scheduler 评估

- [ ] 使用 `scripts/drl_scheduler_eval.sh`（已修复 `max_control_steps=1000`）运行 1418 集。
- [ ] 统计导航精度、系统性能、调度行为全部指标。
- [ ] 确认 CONT_REQ 保持非零。

**Reviewer checkpoint A：**

- PPO 调度器是否退化为 stop-and-go（CR=0%）？
- 调度行为是否展现出策略智能？
- 结果能否作为 PPO Scheduler 的正式评估？

### 阶段三：运行 Rule-based 基线

- [ ] 使用 `scripts/rule_eval.sh`（`max_control_steps=1000`）运行 1418 集。
- [ ] 输出与 PPO Scheduler 完全相同的指标。

### 阶段四：收集 TrajCorr 基线

- [ ] 等待 TrajCorr 实验产出正式 Stop-and-go 和 TC OFF 结果。
- [ ] 确认与 PPO/Rule-based 评估口径一致。

### 阶段五：四组统一对比与分析

- [ ] 输出四组导航精度与系统性能表格。
- [ ] 分析 PPO Scheduler 在精度-时延 Pareto 前沿上的位置。
- [ ] 计算精度保持率和时延节省率。

**Reviewer checkpoint B：**

- PPO Scheduler 是否在精度-时延 trade-off 上优于 Rule-based？
- CONT_REQ 的行为规律是否可以清晰解释？
- 证据是否足以支持论文表述？

### 阶段六：深度分析与可选消融

（同之前）

### 阶段七：论文写作

（同之前）

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
| `scripts/drl_scheduler_train.sh` | 训练超参（gamma、weights、episodes） |
| `scripts/drl_scheduler_eval.sh` | 评估配置（模型路径、fast_eval） |
| `src/common/param.py` | 所有 CLI 参数定义与默认值 |

## 9. 当前下一步

```text
等待 0723-0012 完成（CR 坍缩检测）
→ Reviewer 确认 CR ≠ 0%
→ 用新预算 max_control_steps=1000 重跑 PPO Scheduler 评估
→ Reviewer checkpoint A
→ 跑 Rule-based 基线（scripts/rule_eval.sh）
→ 等 TrajCorr 产出 Stop-and-go + TC OFF 结果
→ 四组统一对比
→ Reviewer checkpoint B
```
