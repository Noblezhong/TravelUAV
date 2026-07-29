# 轨迹修正模块正式实验计划

> 最近更新：2026-07-29
>
> TC ON 代码提交：`251dcd4`
>
> 本文档是轨迹修正实验的唯一工作台账。每完成一个阶段，都必须同步更新“当前状态、下一步任务、实验结果和 reviewer 结论”。

## 0. 如何根据本文档继续工作

当用户说“根据文档继续完成科研实验任务”时，执行者必须：

1. 先读取本文档，不依赖聊天记忆猜测实验状态。
2. 检查“当前状态”和“下一步唯一任务”，只推进第一个未完成任务。
3. 执行前确认不会中断正在运行的实验，也不会覆盖已有结果目录。
4. 不修改第 2 节的统一评估口径；如确需修改，先停止执行并交给 reviewer 决定哪些基线需要重跑。
5. 每完成一个实验或代码里程碑，更新本文档并提交 Git。
6. 到达标记为“Reviewer checkpoint”的位置时，先汇报证据、问题和建议，等待用户 rebuttal 后再进入下一阶段。

职责分工：

- **执行者（Codex）**：检查运行状态、审查代码、执行或指导实验、统计指标、定位异常、更新本文档。
- **Reviewer（用户）**：质疑实验公平性、判断结果是否支持研究假设、批准关键代码设计和下一阶段实验。

工作环境：

- 工作站仓库：`/HDD1/code/TravelUAV`
- Jetson：`zt@192.168.105.13`
- Jetson 仓库：`/home/zt/code/TravelUAV`
- 工作站共享结果：`/HDD1/traveluav_eval_shared`
- Jetson 共享结果：`/home/zt/traveluav_eval_shared`
- 当前 TC OFF 日志：
  `/home/zt/traveluav_eval_shared/eval_trajcorr_off_0723-1551_launcher.log`

运行约束：

- 当前 TC ON 完成前，不重启 evaluator，不在 Jetson 再次 pull，不切换其运行代码。
- 只在用户主动询问、出现异常，或累计增加约 100 个 episode 时检查长时间实验。
- 结果目录必须使用独立 Run ID，禁止覆盖或混合续跑不同代码版本。

## 当前任务看板

### 已完成

- [x] 统一 TC OFF/ON 的 `max_control_steps=1000`。
- [x] 实现 TC ON 的 state-shift gate、目标锁定、请求冻结和回头 waypoint 过滤。
- [x] TC ON 代码提交为 `251dcd4`，静态与单元测试通过。
- [x] 启动正式 TC OFF：`0723-1551`。
- [x] 正式 TC OFF 完成 1418 个 episode，并完成第一轮指标统计。
- [x] Reviewer checkpoint A：TC OFF 的数据、终止条件和控制预算通过检查；其 Fast Eval x1 等价性后续单独复核。
- [x] 在 5090 工作站启动正式 Stop-and-go：`0728-1740`。
- [x] 在 4090 + Jetson 完成 TC ON 跨端 smoke test，并清理临时输出。
- [x] 按用户决定启动正式 TC ON：`0728-1810`。
- [x] 完成 Fast Eval 时序代码审计，确认 TrajCorr evaluator 的虚拟时间实现存在缺口，详见 2.3 节。
- [x] 量化现有 TC OFF：54164 次非冷启动请求平均晚于虚拟 ready time `2772.3ms`，1336 个 episode 出现超过一个动作边界的晚应用。

### 正在进行

- [ ] 正式 Stop-and-go 完整运行 1418 个 episode。
- [ ] 正式 TC ON 完整运行 1418 个 episode。
- [ ] 将当前 TC ON 作为旧版 x10 完整实验继续运行，不在中途修改代码。

### 下一步唯一任务

- [ ] 等待 Stop-and-go 与当前 TC ON 完成，保留并统计现有结果。
- [ ] 修复 TC OFF/ON 共用的 Fast Eval 虚拟时间控制。
- [ ] 使用相同的 50 个 episode 比较“旧 x10、修复后 x10、正常 x1”，再决定是否完整重跑。

### 暂时不要做

- [ ] 不根据旧 Stop-and-go 或旧 Continuous 结果计算正式恢复率。
- [ ] 不调整 TC ON 阈值、目标锁定或请求逻辑，直到 TC OFF 与正式 Stop-and-go 的结果被 reviewer 审查。

## 1. 研究目标

验证 Trajectory Correction 能否在相同异步 Continuous `w=5` 框架下，提高 UAV-VLN 的导航精度，并缩小其与 Stop-and-go 精度上限之间的差距。

正式实验只比较以下三组：

1. **TC OFF**：不启用轨迹修正的 Continuous `w=5` 基线。
2. **TC ON**：启用目标锁定轨迹修正的 Continuous `w=5`。
3. **Stop-and-go**：请求期间停止飞行的精度上限。

历史 Continuous、旧 TC ON 和旧 Stop-and-go 结果只用于研究动机和问题排查，不进入最终严格对比。

## 2. 统一评估口径

三组实验必须固定以下条件：

- 数据集：相同的 1418 个 episode。
- 模型：相同的 LLaMA-UAV、GroundingDINO 和 trajectory DNN 权重。
- 通信：按 `seq_name` 固定带宽 trace 起点，使用相同动态带宽与上行时延。
- Fast Eval：统一使用 `x10`，黑盒模型推理时间仍按真实测量值计入逻辑时间。
- 碰撞：使用 AirSim 物理碰撞，不使用 `tiny diff` 碰撞。
- 终止：统一 `predict_done`、SR、OSR、collision 和 request-level NE regression 规则。
- NE regression：只记为 failure，不记为 collision。
- Stop-and-go 的 OSR：只检查真正执行的 P1-P5，不检查未执行的 P6/P7。

控制预算按理论执行 waypoint 数量对齐：

- Stop-and-go：`maxWaypoints=200` 个决策步，每步执行 5 个 waypoint，理论上限 1000 个 waypoint。
- TC OFF/ON：`max_control_steps=1000` 个逐点控制步。

### 2.1 三种范式的实际执行语义

- **Stop-and-go**：同步完成一次边缘推理后，调用 `moveOnPathAsync`；虽然 DNN 输出 7 点，但 AirSim 客户端通过 `target_idx=5` 在 P5 暂停，因此每个决策实际最多执行 5 点。
- **TC OFF**：异步 Continuous `w=5`；逐点执行当前 buffer，每累计 5 个控制点提交一次请求，结果返回后替换 buffer。
- **TC ON**：正常阶段与 TC OFF 相同；只有 state shift 达到阈值时进入目标锁定轨迹修正。

三者在数据、精度判定、通信 trace 和理论控制预算上对齐，但以下差异属于范式本身，不能强行消除：

- Stop-and-go 请求期间停止；TC OFF/ON 请求期间继续飞行。
- Stop-and-go 的 DINO 在同步决策环执行；TC OFF/ON 的 DINO 随异步 edge result 返回。
- Stop-and-go 的单次 `T_action` 是 5 点 path chunk；TC OFF/ON 的单次 `T_action` 是 1 个控制点。最终比较应优先使用 episode e2e latency；若比较动作时延，必须明确统计单位。
- `Avg waypoints` 表示已消耗/尝试的控制 waypoint 数量，不等同于每个 waypoint 都已精确到达。

### 2.2 指标解释

- `coarse time/state shift`：请求观测到 coarse result 被应用之间的时间和位置差。
- `traj time/state shift`：TC ON 获取当前修正观测到修正轨迹首次执行之间的时间和位置差。
- 轨迹修正不声称消除 coarse shift；它通过当前图像和当前位置重新生成细粒度轨迹，补偿 coarse result 应用时的执行起点失配。
- Time shift 仅记录和分析，当前不作为是否触发轨迹修正的 gate。

### 2.3 Fast Eval 时序审计

Fast Eval 的目标不是简单把所有时间除以 10，而是保持原速系统中“新轨迹在第几个 waypoint 生效”的顺序：

1. AirSim 动作使用 `ClockSpeed=10`。
2. 异步上行只消耗原来的 `1/10` 墙钟时间；UAV 同时以 x10 执行旧 buffer，因此对应完整的原速上行时延。
3. VLM/DINO/DNN 推理时间不乘 10。
4. 使用虚拟原速时间决定新结果何时可见，并用虚拟时间统计动作、time shift 和 episode latency。

代码审计结论：

- Stop-and-go 请求期间本来就停止飞行，其 Fast Eval 时序成立。
- Continuous、Rule-based 和 PPO 的 planner 已包含“真实 VLM 尚未完成时，不允许 x10 仿真额外越过其原速返回时刻”的处理。
- TC OFF/ON 使用独立的 `LatestOnlyEdgeVLMClient`。该客户端记录了虚拟返回时间，但真实 RPC 尚未完成时仍可能继续执行 x10 waypoint，因此 TrajCorr 的 Fast Eval 实现不完整。

2026-07-28 22:01 的只读日志审计：

- 已完成 TC OFF 的 54164 次非冷启动请求中，结果相对计算出的虚拟 ready time 平均晚应用 `2772.3ms`。
- Edge RPC 的平均真实计算时间为 `351.1ms`；在 x10 下，未隔离的理论放大量约为 `351.1 × (10-1) = 3159.9ms`，与日志量级一致。
- TC OFF 中 1336 个 episode 出现结果晚于前一个 waypoint 边界一个以上动作时间的记录。
- 正在运行的 TC ON 同样使用该客户端，但目标锁定和等待会改变受影响程度，因此不能假设 OFF/ON 偏差完全抵消。

当前实验处理：

- 不删除、不覆盖，也不在运行中修改 TC OFF/ON 结果。
- 当前 TC OFF/ON 先标记为“统一 x10 加速设置下的完整实验”，尚未证明与 x1 数值等价。
- 投稿前最低验证：修复 TrajCorr 虚拟时间控制后，用相同的 50 个 episode 比较三组：
  1. 已有旧 x10；
  2. 修复后 x10；
  3. 正常 x1。
- 若修复后 x10 与 x1 的结果应用控制步、轨迹行为和指标趋势接近，并且旧 x10 的 OFF/ON 收益方向一致，则保留已有完整 x10 结果并明确实验设置，不完整重跑。
- 若旧 x10 与修复后 x10/x1 出现明显不同的导航结论，则重新完整运行修复后的 TC OFF 和 TC ON。

## 3. 当前状态

### 3.1 正式 TC OFF 已完成

- Run ID：`0723-1551`
- 工作站结果目录：
  `/HDD1/traveluav_eval_shared/eval_trajcorr_off_0723-1551_fast_x10`
- 配置：
  `trajcorr_mode=off`、`w=5`、`max_control_steps=1000`、Fast Eval `x10`

该实验使用启动时加载的提交 `d42637f`，已完成 `1418/1418` 个 episode，无残留评估进程。profile 包含 1418 条唯一 `episode_end` 记录。其结果完整，但由于 2.3 节记录的 TrajCorr Fast Eval 缺口，当前作为 x10 加速实验保存，正式 x1 等价性待验证。

| 指标 | TC OFF |
|---|---:|
| SR | 2.96% |
| OSR | 4.09% |
| CR | 66.93% |
| Avg waypoints | 197.69 |
| Avg NE | 183.85 m |
| Avg T_dec | 3694.84 ms |
| Avg T_action | 2102.88 ms |
| Avg time shift | 6535.88 ms |
| Avg state drift | 3.47 m |
| Avg episode e2e latency | 523.37 s |

### 3.2 正式 Stop-and-go 正在运行

- 设备：5090 工作站 `192.168.105.9`
- Run ID：`0728-1740`
- 结果目录：
  `/code/TravelUAV/eval_stop_go_0728-1740_fast_x10`
- 代码提交：`20fd509`
- 配置：
  `maxWaypoints=200`、Fast Eval `x10`、通信时延开启
- tmux：
  `srv` 运行 AirSim Server，`eval` 运行 Stop-and-go evaluator

该实验是正式同步精度上限。运行期间不修改评估边界。

### 3.3 已清理的旧实验

`0723-1301` 使用旧的 200 控制步上限，只是 pilot，已删除，不进入任何正式统计。

### 3.4 TC ON 已完成 smoke test，正式评估正在运行

工作站代码已经加入目标锁定状态机、请求冻结、pending request 清理和回头 waypoint 过滤。Jetson 与工作站关键执行文件哈希一致。由于 5090 与 4090 + Jetson 可独立运行，用户决定在 Stop-and-go 尚未完成时并行推进 TC ON；这不改变三组的统一评估边界。

Smoke test 使用 3 条具有 `state shift >= 2.5m` 的 TC OFF episode，确认：3 条均正常终止、控制步未超过 1000、Fast Eval x10 生效、`corrected=19`、`target_lock=43`，且目标锁定阶段没有触发普通 `continuous_w5` 请求。测试输出、日志、临时 JSON 和 manifest 已清理。

正式 TC ON：

- Run ID：`0728-1810`
- Jetson tmux：`tc_on_full`
- 输出目录：`/home/zt/traveluav_eval_shared/eval_trajcorr_on_0728-1810_fast_x10`
- 4090：AirSim Server `25000`，Edge VLM Server `26000`
- 配置：`trajcorr_mode=on`、`max_control_steps=1000`、Fast Eval `x10`

正式 TC ON 正在运行。2026-07-29 10:29 已完成 `267/1418` 个 episode，约 `18.8%`；最近 100 个 episode 平均约为 `250s/episode`。若后续速度保持，预计还需约 `3–3.5 天`，约在 2026-08-01 完成。该估算会随长 episode、场景切换和重试变化。

正式 TC ON 必须同时包含：

1. 在 UAV 当前位置重新构造 coarse vector，使用当前图像和 DNN 生成新 waypoint sequence。
2. state shift 未超过阈值时保持普通 Continuous `w=5`；超过阈值后冻结请求计数，完成旧 coarse goal 的目标锁定后再恢复 `w=5`。
3. 根据当前位姿过滤已越过的目标和回头 waypoint，禁止追赶已经失效的后方点。

### 3.5 本轮 TC ON 已实现的具体行为

- Gate：`state shift < 2.5m` 使用原始轨迹；`state shift >= 2.5m` 进入目标锁定。
- 状态机：`NORMAL / TARGET_LOCK / WAIT_REFRESH`。
- 进入目标锁定时：
  - 锁定旧 `coarse_goal_world`；
  - 冻结全局 `w=5` 计数，保留进入修正前的计数值；
  - 丢弃尚未提交的旧 pending snapshot；
  - 使用当前位置、当前图像、旧目标方向和原 coarse vector 长度调用 DNN。
- waypoint 过滤：
  - 只执行沿目标方向前进且使旧目标距离减小的点；
  - 不执行已越过旧目标或要求 UAV 回头的点。
- 目标锁定结束：
  - 距离旧目标不超过 `0.5m`：`goal_reached`；
  - 已越过旧目标：`goal_passed`；
  - 修正 buffer 耗尽仍未到达：`buffer_exhausted`。
- 三种结束情况均从当前位置提交新 edge request、清零 `w=5` 计数、丢弃修正 buffer 并悬停等待。
- buffer 耗尽时不重复调用本地 DNN，而是按已确定方案重新请求边缘 VLM。
- 新 edge result 返回并激活原始轨迹后，恢复普通 Continuous `w=5`。

新增正式日志：

- `execution_phase`
- `target_lock_active`
- `target_lock_goal_world`
- `target_lock_distance_m`
- `continuous_counter_frozen`
- `target_lock_completion_reason`
- `dropped_pending_request`
- `correction_complete / correction_passed_goal / correction_buffer_exhausted`

当前验证情况：

- `py_compile`、Shell 语法和 `git diff --check` 已通过。
- TrajCorr、评估边界、Fast Eval 和 velocity settle 共 39 项静态/单元测试通过。
- 尚未做 AirSim runtime 验证，因此不能仅凭静态测试宣称 TC ON 已有效。

## 4. 后续执行顺序

### 阶段一：完成并验收 TC OFF

- [x] 等待 `0723-1551` 完成 1418 个 episode。
- [x] 检查所有 episode 均有唯一终止记录，无缺失或重复。
- [x] 检查控制步不超过 1000；22 条达到预算上限，所有 episode 均唯一落盘。
- [x] 统计导航精度：
  `SR / OSR / CR / Avg waypoints / Avg NE`。
- [x] 统计系统性能：
  `Avg T_dec / Avg T_action / Avg time shift / Avg state drift / Avg episode e2e latency`。
- [ ] 保存 episode 列表、带宽映射、代码 commit 和 Fast Eval manifest，作为 TC ON 与 Stop-and-go 的共同实验配置。

**Reviewer checkpoint A：**

- TC OFF 是否确实等价于关闭轨迹修正的 Continuous `w=5`？
- 是否存在异常提前终止、缺失 episode 或统计口径错误？
- 结果能否作为正式异步基线？

只有 reviewer 接受 TC OFF 后，才能启动正式 Stop-and-go。

### 阶段二：运行并验收正式 Stop-and-go

- [ ] 在启动前静态确认 Stop-and-go 与 TC OFF 的数据集、模型、通信 trace、碰撞、DINO、SR/OSR、NE regression 和 Fast Eval 口径一致。
- [ ] 确认 Stop-and-go 使用 `maxWaypoints=200` 个决策步，每步实际最多执行 P1-P5，理论控制点预算为 1000。
- [ ] 使用独立目录运行完整 1418 条：
  `eval_stop_go_<RUN_ID>_fast_x10`。
- [ ] 检查所有 episode 的终止记录、控制预算、重试和结果落盘。
- [ ] 输出与 TC OFF 完全相同的导航精度和系统性能指标。
- [ ] 计算精度上限差：

```text
recoverable SR gap = SR_StopGo - SR_TC_OFF
recoverable OSR gap = OSR_StopGo - OSR_TC_OFF
```

**Reviewer checkpoint B：**

- Stop-and-go 是否可以作为相同模型和预算下的正式精度上限？
- TC OFF 与 Stop-and-go 的差距是否足以支持轨迹修正研究？
- 哪些 episode 属于“TC OFF 失败但 Stop-and-go 成功”的可恢复样本？

若 Stop-and-go 本身出现边界或代码问题，先修复并重跑受影响实验，不进入 TC ON。

### 阶段三：冻结正式对照与构建 TC ON 验证集

- [ ] 固定 TC OFF 和 Stop-and-go 的正式结果目录、commit、episode 列表及带宽映射。
- [ ] 使用 `scripts/build_trajcorr_eval_subset.py` 生成配对验证集。
- [ ] 小规模集合优先包括：
  60 个 TC OFF 失败但 Stop-and-go 成功的 recoverable episode；
  20 个 TC OFF 成功 episode；
  20 个两者均失败的随机 episode。
- [ ] 保存 subset JSON 和 manifest，但不把生成的数据文件提交 Git。

### 阶段四：完成 TC ON 代码运行前验收

已完成的代码工作：

- [x] 只在 corrected trajectory 分支进入目标锁定，TC OFF 保持全局 `w=5`。
- [x] 加入 `NORMAL / TARGET_LOCK / WAIT_REFRESH` 状态机。
- [x] 修正阶段冻结计数并丢弃未提交的旧 pending request。
- [x] 以 `0.5m` 为到达阈值，越过目标时禁止回头。
- [x] 修正 buffer 耗尽时从当前位置重新请求并悬停等待。
- [x] 通过静态检查及 TrajCorr 单元测试。
- [x] 提交并记录 TC ON 代码 commit：`251dcd4`。
- [x] 同步工作站、GitHub 和 Jetson 磁盘仓库。
- [ ] `0723-1551` 完成后再启动新 evaluator，禁止当前 TC OFF 中途切换代码版本。

运行 TC ON 前的最小回归检查：

由于正式 TC OFF `0723-1551` 使用修改前的 Jetson 进程，而 TC ON 修改位于 OFF/ON 共用 evaluator 中，完整 TC ON 前必须验证 OFF 分支未发生行为变化。

- [ ] 从 `0723-1551` 选择相同的 3–5 个 episode。
- [ ] 使用同步后的新代码运行 `trajcorr_mode=off`。
- [ ] 按 episode 对齐请求步、带宽序列、buffer 替换、控制步和终止类型。
- [ ] 若行为一致，保留 `0723-1551` 作为完整正式 TC OFF，不重新运行 1418 条。
- [ ] 若出现 OFF 行为差异，先定位共享代码；只有确认差异影响导航行为或指标时才重跑完整 TC OFF。

### 阶段五：TC ON 小规模运行验证

- [ ] 先运行与回归检查相同的 3–5 个 TC ON episode。
- [ ] 确认小 state shift 保持 Continuous `w=5`。
- [ ] 确认大 state shift 进入目标锁定，修正期间冻结请求计数。
- [ ] 确认到达、越过目标或 buffer 耗尽后重新请求并等待。
- [ ] 确认不会执行回头 waypoint，不会覆盖已有结果。
- [ ] 扩展到正式的 100 个配对 episode。
- [ ] 比较 `SR / OSR / CR / Avg NE`、修正触发率、旧目标到达率和修正阶段局部 NE progress。

**Reviewer checkpoint C：**

- TC ON 的轨迹修正行为是否符合设计，而不是通过改变终止条件获得收益？
- TC ON 是否至少满足：SR 或 OSR 提升、Avg NE 降低、CR 不明显恶化？
- 若不满足，应修改哪一项 TC ON 内部机制？不得顺手修改公共评估边界。

只有 reviewer 批准后，才能运行完整 TC ON。

### 阶段六：完整运行 TC ON

- [ ] 使用与 `0723-1551` 相同的 episode、带宽映射、代码口径和控制预算运行 1418 条。
- [ ] 使用新的时间戳目录：
  `eval_trajcorr_on_<RUN_ID>_fast_x10`
- [ ] 确认 OFF/ON episode 一一对应。
- [ ] 完成后按 episode 配对统计 TC OFF、TC ON 和 Stop-and-go。

### 阶段七：最终结论与论文表格

- [ ] 输出三组统一导航精度与系统性能表格。
- [ ] 统计 `OFF failure -> ON success` 与 `OFF success -> ON failure`。
- [ ] 完成 McNemar 检验和 SR recovery ratio。
- [ ] 区分两类结论：
  - 轨迹修正是否提升 Continuous `w=5` 的精度；
  - 轨迹修正增加了多少请求、等待和 episode e2e latency。
- [ ] 将最终结果、失败案例和 reviewer 结论写回本文档。

**Reviewer checkpoint D：**

- 证据是否足以支持“轨迹修正恢复异步 Continuous 导航精度”的论文表述？
- 是否需要做第 6 节的可选消融？

## 5. 最终比较

导航精度：

- `SR`
- `OSR`
- `CR`
- `Avg waypoints`
- `Avg NE`

系统性能：

- `Avg T_dec`
- `Avg T_action`
- `Avg time shift`
- `Avg state shift`
- `Avg episode e2e latency`
- `request count`

配对分析：

- `OFF failure -> ON success`
- `OFF success -> ON failure`
- McNemar 检验
- 修正触发率
- 旧 coarse goal 到达率
- 修正阶段局部 NE progress
- passed-goal / 回头 waypoint 拒绝次数

轨迹修正的 SR 恢复率定义为：

```text
SR recovery ratio =
    (SR_TC_ON - SR_TC_OFF)
    / (SR_StopGo - SR_TC_OFF)
```

模块有效的最低判断标准：

- TC ON 的 SR 或 OSR 高于 TC OFF；
- Avg NE 降低；
- CR 不出现明显恶化。

若 TC ON 有精度收益但增加时延，保留轨迹修正模块，后续再由 PPO Scheduler 联合优化请求频率和范式切换。

## 6. 可选消融

只有完整 TC ON 已证明有效后，才进行约 100 个 episode 的消融：

- 仅启用“当前位置重构 coarse vector”和“回头 waypoint 过滤”；
- 不启用目标锁定期间的请求冻结。

该消融用于判断精度提升是否主要来自目标锁定请求机制，不需要默认运行完整 1418 条。

## 7. 后续修改与重跑规则

### 7.1 可只重跑 TC ON

以下改动属于轨迹修正模块内部，不改变基线边界：

- state shift 阈值；
- 旧目标到达半径；
- coarse vector 重构方式；
- waypoint 前向/回头过滤；
- 目标锁定状态机；
- 修正 buffer 耗尽处理；
- 修正期间的请求冻结和恢复策略。

这些改动效果不好时，只需重新运行 TC ON。修改后仍应先做 3–5 个 episode smoke test，再进行 100 个配对 episode 验证。

### 7.2 需要验证或重跑 TC OFF

若修改 OFF/ON 共用 evaluator 的以下部分，必须先做 TC OFF 兼容性回放：

- 全局 `w=5` 计数；
- pending/latest-only 请求顺序；
- buffer 替换；
- observation 刷新时机；
- coarse result 应用时机；
- 公共 profile 指标计算。

只有回放出现实际行为或指标差异时，才需要重跑完整 TC OFF。

### 7.3 需要重跑受影响的全部基线

以下评估边界一旦修改，历史结果不可直接继续用于正式对比：

- collision、DINO、SR/OSR、NE regression；
- Stop 的 `maxWaypoints=200` 或 TC 的 `max_control_steps=1000`；
- AirSim waypoint 控制和 velocity settle；
- Fast Eval 逻辑时间；
- 动态带宽 trace 及按 `seq_name` 的映射；
- 数据集 episode 列表或模型权重。

若只影响 TC evaluator，则重跑 TC OFF/ON；若 Stop-and-go 也共享该边界，则三组均需重新运行。

## 8. 实验记录模板

每完成一次正式或小规模实验，在本文档末尾追加以下记录：

```text
Experiment:
Run ID:
Date:
Code commit:
Mode:
Dataset / episode subset:
Output directory:
Key parameters:
Completion:
Retries / failures:

Accuracy:
SR:
OSR:
CR:
Avg waypoints:
Avg NE:

System:
Avg T_dec:
Avg T_action:
Avg time shift:
Avg state drift:
Avg episode e2e latency:

Finding:
Known issue:
Reviewer decision:
Next action:
```

## 9. 当前下一步

当前只执行以下任务：

```text
等待正式 Stop-and-go 0728-1740 与 TC ON 0728-1810 完成
→ 保留并统计现有 x10 结果
→ 修复 TrajCorr Fast Eval 虚拟时间控制
→ 用相同 50 个 episode 比较：旧 x10 / 修复后 x10 / 正常 x1
→ 修复后 x10 与 x1 接近且旧 x10 结论一致：保留完整旧 x10，不重跑
→ 结论明显不同：完整重跑修复后的 TC OFF 和 TC ON
```

当前 TC ON 运行期间不修改或同步 Jetson evaluator。Fast Eval 修复必须使用新进程和新输出目录，禁止与 `0728-1810` 混合续跑。在 Reviewer checkpoint 完成前，不调整 TC ON 轨迹修正策略或其他统一评估边界。
