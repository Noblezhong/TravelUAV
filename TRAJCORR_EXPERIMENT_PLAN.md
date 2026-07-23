# 轨迹修正模块正式实验计划

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

## 3. 当前状态

### 3.1 正式 TC OFF 正在运行

- Run ID：`0723-1551`
- 工作站结果目录：
  `/HDD1/traveluav_eval_shared/eval_trajcorr_off_0723-1551_fast_x10`
- 配置：
  `trajcorr_mode=off`、`w=5`、`max_control_steps=1000`、Fast Eval `x10`

该实验是统一评估口径后的正式 TC OFF 基线。运行期间不得修改评估代码、终止条件或通信逻辑。

这里的“不得修改”指 Jetson 正在运行的代码副本。工作站可以开发 TC ON，但在 `0723-1551` 完成前不得向 Jetson pull、同步或重启 evaluator。

### 3.2 已清理的旧实验

`0723-1301` 使用旧的 200 控制步上限，只是 pilot，已删除，不进入任何正式统计。

### 3.3 TC ON 已完成静态修正，尚未进入正式运行

工作站代码已经加入目标锁定状态机、请求冻结、pending request 清理和回头 waypoint 过滤，并通过静态检查及单元测试。为避免影响正在运行的 `0723-1551 TC OFF`，该修改尚未同步到 Jetson，也没有启动 AirSim smoke test。

TC ON 只有在 `0723-1551` 完成后同步到 Jetson，并通过小规模配对运行验证，才能进入正式 1418 条评估。

正式 TC ON 必须同时包含：

1. 在 UAV 当前位置重新构造 coarse vector，使用当前图像和 DNN 生成新 waypoint sequence。
2. state shift 未超过阈值时保持普通 Continuous `w=5`；超过阈值后冻结请求计数，完成旧 coarse goal 的目标锁定后再恢复 `w=5`。
3. 根据当前位姿过滤已越过的目标和回头 waypoint，禁止追赶已经失效的后方点。

### 3.4 本轮 TC ON 已实现的具体行为

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
- TrajCorr、评估边界、Fast Eval 和 velocity settle 共 34 项静态/单元测试通过。
- 尚未做 AirSim runtime 验证，因此不能仅凭静态测试宣称 TC ON 已有效。

## 4. 后续执行顺序

### 阶段一：完成并验收 TC OFF

- [ ] 等待 `0723-1551` 完成 1418 个 episode。
- [ ] 检查所有 episode 均有唯一终止记录，无缺失或重复。
- [ ] 检查控制步不超过 1000，重试 episode 最终正确落盘。
- [ ] 统计：
  `SR / OSR / CR / Avg waypoints / Avg NE`
- [ ] 统计：
  `Avg T_dec / Avg T_action / Avg time shift / Avg state shift / Avg episode e2e latency`
- [ ] 保存 episode 列表、带宽映射、代码 commit 和 Fast Eval manifest，作为 TC ON 与 Stop-and-go 的共同实验配置。

### 阶段二：完成正式 TC ON 代码

- [x] 只在 corrected trajectory 分支进入目标锁定，TC OFF 保持全局 `w=5`。
- [x] 加入 `NORMAL / TARGET_LOCK / WAIT_REFRESH` 状态机。
- [x] 修正阶段冻结计数并丢弃未提交的旧 pending request。
- [x] 以 `0.5m` 为到达阈值，越过目标时禁止回头。
- [x] 修正 buffer 耗尽时从当前位置重新请求并悬停等待。
- [x] 通过静态检查及 TrajCorr 单元测试。
- [ ] `0723-1551` 完成后同步到 Jetson。
- [ ] 同步前提交并记录 TC ON 代码 commit。

### 阶段三：新代码下的 TC OFF 兼容性回放

由于正式 TC OFF `0723-1551` 使用修改前的 Jetson 进程，而 TC ON 修改位于 OFF/ON 共用 evaluator 中，完整 TC ON 前必须验证 OFF 分支未发生行为变化。

- [ ] 从 `0723-1551` 选择相同的 10–30 个 episode。
- [ ] 使用同步后的新代码运行 `trajcorr_mode=off`。
- [ ] 按 episode 对齐请求步、带宽序列、buffer 替换、控制步和终止类型。
- [ ] 若行为一致，保留 `0723-1551` 作为完整正式 TC OFF，不重新运行 1418 条。
- [ ] 若出现 OFF 行为差异，先定位共享代码；只有确认差异影响导航行为或指标时才重跑完整 TC OFF。

### 阶段四：TC ON 小规模配对验证

- [ ] 使用相同的 10–30 个 episode 运行 TC ON，检查状态机、冻结计数、pending 清理和悬停等待是否符合日志。
- [ ] 扩展到 100 个配对 episode，优先覆盖旧 Continuous 失败但 Stop-and-go 成功的样本。
- [ ] 检查 ON 的 SR/OSR、Avg NE 和 CR；若出现明显退化，先分析修正触发率、目标到达率、回头过滤和局部 NE progress，不直接运行 1418 条。

### 阶段五：完整运行 TC ON

- [ ] 使用与 `0723-1551` 相同的 episode、带宽映射、代码口径和控制预算运行 1418 条。
- [ ] 使用新的时间戳目录：
  `eval_trajcorr_on_<RUN_ID>_fast_x10`
- [ ] 确认 OFF/ON episode 一一对应。

### 阶段六：重跑正式 Stop-and-go

- [ ] 使用已经对齐的 `scripts/eval.sh` 运行 1418 条。
- [ ] 输出目录自动带时间戳和 Fast Eval 后缀：
  `eval_stop_go_<RUN_ID>_fast_x10`
- [ ] Stop-and-go 保持“请求期间停止等待、结果返回后一次执行 5 点”的原始行为。
- [ ] 不再使用历史 `/HDD1/code/TravelUAV/eval_stop_go` 作为正式精度上限。

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

这些改动效果不好时，只需重新运行 TC ON。修改后仍应先做 10–30 个 episode smoke test。

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

## 8. 下一步操作

1. 不干扰地等待 `0723-1551 TC OFF` 完成。
2. 完成后统计并验收 TC OFF 的精度与系统性能指标。
3. 提交当前 TC ON 工作站代码并同步 Jetson。
4. 先运行 10–30 个新代码 TC OFF episode，验证与 `0723-1551` 等价。
5. 再运行相同 episode 的 TC ON，检查状态机和局部导航效果。
6. 通过后扩展到 100 个配对 episode；达到有效性门槛后才运行完整 1418 条 TC ON。
7. 最后运行正式 Stop-and-go，形成三组最终论文结果。
