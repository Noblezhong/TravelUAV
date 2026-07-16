# PPO DRL Scheduler Training Notes

本文档记录当前 `DRL Hybrid UAV-VLN Scheduler` 的代码改动、建模约定和训练流程。另一台设备上的 coding agent 可以先阅读本文档，再继续训练、调参或评估。

## 1. 当前目标

当前目标不是训练 VLN 模型，而是训练一个端侧调度器，用来在 UAV-VLN 闭环中决定：

- 是否停下来等待或重新请求边缘 VLM。
- 是否继续执行当前 waypoint buffer。
- 是否向边缘侧 LLaMA-UAV 发起新的轨迹规划请求。

调度器面对的问题是：纯 stop-and-go 精度较稳但端到端时延较高；纯 continuous 系统时延更低，但通信时延会扩大观测位置和执行位置之间的时序失配，导致导航偏航和碰撞。因此 PPO scheduler 的目标是在精度下降可控的前提下，降低 episode 级导航时延并抑制时序失配。

## 2. 代码入口

新增或修改的核心文件：

- `src/vlnce_src/drl_scheduler_env.py`
  Gymnasium 环境，封装 AirSim/TravelUAV 交互、edge VLM 请求、trajectory buffer、通信时延、state、action、reward 和 profile 日志。

- `src/vlnce_src/drl_scheduler_train.py`
  使用 Stable-Baselines3 PPO 在线训练 scheduler。

- `src/vlnce_src/drl_scheduler_eval.py`
  加载训练好的 PPO scheduler，在完整评估集上做 deterministic evaluation。

- `scripts/drl_scheduler_train.sh`
  PPO 训练脚本。

- `scripts/drl_scheduler_eval.sh`
  PPO scheduler 评估脚本。

- `scripts/build_drl_scheduler_dataset.py`
  根据已有 stop-and-go 与 continuous 评估结果筛选 1418 条轨迹，生成 PPO scheduler 的 curriculum json。

- `requirements_rl.txt`
  PPO 依赖：
  `stable-baselines3[extra]==2.4.1`，`gymnasium==0.29.1`。

## 3. Scheduler State

当前 PPO 输入不使用 oracle 目标距离、oracle success、完整地图或真实目标点。当前 observation 定义为：

```text
o_t = [r_t^B, d_t^B, w_t, f_t, Delta_t, tau_t]
```

各维含义：

- `r_t^B`：当前 waypoint buffer 中剩余未执行 waypoint 数量。
- `d_t^B`：当前位置到下一个待执行 waypoint 的欧式距离。
- `w_t`：当前 scheduler step 采样到的上行带宽。
- `f_t`：当前是否有 edge VLM request in-flight。
- `Delta_t`：active trajectory 的 state drift，即当前 UAV 位置与生成该 trajectory 时观测位置之间的空间偏移。
- `tau_t`：active trajectory 的 time drift，即当前 trajectory 从观测提交到当前时刻经过的时间。

当前代码中的归一化尺度：

```text
r_t^B / 7.0
d_t^B / 1.0
w_t / 100 Mbps
f_t in {0, 1}
Delta_t / 2.5 m
tau_t / 5000 ms
```

注意：后续统一使用 `time drift`，不要再使用 `action age` 作为论文或日志解释术语。

## 4. Scheduler Action

PPO action space 是 4 个离散动作，本质上对应二元决策：

```text
a_t = (motion_decision, request_decision)
motion_decision in {STOP, CONTINUE}
request_decision in {REQUEST, NO_REQUEST}
```

动作编码：

```text
0 = STOP_REQUEST
1 = STOP_NO_REQUEST
2 = CONTINUE_REQUEST
3 = CONTINUE_NO_REQUEST
```

执行语义：

- `STOP_REQUEST`
  停止执行旧 buffer，采集当前位置观测，向 edge VLM 请求新 trajectory，并等待新 trajectory 返回后整体替换 buffer。

- `STOP_NO_REQUEST`
  如果已有 in-flight request，则停下来等待该 request 返回并替换 buffer；如果没有 in-flight request，则先 hover `scheduler_idle_wait_ms`，再基于当前位置观测自动触发一次 `STOP_REQUEST`，避免无意义空等。

- `CONTINUE_REQUEST`
  先执行当前 buffer 中的 1 个 waypoint，然后异步上传当前位置观测，请求 edge VLM 生成新 trajectory。

- `CONTINUE_NO_REQUEST`
  执行当前 buffer 中的 1 个 waypoint，不提交新请求。

当前实现不再定义 illegal action。若 PPO 在 buffer 为空时选择 `CONTINUE`，runtime 会强制 fallback 到 `STOP_REQUEST`，并在 profile 中记录：

```text
forced_fallback = "empty_buffer_stop_request"
```

## 5. Runtime 关键逻辑

冷启动阶段强制执行一次 `STOP_REQUEST`，等待 edge VLM 返回首条 trajectory 后，才进入 PPO scheduler 决策。

每个 PPO step 对应一次 scheduler decision：

1. 当前 scheduler step 采样一次上行带宽 `w_t`。
2. 根据 PPO action 执行 STOP/CONTINUE 和 REQUEST/NO_REQUEST。
3. 如果提交 edge request，则该 request 使用同一次采样到的 `w_t` 计算 uplink latency。
4. 新 planner result 返回后，完整替换 `active_traj`，并重置 `active_index=0`。
5. 每次 `CONTINUE` 只执行 1 个 waypoint。
6. collision、success、oracle_success、done、maxWaypoints 由 termination logic 处理，不交给 scheduler 决定。

为了避免 PPO 初期随机策略无限停等，新增：

```text
--scheduler_max_steps 800
```

该参数表示每个 episode 最大 scheduler 决策次数，超过后按 failure 结束。它是训练环境安全边界，不是主要导航终止条件。

## 6. Reward

当前 reward 使用 NE progress 作为稠密导航进展奖励，同时惩罚调度耗时、时序失配增量和 edge request。

每一步 reward：

```text
reward =
  lambda_NE * clip((NE_before - NE_after) / NE0, -1, 1)
  - lambda_T * elapsed_time / T0
  - lambda_Delta * state_drift_increase / Delta0
  - lambda_tau * time_drift_increase / tau0
  - lambda_Q * I_req
  + terminal_reward
```

终止奖励：

```text
if SR:
    terminal_reward = +R_SR
elif OSR but not SR:
    terminal_reward = +R_OSR
elif collision:
    terminal_reward = -R_c
else:
    terminal_reward = -R_f
```

第一版默认参数：

```text
NE0 = 1.0 m
T0 = 5000 ms
Delta0 = 2.5 m
tau0 = 5000 ms

lambda_NE = 1.0
lambda_T = 0.2
lambda_Delta = 2.0
lambda_tau = 2.0
lambda_Q = 0.2

R_SR = 40
R_OSR = 20
R_c = 20
R_f = 10
```

设计意图：

- `NE progress` 给 PPO 提供更密集的导航进展信号。
- `elapsed_time` 和 `request` 是系统开销，需要惩罚但权重较低。
- `state drift` 和 `time drift` 更容易导致 UAV 偏航，因此惩罚权重更高。
- SR 比 OSR 奖励更高，鼓励最终真正完成导航，而不是只进入 oracle success 范围。

## 7. Curriculum 数据集构建

使用已有 1418 条 seen_valset 评估轨迹构建 scheduler 训练 curriculum。这里不区分 VLN 模型训练集和评估集，因为训练对象是 scheduler，不是 LLaMA-UAV。

输入评估目录：

```text
stop-and-go: /HDD1/code/TravelUAV/eval_output_com
continuous:  /HDD1/code/TravelUAV/eval_pro_con_com_w5
source json: /HDD2/TravelUAV_dataset/TravelUAV_data/data/uav_dataset/seen_valset.json
```

构建命令：

```bash
cd /HDD1/code/TravelUAV

python scripts/build_drl_scheduler_dataset.py \
  --source_json /HDD2/TravelUAV_dataset/TravelUAV_data/data/uav_dataset/seen_valset.json \
  --stop_eval_dir /HDD1/code/TravelUAV/eval_output_com \
  --continuous_eval_dir /HDD1/code/TravelUAV/eval_pro_con_com_w5 \
  --output_json /HDD1/code/TravelUAV/drl_scheduler_seen_curriculum.json \
  --manifest_json /HDD1/code/TravelUAV/drl_scheduler_seen_curriculum_manifest.json
```

当前筛选结果：

```text
total episodes = 1418
selected episodes = 639
selected frames = 31216

recoverable = 547
easy = 92
unsolved = 758
continuous_only = 21
```

训练集选择：

```text
include recoverable
include easy
exclude unsolved
exclude continuous_only
```

生成的两个 json 已加入 `.gitignore`，不会提交到远程仓库。

## 8. 训练步骤

建议在工作站 4090 上训练，不需要 Jetson。

1. 激活环境并安装 PPO 依赖：

```bash
conda activate llamauav
cd /HDD1/code/TravelUAV
pip install -r requirements_rl.txt
```

2. 启动 AirSim server：

```bash
cd /HDD1/code/TravelUAV
python airsim_plugin/AirVLNSimulatorServerTool.py \
  --port 25000 \
  --clock_speed 1 \
  --root_path /HDD2/AeroDuo_envs
```

3. 构建 curriculum json：

```bash
cd /HDD1/code/TravelUAV
bash -lc 'python scripts/build_drl_scheduler_dataset.py \
  --source_json /HDD2/TravelUAV_dataset/TravelUAV_data/data/uav_dataset/seen_valset.json \
  --stop_eval_dir /HDD1/code/TravelUAV/eval_output_com \
  --continuous_eval_dir /HDD1/code/TravelUAV/eval_pro_con_com_w5 \
  --output_json /HDD1/code/TravelUAV/drl_scheduler_seen_curriculum.json \
  --manifest_json /HDD1/code/TravelUAV/drl_scheduler_seen_curriculum_manifest.json'
```

4. 小规模 smoke test：

```bash
cd /HDD1/code/TravelUAV
bash scripts/drl_scheduler_train.sh \
  --scheduler_total_timesteps 64 \
  --maxWaypoints 30 \
  --scheduler_max_steps 80
```

5. 正式训练：

```bash
cd /HDD1/code/TravelUAV
bash scripts/drl_scheduler_train.sh
```

训练输出目录类似：

```text
/HDD1/code/TravelUAV/drl_train_MMDD-HHMM/
```

模型保存位置：

```text
drl_train_MMDD-HHMM/scheduler_models/ppo_scheduler_*.zip
```

## 9. 评估步骤

训练完成后运行：

```bash
cd /HDD1/code/TravelUAV
bash scripts/drl_scheduler_eval.sh /path/to/ppo_scheduler.zip
```

默认评估完整 seen_valset：

```text
/HDD2/TravelUAV_dataset/TravelUAV_data/data/uav_dataset/seen_valset.json
```

默认输出目录：

```text
/HDD1/code/TravelUAV/eval_drl_scheduler_com
```

评估 summary 会输出：

```text
SR
OSR
CR
avg_waypoints
avg_NE_m
avg_episode_latency_ms
avg_time_drift_ms
avg_state_drift_m
avg_T_action_ms
avg_T_dec_ms
action_counts
forced_fallback_count
```

## 10. 当前已知注意事项

1. PPO 当前是在线训练，不是从历史 profile 离线训练。历史 stop-and-go 和 continuous 结果只用于筛选 curriculum episode。

2. `STOP_NO_REQUEST` 没有 in-flight request 时会短暂 hover 后自动请求，这是为了让动作语义闭环，避免训练中出现无意义空等。

3. `scheduler_max_steps` 是训练安全边界，主要防止随机策略阶段卡死。

4. 当前没有使用 Jetson、CMA 或 Continuous DNN，后续不要把这些逻辑混入 PPO scheduler 第一版。

5. 评估和训练中统一使用 `time drift` 字段，不再使用 `action age` 术语。

6. `scripts/rule_eval.sh` 属于 rule-based hybrid baseline，不是 PPO scheduler 训练入口。
