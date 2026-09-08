# AHS 调度器范式修改记录（todo#4 带宽自适应）

> 目的：记录每个 pilot 改了哪些 request/stop 奖励结构，保证可回溯、可复现。
> 运维规则：**每次 PPO 训练 / 评估前先重启 AirSim**（`nohup setsid bash scripts/airsim_server_5090.sh`，等 25000 LISTEN + "AirSim ready on port 25001"）。

## 总览：request 奖励一直在演化的两个维度

| 维度 | 语义 | 状态 |
|---|---|---|
| 该不该 request | buffer 需求 × 带宽成本 | buffer 已达成（pilot4 lo69%/mi0.4%）；带宽待验证（pilot5） |
| 该不该 stop | state_drift × time_drift | 未成形，策略擅用 CONTINUE_REQUEST（Step B 再做） |

## pilot1 — 基线（2026-08-21，20k）
- **改动**：无。固定 request 成本 0.05，无 hunger 无带宽项。
- **结果**：SR 32.9%（最高）。buffer 自然涌现 lo49%/mi0.3%/hi7.8%；固定 buffer 下高带宽 request>低带宽（lo: 53%>44%）。
- **教训**：正确行为可自然涌现，带宽方向也出现过。

## pilot2 — bw_scaled 单侧罚（2026-08-24，20k）
- **改动**：`request_cost = 0.05 × clip(100/bw, 0.25, 8)`（仅低带宽加重、高带宽趋零）。
- **结果**：SR 27.9%。带宽签名拉平（~13%）；CONTINUE_REQUEST 灭绝；buffer 行为反转（hi 31% / lo 0.5%）。

## pilot3 — bw_symmetric 对称激励（2026-08-25，20k）
- **改动**：`request_cost = 0.3 × (1 − clip(100/bw,…))`（低带宽重罚 -2.1、高带宽正激励 +0.2，跨零）。
- **结果**：SR 23.7%。带宽仍拉平；退化 STOP_REQUEST+CONTINUE_NO_REQUEST 两动作；buffer 反转（hi31%/lo0.4%）。
- **教训**：带宽成本放大不能塑造 bandwidth 签名（NE 主导 + 时间对齐混杂）。

## pilot4 — hunger + bw_symmetric 弱成本（2026-08-25，20k）
- **改动**：新增 hunger 需求侧 `+0.8×hunger(buff≤3)`（REQUES 奖/NO_不管罚），bw_symmetric 降到 0.05。
- **结果**：SR 20.9%（四线最低）。**buffer 签名成立且强**（lo 69.3% / mi 0.4% / hi 27.4%）；CONTINUE_REQUEST 复活（11,303 次）；带宽仍反向（低36.9%>高26.9%）。请求过冲 55.3 次/集（hunger 饱和）。
- **教训**：hunger 机制有效但 0.8 过冲；带宽被 hunger 饱和淹没 + 权重太弱。

## pilot5 — Step A：带宽决定 request（2026-08-26，40k，已训完 + 283/283 评估完成）
- **改动**（唯一实验变量 = 带宽权重）：
  - `--scheduler_request_hunger_weight 0.3`（0.8→0.3，去饱和，保留 buffer 需求）
  - `--scheduler_request_bw_symmetric True --scheduler_request_weight 0.2`（0.05→0.2，给带宽主导权）
  - `--scheduler_request_bw_weight 0.0`（不罚 STOP——用户否决普遍罚停，该停时该停）
  - 训练 40k（给弱信号二次机会）
- **判据**：① 固定 buffer 下高带宽 request 率 > 低带宽（pilot1 53%>44% 先例）② SR≥25% 底线。
- **结果（283/283 完整评估）**：SR **27.56%**（≈用户记忆「SR 27.5」），OSR 32.5%；action_counts = STOP_REQUEST 7162 / STOP_NO_REQUEST 0 / CONTINUE_REQUEST 0 / CONTINUE_NO_REQUEST 43093 —— CONTINUE 与 STOP_NO_REQUEST 再灭绝，坍缩两动作；带宽签名未复算。判据② SR≥25% 过、① 带宽签名未验证。
- **后续**：成功 → 论文 + 100k；失败 → 回 pilot1，带宽按不敏感陈述。Step B（drift/td 决定 stop）仅在 Step A 通过后进行。

## pilot6 — Step A2：规则门控 request + PPO 只学 motion（2026-08-27）—— **已被用户否决：不启用、从未训练**
- **思路（2026-08-27 提出，随后被用户否决）**：request 时机交给确定性规则——`request_window = (bw > 30Mbps) ∧ (buffer ≤ 2)` 时才允许 REQUEST；窗口外 REQUEST 被降级成 NO_REQUEST（保留 PPO 的 motion 选择）。带宽差∧buffer尽 → 端侧兜底（暂留最小实现）。PPO 不再学「该不该请求」，只学「该停还是继续飞」（motion）。
- **代码改动**（5090 已就位，COMPILE_OK）：
  - `src/common/param.py`：新增 `scheduler_rule_gate_request` / `scheduler_bw_window_thresh_mbps(30)` / `scheduler_buff_window_thresh(2)` / `scheduler_motion_stabilize_weight(0.5)` / `scheduler_motion_stabilize_thresh(1.0)`
  - `src/vlnce_src/drl_scheduler_env.py`：`_request_window()` 规则函数；step() 内 gate（L421，illegal override 的 empty-buffer 安全网不受 gate 影响）；`_compute_reward` 加 `motion_stabilize` shaping（misalign = drift/2.5 + td/5000，>thresh 时 STOP 奖励 / CONTINUE 惩罚）；reward_parts 加 motion_stabilize 记录
  - 原 hunger/bw_cost 分支保留但训练时权重归零（向后兼容）
- **训练参数**：`--scheduler_rule_gate_request True --scheduler_motion_stabilize_weight 0.5 --scheduler_request_weight 0.0 --scheduler_request_bw_symmetric False --scheduler_request_hunger_weight 0.0 --scheduler_total_timesteps 40000`
- **判据**：① request 只在窗口内出现（规则保证，评估验证 request_gated_out 计数）② STOP 的 misalign 显著高于 CONTINUE（PPO 学 motion）③ SR≥25%。
