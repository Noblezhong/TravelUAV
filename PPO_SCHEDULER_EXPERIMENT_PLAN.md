# PPO Scheduler Experiment Plan — Active Handoff

**Updated:** 2026-08-05  
**Formal evaluator worktree:** `/code/TravelUAV_ppo_trace_eval`  
**PPO evaluator commit:** `7ec6420e7b74b6a667d1696499c4798b53296e30`  
**Shared controller blob:** `087da90de51d132dac5de6d82a3cbd867385be38`  
**Weights:** mounted from 4090 at `/code/TravelUAV/Model`

## Objective

Test scheduler claim C1 by the paired PPO minus Continuous w=5 difference on 5090. Stop-and-go is the completed shared waiting reference, not the PPO causal counterpart.

## Fixed boundary

- 1,418 episodes, manifest SHA256 `af5b7b660ed3b1dea95b53aa72a98f74ae7a37cd381aa0e818ee2a343cf54552`;
- fixed-x10 Fast Eval, 1,000-waypoint budget, unchanged success/OSR/collision/NE and termination rules;
- per-episode bandwidth trace reset by `seq_name` for PPO and the same trace/reset rule for the comparison;
- same mounted LLaMA-UAV, GroundingDINO and trajectory-DNN weights;
- shared controller blob above.

## Evidence already complete

- Stop shared-controller full run: complete, 1,418 episode ends.
- PPO trace-reset pilot: complete, five full episodes; every first observed bandwidth exactly matched the `seq_name` reset sample.
- Continuous w=5: existing full result is retained for the paired C1 comparison.

## Active run — do not modify

- **PPO trace-reset full, 1,418:** tmux `ppo_trace_full1418_7ec6420`.
- Output: `/code/TravelUAV_eval_runs/20260805/eval_ppo_trace_reset_full1418_20260805_fast_x10`.
- This is the formal PPO candidate. The old PPO output remains diagnostic only because it consumed one global trace instead of resetting per episode.

## When this run ends

1. Audit 1,418 unique terminal episodes, manifest, x10 mode, controller blob, terminal fields, and trace reset in the full log.
2. Produce paired PPO–Continuous statistics for SR, OSR, CR, NE, control steps, logical E2E, blocking stall, time shift and state drift.
3. Inspect PPO request behavior: request count, continue-request / overlap rate, in-flight duplicate requests, pending overwrites and stale-result discards.
4. Decide whether Rule-based is needed as an interpretable comparator. Do not start it automatically.
5. Open PPO redesign/retraining only if this analysis establishes that the present PPO cannot support C1. Do not change action feasibility or SMDP semantics during the current evaluation.

## Paper reporting

The main table has no Deployment column. Put Stop, Continuous, PPO and Rule-based in the Scheduler row section, with navigation accuracy (SR/OSR/CR/NE) and system performance (control steps/E2E/blocking stall/time shift/state drift) column groups. Rule-based stays blank until run. Raw `Avg T_dec` and `Avg T_action` do not belong in the main table.

## Do not do

- Do not stop, restart or modify the active PPO evaluator.
- Do not overwrite its output directory.
- Do not treat old PPO as final C1 evidence.
- Do not start Rule-based or PPO retraining before the audited PPO–Continuous report.
