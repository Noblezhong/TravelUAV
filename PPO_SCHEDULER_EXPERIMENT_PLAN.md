# PPO Scheduler Experiment Plan — Current Handoff

**Updated:** 2026-08-03
**Current evaluation code:** `/code/TravelUAV_eval_fix`
**Frozen commit:** `3a516d39d798e46a90877f08ed226f5407c396d6`
**Shared controller blob:** `087da90de51d132dac5de6d82a3cbd867385be38`

## Objective

Test C1 with the paired difference PPO minus Continuous w=5 on the 5090 deployment. The new Stop-and-go run is the shared synchronous reference used to locate both systems on the accuracy--latency Pareto plot.

## Frozen boundary

- 1,418 identical episodes; manifest SHA256 `af5b7b660ed3b1dea95b53aa72a98f74ae7a37cd381aa0e818ee2a343cf54552`;
- same trace and per-`seq_name` reset;
- fixed x10 Fast Eval semantics;
- same LLaMA-UAV, GroundingDINO, and trajectory-DNN weights;
- same low-level controller blob and 1,000-waypoint budget;
- same success, OSR, collision, NE, and termination rules.

## Current state

- Stop-and-go: RUNNING in tmux `stop_full1418_unified_3a516d3`.
  Output: `/code/TravelUAV_eval_runs/20260803/eval_stop_unified_controller_full1418_20260803_fast_x10`.
- Continuous w=5: COMPLETE and reused.
  Output: `/code/TravelUAV/eval_continuous_w5_0729-215750_fast_x10`.
- PPO: COMPLETE and reused.
  Output: `/code/TravelUAV/eval_drl_0723-1845_fast_x10`.
- Rule-based: conditional; do not run before the PPO--Continuous decision gate.

## Next task

1. Let Stop finish without changing runtime code.
2. Validate 1,418 unique episode ends, manifest/trace/controller hashes, and logical E2E.
3. Run paired PPO--Continuous analysis for SR, OSR, collision, NE, E2E, control steps, time shift, and state drift.
4. Use Stop as the shared system reference.
5. Only if PPO is dominated or behaves like Stop, open the retraining gate for single-flight action feasibility and SMDP discount/state/reward revisions.

## Paper result contract

Primary columns: SR, OSR, CR, NE, control steps, episode E2E, time shift, and state drift.

Remove raw `Avg T_dec` and `Avg T_action` from the main table. `T_dec` is not component-aligned with TC, and Stop `T_action` aggregates five waypoint calls while asynchronous evaluators record one. If needed, use an appendix component breakdown and per-waypoint action time.

## Do not do now

- do not rerun Continuous or PPO merely because Stop is running;
- do not modify PPO or start retraining before paired analysis;
- do not add an empty Rule-based row to the final table;
- do not overwrite any result directory.
