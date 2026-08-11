# Trajectory Correction Experiment Plan — Current Handoff

**Updated:** 2026-08-03
**4090 evaluation code:** `/HDD1/code/TravelUAV_eval_fix`
**Jetson evaluation code:** `/home/zt/code/TravelUAV_eval_fix`
**Frozen commit:** `3a516d39d798e46a90877f08ed226f5407c396d6`
**Shared controller blob:** `087da90de51d132dac5de6d82a3cbd867385be38`

## Objective

Test C2 with the paired difference TC-ON minus TC-OFF on the same 4090+Jetson deployment. Stop, Continuous, and PPO may appear in the complete-system comparison, but they do not replace this paired module comparison.

## Frozen boundary

- 1,418 identical episodes; manifest SHA256 `af5b7b660ed3b1dea95b53aa72a98f74ae7a37cd381aa0e818ee2a343cf54552`;
- same trace and per-`seq_name` reset;
- fixed x10 logical-time semantics with `applied_logical_ms >= ready_logical_ms`;
- same edge VLM and Jetson trajectory-DNN weights;
- same controller blob, 1,000-waypoint budget, success/collision/NE, and termination rules.

## Validation already passed

- fixed-x10 TC-OFF 50-episode audit: zero early applications and zero logical-time monotonicity violations;
- post-controller smoke: 16 trajectory results, zero early applications, 12 positive RPC barrier waits;
- the TC-ON pilot was stopped by user decision because the common barrier path had already been validated;
- x1 is intentionally not run, so no x10-versus-x1 equivalence claim is allowed.

## Current state

- TC-OFF corrected full: RUNNING in Jetson tmux `tc_off_full1418_unified_3a516d3`.
  Output: `/home/zt/traveluav_eval_shared/eval_trajcorr_off_unified_controller_full1418_20260803_fast_x10`.
- TC-ON old x10: COMPLETE but diagnostic only because it predates the Fast Eval barrier fix.
- TC-ON corrected full: DEFERRED in the current round.

## Next task

1. Let corrected TC-OFF finish without changing runtime code.
2. Validate 1,418 unique episode ends, manifest/trace/controller hashes, zero early application, monotonic logical time, and barrier timing.
3. Freeze corrected TC-OFF as the asynchronous baseline.
4. Before making the final C2 claim, run same-version TC-ON on the same 1,418 episodes and build a paired report.
5. Report paired SR/OSR transitions, collision, NE, E2E, control steps, time shift, and state drift.

## Paper result contract

Primary columns: SR, OSR, CR, NE, control steps, episode E2E, time shift, and state drift.

Remove raw `Avg T_dec` and `Avg T_action` from the main table. The old TC `T_dec` used only uplink plus edge-VLM latency, while other evaluators included more components; raw action time also uses different event units. If needed, report observation+DINO, uplink, edge VLM, Jetson DNN, barrier wait, and action simulation time per waypoint in an appendix.

## Do not do now

- do not restart or modify the running TC-OFF evaluator;
- do not use old TC-ON as the paper-ready counterpart to corrected TC-OFF;
- do not claim fixed-x10 is empirically equivalent to x1;
- do not overwrite any result directory.
