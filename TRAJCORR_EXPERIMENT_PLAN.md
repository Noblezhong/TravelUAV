# Trajectory Correction Experiment Plan — Active Handoff

**Updated:** 2026-08-05  
**4090 edge code:** `/HDD1/code/TravelUAV_eval_fix`  
**Jetson evaluator code:** `/home/zt/code/TravelUAV_eval_fix`  
**Frozen evaluator commit:** `3a516d39d798e46a90877f08ed226f5407c396d6`  
**Shared controller blob:** `087da90de51d132dac5de6d82a3cbd867385be38`

## Objective

Test correction claim C2 through the paired full-TC-ON minus TC-OFF difference on the same 4090+Jetson deployment. TC is one complete module: current-state trajectory regeneration, target lock, request handling, waypoint filtering and safe refresh behavior are evaluated together.

## Fixed boundary

- 1,418 episode manifest SHA256 `af5b7b660ed3b1dea95b53aa72a98f74ae7a37cd381aa0e818ee2a343cf54552`;
- same bandwidth trace/reset, edge VLM and Jetson trajectory-DNN weights;
- fixed x10 logical-time semantics; `applied_logical_ms >= ready_logical_ms` for every result;
- same shared controller, 1,000-waypoint budget, success/collision/NE and termination rules.

## Timing validation already passed

- corrected TC-OFF audit: zero early applications and zero logical-time monotonicity violations;
- post-controller smoke: 16 results, zero early applications and 12 positive real-RPC barrier waits;
- x1 is intentionally not run, therefore do not claim x10 and x1 are empirically equivalent;
- no new TC-ON pilot is needed: it traverses the same corrected timing path.

## Active run — do not modify

- **Corrected TC-OFF full, 1,418:** Jetson tmux `tc_off_full1418_unified_3a516d3`.
- Output: `/home/zt/traveluav_eval_shared/eval_trajcorr_off_unified_controller_full1418_20260803_fast_x10`.
- 4090 must keep AirSim and `tc_on_edge` running while Jetson evaluates.

## Immediately after TC-OFF finishes

1. Audit 1,418 unique terminal episodes, manifest/trace/controller hashes, terminal metrics, zero early applications, logical-time monotonicity, and edge-barrier fields.
2. Freeze this output as the corrected TC-OFF baseline.
3. Start **corrected TC-ON full, 1,418** with the same configuration and a new empty output directory. No small pilot is required.
4. Audit TC-ON identically, then produce paired TC-ON–OFF SR/OSR transitions, CR, NE, control steps, E2E, blocking stall, time shift and state drift.

## Paper reporting

The main table has no Deployment column. Place TC-OFF and corrected TC-ON in their own TC row section with navigation accuracy and system-performance column groups. Old TC-ON x10 is diagnostic only and must not fill the final TC-ON row. Raw `Avg T_dec` and `Avg T_action` stay out of the main table.

## Do not do

- Do not stop, restart or alter active TC-OFF.
- Do not overwrite the active result directory.
- Do not use old TC-ON as the causal partner of corrected TC-OFF.
- Do not run x1 or another TC-ON pilot before the required corrected TC-ON full run.
