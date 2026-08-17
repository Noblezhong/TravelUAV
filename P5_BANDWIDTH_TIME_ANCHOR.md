# P5: Time-Anchored Bandwidth Sampling (2026-08-16)

Protocol change: bandwidth is a function of (seq hash start, logical time),
NOT request ordinal / step count. Every paradigm observes the same bandwidth
at the same logical time in the same episode.

## New protocol
- `BandwidthTrace.bandwidth_at_ms(elapsed_ms)` (comm_delay.py): sample at
  `(hash_start + elapsed_ms // 1000) mod sample_count`, 1 sample / second,
  cycled. Pure function of time -- does NOT advance a cursor.
- Episode anchor: `reset_for_episode(seq_name)` (sha256 hash) at episode
  start, plus `t0` = logical clock time at episode start. Bandwidth =
  `bandwidth_at_ms(now_ms - t0)`.

## Coverage (all main-table evaluators)
| file | change |
|---|---|
| comm_delay.py | + bandwidth_at_ms |
| drl_scheduler_env.py | reset(): hash start + t0; _sample_bandwidth(): time-anchored (covers PPO + MATCH via inheritance) |
| eval.py (SG) | t0 at episode_clock creation; sample by clock time; assert single-episode batch |
| continue_eval.py (Cont) | planner t0 at construction; worker samples at submit-time logical time |
| rule_based_eval.py (RB) | + reset_for_episode per episode; planner t0; worker samples at submit-time |
| continue_tcm_eval.py | no change (reuses continue_eval planner/worker) |
| match_eval.py | no change (reuses DRLSchedulerEnv) |

NOT changed (outside main table): continue_eval_buffer_trim.py,
edge_dnn_jetson_eval.py -- still request-ordinal. Do NOT use them for
cross-paradigm comparisons.

## Boundary conditions (audited)
1. Start consistency: all 5 evaluators reset trace per episode from seq hash.
2. Single-episode batches: eval.py raises if batch_size > 1 (multi-batch
   would share one trace cursor); continue/rule_based/MATCH are structurally
   single-batch.
3. Clock semantics: unified FastEvalClock.now_ms (logical time under
   fast_eval, wall-clock elapsed otherwise); t0 reset every episode.
4. Overflow: modulo sample_count cycles; max(0, elapsed) clamps clock skew.
5. Request-time semantics: worker-based planners sample at submit-time
   logical time (timing_start = planner_started_logical_ms or
   submitted_logical_ms); step-based (PPO/MATCH) sample each step start.
6. Reproducibility: same seq -> same curve regardless of run/paradigm.
7. Train/eval distribution: PPO training uses the same env path -> same
   protocol as evaluation.

## Verification
- Unit: bandwidth_at_ms offsets / wrap / clamp / cross-paradigm same-time
  same-value / determinism -- ALL PASSED (llamauav env, 2004 samples).
- Smoke: SG / RB / MATCH single episode (seq 9a6e3349) on 5090 -- see
  /tmp/verify_p5_smoke.py results (logical-time vs recorded bandwidth).

## Smoke results (2026-08-16, seq 9a6e3349, fast_x10, 5090)

| paradigm | records | matched | verdict |
|---|---|---|---|
| SG (eval.py) | 67 | 67 | PASS |
| RB (rule_based_eval.py) | 536 | 536 | PASS |
| MATCH (match_eval.py) | 393 | 393 | PASS |

Method: scan offset k in [-60, +10]s around record time; every record's
bandwidth must equal bandwidth_at_ms(t_rec + k*1000). SG/RB offsets -1..-14s
(sample happens before uplink latency advance; 9.25MB payload at low
bandwidth => multi-second uplink), MATCH offsets 0..-1s (step-start sample,
same-step record). All three pass with zero tolerance (<1 bps).
