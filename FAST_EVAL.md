# Fast Eval

Fast Eval accelerates AirSim while keeping an original-speed experiment clock for communication, inference, control, drift, and episode metrics.

Start the updated simulator server normally; the evaluation process sends the requested clock speed when it opens a scene:

```bash
python airsim_plugin/AirVLNSimulatorServerTool.py \
  --port 25000 \
  --clock_speed 1 \
  --root_path /HDD2/AeroDuo_envs
```

The server-level `--clock_speed` remains the normal-mode default. Fast Eval overrides it per scene.

Run an evaluator with the default 5x speed:

```bash
bash scripts/eval.sh --fast_eval True
bash scripts/continue_eval.sh --fast_eval True
bash scripts/hybrid_eval.sh --fast_eval True
bash scripts/drl_scheduler_eval.sh /path/to/ppo_scheduler.zip --fast_eval True
```

Use a conservative speed when validating:

```bash
bash scripts/continue_eval.sh --fast_eval True --fast_eval_speedup 2
```

Fast outputs are isolated automatically with `_fast_x5` or `_fast_x2`. The directory contains `fast_eval_manifest.json`; a mismatched mode or speed cannot resume into it.

Before a full run, compare normal and Fast JSONL profiles from the same episode subset:

```bash
python scripts/compare_fast_eval.py \
  --normal_log /path/to/normal.jsonl \
  --fast_log /path/to/fast.jsonl \
  --output_json /path/to/fast_eval_comparison.json
```

The comparison fails when outcome mismatches, request application steps, metrics, or wall-time speedup exceed the acceptance limits. Do not use a speed for the full evaluation until this check passes.
