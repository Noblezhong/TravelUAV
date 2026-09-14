#!/usr/bin/env bash
# pilot11 (0909) 30k → 主表 1418 消融行（continuous-only + AHS + TCM，无 NCN）
# 与 4090 的 w/ AHS 行同批换版：唯一变量是 checkpoint 由 0901-20k 换为 0909-30k。
# 边界沿用兄弟行：seen_valset(1418)、fast_eval x10、comm_delay、chunk5、
#   trajcorr on / 2.5m、--enable_ncn False、max_control_steps 1000、scheduler_max_steps 2000。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export NCN_CONDA_ENV=llamauav
export CONDA_SH=/home/noble/miniforge3/etc/profile.d/conda.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SCHEDULER_MODEL_PATH="$ROOT_DIR/drl_train_pilot11_0909-2109/scheduler_models/ppo_scheduler_20260909-210915-662188.zip"
export EVAL_SAVE_PATH="$ROOT_DIR/eval_match_edge_only_pilot11_0909_seen1418_fast_x10"
export DDP_MASTER_PORT=80018

echo "model: $SCHEDULER_MODEL_PATH"
echo "save:  $EVAL_SAVE_PATH"

exec bash "$ROOT_DIR/scripts/match_edge_only_eval.sh" "$@"
