#!/bin/bash
# pilot11 dev 列 = AHS+TCM（trajcorr on，NCN off）: 283 heldout fast_x10。
# worktree match_eval.py = legacy AHS+TCM evaluator（本分支无 NCN 代码），作为 dev/选型 proxy。
# 用法:  SCHEDULER_MODEL_PATH=<zip> EVAL_SAVE_PATH=... bash scripts/eval_pilot11_ahstcm_283.sh
# 默认权重 = 0901-20k（先跑它冒烟 + 出 pilot11 对比基线）。
# 依赖: 先起 AirSim ServerTool(25000): bash scripts/airsim_server_5090.sh
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

source /home/noble/miniforge3/etc/profile.d/conda.sh && conda activate llamauav

DEFAULT_WEIGHT=$root_dir/drl_train_pilot10_20k_0901-1507/scheduler_models/ppo_scheduler_20260901-150800-148080.zip
WEIGHT=${SCHEDULER_MODEL_PATH:-$DEFAULT_WEIGHT}
RUN_ID=$(date +%m%d-%H%M)
SAVE=${EVAL_SAVE_PATH:-$root_dir/eval_drl_pilot11_ahstcm_283_fast_x10_$RUN_ID}
EVALJSON=$root_dir/heldout_283_evalformat.json
LOG=${EVAL_LOG:-/tmp/pilot11_ahstcm_eval.log}

[ -f "$WEIGHT" ] || { echo "ERROR: ckpt missing: $WEIGHT"; exit 1; }
[ -f "$EVALJSON" ] || { echo "ERROR: eval json missing: $EVALJSON"; exit 1; }
echo "model: $WEIGHT"
echo "save:  $SAVE"

CUDA_VISIBLE_DEVICES=0 python -u "$root_dir/src/vlnce_src/match_eval.py" \
  --run_type eval --name TravelLLMPCMatchEvalPilot11Dev --gpu_id 0 \
  --simulator_tool_port 25000 --DDP_MASTER_PORT 80017 --batchSize 1 \
  --always_help True --use_gt True --max_control_steps 1000 --scheduler_max_steps 2000 \
  --max_episodes_per_scene 80 \
  --enable_comm_delay True --fast_eval True --fast_eval_speedup 10 \
  --comm_trace_csv_path "$root_dir/bandwidth/ucc4g_bandwidth_trace.csv" \
  --trajcorr_mode on --trajcorr_state_shift_threshold_m 2.5 \
  --scheduler_model_path "$WEIGHT" \
  --scheduler_time_weight 1.0 \
  --scheduler_oracle_success_reward 20.0 \
  --scheduler_illegal_action_penalty 5.0 \
  --scheduler_req_bw_thresh_mbps 25.0 \
  --scheduler_req_buf_thresh 3.0 \
  --scheduler_motion_cont_weight 0.3 \
  --scheduler_req_bw_weight 0.4 \
  --scheduler_req_buf_weight 0.2 \
  --scheduler_req_inflight_penalty 0.5 \
  --scheduler_motion_stop_noreq_penalty 0.5 \
  --dataset_path /HDD2/TravelUAV_dataset/TravelUAV_data/ \
  --eval_save_path "$SAVE" \
  --model_path "$root_dir/Model/LLaMA-UAV/work_dirs/llama-uav-7b" \
  --model_base "$root_dir/Model/LLaMA-UAV/model_zoo/vicuna-7b-v1.5" \
  --vision_tower "$root_dir/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth" \
  --image_processor "$root_dir/Model/LLaMA-UAV/llamavid/processor/clip-patch14-224" \
  --traj_model_path "$root_dir/Model/LLaMA-UAV/work_dirs/traveluav-traj-model" \
  --eval_json_path "$EVALJSON" \
  --map_spawn_area_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/map_spawnarea_info.json \
  --object_name_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/object_description.json \
  --groundingdino_config "$root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  --groundingdino_model_path "$root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth" \
  2>&1 | tee "$LOG"
rc=$?
done_eps=$(ls -d "$SAVE"/*/ 2>/dev/null | grep -vE "/profile_logs/" | wc -l)
echo "评估进程退出 rc=$rc, 完成集: $done_eps/283"
echo "日志: $LOG"
