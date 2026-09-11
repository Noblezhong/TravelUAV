#!/bin/bash
# pilot11 微调: Δ 整体退出 reward 梯度 + req bw 项有界，热启动 0901-20k。
#   r_task:   −elapsed/5000 + (−5×illegal) + terminal{+40 SR / +20 OSR / −20 coll / −10 fail}   [unchanged]
#   r_req:    req·[ w_bw·clip((bw−25)/25, −1, +1) − w_buf·(buffer−3) ] − w_inflight·req·inflight   [bounded bw, pilot11]
#   r_motion: CONTINUE → +w_cont（flat）; STOP+REQUEST / STOP+inflight → 0; STOP+dry-hover → −0.5   [Δ removed, pilot11]
#   state = [buffer/7, bw/1e8, inflight, time_drift/5000, drift/2.5]  (5-dim, unchanged — drift stays as obs input)
# 目标: 冻结消失 + 请求率按 buffer×bw 自适应（行为门，不压 Δ）。
#
# Warm-start 机制（陷阱已核实）:
#   - 该目录被续训到 100k，原地 resume 会 glob 到 100k 漂移版 → 必须在新 run 目录硬链 0901 zip
#     为唯一的 ppo_checkpoint_20000_steps.zip。
#   - SB3 num_timesteps 从 20000 起算（load 保留），StopOnEpisodeCount 用绝对 num_timesteps≥total_timesteps
#     截停 → 微调上限设 --scheduler_total_timesteps 30000；episode 计数每进程归零，
#     --scheduler_total_episodes 150 = 干净的主停条件。
# 依赖: 先起 AirSim ServerTool(25000): bash scripts/airsim_server_5090.sh
set -euo pipefail
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"
set +u
source /home/noble/miniforge3/etc/profile.d/conda.sh
conda activate llamauav
set -u

BASE_CKPT=$root_dir/drl_train_pilot10_20k_0901-1507/scheduler_models/ppo_scheduler_20260901-150800-148080.zip
run_ts=$(date +%m%d-%H%M)
SAVE=$root_dir/drl_train_pilot11_$run_ts
MODEL_DIR=$SAVE/scheduler_models
mkdir -p "$MODEL_DIR"

if [ -n "$(ls "$MODEL_DIR"/*.zip 2>/dev/null)" ]; then
  echo "ERROR: $MODEL_DIR 非空，拒绝覆盖既有 checkpoint。"; exit 1
fi
ln "$BASE_CKPT" "$MODEL_DIR/ppo_checkpoint_20000_steps.zip"
echo "warm-start from: $BASE_CKPT"
echo "hardlinked as:   $MODEL_DIR/ppo_checkpoint_20000_steps.zip"
echo "save:            $SAVE"

CUDA_VISIBLE_DEVICES=0 python -u "$root_dir/src/vlnce_src/drl_scheduler_train.py" \
    --run_type eval \
    --name TravelLLMPCDRLSchedulerTrainPilot11 \
    --gpu_id 0 \
    --simulator_tool_port 25000 \
    --DDP_MASTER_PORT 80018 \
    --batchSize 1 \
    --always_help True \
    --use_gt True \
    --max_control_steps 1000 \
    --max_episodes_per_scene 80 \
    --enable_comm_delay True \
    --fast_eval True \
    --fast_eval_speedup 10 \
    --comm_trace_csv_path $root_dir/bandwidth/ucc4g_bandwidth_trace.csv \
    --scheduler_total_episodes 150 \
    --scheduler_total_timesteps 30000 \
    --scheduler_max_steps 2000 \
    --scheduler_gamma 0.995 \
    --scheduler_n_steps 128 \
    --scheduler_time_weight 1.0 \
    --scheduler_idle_wait_ms 500 \
    --scheduler_success_reward 40.0 \
    --scheduler_oracle_success_reward 20.0 \
    --scheduler_illegal_action_penalty 5.0 \
    --scheduler_collision_penalty 20.0 \
    --scheduler_failure_penalty 10.0 \
    --scheduler_batch_size 32 \
    --scheduler_learning_rate 0.0003 \
    --scheduler_train_bw_iid True \
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
    --eval_json_path "$root_dir/drl_scheduler_seen_curriculum.json" \
    --map_spawn_area_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/map_spawnarea_info.json \
    --object_name_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/object_description.json \
    --groundingdino_config "$root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
    --groundingdino_model_path "$root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth" \
    2>&1 | tee /tmp/pilot11_train_$run_ts.log
