#!/bin/bash
# pilot9 (0902) todo#4: continuous reward + async request, 20k train.
#   r_task:   −elapsed/5000 + (−5×illegal) + terminal{+40 success / +20 OSR / −20 coll / −10 fail}   [unchanged from pilot7]
#   r_req:    req·[ w_bw·(bw−25)/25 − w_buf·(buffer−3) ] − w_inflight·req·inflight
#   r_motion: w_motion·(drift−2.5)·(stop?+1:−1)
#   state = [buffer/7, bw/1e8, inflight, time_drift/5000, drift/2.5]  (5-dim, unchanged)
#   STOP_REQUEST is now async (no wait_result), so STOP_NO_REQUEST (wait-in-place) is expressible.
root_dir=.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source /home/noble/miniforge3/etc/profile.d/conda.sh && conda activate llamauav
enable_comm_delay=True
run_ts=$(date +%m%d-%H%M)

CUDA_VISIBLE_DEVICES=0 python -u $root_dir/src/vlnce_src/drl_scheduler_train.py \
    --run_type eval \
    --name TravelLLMPCDRLSchedulerTrainPilot9 \
    --gpu_id 0 \
    --simulator_tool_port 25000 \
    --DDP_MASTER_PORT 80007 \
    --batchSize 1 \
    --always_help True \
    --use_gt True \
    --max_control_steps 1000 \
    --max_episodes_per_scene 80 \
    --enable_comm_delay $enable_comm_delay \
    --fast_eval True \
    --fast_eval_speedup 10 \
    --comm_trace_csv_path $root_dir/bandwidth/ucc4g_bandwidth_trace.csv \
    --scheduler_total_episodes 639 \
    --scheduler_total_timesteps 20000 \
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
    --scheduler_motion_drift_thresh_m 2.5 \
    --scheduler_motion_cont_weight 0.4 \
    --scheduler_req_bw_weight 0.6 \
    --scheduler_req_buf_weight 0.15 \
    --scheduler_req_inflight_penalty 0.5 \
    --dataset_path /HDD2/TravelUAV_dataset/TravelUAV_data/ \
    --eval_save_path $REPO_ROOT/drl_train_pilot9_20k_$run_ts \
    --model_path "$root_dir/Model/LLaMA-UAV/work_dirs/llama-uav-7b" \
    --model_base "$root_dir/Model/LLaMA-UAV/model_zoo/vicuna-7b-v1.5" \
    --vision_tower "$root_dir/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth" \
    --image_processor "$root_dir/Model/LLaMA-UAV/llamavid/processor/clip-patch14-224" \
    --traj_model_path "$root_dir/Model/LLaMA-UAV/work_dirs/traveluav-traj-model" \
    --eval_json_path $REPO_ROOT/drl_scheduler_seen_curriculum.json \
    --map_spawn_area_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/map_spawnarea_info.json \
    --object_name_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/object_description.json \
    --groundingdino_config $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --groundingdino_model_path $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth \
    "$@"
