#!/bin/bash

root_dir=.
model_dir=$root_dir/Model/LLaMA-UAV
enable_comm_delay=True
run_ts=$(date +%m%d-%H%M)

CUDA_VISIBLE_DEVICES=0 python -u $root_dir/src/vlnce_src/drl_scheduler_train.py \
    --run_type eval \
    --name TravelLLMPCDRLSchedulerTrain \
    --gpu_id 0 \
    --simulator_tool_port 25000 \
    --DDP_MASTER_PORT 80006 \
    --batchSize 1 \
    --always_help True \
    --use_gt True \
    --maxWaypoints 200 \
    --max_episodes_per_scene 80 \
    --enable_comm_delay $enable_comm_delay \
    --comm_trace_csv_path $root_dir/bandwidth/ucc4g_bandwidth_trace.csv \
    --scheduler_total_episodes 639 \
    --scheduler_total_timesteps 100000 \
    --scheduler_max_steps 800 \
    --scheduler_gamma 0.999 \
    --scheduler_n_steps 128 \
    --scheduler_ne_norm_m 1.0 \
    --scheduler_ne_progress_weight 1.0 \
    --scheduler_time_weight 0.05 \
    --scheduler_drift_weight 0.5 \
    --scheduler_time_drift_norm_ms 5000 \
    --scheduler_time_drift_weight 0.5 \
    --scheduler_request_weight 0.0 \
    --scheduler_idle_wait_ms 500 \
    --scheduler_success_reward 40.0 \
    --scheduler_oracle_success_reward 5.0 \
    --scheduler_illegal_action_penalty 2.0 \
    --scheduler_collision_penalty 20.0 \
    --scheduler_failure_penalty 10.0 \
    --scheduler_batch_size 32 \
    --scheduler_learning_rate 0.0003 \
    --dataset_path /HDD2/TravelUAV_dataset/TravelUAV_data/ \
    --eval_save_path /code/TravelUAV/drl_train_$run_ts \
    --model_path $model_dir/work_dirs/llama-uav-7b \
    --model_base $model_dir/model_zoo/vicuna-7b-v1.5 \
    --vision_tower $model_dir/model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor $model_dir/llamavid/processor/clip-patch14-224 \
    --traj_model_path $model_dir/work_dirs/traveluav-traj-model \
    --eval_json_path /code/TravelUAV/drl_scheduler_seen_curriculum.json \
    --map_spawn_area_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/map_spawnarea_info.json \
    --object_name_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/object_description.json \
    --groundingdino_config $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --groundingdino_model_path $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth \
    "$@"
