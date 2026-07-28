#!/bin/bash

root_dir=.
enable_comm_delay=True

CUDA_VISIBLE_DEVICES=0 python -u $root_dir/src/vlnce_src/drl_scheduler_eval.py \
    --run_type eval \
    --name TravelLLMPCDRLSchedulerEval \
    --gpu_id 0 \
    --simulator_tool_port 25000 \
    --DDP_MASTER_PORT 80007 \
    --batchSize 1 \
    --always_help True \
    --use_gt True \
    --max_control_steps 1000 \
    --scheduler_max_steps 2000 \
    --enable_comm_delay $enable_comm_delay \
    --fast_eval True \
    --fast_eval_speedup 10 \
    --comm_trace_csv_path $root_dir/bandwidth/ucc4g_bandwidth_trace.csv \
    --scheduler_model_path $root_dir/drl_train_0722-1254/scheduler_models/ppo_scheduler_20260722-125409-242489.zip \
    --dataset_path /HDD2/TravelUAV_dataset/TravelUAV_data/ \
    --eval_save_path /code/TravelUAV/eval_drl_$(date +%m%d-%H%M)_fast_x10 \
    --model_path "$root_dir/Model/LLaMA-UAV/work_dirs/llama-uav-7b" \
    --model_base "$root_dir/Model/LLaMA-UAV/model_zoo/vicuna-7b-v1.5" \
    --vision_tower "$root_dir/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth" \
    --image_processor "$root_dir/Model/LLaMA-UAV/llamavid/processor/clip-patch14-224" \
    --traj_model_path "$root_dir/Model/LLaMA-UAV/work_dirs/traveluav-traj-model" \
    --eval_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/uav_dataset/seen_valset.json \
    --map_spawn_area_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/map_spawnarea_info.json \
    --object_name_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/object_description.json \
    --groundingdino_config $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --groundingdino_model_path $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth \
    "$@"
