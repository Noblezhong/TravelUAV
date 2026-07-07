#!/bin/bash

root_dir=.
model_dir=$root_dir/Model/LLaMA-UAV
enable_comm_delay=True
scheduler_model_path="$1"
if [ -z "$scheduler_model_path" ]; then
    echo "Usage: bash scripts/drl_scheduler_eval.sh /path/to/ppo_scheduler.zip [extra args]"
    exit 1
fi
shift

CUDA_VISIBLE_DEVICES=0 python -u $root_dir/src/vlnce_src/drl_scheduler_eval.py \
    --run_type eval \
    --name TravelLLMPCDRLSchedulerEval \
    --gpu_id 0 \
    --simulator_tool_port 25000 \
    --DDP_MASTER_PORT 80007 \
    --batchSize 1 \
    --always_help True \
    --use_gt True \
    --maxWaypoints 200 \
    --scheduler_max_steps 800 \
    --enable_comm_delay $enable_comm_delay \
    --comm_trace_csv_path $root_dir/bandwidth/ucc4g_bandwidth_trace.csv \
    --scheduler_model_path "$scheduler_model_path" \
    --dataset_path /HDD2/TravelUAV_dataset/TravelUAV_data/ \
    --eval_save_path /HDD1/code/TravelUAV/eval_drl_scheduler_com \
    --model_path $model_dir/work_dirs/llama-uav-7b \
    --model_base $model_dir/model_zoo/vicuna-7b-v1.5 \
    --vision_tower $model_dir/model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor $model_dir/llamavid/processor/clip-patch14-224 \
    --traj_model_path $model_dir/work_dirs/traveluav-traj-model \
    --eval_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/uav_dataset/seen_valset.json \
    --map_spawn_area_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/map_spawnarea_info.json \
    --object_name_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/object_description.json \
    --groundingdino_config $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --groundingdino_model_path $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth \
    "$@"
