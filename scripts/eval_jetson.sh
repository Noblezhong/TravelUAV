#!/bin/bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

CUDA_VISIBLE_DEVICES=0 python -u "$root_dir/src/vlnce_src/profile_eval_jetson.py" \
    --run_type eval \
    --name TravelLLMProfileJetson \
    --gpu_id 0 \
    --simulator_tool_port 25000 \
    --DDP_MASTER_PORT 80005 \
    --batchSize 1 \
    --always_help True \
    --use_gt True \
    --maxWaypoints 200 \
    --dataset_path /home/zt/traveluav_shared/data \
    --eval_save_path "$root_dir/eval_output_profile_jetson" \
    --model_path "$root_dir/Model/LLaMA-UAV/work_dirs/llama-uav-7b" \
    --model_base "$root_dir/Model/LLaMA-UAV/model_zoo/vicuna-7b-v1.5" \
    --vision_tower "$root_dir/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth" \
    --image_processor "$root_dir/Model/LLaMA-UAV/llamavid/processor/clip-patch14-224" \
    --traj_model_path "$root_dir/Model/LLaMA-UAV/work_dirs/traveluav-traj-model" \
    --eval_json_path /home/zt/traveluav_shared/data/data/uav_dataset/seen_valset.json \
    --map_spawn_area_json_path /home/zt/traveluav_shared/data/data/meta/map_spawnarea_info.json \
    --object_name_json_path /home/zt/traveluav_shared/data/data/meta/object_description.json \
    --groundingdino_config "$root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
    --groundingdino_model_path "$root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth"
