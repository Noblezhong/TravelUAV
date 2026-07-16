#!/bin/bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$root_dir"

CUDA_VISIBLE_DEVICES=0 python -u "$root_dir/src/vlnce_src/edge_vlm_server.py" \
    --run_type eval \
    --name EdgeVLMServer \
    --edge_vlm_bind_host 0.0.0.0 \
    --edge_vlm_port 26000 \
    --model_path "$root_dir/Model/LLaMA-UAV/work_dirs/llama-uav-7b" \
    --model_base "$root_dir/Model/LLaMA-UAV/model_zoo/vicuna-7b-v1.5" \
    --vision_tower "$root_dir/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth" \
    --image_processor "$root_dir/Model/LLaMA-UAV/llamavid/processor/clip-patch14-224" \
    --traj_model_path "$root_dir/Model/LLaMA-UAV/work_dirs/traveluav-traj-model" \
    --object_name_json_path "/HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/object_description.json" \
    --groundingdino_config "$root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
    --groundingdino_model_path "$root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth"
