#!/bin/bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${JETSON_MODEL_DIR:-$root_dir/Model/LLaMA-UAV}"
dataset_path="${JETSON_DATASET_PATH:-$root_dir/data/TravelUAV_data/}"
eval_save_path="${JETSON_EVAL_SAVE_PATH:-$root_dir/eval_output_profile_jetson}"
eval_json_path="${JETSON_EVAL_JSON_PATH:-$dataset_path/data/uav_dataset/seen_valset.json}"
map_spawn_area_json_path="${JETSON_MAP_SPAWN_AREA_JSON_PATH:-$dataset_path/data/meta/map_spawnarea_info.json}"
object_name_json_path="${JETSON_OBJECT_NAME_JSON_PATH:-$dataset_path/data/meta/object_description.json}"
groundingdino_config="${JETSON_GROUNDINGDINO_CONFIG:-$root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
groundingdino_model_path="${JETSON_GROUNDINGDINO_MODEL_PATH:-$root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth}"

mkdir -p "$eval_save_path"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u "$root_dir/src/vlnce_src/profile_eval_jetson.py" \
    --run_type eval \
    --name TravelLLMProfileJetson \
    --gpu_id 0 \
    --simulator_tool_port 25000 \
    --DDP_MASTER_PORT 80005 \
    --batchSize 1 \
    --always_help True \
    --use_gt True \
    --maxWaypoints "${MAX_WAYPOINTS:-200}" \
    --dataset_path "$dataset_path" \
    --eval_save_path "$eval_save_path" \
    --model_path "$model_dir/work_dirs/llama-uav-7b" \
    --model_base "$model_dir/model_zoo/vicuna-7b-v1.5" \
    --vision_tower "$model_dir/model_zoo/LAVIS/eva_vit_g.pth" \
    --image_processor "$model_dir/llamavid/processor/clip-patch14-224" \
    --traj_model_path "$model_dir/work_dirs/traveluav-traj-model" \
    --eval_json_path "$eval_json_path" \
    --map_spawn_area_json_path "$map_spawn_area_json_path" \
    --object_name_json_path "$object_name_json_path" \
    --groundingdino_config "$groundingdino_config" \
    --groundingdino_model_path "$groundingdino_model_path"
