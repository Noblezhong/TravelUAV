#!/bin/bash

set -euo pipefail

RUN_ID="${RUN_ID:-$(date +%m%d-%H%M)}"
TRAJCORR_MODE="${TRAJCORR_MODE:-on}"
EVAL_JSON_PATH="${EVAL_JSON_PATH:-/home/zt/traveluav_shared/data/data/uav_dataset/seen_valset.json}"
EVAL_SAVE_ROOT="${EVAL_SAVE_ROOT:-/home/zt/traveluav_eval_shared}"
if [[ "$TRAJCORR_MODE" != "off" && "$TRAJCORR_MODE" != "on" ]]; then
    echo "TRAJCORR_MODE must be off or on" >&2
    exit 2
fi
if ! mountpoint -q "$EVAL_SAVE_ROOT"; then
    echo "Evaluation output directory is not mounted: $EVAL_SAVE_ROOT" >&2
    exit 2
fi

PYTHONPATH=/home/zt/code/TravelUAV CUDA_VISIBLE_DEVICES=0 python -u /home/zt/code/TravelUAV/src/vlnce_src/edge_dnn_jetson_eval.py \
    --run_type eval \
    --name "TrajCorr${TRAJCORR_MODE}" \
    --gpu_id 0 \
    --simulator_tool_host 192.168.105.17 \
    --simulator_tool_port 25000 \
    --DDP_MASTER_PORT 80005 \
    --batchSize 1 \
    --always_help True \
    --use_gt True \
    --max_control_steps 1000 \
    --edge_vlm_host 192.168.105.17 \
    --edge_vlm_port 26000 \
    --chunk_waypoints 5 \
    --trajcorr_mode "$TRAJCORR_MODE" \
    --trajcorr_state_shift_threshold_m 2.5 \
    --enable_comm_delay True \
    --fast_eval True \
    --fast_eval_speedup 10 \
    --max_episodes_per_scene 100 \
    --comm_trace_csv_path /home/zt/code/TravelUAV/bandwidth/ucc4g_bandwidth_trace.csv \
    --dataset_path /home/zt/traveluav_shared/data \
    --eval_save_path "${EVAL_SAVE_ROOT}/eval_trajcorr_${TRAJCORR_MODE}_${RUN_ID}" \
    --model_path /home/zt/code/TravelUAV/Model/LLaMA-UAV/work_dirs/llama-uav-7b \
    --model_base /home/zt/code/TravelUAV/Model/LLaMA-UAV/model_zoo/vicuna-7b-v1.5 \
    --vision_tower /home/zt/code/TravelUAV/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor /home/zt/code/TravelUAV/Model/LLaMA-UAV/llamavid/processor/clip-patch14-224 \
    --traj_model_path /home/zt/code/TravelUAV/Model/LLaMA-UAV/work_dirs/traveluav-traj-model \
    --eval_json_path "$EVAL_JSON_PATH" \
    --map_spawn_area_json_path /home/zt/traveluav_shared/data/data/meta/map_spawnarea_info.json \
    --object_name_json_path /home/zt/traveluav_shared/data/data/meta/object_description.json \
    --groundingdino_config /home/zt/code/TravelUAV/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --groundingdino_model_path /home/zt/code/TravelUAV/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth
