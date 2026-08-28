#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_id="${RUN_ID:-$(date +%m%d-%H%M)}"
simulator_port="${SIMULATOR_PORT:-25000}"
ddp_port="${DDP_MASTER_PORT:-20015}"
max_waypoints="${MAX_WAYPOINTS:-200}"
eval_json_path="${EVAL_JSON_PATH:-/HDD2/TravelUAV_dataset/TravelUAV_data/data/uav_dataset/seen_valset.json}"
eval_save_path="${EVAL_SAVE_PATH:-${root_dir}/eval_aerodpo_stopgo_${run_id}}"
aerodpo_lora_path="${AERODPO_LORA_PATH:-/HDD1/code/AeroDPO/model_zoo/AeroDPO-2B-384-LoRA}"
aerodpo_base_model_path="${AERODPO_BASE_MODEL_PATH:-/HDD1/code/AeroDPO/model_zoo/AeroVLA-2B-384-Merged}"

source /home/zt/miniconda3/etc/profile.d/conda.sh
conda activate aerodpo

cd "${root_dir}"
export PYTHONPATH="${root_dir}${PYTHONPATH:+:${PYTHONPATH}}"
CUDA_VISIBLE_DEVICES=0 python -u src/vlnce_src/aerodpo_stopgo_eval.py \
    --run_type eval \
    --name AeroDPOStopGo \
    --gpu_id 0 \
    --simulator_tool_port "${simulator_port}" \
    --DDP_MASTER_PORT "${ddp_port}" \
    --batchSize 1 \
    --always_help False \
    --use_gt False \
    --maxWaypoints "${max_waypoints}" \
    --enable_comm_delay True \
    --fast_eval True \
    --fast_eval_speedup 10 \
    --max_episodes_per_scene 80 \
    --comm_trace_csv_path "${root_dir}/bandwidth/ucc4g_bandwidth_trace.csv" \
    --dataset_path /HDD2/TravelUAV_dataset/TravelUAV_data/ \
    --eval_save_path "${eval_save_path}" \
    --model_path "${aerodpo_lora_path}" \
    --base_model_path "${aerodpo_base_model_path}" \
    --eval_json_path "${eval_json_path}" \
    --map_spawn_area_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/map_spawnarea_info.json \
    --object_name_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/object_description.json \
    "$@"
