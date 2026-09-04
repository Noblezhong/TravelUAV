#!/usr/bin/env bash
# Continuous-only + TCM (Table V) local evaluator.
#
# All defaults mirror scripts/continue_eval.sh.  Override any *_PATH variable,
# GPU_ID, CUDA_VISIBLE_DEVICES, SIMULATOR_TOOL_PORT, or EVAL_SAVE_PATH to run
# the same code on another host without editing this file.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${TRAVELUAV_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
SIMULATOR_TOOL_PORT="${SIMULATOR_TOOL_PORT:-25000}"
DDP_MASTER_PORT="${DDP_MASTER_PORT:-80005}"
RUN_ID="${RUN_ID:-$(date +%m%d-%H%M)}"

DATASET_PATH="${DATASET_PATH:-/HDD2/TravelUAV_dataset/TravelUAV_data}"
EVAL_JSON_PATH="${EVAL_JSON_PATH:-${DATASET_PATH}/data/uav_dataset/seen_valset.json}"
MAP_SPAWN_AREA_JSON_PATH="${MAP_SPAWN_AREA_JSON_PATH:-${DATASET_PATH}/data/meta/map_spawnarea_info.json}"
OBJECT_NAME_JSON_PATH="${OBJECT_NAME_JSON_PATH:-${DATASET_PATH}/data/meta/object_description.json}"
COMM_TRACE_CSV_PATH="${COMM_TRACE_CSV_PATH:-${ROOT_DIR}/bandwidth/ucc4g_bandwidth_trace.csv}"
EVAL_SAVE_PATH="${EVAL_SAVE_PATH:-${ROOT_DIR}/eval_continuous_tcm_${RUN_ID}_fast_x10}"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/Model/LLaMA-UAV/work_dirs/llama-uav-7b}"
MODEL_BASE="${MODEL_BASE:-${ROOT_DIR}/Model/LLaMA-UAV/model_zoo/vicuna-7b-v1.5}"
VISION_TOWER="${VISION_TOWER:-${ROOT_DIR}/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth}"
IMAGE_PROCESSOR="${IMAGE_PROCESSOR:-${ROOT_DIR}/Model/LLaMA-UAV/llamavid/processor/clip-patch14-224}"
TRAJ_MODEL_PATH="${TRAJ_MODEL_PATH:-${ROOT_DIR}/Model/LLaMA-UAV/work_dirs/traveluav-traj-model}"
GROUNDINGDINO_CONFIG="${GROUNDINGDINO_CONFIG:-${ROOT_DIR}/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
GROUNDINGDINO_MODEL_PATH="${GROUNDINGDINO_MODEL_PATH:-${ROOT_DIR}/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth}"

# Inference does not use bitsandbytes.  Older host installations can otherwise
# auto-load a CUDA-11-only bitsandbytes build before evaluation starts.
export NCN_DISABLE_BITSANDBYTES="${NCN_DISABLE_BITSANDBYTES:-1}"
export PYTHONPATH="${ROOT_DIR}/tools/ncn_python_shim${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT_DIR}"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" -u src/vlnce_src/continue_tcm_eval.py \
    --run_type eval \
    --name TravelLLMPCChunkWPTCM \
    --gpu_id "${GPU_ID}" \
    --simulator_tool_port "${SIMULATOR_TOOL_PORT}" \
    --DDP_MASTER_PORT "${DDP_MASTER_PORT}" \
    --batchSize 1 \
    --always_help True \
    --use_gt True \
    --max_control_steps 1000 \
    --max_episodes_per_scene 80 \
    --enable_comm_delay True \
    --fast_eval True \
    --fast_eval_speedup 10 \
    --comm_trace_csv_path "${COMM_TRACE_CSV_PATH}" \
    --chunk_waypoints 5 \
    --trajcorr_mode on \
    --trajcorr_state_shift_threshold_m 2.5 \
    --dataset_path "${DATASET_PATH}" \
    --eval_save_path "${EVAL_SAVE_PATH}" \
    --model_path "${MODEL_PATH}" \
    --model_base "${MODEL_BASE}" \
    --vision_tower "${VISION_TOWER}" \
    --image_processor "${IMAGE_PROCESSOR}" \
    --traj_model_path "${TRAJ_MODEL_PATH}" \
    --eval_json_path "${EVAL_JSON_PATH}" \
    --map_spawn_area_json_path "${MAP_SPAWN_AREA_JSON_PATH}" \
    --object_name_json_path "${OBJECT_NAME_JSON_PATH}" \
    --groundingdino_config "${GROUNDINGDINO_CONFIG}" \
    --groundingdino_model_path "${GROUNDINGDINO_MODEL_PATH}" \
    "$@"
