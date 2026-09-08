#!/usr/bin/env bash
# Final MATCH: AHS + TCM + local Q8 AeroDPO NCN.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${TRAVELUAV_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ARTIFACT_DIR="${AERODPO_GGUF_DIR:?Set AERODPO_GGUF_DIR to the external AeroDPO GGUF directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
SIMULATOR_TOOL_PORT="${SIMULATOR_TOOL_PORT:-25000}"
DDP_MASTER_PORT="${DDP_MASTER_PORT:-80008}"
RUN_ID="${RUN_ID:-$(date +%m%d-%H%M)}"

DATASET_PATH="${DATASET_PATH:-/HDD2/TravelUAV_dataset/TravelUAV_data}"
EVAL_JSON_PATH="${EVAL_JSON_PATH:-${DATASET_PATH}/data/uav_dataset/seen_valset.json}"
EVAL_SAVE_PATH="${EVAL_SAVE_PATH:-${ROOT_DIR}/eval_match_ncn_${RUN_ID}_fast_x10}"
SCHEDULER_MODEL_PATH="${SCHEDULER_MODEL_PATH:-${ROOT_DIR}/drl_train_0722-1254/scheduler_models/ppo_scheduler_20260722-125409-242489.zip}"

export NCN_DISABLE_BITSANDBYTES="${NCN_DISABLE_BITSANDBYTES:-1}"
export PYTHONPATH="${ROOT_DIR}/tools/ncn_python_shim${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${NCN_CONDA_ENV:-}" ]]; then
  # Optional, so hosts that already activated their environment keep working.
  source "${CONDA_SH:-/home/zt/miniconda3/etc/profile.d/conda.sh}"
  conda activate "${NCN_CONDA_ENV}"
fi
cd "${ROOT_DIR}"

exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" -u src/vlnce_src/match_eval.py \
  --run_type eval --name MATCH --gpu_id "${GPU_ID}" --simulator_tool_port "${SIMULATOR_TOOL_PORT}" \
  --DDP_MASTER_PORT "${DDP_MASTER_PORT}" --batchSize 1 --always_help True --use_gt True \
  --max_control_steps 1000 --scheduler_max_steps 2000 --max_episodes_per_scene 80 \
  --enable_comm_delay True --fast_eval True --fast_eval_speedup 10 --chunk_waypoints 5 \
  --trajcorr_mode on --trajcorr_state_shift_threshold_m 2.5 --enable_ncn True \
  --ncn_max_consecutive_actions "${NCN_MAX_ACTIONS:-200}" \
  --ncn_response_safety_margin_ms "${NCN_SAFETY_MARGIN_MS:-0}" \
  --ncn_edge_timeout_ms "${NCN_EDGE_TIMEOUT_MS:-0}" \
  --ncn_outage_duration_ms "${NCN_OUTAGE_DURATION_MS:-0}" \
  --comm_trace_csv_path "${COMM_TRACE_CSV_PATH:-${ROOT_DIR}/bandwidth/ucc4g_bandwidth_trace.csv}" \
  --scheduler_model_path "${SCHEDULER_MODEL_PATH}" \
  --dataset_path "${DATASET_PATH}" --eval_save_path "${EVAL_SAVE_PATH}" \
  --model_path "${EDGE_MODEL_PATH:-${ROOT_DIR}/Model/LLaMA-UAV/work_dirs/llama-uav-7b}" \
  --model_base "${EDGE_MODEL_BASE:-${ROOT_DIR}/Model/LLaMA-UAV/model_zoo/vicuna-7b-v1.5}" \
  --vision_tower "${EDGE_VISION_TOWER:-${ROOT_DIR}/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth}" \
  --image_processor "${EDGE_IMAGE_PROCESSOR:-${ROOT_DIR}/Model/LLaMA-UAV/llamavid/processor/clip-patch14-224}" \
  --traj_model_path "${TRAJ_MODEL_PATH:-${ROOT_DIR}/Model/LLaMA-UAV/work_dirs/traveluav-traj-model}" \
  --groundingdino_config "${GROUNDINGDINO_CONFIG:-${ROOT_DIR}/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}" \
  --groundingdino_model_path "${GROUNDINGDINO_MODEL_PATH:-${ROOT_DIR}/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth}" \
  --ncn_text_model_path "${NCN_TEXT_MODEL:-${ARTIFACT_DIR}/language-base-q8_0.gguf}" \
  --ncn_lora_model_path "${NCN_LORA_MODEL:-${ARTIFACT_DIR}/aerodpo-language-lora-f16.gguf}" \
  --ncn_mmproj_path "${NCN_MMPROJ_MODEL:-${ARTIFACT_DIR}/aerodpo-mmproj-bf16.gguf}" \
  --ncn_local_bridge_path "${NCN_LOCAL_BRIDGE:-${ROOT_DIR}/native/aerodpo_llamacpp_local/build/libaerodpo_llamacpp_local.so}" \
  --eval_json_path "${EVAL_JSON_PATH}" \
  --map_spawn_area_json_path "${MAP_SPAWN_PATH:-${DATASET_PATH}/data/meta/map_spawnarea_info.json}" \
  --object_name_json_path "${OBJECT_NAME_PATH:-${DATASET_PATH}/data/meta/object_description.json}" "$@"
