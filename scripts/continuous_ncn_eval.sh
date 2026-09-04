#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_dir="${AERODPO_GGUF_DIR:?Set AERODPO_GGUF_DIR to the external Q8 artifact directory}"
run_id="${RUN_ID:-$(date +%m%d-%H%M)}"

source /home/zt/miniconda3/etc/profile.d/conda.sh
conda activate "${NCN_CONDA_ENV:-llamauav}"
export NCN_DISABLE_BITSANDBYTES="${NCN_DISABLE_BITSANDBYTES:-1}"
export PYTHONPATH="${root_dir}/tools/ncn_python_shim${PYTHONPATH:+:${PYTHONPATH}}"
cd "${root_dir}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u src/vlnce_src/continue_ncn_eval.py \
  --run_type eval --name ContinuousNCN --gpu_id 0 --simulator_tool_port "${SIMULATOR_PORT:-25000}" \
  --DDP_MASTER_PORT "${DDP_MASTER_PORT:-20031}" --batchSize 1 --always_help True --use_gt True \
  --max_control_steps "${MAX_CONTROL_STEPS:-1000}" --max_episodes_per_scene 80 \
  --enable_comm_delay True --fast_eval True --fast_eval_speedup 10 --chunk_waypoints 5 \
  --enable_ncn True --ncn_max_consecutive_actions "${NCN_MAX_ACTIONS:-200}" \
  --comm_trace_csv_path "${root_dir}/bandwidth/ucc4g_bandwidth_trace.csv" \
  --dataset_path "${DATASET_PATH:-/HDD2/TravelUAV_dataset/TravelUAV_data}" \
  --eval_save_path "${EVAL_SAVE_PATH:-${root_dir}/eval_continuous_ncn_${run_id}}" \
  --model_path "${EDGE_MODEL_PATH:-${root_dir}/Model/LLaMA-UAV/work_dirs/llama-uav-7b}" \
  --model_base "${EDGE_MODEL_BASE:-${root_dir}/Model/LLaMA-UAV/model_zoo/vicuna-7b-v1.5}" \
  --vision_tower "${EDGE_VISION_TOWER:-${root_dir}/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth}" \
  --image_processor "${EDGE_IMAGE_PROCESSOR:-${root_dir}/Model/LLaMA-UAV/llamavid/processor/clip-patch14-224}" \
  --traj_model_path "${TRAJ_MODEL_PATH:-${root_dir}/Model/LLaMA-UAV/work_dirs/traveluav-traj-model}" \
  --groundingdino_config "${GROUNDINGDINO_CONFIG:-${root_dir}/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}" \
  --groundingdino_model_path "${GROUNDINGDINO_MODEL_PATH:-${root_dir}/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth}" \
  --ncn_text_model_path "${NCN_TEXT_MODEL:-${artifact_dir}/language-base-q8_0.gguf}" \
  --ncn_lora_model_path "${NCN_LORA_MODEL:-${artifact_dir}/aerodpo-language-lora-f16.gguf}" \
  --ncn_mmproj_path "${NCN_MMPROJ_MODEL:-${artifact_dir}/aerodpo-mmproj-bf16.gguf}" \
  --ncn_local_bridge_path "${NCN_LOCAL_BRIDGE:-${root_dir}/native/aerodpo_llamacpp_local/build/libaerodpo_llamacpp_local.so}" \
  --eval_json_path "${EVAL_JSON_PATH:-/HDD2/TravelUAV_dataset/TravelUAV_data/data/uav_dataset/seen_valset.json}" \
  --map_spawn_area_json_path "${MAP_SPAWN_PATH:-/HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/map_spawnarea_info.json}" \
  --object_name_json_path "${OBJECT_NAME_PATH:-/HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/object_description.json}" "$@"
