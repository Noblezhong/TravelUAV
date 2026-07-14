#!/bin/bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${JETSON_MODEL_DIR:-$root_dir/Model/LLaMA-UAV}"
dataset_path="${JETSON_DATASET_PATH:-$HOME/traveluav_shared/data}"
chunk_waypoints="${CHUNK_WAYPOINTS:-5}"
enable_comm_delay="${ENABLE_COMM_DELAY:-True}"
com_suffix=""
name_suffix=""
if [ "$enable_comm_delay" = "True" ]; then
    com_suffix="_com"
    name_suffix="Com"
fi
eval_save_path="$root_dir/eval_dnn${com_suffix}_w${chunk_waypoints}"
eval_json_path="${JETSON_EVAL_JSON_PATH:-$dataset_path/data/uav_dataset/seen_valset.json}"
map_spawn_area_json_path="${JETSON_MAP_SPAWN_AREA_JSON_PATH:-$dataset_path/data/meta/map_spawnarea_info.json}"
object_name_json_path="${JETSON_OBJECT_NAME_JSON_PATH:-$dataset_path/data/meta/object_description.json}"
groundingdino_config="${JETSON_GROUNDINGDINO_CONFIG:-$root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
groundingdino_model_path="${JETSON_GROUNDINGDINO_MODEL_PATH:-$root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth}"
edge_vlm_host="${EDGE_VLM_HOST:-192.168.105.17}"
edge_vlm_port="${EDGE_VLM_PORT:-26000}"
simulator_tool_host="${SIMULATOR_TOOL_HOST:-192.168.105.17}"
simulator_tool_port="${SIMULATOR_TOOL_PORT:-25000}"
fast_eval="${FAST_EVAL:-True}"
fast_eval_speedup="${FAST_EVAL_SPEEDUP:-10}"
# Carla/UE memory grows during long fast-eval runs. Reopen the same scene
# periodically without changing episode termination or navigation logic.
max_episodes_per_scene="${MAX_EPISODES_PER_SCENE:-100}"

# One evaluator owns one simulator port. This prevents two tmux jobs from
# concurrently calling reopen_scenes() and killing each other's UE process.
lock_path="/tmp/traveluav_edge_dnn_${simulator_tool_port}.lock"
exec 9>"$lock_path"
if ! flock -n 9; then
    echo "simulator port ${simulator_tool_port} is already owned by another continuous DNN evaluator" >&2
    exit 2
fi

mkdir -p "$eval_save_path"
cd "$root_dir"

CUDA_VISIBLE_DEVICES=0 python -u "$root_dir/src/vlnce_src/edge_dnn_jetson_eval.py" \
    --run_type eval \
    --name EdgeDNNJetson${name_suffix} \
    --gpu_id 0 \
    --simulator_tool_host "$simulator_tool_host" \
    --simulator_tool_port "$simulator_tool_port" \
    --DDP_MASTER_PORT 80005 \
    --batchSize 1 \
    --always_help True \
    --use_gt True \
    --maxWaypoints 200 \
    --edge_vlm_host "$edge_vlm_host" \
    --edge_vlm_port "$edge_vlm_port" \
    --chunk_waypoints "$chunk_waypoints" \
    --enable_comm_delay "$enable_comm_delay" \
    --fast_eval "$fast_eval" \
    --fast_eval_speedup "$fast_eval_speedup" \
    --max_episodes_per_scene "$max_episodes_per_scene" \
    --comm_trace_csv_path "$root_dir/bandwidth/ucc4g_bandwidth_trace.csv" \
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
    --groundingdino_model_path "$groundingdino_model_path" \
    "$@"
