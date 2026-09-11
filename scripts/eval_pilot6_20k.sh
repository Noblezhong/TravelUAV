#!/bin/bash
# pilot6 (0827) 20k reward固定0.05+iid带宽训练 模型 heldout 283 评估（todo#4 带宽自适应判据）
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"
source /home/noble/miniforge3/etc/profile.d/conda.sh && conda activate llamauav

WEIGHT=$root_dir/drl_train_pilot6_0827-2022/scheduler_models/ppo_scheduler_20260827-202229-448535.zip
EVALJSON=$root_dir/heldout_283_evalformat.json
SAVE=$root_dir/eval_drl_0828_pilot6_20k_fast_x10
LOG=/tmp/pilot6_eval20k.log

echo "model: $WEIGHT"
CUDA_VISIBLE_DEVICES=0 python -u "$root_dir/src/vlnce_src/drl_scheduler_eval.py" \
  --run_type eval --name TravelLLMPCDRLSchedulerEvalPilot6 --gpu_id 0 \
  --simulator_tool_port 25000 --DDP_MASTER_PORT 80014 --batchSize 1 \
  --always_help True --use_gt True --max_control_steps 1000 --scheduler_max_steps 2000 \
  --enable_comm_delay True --fast_eval True --fast_eval_speedup 10 \
  --comm_trace_csv_path "$root_dir/bandwidth/ucc4g_bandwidth_trace.csv" \
  --scheduler_model_path "$WEIGHT" \
  --dataset_path /HDD2/TravelUAV_dataset/TravelUAV_data/ \
  --eval_save_path "$SAVE" \
  --model_path "$root_dir/Model/LLaMA-UAV/work_dirs/llama-uav-7b" \
  --model_base "$root_dir/Model/LLaMA-UAV/model_zoo/vicuna-7b-v1.5" \
  --vision_tower "$root_dir/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth" \
  --image_processor "$root_dir/Model/LLaMA-UAV/llamavid/processor/clip-patch14-224" \
  --traj_model_path "$root_dir/Model/LLaMA-UAV/work_dirs/traveluav-traj-model" \
  --eval_json_path "$EVALJSON" \
  --map_spawn_area_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/map_spawnarea_info.json \
  --object_name_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/object_description.json \
  --groundingdino_config "$root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  --groundingdino_model_path "$root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth" \
  2>&1 | tee "$LOG"
rc=$?
done_eps=$(ls -d ${SAVE}/*/ 2>/dev/null | grep -vE "/profile_logs/" | wc -l)
echo "评估进程退出 rc=$rc, 完成集: $done_eps/283"
echo "日志: $LOG"
