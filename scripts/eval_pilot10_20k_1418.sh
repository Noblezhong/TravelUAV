#!/bin/bash
# pilot10 (0901) 20k → 主表 1418 消融行（continuous-only + AHS）
# 边界与兄弟行一致：eval_json=seen_valset(harness 按 item['json'] 去重 → ~1418 集)、fast_eval x10、
#   ucc4g 真实带宽 trace、paradigm=drl_based_hybrid、timing_contract=original-speed logical clock（P5）
# 权重【显式 pin】0901-20k：ppo_scheduler_20260901-150800-148080.zip
#   ⚠️ 严禁 ls -dt | head -1 —— 同目录有 100k 续训漂移版 ppo_scheduler_20260902-122129-468610.zip（mtime 更新）
# 依赖：先起 AirSim ServerTool(25000): bash scripts/airsim_server_5090.sh
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

source /home/noble/miniforge3/etc/profile.d/conda.sh && conda activate llamauav

WEIGHT=$root_dir/drl_train_pilot10_20k_0901-1507/scheduler_models/ppo_scheduler_20260901-150800-148080.zip
EVALJSON=/HDD2/TravelUAV_dataset/TravelUAV_data/data/uav_dataset/seen_valset.json
SAVE=$root_dir/eval_drl_0904_pilot10_20k_1418_fast_x10
LOG=/tmp/pilot10_eval1418.log

[ -f "$WEIGHT" ] || { echo "ERROR: ckpt missing: $WEIGHT"; exit 1; }
[ -f "$EVALJSON" ] || { echo "ERROR: eval json missing: $EVALJSON"; exit 1; }
echo "model: $WEIGHT"
echo "eval:  $EVALJSON"
echo "save:  $SAVE"

CUDA_VISIBLE_DEVICES=0 python -u "$root_dir/src/vlnce_src/drl_scheduler_eval.py" \
  --run_type eval --name TravelLLMPCDRLSchedulerEvalPilot10 --gpu_id 0 \
  --simulator_tool_port 25000 --DDP_MASTER_PORT 80016 --batchSize 1 \
  --always_help True --use_gt True --max_control_steps 1000 --scheduler_max_steps 2000 \
  --enable_comm_delay True --fast_eval True --fast_eval_speedup 10 \
  --comm_trace_csv_path "$root_dir/bandwidth/ucc4g_bandwidth_trace.csv" \
  --scheduler_model_path "$WEIGHT" \
  --scheduler_time_weight 1.0 \
  --scheduler_oracle_success_reward 20.0 \
  --scheduler_illegal_action_penalty 5.0 \
  --scheduler_req_bw_thresh_mbps 25.0 \
  --scheduler_req_buf_thresh 3.0 \
  --scheduler_motion_drift_thresh_m 2.5 \
  --scheduler_motion_cont_weight 0.4 \
  --scheduler_req_bw_weight 0.6 \
  --scheduler_req_inflight_penalty 0.5 \
  --scheduler_req_buf_weight 0.15 \
  --scheduler_motion_stop_noreq_penalty 0.5 \
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
echo "评估进程退出 rc=$rc, 已完成 episode 目录: $done_eps"
echo "日志: $LOG (tail -f $LOG 查看)"
