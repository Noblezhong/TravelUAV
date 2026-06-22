#!/bin/bash
# CMA model evaluation on TravelUAV dataset
# Usage: bash scripts/eval_cma.sh

root_dir=.  # TravelUAV directory

CUDA_VISIBLE_DEVICES=0 python -u $root_dir/src/vlnce_src/eval_cma.py \
    --run_type eval \
    --name CMA-TravelUAV \
    --gpu_id 0 \
    --simulator_tool_port 25000 \
    --DDP_MASTER_PORT 80005 \
    --batchSize 1 \
    --always_help True \
    --use_gt True \
    --maxWaypoints 200 \
    --dataset_path /HDD2/TravelUAV_dataset/TravelUAV_data/ \
    --eval_save_path /HDD1/code/TravelUAV/eval_output_cma \
    --eval_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/uav_dataset/seen_valset.json \
    --map_spawn_area_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/map_spawnarea_info.json \
    --object_name_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/object_description.json \
    --cma_ckpt_path /HDD1/code/AirVLN_ws/DATA/output/AirVLN-cma-traveluav/train/checkpoint/__CKPT_DIR__/ckpt.LAST.pth \
    --cma_vocab_path /HDD1/code/AirVLN_ws/DATA/data/traveluav/train_vocab.txt

echo ""
echo "Results saved to: eval_output_cma/"
