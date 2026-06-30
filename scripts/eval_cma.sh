#!/bin/bash
# CMA model evaluation on TravelUAV dataset
# Usage: bash scripts/eval_cma.sh

root_dir=.  # TravelUAV directory

# Use ckpt.49 (epoch 49, converged: mean loss ~0.055)
CKPT_DIR=20260627-014453-749496
CKPT_FILE=ckpt.54.pth

CMA_EVAL_ONLY=1 CMA_TELEPORT=1 CUDA_VISIBLE_DEVICES=0 python -u $root_dir/src/vlnce_src/eval_cma.py \
    --run_type eval \
    --name CMA-TravelUAV \
    --gpu_id 0 \
    --simulator_tool_port 30000 \
    --DDP_MASTER_PORT 80005 \
    --batchSize 1 \
    --always_help False \
    --use_gt False \
    --maxWaypoints 200 \
    --dataset_path /mnt/zt-hdd2/TravelUAV_dataset/TravelUAV_data/ \
    --eval_save_path /HDD1/code/TravelUAV/eval_output_cma_v2 \
    --eval_json_path /mnt/zt-hdd2/TravelUAV_dataset/TravelUAV_data_json/data/uav_dataset/seen_valset.json \
    --map_spawn_area_json_path /mnt/zt-hdd2/TravelUAV_dataset/TravelUAV_data_json/data/meta/map_spawnarea_info.json \
    --object_name_json_path /mnt/zt-hdd2/TravelUAV_dataset/TravelUAV_data_json/data/meta/object_description.json \
    --cma_ckpt_path /HDD1/code/AirVLN_ws/DATA/output/AirVLN-cma-traveluav/train/checkpoint/${CKPT_DIR}/${CKPT_FILE} \
    --cma_vocab_path /HDD1/code/AirVLN_ws/DATA/data/traveluav/train_vocab.txt

echo ""
echo "Results saved to: eval_output_cma/"
