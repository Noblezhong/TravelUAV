#!/bin/bash
# 5090 启动 AirSim ServerTool: 前台运行, 日志直接打在终端/对应 tmux pane 上
# 用法: 在 tmux 会话里 bash scripts/airsim_server_5090.sh
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

source /home/noble/miniforge3/etc/profile.d/conda.sh && conda activate llamauav

# 清掉旧 ServerTool, 避免端口占用
pkill -9 -f "AirVLNSimulatorServerTool" 2>/dev/null && echo "清理旧 ServerTool" || echo "无旧 ServerTool"
sleep 1

echo ">>> 启动 ServerTool (前台, 监听 25000), Ctrl-C 可退出"
python -u airsim_plugin/AirVLNSimulatorServerTool.py \
  --root_path /HDD2/AeroDuo_envs/ \
  --port 25000 \
  --gpus 0
