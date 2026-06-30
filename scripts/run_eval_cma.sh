#!/bin/bash
# CMA Evaluation — start server, warm up scene, run eval, all in one shot
set -e
cd /HDD1/code/TravelUAV

echo "=== 1. Cleanup ==="
pkill -9 -f CarlaUE4 2>/dev/null || true
pkill -9 -f SimulatorServerTool 2>/dev/null || true
sleep 2

echo "=== 2. Start AirSim Server ==="
python airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0 --root_path /mnt/zt-hdd2/AeroDuo_envs/ &
SERVER_PID=$!
sleep 3

# Verify server RPC
python3 -c "
import msgpackrpc, sys
c = msgpackrpc.Client(msgpackrpc.Address('127.0.0.1', 30000), timeout=5)
if c.call('ping'):
    print('Server RPC OK')
else:
    print('Server RPC FAILED')
    sys.exit(1)
" || { echo 'Server failed'; exit 1; }

echo "=== 3. Pre-warm: open first scene ==="
python3 -c "
import msgpackrpc, time, socket
c = msgpackrpc.Client(msgpackrpc.Address('127.0.0.1', 30000), timeout=300)
print('Calling reopen_scenes...')
result = c.call('reopen_scenes', '127.0.0.1', [['Carla_Town01', 0]])
if result[0]:
    ip, ports = result[1]
    print(f'Scene launched on ports: {ports}')
else:
    print(f'Server returned failure: {result}')
    # Check if port is still open (previous UE running)
    import subprocess
    subprocess.run(['ss', '-tlnp', '|', 'grep', '30001'], shell=True)
    raise SystemExit('Pre-warm failed')
print('Scene ready!')
"

echo "=== 4. Run CMA Evaluation ==="
bash scripts/eval_cma.sh

echo "=== 5. Cleanup ==="
kill $SERVER_PID 2>/dev/null || true
