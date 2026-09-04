# 5090: TravelUAV NCN deployment

The repository contains code only.  Do not add Qwen, AeroDPO, GGUF, dataset,
evaluation, or build files to Git.

## One-time setup

```bash
git checkout edge-collab-vln
git pull --ff-only origin edge-collab-vln
git clone https://github.com/ggml-org/llama.cpp /opt/llama.cpp
cd /opt/llama.cpp && git checkout b19cbe925be361d229f0fe03435affe4a2717f37
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --parallel
```

Use paths outside the repository for all model artifacts.  Given the original
Qwen3-VL base and the original AeroDPO adapter, create Q8 once:

```bash
conda activate aerodpo  # conversion dependencies
python scripts/prepare_aerodpo_llamacpp.py \
  --base-model-dir /models/AeroVLA-2B-384-Merged \
  --adapter-dir /models/AeroDPO-2B-384-LoRA \
  --llama-cpp-dir /opt/llama.cpp \
  --output-dir /models/aerodpo-llamacpp-q8_0 \
  --staging-dir /models/aerodpo-mmproj-staging
LLAMA_CPP_DIR=/opt/llama.cpp scripts/build_aerodpo_llamacpp_local.sh
```

The output contains `language-base-q8_0.gguf`,
`aerodpo-language-lora-f16.gguf`, and `aerodpo-mmproj-bf16.gguf`; it is ignored
by Git.  The conversion refuses to overwrite a non-empty output directory.

## Preflight and run

On an otherwise idle 5090, first load the full edge stack and NCN in a three
episode smoke split.  Record `nvidia-smi` before, during, and after load.  If
the process cannot load all edge models and the Q8 NCN together, stop: do not
silently unload TCM/AHS dependencies or alter precision.

```bash
export AERODPO_GGUF_DIR=/models/aerodpo-llamacpp-q8_0
export LLAMA_CPP_DIR=/opt/llama.cpp
scripts/build_aerodpo_llamacpp_local.sh
export NCN_CONDA_ENV=llamauav  # evaluation uses the LLaMA-UAV environment
EVAL_JSON_PATH=/path/to/three_episode_seen_smoke.json \
EVAL_SAVE_PATH=/results/continuous_ncn_smoke \
scripts/continuous_ncn_eval.sh
```

The NCN launchers disable unused bitsandbytes auto-discovery by default.  This
avoids importing the environment's old CUDA 11 package into the llama.cpp
CUDA process.  Set `NCN_DISABLE_BITSANDBYTES=0` only if the edge stack
explicitly requires bitsandbytes and its CUDA libraries are compatible with
the llama.cpp build.

Use `scripts/stopgo_ncn_eval.sh` for the Stop-go+NCN ablation.  For a formal
run, omit the smoke split override and choose a new empty result directory.
Both scripts use the existing seen split, Fast Eval x10, batch size one, and
the existing bandwidth trace by default.
