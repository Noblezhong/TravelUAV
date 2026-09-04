#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llama_dir="${LLAMA_CPP_DIR:-/HDD1/code/llama.cpp}"
build_dir="${AERODPO_LOCAL_BUILD_DIR:-${repo_root}/native/aerodpo_llamacpp_local/build}"

cmake -S "${repo_root}/native/aerodpo_llamacpp_local" -B "${build_dir}" \
    -DLLAMA_CPP_DIR="${llama_dir}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --parallel "${AERODPO_LOCAL_BUILD_JOBS:-4}"

printf '%s\n' "${build_dir}/libaerodpo_llamacpp_local.so"
