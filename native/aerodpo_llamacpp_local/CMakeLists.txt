cmake_minimum_required(VERSION 3.18)
project(aerodpo_llamacpp_local LANGUAGES CXX)

set(LLAMA_CPP_DIR "" CACHE PATH "llama.cpp source tree")
if(NOT LLAMA_CPP_DIR)
    message(FATAL_ERROR "Set -DLLAMA_CPP_DIR=/path/to/llama.cpp")
endif()

find_library(LLAMA_LIBRARY NAMES llama REQUIRED
    PATHS "${LLAMA_CPP_DIR}/build/bin" NO_DEFAULT_PATH)
find_library(MTMD_LIBRARY NAMES mtmd REQUIRED
    PATHS "${LLAMA_CPP_DIR}/build/bin" NO_DEFAULT_PATH)

add_library(aerodpo_llamacpp_local SHARED aerodpo_llamacpp_local.cpp)
target_compile_features(aerodpo_llamacpp_local PRIVATE cxx_std_17)
target_include_directories(aerodpo_llamacpp_local PRIVATE
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/tools/mtmd")
target_link_libraries(aerodpo_llamacpp_local PRIVATE "${LLAMA_LIBRARY}" "${MTMD_LIBRARY}")
set_target_properties(aerodpo_llamacpp_local PROPERTIES
    BUILD_RPATH "${LLAMA_CPP_DIR}/build/bin"
    INSTALL_RPATH "${LLAMA_CPP_DIR}/build/bin")
