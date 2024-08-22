# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Minimum CMake required
cmake_minimum_required(VERSION 3.26)

cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

cmake_policy(SET CMP0092 NEW)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0117 NEW)
# Don't let cmake set a default value for CMAKE_CUDA_ARCHITECTURES
cmake_policy(SET CMP0104 OLD)

# Enable Hot Reload for MSVC compilers if supported.
cmake_policy(SET CMP0141 NEW)

# Project
project(custom_op_library C CXX ASM)

include(CheckCXXCompilerFlag)
include(CheckLanguage)
include(${CMAKE_CURRENT_LIST_DIR}/onnxruntime.cmake)

# TODO: update this once all system adapt c++20
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
set(CMAKE_CXX_STANDARD 20)
else()
set(CMAKE_CXX_STANDARD 17)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
# NOTE: POSITION INDEPENDENT CODE hurts performance, and it only make sense on POSIX systems
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")



include_directories("${PROJECT_BINARY_DIR}/_deps/onnxruntime-src/include")
include_directories("${PROJECT_SOURCE_DIR}/extra_ort_headers/")
set(custom_op_src_patterns
"${PROJECT_SOURCE_DIR}/custom_op_library/*.h"
"${PROJECT_SOURCE_DIR}/custom_op_library/*.cc"
)


#set(custom_op_lib_include ${PROJECT_SOURCE_DIR}/custom_op_library/onnxruntime-linux-x64-gpu-1.16.1/include)
#list(APPEND custom_op_lib_include /home/jicwen/work/anaconda3/envs/TNLGv4_env/lib/python3.8/site-packages/torch/include)
#list(APPEND custom_op_lib_include  /home/jicwen/work/anaconda3/envs/TNLGv4_env/lib/python3.8/site-packages/torch/include/torch/csrc/api/include)
#list(APPEND custom_op_lib_include  /home/jicwen/work/anaconda3/envs/TNLGv4_env/lib/python3.8/site-packages/torch/include/TH)
#list(APPEND custom_op_lib_include /home/jicwen/work/anaconda3/envs/TNLGv4_env/lib/python3.8/site-packages/torch/include/THC)
#list(APPEND custom_op_lib_include  /home/jicwen/work/anaconda3/envs/TNLGv4_env/include/python3.8)
#set(custom_op_lib_option)
#set(custom_op_lib_dir /home/jicwen/work/anaconda3/envs/TNLGv4_env/lib/python3.8/site-packages/torch/lib)
#set(custom_op_lib_link ${GSL_TARGET} c10 torch torch_cpu torch_python)


if (NOT "$ENV{CUDNN_HOME}" STREQUAL "")
    set(CUDNN_HOME "$ENV{CUDNN_HOME}" CACHE INTERNAL "Copied from environment variable")
else()
    set(CUDNN_HOME "/usr/lib/x86_64-linux-gnu" )
endif()

if (NOT "$ENV{CUDA_PATH}" STREQUAL "")
    set(CUDA_PATH "$ENV{CUDA_PATH}" CACHE INTERNAL "Copied from environment variable")
else()
    set(CUDA_PATH "/usr/lib/x86_64-linux-gnu/")
endif()


list(APPEND custom_op_src_patterns
    "${PROJECT_SOURCE_DIR}/custom_op_library/kernels/cuda_ops.cu"
    "${PROJECT_SOURCE_DIR}/custom_op_library/cuda/cuda_ops.*")
list(APPEND custom_op_lib_include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CUDNN_HOME}/include ${CUDA_PATH}/include)
if (HAS_QSPECTRE)
    list(APPEND custom_op_lib_option "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /Qspectre>")
endif()

file(GLOB custom_op_src ${custom_op_src_patterns})
add_library(custom_op_library SHARED ${custom_op_src})
target_compile_options(custom_op_library PRIVATE ${custom_op_lib_option})
target_include_directories(custom_op_library PRIVATE ${PROJECT_SOURCE_DIR}/include ${custom_op_lib_include})
target_link_directories(custom_op_library PRIVATE ${custom_op_lib_dir})
#link_directories(${custom_op_lib_dir})
target_link_libraries(custom_op_library PRIVATE ${GSL_TARGET} ${custom_op_lib_link})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

