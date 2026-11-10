# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTE: this CMakeLists.txt was NOT exercised by the agent that wrote it --
# cmake was not available in that sandbox, so the actual verification build
# used build.sh (a direct clang++ invocation against a pip-installed
# libtorch) instead; see README.md's "What was and wasn't verified" section.
# This file is provided as the conventional/portable build path for anyone
# with a normal CMake + libtorch (or `python -c "import torch;
# print(torch.utils.cmake_prefix_path)"`) setup, mirroring build.sh's
# compiler flags and file list; please open an issue/PR if it doesn't
# configure cleanly as-is.
cmake_minimum_required(VERSION 3.18)
project(verl_core_algos CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_library(verl_core_algos STATIC src/core_algos.cpp)
target_include_directories(verl_core_algos PUBLIC include)
target_link_libraries(verl_core_algos PUBLIC ${TORCH_LIBRARIES})
target_compile_options(verl_core_algos PUBLIC ${TORCH_CXX_FLAGS})

option(VERL_CORE_ALGOS_BUILD_TESTS "Build the core_algos C++ validation harness" ON)
if(VERL_CORE_ALGOS_BUILD_TESTS)
  add_executable(validate_core_algos tests/validate_core_algos.cpp)
  target_link_libraries(validate_core_algos PRIVATE verl_core_algos)
endif()
