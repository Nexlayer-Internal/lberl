# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
# NOTE: like csrc/core_algos/CMakeLists.txt, this file was NOT exercised by
# the agent that wrote it -- no `cmake` binary was available in that
# sandbox, so the actual verification build used build.sh (a direct clang++
# invocation against a pip-installed libtorch) instead; see README.md's
# "What was and wasn't verified" section. Provided as the conventional/
# portable build path, mirroring build.sh's compiler flags and file list.
cmake_minimum_required(VERSION 3.18)
project(verl_rollout_corr CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

# Depends on verl_core_algos's masked_sum/masked_mean -- reuse the sibling
# port's target rather than duplicating those two helpers here. Disable its
# own test executables first: core_algos/CMakeLists.txt also defines a
# `cuda_smoke_test` target, which would collide with this directory's own
# target of the same name.
set(VERL_CORE_ALGOS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../core_algos
                 ${CMAKE_CURRENT_BINARY_DIR}/core_algos_build)

add_library(verl_rollout_corr STATIC src/rollout_corr_helper.cpp src/score_centering.cpp)
target_include_directories(verl_rollout_corr PUBLIC include)
target_link_libraries(verl_rollout_corr PUBLIC verl_core_algos ${TORCH_LIBRARIES})
target_compile_options(verl_rollout_corr PUBLIC ${TORCH_CXX_FLAGS})

option(VERL_ROLLOUT_CORR_BUILD_TESTS "Build the rollout_corr C++ validation harness" ON)
if(VERL_ROLLOUT_CORR_BUILD_TESTS)
  add_executable(validate_rollout_corr tests/validate_rollout_corr.cpp)
  target_link_libraries(validate_rollout_corr PRIVATE verl_rollout_corr)

  # CUDA-tensor smoke test exercising every ported function on real CUDA
  # tensors (see tests/cuda_smoke_test.cpp). Builds fine against a CPU-only
  # libtorch too (torch::cuda::is_available() reports false and the test
  # self-reports SKIP, exit code 77, rather than a false PASS).
  add_executable(cuda_smoke_test tests/cuda_smoke_test.cpp)
  target_link_libraries(cuda_smoke_test PRIVATE verl_rollout_corr)
endif()
