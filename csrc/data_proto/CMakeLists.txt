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
# no cmake binary was available in that sandbox, so the actual verification
# build used build.sh (a direct clang++ invocation against a pip-installed
# libtorch) instead; see README.md's "What was and wasn't verified" section.
# This file mirrors build.sh's compiler flags and file list, following the
# same conventional/portable build path csrc/core_algos and csrc/rollout_corr
# already established; please open an issue/PR if it doesn't configure
# cleanly as-is.
cmake_minimum_required(VERSION 3.18)
project(verl_data_proto CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_library(verl_data_proto STATIC src/tensor_batch.cpp)
target_include_directories(verl_data_proto PUBLIC include)
target_link_libraries(verl_data_proto PUBLIC ${TORCH_LIBRARIES})
target_compile_options(verl_data_proto PUBLIC ${TORCH_CXX_FLAGS})

option(VERL_DATA_PROTO_BUILD_TESTS "Build the data_proto C++ validation harness" ON)
if(VERL_DATA_PROTO_BUILD_TESTS)
  add_executable(validate_data_proto tests/validate_data_proto.cpp)
  target_link_libraries(validate_data_proto PRIVATE verl_data_proto)

  # CUDA-tensor smoke test: exercises `to(device)` plus a representative
  # subset of the other operations on real CUDA tensors. Builds fine against
  # a CPU-only libtorch too (torch::cuda::is_available() just returns false
  # and the test self-reports SKIP, exit code 77, rather than a false PASS)
  # -- but only running it against a real CUDA-enabled libtorch + GPU
  # actually exercises anything. See tests/cuda_smoke_test.cpp's header
  # comment and README.md's CUDA-safety audit section.
  add_executable(cuda_smoke_test tests/cuda_smoke_test.cpp)
  target_link_libraries(cuda_smoke_test PRIVATE verl_data_proto)
endif()
