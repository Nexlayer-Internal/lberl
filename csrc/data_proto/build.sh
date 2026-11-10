#!/usr/bin/env bash
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
# Direct clang++/g++ build against a pip-installed libtorch, bypassing
# CMake -- same no-cmake-in-this-sandbox convention as csrc/core_algos and
# csrc/rollout_corr (see README.md). Works wherever `python3 -c "import
# torch"` succeeds; CPU-only is fine for the fixture-comparison harness
# (this code has no CUDA-only-at-compile-time dependency), but
# cuda_smoke_test needs a CUDA-enabled libtorch + real GPU to actually
# exercise anything beyond its own SKIP path.
#
# Usage:
#   TORCH_PYTHON=python3 ./build.sh
#   ./build/validate_data_proto ./tests/fixtures
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCH_PYTHON="${TORCH_PYTHON:-python3}"
BUILD_DIR="${BUILD_DIR:-$HERE/build}"
CXX="${CXX:-c++}"

TORCH_DIR="$("$TORCH_PYTHON" -c 'import torch, os; print(os.path.dirname(torch.__file__))')"
echo "Using libtorch from: $TORCH_DIR"

mkdir -p "$BUILD_DIR"

CXXFLAGS=(-std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI=1
  -I"$TORCH_DIR/include"
  -I"$TORCH_DIR/include/torch/csrc/api/include"
  -I"$HERE/include")

"$CXX" "${CXXFLAGS[@]}" -c "$HERE/src/tensor_batch.cpp" -o "$BUILD_DIR/tensor_batch.o"
"$CXX" "${CXXFLAGS[@]}" -c "$HERE/tests/validate_data_proto.cpp" -o "$BUILD_DIR/validate_data_proto.o"
"$CXX" "${CXXFLAGS[@]}" -c "$HERE/tests/cuda_smoke_test.cpp" -o "$BUILD_DIR/cuda_smoke_test.o"

# Link flags for an optional CUDA-enabled libtorch (ltorch_cuda/lc10_cuda).
# Harmless to pass even against a CPU-only libtorch build IF those .so/.dylib
# files simply don't exist there -- so only add them when present, rather
# than unconditionally requiring a CUDA libtorch to build at all.
EXTRA_LIBS=()
for lib in torch_cuda c10_cuda; do
  if ls "$TORCH_DIR"/lib/lib${lib}.* >/dev/null 2>&1; then
    EXTRA_LIBS+=("-l${lib}")
  fi
done

"$CXX" -std=c++17 -O2 \
  "$BUILD_DIR/tensor_batch.o" "$BUILD_DIR/validate_data_proto.o" \
  -o "$BUILD_DIR/validate_data_proto" \
  -L"$TORCH_DIR/lib" -ltorch -ltorch_cpu -lc10 "${EXTRA_LIBS[@]+"${EXTRA_LIBS[@]}"}" \
  -Wl,-rpath,"$TORCH_DIR/lib"

"$CXX" -std=c++17 -O2 \
  "$BUILD_DIR/tensor_batch.o" "$BUILD_DIR/cuda_smoke_test.o" \
  -o "$BUILD_DIR/cuda_smoke_test" \
  -L"$TORCH_DIR/lib" -ltorch -ltorch_cpu -lc10 "${EXTRA_LIBS[@]+"${EXTRA_LIBS[@]}"}" \
  -Wl,-rpath,"$TORCH_DIR/lib"

echo "Built: $BUILD_DIR/validate_data_proto"
echo "Run:   $BUILD_DIR/validate_data_proto $HERE/tests/fixtures"
echo "Built: $BUILD_DIR/cuda_smoke_test"
echo "Run:   $BUILD_DIR/cuda_smoke_test   (exits 77/SKIP without a GPU)"
