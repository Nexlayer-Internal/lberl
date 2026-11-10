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
# CMake. This is the exact build path used to actually compile and run
# the validation harness during development (no CMake was available in
# that environment) -- see README.md. Works wherever `python3 -c "import
# torch"` succeeds, CPU-only is fine (this code has no CUDA dependency).
#
# Usage:
#   TORCH_PYTHON=python3 ./build.sh
#   ./build/validate_core_algos ./tests/fixtures
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

"$CXX" "${CXXFLAGS[@]}" -c "$HERE/src/core_algos.cpp" -o "$BUILD_DIR/core_algos.o"
"$CXX" "${CXXFLAGS[@]}" -c "$HERE/tests/validate_core_algos.cpp" -o "$BUILD_DIR/validate_core_algos.o"

"$CXX" -std=c++17 -O2 \
  "$BUILD_DIR/core_algos.o" "$BUILD_DIR/validate_core_algos.o" \
  -o "$BUILD_DIR/validate_core_algos" \
  -L"$TORCH_DIR/lib" -ltorch -ltorch_cpu -lc10 \
  -Wl,-rpath,"$TORCH_DIR/lib"

echo "Built: $BUILD_DIR/validate_core_algos"
echo "Run:   $BUILD_DIR/validate_core_algos $HERE/tests/fixtures"
