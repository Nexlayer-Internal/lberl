#!/usr/bin/env bash
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
# Direct clang++/g++ build against a pip-installed libtorch, bypassing
# CMake -- same no-cmake sandbox constraint as csrc/core_algos/build.sh (see
# that file / README for why). Works wherever `python3 -c "import torch"`
# succeeds; CPU-only is fine for the CPU fixture-comparison harness.
#
# This module depends on verl_core_algos's masked_sum/masked_mean, so it
# compiles ../core_algos/src/core_algos.cpp as an extra translation unit
# rather than duplicating those two helpers.
#
# Usage:
#   TORCH_PYTHON=python3 ./build.sh
#   ./build/validate_rollout_corr ./tests/fixtures
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_ALGOS_DIR="$(cd "$HERE/../core_algos" && pwd)"
TORCH_PYTHON="${TORCH_PYTHON:-python3}"
BUILD_DIR="${BUILD_DIR:-$HERE/build}"
CXX="${CXX:-c++}"

TORCH_DIR="$("$TORCH_PYTHON" -c 'import torch, os; print(os.path.dirname(torch.__file__))')"
echo "Using libtorch from: $TORCH_DIR"

mkdir -p "$BUILD_DIR"

CXXFLAGS=(-std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI=1
  -I"$TORCH_DIR/include"
  -I"$TORCH_DIR/include/torch/csrc/api/include"
  -I"$HERE/include"
  -I"$CORE_ALGOS_DIR/include")

"$CXX" "${CXXFLAGS[@]}" -c "$CORE_ALGOS_DIR/src/core_algos.cpp" -o "$BUILD_DIR/core_algos.o"
"$CXX" "${CXXFLAGS[@]}" -c "$HERE/src/rollout_corr_helper.cpp" -o "$BUILD_DIR/rollout_corr_helper.o"
"$CXX" "${CXXFLAGS[@]}" -c "$HERE/src/score_centering.cpp" -o "$BUILD_DIR/score_centering.o"
"$CXX" "${CXXFLAGS[@]}" -c "$HERE/tests/validate_rollout_corr.cpp" -o "$BUILD_DIR/validate_rollout_corr.o"
"$CXX" "${CXXFLAGS[@]}" -c "$HERE/tests/cuda_smoke_test.cpp" -o "$BUILD_DIR/cuda_smoke_test.o"

# Optional CUDA-enabled libtorch libs -- harmless to pass if they simply
# don't exist (CPU-only libtorch build), same pattern as core_algos/build.sh.
EXTRA_LIBS=()
for lib in torch_cuda c10_cuda; do
  if ls "$TORCH_DIR"/lib/lib${lib}.* >/dev/null 2>&1; then
    EXTRA_LIBS+=("-l${lib}")
  fi
done

OBJS=("$BUILD_DIR/core_algos.o" "$BUILD_DIR/rollout_corr_helper.o" "$BUILD_DIR/score_centering.o")

"$CXX" -std=c++17 -O2 \
  "${OBJS[@]}" "$BUILD_DIR/validate_rollout_corr.o" \
  -o "$BUILD_DIR/validate_rollout_corr" \
  -L"$TORCH_DIR/lib" -ltorch -ltorch_cpu -lc10 "${EXTRA_LIBS[@]+"${EXTRA_LIBS[@]}"}" \
  -Wl,-rpath,"$TORCH_DIR/lib"

"$CXX" -std=c++17 -O2 \
  "${OBJS[@]}" "$BUILD_DIR/cuda_smoke_test.o" \
  -o "$BUILD_DIR/cuda_smoke_test" \
  -L"$TORCH_DIR/lib" -ltorch -ltorch_cpu -lc10 "${EXTRA_LIBS[@]+"${EXTRA_LIBS[@]}"}" \
  -Wl,-rpath,"$TORCH_DIR/lib"

echo "Built: $BUILD_DIR/validate_rollout_corr"
echo "Run:   $BUILD_DIR/validate_rollout_corr $HERE/tests/fixtures"
echo "Built: $BUILD_DIR/cuda_smoke_test"
echo "Run:   $BUILD_DIR/cuda_smoke_test   (exits 77/SKIP if no CUDA device is visible)"
