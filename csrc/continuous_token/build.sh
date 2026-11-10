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
# Direct clang++/g++ build, mirroring the sibling ports' build.sh convention
# (core_algos/rollout_corr/data_proto), extended for the one genuinely new
# dependency this port has that they didn't: a Rust toolchain.
#
# Unlike the other three C++ ports, this one has NO libtorch/CUDA dependency
# at all -- it's a tokenization/chat-template port, not tensor math (see
# README.md). What it does need:
#   1. `cargo`/`rustc` (any reasonably recent stable toolchain) to build the
#      Rust shim crate (rust/) that binds the real HF `tokenizers` crate +
#      minijinja for chat-template rendering.
#   2. `cbindgen` (`cargo install cbindgen`) to regenerate
#      include/verl_continuous_token/ffi/tokenizer_shim.h from the Rust
#      shim's actual `extern "C"` signatures, so the C++ side can never see
#      a header that's drifted from the real FFI surface.
#
# Usage:
#   ./build.sh
#   ./tests/fetch_tokenizer.sh                       # one-time, needs network
#   ./build/validate_continuous_token ./tests/fixtures \
#       ./tests/fixtures/qwen3_tokenizer/tokenizer.json \
#       ./tests/fixtures/qwen3_tokenizer/chat_template.jinja
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$HERE/build}"
CXX="${CXX:-c++}"
RUST_PROFILE="${RUST_PROFILE:-release}"

mkdir -p "$BUILD_DIR"

command -v cargo >/dev/null 2>&1 || {
  echo "error: cargo not found on PATH. Install a Rust toolchain (e.g. 'brew install rust' or https://rustup.rs)." >&2
  exit 1
}

echo "=== 1/3: building Rust shim crate (verl_continuous_token_shim, $RUST_PROFILE) ==="
( cd "$HERE/rust" && cargo build "--$RUST_PROFILE" )
RUST_TARGET_DIR="$HERE/rust/target/$RUST_PROFILE"
RUST_LIB="$RUST_TARGET_DIR/libverl_continuous_token_shim.a"
if [[ ! -f "$RUST_LIB" ]]; then
  echo "error: expected static lib not found at $RUST_LIB" >&2
  exit 1
fi

echo "=== 2/3: regenerating ffi/tokenizer_shim.h via cbindgen ==="
if command -v cbindgen >/dev/null 2>&1; then
  ( cd "$HERE/rust" && cbindgen --config cbindgen.toml --crate verl_continuous_token_shim \
      --output "$HERE/include/verl_continuous_token/ffi/tokenizer_shim.h" )
else
  echo "warning: cbindgen not found (cargo install cbindgen) -- using the checked-in" >&2
  echo "         include/verl_continuous_token/ffi/tokenizer_shim.h as-is. It was generated" >&2
  echo "         from the same lib.rs by the port's author; regenerate it yourself to be" >&2
  echo "         sure it hasn't drifted if you've since edited rust/src/lib.rs." >&2
fi

echo "=== 3/3: building C++ wrapper + validation harness ==="
CXXFLAGS=(-std=c++17 -O2 -Wall -Wextra -I"$HERE/include")

"$CXX" "${CXXFLAGS[@]}" -c "$HERE/src/continuous_token.cpp" -o "$BUILD_DIR/continuous_token.o"
"$CXX" "${CXXFLAGS[@]}" -c "$HERE/tests/validate_continuous_token.cpp" -o "$BUILD_DIR/validate_continuous_token.o"

# macOS/Linux system libs the Rust staticlib needs (pthread, dl, m; on macOS
# `-framework Security` for getrandom's keychain-based fallback path, harmless
# to pass on Linux builds too... except it isn't, so gate it on uname).
EXTRA_LIBS=(-lpthread -ldl -lm)
if [[ "$(uname -s)" == "Darwin" ]]; then
  EXTRA_LIBS+=(-framework Security -framework CoreFoundation)
fi

"$CXX" -std=c++17 -O2 \
  "$BUILD_DIR/continuous_token.o" "$BUILD_DIR/validate_continuous_token.o" \
  "$RUST_LIB" \
  -o "$BUILD_DIR/validate_continuous_token" \
  "${EXTRA_LIBS[@]}"

echo
echo "Built: $BUILD_DIR/validate_continuous_token"
echo "Run (after ./tests/fetch_tokenizer.sh once):"
echo "  $BUILD_DIR/validate_continuous_token $HERE/tests/fixtures \\"
echo "      $HERE/tests/fixtures/qwen3_tokenizer/tokenizer.json \\"
echo "      $HERE/tests/fixtures/qwen3_tokenizer/chat_template.jinja"
