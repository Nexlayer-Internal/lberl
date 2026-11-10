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
# Unlike csrc/core_algos, csrc/rollout_corr, and csrc/data_proto, this port
# has NO libtorch/CUDA dependency (it's tokenization/chat-template, not
# tensor math -- see README.md). It DOES have one dependency those three
# don't: a Rust toolchain (cargo + rustc), used here via a custom command
# that invokes `cargo build` for the rust/ shim crate and `cbindgen` to
# regenerate include/verl_continuous_token/ffi/tokenizer_shim.h, so building
# this target through CMake "just works" the same way `cargo build`
# transparently builds a crate's C dependencies via `cc`/`build.rs` --
# except here it's the other direction (C++ build pulling in a Rust crate).
#
# This file WAS exercised for real by the author of this port (unlike the
# other three ports' CMakeLists.txt, which were written but not built by
# their authors) -- see README.md's "What was, and wasn't, verified"
# section.
cmake_minimum_required(VERSION 3.18)
project(verl_continuous_token CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_program(CARGO_EXECUTABLE cargo REQUIRED)
find_program(CBINDGEN_EXECUTABLE cbindgen)

set(VCT_RUST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rust")
set(VCT_RUST_PROFILE "release" CACHE STRING "Cargo profile for the Rust shim crate (release|debug)")
if(VCT_RUST_PROFILE STREQUAL "release")
  set(VCT_CARGO_BUILD_ARGS --release)
  set(VCT_RUST_TARGET_SUBDIR "release")
else()
  set(VCT_CARGO_BUILD_ARGS "")
  set(VCT_RUST_TARGET_SUBDIR "debug")
endif()

set(VCT_RUST_LIB "${VCT_RUST_DIR}/target/${VCT_RUST_TARGET_SUBDIR}/libverl_continuous_token_shim.a")
set(VCT_FFI_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/verl_continuous_token/ffi/tokenizer_shim.h")

# Build the Rust shim crate as part of the normal CMake build -- this is
# what makes "build the C++ library normally also builds the Rust shim"
# true, rather than requiring a separately-documented prerequisite step.
add_custom_command(
  OUTPUT "${VCT_RUST_LIB}"
  COMMAND "${CARGO_EXECUTABLE}" build ${VCT_CARGO_BUILD_ARGS}
  WORKING_DIRECTORY "${VCT_RUST_DIR}"
  DEPENDS "${VCT_RUST_DIR}/src/lib.rs" "${VCT_RUST_DIR}/Cargo.toml"
  COMMENT "Building Rust shim crate (verl_continuous_token_shim, ${VCT_RUST_PROFILE})"
  VERBATIM)
add_custom_target(verl_continuous_token_rust_shim DEPENDS "${VCT_RUST_LIB}")

if(CBINDGEN_EXECUTABLE)
  add_custom_command(
    OUTPUT "${VCT_FFI_HEADER}"
    COMMAND "${CBINDGEN_EXECUTABLE}" --config cbindgen.toml --crate verl_continuous_token_shim
            --output "${VCT_FFI_HEADER}"
    WORKING_DIRECTORY "${VCT_RUST_DIR}"
    DEPENDS "${VCT_RUST_DIR}/src/lib.rs" "${VCT_RUST_DIR}/cbindgen.toml"
    COMMENT "Regenerating ffi/tokenizer_shim.h via cbindgen (never hand-edit this file)"
    VERBATIM)
  add_custom_target(verl_continuous_token_ffi_header DEPENDS "${VCT_FFI_HEADER}")
else()
  message(WARNING
    "cbindgen not found (cargo install cbindgen) -- using the checked-in "
    "include/verl_continuous_token/ffi/tokenizer_shim.h as-is instead of regenerating it. "
    "It was generated from the same rust/src/lib.rs by this port's author; if you've since "
    "edited lib.rs, install cbindgen and reconfigure to keep the header from drifting.")
  add_custom_target(verl_continuous_token_ffi_header)
endif()

add_library(verl_continuous_token STATIC src/continuous_token.cpp)
add_dependencies(verl_continuous_token verl_continuous_token_rust_shim verl_continuous_token_ffi_header)
target_include_directories(verl_continuous_token PUBLIC include)
target_link_libraries(verl_continuous_token PUBLIC "${VCT_RUST_LIB}")

if(APPLE)
  target_link_libraries(verl_continuous_token PUBLIC "-framework Security" "-framework CoreFoundation")
endif()
find_package(Threads REQUIRED)
target_link_libraries(verl_continuous_token PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

option(VERL_CONTINUOUS_TOKEN_BUILD_TESTS "Build the continuous_token C++ validation harness" ON)
if(VERL_CONTINUOUS_TOKEN_BUILD_TESTS)
  add_executable(validate_continuous_token tests/validate_continuous_token.cpp)
  target_link_libraries(validate_continuous_token PRIVATE verl_continuous_token)
endif()
