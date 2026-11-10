// Copyright 2024 Bytedance Ltd. and/or its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// CUDA-tensor smoke test for compute_grpo_outcome_advantage.
//
// Background: an earlier version of this port called
// `Tensor::accessor<float, 1>()` directly on a tensor that could be a CUDA
// tensor. `accessor<T,N>()` assumes host-dereferenceable memory -- it is
// CPU-only -- so indexing into it when the tensor actually lives on a CUDA
// device reads/writes device memory as if it were host memory, which
// segfaults (or silently corrupts memory, absent a debugger). The fix moves
// the per-group mean/std bookkeeping onto an explicit `.cpu()` copy and
// moves the final adjusted scores back to the original device before any
// further tensor ops. This test exists specifically to catch a regression
// of that bug: it constructs CUDA tensors, runs them through the real
// function, and cross-checks the result against a CPU run with identical
// input VALUES (not identical tensors -- a CPU reference tensor and the
// CUDA input are both built from the same host-side float vectors, since
// compute_grpo_outcome_advantage requires same-device response_mask/scores
// internally and we want a completely independent reference pass).
//
// IMPORTANT / honesty note (see also csrc/core_algos/README.md): this file
// was written and syntax/API-checked by compiling it against a CPU-only
// libtorch build (no GPU was available in the environment this port was
// developed in). On that CPU-only build, `torch::cuda::is_available()`
// correctly reports `false` and the test exits early with code 77 (skipped)
// rather than silently reporting success -- this is NOT a substitute for
// running it on real CUDA hardware, only a guard against misreporting a
// skip as a pass. Actual GPU execution of this test still needs to be done
// on a CUDA-enabled libtorch + real GPU (e.g. the IP66 box used to find the
// original bug) before this fix can be considered GPU-verified.
//
// Exit codes: 0 = pass, 1 = fail (mismatch or crash-adjacent assertion
// failure), 2 = usage error, 77 = skipped (no CUDA available).

#include <cmath>
#include <iostream>
#include <vector>

#include "verl_core_algos/core_algos.h"

namespace {

constexpr double kTol = 1e-5;

bool nearly_equal(double a, double b, double tol = kTol) { return std::abs(a - b) <= tol; }

}  // namespace

int main() {
  if (!torch::cuda::is_available()) {
    std::cout << "[SKIP] CUDA not available in this environment -- this test must be run on a "
                 "CUDA-enabled libtorch + real GPU to actually validate the fix (see file header)."
              << std::endl;
    return 77;
  }

  const int64_t bsz = 8;
  const int64_t resp_len = 16;
  const double epsilon = 1e-6;

  // Same shape/edge-case spirit as the CPU fixtures: one ordinary group of
  // 4, one all-identical-reward (zero-variance) group of 4 -- the exact
  // pattern that previously segfaulted once scores lived on a CUDA device.
  std::vector<float> reward_values = {0.5f, -0.3f, 1.2f, 0.1f, 0.7f, 0.7f, 0.7f, 0.7f};
  std::vector<int64_t> index = {0, 0, 0, 0, 1, 1, 1, 1};

  torch::Tensor token_level_rewards_cpu = torch::zeros({bsz, resp_len}, torch::kFloat);
  torch::Tensor response_mask_cpu = torch::ones({bsz, resp_len}, torch::kFloat);
  for (int64_t i = 0; i < bsz; ++i) {
    token_level_rewards_cpu[i][resp_len - 1] = reward_values[i];
  }

  // Reference pass: identical input values, entirely on CPU.
  auto cpu_result = verl_core_algos::compute_grpo_outcome_advantage(token_level_rewards_cpu, response_mask_cpu,
                                                                     index, epsilon, /*norm_adv_by_std_in_grpo=*/true);

  // CUDA pass: same values, moved to device 0.
  torch::Tensor token_level_rewards_cuda = token_level_rewards_cpu.to(torch::kCUDA);
  torch::Tensor response_mask_cuda = response_mask_cpu.to(torch::kCUDA);

  auto cuda_result = verl_core_algos::compute_grpo_outcome_advantage(
      token_level_rewards_cuda, response_mask_cuda, index, epsilon, /*norm_adv_by_std_in_grpo=*/true);

  // If we get here at all, the segfault regression is gone (the old code
  // would have crashed the process before returning).
  std::cout << "[INFO] compute_grpo_outcome_advantage returned without crashing on CUDA tensors." << std::endl;

  TORCH_CHECK(cuda_result.advantages.device().is_cuda(), "expected CUDA result tensor to stay on a CUDA device");
  TORCH_CHECK(!torch::isnan(cuda_result.advantages).any().item<bool>(),
              "zero-variance group produced NaN advantages on CUDA");

  torch::Tensor cuda_advantages_on_cpu = cuda_result.advantages.cpu();
  double diff = (cuda_advantages_on_cpu - cpu_result.advantages).abs().max().item<double>();

  bool ok = std::isfinite(diff) && diff <= kTol;
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << "cuda_vs_cpu advantages max_abs_diff=" << diff << " (tol=" << kTol
            << ")" << std::endl;

  if (!ok) return 1;

  // Also exercise norm_adv_by_std_in_grpo=false on CUDA, since that branch
  // takes a different path through the same accessor-based loop.
  auto cuda_result_no_norm = verl_core_algos::compute_grpo_outcome_advantage(
      token_level_rewards_cuda, response_mask_cuda, index, epsilon, /*norm_adv_by_std_in_grpo=*/false);
  auto cpu_result_no_norm = verl_core_algos::compute_grpo_outcome_advantage(
      token_level_rewards_cpu, response_mask_cpu, index, epsilon, /*norm_adv_by_std_in_grpo=*/false);
  double diff_no_norm =
      (cuda_result_no_norm.advantages.cpu() - cpu_result_no_norm.advantages).abs().max().item<double>();
  bool ok_no_norm = std::isfinite(diff_no_norm) && diff_no_norm <= kTol;
  std::cout << (ok_no_norm ? "[PASS] " : "[FAIL] ") << "cuda_vs_cpu advantages (no_std_norm) max_abs_diff="
            << diff_no_norm << " (tol=" << kTol << ")" << std::endl;
  if (!ok_no_norm) return 1;

  std::cout << "\nALL CUDA SMOKE CHECKS PASSED" << std::endl;
  return 0;
}
