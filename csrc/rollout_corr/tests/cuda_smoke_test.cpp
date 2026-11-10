// Copyright 2025 Bytedance Ltd. and/or its affiliates
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
// CUDA-tensor smoke test for every function ported in csrc/rollout_corr.
//
// Background (see csrc/core_algos/README.md's "CUDA fix" section for the
// precedent this guards against): a CPU-only accessor<T,N>() call on a
// tensor that could be on CUDA silently misbehaves/segfaults rather than
// raising. This module's entire implementation was specifically audited for
// that pattern -- `grep -n '\.accessor<\|\.data_ptr<' csrc/rollout_corr/src/*.cpp`
// returns nothing: every raw-value extraction in both rollout_corr_helper.cpp
// and score_centering.cpp is `.item<double>()`/`.item<bool>()` on a 0-dim
// reduction tensor (device-to-host copy handled internally, safe on any
// device) or a standard device-dispatched ATen op (masked_fill, gather,
// scatter_add_, logsumexp, etc.). There is therefore no single "this exact
// line segfaults on CUDA" bug this test is chasing (unlike core_algos's); it
// instead exists as a broad regression guard that every ported function
// actually runs on CUDA tensors and produces identical results to an
// identical-value CPU run, so that invariant doesn't silently rot later.
//
// IMPORTANT / honesty note: written and compile/link-checked against a
// CPU-only libtorch (no GPU was available in the environment this port was
// developed in; same constraint csrc/core_algos's cuda_smoke_test.cpp
// documents). On that build, torch::cuda::is_available() reports false and
// the test exits early with code 77 (skipped) rather than silently
// reporting success. Actual GPU execution still needs to happen on a
// CUDA-enabled libtorch + real GPU before this is GPU-verified, not just
// compile-verified -- see this PR's description for the explicit ask to the
// coordinator to do that on real hardware (IP66) before merge.
//
// Exit codes: 0 = pass, 1 = fail, 2 = usage error, 77 = skipped (no CUDA).

#include <cmath>
#include <iostream>
#include <vector>

#include "verl_rollout_corr/rollout_corr_helper.h"
#include "verl_rollout_corr/score_centering.h"

using verl_rollout_corr::ThresholdSpec;

namespace {

constexpr double kTol = 1e-4;
bool g_any_failed = false;

double max_abs_diff(const torch::Tensor& a, const torch::Tensor& b) {
  if (a.numel() == 0) return 0.0;
  torch::Tensor diff = (a.cpu() - b.cpu()).abs();
  torch::Tensor both_inf = a.cpu().isinf() & b.cpu().isinf() & (a.cpu().sign() == b.cpu().sign());
  diff = torch::where(both_inf, torch::zeros_like(diff), diff);
  return diff.max().item<double>();
}

void check(const std::string& name, double diff) {
  bool ok = std::isfinite(diff) && diff <= kTol;
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "  max_abs_diff=" << diff << std::endl;
  if (!ok) g_any_failed = true;
}

void check_metrics_match(const std::string& name, const std::unordered_map<std::string, double>& cuda_metrics,
                          const std::unordered_map<std::string, double>& cpu_metrics) {
  double max_diff = 0.0;
  bool keys_ok = cuda_metrics.size() == cpu_metrics.size();
  for (const auto& kv : cpu_metrics) {
    auto it = cuda_metrics.find(kv.first);
    if (it == cuda_metrics.end()) {
      keys_ok = false;
      continue;
    }
    max_diff = std::max(max_diff, std::abs(it->second - kv.second));
  }
  bool ok = keys_ok && std::isfinite(max_diff) && max_diff <= kTol;
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << " (metrics)  max_abs_diff=" << max_diff
            << (keys_ok ? "" : "  KEY MISMATCH") << std::endl;
  if (!ok) g_any_failed = true;
}

}  // namespace

int main() {
  if (!torch::cuda::is_available()) {
    std::cout << "[SKIP] CUDA not available in this environment -- this test must be run on a "
                 "CUDA-enabled libtorch + real GPU to actually validate CUDA execution (see file header)."
              << std::endl;
    return 77;
  }

  torch::manual_seed(0);
  const int64_t bsz = 8, resp_len = 16;
  std::vector<int64_t> lengths = {16, 14, 10, 16, 8, 12, 6, 16};

  torch::Tensor response_mask_cpu = torch::zeros({bsz, resp_len});
  for (int64_t i = 0; i < bsz; ++i) response_mask_cpu[i].slice(0, 0, lengths[i]).fill_(1.0);
  torch::Tensor log_ratio_cpu = (torch::rand({bsz, resp_len}) - 0.5) * 1.2;

  torch::Tensor response_mask_cuda = response_mask_cpu.to(torch::kCUDA);
  torch::Tensor log_ratio_cuda = log_ratio_cpu.to(torch::kCUDA);

  // --- compute_rollout_rejection_mask ---
  {
    auto cpu_r = verl_rollout_corr::compute_rollout_rejection_mask(log_ratio_cpu, response_mask_cpu, "token_k1",
                                                                     ThresholdSpec("0.5_1.8"));
    auto cuda_r = verl_rollout_corr::compute_rollout_rejection_mask(log_ratio_cuda, response_mask_cuda, "token_k1",
                                                                      ThresholdSpec("0.5_1.8"));
    TORCH_CHECK(cuda_r.modified_response_mask.device().is_cuda(), "expected CUDA result to stay on CUDA device");
    check("rejection_mask (mask)", max_abs_diff(cuda_r.modified_response_mask, cpu_r.modified_response_mask));
    check_metrics_match("rejection_mask", cuda_r.metrics, cpu_r.metrics);
  }

  // --- compute_rs_metrics (standalone) ---
  {
    torch::Tensor seq_valid_mask_cpu = response_mask_cpu.sum(-1) > 0;
    torch::Tensor seq_valid_mask_cuda = seq_valid_mask_cpu.to(torch::kCUDA);
    torch::Tensor rs_stat_cpu = log_ratio_cpu.abs();
    torch::Tensor rs_stat_cuda = log_ratio_cuda.abs();
    auto cpu_m = verl_rollout_corr::compute_rs_metrics("token_k1", rs_stat_cpu, response_mask_cpu, seq_valid_mask_cpu,
                                                         "token", std::nullopt, 0.6, 0.0, false);
    auto cuda_m = verl_rollout_corr::compute_rs_metrics("token_k1", rs_stat_cuda, response_mask_cuda,
                                                          seq_valid_mask_cuda, "token", std::nullopt, 0.6, 0.0, false);
    check_metrics_match("rs_metrics", cuda_m, cpu_m);
  }

  // --- compute_rollout_correction_weights / compute_is_metrics ---
  {
    auto cpu_w = verl_rollout_corr::compute_rollout_correction_weights(log_ratio_cpu, response_mask_cpu, "token",
                                                                         ThresholdSpec(2.0), /*batch_normalize=*/true);
    auto cuda_w = verl_rollout_corr::compute_rollout_correction_weights(
        log_ratio_cuda, response_mask_cuda, "token", ThresholdSpec(2.0), /*batch_normalize=*/true);
    TORCH_CHECK(cuda_w.rollout_is_weights.device().is_cuda(), "expected CUDA result to stay on CUDA device");
    check("is_weights (weights)", max_abs_diff(cuda_w.rollout_is_weights, cpu_w.rollout_is_weights));
    check_metrics_match("is_weights", cuda_w.metrics, cpu_w.metrics);

    auto cpu_seq_w = verl_rollout_corr::compute_rollout_correction_weights(log_ratio_cpu, response_mask_cpu,
                                                                             "sequence", ThresholdSpec("0.3_2.5"));
    auto cuda_seq_w = verl_rollout_corr::compute_rollout_correction_weights(log_ratio_cuda, response_mask_cuda,
                                                                              "sequence", ThresholdSpec("0.3_2.5"));
    check("is_weights_sequence (weights)", max_abs_diff(cuda_seq_w.rollout_is_weights, cpu_seq_w.rollout_is_weights));
    check_metrics_match("is_weights_sequence", cuda_seq_w.metrics, cpu_seq_w.metrics);
  }

  // --- compute_offpolicy_metrics / compute_rollout_corr_metrics_from_logprobs ---
  {
    torch::Tensor old_lp_cpu = -torch::rand({bsz, resp_len}) * 2.0 - 0.1;
    torch::Tensor rollout_lp_cpu = old_lp_cpu + (torch::rand({bsz, resp_len}) - 0.5) * 0.8;
    torch::Tensor old_lp_cuda = old_lp_cpu.to(torch::kCUDA);
    torch::Tensor rollout_lp_cuda = rollout_lp_cpu.to(torch::kCUDA);

    auto cpu_m = verl_rollout_corr::compute_offpolicy_metrics(old_lp_cpu, rollout_lp_cpu, response_mask_cpu);
    auto cuda_m = verl_rollout_corr::compute_offpolicy_metrics(old_lp_cuda, rollout_lp_cuda, response_mask_cuda);
    check_metrics_match("offpolicy_metrics", cuda_m, cpu_m);

    auto cpu_prefixed =
        verl_rollout_corr::compute_rollout_corr_metrics_from_logprobs(old_lp_cpu, rollout_lp_cpu, response_mask_cpu);
    auto cuda_prefixed = verl_rollout_corr::compute_rollout_corr_metrics_from_logprobs(old_lp_cuda, rollout_lp_cuda,
                                                                                         response_mask_cuda);
    check_metrics_match("corr_metrics_from_logprobs", cuda_prefixed, cpu_prefixed);
  }

  // --- score_centering: dummy_rollout_topk / pad_rollout_topk ---
  {
    auto cpu_head = verl_rollout_corr::dummy_rollout_topk(5, 4, torch::kCPU);
    auto cuda_head = verl_rollout_corr::dummy_rollout_topk(5, 4, torch::kCUDA);
    TORCH_CHECK(cuda_head.ids.device().is_cuda(), "expected CUDA result to stay on CUDA device");
    check("dummy_topk (ids)", max_abs_diff(cuda_head.ids.to(torch::kFloat), cpu_head.ids.to(torch::kFloat)));
    check("dummy_topk (log_probs)", max_abs_diff(cuda_head.log_probs, cpu_head.log_probs));

    torch::Tensor resp_ids_cpu = torch::randint(0, 100, {5, 4});
    torch::Tensor resp_lp_cpu = -torch::rand({5, 4}) * 3.0;
    torch::Tensor resp_ids_cuda = resp_ids_cpu.to(torch::kCUDA);
    torch::Tensor resp_lp_cuda = resp_lp_cpu.to(torch::kCUDA);
    auto cpu_pad = verl_rollout_corr::pad_rollout_topk(resp_ids_cpu, resp_lp_cpu, /*prompt_length=*/7);
    auto cuda_pad = verl_rollout_corr::pad_rollout_topk(resp_ids_cuda, resp_lp_cuda, /*prompt_length=*/7);
    TORCH_CHECK(cuda_pad.ids.device().is_cuda(), "expected CUDA result to stay on CUDA device");
    check("pad_topk (ids)", max_abs_diff(cuda_pad.ids.to(torch::kFloat), cpu_pad.ids.to(torch::kFloat)));
    check("pad_topk (log_probs)", max_abs_diff(cuda_pad.log_probs, cpu_pad.log_probs));
  }

  // --- score_centering: topk_log_probs_from_logits (forward + backward) ---
  {
    const int64_t n = 10, v = 64, k = 4;
    torch::Tensor logits_cpu = torch::randn({n, v});
    torch::Tensor token_ids_cpu = torch::randint(0, v, {n, k});
    torch::Tensor grad_output_cpu = torch::randn({n, k});

    torch::Tensor logits_cpu_leaf = logits_cpu.clone().set_requires_grad(true);
    torch::Tensor out_cpu = verl_rollout_corr::topk_log_probs_from_logits(logits_cpu_leaf, token_ids_cpu, /*chunk_size=*/3);
    out_cpu.backward(grad_output_cpu);

    torch::Tensor logits_cuda_leaf = logits_cpu.to(torch::kCUDA).set_requires_grad(true);
    torch::Tensor token_ids_cuda = token_ids_cpu.to(torch::kCUDA);
    torch::Tensor grad_output_cuda = grad_output_cpu.to(torch::kCUDA);
    torch::Tensor out_cuda =
        verl_rollout_corr::topk_log_probs_from_logits(logits_cuda_leaf, token_ids_cuda, /*chunk_size=*/3);
    out_cuda.backward(grad_output_cuda);

    TORCH_CHECK(out_cuda.device().is_cuda(), "expected CUDA result to stay on CUDA device");
    check("topk_log_probs (forward)", max_abs_diff(out_cuda, out_cpu));
    check("topk_log_probs (grad)", max_abs_diff(logits_cuda_leaf.grad(), logits_cpu_leaf.grad()));
  }

  // --- score_centering_correction / *_support_correction / *_auto_correction ---
  {
    const int64_t n = 10, k = 6, vocab = 64;
    torch::Tensor logits_a = torch::randn({n, vocab});
    torch::Tensor logits_b = torch::randn({n, vocab});
    torch::Tensor train_head_cpu = std::get<0>(torch::log_softmax(logits_a, -1).topk(k, -1));
    torch::Tensor sampler_head_cpu = std::get<0>(torch::log_softmax(logits_b, -1).topk(k, -1));

    auto weight_fn_cpu = verl_rollout_corr::score_centering_weight_fn("token", ThresholdSpec(2.0));
    auto weight_fn_cuda = verl_rollout_corr::score_centering_weight_fn("token", ThresholdSpec(2.0));

    torch::Tensor train_cpu_leaf = train_head_cpu.clone().set_requires_grad(true);
    auto cpu_corr = verl_rollout_corr::score_centering_correction(train_cpu_leaf, sampler_head_cpu, weight_fn_cpu);
    cpu_corr.correction.sum().backward();

    torch::Tensor train_cuda_leaf = train_head_cpu.to(torch::kCUDA).set_requires_grad(true);
    torch::Tensor sampler_head_cuda = sampler_head_cpu.to(torch::kCUDA);
    auto cuda_corr = verl_rollout_corr::score_centering_correction(train_cuda_leaf, sampler_head_cuda, weight_fn_cuda);
    cuda_corr.correction.sum().backward();

    TORCH_CHECK(cuda_corr.correction.device().is_cuda(), "expected CUDA result to stay on CUDA device");
    check("score_centering_correction (value)", max_abs_diff(cuda_corr.correction, cpu_corr.correction));
    check("score_centering_correction (grad)", max_abs_diff(train_cuda_leaf.grad(), train_cpu_leaf.grad()));

    // Incomplete support (-inf in the last head slot) -- the branch most
    // likely to interact badly with a CUDA-only masked_fill/logsumexp if
    // anything here were CPU-only.
    torch::Tensor sampler_incomplete_cpu = sampler_head_cpu.clone();
    sampler_incomplete_cpu.index_put_({torch::indexing::Slice(), -1}, -std::numeric_limits<float>::infinity());
    torch::Tensor sampler_incomplete_cuda = sampler_incomplete_cpu.to(torch::kCUDA);

    torch::Tensor train_cpu_leaf2 = train_head_cpu.clone().set_requires_grad(true);
    auto cpu_support = verl_rollout_corr::score_centering_support_correction(train_cpu_leaf2, sampler_incomplete_cpu,
                                                                               weight_fn_cpu);
    (cpu_support.correction.sum() + cpu_support.log_support_mass.sum()).backward();

    torch::Tensor train_cuda_leaf2 = train_head_cpu.to(torch::kCUDA).set_requires_grad(true);
    auto cuda_support = verl_rollout_corr::score_centering_support_correction(train_cuda_leaf2, sampler_incomplete_cuda,
                                                                                weight_fn_cuda);
    (cuda_support.correction.sum() + cuda_support.log_support_mass.sum()).backward();

    TORCH_CHECK(cuda_support.correction.device().is_cuda(), "expected CUDA result to stay on CUDA device");
    check("score_centering_support_correction (value)", max_abs_diff(cuda_support.correction, cpu_support.correction));
    check("score_centering_support_correction (log_support_mass)",
          max_abs_diff(cuda_support.log_support_mass, cpu_support.log_support_mass));
    check("score_centering_support_correction (grad)", max_abs_diff(train_cuda_leaf2.grad(), train_cpu_leaf2.grad()));

    torch::Tensor support_complete_cpu =
        torch::tensor({true, false, true, false, true, false, true, false, true, false});
    torch::Tensor support_complete_cuda = support_complete_cpu.to(torch::kCUDA);

    torch::Tensor train_cpu_leaf3 = train_head_cpu.clone().set_requires_grad(true);
    auto cpu_auto = verl_rollout_corr::score_centering_auto_correction(train_cpu_leaf3, sampler_incomplete_cpu,
                                                                         weight_fn_cpu, support_complete_cpu);
    (cpu_auto.correction.sum() + cpu_auto.log_norm_shift.sum()).backward();

    torch::Tensor train_cuda_leaf3 = train_head_cpu.to(torch::kCUDA).set_requires_grad(true);
    auto cuda_auto = verl_rollout_corr::score_centering_auto_correction(
        train_cuda_leaf3, sampler_incomplete_cuda, weight_fn_cuda, support_complete_cuda);
    (cuda_auto.correction.sum() + cuda_auto.log_norm_shift.sum()).backward();

    TORCH_CHECK(cuda_auto.correction.device().is_cuda(), "expected CUDA result to stay on CUDA device");
    check("score_centering_auto_correction (correction)", max_abs_diff(cuda_auto.correction, cpu_auto.correction));
    check("score_centering_auto_correction (log_norm_shift)",
          max_abs_diff(cuda_auto.log_norm_shift, cpu_auto.log_norm_shift));
    check("score_centering_auto_correction (train_head_mass)",
          max_abs_diff(cuda_auto.train_head_mass, cpu_auto.train_head_mass));
    check("score_centering_auto_correction (grad)", max_abs_diff(train_cuda_leaf3.grad(), train_cpu_leaf3.grad()));
  }

  if (g_any_failed) {
    std::cout << "\nSOME CUDA SMOKE CHECKS FAILED" << std::endl;
    return 1;
  }
  std::cout << "\nALL CUDA SMOKE CHECKS PASSED" << std::endl;
  return 0;
}
