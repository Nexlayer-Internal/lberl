// Copyright 2026 Bytedance Ltd. and/or its affiliates
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
// C++/libtorch port of `verl/trainer/ppo/score_centering.py` (PLAN.md table
// item #2's other half): training-inference-mismatch score centering per
// https://arxiv.org/abs/2609.20807. The sampler's top-k head is exact; its
// tail is modeled as the trainer's tail rescaled to the sampler's tail mass,
// so the centering term only needs the k head log-probs of the trainer. When
// the recorded head is verified to cover the whole sampling support, the
// trainer is renormalized over that support; otherwise the recorded head and
// trainer tail supply the predefined approximation.
//
// Scope (see csrc/rollout_corr/README.md for the full rationale): this file
// ports the actual math -- topk_log_probs_from_logits (incl. the custom
// autograd::Function), score_centering_weight_fn,
// score_centering_correction, score_centering_support_correction,
// score_centering_auto_correction -- plus dummy_rollout_topk/
// pad_rollout_topk, which turned out on inspection to be plain tensor math
// with no framework coupling (no DataProto/TensorDict/HF dependency) despite
// looking test-fixture-shaped from their names; see the README for why they
// were ported rather than left as glue. `score_centering_logits_processor`
// (TensorDict + ulysses sequence-parallel slicing + ActorConfig) and
// `score_centering_ppo_loss` (dispatches into the above plus the
// not-in-scope `ppo_loss`) are genuine framework glue and are NOT ported --
// they stay Python-side call sites for the functions below.
#pragma once

#include <functional>
#include <optional>
#include <string>

#include <torch/torch.h>

#include "verl_rollout_corr/rollout_corr_helper.h"

namespace verl_rollout_corr {

// score_centering.py: TOPK_LOG_PROB_CHUNK_SIZE
constexpr int64_t kTopkLogProbChunkSize = 4096;

// ---------------------------------------------------------------------------
// score_centering.py: dummy_rollout_topk / pad_rollout_topk
// ---------------------------------------------------------------------------

struct TopkHead {
  torch::Tensor ids;        // (rows, k), int32
  torch::Tensor log_probs;  // (rows, k), float32
};

// Uniform top-k head (ids 0..k-1, log-prob -log(k)) for rows the loss masks
// out (e.g. prompt or padding positions).
TopkHead dummy_rollout_topk(int64_t num_rows, int64_t k, torch::Device device = torch::kCPU);

// Lays the sampler's response-token heads over the unpadded prompt+response
// sequence: response token r is predicted by logits row
// `prompt_length - 1 + r`, so its sampler head goes there; other rows get
// the uniform dummy head.
TopkHead pad_rollout_topk(const torch::Tensor& response_topk_ids, const torch::Tensor& response_topk_log_probs,
                           int64_t prompt_length);

// ---------------------------------------------------------------------------
// score_centering.py: _TopKLogProbsFromLogits / topk_log_probs_from_logits
//
// Ported with torch::autograd::Function<T>, the same custom-C++-autograd-
// function pattern already validated in the sibling lbtorch project's
// torch::utils::checkpoint(). No per-element/raw accessor access anywhere in
// forward or backward -- every op here (max/gather/sub_/exp_/sum/log/
// scatter_add_) is a standard device-dispatched ATen op, so this is CUDA-safe
// by construction (see csrc/rollout_corr/README.md's CUDA-safety audit).
// ---------------------------------------------------------------------------

// log_softmax(logits, dim=-1).gather(-1, token_ids), computed chunk-by-chunk
// so the full (N, V) softmax workspace is never materialized at once (bounds
// extra fp32 workspace to (chunk_size, V)). Differentiable w.r.t. `logits`.
torch::Tensor topk_log_probs_from_logits(const torch::Tensor& logits, const torch::Tensor& token_ids,
                                          int64_t chunk_size = kTopkLogProbChunkSize);

// ---------------------------------------------------------------------------
// score_centering.py: score_centering_weight_fn / *_correction /
// *_support_correction / *_auto_correction
// ---------------------------------------------------------------------------

using WeightFn = std::function<torch::Tensor(const torch::Tensor&)>;

// Builds the token-level IS rule as a function of a ratio, shared by the
// head and the sampled token. `rollout_is` unset means no IS weighting
// (identity / ones_like); "token" with a plain-numeric threshold means TIS
// (upper clamp); a "lower_upper" threshold means IcePop (zero outside the
// band). Mirrors score_centering_weight_fn's 1:1 use of
// `_parse_rollout_is_threshold` (ported above as `parse_rollout_is_threshold`).
WeightFn score_centering_weight_fn(const std::optional<std::string>& rollout_is,
                                    const ThresholdSpec& rollout_is_threshold);

struct CorrectionResult {
  torch::Tensor correction;         // (N,), float32, grad to the trainer logits
  torch::Tensor sampler_head_mass;  // (N,), float32, q_mass
  torch::Tensor train_head_mass;    // (N,), float32, p_mass
};

// Per-token centering term `sum_H sg[q w - alpha p] log p` and the head
// masses; see score_centering.py's docstring for the full derivation
// (rho/alpha trainer-tail-rescaling argument). `train_head_log_probs` must
// carry gradient to the trainer logits (as produced by
// topk_log_probs_from_logits); the returned `correction` does too.
CorrectionResult score_centering_correction(const torch::Tensor& train_head_log_probs,
                                             const torch::Tensor& sampler_head_log_probs, const WeightFn& weight_fn,
                                             double eps = 1e-6);

struct SupportCorrectionResult {
  torch::Tensor correction;         // (N,), float32, grad to the trainer logits
  torch::Tensor log_support_mass;   // (N,), float32, grad to the trainer logits (log p(S))
  torch::Tensor sampler_head_mass;  // (N,), float32
};

// Centering term over the sampler's support held in the head (slots with
// -inf sampler log-prob lie outside the support S); the trainer is
// renormalized over S. See score_centering.py's docstring.
SupportCorrectionResult score_centering_support_correction(const torch::Tensor& train_head_log_probs,
                                                            const torch::Tensor& sampler_head_log_probs,
                                                            const WeightFn& weight_fn);

struct AutoCorrectionResult {
  torch::Tensor correction;          // (N,), float32, grad to the trainer logits
  torch::Tensor log_norm_shift;      // (N,), float32: log_support_mass where aligned, else 0
  torch::Tensor sampler_head_mass;   // (N,), float32 (from the unconditional correction call)
  torch::Tensor train_head_mass;     // (N,), float32: trainer mass on the recorded support/head
};

// Aligns to verified support per token (via `support_complete`); otherwise
// falls back to the trainer-tail approximation. See score_centering.py's
// docstring.
AutoCorrectionResult score_centering_auto_correction(const torch::Tensor& train_head_log_probs,
                                                      const torch::Tensor& sampler_head_log_probs,
                                                      const WeightFn& weight_fn,
                                                      const torch::Tensor& support_complete);

}  // namespace verl_rollout_corr
