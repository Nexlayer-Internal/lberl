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
// C++/libtorch port of the GRPO advantage computation, PPO-clip policy loss,
// and KL-penalty math from `verl/trainer/ppo/core_algos.py`.
//
// Scope (see PLAN.md table item #1 / section 6, and the port's own README):
// this file ports ONLY the functions that are the literal algorithm
// executed every training step for the validated GRPO config
// (`algorithm.adv_estimator=grpo`, `actor_rollout_ref.actor.policy_loss.loss_mode=vanilla`):
//   - compute_grpo_outcome_advantage   (core_algos.py, registered AdvantageEstimator.GRPO)
//   - agg_loss                        (core_algos.py, loss aggregation used by the policy loss)
//   - compute_policy_loss_vanilla      (core_algos.py, registered "vanilla" policy loss: PPO
//                                       clipping + dual-clip, source of actor/pg_clipfrac,
//                                       actor/ppo_kl, actor/pg_clipfrac_lower, actor/pg_loss)
//   - kl_penalty / kl_penalty_forward  (core_algos.py, k1/k2/k3/abs KL estimators + the k3+
//                                       straight-through gradient trick)
//   - compute_rewards                 (core_algos.py, applies the KL penalty term to the
//                                       token-level reward before advantage computation)
// plus the two small masked-reduction helpers (`masked_sum`/`masked_mean` from
// `verl/utils/torch_functional.py`) these functions depend on.
//
// Deliberately NOT ported here (see csrc/core_algos/README.md for the full
// rationale): the 15+ other advantage-estimator/policy-loss variants in the
// same Python file (GAE, RLOO, GSPO, DAPO clip-cov, CISPO, etc. — not used by
// the validated GRPO config), and `rollout_corr_helper.py`/`score_centering.py`
// (PLAN.md table item #2, off-policy/rollout-correction math) — real
// "pure tensor math" per the plan, but ~1500 lines of additional surface
// relative to the ~250 lines ported here, so left for a follow-up port rather
// than folded into this one.
#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

namespace verl_core_algos {

// ---------------------------------------------------------------------------
// Masked reduction helpers (verl/utils/torch_functional.py: masked_sum, masked_mean)
// ---------------------------------------------------------------------------

// Sum of `values` where `mask` is nonzero. NaNs outside the mask are zeroed
// first so they cannot contaminate the sum, matching the Python implementation
// exactly (`torch.where(mask.bool(), values, 0.0)` then `(valid * mask).sum(dim)`).
torch::Tensor masked_sum(const torch::Tensor& values, const torch::Tensor& mask,
                          std::optional<std::vector<int64_t>> dim = std::nullopt);

// masked_sum(values, mask, dim) / (mask.sum(dim) + 1e-8)
torch::Tensor masked_mean(const torch::Tensor& values, const torch::Tensor& mask,
                           std::optional<std::vector<int64_t>> dim = std::nullopt);

// ---------------------------------------------------------------------------
// GRPO outcome-advantage (core_algos.py: compute_grpo_outcome_advantage, L268)
// ---------------------------------------------------------------------------

struct GrpoAdvantageResult {
  torch::Tensor advantages;  // (bsz, response_length)
  torch::Tensor returns;     // == advantages, for GRPO's outcome formulation
};

// `index` is a group id per sample (bsz,), analogous to the Python `np.ndarray`
// index array used to bucket samples into GRPO groups (all rollouts sampled
// for the same prompt share the same group id). Grouping/mean/std computation
// mirrors the Python implementation's per-group Python loop exactly, including
// the size-1 special case (mean=0, std=1, so no NaN from zero-variance /
// zero-size normalization) and `torch.std`'s unbiased (N-1 denominator)
// estimator for groups of size >= 2.
GrpoAdvantageResult compute_grpo_outcome_advantage(const torch::Tensor& token_level_rewards,
                                                    const torch::Tensor& response_mask,
                                                    const std::vector<int64_t>& index,
                                                    double epsilon = 1e-6,
                                                    bool norm_adv_by_std_in_grpo = true);

// ---------------------------------------------------------------------------
// Loss aggregation (core_algos.py: agg_loss, L1138)
// ---------------------------------------------------------------------------

enum class LossAggMode {
  kTokenMean,
  kSeqMeanTokenSum,
  kSeqMeanTokenSumNorm,
  kSeqMeanTokenMean,
};

LossAggMode parse_loss_agg_mode(const std::string& name);

struct AggLossOptions {
  int64_t dp_size = 1;
  std::optional<double> batch_num_tokens = std::nullopt;
  std::optional<double> global_batch_size = std::nullopt;
  std::optional<double> loss_scale_factor = std::nullopt;
};

torch::Tensor agg_loss(const torch::Tensor& loss_mat, const torch::Tensor& loss_mask,
                        LossAggMode loss_agg_mode, const AggLossOptions& opts = {});

// ---------------------------------------------------------------------------
// PPO-clip / dual-clip policy loss (core_algos.py: compute_policy_loss_vanilla, L1278)
// ---------------------------------------------------------------------------

struct PolicyLossConfig {
  double clip_ratio;
  std::optional<double> clip_ratio_low = std::nullopt;   // defaults to clip_ratio
  std::optional<double> clip_ratio_high = std::nullopt;  // defaults to clip_ratio
  double clip_ratio_c = 3.0;                             // dual-clip lower bound, must be > 1.0
  LossAggMode loss_agg_mode = LossAggMode::kTokenMean;
  AggLossOptions agg_opts = {};
};

struct PolicyLossResult {
  torch::Tensor pg_loss;  // scalar
  // Metric family matching the real GSM8K smoke-test logs:
  // actor/pg_clipfrac, actor/ppo_kl, actor/pg_clipfrac_lower.
  double pg_clipfrac;
  double ppo_kl;
  double pg_clipfrac_lower;
};

PolicyLossResult compute_policy_loss_vanilla(const torch::Tensor& old_log_prob,
                                              const torch::Tensor& log_prob,
                                              const torch::Tensor& advantages,
                                              const torch::Tensor& response_mask,
                                              const PolicyLossConfig& config,
                                              const std::optional<torch::Tensor>& rollout_is_weights =
                                                  std::nullopt);

// ---------------------------------------------------------------------------
// KL penalty estimators (core_algos.py: kl_penalty / kl_penalty_forward, L2126/L2154)
// ---------------------------------------------------------------------------

// Per-token KL estimate for one of: "kl"/"k1", "abs", "mse"/"k2", "low_var_kl"/"k3".
// ("full" and anything else raise, matching the Python `NotImplementedError`.)
torch::Tensor kl_penalty_forward(const torch::Tensor& logprob, const torch::Tensor& ref_logprob,
                                  const std::string& kl_penalty_kind);

// Same estimators, plus the "<kind>+" straight-through variant (e.g. "k3+"):
// forward value uses the named estimator, backward gradient uses the k2/mse
// estimator's gradient (the unbiased-gradient trick from the Python
// implementation's docstring).
torch::Tensor kl_penalty(const torch::Tensor& logprob, const torch::Tensor& ref_logprob,
                          const std::string& kl_penalty_kind);

// core_algos.py: compute_rewards, L1122 — applies the KL penalty term to the
// token-level reward: token_level_scores - (old_log_prob - ref_log_prob) * kl_ratio.
torch::Tensor compute_rewards(const torch::Tensor& token_level_scores, const torch::Tensor& old_log_prob,
                               const torch::Tensor& ref_log_prob, double kl_ratio);

}  // namespace verl_core_algos
