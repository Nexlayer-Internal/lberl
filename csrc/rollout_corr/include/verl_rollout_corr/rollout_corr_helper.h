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
// C++/libtorch port of `verl/trainer/ppo/rollout_corr_helper.py`
// (PLAN.md table item #2: off-policy / rollout-correction math -- truncated
// importance-sampling (IS) correction for rollout-vs-train policy mismatch,
// rejection sampling (RS), and the KL/PPL/chi^2 off-policy diagnostics).
//
// Scope (see csrc/rollout_corr/README.md for the full rationale): this file
// ports the pure-tensor-in/tensor-and-scalars-out compute functions. It does
// NOT port the two functions that construct/mutate a `DataProto` directly
// (`compute_rollout_correction_and_rejection_mask`,
// `compute_rollout_correction_and_add_to_batch`) or `apply_bypass_mode`
// (pure DataProto/omegaconf-config mutation, no tensor math at all) --
// those stay Python-side glue that calls into the functions below, matching
// the `core_algos.cpp` precedent of leaving DataProto-touching code as a
// boundary.
#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

namespace verl_rollout_corr {

// Safety bound to prevent numerical overflow/underflow when exponentiating.
// exp(20) ~= 485 million (upper limit for stable weights), exp(-20) ~= 2e-9
// (lower limit). Matches rollout_corr_helper.py's SAFETY_BOUND.
constexpr double kSafetyBound = 20.0;

// ---------------------------------------------------------------------------
// Threshold-spec parsing (rollout_corr_helper.py: _parse_rollout_is_threshold,
// _parse_rollout_rs_thresholds)
//
// Python accepts `str | float` for these threshold specs. Rather than
// replicating Python's dynamic isinstance() dispatch, `ThresholdSpec` is a
// small tagged union constructible from either a double or a string --
// equivalent semantics, type-safe at the C++ call site.
// ---------------------------------------------------------------------------

class ThresholdSpec {
 public:
  ThresholdSpec(double value) : value_(value) {}
  ThresholdSpec(const char* text) : text_(std::string(text)) {}
  ThresholdSpec(std::string text) : text_(std::move(text)) {}

  bool is_numeric() const { return value_.has_value(); }
  double numeric_value() const { return *value_; }
  const std::string& text_value() const { return *text_; }

 private:
  std::optional<double> value_;
  std::optional<std::string> text_;
};

struct IsThreshold {
  double upper;
  std::optional<double> lower;
};

// rollout_corr_helper.py: _parse_rollout_is_threshold
// A bare numeric spec is the upper (TIS) bound with no lower bound; a
// "lower_upper" string spec is the IcePop [lower, upper] band; a bare
// numeric string ("2.0") is equivalent to the numeric-upper-only case.
IsThreshold parse_rollout_is_threshold(const ThresholdSpec& threshold_spec);

struct RsThreshold {
  std::optional<double> lower;  // only set for "*_k1" options
  std::optional<double> upper;
};

// rollout_corr_helper.py: _parse_rollout_rs_thresholds
// `options` are already-validated, de-duplicated rollout_rs option names
// (e.g. {"token_k1", "seq_sum_k3"}); `threshold_spec` supplies either one
// shared threshold (numeric, or a single comma-free string) or exactly
// `options.size()` comma-separated entries.
std::unordered_map<std::string, RsThreshold> parse_rollout_rs_thresholds(
    const std::vector<std::string>& options, const std::optional<ThresholdSpec>& threshold_spec);

// ---------------------------------------------------------------------------
// Rejection sampling (rollout_corr_helper.py: compute_rollout_rejection_mask,
// compute_rs_metrics)
// ---------------------------------------------------------------------------

struct RejectionMaskResult {
  torch::Tensor modified_response_mask;
  std::unordered_map<std::string, double> metrics;
};

// `rollout_rs` is a comma-separated list of options from
// SUPPORTED_ROLLOUT_RS_OPTIONS (see the .cpp for the exact set); every listed
// option must pass for a token/sequence to remain valid (logical AND).
RejectionMaskResult compute_rollout_rejection_mask(const torch::Tensor& log_ratio,
                                                    const torch::Tensor& response_mask,
                                                    const std::string& rollout_rs,
                                                    const ThresholdSpec& rollout_rs_threshold);

// rollout_corr_helper.py: compute_rs_metrics. Per-option trust-region
// metrics; called internally by compute_rollout_rejection_mask for each
// option, and exposed standalone to match the Python module's public API
// (and so it can be fixture-tested independently).
//
// Honest note on dict-key presence: when `seq_values_valid` is non-empty
// AND `apply_lower_threshold` is false, the Python source simply never
// assigns `{prefix}_seq_fraction_low` into the returned dict (no `else`
// branch covers that combination) -- this port replicates that exactly: the
// key is *absent* from the returned map in that case, not present-with-0.0.
std::unordered_map<std::string, double> compute_rs_metrics(
    const std::string& option_name, const torch::Tensor& rs_statistic, const torch::Tensor& response_mask,
    const torch::Tensor& seq_valid_mask, const std::string& level,
    const std::optional<torch::Tensor>& per_sequence_values, double rollout_rs_threshold,
    double rollout_rs_threshold_lower, bool apply_lower_threshold);

// ---------------------------------------------------------------------------
// Importance-sampling (IS) weights (rollout_corr_helper.py:
// compute_rollout_correction_weights, compute_is_metrics)
// ---------------------------------------------------------------------------

struct IsWeightsResult {
  torch::Tensor rollout_is_weights;
  std::unordered_map<std::string, double> metrics;
};

// `rollout_is_batch_normalize`'s distributed branch (`torch.distributed`
// all-reduce across DP ranks) is orchestration, not pure tensor math --
// same "not this file's job" boundary as `core_algos.cpp`'s `agg_loss`
// requiring a pre-aggregated `batch_num_tokens`/`global_batch_size` when
// `dp_size > 1`. `distributed_weights_mean` plays the same role here: pass
// an already-all-reduced global mean when running multi-rank DP; leave it
// unset for the single-process case (this port's only directly-verified
// path), which computes the mean locally exactly like the Python
// non-distributed branch.
IsWeightsResult compute_rollout_correction_weights(const torch::Tensor& log_ratio,
                                                    const torch::Tensor& response_mask,
                                                    const std::string& rollout_is = "token",
                                                    const ThresholdSpec& rollout_is_threshold = ThresholdSpec(2.0),
                                                    bool rollout_is_batch_normalize = false,
                                                    std::optional<double> distributed_weights_mean = std::nullopt);

std::unordered_map<std::string, double> compute_is_metrics(
    const torch::Tensor& rollout_is_weights, const torch::Tensor& raw_rollout_is_weights,
    const torch::Tensor& log_ratio_for_metrics, const torch::Tensor& response_mask, const std::string& rollout_is,
    double rollout_is_threshold, std::optional<double> rollout_is_threshold_lower = std::nullopt);

// ---------------------------------------------------------------------------
// Off-policy diagnostics (rollout_corr_helper.py: compute_offpolicy_metrics,
// compute_rollout_corr_metrics_from_logprobs)
// ---------------------------------------------------------------------------

// `rollout_log_prob` unset mirrors Python's `rollout_log_prob=None`: only
// the training-policy perplexity metrics are returned.
std::unordered_map<std::string, double> compute_offpolicy_metrics(const torch::Tensor& old_log_prob,
                                                                    const std::optional<torch::Tensor>& rollout_log_prob,
                                                                    const torch::Tensor& response_mask);

// rollout_corr_helper.py: compute_rollout_corr_metrics_from_logprobs --
// compute_offpolicy_metrics with every key prefixed "rollout_corr/".
std::unordered_map<std::string, double> compute_rollout_corr_metrics_from_logprobs(const torch::Tensor& log_prob,
                                                                                    const torch::Tensor& rollout_log_prob,
                                                                                    const torch::Tensor& response_mask);

}  // namespace verl_rollout_corr
