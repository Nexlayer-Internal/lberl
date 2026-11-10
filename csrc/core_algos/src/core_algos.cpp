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

#include "verl_core_algos/core_algos.h"

#include <stdexcept>

namespace verl_core_algos {

namespace {

torch::IntArrayRef dim_or_all(const std::optional<std::vector<int64_t>>& dim) {
  static const std::vector<int64_t> kEmpty;
  return dim.has_value() ? torch::IntArrayRef(*dim) : torch::IntArrayRef(kEmpty);
}

}  // namespace

torch::Tensor masked_sum(const torch::Tensor& values, const torch::Tensor& mask,
                          std::optional<std::vector<int64_t>> dim) {
  // valid_values = torch.where(mask.bool(), values, 0.0)
  torch::Tensor valid_values = torch::where(mask.to(torch::kBool), values, torch::zeros_like(values));
  torch::Tensor masked = valid_values * mask;
  if (!dim.has_value()) {
    return masked.sum();
  }
  return masked.sum(dim_or_all(dim));
}

torch::Tensor masked_mean(const torch::Tensor& values, const torch::Tensor& mask,
                           std::optional<std::vector<int64_t>> dim) {
  torch::Tensor s = masked_sum(values, mask, dim);
  torch::Tensor denom = dim.has_value() ? mask.sum(dim_or_all(dim)) : mask.sum();
  return s / (denom + 1e-8);
}

GrpoAdvantageResult compute_grpo_outcome_advantage(const torch::Tensor& token_level_rewards,
                                                    const torch::Tensor& response_mask,
                                                    const std::vector<int64_t>& index, double epsilon,
                                                    bool norm_adv_by_std_in_grpo) {
  torch::NoGradGuard no_grad;

  // scores = token_level_rewards.sum(dim=-1)
  torch::Tensor scores = token_level_rewards.sum(/*dim=*/-1).clone();
  int64_t bsz = scores.size(0);
  TORCH_CHECK(static_cast<int64_t>(index.size()) == bsz,
              "index must have one entry per sample (bsz=", bsz, ", index.size()=", index.size(), ")");

  // Group samples by index, exactly mirroring the Python defaultdict(list) loop.
  std::unordered_map<int64_t, std::vector<int64_t>> id2rows;
  id2rows.reserve(bsz);
  for (int64_t i = 0; i < bsz; ++i) {
    id2rows[index[i]].push_back(i);
  }

  std::unordered_map<int64_t, double> id2mean;
  std::unordered_map<int64_t, double> id2std;

  // Tensor::accessor<T,N>() assumes host-dereferenceable memory; it is
  // CPU-only and will segfault (silently corrupt, in the absence of a
  // debugger) if `scores` lives on a CUDA device. The per-group mean/std
  // loop below is inherently host-side bookkeeping (mirroring Python's
  // defaultdict loop) regardless of where the caller's tensors live, so do
  // it on an explicit CPU copy and move the result back to the original
  // device before using it in any further tensor ops.
  const torch::Device orig_device = scores.device();
  torch::Tensor scores_cpu = scores.cpu().contiguous();
  auto scores_acc = scores_cpu.accessor<float, 1>();

  for (const auto& kv : id2rows) {
    const int64_t group_id = kv.first;
    const std::vector<int64_t>& rows = kv.second;
    const int64_t n = static_cast<int64_t>(rows.size());
    if (n == 1) {
      id2mean[group_id] = 0.0;
      id2std[group_id] = 1.0;
    } else if (n > 1) {
      double sum = 0.0;
      for (int64_t r : rows) sum += static_cast<double>(scores_acc[r]);
      double mean = sum / static_cast<double>(n);
      double sq = 0.0;
      for (int64_t r : rows) {
        double d = static_cast<double>(scores_acc[r]) - mean;
        sq += d * d;
      }
      // torch.std default is unbiased (N-1 denominator) — match exactly.
      double variance = sq / static_cast<double>(n - 1);
      id2mean[group_id] = mean;
      id2std[group_id] = std::sqrt(variance);
    } else {
      throw std::runtime_error("no score in prompt index: " + std::to_string(group_id));
    }
  }

  for (int64_t i = 0; i < bsz; ++i) {
    const int64_t group_id = index[i];
    if (norm_adv_by_std_in_grpo) {
      scores_acc[i] = static_cast<float>((static_cast<double>(scores_acc[i]) - id2mean[group_id]) /
                                          (id2std[group_id] + epsilon));
    } else {
      scores_acc[i] = static_cast<float>(static_cast<double>(scores_acc[i]) - id2mean[group_id]);
    }
  }

  scores = scores_cpu.to(orig_device);
  torch::Tensor advantages = scores.unsqueeze(-1) * response_mask;
  return {advantages, advantages};
}

LossAggMode parse_loss_agg_mode(const std::string& name) {
  if (name == "token-mean") return LossAggMode::kTokenMean;
  if (name == "seq-mean-token-sum") return LossAggMode::kSeqMeanTokenSum;
  if (name == "seq-mean-token-sum-norm") return LossAggMode::kSeqMeanTokenSumNorm;
  if (name == "seq-mean-token-mean") return LossAggMode::kSeqMeanTokenMean;
  throw std::invalid_argument("Invalid loss_agg_mode: " + name);
}

torch::Tensor agg_loss(const torch::Tensor& loss_mat, const torch::Tensor& loss_mask, LossAggMode loss_agg_mode,
                        const AggLossOptions& opts) {
  switch (loss_agg_mode) {
    case LossAggMode::kTokenMean: {
      double batch_num_tokens;
      if (opts.batch_num_tokens.has_value()) {
        batch_num_tokens = *opts.batch_num_tokens;
      } else {
        TORCH_CHECK(opts.dp_size <= 1, "(global) batch_num_tokens is required when dp_size > 1");
        batch_num_tokens = loss_mask.sum().item<double>();
      }
      return masked_sum(loss_mat, loss_mask) / batch_num_tokens * static_cast<double>(opts.dp_size);
    }
    case LossAggMode::kSeqMeanTokenSum:
    case LossAggMode::kSeqMeanTokenSumNorm: {
      torch::Tensor seq_losses = (loss_mat * loss_mask).sum(-1);
      torch::Tensor seq_mask = (loss_mask.sum(-1) > 0).to(loss_mat.scalar_type());
      double global_batch_size;
      if (opts.global_batch_size.has_value()) {
        global_batch_size = *opts.global_batch_size;
      } else {
        TORCH_CHECK(opts.dp_size <= 1, "global_batch_size is required when dp_size > 1");
        global_batch_size = seq_mask.sum().item<double>();
      }
      torch::Tensor loss = masked_sum(seq_losses, seq_mask) / global_batch_size * static_cast<double>(opts.dp_size);
      if (loss_agg_mode == LossAggMode::kSeqMeanTokenSumNorm) {
        double loss_scale_factor =
            opts.loss_scale_factor.has_value() ? *opts.loss_scale_factor : static_cast<double>(loss_mask.size(-1));
        loss = loss / loss_scale_factor;
      }
      return loss;
    }
    case LossAggMode::kSeqMeanTokenMean: {
      torch::Tensor seq_token_count = loss_mask.sum(-1);
      torch::Tensor seq_losses = (loss_mat * loss_mask).sum(-1) / (seq_token_count + 1e-8);
      torch::Tensor seq_mask = (seq_token_count > 0).to(loss_mat.scalar_type());
      double global_batch_size;
      if (opts.global_batch_size.has_value()) {
        global_batch_size = *opts.global_batch_size;
      } else {
        TORCH_CHECK(opts.dp_size <= 1, "global_batch_size is required when dp_size > 1");
        global_batch_size = seq_mask.sum().item<double>();
      }
      return masked_sum(seq_losses, seq_mask) / global_batch_size * static_cast<double>(opts.dp_size);
    }
  }
  throw std::invalid_argument("Unhandled LossAggMode");
}

PolicyLossResult compute_policy_loss_vanilla(const torch::Tensor& old_log_prob, const torch::Tensor& log_prob,
                                              const torch::Tensor& advantages, const torch::Tensor& response_mask,
                                              const PolicyLossConfig& config,
                                              const std::optional<torch::Tensor>& rollout_is_weights) {
  const double clip_ratio = config.clip_ratio;
  const double clip_ratio_low = config.clip_ratio_low.value_or(clip_ratio);
  const double clip_ratio_high = config.clip_ratio_high.value_or(clip_ratio);
  const double clip_ratio_c = config.clip_ratio_c;

  TORCH_CHECK(clip_ratio_c > 1.0,
              "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the "
              "value: ",
              clip_ratio_c, ".");

  torch::Tensor negative_approx_kl = log_prob - old_log_prob;
  negative_approx_kl = torch::clamp(negative_approx_kl, /*min=*/-20.0, /*max=*/20.0);
  torch::Tensor ratio = torch::exp(negative_approx_kl);
  torch::Tensor ppo_kl = masked_mean(-negative_approx_kl, response_mask);

  torch::Tensor pg_losses1 = -advantages * ratio;
  torch::Tensor pg_losses2 = -advantages * torch::clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high);
  torch::Tensor clip_pg_losses1 = torch::maximum(pg_losses1, pg_losses2);
  torch::Tensor pg_clipfrac = masked_mean(torch::gt(pg_losses2, pg_losses1).to(torch::kFloat), response_mask);

  torch::Tensor pg_losses3 = -advantages * clip_ratio_c;
  torch::Tensor clip_pg_losses2 = torch::min(pg_losses3, clip_pg_losses1);
  torch::Tensor pg_clipfrac_lower = masked_mean(
      torch::gt(clip_pg_losses1, pg_losses3) * (advantages < 0).to(torch::kFloat), response_mask);

  torch::Tensor pg_losses = torch::where(advantages < 0, clip_pg_losses2, clip_pg_losses1);

  if (rollout_is_weights.has_value()) {
    pg_losses = pg_losses * (*rollout_is_weights);
  }

  torch::Tensor pg_loss = agg_loss(pg_losses, response_mask, config.loss_agg_mode, config.agg_opts);

  PolicyLossResult result;
  result.pg_loss = pg_loss;
  result.pg_clipfrac = pg_clipfrac.item<double>();
  result.ppo_kl = ppo_kl.item<double>();
  result.pg_clipfrac_lower = pg_clipfrac_lower.item<double>();
  return result;
}

torch::Tensor kl_penalty_forward(const torch::Tensor& logprob, const torch::Tensor& ref_logprob,
                                  const std::string& kl_penalty_kind) {
  if (kl_penalty_kind == "kl" || kl_penalty_kind == "k1") {
    return logprob - ref_logprob;
  }
  if (kl_penalty_kind == "abs") {
    return (logprob - ref_logprob).abs();
  }
  if (kl_penalty_kind == "mse" || kl_penalty_kind == "k2") {
    return 0.5 * (logprob - ref_logprob).pow(2);
  }
  if (kl_penalty_kind == "low_var_kl" || kl_penalty_kind == "k3") {
    torch::Tensor kl = ref_logprob - logprob;
    kl = torch::clamp(kl, /*min=*/-20, /*max=*/20);
    torch::Tensor ratio = torch::exp(kl);
    torch::Tensor kld = (ratio - kl - 1).contiguous();
    return torch::clamp(kld, /*min=*/-10, /*max=*/10);
  }
  if (kl_penalty_kind == "full") {
    throw std::runtime_error("kl_penalty_forward: 'full' is not implemented");
  }
  throw std::runtime_error("kl_penalty_forward: unknown kl_penalty kind '" + kl_penalty_kind + "'");
}

torch::Tensor kl_penalty(const torch::Tensor& logprob, const torch::Tensor& ref_logprob,
                          const std::string& kl_penalty_kind) {
  const bool has_plus = !kl_penalty_kind.empty() && kl_penalty_kind.back() == '+';
  const std::string base_kl_penalty = has_plus ? kl_penalty_kind.substr(0, kl_penalty_kind.size() - 1)
                                                : kl_penalty_kind;
  torch::Tensor forward_score = kl_penalty_forward(logprob, ref_logprob, base_kl_penalty);
  if (!has_plus || base_kl_penalty == "mse" || base_kl_penalty == "k2") {
    return forward_score;
  }
  torch::Tensor backward_score = 0.5 * (logprob - ref_logprob).pow(2);
  return backward_score - backward_score.detach() + forward_score.detach();
}

torch::Tensor compute_rewards(const torch::Tensor& token_level_scores, const torch::Tensor& old_log_prob,
                               const torch::Tensor& ref_log_prob, double kl_ratio) {
  torch::Tensor kl = old_log_prob - ref_log_prob;
  return token_level_scores - kl * kl_ratio;
}

}  // namespace verl_core_algos
