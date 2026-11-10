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

#include "verl_rollout_corr/score_centering.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <torch/csrc/autograd/custom_function.h>

namespace verl_rollout_corr {

TopkHead dummy_rollout_topk(int64_t num_rows, int64_t k, torch::Device device) {
  torch::Tensor ids = torch::arange(k, torch::TensorOptions().dtype(torch::kInt32).device(device))
                           .expand({num_rows, k})
                           .clone();
  torch::Tensor log_probs = torch::full({num_rows, k}, -std::log(static_cast<double>(k)),
                                         torch::TensorOptions().dtype(torch::kFloat32).device(device));
  return {ids, log_probs};
}

TopkHead pad_rollout_topk(const torch::Tensor& response_topk_ids, const torch::Tensor& response_topk_log_probs,
                           int64_t prompt_length) {
  torch::Tensor ids = response_topk_ids.to(torch::kInt32);
  torch::Tensor log_probs = response_topk_log_probs.to(torch::kFloat32);
  const int64_t response_length = ids.size(0);
  const int64_t k = ids.size(1);
  const int64_t start = prompt_length - 1;
  // Defensive addition relative to the Python source: Python's
  // full_ids[start:start+response_length] with a negative `start` would
  // silently slice from the end instead of raising -- this port fails fast
  // instead (only ever triggered by prompt_length < 1, which is not a
  // physically meaningful input: every real prompt has at least one token).
  TORCH_CHECK(start >= 0, "pad_rollout_topk: prompt_length must be >= 1, got ", prompt_length);

  TopkHead full = dummy_rollout_topk(prompt_length + response_length, k, ids.device());
  full.ids.narrow(0, start, response_length).copy_(ids);
  full.log_probs.narrow(0, start, response_length).copy_(log_probs);
  return full;
}

namespace {

// score_centering.py: _TopKLogProbsFromLogits(torch.autograd.Function)
//
// Ported using torch::autograd::Function<T>, the templated CRTP pattern for
// custom C++ autograd ops (same approach already validated in the sibling
// lbtorch project's torch::utils::checkpoint()). Saved state: the original
// logits, the gathered token ids, and two (N,1) scalars-per-row (the
// row-wise max and the softmax denominator) -- the softmax numerator is
// recomputed per chunk in backward rather than cached, matching the
// Python version's "bounded fp32 workspace" design.
class TopKLogProbsFromLogits : public torch::autograd::Function<TopKLogProbsFromLogits> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor logits, torch::Tensor token_ids,
                                int64_t chunk_size) {
    const int64_t n = logits.size(0);
    chunk_size = chunk_size > 0 ? chunk_size : std::max<int64_t>(n, 1);

    torch::Tensor output = torch::empty(token_ids.sizes(), logits.options().dtype(torch::kFloat32));
    torch::Tensor maxima = torch::empty({n, 1}, logits.options().dtype(torch::kFloat32));
    torch::Tensor denominators = torch::empty_like(maxima);

    for (int64_t start = 0; start < n; start += chunk_size) {
      const int64_t end = std::min(start + chunk_size, n);
      // `.to(..., /*copy=*/true)` mirrors Python's
      // `logits[start:end].to(dtype=torch.float32, copy=True)`: an
      // explicit owned fp32 workspace tensor, never a view into `logits`,
      // since the in-place ops below (sub_/exp_) would otherwise corrupt
      // the saved-for-backward `logits` tensor.
      torch::Tensor work = logits.slice(0, start, end).to(torch::kFloat32, /*non_blocking=*/false, /*copy=*/true);
      torch::Tensor maximum = std::get<0>(work.max(/*dim=*/-1, /*keepdim=*/true));
      torch::Tensor targets = work.gather(-1, token_ids.slice(0, start, end));
      work.sub_(maximum).exp_();
      torch::Tensor denominator = work.sum(-1, /*keepdim=*/true);
      output.slice(0, start, end).copy_((targets - maximum) - denominator.log());
      maxima.slice(0, start, end).copy_(maximum);
      denominators.slice(0, start, end).copy_(denominator);
    }

    ctx->save_for_backward({logits, token_ids, maxima, denominators});
    ctx->saved_data["chunk_size"] = chunk_size;
    return output;
  }

  // Mirrors Python's @torch.autograd.function.once_differentiable: nothing
  // in this backward builds a new autograd graph (it operates on the saved,
  // non-grad-tracked tensors with plain in-place ops), so higher-order
  // gradients through it are simply not supported, same as the Python
  // version.
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                   torch::autograd::variable_list grad_outputs) {
    torch::autograd::variable_list saved = ctx->get_saved_variables();
    torch::Tensor logits = saved[0];
    torch::Tensor token_ids = saved[1];
    torch::Tensor maxima = saved[2];
    torch::Tensor denominators = saved[3];
    const int64_t chunk_size = ctx->saved_data["chunk_size"].toInt();
    const int64_t n = logits.size(0);
    const torch::Tensor& grad_output = grad_outputs[0];

    torch::Tensor grad_input = torch::empty_like(logits);

    for (int64_t start = 0; start < n; start += chunk_size) {
      const int64_t end = std::min(start + chunk_size, n);
      torch::Tensor work = logits.slice(0, start, end).to(torch::kFloat32, /*non_blocking=*/false, /*copy=*/true);
      work.sub_(maxima.slice(0, start, end)).exp_().div_(denominators.slice(0, start, end));
      torch::Tensor grad = grad_output.slice(0, start, end).to(torch::kFloat32);
      work.mul_(-grad.sum(-1, /*keepdim=*/true));
      work.scatter_add_(-1, token_ids.slice(0, start, end), grad);
      grad_input.slice(0, start, end).copy_(work);
    }

    // One Variable per forward() argument: logits, token_ids, chunk_size.
    return {grad_input, torch::Tensor(), torch::Tensor()};
  }
};

}  // namespace

torch::Tensor topk_log_probs_from_logits(const torch::Tensor& logits, const torch::Tensor& token_ids,
                                          int64_t chunk_size) {
  return TopKLogProbsFromLogits::apply(logits, token_ids.to(torch::kInt64), chunk_size);
}

WeightFn score_centering_weight_fn(const std::optional<std::string>& rollout_is,
                                    const ThresholdSpec& rollout_is_threshold) {
  if (!rollout_is.has_value()) {
    return [](const torch::Tensor& ratio) { return torch::ones_like(ratio); };
  }
  IsThreshold parsed = parse_rollout_is_threshold(rollout_is_threshold);
  if (!parsed.lower.has_value()) {
    const double upper = parsed.upper;
    return [upper](const torch::Tensor& ratio) { return ratio.clamp(c10::nullopt, upper); };
  }
  const double upper = parsed.upper;
  const double lower = *parsed.lower;
  return [lower, upper](const torch::Tensor& ratio) {
    torch::Tensor mask = (ratio >= lower) & (ratio <= upper);
    return torch::where(mask, ratio, torch::zeros_like(ratio));
  };
}

CorrectionResult score_centering_correction(const torch::Tensor& train_head_log_probs_in,
                                             const torch::Tensor& sampler_head_log_probs, const WeightFn& weight_fn,
                                             double eps) {
  torch::Tensor train_head_log_probs = train_head_log_probs_in.to(torch::kFloat32);
  torch::Tensor residual, p_mass, q_mass;
  {
    torch::NoGradGuard no_grad;
    torch::Tensor p = train_head_log_probs.exp();
    torch::Tensor q = sampler_head_log_probs.to(torch::kFloat32).exp();
    p_mass = p.sum(-1);
    q_mass = q.sum(-1);
    torch::Tensor rho = (1 - q_mass).clamp_min(eps) / (1 - p_mass).clamp_min(eps);
    torch::Tensor alpha = rho * weight_fn(1.0 / rho);
    torch::Tensor head_weights = weight_fn((train_head_log_probs - sampler_head_log_probs.to(torch::kFloat32)).exp());
    residual = q * head_weights - alpha.unsqueeze(-1) * p;
  }
  // `residual` was built entirely under NoGradGuard (matching Python's
  // `with torch.no_grad(): ...` block) -- i.e. it is the stop-gradient
  // coefficient from the docstring's `sg[...]` notation. Multiplying it by
  // `train_head_log_probs` here, outside the no-grad scope, is what lets
  // gradient flow to the trainer logits through `train_head_log_probs`
  // while `residual` itself contributes no gradient.
  torch::Tensor correction = (residual * train_head_log_probs).sum(-1);
  return {correction, q_mass, p_mass};
}

SupportCorrectionResult score_centering_support_correction(const torch::Tensor& train_head_log_probs_in,
                                                            const torch::Tensor& sampler_head_log_probs_in,
                                                            const WeightFn& weight_fn) {
  torch::Tensor sampler_head_log_probs = sampler_head_log_probs_in.to(torch::kFloat32);
  torch::Tensor in_support = sampler_head_log_probs.isfinite();
  torch::Tensor neg_inf = torch::full({}, -std::numeric_limits<double>::infinity(),
                                       train_head_log_probs_in.options().dtype(torch::kFloat32));
  torch::Tensor train_head_log_probs = train_head_log_probs_in.to(torch::kFloat32).masked_fill(~in_support, neg_inf);
  torch::Tensor log_support_mass = train_head_log_probs.logsumexp(-1);
  torch::Tensor train_support_log_probs =
      (train_head_log_probs - log_support_mass.unsqueeze(-1)).masked_fill(~in_support, 0.0);

  torch::Tensor residual, q_sum;
  {
    torch::NoGradGuard no_grad;
    torch::Tensor q = sampler_head_log_probs.exp();
    torch::Tensor ratio = torch::where(in_support, (train_support_log_probs - sampler_head_log_probs).exp(),
                                        torch::ones_like(q));
    torch::Tensor weighted = q * weight_fn(ratio);
    // `in_support` (bool) multiplies the float tensor via ATen's standard
    // bool->float type promotion, same as the implicit promotion Python's
    // `* in_support` relies on.
    residual = weighted - weighted.sum(-1, /*keepdim=*/true) * train_support_log_probs.exp() * in_support;
    q_sum = q.sum(-1);
  }
  torch::Tensor correction = (residual * train_support_log_probs).sum(-1);
  return {correction, log_support_mass, q_sum};
}

AutoCorrectionResult score_centering_auto_correction(const torch::Tensor& train_head_log_probs,
                                                      const torch::Tensor& sampler_head_log_probs,
                                                      const WeightFn& weight_fn,
                                                      const torch::Tensor& support_complete) {
  CorrectionResult base = score_centering_correction(train_head_log_probs, sampler_head_log_probs, weight_fn);

  // Unselected-branch values must also be finite -- a malformed/empty
  // support may otherwise propagate NaN gradients through torch::where's
  // unselected branch, matching the Python docstring's explicit caution.
  torch::Tensor uniform_fill = torch::full_like(
      sampler_head_log_probs, -std::log(static_cast<double>(sampler_head_log_probs.size(-1))));
  torch::Tensor safe_sampler = torch::where(support_complete.unsqueeze(-1), sampler_head_log_probs, uniform_fill);

  SupportCorrectionResult support =
      score_centering_support_correction(train_head_log_probs, safe_sampler, weight_fn);

  torch::Tensor correction = torch::where(support_complete, support.correction, base.correction);
  torch::Tensor log_norm_shift = torch::where(support_complete, support.log_support_mass,
                                               torch::zeros_like(support.log_support_mass));
  torch::Tensor train_head_mass =
      torch::where(support_complete, support.log_support_mass.detach().exp(), base.train_head_mass);

  // `base.sampler_head_mass` (from the unconditional score_centering_correction
  // call), not `support.sampler_head_mass` -- the Python source discards the
  // support-branch sampler mass with `_` and keeps only the unconditional one.
  return {correction, log_norm_shift, base.sampler_head_mass, train_head_mass};
}

}  // namespace verl_rollout_corr
