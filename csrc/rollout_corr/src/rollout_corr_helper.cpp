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

#include "verl_rollout_corr/rollout_corr_helper.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "verl_core_algos/core_algos.h"

namespace verl_rollout_corr {

namespace {

// -- small string helpers for the threshold-spec parsers (no tensors here) --

std::string trim(const std::string& s) {
  size_t start = s.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) return "";
  size_t end = s.find_last_not_of(" \t\n\r");
  return s.substr(start, end - start + 1);
}

std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> parts;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    parts.push_back(item);
  }
  return parts;
}

double parse_double(const std::string& spec) {
  try {
    size_t pos = 0;
    double v = std::stod(spec, &pos);
    if (pos != spec.size()) throw std::invalid_argument("trailing characters");
    return v;
  } catch (const std::exception&) {
    throw std::invalid_argument("Invalid numeric threshold value '" + spec + "'.");
  }
}

// Round-trips a double through text exactly (used only when a caller passes
// a numeric ThresholdSpec into the multi-option RS parser, which otherwise
// operates on strings -- mirrors Python's `str(threshold_spec)` on a float,
// but with full round-trip precision rather than Python's repr rounding).
std::string format_double(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return std::string(buf);
}

bool ends_with(const std::string& s, const std::string& suffix) {
  if (suffix.size() > s.size()) return false;
  return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

const std::unordered_set<std::string>& supported_rollout_rs_options() {
  static const std::unordered_set<std::string> kOptions = {
      "token_k1",    "token_k2",    "token_k3",    "seq_sum_k1",  "seq_sum_k2",
      "seq_sum_k3",  "seq_mean_k1", "seq_mean_k2", "seq_mean_k3", "seq_max_k2",
      "seq_max_k3",
  };
  return kOptions;
}

const std::unordered_set<std::string>& token_level_rollout_rs_options() {
  static const std::unordered_set<std::string> kOptions = {"token_k1", "token_k2", "token_k3"};
  return kOptions;
}

torch::Tensor masked_sum_dim(const torch::Tensor& values, const torch::Tensor& mask) {
  return verl_core_algos::masked_sum(values, mask, std::vector<int64_t>{-1});
}

torch::Tensor masked_mean_dim(const torch::Tensor& values, const torch::Tensor& mask) {
  return verl_core_algos::masked_mean(values, mask, std::vector<int64_t>{-1});
}

torch::Tensor neg_inf_like(const torch::Tensor& t) {
  return torch::full({}, -std::numeric_limits<double>::infinity(), t.options());
}

torch::Tensor sequence_max(const torch::Tensor& values, const torch::Tensor& response_mask) {
  torch::Tensor mask_bool = response_mask.to(torch::kBool);
  torch::Tensor neg_inf = neg_inf_like(values);
  torch::Tensor masked_values = values.masked_fill(~mask_bool, neg_inf);
  torch::Tensor max_values = std::get<0>(masked_values.max(/*dim=*/-1));
  return torch::where(max_values == neg_inf, torch::zeros_like(max_values), max_values);
}

}  // namespace

IsThreshold parse_rollout_is_threshold(const ThresholdSpec& threshold_spec) {
  double upper;
  std::optional<double> lower;

  if (threshold_spec.is_numeric()) {
    upper = threshold_spec.numeric_value();
    lower = std::nullopt;
  } else {
    std::string spec = trim(threshold_spec.text_value());
    if (spec.empty()) {
      throw std::invalid_argument("rollout_is_threshold must not be an empty string.");
    }
    size_t underscore = spec.find('_');
    if (underscore != std::string::npos) {
      std::string lower_str = spec.substr(0, underscore);
      std::string upper_str = spec.substr(underscore + 1);
      lower = parse_double(lower_str);
      upper = parse_double(upper_str);
    } else {
      upper = parse_double(spec);
      lower = std::nullopt;
    }
  }

  if (upper <= 0) {
    throw std::invalid_argument("rollout_is_threshold upper bound must be positive, got " +
                                 format_double(upper) + ".");
  }
  if (lower.has_value()) {
    if (*lower <= 0) {
      throw std::invalid_argument("rollout_is_threshold lower bound must be positive, got " +
                                   format_double(*lower) + ".");
    }
    if (*lower > upper) {
      throw std::invalid_argument("rollout_is_threshold lower bound must be <= upper bound.");
    }
  }
  return {upper, lower};
}

std::unordered_map<std::string, RsThreshold> parse_rollout_rs_thresholds(
    const std::vector<std::string>& options, const std::optional<ThresholdSpec>& threshold_spec) {
  if (!threshold_spec.has_value()) {
    throw std::invalid_argument("rollout_rs_threshold must be provided for rejection sampling.");
  }

  std::vector<std::string> raw_specs;
  if (threshold_spec->is_numeric()) {
    raw_specs = {format_double(threshold_spec->numeric_value())};
  } else {
    for (const auto& part : split(threshold_spec->text_value(), ',')) {
      std::string trimmed = trim(part);
      if (!trimmed.empty()) raw_specs.push_back(trimmed);
    }
  }

  if (raw_specs.empty()) {
    throw std::invalid_argument("rollout_rs_threshold must contain at least one threshold value.");
  }
  if (raw_specs.size() != 1 && raw_specs.size() != options.size()) {
    throw std::invalid_argument(
        "rollout_rs_threshold expects either one threshold shared by all options or exactly "
        "options.size() thresholds to match the provided rollout_rs options.");
  }
  if (raw_specs.size() == 1 && options.size() > 1) {
    std::string shared = raw_specs[0];
    raw_specs.assign(options.size(), shared);
  }

  std::unordered_map<std::string, RsThreshold> thresholds;
  for (size_t i = 0; i < options.size(); ++i) {
    const std::string& option = options[i];
    const std::string& spec = raw_specs[i];
    if (ends_with(option, "k1")) {
      std::string lower_str, upper_str;
      size_t underscore = spec.find('_');
      if (underscore != std::string::npos) {
        lower_str = spec.substr(0, underscore);
        upper_str = spec.substr(underscore + 1);
      } else {
        upper_str = spec;
        lower_str = format_double(1.0 / parse_double(upper_str));
      }
      double lower = parse_double(lower_str);
      double upper = parse_double(upper_str);
      if (lower <= 0 || upper <= 0) {
        throw std::invalid_argument("Thresholds for option '" + option + "' must be positive, got " + spec + ".");
      }
      if (lower > upper) {
        throw std::invalid_argument("Thresholds for option '" + option + "' must be lower <= upper, got " + spec +
                                     ".");
      }
      thresholds[option] = {lower, upper};
    } else {
      if (spec.find('_') != std::string::npos) {
        throw std::invalid_argument("rollout_rs_threshold for option '" + option +
                                     "' must provide a single upper bound without '_'. Received '" + spec + "'.");
      }
      double upper = parse_double(spec);
      if (upper <= 0) {
        throw std::invalid_argument("Threshold for option '" + option + "' must be positive, got " + spec + ".");
      }
      thresholds[option] = {std::nullopt, upper};
    }
  }
  return thresholds;
}

std::unordered_map<std::string, double> compute_rs_metrics(
    const std::string& option_name, const torch::Tensor& rs_statistic, const torch::Tensor& response_mask,
    const torch::Tensor& seq_valid_mask, const std::string& level,
    const std::optional<torch::Tensor>& per_sequence_values, double rollout_rs_threshold,
    double rollout_rs_threshold_lower, bool apply_lower_threshold) {
  TORCH_CHECK(response_mask.any().item<bool>(), "response_mask must contain at least one valid token (1).");

  std::unordered_map<std::string, double> metrics;
  const std::string prefix = "rollout_rs_" + option_name;
  torch::Tensor mask_bool = response_mask.to(torch::kBool);

  torch::Tensor seq_values =
      per_sequence_values.has_value() ? *per_sequence_values : masked_mean_dim(rs_statistic, response_mask);
  if (seq_values.dim() > 1) seq_values = seq_values.squeeze(-1);
  torch::Tensor seq_values_valid = seq_values.masked_select(seq_valid_mask.to(torch::kBool));

  metrics[prefix + "_mean"] = verl_core_algos::masked_mean(rs_statistic, response_mask).item<double>();

  const bool seq_level_with_values = (level == "sequence") && seq_values_valid.numel() > 0;
  if (seq_level_with_values) {
    metrics[prefix + "_max"] = seq_values_valid.max().item<double>();
    metrics[prefix + "_min"] = seq_values_valid.min().item<double>();
  } else {
    torch::Tensor neg_inf = neg_inf_like(rs_statistic);
    torch::Tensor pos_inf = torch::full({}, std::numeric_limits<double>::infinity(), rs_statistic.options());
    metrics[prefix + "_max"] = rs_statistic.masked_fill(~mask_bool, neg_inf).max().item<double>();
    metrics[prefix + "_min"] = rs_statistic.masked_fill(~mask_bool, pos_inf).min().item<double>();
  }

  double fraction_high, fraction_low;
  if (seq_level_with_values) {
    fraction_high = (seq_values_valid > rollout_rs_threshold).to(torch::kFloat).mean().item<double>();
    fraction_low = apply_lower_threshold
                       ? (seq_values_valid < rollout_rs_threshold_lower).to(torch::kFloat).mean().item<double>()
                       : 0.0;
  } else {
    fraction_high =
        verl_core_algos::masked_mean((rs_statistic > rollout_rs_threshold).to(torch::kFloat), response_mask)
            .item<double>();
    fraction_low = apply_lower_threshold
                       ? verl_core_algos::masked_mean((rs_statistic < rollout_rs_threshold_lower).to(torch::kFloat),
                                                       response_mask)
                             .item<double>()
                       : 0.0;
  }
  metrics[prefix + "_fraction_high"] = fraction_high;
  metrics[prefix + "_fraction_low"] = fraction_low;

  double mask_count = response_mask.sum().item<double>();
  if (mask_count > 1) {
    double clamp_min = apply_lower_threshold ? rollout_rs_threshold_lower : 0.0;
    torch::Tensor stat_for_std = rs_statistic.clamp(clamp_min, rollout_rs_threshold);
    torch::Tensor mean_clamped = verl_core_algos::masked_mean(stat_for_std, response_mask);
    torch::Tensor stat_var = verl_core_algos::masked_mean(stat_for_std.square(), response_mask) - mean_clamped.square();
    metrics[prefix + "_std"] = torch::sqrt(torch::clamp(stat_var, 0.0)).item<double>();
  } else {
    metrics[prefix + "_std"] = 0.0;
  }

  if (seq_values_valid.numel() > 0) {
    metrics[prefix + "_seq_mean"] = seq_values_valid.mean().item<double>();
    metrics[prefix + "_seq_std"] = seq_values_valid.numel() > 1 ? seq_values_valid.std().item<double>() : 0.0;
    metrics[prefix + "_seq_max"] = seq_values_valid.max().item<double>();
    metrics[prefix + "_seq_min"] = seq_values_valid.min().item<double>();
    metrics[prefix + "_seq_max_deviation"] = seq_values_valid.abs().max().item<double>();
    metrics[prefix + "_seq_fraction_high"] =
        (seq_values_valid > rollout_rs_threshold).to(torch::kFloat).mean().item<double>();
    if (apply_lower_threshold) {
      // NOTE: matches Python exactly -- this key is only present when
      // apply_lower_threshold is true in this branch; see the header's
      // "Honest note on dict-key presence".
      metrics[prefix + "_seq_fraction_low"] =
          (seq_values_valid < rollout_rs_threshold_lower).to(torch::kFloat).mean().item<double>();
    }
  } else {
    metrics[prefix + "_seq_mean"] = 0.0;
    metrics[prefix + "_seq_std"] = 0.0;
    metrics[prefix + "_seq_max"] = 0.0;
    metrics[prefix + "_seq_min"] = 0.0;
    metrics[prefix + "_seq_max_deviation"] = 0.0;
    metrics[prefix + "_seq_fraction_high"] = 0.0;
    metrics[prefix + "_seq_fraction_low"] = 0.0;
  }

  return metrics;
}

RejectionMaskResult compute_rollout_rejection_mask(const torch::Tensor& log_ratio, const torch::Tensor& response_mask,
                                                    const std::string& rollout_rs,
                                                    const ThresholdSpec& rollout_rs_threshold) {
  if (log_ratio.size(0) == 0) {
    return {response_mask, {}};
  }

  std::vector<std::string> option_modes;
  for (const auto& part : split(rollout_rs, ',')) {
    std::string trimmed = trim(part);
    if (!trimmed.empty()) option_modes.push_back(trimmed);
  }
  if (option_modes.empty()) {
    throw std::invalid_argument("rollout_rs must contain at least one valid option.");
  }

  std::vector<std::string> normalized_options;
  std::unordered_set<std::string> seen;
  for (const auto& opt : option_modes) {
    if (supported_rollout_rs_options().find(opt) == supported_rollout_rs_options().end()) {
      throw std::invalid_argument("Invalid rollout_rs option: " + opt + ".");
    }
    if (seen.insert(opt).second) normalized_options.push_back(opt);
  }

  std::unordered_map<std::string, RsThreshold> threshold_specs =
      parse_rollout_rs_thresholds(normalized_options, rollout_rs_threshold);

  torch::Tensor log_ratio_safe = torch::clamp(log_ratio, -kSafetyBound, kSafetyBound);
  torch::Tensor token_k1 = -log_ratio_safe;
  torch::Tensor token_k2 = 0.5 * log_ratio_safe.pow(2);
  torch::Tensor token_k3 = torch::exp(log_ratio_safe) - 1.0 - log_ratio_safe;

  torch::Tensor response_mask_bool = response_mask.to(torch::kBool);
  torch::Tensor seq_valid_mask = response_mask.sum(-1) > 0;
  // dtype=log_ratio.dtype, device kept at response_mask's device -- matches
  // Python's torch.ones_like(response_mask, dtype=log_ratio.dtype) exactly
  // (the *_like factory functions default device to the input tensor's).
  torch::Tensor combined_mask = torch::ones_like(response_mask, response_mask.options().dtype(log_ratio.scalar_type()));
  std::unordered_map<std::string, double> metrics;

  for (const std::string& option_name : normalized_options) {
    const RsThreshold& thresholds_info = threshold_specs.at(option_name);
    const bool is_k1_option = ends_with(option_name, "k1");
    const std::optional<double> upper_value = thresholds_info.upper;
    const std::optional<double> lower_value = thresholds_info.lower;
    const bool apply_lower_threshold = is_k1_option;
    std::optional<double> lower_log, upper_log;

    if (is_k1_option) {
      TORCH_CHECK(lower_value.has_value() && upper_value.has_value(),
                  "rollout_rs_threshold for option '", option_name, "' must specify both lower and upper bounds.");
      lower_log = std::log(*lower_value);
      upper_log = std::log(*upper_value);
    } else {
      TORCH_CHECK(upper_value.has_value(), "rollout_rs_threshold for option '", option_name,
                  "' must specify an upper bound.");
    }

    const std::string level = token_level_rollout_rs_options().count(option_name) ? "token" : "sequence";

    torch::Tensor per_token_stat;
    std::optional<torch::Tensor> per_sequence_stat;
    torch::Tensor token_keep_bool;

    auto seq_reduce_sum = [&](const torch::Tensor& values) { return masked_sum_dim(values, response_mask); };
    auto seq_reduce_mean = [&](const torch::Tensor& values) { return masked_mean_dim(values, response_mask); };
    auto seq_reduce_max = [&](const torch::Tensor& values) { return sequence_max(values, response_mask); };

    if (option_name == "token_k1") {
      TORCH_CHECK(lower_log.has_value(), "Threshold specification for token_k1 must include lower and upper bounds.");
      per_token_stat = token_k1;
      token_keep_bool = (per_token_stat >= *lower_log) & (per_token_stat <= *upper_log);
    } else if (option_name == "token_k2") {
      per_token_stat = token_k2;
      token_keep_bool = per_token_stat <= *upper_value;
    } else if (option_name == "token_k3") {
      per_token_stat = token_k3;
      token_keep_bool = per_token_stat <= *upper_value;
    } else if (option_name.rfind("seq_sum", 0) == 0 || option_name.rfind("seq_mean", 0) == 0) {
      const bool is_sum = option_name.rfind("seq_sum", 0) == 0;
      torch::Tensor seq_stat;
      torch::Tensor seq_keep_bool_direct;
      if (ends_with(option_name, "k1")) {
        TORCH_CHECK(lower_log.has_value(), "Threshold specification for option '", option_name,
                    "' must include lower and upper bounds.");
        seq_stat = is_sum ? seq_reduce_sum(token_k1) : seq_reduce_mean(token_k1);
        seq_keep_bool_direct = (seq_stat >= *lower_log) & (seq_stat <= *upper_log);
      } else if (ends_with(option_name, "k2")) {
        seq_stat = is_sum ? seq_reduce_sum(token_k2) : seq_reduce_mean(token_k2);
        seq_keep_bool_direct = seq_stat <= *upper_value;
      } else if (ends_with(option_name, "k3")) {
        seq_stat = is_sum ? seq_reduce_sum(token_k3) : seq_reduce_mean(token_k3);
        seq_keep_bool_direct = seq_stat <= *upper_value;
      } else {
        throw std::invalid_argument("Unsupported rollout_rs option: " + option_name + ".");
      }
      per_sequence_stat = seq_stat;
      token_keep_bool = seq_keep_bool_direct.unsqueeze(-1).expand_as(response_mask_bool);
      per_token_stat = seq_stat.unsqueeze(-1).expand_as(response_mask);
    } else if (option_name.rfind("seq_max", 0) == 0) {
      torch::Tensor seq_stat;
      torch::Tensor seq_keep_bool_direct;
      if (ends_with(option_name, "k2")) {
        seq_stat = seq_reduce_max(token_k2);
        seq_keep_bool_direct = seq_stat <= *upper_value;
      } else if (ends_with(option_name, "k3")) {
        seq_stat = seq_reduce_max(token_k3);
        seq_keep_bool_direct = seq_stat <= *upper_value;
      } else {
        throw std::invalid_argument("Unsupported rollout_rs option: " + option_name + ".");
      }
      per_sequence_stat = seq_stat;
      token_keep_bool = seq_keep_bool_direct.unsqueeze(-1).expand_as(response_mask_bool);
      per_token_stat = seq_stat.unsqueeze(-1).expand_as(response_mask);
    } else {
      throw std::invalid_argument("Unsupported rollout_rs option: " + option_name + ".");
    }

    const double metrics_upper_threshold = is_k1_option ? *upper_log : *upper_value;
    const double metrics_lower_threshold = (is_k1_option && lower_log.has_value()) ? *lower_log : 0.0;

    torch::Tensor token_keep_mask = token_keep_bool.to(log_ratio.scalar_type());
    combined_mask = combined_mask * token_keep_mask;
    torch::Tensor seq_keep_bool_tensor = (~((~token_keep_bool) & response_mask_bool)).all(-1);

    std::unordered_map<std::string, double> option_metrics =
        compute_rs_metrics(option_name, per_token_stat, response_mask, seq_valid_mask, level, per_sequence_stat,
                            metrics_upper_threshold, metrics_lower_threshold, apply_lower_threshold);
    metrics.insert(option_metrics.begin(), option_metrics.end());

    double token_masked_fraction =
        verl_core_algos::masked_mean(1 - token_keep_mask, response_mask).item<double>();
    torch::Tensor seq_valid_float = seq_valid_mask.to(torch::kFloat);
    double seq_masked_fraction;
    if (seq_valid_float.sum().item<double>() > 0) {
      torch::Tensor seq_keep_float = seq_keep_bool_tensor.to(log_ratio.scalar_type());
      seq_masked_fraction =
          (((1.0 - seq_keep_float) * seq_valid_float).sum() / seq_valid_float.sum()).item<double>();
    } else {
      seq_masked_fraction = 0.0;
    }
    metrics["rollout_rs_" + option_name + "_masked_fraction"] = token_masked_fraction;
    metrics["rollout_rs_" + option_name + "_seq_masked_fraction"] = seq_masked_fraction;
  }

  torch::Tensor final_mask = combined_mask;
  metrics["rollout_rs_masked_fraction"] =
      verl_core_algos::masked_mean(1 - final_mask, response_mask).item<double>();
  torch::Tensor final_keep_bool = (final_mask > 0.5) & response_mask_bool;
  torch::Tensor seq_has_masked = (~final_keep_bool & response_mask_bool).any(-1);
  metrics["rollout_rs_seq_masked_fraction"] = seq_has_masked.to(torch::kFloat).mean().item<double>();

  torch::Tensor modified_response_mask = (response_mask * final_mask).to(response_mask.scalar_type());
  return {modified_response_mask, metrics};
}

std::unordered_map<std::string, double> compute_is_metrics(const torch::Tensor& rollout_is_weights,
                                                             const torch::Tensor& raw_rollout_is_weights,
                                                             const torch::Tensor& log_ratio_for_metrics,
                                                             const torch::Tensor& response_mask,
                                                             const std::string& rollout_is,
                                                             double rollout_is_threshold,
                                                             std::optional<double> rollout_is_threshold_lower) {
  TORCH_CHECK(response_mask.any().item<bool>(), "response_mask must contain at least one valid token (1).");

  std::unordered_map<std::string, double> metrics;
  const double threshold_lower =
      rollout_is_threshold_lower.has_value() ? *rollout_is_threshold_lower : 1.0 / rollout_is_threshold;
  const double log_threshold_upper = std::log(rollout_is_threshold);
  const double log_threshold_lower = std::log(threshold_lower);

  if (rollout_is == "sequence") {
    torch::Tensor log_max = log_ratio_for_metrics.max();
    torch::Tensor log_min = log_ratio_for_metrics.min();
    metrics["rollout_is_max"] = torch::exp(torch::clamp(log_max, c10::nullopt, kSafetyBound)).item<double>();
    metrics["rollout_is_min"] = torch::exp(log_min).item<double>();

    metrics["rollout_is_mean"] = verl_core_algos::masked_mean(rollout_is_weights, response_mask).item<double>();

    torch::Tensor exceeds_upper = log_ratio_for_metrics > log_threshold_upper;
    torch::Tensor below_lower = log_ratio_for_metrics < log_threshold_lower;
    metrics["rollout_is_ratio_fraction_high"] = exceeds_upper.to(torch::kFloat).mean().item<double>();
    metrics["rollout_is_ratio_fraction_low"] = below_lower.to(torch::kFloat).mean().item<double>();
  } else {
    metrics["rollout_is_mean"] = verl_core_algos::masked_mean(rollout_is_weights, response_mask).item<double>();

    torch::Tensor above = raw_rollout_is_weights > rollout_is_threshold;
    torch::Tensor below = raw_rollout_is_weights < threshold_lower;
    metrics["rollout_is_ratio_fraction_high"] =
        verl_core_algos::masked_mean(above.to(torch::kFloat), response_mask).item<double>();
    metrics["rollout_is_ratio_fraction_low"] =
        verl_core_algos::masked_mean(below.to(torch::kFloat), response_mask).item<double>();

    torch::Tensor mask_bool = response_mask.to(torch::kBool);
    torch::Tensor neg_inf = neg_inf_like(rollout_is_weights);
    torch::Tensor pos_inf = torch::full({}, std::numeric_limits<double>::infinity(), rollout_is_weights.options());
    metrics["rollout_is_max"] = rollout_is_weights.masked_fill(~mask_bool, neg_inf).max().item<double>();
    metrics["rollout_is_min"] = rollout_is_weights.masked_fill(~mask_bool, pos_inf).min().item<double>();
  }

  double mask_count = response_mask.sum().item<double>();
  if (mask_count > 1) {
    torch::Tensor weights_for_std = rollout_is_weights.clamp(0.0, rollout_is_threshold);
    torch::Tensor mean_clamped = verl_core_algos::masked_mean(weights_for_std, response_mask);
    torch::Tensor rollout_is_var =
        verl_core_algos::masked_mean(weights_for_std.square(), response_mask) - mean_clamped.square();
    metrics["rollout_is_std"] = torch::sqrt(torch::clamp(rollout_is_var, 0.0)).item<double>();
  } else {
    metrics["rollout_is_std"] = 0.0;
  }

  torch::Tensor weights_for_ess = rollout_is_weights.clamp(0.0, rollout_is_threshold);
  torch::Tensor mean_for_ess = verl_core_algos::masked_mean(weights_for_ess, response_mask);
  torch::Tensor is_weights_normalized = weights_for_ess / (mean_for_ess + 1e-8);
  metrics["rollout_is_eff_sample_size"] =
      1.0 / verl_core_algos::masked_mean(is_weights_normalized.square(), response_mask).item<double>();

  if (rollout_is_weights.dim() > 1) {
    torch::Tensor seq_mean_weights = masked_mean_dim(rollout_is_weights, response_mask);

    metrics["rollout_is_seq_mean"] = seq_mean_weights.mean().item<double>();
    metrics["rollout_is_seq_std"] = seq_mean_weights.numel() > 1 ? seq_mean_weights.std().item<double>() : 0.0;
    metrics["rollout_is_seq_max"] = seq_mean_weights.max().item<double>();
    metrics["rollout_is_seq_min"] = seq_mean_weights.min().item<double>();

    torch::Tensor seq_deviation = (seq_mean_weights - 1.0).abs();
    metrics["rollout_is_seq_max_deviation"] = seq_deviation.max().item<double>();

    torch::Tensor raw_seq_mean_weights = masked_mean_dim(raw_rollout_is_weights, response_mask);
    metrics["rollout_is_seq_fraction_high"] =
        (raw_seq_mean_weights > rollout_is_threshold).to(torch::kFloat).mean().item<double>();
    metrics["rollout_is_seq_fraction_low"] =
        (raw_seq_mean_weights < threshold_lower).to(torch::kFloat).mean().item<double>();
  }

  return metrics;
}

IsWeightsResult compute_rollout_correction_weights(const torch::Tensor& log_ratio, const torch::Tensor& response_mask,
                                                    const std::string& rollout_is,
                                                    const ThresholdSpec& rollout_is_threshold,
                                                    bool rollout_is_batch_normalize,
                                                    std::optional<double> distributed_weights_mean) {
  TORCH_CHECK(rollout_is == "token" || rollout_is == "sequence", "Invalid rollout_is: ", rollout_is,
              ". Must be 'token' or 'sequence'.");

  IsThreshold parsed = parse_rollout_is_threshold(rollout_is_threshold);
  const double upper = parsed.upper;
  const std::optional<double> lower = parsed.lower;
  const bool use_icepop = lower.has_value();

  torch::Tensor log_ratio_for_metrics;
  torch::Tensor raw_rollout_is_weights;

  if (rollout_is == "token") {
    log_ratio_for_metrics = log_ratio;
    torch::Tensor log_ratio_safe = torch::clamp(log_ratio, -kSafetyBound, kSafetyBound);
    raw_rollout_is_weights = torch::exp(log_ratio_safe);
  } else {
    torch::Tensor log_ratio_sum = masked_sum_dim(log_ratio, response_mask).unsqueeze(-1);
    log_ratio_for_metrics = log_ratio_sum;
    torch::Tensor log_ratio_sum_safe = torch::clamp(log_ratio_sum, -kSafetyBound, kSafetyBound);
    raw_rollout_is_weights = torch::exp(log_ratio_sum_safe).expand_as(log_ratio);
  }

  raw_rollout_is_weights = raw_rollout_is_weights * response_mask;

  torch::Tensor rollout_is_weights;
  if (!use_icepop) {
    rollout_is_weights = raw_rollout_is_weights.clamp(c10::nullopt, upper);
  } else {
    torch::Tensor token_kept_mask = (raw_rollout_is_weights >= *lower) & (raw_rollout_is_weights <= upper);
    rollout_is_weights = torch::where(token_kept_mask, raw_rollout_is_weights, torch::zeros_like(raw_rollout_is_weights));
  }

  std::unordered_map<std::string, double> metrics = compute_is_metrics(
      rollout_is_weights, raw_rollout_is_weights, log_ratio_for_metrics, response_mask, rollout_is, upper, lower);

  if (use_icepop) {
    torch::Tensor oob_mask = (raw_rollout_is_weights < *lower) | (raw_rollout_is_weights > upper);
    metrics["rollout_is_oob_ratio"] =
        verl_core_algos::masked_mean(oob_mask.to(torch::kFloat), response_mask).item<double>();
  }

  rollout_is_weights = rollout_is_weights.detach();

  if (rollout_is_batch_normalize) {
    double weights_mean;
    if (rollout_is == "token") {
      weights_mean = distributed_weights_mean.has_value()
                          ? *distributed_weights_mean
                          : verl_core_algos::masked_mean(rollout_is_weights, response_mask).item<double>();
    } else {
      if (distributed_weights_mean.has_value()) {
        weights_mean = *distributed_weights_mean;
      } else {
        torch::Tensor seq_weights = masked_mean_dim(rollout_is_weights, response_mask);
        torch::Tensor seq_mask = (response_mask.sum(-1) > 0).to(rollout_is_weights.scalar_type());
        weights_mean = ((seq_weights * seq_mask).sum() / seq_mask.sum().clamp_min(1e-8)).item<double>();
      }
    }

    if (weights_mean > 1e-8) {
      rollout_is_weights = rollout_is_weights / weights_mean;
      metrics["rollout_is_batch_norm_factor"] = weights_mean;
    } else {
      metrics["rollout_is_batch_norm_factor"] = 1.0;
    }
  }

  return {rollout_is_weights, metrics};
}

std::unordered_map<std::string, double> compute_offpolicy_metrics(const torch::Tensor& old_log_prob,
                                                                    const std::optional<torch::Tensor>& rollout_log_prob,
                                                                    const torch::Tensor& response_mask) {
  TORCH_CHECK(response_mask.any().item<bool>(), "Expected at least one valid token in response_mask");

  std::unordered_map<std::string, double> metrics;

  torch::Tensor mean_log_prob_training = masked_mean_dim(old_log_prob, response_mask);
  torch::Tensor training_ppl = torch::exp(-mean_log_prob_training).mean();
  metrics["training_ppl"] = training_ppl.detach().item<double>();
  metrics["training_log_ppl"] = (-mean_log_prob_training).mean().detach().item<double>();

  if (rollout_log_prob.has_value()) {
    const torch::Tensor& rlp = *rollout_log_prob;

    metrics["kl"] = verl_core_algos::masked_mean(rlp - old_log_prob, response_mask).detach().item<double>();

    torch::Tensor log_ratio = old_log_prob - rlp;
    torch::Tensor k3_kl_matrix = torch::exp(log_ratio) - log_ratio - 1;
    metrics["k3_kl"] = verl_core_algos::masked_mean(k3_kl_matrix, response_mask).detach().item<double>();

    torch::Tensor mean_log_prob_rollout = masked_mean_dim(rlp, response_mask);
    torch::Tensor rollout_ppl = torch::exp(-mean_log_prob_rollout).mean();
    metrics["rollout_ppl"] = rollout_ppl.detach().item<double>();
    metrics["rollout_log_ppl"] = (-mean_log_prob_rollout).mean().detach().item<double>();

    torch::Tensor log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training;
    metrics["log_ppl_diff"] = log_ppl_diff.mean().detach().item<double>();
    metrics["log_ppl_abs_diff"] = log_ppl_diff.abs().mean().detach().item<double>();
    metrics["log_ppl_diff_max"] = log_ppl_diff.max().detach().item<double>();
    metrics["log_ppl_diff_min"] = log_ppl_diff.min().detach().item<double>();

    torch::Tensor ppl_ratio = torch::exp(log_ppl_diff).mean();
    metrics["ppl_ratio"] = ppl_ratio.detach().item<double>();

    torch::Tensor log_ratio_safe = torch::clamp(log_ratio, -kSafetyBound, kSafetyBound);
    torch::Tensor rho_token = torch::exp(log_ratio_safe);
    torch::Tensor rho_squared_token = rho_token.square();
    torch::Tensor chi2_token = verl_core_algos::masked_mean(rho_squared_token, response_mask) - 1.0;
    metrics["chi2_token"] = chi2_token.detach().item<double>();

    torch::Tensor log_ratio_sum = masked_sum_dim(log_ratio, response_mask);
    torch::Tensor log_ratio_sum_safe = torch::clamp(log_ratio_sum, -kSafetyBound, kSafetyBound);
    torch::Tensor rho_squared_seq = torch::exp(2.0 * log_ratio_sum_safe);
    torch::Tensor chi2_seq = rho_squared_seq.mean() - 1.0;
    metrics["chi2_seq"] = chi2_seq.detach().item<double>();
  }

  return metrics;
}

std::unordered_map<std::string, double> compute_rollout_corr_metrics_from_logprobs(const torch::Tensor& log_prob,
                                                                                    const torch::Tensor& rollout_log_prob,
                                                                                    const torch::Tensor& response_mask) {
  std::unordered_map<std::string, double> offpolicy_metrics =
      compute_offpolicy_metrics(log_prob, rollout_log_prob, response_mask);
  std::unordered_map<std::string, double> metrics_with_prefix;
  metrics_with_prefix.reserve(offpolicy_metrics.size());
  for (const auto& kv : offpolicy_metrics) {
    metrics_with_prefix["rollout_corr/" + kv.first] = kv.second;
  }
  return metrics_with_prefix;
}

}  // namespace verl_rollout_corr
