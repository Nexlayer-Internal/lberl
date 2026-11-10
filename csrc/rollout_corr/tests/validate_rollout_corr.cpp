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
// Reads the text fixtures written by gen_fixtures.py (real Python
// rollout_corr_helper.py / score_centering.py inputs + reference outputs)
// and checks the libtorch port in verl_rollout_corr::* reproduces them
// within a tight numerical tolerance. Same style as
// csrc/core_algos/tests/validate_core_algos.cpp.
//
// Usage: validate_rollout_corr <fixtures_dir>
// Exit code 0 if every scenario passes, nonzero otherwise.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "verl_rollout_corr/rollout_corr_helper.h"
#include "verl_rollout_corr/score_centering.h"

namespace fs = std::filesystem;
using verl_rollout_corr::ThresholdSpec;

namespace {

constexpr double kTol = 1e-4;  // float32 reference vs libtorch float32 compute, through a text round-trip

bool g_any_failed = false;

// macOS libc++'s istream >> double does not reliably parse "inf"/"-inf"/"nan"
// tokens (verified: it silently produces garbage rather than throwing) --
// Python's repr() of such floats produces exactly these tokens (score
// centering's "incomplete support" fixtures contain real -inf entries), so
// fixture parsing needs an explicit special-case rather than relying on the
// standard numeric extraction operator.
double parse_fixture_double(const std::string& token) {
  if (token == "inf" || token == "Infinity") return std::numeric_limits<double>::infinity();
  if (token == "-inf" || token == "-Infinity") return -std::numeric_limits<double>::infinity();
  if (token == "nan" || token == "-nan") return std::numeric_limits<double>::quiet_NaN();
  return std::stod(token);
}

struct Fixture {
  std::unordered_map<std::string, torch::Tensor> tensors;
  std::unordered_map<std::string, double> scalars;
  std::unordered_map<std::string, std::string> strs;
  std::unordered_map<std::string, double> metrics;

  const torch::Tensor& t(const std::string& name) const { return tensors.at(name); }
  double s(const std::string& name) const { return scalars.at(name); }
  const std::string& str(const std::string& name) const { return strs.at(name); }
  bool has_tensor(const std::string& name) const { return tensors.count(name) > 0; }
};

Fixture load_fixture(const fs::path& path) {
  std::ifstream in(path);
  TORCH_CHECK(in.is_open(), "failed to open fixture: ", path.string());
  Fixture fx;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream header(line);
    std::string kind;
    header >> kind;
    if (kind == "TENSOR") {
      std::string name;
      int ndim;
      header >> name >> ndim;
      std::vector<int64_t> shape(ndim);
      for (auto& d : shape) header >> d;
      int64_t numel;
      header >> numel;
      std::string data_line;
      std::getline(in, data_line);
      std::istringstream data(data_line);
      std::vector<float> values(numel);
      std::string token;
      for (int64_t k = 0; k < numel; ++k) {
        data >> token;
        values[k] = static_cast<float>(parse_fixture_double(token));
      }
      torch::Tensor t = torch::from_blob(values.data(), shape, torch::kFloat).clone();
      fx.tensors[name] = t;
    } else if (kind == "SCALAR") {
      std::string name, token;
      header >> name >> token;
      fx.scalars[name] = parse_fixture_double(token);
    } else if (kind == "INTS") {
      // Not used by this module's fixtures, but tolerate/skip the line pair
      // for forward-compatibility with the core_algos-style format.
      std::string name;
      int64_t n;
      header >> name >> n;
      std::string data_line;
      std::getline(in, data_line);
    } else if (kind == "STR") {
      std::string name;
      header >> name;
      std::string value;
      std::getline(header, value);
      // Strip a single leading space left by the `>>` extraction above.
      if (!value.empty() && value.front() == ' ') value.erase(0, 1);
      fx.strs[name] = value;
    } else if (kind == "NMETRICS") {
      int64_t n;
      header >> n;
      for (int64_t i = 0; i < n; ++i) {
        std::string metric_line;
        std::getline(in, metric_line);
        std::istringstream ms(metric_line);
        std::string tag, name, token;
        ms >> tag >> name >> token;
        TORCH_CHECK(tag == "METRIC", "expected METRIC line, got: ", metric_line);
        fx.metrics[name] = parse_fixture_double(token);
      }
    } else {
      TORCH_CHECK(false, "unknown fixture record kind: ", kind);
    }
  }
  return fx;
}

double max_abs_diff(const torch::Tensor& a, const torch::Tensor& b) {
  TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch: ", a.sizes(), " vs ", b.sizes());
  if (a.numel() == 0) return 0.0;
  torch::Tensor diff = (a - b).abs();
  // NaN-aware: (-inf) - (-inf) = NaN in plain subtraction, but both being -inf
  // at the same position is itself a match (score centering's "incomplete
  // support" fixtures legitimately produce -inf head log-probs). Treat
  // same-value infinities as zero diff explicitly.
  torch::Tensor both_neg_inf = a.isinf() & b.isinf() & (a.sign() == b.sign());
  diff = torch::where(both_neg_inf, torch::zeros_like(diff), diff);
  return diff.max().item<double>();
}

void report(const std::string& scenario, double diff, double tol = kTol) {
  bool ok = std::isfinite(diff) && diff <= tol;
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << scenario << "  max_abs_diff=" << diff << " (tol=" << tol << ")"
            << std::endl;
  if (!ok) g_any_failed = true;
}

struct MetricsComparison {
  double max_abs_diff = 0.0;
  std::vector<std::string> missing_keys;
  std::vector<std::string> unexpected_keys;
};

MetricsComparison compare_metrics(const std::unordered_map<std::string, double>& computed,
                                   const std::unordered_map<std::string, double>& reference) {
  MetricsComparison cmp;
  for (const auto& kv : reference) {
    auto it = computed.find(kv.first);
    if (it == computed.end()) {
      cmp.missing_keys.push_back(kv.first);
      continue;
    }
    double diff = std::abs(it->second - kv.second);
    if (!std::isfinite(diff)) diff = std::numeric_limits<double>::infinity();
    cmp.max_abs_diff = std::max(cmp.max_abs_diff, diff);
  }
  for (const auto& kv : computed) {
    if (!reference.count(kv.first)) cmp.unexpected_keys.push_back(kv.first);
  }
  return cmp;
}

void report_metrics(const std::string& scenario, const MetricsComparison& cmp, double tol = kTol) {
  bool ok = cmp.missing_keys.empty() && cmp.unexpected_keys.empty() && std::isfinite(cmp.max_abs_diff) &&
            cmp.max_abs_diff <= tol;
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << scenario << " (metrics)  max_abs_diff=" << cmp.max_abs_diff
            << " (tol=" << tol << ")";
  if (!cmp.missing_keys.empty()) {
    std::cout << "  MISSING:";
    for (const auto& k : cmp.missing_keys) std::cout << " " << k;
  }
  if (!cmp.unexpected_keys.empty()) {
    std::cout << "  UNEXPECTED:";
    for (const auto& k : cmp.unexpected_keys) std::cout << " " << k;
  }
  std::cout << std::endl;
  if (!ok) g_any_failed = true;
}

std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> parts;
  std::istringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) parts.push_back(item);
  return parts;
}

// ---------------------------------------------------------------------------
// 1. Threshold-spec parsing
// ---------------------------------------------------------------------------

void check_threshold_parsing(const fs::path& dir) {
  std::ifstream in(dir / "threshold_parsing.txt");
  TORCH_CHECK(in.is_open(), "failed to open threshold_parsing.txt");

  int64_t num_is_cases = 0;
  {
    std::string tag;
    in >> tag >> num_is_cases;
  }
  for (int64_t i = 0; i < num_is_cases; ++i) {
    std::string tag, kind, spec, upper_str, has_lower_str, lower_str;
    in >> tag >> kind >> spec >> upper_str >> has_lower_str >> lower_str;
    TORCH_CHECK(tag == "IS_THRESHOLD", "malformed threshold_parsing.txt (IS_THRESHOLD)");
    double expected_upper = parse_fixture_double(upper_str);
    bool expected_has_lower = has_lower_str == "1";
    double expected_lower = parse_fixture_double(lower_str);

    ThresholdSpec threshold = (kind == "numeric") ? ThresholdSpec(parse_fixture_double(spec)) : ThresholdSpec(spec);
    verl_rollout_corr::IsThreshold parsed = verl_rollout_corr::parse_rollout_is_threshold(threshold);

    double diff = std::abs(parsed.upper - expected_upper);
    bool lower_ok = parsed.lower.has_value() == expected_has_lower;
    if (lower_ok && expected_has_lower) {
      diff = std::max(diff, std::abs(*parsed.lower - expected_lower));
    }
    std::string name = "is_threshold[" + std::to_string(i) + "]:" + kind + ":" + spec;
    if (!lower_ok) {
      std::cout << "[FAIL] " << name << "  has_lower mismatch (expected " << expected_has_lower << ")" << std::endl;
      g_any_failed = true;
    } else {
      report(name, diff);
    }
  }

  int64_t num_rs_cases = 0;
  {
    std::string tag;
    in >> tag >> num_rs_cases;
  }
  for (int64_t i = 0; i < num_rs_cases; ++i) {
    std::string tag, options_csv, spec;
    int64_t n_options;
    in >> tag >> options_csv >> spec >> n_options;
    TORCH_CHECK(tag == "RS_THRESHOLD", "malformed threshold_parsing.txt (RS_THRESHOLD)");
    std::vector<std::string> options = split_csv(options_csv);

    std::unordered_map<std::string, std::pair<bool, double>> expected_lower_map;
    std::unordered_map<std::string, double> expected_upper_map;
    for (int64_t j = 0; j < n_options; ++j) {
      std::string opt, has_lower_str, lower_str, upper_str;
      in >> opt >> has_lower_str >> lower_str >> upper_str;
      expected_lower_map[opt] = {has_lower_str == "1", parse_fixture_double(lower_str)};
      expected_upper_map[opt] = parse_fixture_double(upper_str);
    }

    auto result = verl_rollout_corr::parse_rollout_rs_thresholds(options, ThresholdSpec(spec));
    double max_diff = 0.0;
    bool ok = true;
    for (const auto& opt : options) {
      const auto& r = result.at(opt);
      double expected_upper = expected_upper_map.at(opt);
      max_diff = std::max(max_diff, std::abs(r.upper.value_or(0.0) - expected_upper));
      bool expect_lower = expected_lower_map.at(opt).first;
      if (r.lower.has_value() != expect_lower) {
        ok = false;
      } else if (expect_lower) {
        max_diff = std::max(max_diff, std::abs(*r.lower - expected_lower_map.at(opt).second));
      }
    }
    std::string name = "rs_threshold[" + std::to_string(i) + "]:" + options_csv;
    if (!ok) {
      std::cout << "[FAIL] " << name << "  has_lower mismatch" << std::endl;
      g_any_failed = true;
    } else {
      report(name, max_diff);
    }
  }
}

// ---------------------------------------------------------------------------
// 2. compute_rollout_rejection_mask / compute_rs_metrics
// ---------------------------------------------------------------------------

void check_rejection_mask(const fs::path& dir, const std::string& name) {
  Fixture fx = load_fixture(dir / ("rejection_mask_" + name + ".txt"));
  auto result = verl_rollout_corr::compute_rollout_rejection_mask(
      fx.t("log_ratio"), fx.t("response_mask"), fx.str("rollout_rs"), ThresholdSpec(fx.str("rollout_rs_threshold")));
  report("rejection_mask:" + name + " (mask)", max_abs_diff(result.modified_response_mask, fx.t("ref_modified_response_mask")));
  report_metrics("rejection_mask:" + name, compare_metrics(result.metrics, fx.metrics));
}

void check_rs_metrics(const fs::path& dir, const std::string& name) {
  Fixture fx = load_fixture(dir / ("rs_metrics_" + name + ".txt"));
  std::optional<torch::Tensor> per_sequence_values;
  if (fx.s("has_per_sequence_values") > 0.5) per_sequence_values = fx.t("per_sequence_values");
  auto metrics = verl_rollout_corr::compute_rs_metrics(
      fx.str("option_name"), fx.t("rs_statistic"), fx.t("response_mask"), fx.t("seq_valid_mask").to(torch::kBool),
      fx.str("level"), per_sequence_values, fx.s("rollout_rs_threshold"), fx.s("rollout_rs_threshold_lower"),
      fx.s("apply_lower_threshold") > 0.5);
  report_metrics("rs_metrics:" + name, compare_metrics(metrics, fx.metrics));
}

// ---------------------------------------------------------------------------
// 3. compute_rollout_correction_weights / compute_is_metrics
// ---------------------------------------------------------------------------

void check_is_weights(const fs::path& dir, const std::string& name) {
  Fixture fx = load_fixture(dir / ("is_weights_" + name + ".txt"));
  auto result = verl_rollout_corr::compute_rollout_correction_weights(
      fx.t("log_ratio"), fx.t("response_mask"), fx.str("rollout_is"), ThresholdSpec(fx.str("rollout_is_threshold")),
      fx.s("rollout_is_batch_normalize") > 0.5);
  report("is_weights:" + name + " (weights)", max_abs_diff(result.rollout_is_weights, fx.t("ref_weights")));
  report_metrics("is_weights:" + name, compare_metrics(result.metrics, fx.metrics));
}

void check_is_metrics(const fs::path& dir, const std::string& name) {
  Fixture fx = load_fixture(dir / ("is_metrics_" + name + ".txt"));
  std::optional<double> threshold_lower;
  if (fx.s("has_threshold_lower") > 0.5) threshold_lower = fx.s("rollout_is_threshold_lower");
  auto metrics = verl_rollout_corr::compute_is_metrics(fx.t("rollout_is_weights"), fx.t("raw_rollout_is_weights"),
                                                        fx.t("log_ratio_for_metrics"), fx.t("response_mask"),
                                                        fx.str("rollout_is"), fx.s("rollout_is_threshold"),
                                                        threshold_lower);
  report_metrics("is_metrics:" + name, compare_metrics(metrics, fx.metrics));
}

// ---------------------------------------------------------------------------
// 4. compute_offpolicy_metrics / compute_rollout_corr_metrics_from_logprobs
// ---------------------------------------------------------------------------

void check_offpolicy_metrics(const fs::path& dir, const std::string& name) {
  Fixture fx = load_fixture(dir / ("offpolicy_metrics_" + name + ".txt"));
  std::optional<torch::Tensor> rollout_log_prob;
  if (fx.s("has_rollout_log_prob") > 0.5) rollout_log_prob = fx.t("rollout_log_prob");
  auto metrics = verl_rollout_corr::compute_offpolicy_metrics(fx.t("old_log_prob"), rollout_log_prob, fx.t("response_mask"));
  report_metrics("offpolicy_metrics:" + name, compare_metrics(metrics, fx.metrics));
}

void check_corr_metrics_from_logprobs(const fs::path& dir) {
  Fixture fx = load_fixture(dir / "corr_metrics_from_logprobs.txt");
  auto metrics = verl_rollout_corr::compute_rollout_corr_metrics_from_logprobs(fx.t("log_prob"), fx.t("rollout_log_prob"),
                                                                                fx.t("response_mask"));
  report_metrics("corr_metrics_from_logprobs", compare_metrics(metrics, fx.metrics));
}

// ---------------------------------------------------------------------------
// 5. score_centering: dummy_rollout_topk / pad_rollout_topk
// ---------------------------------------------------------------------------

void check_dummy_topk(const fs::path& dir) {
  Fixture fx = load_fixture(dir / "dummy_topk_basic.txt");
  int64_t num_rows = static_cast<int64_t>(fx.s("num_rows"));
  int64_t k = static_cast<int64_t>(fx.s("k"));
  auto head = verl_rollout_corr::dummy_rollout_topk(num_rows, k);
  report("dummy_topk:basic (ids)", max_abs_diff(head.ids.to(torch::kFloat), fx.t("ref_ids")));
  report("dummy_topk:basic (log_probs)", max_abs_diff(head.log_probs, fx.t("ref_log_probs")));
}

void check_pad_topk(const fs::path& dir, const std::string& name) {
  Fixture fx = load_fixture(dir / ("pad_topk_" + name + ".txt"));
  int64_t prompt_length = static_cast<int64_t>(fx.s("prompt_length"));
  auto head = verl_rollout_corr::pad_rollout_topk(fx.t("response_topk_ids"), fx.t("response_topk_log_probs"), prompt_length);
  report("pad_topk:" + name + " (ids)", max_abs_diff(head.ids.to(torch::kFloat), fx.t("ref_full_ids")));
  report("pad_topk:" + name + " (log_probs)", max_abs_diff(head.log_probs, fx.t("ref_full_log_probs")));
}

// ---------------------------------------------------------------------------
// 6. score_centering: topk_log_probs_from_logits (forward + backward)
// ---------------------------------------------------------------------------

void check_topk_log_probs(const fs::path& dir, const std::string& name) {
  Fixture fx = load_fixture(dir / ("topk_log_probs_" + name + ".txt"));
  torch::Tensor logits = fx.t("logits").clone().set_requires_grad(true);
  torch::Tensor token_ids = fx.t("token_ids").to(torch::kInt64);
  int64_t chunk_size = static_cast<int64_t>(fx.s("chunk_size"));

  torch::Tensor output = verl_rollout_corr::topk_log_probs_from_logits(logits, token_ids, chunk_size);
  report("topk_log_probs:" + name + " (forward)", max_abs_diff(output, fx.t("ref_output")));

  output.backward(fx.t("grad_output"));
  report("topk_log_probs:" + name + " (grad_logits)", max_abs_diff(logits.grad(), fx.t("ref_grad_logits")));
}

// ---------------------------------------------------------------------------
// 7. score_centering: score_centering_weight_fn
// ---------------------------------------------------------------------------

void check_weight_fn(const fs::path& dir, const std::string& name) {
  Fixture fx = load_fixture(dir / ("weight_fn_" + name + ".txt"));
  std::optional<std::string> rollout_is;
  if (fx.s("has_rollout_is") > 0.5) rollout_is = fx.str("rollout_is");
  auto weight_fn = verl_rollout_corr::score_centering_weight_fn(rollout_is, ThresholdSpec(fx.str("rollout_is_threshold")));
  torch::Tensor out = weight_fn(fx.t("ratio"));
  report("weight_fn:" + name, max_abs_diff(out, fx.t("ref_weight")));
}

// ---------------------------------------------------------------------------
// 8. score_centering_correction / *_support_correction / *_auto_correction
// ---------------------------------------------------------------------------

// Reconstructs the WeightFn gen_fixtures.py used for a given scenario name
// (weight_fn isn't tensor data, so it can't be read back from the fixture
// file -- this mirrors gen_score_centering_correction_scenarios()'s choices
// exactly: "icepop" scenarios used token-IcePop(0.3, 2.0), "identity"
// scenarios used no IS weighting, everything else used token-TIS(2.0)).
verl_rollout_corr::WeightFn weight_fn_for_scenario(const std::string& name) {
  if (name.find("icepop") != std::string::npos) {
    return verl_rollout_corr::score_centering_weight_fn("token", ThresholdSpec("0.3_2.0"));
  }
  if (name.find("identity") != std::string::npos) {
    return verl_rollout_corr::score_centering_weight_fn(std::nullopt, ThresholdSpec(2.0));
  }
  return verl_rollout_corr::score_centering_weight_fn("token", ThresholdSpec(2.0));
}

void check_correction(const fs::path& dir, const std::string& name) {
  Fixture fx = load_fixture(dir / ("correction_" + name + ".txt"));
  torch::Tensor train = fx.t("train_head_log_probs").clone().set_requires_grad(true);
  torch::Tensor sampler = fx.t("sampler_head_log_probs");
  auto weight_fn = weight_fn_for_scenario(name);

  auto result = verl_rollout_corr::score_centering_correction(train, sampler, weight_fn, fx.s("eps"));
  report("correction:" + name + " (correction)", max_abs_diff(result.correction, fx.t("ref_correction")));
  report("correction:" + name + " (sampler_head_mass)", max_abs_diff(result.sampler_head_mass, fx.t("ref_sampler_head_mass")));
  report("correction:" + name + " (train_head_mass)", max_abs_diff(result.train_head_mass, fx.t("ref_train_head_mass")));

  result.correction.sum().backward();
  report("correction:" + name + " (grad)", max_abs_diff(train.grad(), fx.t("ref_grad_train_head_log_probs")));
}

void check_support_correction(const fs::path& dir, const std::string& name) {
  Fixture fx = load_fixture(dir / ("support_correction_" + name + ".txt"));
  torch::Tensor train = fx.t("train_head_log_probs").clone().set_requires_grad(true);
  torch::Tensor sampler = fx.t("sampler_head_log_probs");
  auto weight_fn = weight_fn_for_scenario(name);

  auto result = verl_rollout_corr::score_centering_support_correction(train, sampler, weight_fn);
  report("support_correction:" + name + " (correction)", max_abs_diff(result.correction, fx.t("ref_correction")));
  report("support_correction:" + name + " (log_support_mass)",
         max_abs_diff(result.log_support_mass, fx.t("ref_log_support_mass")));
  report("support_correction:" + name + " (sampler_head_mass)",
         max_abs_diff(result.sampler_head_mass, fx.t("ref_sampler_head_mass")));

  (result.correction.sum() + result.log_support_mass.sum()).backward();
  report("support_correction:" + name + " (grad)", max_abs_diff(train.grad(), fx.t("ref_grad_train_head_log_probs")));
}

void check_auto_correction(const fs::path& dir, const std::string& name) {
  Fixture fx = load_fixture(dir / ("auto_correction_" + name + ".txt"));
  torch::Tensor train = fx.t("train_head_log_probs").clone().set_requires_grad(true);
  torch::Tensor sampler = fx.t("sampler_head_log_probs");
  torch::Tensor support_complete = fx.t("support_complete").to(torch::kBool);
  auto weight_fn = weight_fn_for_scenario(name);

  auto result = verl_rollout_corr::score_centering_auto_correction(train, sampler, weight_fn, support_complete);
  report("auto_correction:" + name + " (correction)", max_abs_diff(result.correction, fx.t("ref_correction")));
  report("auto_correction:" + name + " (log_norm_shift)", max_abs_diff(result.log_norm_shift, fx.t("ref_log_norm_shift")));
  report("auto_correction:" + name + " (sampler_head_mass)",
         max_abs_diff(result.sampler_head_mass, fx.t("ref_sampler_head_mass")));
  report("auto_correction:" + name + " (train_head_mass)", max_abs_diff(result.train_head_mass, fx.t("ref_train_head_mass")));

  (result.correction.sum() + result.log_norm_shift.sum()).backward();
  report("auto_correction:" + name + " (grad)", max_abs_diff(train.grad(), fx.t("ref_grad_train_head_log_probs")));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: validate_rollout_corr <fixtures_dir>" << std::endl;
    return 2;
  }
  fs::path dir(argv[1]);

  check_threshold_parsing(dir);

  for (const std::string& name :
       {"token_k1_icepop", "token_k2_token_k3_chained", "seq_sum_k2", "seq_mean_k3", "seq_max_k2",
        "seq_sum_k1_icepop", "extreme_all_rejected", "extreme_all_kept", "singleton_batch", "empty_batch",
        "zero_variance_log_ratio"}) {
    check_rejection_mask(dir, name);
  }

  for (const std::string& name :
       {"token_level_normal", "sequence_level_no_lower", "sequence_level_with_lower", "contrived_empty_seq_valid"}) {
    check_rs_metrics(dir, name);
  }

  for (const std::string& name : {"token_tis", "token_icepop", "sequence_tis", "sequence_icepop",
                                   "token_tis_batch_normalize", "sequence_batch_normalize", "extreme_tight_tis",
                                   "singleton_batch", "zero_variance"}) {
    check_is_weights(dir, name);
  }

  for (const std::string& name : {"token_default_lower", "token_explicit_lower", "sequence_default_lower"}) {
    check_is_metrics(dir, name);
  }

  for (const std::string& name : {"with_rollout", "without_rollout", "identical_policies", "singleton_batch"}) {
    check_offpolicy_metrics(dir, name);
  }
  check_corr_metrics_from_logprobs(dir);

  check_dummy_topk(dir);
  check_pad_topk(dir, "basic");
  check_pad_topk(dir, "prompt_length_one");

  for (const std::string& name : {"single_chunk", "multi_chunk", "large_vocab"}) {
    check_topk_log_probs(dir, name);
  }

  for (const std::string& name : {"none", "token_tis", "token_icepop"}) {
    check_weight_fn(dir, name);
  }

  for (const std::string& name : {"normal_tis", "weight_fn_identity", "weight_fn_icepop", "near_full_head_mass"}) {
    check_correction(dir, name);
  }
  for (const std::string& name : {"full_support", "incomplete_support", "incomplete_support_icepop"}) {
    check_support_correction(dir, name);
  }
  for (const std::string& name : {"mixed_support", "all_complete", "all_incomplete"}) {
    check_auto_correction(dir, name);
  }

  if (g_any_failed) {
    std::cout << "\nSOME SCENARIOS FAILED" << std::endl;
    return 1;
  }
  std::cout << "\nALL SCENARIOS PASSED" << std::endl;
  return 0;
}
