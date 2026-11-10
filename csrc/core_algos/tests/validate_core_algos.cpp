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
// Reads the text fixtures written by gen_fixtures.py (real Python
// `core_algos.py` inputs + reference outputs) and checks the libtorch port
// in verl_core_algos::* reproduces them within a tight numerical tolerance.
//
// Usage: validate_core_algos <fixtures_dir>
// Exit code 0 if every scenario passes, nonzero otherwise. Prints
// max-abs-diff per scenario either way.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "verl_core_algos/core_algos.h"

namespace fs = std::filesystem;

namespace {

constexpr double kTol = 1e-5;  // float32 reference vs libtorch float32 compute

bool g_any_failed = false;

struct Fixture {
  std::unordered_map<std::string, torch::Tensor> tensors;
  std::unordered_map<std::string, double> scalars;
  std::unordered_map<std::string, std::vector<int64_t>> ints;
  std::unordered_map<std::string, std::string> strs;

  const torch::Tensor& t(const std::string& name) const { return tensors.at(name); }
  double s(const std::string& name) const { return scalars.at(name); }
  const std::vector<int64_t>& i(const std::string& name) const { return ints.at(name); }
  const std::string& str(const std::string& name) const { return strs.at(name); }
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
      for (int64_t k = 0; k < numel; ++k) {
        double v;
        data >> v;
        values[k] = static_cast<float>(v);
      }
      torch::Tensor t = torch::from_blob(values.data(), shape, torch::kFloat).clone();
      fx.tensors[name] = t;
    } else if (kind == "SCALAR") {
      std::string name;
      double v;
      header >> name >> v;
      fx.scalars[name] = v;
    } else if (kind == "INTS") {
      std::string name;
      int64_t n;
      header >> name >> n;
      std::string data_line;
      std::getline(in, data_line);
      std::istringstream data(data_line);
      std::vector<int64_t> values(n);
      for (auto& v : values) data >> v;
      fx.ints[name] = values;
    } else if (kind == "STR") {
      std::string name, value;
      header >> name >> value;
      fx.strs[name] = value;
    } else {
      TORCH_CHECK(false, "unknown fixture record kind: ", kind);
    }
  }
  return fx;
}

double max_abs_diff(const torch::Tensor& a, const torch::Tensor& b) {
  return (a - b).abs().max().item<double>();
}

void report(const std::string& scenario, double diff, double tol = kTol) {
  bool ok = std::isfinite(diff) && diff <= tol;
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << scenario << "  max_abs_diff=" << diff << " (tol=" << tol << ")"
            << std::endl;
  if (!ok) g_any_failed = true;
}

void check_grpo_advantage(const fs::path& dir, const std::string& name) {
  Fixture fx = load_fixture(dir / ("grpo_adv_" + name + ".txt"));
  bool norm = fx.s("norm_adv_by_std_in_grpo") > 0.5;
  auto result = verl_core_algos::compute_grpo_outcome_advantage(fx.t("token_level_rewards"), fx.t("response_mask"),
                                                                 fx.i("index"), fx.s("epsilon"), norm);
  bool has_nan = torch::isnan(result.advantages).any().item<bool>();
  double diff = max_abs_diff(result.advantages, fx.t("ref_advantages"));
  double diff_ret = max_abs_diff(result.returns, fx.t("ref_returns"));
  report("grpo_adv:" + name + " (advantages)", diff);
  report("grpo_adv:" + name + " (returns)", diff_ret);
  if (has_nan) {
    std::cout << "[FAIL] grpo_adv:" << name << " produced NaN advantages" << std::endl;
    g_any_failed = true;
  }
}

void check_policy_loss(const fs::path& dir, const std::string& name) {
  Fixture fx = load_fixture(dir / ("policy_loss_" + name + ".txt"));

  verl_core_algos::PolicyLossConfig cfg;
  cfg.clip_ratio = fx.s("clip_ratio");
  cfg.clip_ratio_low = fx.s("clip_ratio_low");
  cfg.clip_ratio_high = fx.s("clip_ratio_high");
  cfg.clip_ratio_c = fx.s("clip_ratio_c");
  cfg.loss_agg_mode = verl_core_algos::parse_loss_agg_mode(fx.str("loss_agg_mode"));

  std::optional<torch::Tensor> rollout_is_weights = std::nullopt;
  if (fx.tensors.count("rollout_is_weights")) {
    rollout_is_weights = fx.t("rollout_is_weights");
  }

  auto result = verl_core_algos::compute_policy_loss_vanilla(fx.t("old_log_prob"), fx.t("log_prob"),
                                                               fx.t("advantages"), fx.t("response_mask"), cfg,
                                                               rollout_is_weights);

  double diff_loss = std::abs(result.pg_loss.item<double>() - fx.t("ref_pg_loss").item<double>());
  double diff_clipfrac = std::abs(result.pg_clipfrac - fx.s("ref_pg_clipfrac"));
  double diff_kl = std::abs(result.ppo_kl - fx.s("ref_ppo_kl"));
  double diff_clipfrac_lower = std::abs(result.pg_clipfrac_lower - fx.s("ref_pg_clipfrac_lower"));

  report("policy_loss:" + name + " (pg_loss)", diff_loss);
  report("policy_loss:" + name + " (pg_clipfrac)", diff_clipfrac);
  report("policy_loss:" + name + " (ppo_kl)", diff_kl);
  report("policy_loss:" + name + " (pg_clipfrac_lower)", diff_clipfrac_lower);
}

void check_kl(const fs::path& dir, const std::string& name) {
  Fixture fx = load_fixture(dir / ("kl_" + name + ".txt"));
  std::string kind = fx.str("kind");
  std::string base_kind = (!kind.empty() && kind.back() == '+') ? kind.substr(0, kind.size() - 1) : kind;

  torch::Tensor fwd = verl_core_algos::kl_penalty_forward(fx.t("logprob"), fx.t("ref_logprob"), base_kind);
  torch::Tensor full = verl_core_algos::kl_penalty(fx.t("logprob"), fx.t("ref_logprob"), kind);

  report("kl:" + name + " (forward)", max_abs_diff(fwd, fx.t("ref_forward")));
  report("kl:" + name + " (full)", max_abs_diff(full, fx.t("ref_full")));
}

void check_rewards(const fs::path& dir, const std::string& name) {
  Fixture fx = load_fixture(dir / ("rewards_" + name + ".txt"));
  torch::Tensor rewards = verl_core_algos::compute_rewards(fx.t("token_level_scores"), fx.t("old_log_prob"),
                                                            fx.t("ref_log_prob"), fx.s("kl_ratio"));
  report("rewards:" + name, max_abs_diff(rewards, fx.t("ref_rewards")));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: validate_core_algos <fixtures_dir>" << std::endl;
    return 2;
  }
  fs::path dir(argv[1]);

  check_grpo_advantage(dir, "mixed");
  check_grpo_advantage(dir, "mixed_no_std_norm");
  check_grpo_advantage(dir, "zero_variance_group");
  check_grpo_advantage(dir, "singleton_groups");

  check_policy_loss(dir, "mixed_clip_token_mean");
  check_policy_loss(dir, "near_one_seq_mean_token_mean");
  check_policy_loss(dir, "extreme_dual_clip");
  check_policy_loss(dir, "seq_mean_token_sum_norm");
  check_policy_loss(dir, "with_rollout_is_weights");

  check_kl(dir, "k1");
  check_kl(dir, "abs");
  check_kl(dir, "k2");
  check_kl(dir, "k3");
  check_kl(dir, "k3plus");
  check_kl(dir, "mse");
  check_kl(dir, "zero_kl");

  check_rewards(dir, "kl_on");
  check_rewards(dir, "kl_off");

  if (g_any_failed) {
    std::cout << "\nSOME SCENARIOS FAILED" << std::endl;
    return 1;
  }
  std::cout << "\nALL SCENARIOS PASSED" << std::endl;
  return 0;
}
