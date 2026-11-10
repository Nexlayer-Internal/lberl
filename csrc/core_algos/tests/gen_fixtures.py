#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate validation fixtures for the C++ port of core_algos.py's GRPO
advantage / PPO-clip policy-loss / KL-penalty math.

Runs the REAL `verl/trainer/ppo/core_algos.py` functions (imported directly
from this checkout, no reimplementation) on several labeled scenarios and
dumps inputs + reference outputs as plain text, which the C++ validation
harness (validate_core_algos.cpp) reads back and compares against the
libtorch port bit-for-bit (within float tolerance).

Shapes/scale note (honesty disclosure, see also csrc/core_algos/README.md):
the original validated GSM8K/Qwen3-4B smoke test's raw logs/launch script
were not recoverable in this environment -- only a summary README survived,
which records `data.train_batch_size=8` and aggregate scalar metrics
(actor/loss, actor/kl_loss, actor/entropy) from that run. Group size and
per-token log-prob/advantage tensors were never preserved at that
granularity (they are intermediate values, not logged), so they cannot be
"recovered" even in principle from the surviving scalar metrics. The
scenarios below therefore use clearly-labeled realistic synthetic values:
batch size 8 (matching the recovered train_batch_size), GRPO group size 4
(a common `rollout.n` default), response length 64 (shorter than real
GSM8K generations to keep fixtures small and diffable; the math has no
dependence on sequence length beyond elementwise/reduction ops, so this
does not reduce coverage of the ported logic).
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import verl.trainer.ppo.core_algos as core_algos  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "fixtures"
OUT_DIR.mkdir(exist_ok=True)


class FakeActorConfig:
    """Minimal duck-typed stand-in for verl.workers.config.ActorConfig.

    compute_policy_loss_vanilla only reads these specific attributes/methods
    off `config` (see core_algos.py L1311-1361); it is not required to be an
    actual ActorConfig instance (only NOT an AlgoConfig instance, which a
    plain object satisfies).
    """

    def __init__(self, clip_ratio, clip_ratio_low=None, clip_ratio_high=None, clip_ratio_c=3.0):
        self.clip_ratio = clip_ratio
        self.clip_ratio_low = clip_ratio_low
        self.clip_ratio_high = clip_ratio_high
        self._clip_ratio_c = clip_ratio_c
        self.global_batch_info = {}

    def get(self, key, default=None):
        if key == "clip_ratio_c":
            return self._clip_ratio_c
        return getattr(self, key, default)


def dump_tensor(f, name, t):
    t = t.detach().to(torch.float64).contiguous()
    flat = t.reshape(-1).tolist()
    shape = list(t.shape)
    f.write(f"TENSOR {name} {len(shape)} {' '.join(str(s) for s in shape)} {len(flat)}\n")
    f.write(" ".join(repr(v) for v in flat) + "\n")


def dump_scalar(f, name, v):
    f.write(f"SCALAR {name} {repr(float(v))}\n")


def dump_ints(f, name, values):
    f.write(f"INTS {name} {len(values)}\n")
    f.write(" ".join(str(int(v)) for v in values) + "\n")


def make_index(bsz, group_size):
    # Groups of `group_size` consecutive rows share a prompt (group id),
    # mirroring how GRPO groups rollout.n samples per prompt. Allows a
    # remainder group smaller than group_size if bsz is not a multiple.
    idx = []
    gid = 0
    i = 0
    while i < bsz:
        n = min(group_size, bsz - i)
        idx += [gid] * n
        i += n
        gid += 1
    return np.array(idx)


def run_grpo_advantage_scenario(name, token_level_rewards, response_mask, index, epsilon=1e-6,
                                 norm_adv_by_std_in_grpo=True):
    advantages, returns = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards.clone(),
        response_mask=response_mask,
        index=index,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
    path = OUT_DIR / f"grpo_adv_{name}.txt"
    with open(path, "w") as f:
        dump_tensor(f, "token_level_rewards", token_level_rewards)
        dump_tensor(f, "response_mask", response_mask)
        dump_ints(f, "index", index.tolist())
        dump_scalar(f, "epsilon", epsilon)
        dump_scalar(f, "norm_adv_by_std_in_grpo", 1.0 if norm_adv_by_std_in_grpo else 0.0)
        dump_tensor(f, "ref_advantages", advantages)
        dump_tensor(f, "ref_returns", returns)
    has_nan = torch.isnan(advantages).any().item()
    print(f"[grpo_adv:{name}] wrote {path.name}  has_nan={has_nan}")
    assert not has_nan, f"scenario {name} produced NaN advantages"


def run_policy_loss_scenario(name, old_log_prob, log_prob, advantages, response_mask, config,
                              loss_agg_mode="token-mean", rollout_is_weights=None):
    pg_loss, pg_metrics = core_algos.compute_policy_loss_vanilla(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )
    path = OUT_DIR / f"policy_loss_{name}.txt"
    with open(path, "w") as f:
        dump_tensor(f, "old_log_prob", old_log_prob)
        dump_tensor(f, "log_prob", log_prob)
        dump_tensor(f, "advantages", advantages)
        dump_tensor(f, "response_mask", response_mask)
        f.write(f"STR loss_agg_mode {loss_agg_mode}\n")
        dump_scalar(f, "clip_ratio", config.clip_ratio)
        dump_scalar(f, "clip_ratio_low", config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio)
        dump_scalar(f, "clip_ratio_high", config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio)
        dump_scalar(f, "clip_ratio_c", config.get("clip_ratio_c", 3.0))
        if rollout_is_weights is not None:
            dump_tensor(f, "rollout_is_weights", rollout_is_weights)
        dump_tensor(f, "ref_pg_loss", pg_loss.reshape(1))
        dump_scalar(f, "ref_pg_clipfrac", pg_metrics["actor/pg_clipfrac"])
        dump_scalar(f, "ref_ppo_kl", pg_metrics["actor/ppo_kl"])
        dump_scalar(f, "ref_pg_clipfrac_lower", pg_metrics["actor/pg_clipfrac_lower"])
    print(f"[policy_loss:{name}] wrote {path.name}  pg_loss={pg_loss.item():.8f} "
          f"clipfrac={pg_metrics['actor/pg_clipfrac']:.6f} ppo_kl={pg_metrics['actor/ppo_kl']:.6f} "
          f"clipfrac_lower={pg_metrics['actor/pg_clipfrac_lower']:.6f}")


def run_kl_scenario(name, logprob, ref_logprob, kind):
    # kl_penalty_forward (unlike kl_penalty) does not strip a trailing "+"
    # itself -- mirror kl_penalty's own stripping so this scenario still
    # exercises kl_penalty_forward standalone for "+" variants.
    base_kind = kind[:-1] if kind.endswith("+") else kind
    fwd = core_algos.kl_penalty_forward(logprob, ref_logprob, base_kind)
    full = core_algos.kl_penalty(logprob, ref_logprob, kind)
    path = OUT_DIR / f"kl_{name}.txt"
    with open(path, "w") as f:
        dump_tensor(f, "logprob", logprob)
        dump_tensor(f, "ref_logprob", ref_logprob)
        f.write(f"STR kind {kind}\n")
        dump_tensor(f, "ref_forward", fwd)
        dump_tensor(f, "ref_full", full)
    print(f"[kl:{name}] wrote {path.name}  kind={kind}")


def run_compute_rewards_scenario(name, token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    rewards = core_algos.compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio)
    path = OUT_DIR / f"rewards_{name}.txt"
    with open(path, "w") as f:
        dump_tensor(f, "token_level_scores", token_level_scores)
        dump_tensor(f, "old_log_prob", old_log_prob)
        dump_tensor(f, "ref_log_prob", ref_log_prob)
        dump_scalar(f, "kl_ratio", kl_ratio)
        dump_tensor(f, "ref_rewards", rewards)
    print(f"[rewards:{name}] wrote {path.name}")


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    bsz, group_size, resp_len = 8, 4, 64
    index = make_index(bsz, group_size)

    # --- Scenario 1: mixed/realistic rewards, response_mask with real padding
    # (variable-length completions -- later tokens masked out, as in real
    # generation where responses end with EOS before max_response_length).
    token_level_rewards = torch.zeros(bsz, resp_len)
    lengths = [64, 50, 40, 64, 30, 64, 20, 55]
    response_mask = torch.zeros(bsz, resp_len)
    for i, length in enumerate(lengths):
        response_mask[i, :length] = 1.0
        # Outcome reward: GRPO's reward is scalar-per-sequence, placed at the
        # last valid (unmasked) token, as a real reward manager would do.
        token_level_rewards[i, length - 1] = float(np.random.uniform(-1.0, 1.0))
    run_grpo_advantage_scenario("mixed", token_level_rewards, response_mask, index)
    run_grpo_advantage_scenario("mixed_no_std_norm", token_level_rewards, response_mask, index,
                                 norm_adv_by_std_in_grpo=False)

    # --- Scenario 2: all-identical-reward group (zero variance) --
    # exercises the epsilon-guarded std path; must not produce NaN.
    token_level_rewards_const = torch.zeros(bsz, resp_len)
    response_mask_const = torch.ones(bsz, resp_len)
    const_rewards = [1.0, 1.0, 1.0, 1.0, 0.3, 0.7, -0.5, 0.2]  # first group of 4 identical
    for i, r in enumerate(const_rewards):
        token_level_rewards_const[i, -1] = r
    run_grpo_advantage_scenario("zero_variance_group", token_level_rewards_const, response_mask_const, index)

    # --- Scenario 3: singleton groups (group size 1) -- exercises the
    # explicit mean=0/std=1 special case.
    index_singleton = np.arange(bsz)
    run_grpo_advantage_scenario("singleton_groups", token_level_rewards, response_mask, index_singleton)

    # --- Policy loss scenarios ---
    def make_logprobs(resp_len_, ratio_kind):
        old_lp = -torch.rand(bsz, resp_len_) * 2.0 - 0.1  # realistic negative log-probs
        if ratio_kind == "near_one":
            lp = old_lp + (torch.rand(bsz, resp_len_) - 0.5) * 0.05  # small drift -> ratio near 1
        elif ratio_kind == "mixed_clip":
            lp = old_lp + (torch.rand(bsz, resp_len_) - 0.5) * 0.6  # bigger drift -> exercises clipping
        elif ratio_kind == "extreme":
            lp = old_lp + (torch.rand(bsz, resp_len_) - 0.2) * 4.0  # large drift -> exercises dual-clip
        return old_lp, lp

    old_lp1, lp1 = make_logprobs(resp_len, "mixed_clip")
    advantages1 = (torch.rand(bsz, resp_len) - 0.5) * 4.0 * response_mask
    cfg1 = FakeActorConfig(clip_ratio=0.2)
    run_policy_loss_scenario("mixed_clip_token_mean", old_lp1, lp1, advantages1, response_mask, cfg1,
                              loss_agg_mode="token-mean")

    old_lp2, lp2 = make_logprobs(resp_len, "near_one")
    advantages2 = (torch.rand(bsz, resp_len) - 0.5) * 0.5 * response_mask
    cfg2 = FakeActorConfig(clip_ratio=0.2)
    run_policy_loss_scenario("near_one_seq_mean_token_mean", old_lp2, lp2, advantages2, response_mask, cfg2,
                              loss_agg_mode="seq-mean-token-mean")

    old_lp3, lp3 = make_logprobs(resp_len, "extreme")
    advantages3 = (torch.rand(bsz, resp_len) - 0.5) * 6.0 * response_mask
    # asymmetric clip range + explicit dual-clip lower bound, to exercise
    # both the clip_ratio_low/high branch and the clip_ratio_c dual-clip branch.
    cfg3 = FakeActorConfig(clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.28, clip_ratio_c=2.0)
    run_policy_loss_scenario("extreme_dual_clip", old_lp3, lp3, advantages3, response_mask, cfg3,
                              loss_agg_mode="token-mean")

    old_lp4, lp4 = make_logprobs(resp_len, "mixed_clip")
    advantages4 = (torch.rand(bsz, resp_len) - 0.5) * 4.0 * response_mask
    cfg4 = FakeActorConfig(clip_ratio=0.2)
    run_policy_loss_scenario("seq_mean_token_sum_norm", old_lp4, lp4, advantages4, response_mask, cfg4,
                              loss_agg_mode="seq-mean-token-sum-norm")

    # rollout correction weights (off-policy IS weight), simple multiplicative case
    old_lp5, lp5 = make_logprobs(resp_len, "mixed_clip")
    advantages5 = (torch.rand(bsz, resp_len) - 0.5) * 4.0 * response_mask
    cfg5 = FakeActorConfig(clip_ratio=0.2)
    rollout_is_weights5 = torch.rand(bsz, resp_len).clamp(0.5, 1.5) * response_mask
    run_policy_loss_scenario("with_rollout_is_weights", old_lp5, lp5, advantages5, response_mask, cfg5,
                              loss_agg_mode="token-mean", rollout_is_weights=rollout_is_weights5)

    # --- KL penalty scenarios ---
    logprob = -torch.rand(bsz, resp_len) * 2.0 - 0.1
    ref_logprob = logprob + (torch.rand(bsz, resp_len) - 0.5) * 0.8
    for kind in ["k1", "abs", "k2", "k3", "k3+", "mse"]:
        run_kl_scenario(kind.replace("+", "plus"), logprob, ref_logprob, kind)

    # KL penalty "off" sanity scenario: identical logprob/ref_logprob -> kl == 0 everywhere.
    run_kl_scenario("zero_kl", logprob, logprob.clone(), "k3")

    # --- compute_rewards (applies KL penalty term to token-level reward) ---
    token_level_scores = torch.rand(bsz, resp_len) * response_mask
    run_compute_rewards_scenario("kl_on", token_level_scores, logprob, ref_logprob, kl_ratio=0.1)
    run_compute_rewards_scenario("kl_off", token_level_scores, logprob, ref_logprob, kl_ratio=0.0)

    print(f"\nAll fixtures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
