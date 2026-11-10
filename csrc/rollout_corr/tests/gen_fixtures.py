#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Generate validation fixtures for the C++ port of `rollout_corr_helper.py` +
`score_centering.py` (PLAN.md table item #2).

Runs the REAL, unmodified functions from this checkout (no reimplementation)
across labeled scenarios and dumps inputs + reference outputs as plain text,
mirroring `csrc/core_algos/tests/gen_fixtures.py`'s approach exactly. The C++
validation harness (`validate_rollout_corr.cpp`) reads these back and
compares against the libtorch port.

Environment note (honesty disclosure): this checkout's system Python is 3.9,
but the repo requires >=3.10 (`dict | type` union syntax in
`verl/utils/config.py` is a hard syntax/runtime error on 3.9). Fixtures here
were generated with a fresh Python 3.11 venv
(`torch==2.15.0` CPU wheel, `numpy==2.4.6`, `omegaconf==2.4.0`,
`tensordict==0.15.5`, `transformers==5.22.2`, `ray==2.60.0`, `pillow==13.0.0`
-- installed only to make `import verl` succeed; none of their pinned
versions in `requirements.txt` matter for this script, which only calls pure
tensor-math functions that don't touch any of those packages' actual APIs).
"""

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import verl.trainer.ppo.rollout_corr_helper as rc  # noqa: E402
import verl.trainer.ppo.score_centering as sc  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "fixtures"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Generic dump helpers (same text format as csrc/core_algos/tests/gen_fixtures.py)
# ---------------------------------------------------------------------------


def dump_tensor(f, name, t):
    t = t.detach().to(torch.float64).contiguous()
    flat = t.reshape(-1).tolist()
    shape = list(t.shape)
    f.write(f"TENSOR {name} {len(shape)} {' '.join(str(s) for s in shape)} {len(flat)}\n")
    f.write(" ".join(repr(v) for v in flat) + "\n")


def dump_bool_tensor(f, name, t):
    dump_tensor(f, name, t.to(torch.float64))


def dump_scalar(f, name, v):
    f.write(f"SCALAR {name} {repr(float(v))}\n")


def dump_ints(f, name, values):
    f.write(f"INTS {name} {len(values)}\n")
    f.write(" ".join(str(int(v)) for v in values) + "\n")


def dump_str(f, name, value):
    f.write(f"STR {name} {value}\n")


def dump_metrics(f, metrics):
    f.write(f"NMETRICS {len(metrics)}\n")
    for k, v in metrics.items():
        f.write(f"METRIC {k} {repr(float(v))}\n")


def make_response_mask(bsz, resp_len, lengths):
    mask = torch.zeros(bsz, resp_len)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1.0
    return mask


# ---------------------------------------------------------------------------
# 1. Threshold-spec parsing (_parse_rollout_is_threshold, _parse_rollout_rs_thresholds)
#    No tensors involved -- custom compact text format (see README for why).
# ---------------------------------------------------------------------------


def gen_threshold_parsing():
    path = OUT_DIR / "threshold_parsing.txt"
    with open(path, "w") as f:
        # IS_THRESHOLD <kind> <spec> <expected_upper> <has_lower> <expected_lower_or_0>
        is_cases = [
            (2.0, None),  # bare numeric -> upper only
            ("2.5", None),  # numeric string, no underscore -> upper only
            ("0.1_1.2", None),  # lower_upper string -> IcePop band
            ("1.0_1.0", None),  # degenerate equal bounds (lower == upper, allowed)
        ]
        f.write(f"NUM_IS_CASES {len(is_cases)}\n")
        for spec, _ in is_cases:
            upper, lower = rc._parse_rollout_is_threshold(spec)
            kind = "numeric" if isinstance(spec, (int, float)) and not isinstance(spec, bool) else "string"
            spec_repr = str(spec)
            f.write(
                f"IS_THRESHOLD {kind} {spec_repr} {repr(upper)} {1 if lower is not None else 0} "
                f"{repr(lower) if lower is not None else 0.0}\n"
            )

        # RS_THRESHOLD <options_csv> <spec> <n_options> then per option:
        #   <name> <has_lower> <lower_or_0> <upper>
        rs_cases = [
            (["token_k1"], "0.5_2.0"),
            (["token_k2"], "3.0"),
            (["token_k1", "seq_sum_k2"], "0.5_2.0,3.0"),
            (["seq_mean_k3", "seq_max_k2"], "1.5"),  # single shared threshold broadcast to both options
            (["token_k1"], 2.0),  # bare numeric threshold for a k1 option (reciprocal-lower default)
        ]
        f.write(f"NUM_RS_CASES {len(rs_cases)}\n")
        for options, spec in rs_cases:
            thresholds = rc._parse_rollout_rs_thresholds(options, spec)
            spec_repr = str(spec)
            options_csv = ",".join(options)
            f.write(f"RS_THRESHOLD {options_csv} {spec_repr} {len(options)}\n")
            for opt in options:
                info = thresholds[opt]
                has_lower = 1 if info["lower"] is not None else 0
                lower_val = info["lower"] if info["lower"] is not None else 0.0
                f.write(f"  {opt} {has_lower} {repr(lower_val)} {repr(info['upper'])}\n")
    print(f"[threshold_parsing] wrote {path.name}")


# ---------------------------------------------------------------------------
# 2. compute_rollout_rejection_mask (+ compute_rs_metrics via its internals)
# ---------------------------------------------------------------------------


def run_rejection_mask_scenario(name, log_ratio, response_mask, rollout_rs, rollout_rs_threshold):
    modified_mask, metrics = rc.compute_rollout_rejection_mask(
        log_ratio=log_ratio,
        response_mask=response_mask,
        rollout_rs=rollout_rs,
        rollout_rs_threshold=rollout_rs_threshold,
    )
    path = OUT_DIR / f"rejection_mask_{name}.txt"
    with open(path, "w") as f:
        dump_tensor(f, "log_ratio", log_ratio)
        dump_tensor(f, "response_mask", response_mask)
        dump_str(f, "rollout_rs", rollout_rs)
        dump_str(f, "rollout_rs_threshold", str(rollout_rs_threshold))
        dump_tensor(f, "ref_modified_response_mask", modified_mask)
        dump_metrics(f, metrics)
    print(f"[rejection_mask:{name}] wrote {path.name}  n_metrics={len(metrics)}")


def gen_rejection_mask_scenarios():
    torch.manual_seed(1)
    bsz, resp_len = 8, 32
    lengths = [32, 28, 20, 32, 15, 32, 10, 25]
    response_mask = make_response_mask(bsz, resp_len, lengths)
    log_ratio = (torch.rand(bsz, resp_len) - 0.5) * 1.2  # realistic small log-ratio drift

    run_rejection_mask_scenario("token_k1_icepop", log_ratio, response_mask, "token_k1", "0.5_1.8")
    run_rejection_mask_scenario("token_k2_token_k3_chained", log_ratio, response_mask, "token_k2,token_k3", "0.4,0.4")
    run_rejection_mask_scenario("seq_sum_k2", log_ratio, response_mask, "seq_sum_k2", "4.0")
    run_rejection_mask_scenario("seq_mean_k3", log_ratio, response_mask, "seq_mean_k3", "0.3")
    run_rejection_mask_scenario("seq_max_k2", log_ratio, response_mask, "seq_max_k2", "0.3")
    run_rejection_mask_scenario("seq_sum_k1_icepop", log_ratio, response_mask, "seq_sum_k1", "1.0_12.0")

    # Extreme thresholds: everything rejected / nothing rejected.
    run_rejection_mask_scenario("extreme_all_rejected", log_ratio, response_mask, "token_k2", "1e-6")
    run_rejection_mask_scenario("extreme_all_kept", log_ratio, response_mask, "token_k2", "1e6")

    # Singleton batch.
    singleton_mask = torch.ones(1, resp_len)
    singleton_log_ratio = (torch.rand(1, resp_len) - 0.5) * 1.2
    run_rejection_mask_scenario("singleton_batch", singleton_log_ratio, singleton_mask, "token_k1", "0.5_1.8")

    # Empty batch -- exercises the `log_ratio.shape[0] == 0` early return.
    empty_log_ratio = torch.zeros(0, resp_len)
    empty_mask = torch.zeros(0, resp_len)
    run_rejection_mask_scenario("empty_batch", empty_log_ratio, empty_mask, "token_k1", "0.5_1.8")

    # Zero-variance log_ratio (exact zero everywhere -> ratio == 1 everywhere).
    zero_log_ratio = torch.zeros(bsz, resp_len)
    run_rejection_mask_scenario("zero_variance_log_ratio", zero_log_ratio, response_mask, "token_k1", "0.5_1.8")


def gen_rs_metrics_scenarios():
    torch.manual_seed(2)
    bsz, resp_len = 6, 16
    lengths = [16, 14, 10, 16, 8, 12]
    response_mask = make_response_mask(bsz, resp_len, lengths)
    log_ratio = (torch.rand(bsz, resp_len) - 0.5) * 1.0
    log_ratio_safe = torch.clamp(log_ratio, min=-rc.SAFETY_BOUND, max=rc.SAFETY_BOUND)
    token_k2 = 0.5 * log_ratio_safe**2
    seq_valid_mask = response_mask.sum(dim=-1) > 0

    def run(name, option_name, rs_statistic, level, per_sequence_values, upper, lower, apply_lower):
        metrics = rc.compute_rs_metrics(
            option_name=option_name,
            rs_statistic=rs_statistic,
            response_mask=response_mask,
            seq_valid_mask=seq_valid_mask,
            level=level,
            per_sequence_values=per_sequence_values,
            rollout_rs_threshold=upper,
            rollout_rs_threshold_lower=lower,
            apply_lower_threshold=apply_lower,
        )
        path = OUT_DIR / f"rs_metrics_{name}.txt"
        with open(path, "w") as f:
            dump_tensor(f, "rs_statistic", rs_statistic)
            dump_tensor(f, "response_mask", response_mask)
            dump_bool_tensor(f, "seq_valid_mask", seq_valid_mask)
            dump_str(f, "option_name", option_name)
            dump_str(f, "level", level)
            dump_scalar(f, "has_per_sequence_values", 1.0 if per_sequence_values is not None else 0.0)
            if per_sequence_values is not None:
                dump_tensor(f, "per_sequence_values", per_sequence_values)
            dump_scalar(f, "rollout_rs_threshold", upper)
            dump_scalar(f, "rollout_rs_threshold_lower", lower)
            dump_scalar(f, "apply_lower_threshold", 1.0 if apply_lower else 0.0)
            dump_metrics(f, metrics)
        print(f"[rs_metrics:{name}] wrote {path.name}  n_metrics={len(metrics)}")

    run("token_level_normal", "token_k2", token_k2, "token", None, 0.4, 0.0, False)

    # Sequence-level options pass compute_rs_metrics a PER-TOKEN-SHAPED
    # `rs_statistic` (the per-sequence stat broadcast back to (bsz, resp_len)
    # via `.unsqueeze(-1).expand_as(response_mask)`), exactly as
    # compute_rollout_rejection_mask's own seq_sum/seq_mean/seq_max branches
    # do internally -- `rs_statistic` must always be response_mask-shaped
    # since `{prefix}_mean` unconditionally calls masked_mean(rs_statistic,
    # response_mask) regardless of `level`. `per_sequence_values` carries the
    # true (bsz,) per-sequence tensor separately.
    seq_mean_k2 = rc.verl_F.masked_mean(token_k2, response_mask, axis=-1)
    seq_mean_k2_expanded = seq_mean_k2.unsqueeze(-1).expand_as(response_mask)
    run("sequence_level_no_lower", "seq_mean_k2", seq_mean_k2_expanded, "sequence", seq_mean_k2, 0.3, 0.0, False)
    run("sequence_level_with_lower", "seq_mean_k2", seq_mean_k2_expanded, "sequence", seq_mean_k2, 0.3, 0.05, True)
    # Contrived: seq_valid_mask all False despite response_mask having valid tokens,
    # forcing seq_values_valid.numel() == 0 with level="sequence" (see rs_metrics
    # header's note: this combination can't arise from compute_rollout_rejection_mask's
    # own internal call, since it always derives seq_valid_mask consistently from
    # response_mask -- but compute_rs_metrics is a public standalone function with no
    # such consistency check, so this is still real fixture coverage of that branch).
    contrived_seq_valid_mask = torch.zeros(bsz, dtype=torch.bool)
    metrics = rc.compute_rs_metrics(
        option_name="seq_mean_k2",
        rs_statistic=seq_mean_k2_expanded,
        response_mask=response_mask,
        seq_valid_mask=contrived_seq_valid_mask,
        level="sequence",
        per_sequence_values=seq_mean_k2,
        rollout_rs_threshold=0.3,
        rollout_rs_threshold_lower=0.05,
        apply_lower_threshold=True,
    )
    path = OUT_DIR / "rs_metrics_contrived_empty_seq_valid.txt"
    with open(path, "w") as f:
        dump_tensor(f, "rs_statistic", seq_mean_k2_expanded)
        dump_tensor(f, "response_mask", response_mask)
        dump_bool_tensor(f, "seq_valid_mask", contrived_seq_valid_mask)
        dump_str(f, "option_name", "seq_mean_k2")
        dump_str(f, "level", "sequence")
        dump_scalar(f, "has_per_sequence_values", 1.0)
        dump_tensor(f, "per_sequence_values", seq_mean_k2)
        dump_scalar(f, "rollout_rs_threshold", 0.3)
        dump_scalar(f, "rollout_rs_threshold_lower", 0.05)
        dump_scalar(f, "apply_lower_threshold", 1.0)
        dump_metrics(f, metrics)
    print(f"[rs_metrics:contrived_empty_seq_valid] wrote {path.name}  n_metrics={len(metrics)}")


# ---------------------------------------------------------------------------
# 3. compute_rollout_correction_weights / compute_is_metrics
# ---------------------------------------------------------------------------


def run_is_weights_scenario(name, log_ratio, response_mask, rollout_is, rollout_is_threshold, batch_normalize):
    weights, metrics = rc.compute_rollout_correction_weights(
        log_ratio=log_ratio,
        response_mask=response_mask,
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
        rollout_is_batch_normalize=batch_normalize,
    )
    path = OUT_DIR / f"is_weights_{name}.txt"
    with open(path, "w") as f:
        dump_tensor(f, "log_ratio", log_ratio)
        dump_tensor(f, "response_mask", response_mask)
        dump_str(f, "rollout_is", rollout_is)
        dump_str(f, "rollout_is_threshold", str(rollout_is_threshold))
        dump_scalar(f, "rollout_is_batch_normalize", 1.0 if batch_normalize else 0.0)
        dump_tensor(f, "ref_weights", weights)
        dump_metrics(f, metrics)
    print(f"[is_weights:{name}] wrote {path.name}  n_metrics={len(metrics)}")


def gen_is_weights_scenarios():
    torch.manual_seed(3)
    bsz, resp_len = 8, 24
    lengths = [24, 20, 16, 24, 12, 24, 8, 18]
    response_mask = make_response_mask(bsz, resp_len, lengths)
    log_ratio = (torch.rand(bsz, resp_len) - 0.5) * 1.5

    run_is_weights_scenario("token_tis", log_ratio, response_mask, "token", 2.0, False)
    run_is_weights_scenario("token_icepop", log_ratio, response_mask, "token", "0.2_3.0", False)
    run_is_weights_scenario("sequence_tis", log_ratio, response_mask, "sequence", 1.5, False)
    run_is_weights_scenario("sequence_icepop", log_ratio, response_mask, "sequence", "0.3_2.5", False)
    run_is_weights_scenario("token_tis_batch_normalize", log_ratio, response_mask, "token", 2.0, True)
    run_is_weights_scenario("sequence_batch_normalize", log_ratio, response_mask, "sequence", 1.5, True)
    run_is_weights_scenario("extreme_tight_tis", log_ratio, response_mask, "token", 1.001, False)

    singleton_mask = torch.ones(1, resp_len)
    singleton_log_ratio = (torch.rand(1, resp_len) - 0.5) * 1.5
    run_is_weights_scenario("singleton_batch", singleton_log_ratio, singleton_mask, "token", 2.0, False)

    zero_log_ratio = torch.zeros(bsz, resp_len)
    run_is_weights_scenario("zero_variance", zero_log_ratio, response_mask, "token", 2.0, False)


def gen_is_metrics_scenarios():
    torch.manual_seed(4)
    bsz, resp_len = 6, 16
    lengths = [16, 14, 10, 16, 8, 12]
    response_mask = make_response_mask(bsz, resp_len, lengths)
    log_ratio = (torch.rand(bsz, resp_len) - 0.5) * 1.2

    def run(name, rollout_is, threshold_lower):
        log_ratio_safe = torch.clamp(log_ratio, min=-rc.SAFETY_BOUND, max=rc.SAFETY_BOUND)
        if rollout_is == "token":
            log_ratio_for_metrics = log_ratio
            raw_weights = torch.exp(log_ratio_safe)
        else:
            log_ratio_sum = rc.verl_F.masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(-1)
            log_ratio_for_metrics = log_ratio_sum
            raw_weights = torch.exp(torch.clamp(log_ratio_sum, min=-rc.SAFETY_BOUND, max=rc.SAFETY_BOUND)).expand_as(
                log_ratio
            )
        raw_weights = raw_weights * response_mask
        weights = raw_weights.clamp(max=2.0)
        metrics = rc.compute_is_metrics(
            rollout_is_weights=weights,
            raw_rollout_is_weights=raw_weights,
            log_ratio_for_metrics=log_ratio_for_metrics,
            response_mask=response_mask,
            rollout_is=rollout_is,
            rollout_is_threshold=2.0,
            rollout_is_threshold_lower=threshold_lower,
        )
        path = OUT_DIR / f"is_metrics_{name}.txt"
        with open(path, "w") as f:
            dump_tensor(f, "rollout_is_weights", weights)
            dump_tensor(f, "raw_rollout_is_weights", raw_weights)
            dump_tensor(f, "log_ratio_for_metrics", log_ratio_for_metrics)
            dump_tensor(f, "response_mask", response_mask)
            dump_str(f, "rollout_is", rollout_is)
            dump_scalar(f, "rollout_is_threshold", 2.0)
            dump_scalar(f, "has_threshold_lower", 1.0 if threshold_lower is not None else 0.0)
            if threshold_lower is not None:
                dump_scalar(f, "rollout_is_threshold_lower", threshold_lower)
            dump_metrics(f, metrics)
        print(f"[is_metrics:{name}] wrote {path.name}  n_metrics={len(metrics)}")

    run("token_default_lower", "token", None)
    run("token_explicit_lower", "token", 0.25)
    run("sequence_default_lower", "sequence", None)


# ---------------------------------------------------------------------------
# 4. compute_offpolicy_metrics / compute_rollout_corr_metrics_from_logprobs
# ---------------------------------------------------------------------------


def run_offpolicy_scenario(name, old_log_prob, rollout_log_prob, response_mask):
    metrics = rc.compute_offpolicy_metrics(
        old_log_prob=old_log_prob, rollout_log_prob=rollout_log_prob, response_mask=response_mask
    )
    path = OUT_DIR / f"offpolicy_metrics_{name}.txt"
    with open(path, "w") as f:
        dump_tensor(f, "old_log_prob", old_log_prob)
        dump_scalar(f, "has_rollout_log_prob", 1.0 if rollout_log_prob is not None else 0.0)
        if rollout_log_prob is not None:
            dump_tensor(f, "rollout_log_prob", rollout_log_prob)
        dump_tensor(f, "response_mask", response_mask)
        dump_metrics(f, metrics)
    print(f"[offpolicy_metrics:{name}] wrote {path.name}  n_metrics={len(metrics)}")


def gen_offpolicy_scenarios():
    torch.manual_seed(5)
    bsz, resp_len = 8, 20
    lengths = [20, 18, 14, 20, 10, 20, 6, 16]
    response_mask = make_response_mask(bsz, resp_len, lengths)
    old_log_prob = -torch.rand(bsz, resp_len) * 2.0 - 0.1
    rollout_log_prob = old_log_prob + (torch.rand(bsz, resp_len) - 0.5) * 0.8

    run_offpolicy_scenario("with_rollout", old_log_prob, rollout_log_prob, response_mask)
    run_offpolicy_scenario("without_rollout", old_log_prob, None, response_mask)
    run_offpolicy_scenario("identical_policies", old_log_prob, old_log_prob.clone(), response_mask)

    singleton_mask = torch.ones(1, resp_len)
    singleton_old = -torch.rand(1, resp_len) * 2.0 - 0.1
    singleton_rollout = singleton_old + (torch.rand(1, resp_len) - 0.5) * 0.8
    run_offpolicy_scenario("singleton_batch", singleton_old, singleton_rollout, singleton_mask)

    # compute_rollout_corr_metrics_from_logprobs -- thin "rollout_corr/" prefix wrapper.
    metrics = rc.compute_rollout_corr_metrics_from_logprobs(
        log_prob=old_log_prob, rollout_log_prob=rollout_log_prob, response_mask=response_mask
    )
    path = OUT_DIR / "corr_metrics_from_logprobs.txt"
    with open(path, "w") as f:
        dump_tensor(f, "log_prob", old_log_prob)
        dump_tensor(f, "rollout_log_prob", rollout_log_prob)
        dump_tensor(f, "response_mask", response_mask)
        dump_metrics(f, metrics)
    print(f"[corr_metrics_from_logprobs] wrote {path.name}  n_metrics={len(metrics)}")


# ---------------------------------------------------------------------------
# 5. score_centering: dummy_rollout_topk / pad_rollout_topk
# ---------------------------------------------------------------------------


def gen_dummy_pad_topk_scenarios():
    ids, log_probs = sc.dummy_rollout_topk(num_rows=5, k=4)
    path = OUT_DIR / "dummy_topk_basic.txt"
    with open(path, "w") as f:
        dump_scalar(f, "num_rows", 5)
        dump_scalar(f, "k", 4)
        dump_tensor(f, "ref_ids", ids.float())
        dump_tensor(f, "ref_log_probs", log_probs)
    print(f"[dummy_topk:basic] wrote {path.name}")

    torch.manual_seed(6)
    response_length, k, prompt_length = 5, 4, 7
    response_topk_ids = torch.randint(0, 100, (response_length, k))
    response_topk_log_probs = -torch.rand(response_length, k) * 3.0
    full_ids, full_log_probs = sc.pad_rollout_topk(response_topk_ids, response_topk_log_probs, prompt_length)
    path = OUT_DIR / "pad_topk_basic.txt"
    with open(path, "w") as f:
        dump_tensor(f, "response_topk_ids", response_topk_ids.float())
        dump_tensor(f, "response_topk_log_probs", response_topk_log_probs)
        dump_scalar(f, "prompt_length", prompt_length)
        dump_tensor(f, "ref_full_ids", full_ids.float())
        dump_tensor(f, "ref_full_log_probs", full_log_probs)
    print(f"[pad_topk:basic] wrote {path.name}")

    # prompt_length == 1 -> start == 0, the boundary this port's extra TORCH_CHECK
    # guards (see score_centering.cpp); still a valid, Python-matching case.
    full_ids2, full_log_probs2 = sc.pad_rollout_topk(response_topk_ids, response_topk_log_probs, 1)
    path = OUT_DIR / "pad_topk_prompt_length_one.txt"
    with open(path, "w") as f:
        dump_tensor(f, "response_topk_ids", response_topk_ids.float())
        dump_tensor(f, "response_topk_log_probs", response_topk_log_probs)
        dump_scalar(f, "prompt_length", 1)
        dump_tensor(f, "ref_full_ids", full_ids2.float())
        dump_tensor(f, "ref_full_log_probs", full_log_probs2)
    print(f"[pad_topk:prompt_length_one] wrote {path.name}")


# ---------------------------------------------------------------------------
# 6. score_centering: topk_log_probs_from_logits (forward + backward)
# ---------------------------------------------------------------------------


def run_topk_log_probs_scenario(name, n, v, k, chunk_size):
    logits = torch.randn(n, v, dtype=torch.float32, requires_grad=True)
    token_ids = torch.randint(0, v, (n, k))
    output = sc.topk_log_probs_from_logits(logits, token_ids, chunk_size=chunk_size)
    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    path = OUT_DIR / f"topk_log_probs_{name}.txt"
    with open(path, "w") as f:
        dump_tensor(f, "logits", logits)
        dump_tensor(f, "token_ids", token_ids.float())
        dump_scalar(f, "chunk_size", chunk_size)
        dump_tensor(f, "grad_output", grad_output)
        dump_tensor(f, "ref_output", output)
        dump_tensor(f, "ref_grad_logits", logits.grad)
    print(f"[topk_log_probs:{name}] wrote {path.name}")


def gen_topk_log_probs_scenarios():
    torch.manual_seed(7)
    run_topk_log_probs_scenario("single_chunk", n=6, v=50, k=4, chunk_size=8)
    run_topk_log_probs_scenario("multi_chunk", n=10, v=50, k=4, chunk_size=3)
    run_topk_log_probs_scenario("large_vocab", n=4, v=500, k=8, chunk_size=2)


# ---------------------------------------------------------------------------
# 7. score_centering: score_centering_weight_fn
# ---------------------------------------------------------------------------


def gen_weight_fn_scenarios():
    torch.manual_seed(8)
    ratio = torch.rand(20) * 4.0  # spans below/within/above typical thresholds

    def run(name, rollout_is, threshold):
        weight_fn = sc.score_centering_weight_fn(rollout_is, threshold)
        out = weight_fn(ratio)
        path = OUT_DIR / f"weight_fn_{name}.txt"
        with open(path, "w") as f:
            dump_tensor(f, "ratio", ratio)
            dump_scalar(f, "has_rollout_is", 1.0 if rollout_is is not None else 0.0)
            if rollout_is is not None:
                dump_str(f, "rollout_is", rollout_is)
            dump_str(f, "rollout_is_threshold", str(threshold))
            dump_tensor(f, "ref_weight", out)
        print(f"[weight_fn:{name}] wrote {path.name}")

    run("none", None, 2.0)
    run("token_tis", "token", 2.0)
    run("token_icepop", "token", "0.5_2.0")


# ---------------------------------------------------------------------------
# 8. score_centering: score_centering_correction / *_support_correction / *_auto_correction
# ---------------------------------------------------------------------------


def make_head(n, k, seed, concentrate=False):
    """Builds a valid (<=1 total mass) top-k log-prob head via log_softmax of random logits."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, 64, generator=g)
    if concentrate:
        logits[:, :k] += 6.0  # pushes most mass into the first k logits (head ~= full support)
    log_probs_full = torch.log_softmax(logits, dim=-1)
    topk = log_probs_full.topk(k, dim=-1)
    return topk.values, topk.indices, log_probs_full


def run_correction_scenario(name, train_head_log_probs, sampler_head_log_probs, weight_fn, eps=1e-6):
    train = train_head_log_probs.clone().requires_grad_(True)
    correction, sampler_mass, train_mass = sc.score_centering_correction(train, sampler_head_log_probs, weight_fn, eps)
    correction.sum().backward()
    path = OUT_DIR / f"correction_{name}.txt"
    with open(path, "w") as f:
        dump_tensor(f, "train_head_log_probs", train_head_log_probs)
        dump_tensor(f, "sampler_head_log_probs", sampler_head_log_probs)
        dump_scalar(f, "eps", eps)
        dump_tensor(f, "ref_correction", correction)
        dump_tensor(f, "ref_sampler_head_mass", sampler_mass)
        dump_tensor(f, "ref_train_head_mass", train_mass)
        dump_tensor(f, "ref_grad_train_head_log_probs", train.grad)
    print(f"[correction:{name}] wrote {path.name}")


def run_support_correction_scenario(name, train_head_log_probs, sampler_head_log_probs, weight_fn):
    train = train_head_log_probs.clone().requires_grad_(True)
    correction, log_support_mass, sampler_mass = sc.score_centering_support_correction(
        train, sampler_head_log_probs, weight_fn
    )
    (correction.sum() + log_support_mass.sum()).backward()
    path = OUT_DIR / f"support_correction_{name}.txt"
    with open(path, "w") as f:
        dump_tensor(f, "train_head_log_probs", train_head_log_probs)
        dump_tensor(f, "sampler_head_log_probs", sampler_head_log_probs)
        dump_tensor(f, "ref_correction", correction)
        dump_tensor(f, "ref_log_support_mass", log_support_mass)
        dump_tensor(f, "ref_sampler_head_mass", sampler_mass)
        dump_tensor(f, "ref_grad_train_head_log_probs", train.grad)
    print(f"[support_correction:{name}] wrote {path.name}")


def run_auto_correction_scenario(name, train_head_log_probs, sampler_head_log_probs, weight_fn, support_complete):
    train = train_head_log_probs.clone().requires_grad_(True)
    correction, log_norm_shift, sampler_mass, train_mass = sc.score_centering_auto_correction(
        train, sampler_head_log_probs, weight_fn, support_complete
    )
    (correction.sum() + log_norm_shift.sum()).backward()
    path = OUT_DIR / f"auto_correction_{name}.txt"
    with open(path, "w") as f:
        dump_tensor(f, "train_head_log_probs", train_head_log_probs)
        dump_tensor(f, "sampler_head_log_probs", sampler_head_log_probs)
        dump_bool_tensor(f, "support_complete", support_complete)
        dump_tensor(f, "ref_correction", correction)
        dump_tensor(f, "ref_log_norm_shift", log_norm_shift)
        dump_tensor(f, "ref_sampler_head_mass", sampler_mass)
        dump_tensor(f, "ref_train_head_mass", train_mass)
        dump_tensor(f, "ref_grad_train_head_log_probs", train.grad)
    print(f"[auto_correction:{name}] wrote {path.name}")


def gen_score_centering_correction_scenarios():
    n, k = 10, 6

    train_head, _, _ = make_head(n, k, seed=10)
    sampler_head, _, _ = make_head(n, k, seed=11)

    tis_weight_fn = sc.score_centering_weight_fn("token", 2.0)
    identity_weight_fn = sc.score_centering_weight_fn(None, 2.0)
    icepop_weight_fn = sc.score_centering_weight_fn("token", "0.3_2.0")

    run_correction_scenario("normal_tis", train_head, sampler_head, tis_weight_fn)
    run_correction_scenario("weight_fn_identity", train_head, sampler_head, identity_weight_fn)
    run_correction_scenario("weight_fn_icepop", train_head, sampler_head, icepop_weight_fn)

    # Near-full head mass (ill-conditioned rho per the docstring -- gradient should
    # still be well-defined even though the correction's own value is not meaningful).
    train_head_full, _, _ = make_head(n, k, seed=12, concentrate=True)
    sampler_head_full, _, _ = make_head(n, k, seed=13, concentrate=True)
    run_correction_scenario("near_full_head_mass", train_head_full, sampler_head_full, tis_weight_fn)

    # --- support correction: "head covers full support" vs "incomplete head" ---
    # Full support: sampler_head_log_probs has no -inf entries at all.
    run_support_correction_scenario("full_support", train_head, sampler_head, tis_weight_fn)

    # Incomplete head: mark the lowest-probability head slot as out-of-support (-inf)
    # for every row, i.e. "predefined approximation" per the module docstring.
    sampler_incomplete = sampler_head.clone()
    sampler_incomplete[:, -1] = float("-inf")
    run_support_correction_scenario("incomplete_support", train_head, sampler_incomplete, tis_weight_fn)
    run_support_correction_scenario("incomplete_support_icepop", train_head, sampler_incomplete, icepop_weight_fn)

    # --- auto correction: mixes both branches within one batch via support_complete ---
    support_complete = torch.tensor([True, False, True, False, True, False, True, False, True, False])
    sampler_mixed = sampler_head.clone()
    # Rows marked incomplete additionally carry a -inf slot (consistent with how a
    # real incomplete recorded head would look); rows marked complete keep their
    # full (finite) head as the verified whole-support case.
    sampler_mixed[~support_complete, -1] = float("-inf")
    run_auto_correction_scenario("mixed_support", train_head, sampler_mixed, tis_weight_fn, support_complete)

    all_complete = torch.ones(n, dtype=torch.bool)
    run_auto_correction_scenario("all_complete", train_head, sampler_head, tis_weight_fn, all_complete)

    all_incomplete = torch.zeros(n, dtype=torch.bool)
    run_auto_correction_scenario("all_incomplete", train_head, sampler_incomplete, tis_weight_fn, all_incomplete)


def main():
    gen_threshold_parsing()
    gen_rejection_mask_scenarios()
    gen_rs_metrics_scenarios()
    gen_is_weights_scenarios()
    gen_is_metrics_scenarios()
    gen_offpolicy_scenarios()
    gen_dummy_pad_topk_scenarios()
    gen_topk_log_probs_scenarios()
    gen_weight_fn_scenarios()
    gen_score_centering_correction_scenarios()
    print(f"\nAll fixtures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
