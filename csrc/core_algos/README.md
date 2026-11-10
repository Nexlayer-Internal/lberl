# `csrc/core_algos` — C++/libtorch port of verl's GRPO advantage + PPO-clip loss math

First concrete migration step of the Python-elimination plan in `PLAN.md`
(section 6, item 1 / table item #1): a libtorch port of the GRPO
advantage computation, the PPO-clip ("vanilla") policy loss, and the KL
penalty estimators from `verl/trainer/ppo/core_algos.py`. This is the
literal algorithm executed every training step for the validated
`algorithm.adv_estimator=grpo` / `actor_rollout_ref.actor.policy_loss.loss_mode=vanilla`
config — confirmed framework-agnostic (only `torch`/`numpy`/`omegaconf`
imports in the source file), which is exactly why `PLAN.md` recommended
porting it first.

## Why `csrc/` at the repo root

There is no existing C++ code anywhere in this repository (checked before
adding this), so there was no established convention to follow here. `csrc/`
is the common convention for "C++ source living alongside a primarily-Python
package" (used by PyTorch itself, among others) and reads unambiguously next
to `verl/`. Placed at the repo root rather than nested under `verl/` since
this is explicitly the first of several planned C++ ports (PLAN.md section 6
lists `rollout_corr_helper.py`/`score_centering.py`, a `DataProto` tensor-batch
boundary, etc. as likely next additions) and a root-level `csrc/` gives each
its own subdirectory (`csrc/core_algos/`, presumably `csrc/rollout_corr/`
later, ...) without the directory accumulating unrelated Python-package
baggage.

## What's ported (and exactly where it came from)

From `verl/trainer/ppo/core_algos.py` on `main` (this branch was created from
a fresh clone of `main`, which does not itself contain `PLAN.md` — see
"A correction to the task's framing" below for why):

| C++ (`verl_core_algos::`) | Python source | Lines (this checkout) |
|---|---|---|
| `compute_grpo_outcome_advantage` | `compute_grpo_outcome_advantage` (`@register_adv_est(AdvantageEstimator.GRPO)`) | L267-331 |
| `agg_loss` | `agg_loss` | L1138-1199 |
| `compute_policy_loss_vanilla` | `compute_policy_loss_vanilla` (`@register_policy_loss("vanilla")`) | L1278-1369 |
| `kl_penalty_forward` | `kl_penalty_forward` | L2154-2189 |
| `kl_penalty` | `kl_penalty` (incl. the `k3+`-style straight-through gradient trick) | L2126-2151 |
| `compute_rewards` | `compute_rewards` (applies the KL penalty term to token-level reward) | L1122-1135 |
| `masked_sum` / `masked_mean` | `verl/utils/torch_functional.py::masked_sum`/`masked_mean` (dependency of the above, not independently interesting, ported because it's required) | — |

`compute_policy_loss_vanilla` is the actively-registered, non-deprecated
function; the older `compute_policy_loss` (same math, pre-registry API,
marked `@deprecated`) was read for reference but intentionally not ported —
porting both would just duplicate the same formulas.

## What's deliberately NOT ported here (scope boundary)

Per the task's own scope instructions (port the core math; include
PLAN.md's #2 priority *only* if it's "not much additional surface" — it
isn't, here):

- **The other 15+ advantage-estimator/policy-loss variants in the same
  Python file**: GAE, GRPO_VECTORIZED, GDPO, GRPO_PASSK,
  REINFORCE(++)/baseline, RLOO(_VECTORIZED), OPO, REMAX, GPG,
  optimal-token-baseline (+ multi-turn variant), `dppo_tv`, `dppo_kl`,
  `gspo`, `sapo`, `clip_cov`, `kl_cov`, `geo_mean`, `cispo`,
  `compute_entropy_loss`, `compute_value_loss`,
  `compute_pf_ppo_reweight_data`, `compute_policy_loss_reinforce`,
  `bypass_mode`. None of these are exercised by the validated GRPO/vanilla
  config this port targets; porting them now would be scope creep beyond
  "the literal GRPO algorithm running every training step."
- **`rollout_corr_helper.py` (1144 lines) + `score_centering.py` (400
  lines)** — `PLAN.md`'s own table item #2, correctly identified there as
  "same pure-tensor-math profile" as this port and a reasonable
  next-in-line candidate. Explicitly *not* folded into this PR: at ~1,500
  lines combined it is roughly 6x the surface of what's ported here, which
  is exactly the kind of scope creep the task asked to avoid ("not much
  additional surface"). Left for a dedicated follow-up port, as `PLAN.md`
  section 6 itself sequences it as a separate (if adjacent) step.

## A correction to the task's framing (documented, not silently fixed)

The task description states `PLAN.md` was "just merged to `main`." That is
not quite what the repository shows: `PLAN.md` was merged via PR #1
(`Nexlayer-Internal/lberl#1`, "Python-elimination survey and prioritized
migration plan") into the branch `codex/fix-nvshmem-image-20261023`, **not**
into `main` — `main` does not contain `PLAN.md` at all (verified directly:
`git show origin/main:PLAN.md` fails). This port was read from `PLAN.md` as
it exists on that branch, cross-checked for line-number accuracy against
`main`'s copy of `core_algos.py` (the two differ only in unrelated regions —
a `token-sum` loss-agg-mode branch, a `returns`-masking fix, a new `dro`
policy loss — none of which touch the functions ported here), and this PR's
branch/diff is against `main`, as asked. Flagging this now so it isn't
rediscovered independently later, same spirit as `PLAN.md`'s own "flagging
the second seam so it isn't rediscovered independently" footnote.

## Validation methodology

`tests/gen_fixtures.py` imports the **real, unmodified**
`verl/trainer/ppo/core_algos.py` from this checkout (not a reimplementation)
and runs it on several labeled scenarios, dumping both inputs and Python's
own reference outputs as plain text to `tests/fixtures/*.txt`.
`tests/validate_core_algos.cpp` reads those fixtures, runs the same inputs
through the C++ port above, and reports max-abs-diff per scenario/output
against a `1e-5` tolerance (float32 compute; actual diffs observed were
`~1e-7`-`1e-9`, i.e. float32 rounding noise from the text round-trip, not a
real numerical gap — see the run log below).

Scenarios (18 total, `tests/fixtures/*.txt`):

- **GRPO advantage** (batch=8, group size=4, response_length=64):
  - `mixed` / `mixed_no_std_norm`: realistic variable-length completions
    (padding via `response_mask`), one outcome reward per sequence placed at
    the last valid token, both with and without std-normalization
    (Dr.GRPO-style).
  - `zero_variance_group`: a group of 4 samples with an **identical**
    reward — exercises the `epsilon`-guarded std path; asserted
    not-NaN both in the Python generator and the C++ harness.
  - `singleton_groups`: group size 1 for every sample — exercises the
    explicit `mean=0, std=1` special case in the Python source.
- **Policy loss** (same batch/shape):
  - `mixed_clip_token_mean`: moderate log-prob drift, exercises the
    standard (non-dual) clip branch, `loss_agg_mode="token-mean"`.
  - `near_one_seq_mean_token_mean`: ratio kept near 1.0 (near-zero
    clipfrac), `loss_agg_mode="seq-mean-token-mean"`.
  - `extreme_dual_clip`: large log-prob drift + asymmetric
    `clip_ratio_low`/`clip_ratio_high` + a tight `clip_ratio_c=2.0` —
    exercises the dual-clip (`clip_ratio_c`) branch, confirmed nonzero
    `pg_clipfrac_lower` in the reference output.
  - `seq_mean_token_sum_norm`: exercises the `seq-mean-token-sum-norm`
    aggregation mode (the `loss_scale_factor` branch of `agg_loss`).
  - `with_rollout_is_weights`: exercises the optional
    `rollout_is_weights` multiplicative correction term.
- **KL penalty** (`k1`, `abs`, `k2`/`mse`, `k3`, `k3+` straight-through,
  plus a `zero_kl` sanity check with `logprob == ref_logprob`).
- **`compute_rewards`**: KL-penalty-on (`kl_ratio=0.1`) and KL-penalty-off
  (`kl_ratio=0.0`) scenarios.

### Honesty disclosure: input realism

The original validated GSM8K/Qwen3-4B smoke test's raw launch script and
per-step logs were not recoverable in this environment — only a summary
README survived (recorded `data.train_batch_size=8` and aggregate scalar
metrics: `actor/loss: 7.68e-07`, `actor/kl_loss: 0.000768`,
`actor/entropy: 0.115`). Per-token log-probs/advantages/group assignments
are intermediate values that were never logged at that granularity in the
first place, so they are not recoverable even in principle from the
surviving scalars, with or without the original log file. The fixtures
above therefore use clearly-labeled realistic synthetic values: batch size
8 (matching the one recovered real hyperparameter), GRPO group size 4 (a
common `rollout.n` default), response length 64. The ported math has no
dependence on sequence length beyond elementwise ops and masked
reductions, so a shorter synthetic response length does not reduce
coverage of the logic actually being validated.

## Build + run

No CMake was available in the environment this was developed in, so the
build that was actually exercised is `build.sh` (direct `clang++`/`c++`
invocation against whatever `python3 -c "import torch"` resolves to —
tested here against a pip-installed `torch==2.8.0` CPU wheel for
arm64 macOS; this code has no CUDA dependency, so a CPU-only libtorch is
sufficient for both building and validating it):

```sh
cd csrc/core_algos
TORCH_PYTHON=python3 ./build.sh
./build/validate_core_algos ./tests/fixtures
```

A `CMakeLists.txt` is also provided for the conventional build path
(`find_package(Torch)`), mirroring `build.sh`'s flags and file list, but it
was **not** exercised in this environment (no `cmake` binary available) —
treat it as unverified until someone builds it for real.

To regenerate fixtures from a fresh Python environment (needs the real
`verl` package importable, i.e. its dependencies installed — `torch`,
`numpy`, `omegaconf`, plus whatever else `verl/__init__.py`'s import chain
pulls in for your checkout):

```sh
python3 csrc/core_algos/tests/gen_fixtures.py
```

### Actual run output (this environment, arm64 macOS, CPU libtorch 2.8.0)

```
[PASS] grpo_adv:mixed (advantages)  max_abs_diff=1.19209e-07 (tol=1e-05)
[PASS] grpo_adv:mixed (returns)  max_abs_diff=1.19209e-07 (tol=1e-05)
[PASS] grpo_adv:mixed_no_std_norm (advantages)  max_abs_diff=5.96046e-08 (tol=1e-05)
[PASS] grpo_adv:zero_variance_group (advantages)  max_abs_diff=1.19209e-07 (tol=1e-05)
[PASS] grpo_adv:singleton_groups (advantages)  max_abs_diff=2.98023e-08 (tol=1e-05)
[PASS] policy_loss:mixed_clip_token_mean (pg_loss)  max_abs_diff=1.39698e-09 (tol=1e-05)
[PASS] policy_loss:extreme_dual_clip (pg_loss)  max_abs_diff=0 (tol=1e-05)
[PASS] policy_loss:extreme_dual_clip (pg_clipfrac_lower)  max_abs_diff=0 (tol=1e-05)
[PASS] kl:k3plus (full)  max_abs_diff=0 (tol=1e-05)
[PASS] rewards:kl_on  max_abs_diff=0 (tol=1e-05)
...
ALL SCENARIOS PASSED
```

Full output (all 18 scenarios x their respective sub-checks, 46 individual
PASS lines) is reproducible via the commands above; not all lines are
reproduced here for brevity.

## What was, and wasn't, verified

- **Verified for real**: the port compiles cleanly (zero warnings) against
  a real libtorch (pip `torch==2.8.0`, CPU, arm64 macOS) via direct
  `clang++` invocation, links, runs, and numerically matches the real
  (unmodified) Python `core_algos.py` across 18 scenarios / 46 individual
  value comparisons, all within `1e-5` (observed: `1e-7`-`1e-9`, i.e.
  float32 rounding noise).
- **Not verified**: CUDA execution (no GPU in this environment; the ported
  code has no CUDA-specific logic — it's dtype/device-generic libtorch
  tensor ops — but "should work on GPU too" was not actually exercised on
  one). The provided `CMakeLists.txt` build path. Integration into the
  actual Python/Ray training loop (out of scope for this PR; the plan's own
  sequencing — section 6 — treats that as a later step once this core is
  proven standalone).
- **SSH/remote build machines**: the task description mentioned a
  possible existing libtorch build reachable over SSH as a time-saving
  option. That path was not used — a working CPU libtorch was already
  available locally via `pip show torch`, which was both sufficient (this
  code is pure tensor math, no CUDA dependency) and avoided touching any
  shared machine or another agent's in-progress build directories.
