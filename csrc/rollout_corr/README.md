# `csrc/rollout_corr` — C++/libtorch port of verl's off-policy / rollout-correction math

Second concrete migration step of the Python-elimination plan in `PLAN.md`
(section 6, table item #2): a libtorch port of `verl/trainer/ppo/
rollout_corr_helper.py` (1144 lines) and `verl/trainer/ppo/score_centering.py`
(400 lines) — truncated importance-sampling (IS) correction for
rollout-vs-train policy mismatch, rejection sampling (RS), and
training-inference-mismatch score centering per arXiv:2609.20807. `PLAN.md`
calls this "the same pure tensor math profile as #1 [`csrc/core_algos`],
larger and more intricate" and flags it as directly relevant to the future
`lbx` rollout integration (any precision gap between lbx-served logits and
libtorch-trained logits reintroduces exactly the mismatch this code
corrects) — treated accordingly, not as throwaway scaffolding.

Follows `csrc/core_algos`'s conventions throughout: same file layout
(`include/`, `src/`, `tests/`, `build.sh` + `CMakeLists.txt`), same
fixture-generation-from-real-Python-then-compare-in-C++ test harness style,
same CPU/CUDA dual-build path, same honesty-disclosure norms.

## What's ported vs. left as boundary, and why

### `rollout_corr_helper.py`

Ported (pure tensor-in/tensor-and-scalars-out math, `verl_rollout_corr`
namespace, `include/verl_rollout_corr/rollout_corr_helper.h` +
`src/rollout_corr_helper.cpp`):

| C++ | Python source |
|---|---|
| `parse_rollout_is_threshold` | `_parse_rollout_is_threshold` |
| `parse_rollout_rs_thresholds` | `_parse_rollout_rs_thresholds` |
| `compute_rollout_rejection_mask` | `compute_rollout_rejection_mask` |
| `compute_rs_metrics` | `compute_rs_metrics` |
| `compute_rollout_correction_weights` | `compute_rollout_correction_weights` |
| `compute_is_metrics` | `compute_is_metrics` |
| `compute_offpolicy_metrics` | `compute_offpolicy_metrics` |
| `compute_rollout_corr_metrics_from_logprobs` | `compute_rollout_corr_metrics_from_logprobs` |

Left as Python-side glue (does not touch tensor math, matches the
`core_algos.cpp` precedent of leaving `DataProto`-touching code as a
boundary):

- **`compute_rollout_correction_and_rejection_mask`,
  `compute_rollout_correction_and_add_to_batch`** — construct/mutate a
  `DataProto` directly. Not ported, per the task's own scope instructions and
  `PLAN.md` table item #7 (`DataProto`'s tensor half is a separate,
  not-yet-started roadmap item).
- **`apply_bypass_mode`** — **this is a deliberate deviation from the task's
  suggested port list, flagged honestly rather than silently done
  differently.** The task listed this as a function to port, reasoning by
  analogy to `core_algos.py`'s `omegaconf.DictConfig` scalar-hyperparameter
  usage. On actually reading it, `apply_bypass_mode` does zero tensor math:
  it (1) checks a key's presence in `batch.batch` (a `DataProto`'s tensor
  dict), (2) aliases one dict entry to another
  (`batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]`), and
  (3) mutates an `omegaconf` config object's fields
  (`policy_loss_config["rollout_correction"] = ...`,
  `policy_loss_config["loss_mode"] = "bypass_mode"`) inside a `with
  open_dict(...)` block. There is no numeric computation to port — a C++
  "port" of this function would just be renaming dict/config mutations with
  no libtorch value-add, and it directly touches both `DataProto` and
  `omegaconf.DictConfig`, the two boundary types this port (and its
  `core_algos` predecessor) deliberately stays out of. Left as Python-side
  glue for the same reason `compute_rollout_correction_and_add_to_batch` is:
  it is 100% orchestration/config plumbing, not math.

`_parse_rollout_is_threshold`/`_parse_rollout_rs_thresholds` are pure
string/scalar logic with no tensors at all (not even `omegaconf` objects —
they take already-unpacked `str | float` values) — ported anyway since
`compute_rollout_rejection_mask`/`compute_rollout_correction_weights`/
`score_centering_weight_fn` all call them directly and a C++ port of the
callers needs a C++ implementation of the parsers too. `ThresholdSpec` (a
small tagged union constructible from a `double` or a `std::string`) stands
in for Python's `str | float` dynamic dispatch — see
`rollout_corr_helper.h`.

**Multi-DP distributed reduction** (`compute_rollout_correction_weights`'s
`rollout_is_batch_normalize=True` branch calls `verl_F.distributed_masked_mean`,
which does a `torch.distributed.all_reduce`, when `torch.distributed` is
initialized): this is orchestration, not pure tensor math, so it is **not**
ported — exactly the same boundary `core_algos.cpp`'s `agg_loss` draws for
`dp_size > 1` (requiring a pre-aggregated `batch_num_tokens`/
`global_batch_size` from the caller rather than doing the all-reduce
itself). `compute_rollout_correction_weights` takes an optional
`distributed_weights_mean` parameter for the same purpose: pass an
already-all-reduced mean for the multi-rank case, leave it unset for
single-process (the only path this port directly validates).

### `score_centering.py`

Ported (actual math, same namespace, `include/verl_rollout_corr/
score_centering.h` + `src/score_centering.cpp`):

| C++ | Python source |
|---|---|
| `TopKLogProbsFromLogits` (`torch::autograd::Function<T>`) | `_TopKLogProbsFromLogits(torch.autograd.Function)` |
| `topk_log_probs_from_logits` | `topk_log_probs_from_logits` |
| `score_centering_weight_fn` | `score_centering_weight_fn` |
| `score_centering_correction` | `score_centering_correction` |
| `score_centering_support_correction` | `score_centering_support_correction` |
| `score_centering_auto_correction` | `score_centering_auto_correction` |
| `dummy_rollout_topk` | `dummy_rollout_topk` |
| `pad_rollout_topk` | `pad_rollout_topk` |

**`dummy_rollout_topk`/`pad_rollout_topk` judgment call**: the task
description flagged these as "look like test-fixture/HF-generation-hook
glue" and left the call to this port's judgment. On inspection, they are
**not** glue — they're plain tensor construction/slicing math
(`torch.arange`/`torch.full`/`torch.as_tensor`/tensor-slice-assignment) with
**zero** framework dependency: no `DataProto`, no `TensorDict`, no HF
tokenizer/generation API, nothing beyond `torch` and `math.log`. They build
the uniform-dummy and sampler-recorded top-k "head" tensors that feed
directly into `topk_log_probs_from_logits`/`score_centering_correction`'s
real math — i.e. they're part of the actual data-assembly path for score
centering, not test scaffolding (their names suggest "dummy"/"pad" as in
test fixtures, but `dummy_rollout_topk` is the real fallback for masked-out
rows in production, not a test helper). Ported for that reason; see
`score_centering.h`'s header comment for the one-line version of this same
argument.

**Not ported** (genuine framework glue, call sites for the functions above,
not math):

- **`score_centering_logits_processor`** — reads a `TensorDict` (`data["rollout_topk_ids"].values()`,
  nested-tensor semantics), calls `get_ulysses_sequence_parallel_world_size`/
  `slice_input_tensor` (Ulysses sequence-parallel slicing), and reads
  `ActorConfig`. This is the actual "boundary" the ported math functions are
  called from in production (inside the trainer's logits-processor hook);
  porting it would mean porting `TensorDict` and the Ulysses SP slicing
  logic too, well outside this PR's scope.
- **`score_centering_ppo_loss`** — dispatches between the logits processor
  above and `verl.workers.utils.losses.ppo_loss` (not ported, not in scope;
  itself a further boundary onto `ActorConfig`/`TensorDict`).

## CUDA-safety audit

Per the task's explicit ask (this exact bug — `Tensor::accessor<T,N>()`,
a CPU-only API, called on a tensor that could be on CUDA — was found and
fixed in `csrc/core_algos`'s GRPO-advantage port after an initial PR):

```sh
$ grep -n '\.accessor<\|\.data_ptr<' csrc/rollout_corr/src/*.cpp
# (no output)
```

**Zero hits.** Every raw-value extraction in both `rollout_corr_helper.cpp`
and `score_centering.cpp` is one of:
- `.item<double>()` / `.item<bool>()` on a 0-dim (scalar) reduction tensor —
  explicitly safe on any device (performs the device-to-host copy itself),
  same safe pattern `core_algos.cpp`'s fix moved everything else onto.
- A standard device-dispatched ATen op (`masked_fill`, `gather`,
  `scatter_add_`, `logsumexp`, `topk`, `where`, `clamp`, `narrow`/`copy_`,
  etc.) — these dispatch correctly on CUDA tensors by construction; nothing
  in this port does raw per-element/scalar pointer access the way
  `core_algos.py`'s original `compute_grpo_outcome_advantage` port did (that
  function's per-group mean/std bookkeeping needed an explicit host-side
  loop; nothing in `rollout_corr_helper.py`/`score_centering.py`'s math
  needs a comparable host-side loop over individual elements — it's
  masked-reduction/broadcast math throughout).

There is therefore no single "this line segfaults on CUDA" fix to report
here (unlike `core_algos`'s); `tests/cuda_smoke_test.cpp` instead exists as
a broad regression guard (every ported function, run on real CUDA tensors,
cross-checked against an identical-value CPU run) so this invariant doesn't
silently rot as the port evolves.

The one place genuinely worth flagging: `pad_rollout_topk`'s
`start = prompt_length - 1` is used as a `Tensor::narrow` offset. Unlike
Python's slice (which would silently wrap for a negative `start`), libtorch's
`narrow` would throw on a negative offset regardless of device — this port
adds an explicit `TORCH_CHECK(start >= 0, ...)` ahead of it for a clearer
error message, not a CUDA-specific fix (the underlying behavior difference
from Python is intentional and documented in `score_centering.cpp`).

## Build + run

Same no-cmake sandbox constraint as `csrc/core_algos` (see that module's
README): the build actually exercised here is `build.sh` (direct `clang++`
invocation against a pip-installed libtorch). This module links against
`verl_core_algos`'s `masked_sum`/`masked_mean` (reused, not duplicated) —
`build.sh` compiles `../core_algos/src/core_algos.cpp` as an extra
translation unit; `CMakeLists.txt` `add_subdirectory()`s the sibling
`csrc/core_algos` target instead.

```sh
cd csrc/rollout_corr
TORCH_PYTHON=python3 ./build.sh
./build/validate_rollout_corr ./tests/fixtures   # CPU fixture comparisons
./build/cuda_smoke_test                          # CUDA regression test; exit 77/SKIP without a GPU
```

`CMakeLists.txt` mirrors `build.sh`'s flags/file list and `add_subdirectory`s
`../core_algos` (disabling that module's own test targets first, to avoid a
duplicate `cuda_smoke_test` target name). **Not exercised by the author of
this port** — no `cmake` binary was available in this environment, same as
`core_algos`'s own `CMakeLists.txt`; please open an issue/PR if it doesn't
configure cleanly as-is, same ask that file made.

### Actual run output (this environment, arm64 macOS, CPU libtorch 2.8.0, pip wheel)

Full build: zero warnings from this port's own source files (`src/*.cpp`,
`tests/*.cpp`) under `-Wall -Wextra` — verified explicitly (not just the
default `build.sh` flags), since `build.sh` itself doesn't pass
`-Wall -Wextra`:

```sh
$ c++ -std=c++17 -O2 -Wall -Wextra ... -c src/rollout_corr_helper.cpp -o /tmp/out.o 2>&1 \
    | grep 'rollout_corr/\(src\|tests\)'
# (no output, for all four of src/*.cpp and tests/*.cpp)
```

`./build/validate_rollout_corr ./tests/fixtures` (119 individual `[PASS]`
lines, 0 `[FAIL]`, across the fixture files listed in "Validation
methodology" below; abbreviated here, full output reproducible via the
command above):

```
[PASS] is_threshold[2]:string:0.1_1.2  max_abs_diff=0 (tol=0.0001)
[PASS] rs_threshold[2]:token_k1,seq_sum_k2  max_abs_diff=0 (tol=0.0001)
[PASS] rejection_mask:token_k1_icepop (mask)  max_abs_diff=0 (tol=0.0001)
[PASS] rejection_mask:token_k1_icepop (metrics)  max_abs_diff=1.11759e-08 (tol=0.0001)
[PASS] rejection_mask:empty_batch (mask)  max_abs_diff=0 (tol=0.0001)
[PASS] rejection_mask:empty_batch (metrics)  max_abs_diff=0 (tol=0.0001)
[PASS] rs_metrics:sequence_level_no_lower (metrics)  max_abs_diff=5.58794e-09 (tol=0.0001)
[PASS] is_weights:token_tis_batch_normalize (weights)  max_abs_diff=0 (tol=0.0001)
[PASS] is_metrics:sequence_default_lower (metrics)  max_abs_diff=3.57628e-07 (tol=0.0001)
[PASS] offpolicy_metrics:without_rollout (metrics)  max_abs_diff=0 (tol=0.0001)
[PASS] topk_log_probs:multi_chunk (forward)  max_abs_diff=4.76837e-07 (tol=0.0001)
[PASS] topk_log_probs:multi_chunk (grad_logits)  max_abs_diff=5.96046e-08 (tol=0.0001)
[PASS] correction:near_full_head_mass (grad)  max_abs_diff=2.14577e-06 (tol=0.0001)
[PASS] support_correction:incomplete_support (correction)  max_abs_diff=1.65775e-07 (tol=0.0001)
[PASS] auto_correction:mixed_support (correction)  max_abs_diff=2.95229e-07 (tol=0.0001)
[PASS] auto_correction:all_incomplete (grad)  max_abs_diff=0 (tol=0.0001)
...
ALL SCENARIOS PASSED
```

All observed diffs are in the `3.7e-9`-`4.4e-6` range (float32 rounding /
text-round-trip noise, same order of magnitude `core_algos`'s own run log
reports), well inside the `1e-4` tolerance used here (slightly looser than
`core_algos`'s `1e-5` — this module's metrics involve more chained
reductions (ESS, chi^2, PPL ratios) where float32 rounding compounds a bit
more; still two to four orders of magnitude tighter than any observed
diff).

`./build/cuda_smoke_test`:

```
[SKIP] CUDA not available in this environment -- this test must be run on a
CUDA-enabled libtorch + real GPU to actually validate CUDA execution (see file header).
```

(exit code 77, correctly self-reporting skip rather than a false pass — no
GPU was available in the environment this port was developed in.)

## Validation methodology

`tests/gen_fixtures.py` imports the **real, unmodified**
`verl/trainer/ppo/rollout_corr_helper.py` and `score_centering.py` from this
checkout and runs them across labeled scenarios, dumping inputs + Python's
own reference outputs (plus, for the two autograd-involving functions in
`score_centering.py`, reference gradients from a real `.backward()` call) to
`tests/fixtures/*.txt`. `tests/validate_rollout_corr.cpp` reads those back
and cross-checks the C++ port.

**Scenario coverage** (52 fixture files / 119 individual `[PASS]` comparisons
in the run log above):

- **Threshold parsing** (`_parse_rollout_is_threshold`,
  `_parse_rollout_rs_thresholds`): numeric and string specs, TIS
  (upper-only) and IcePop (`lower_upper`) forms, a degenerate
  `lower == upper` band, single-shared-threshold broadcast across multiple
  `rollout_rs` options, and the `k1`-option reciprocal-default-lower case.
- **`compute_rollout_rejection_mask`**: every RS option family
  (`token_k1`/`k2`/`k3`, `seq_sum_*`, `seq_mean_*`, `seq_max_*`), a chained
  multi-option case (`token_k2,token_k3`), IcePop thresholds, extreme
  thresholds (near-zero upper bound → everything rejected; `1e6` → nothing
  rejected), a singleton batch, an **empty batch** (`log_ratio.shape[0] ==
  0`, exercising the Python source's own early-return branch), and a
  zero-variance (`log_ratio == 0` everywhere) case.
- **`compute_rs_metrics`** (standalone): token-level and sequence-level
  calls, with and without a lower threshold (the `apply_lower_threshold`
  branch that changes which dict keys are even present — see below), plus a
  deliberately contrived `seq_valid_mask`-all-`False` call to exercise the
  `seq_values_valid.numel() == 0` branch, which can't arise from
  `compute_rollout_rejection_mask`'s own internal call (it always derives
  `seq_valid_mask` consistently from `response_mask`) but is real coverage
  of `compute_rs_metrics` as a standalone public function.
  - **Dict-key-presence correctness, not just value closeness**: the C++
    harness's `compare_metrics`/`report_metrics` explicitly checks for
    missing/unexpected keys, not just numeric closeness of common keys. This
    specifically catches a subtlety in the Python source: when
    `seq_values_valid.numel() > 0` **and** `apply_lower_threshold` is
    `False`, `{prefix}_seq_fraction_low` is never assigned into the returned
    dict at all (no `else` branch covers that exact combination) — the C++
    port replicates the *absence*, not a present-with-`0.0` value (see
    `rollout_corr_helper.h`'s "Honest note on dict-key presence" and the
    `rs_metrics_sequence_level_no_lower` fixture/check). Every metrics-bearing
    fixture's key set matched exactly (no `[FAIL] ... KEY MISMATCH`/`MISSING`/
    `UNEXPECTED` lines anywhere in the run log).
- **`compute_rollout_correction_weights`/`compute_is_metrics`**: token- and
  sequence-level aggregation, TIS and IcePop thresholds, batch
  normalization (both aggregation levels), an extremely tight threshold
  (`1.001`, heavy clipping), a singleton batch, zero-variance
  (`log_ratio == 0` → ratio `1.0` everywhere, ESS ≈ 1), and both the
  default-reciprocal-lower and explicit-lower `compute_is_metrics` paths.
- **`compute_offpolicy_metrics`**: with and without `rollout_log_prob`
  (`None` → only the two training-perplexity metrics, matching Python's own
  early-return shape), identical policies (KL/χ² ≈ 0), and a singleton
  batch.
- **`dummy_rollout_topk`/`pad_rollout_topk`**: basic case, and
  `prompt_length == 1` (the `start == 0` boundary this port's extra
  `TORCH_CHECK` guards).
- **`topk_log_probs_from_logits`**: a single-chunk case (`chunk_size`
  covers the whole batch in one pass), a multi-chunk case (`chunk_size`
  forces several passes through the forward/backward loop), and a
  larger-vocabulary case — forward value **and** backward gradient (via a
  real `logits.grad` from `.backward(grad_output)` with a random, non-uniform
  `grad_output`, not just a trivial sum-reduction) checked against the C++
  port's `torch::autograd::Function<T>` implementation.
- **`score_centering_weight_fn`**: no IS weighting (identity), TIS, IcePop.
- **`score_centering_correction`/`*_support_correction`/`*_auto_correction`**:
  normal case, each weight-fn variant (identity/TIS/IcePop), a
  near-full-head-mass case (the docstring's "ill-conditioned `rho`, but the
  gradient is still meaningful" scenario), and — per the task's explicit ask
  to cover both branches the module's own docstring calls out — both
  **"head covers full support"** (`support_correction_full_support`: no
  `-inf` entries in the sampler head at all) and **"incomplete head,
  predefined approximation"** (`support_correction_incomplete_support[_icepop]`:
  the lowest-probability head slot marked `-inf`, i.e. out of the recorded
  support) for `score_centering_support_correction`, plus
  `score_centering_auto_correction` scenarios that mix both branches within
  one batch via `support_complete` (`auto_correction_mixed_support`) and
  exercise each branch exclusively (`all_complete`/`all_incomplete`). Every
  one of these checks a real backward-pass gradient, not just forward
  values.

**Honesty disclosure: Python environment.** This checkout's system Python is
3.9, but the repo requires `>=3.10` (`verl/utils/config.py` uses
`DictConfig | dict` union syntax, a hard `TypeError` at runtime on 3.9 — this
is also why `csrc/core_algos`'s own `gen_fixtures.py` cannot actually be run
as-is in this same environment with the system Python; it was presumably run
against a different, no-longer-available Python 3.10+ environment by that
port's author). Fixtures for this PR were generated with a fresh Python 3.11
venv: `torch==2.15.0` (CPU wheel, via `--index-url
https://download.pytorch.org/whl/cpu`), `numpy==2.4.6`, `omegaconf==2.4.0`,
`tensordict==0.15.5`, `transformers==5.22.2`, `ray==2.60.0`,
`pillow==13.0.0` — installed only to make `import verl` succeed end-to-end
(the module import chain pulls in `verl.protocol`/`verl.utils.dataset`/etc.
even though none of their actual APIs are touched by the functions being
fixture-tested). None of these versions match `requirements.txt`'s exact
pins, and that's fine for this specific purpose: the goal is running the
real `rollout_corr_helper.py`/`score_centering.py` source as unmodified
tensor-math functions, not reproducing a full production training
environment. The C++ build itself (`build.sh`) still uses the repo's
originally-available `torch==2.8.0` CPU wheel for arm64 macOS (Python 3.9),
same as `core_algos`'s build — the fixture-generation Python version and the
C++-build libtorch version are independent of each other by design (the
fixtures are plain text files with no Python/torch-version-specific
encoding).

## What was, and wasn't, verified

- **Verified for real (author, CPU, arm64 macOS)**: the port compiles
  cleanly (zero warnings from this port's own files under `-Wall -Wextra`)
  against a real libtorch (pip `torch==2.8.0`, CPU) via direct `clang++`
  invocation, links, runs, and numerically matches the real (unmodified)
  Python source across 52 fixture files / 119 individual `[PASS]` comparisons
  (tensor values, scalar metrics, dict key-sets, and real `.backward()`
  gradients), all within `1e-4` (observed: `3.7e-9`-`4.4e-6`).
- **Not verified (by the author of this port)**: CUDA execution of any
  kind — no GPU was available in the environment this port was developed
  in. `tests/cuda_smoke_test.cpp` was written and compile/link-checked
  against the same CPU-only libtorch (correctly self-reports `[SKIP]`/exit
  77 there, per `torch::cuda::is_available()`), but has not been run on real
  CUDA hardware by the author. Per this project's established process (see
  `core_algos.cpp`'s own PR, where this is exactly how the
  `Tensor::accessor` bug was actually found), **the coordinator should run
  `./build/cuda_smoke_test` on real GPU hardware (e.g. IP66) before
  merging** — this audit found no CPU-only-accessor bugs by inspection, but
  inspection is not a substitute for actually running it on a GPU, which is
  exactly the lesson the `core_algos` PR's own post-mortem drew.
- **`CMakeLists.txt` build path**: not exercised (no `cmake` binary in this
  environment), same as `core_algos`'s own `CMakeLists.txt` at the time it
  was written (that file has since been built and GPU-verified by a
  reviewer, per `core_algos/README.md`) — the coordinator's independent
  rebuild is the natural place to also exercise this path for the first
  time.
- Integration into the actual Python call sites
  (`compute_rollout_correction_and_rejection_mask`,
  `compute_rollout_correction_and_add_to_batch`, `apply_bypass_mode`,
  `score_centering_logits_processor`/`score_centering_ppo_loss`) is out of
  scope for this PR, same as `core_algos.cpp`'s own scope boundary — those
  remain Python-side glue that would call into this new C++ library once
  such integration work is undertaken.
