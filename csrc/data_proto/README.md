# `csrc/data_proto` — C++/libtorch port of the tensor-batch half of `verl/protocol.py`'s `DataProto`

Third concrete migration step of the Python-elimination plan in `PLAN.md`
(section 6, item 3 / table item #7): a libtorch port of `DataProto.batch`
(a `TensorDict`-backed dict of named tensors sharing a leading batch-size
dimension) from `verl/protocol.py` (1342 lines, class at L318). Follows
`csrc/core_algos` and `csrc/rollout_corr`'s established conventions
throughout: same file layout (`include/`, `src/`, `tests/`, `build.sh` +
`CMakeLists.txt`), same fixture-generation-from-real-Python-then-compare-
in-C++ test harness style, same CPU/CUDA dual-build path, same
honesty-disclosure norms.

Per `PLAN.md` section 6, item 3: this is "just enough of a tensor-batch
struct to let [`core_algos`/`rollout_corr`] run standalone against real
GSM8K data, without yet solving the dynamic `non_tensor_batch`/`meta_info`
problem in full generality" — i.e. a foundational data-structure port, not
a math port, and (per the task's own framing) one that does not need to
`add_subdirectory()` either sibling module.

## What's ported, and the C++ mapping

`verl_data_proto::TensorBatch` (`include/verl_data_proto/tensor_batch.h` +
`src/tensor_batch.cpp`) stands in for `DataProto.batch`: a
`std::unordered_map<std::string, torch::Tensor>` plus an explicit
`int64_t batch_size`, the C++ analog of a `TensorDict`'s own
`(keys dict, batch_size tuple)` pair restricted to `num_batch_dims == 1`
(the only case `DataProto.check_consistency` itself supports — "only
support num_batch_dims=1" appears twice in its assertions).

| C++ (`verl_data_proto::TensorBatch::`) | Python source (`DataProto.`) |
|---|---|
| `check_consistency` | `check_consistency` |
| `from_dict` (two overloads) | `from_dict` (tensor-only slice of it — see "Scope" below) |
| `select` | `select` (tensor-only slice; see "Scope") |
| `select_idxs` | `select_idxs` |
| `slice` | `slice` |
| `pop` | `pop` (tensor-only slice; see "Scope") |
| `rename` | `rename` (+ its `validate_input` helper) |
| `union_with` | `union` (tensor-only slice; see "Scope") |
| `concat` (static) | `concat` (static; tensor-only slice) |
| `chunk` | `chunk` |
| `split` | `split` |
| `reorder` | `reorder` |
| `repeat` | `repeat` |
| `sample_level_repeat` | `sample_level_repeat` |
| `unfold_column_chunks` | `unfold_column_chunks` |
| `to` | `to` |

**Mutate-self vs. return-new-object split is preserved exactly**, not
normalized to one C++ convention, because it's part of the semantics being
ported: `to`, `rename`, `union_with` mutate `*this` and return `*this` by
reference (matching `DataProto.to`/`.rename`/`.union`'s own `return self`);
`reorder` mutates `*this` with no return value (matching the Python source,
which has no `return` statement either — its own docstring says "Note that
this operation is in-place"); `pop` mutates `*this` (removes keys) *and*
returns a new `TensorBatch` of the popped entries (matching `DataProto.pop`
exactly); everything else (`select`, `select_idxs`, `slice`, `concat`,
`chunk`, `split`, `repeat`, `sample_level_repeat`, `unfold_column_chunks`)
returns a new `TensorBatch` without touching `*this`. See
`tensor_batch.h`'s file header and each method's own doc comment for the
one-line version of this same point, attached to the specific method.

## Scope: what's deliberately NOT ported, and why

Per the task's own scope instructions and `PLAN.md`'s framing of this item:

- **`non_tensor_batch`** (`dict[str, np.ndarray]` of numpy object arrays
  holding arbitrary Python objects — e.g. multi-turn chat message dicts,
  reward metadata strings): genuinely awkward to express in C++ without a
  variant/any type, exactly as `PLAN.md` says. Every ported method above
  that touches both `self.batch` *and* `self.non_tensor_batch` in the
  Python source (`select`, `select_idxs`, `slice`, `pop`, `union`, `concat`,
  `chunk`, `repeat`, `sample_level_repeat`, `unfold_column_chunks`) is
  ported here as its **tensor-only slice**: the exact same indexing/
  reshaping/combining logic, applied only to the `TensorBatch`'s own
  tensors. The non-tensor half of each of those methods is a parallel,
  independent block of numpy-array logic in the Python source (not
  entangled with the tensor half's correctness) — dropping it does not
  change what the tensor half computes, only what else the Python method
  additionally does.
- **`meta_info`** (a plain Python dict): same story. `concat`'s Python
  source does real, somewhat involved meta_info-merging logic (special-
  cased "metrics" list-aggregation across workers) — not ported, since it's
  meta_info logic, not tensor logic. `DataProtoConfig`/`auto_padding`/
  `is_padding_enabled` (an auto-padding policy flag stored *in* meta_info)
  are also not modeled; see `chunk`'s `require_equal_chunks` parameter
  below for how this port still exercises the real uneven-split code path
  that flag unlocks in Python, without modeling the flag itself.
- **Ray-serialization machinery**: `__getstate__`/`__setstate__`,
  `DataProtoFuture`, `all_gather_data_proto`. Entirely about Ray object
  transport, not tensor semantics.
- **`save_to_disk`/`load_from_disk`**: `pickle` file I/O, not tensor math.
- **`make_iterator`**: wraps a PyTorch `DataLoader` with Hydra
  (`dataloader_kwargs`) config — orchestration, not tensor math.
- **`auto_padding`/`padding`/`is_padding_enabled`**: tokenizer-padding-policy
  logic gated on a `meta_info` flag (see above).
- **`print_size`/`get_data_info`/`_get_type_info`**: diagnostics/logging,
  `print()`-based.
- **`from_single_dict`/`from_tensordict`/`to_tensordict`**: conversion
  from/to the Python-side `TensorDict`/numpy-array types themselves — the
  actual boundary-crossing constructor this struct needs instead is
  `TensorBatch::from_dict(std::unordered_map<std::string, torch::Tensor>)`,
  provided (two overloads: one infers `batch_size` from the first tensor,
  matching `from_dict`'s own "pivot key" batch-size-consistency check; one
  takes an explicit `batch_size` for the empty-map case, where there's no
  tensor to infer it from).
- **`__getitem__`/`__len__`**: Python dunder dispatch, not listed in the
  task's own scope table. `select_idxs`/`slice` (what `__getitem__`
  dispatches to) are ported directly; `batch_size()` is the natural C++
  equivalent of `__len__` and is provided as a plain accessor.
- **`DataProtoItem`/`BatchData`/`collate_fn`**: `DataProtoItem` is a
  single-row, non-tensor-batch-aware dataclass returned by `__getitem__`'s
  integer-index case (not in scope, same reasoning as `__getitem__` itself);
  `BatchData`/`collate_fn` exist to support `make_iterator`'s `DataLoader`
  integration (not in scope, see above).

## Judgment calls and deviations from literal Python behavior (documented, not silently fixed)

Per the task's explicit ask to flag deviations honestly:

**`rename` with 2+ keys is actually broken in the real Python source today
— confirmed by running it, not just reading it.** `DataProto.rename`
(`verl/protocol.py:751`) does:

```python
self.batch.rename_key_(tuple(old_keys), tuple(new_keys))
```

`TensorDict.rename_key_(old_key: NestedKey, new_key: NestedKey, ...)` takes
a **single** key (`str` or a tuple representing one *nested* key path, e.g.
`("a", "b")` meaning "the `"b"` entry inside the sub-tensordict at `"a"`") —
not a batch of independent `(old, new)` pairs. Passing `tuple(["obs",
"mask"])` therefore gets interpreted as "look for a nested entry `"mask"`
inside the top-level entry `"obs"`", which raises
`tensordict.base._KeyThroughLeafError: 'obs' is a Tensor, not a tensordict,
so it has no entry 'mask'` the moment `"obs"` is a plain tensor (which it
always is here — `DataProto.batch` is a flat, non-nested `TensorDict` by
construction everywhere in this codebase). Verified directly by actually running
`DataProto.rename(old_keys=["obs", "mask"], new_keys=[...])` against the
real `tensordict==0.15.5` dependency in this checkout and observing the
exception above (not inferred from reading the source alone). The
**single-key** case
(`old_keys=["x"]`, a 1-tuple) works fine (`unravel_key(("x",))` resolves
back to the plain string `"x"`) — it's specifically 2+ keys in one call
that's broken. Grepping the repo (`grep -rn '\.rename(' --include='*.py'`)
turns up **zero** real call sites of `DataProto.rename` anywhere in this
codebase outside this port's own test file, so this bug has apparently
never been exercised in practice.

This port's `TensorBatch::rename(old_keys, new_keys)` implements the
clearly-*intended* semantics instead of replicating the bug: independent
per-pair renaming via a plain erase+insert loop (see `tensor_batch.cpp`),
which is exactly what a real caller wanting to rename 2+ keys would need
and the single-key path already proves is the intended behavior. The
`rename_basic` fixture therefore validates against real Python by calling
`rename` **twice**, once per key (the only way to actually achieve a
two-key rename against the real Python source today) and comparing the
combined final state — see `tests/gen_fixtures.py`'s `rename_basic` comment
for the full trace of this finding.

**`slice`'s negative `step` is rejected, not independently supported —
also confirmed by running it, not assumed.** `DataProto.slice(None, None,
-1)` against the real `tensordict==0.15.5` dependency in this checkout
raises `"step must be greater than zero"` — `TensorDict`'s own indexing
doesn't support reversal via a negative step at all. `TensorBatch::slice`
matches this with a `TORCH_CHECK` rather than implementing Python-style
negative-step reversal logic for behavior the real dependency doesn't
actually have.

**`chunk`'s `require_equal_chunks` parameter stands in for
`is_padding_enabled()`, not `auto_padding` itself.** The Python source:

```python
if not self.is_padding_enabled():
    assert len(self) % chunks == 0, ...
```

`is_padding_enabled()` reads a flag out of `meta_info` (set via
`DataProto.from_dict(..., auto_padding=True)`) — `meta_info` itself is out
of scope here (see "Scope" above), so there is no flag to read. But the
underlying code path it gates — `torch.chunk`'s own uneven-split behavior
(ceil-division sizes, last chunk smaller) — **is** real, pure tensor logic,
and a real code path production `auto_padding=True` callers rely on (per
the task's own validation-bar ask: "chunk/split with sizes that don't
divide evenly if that's a real code path"). `require_equal_chunks` (default
`true`, matching the *un*-padded default) is this port's way of exposing
that real branch without modeling the `meta_info` flag that gates it in
Python. The `chunk_uneven` fixture constructs its Python reference with
`auto_padding=True` specifically to reach this branch for real (confirmed:
`chunk(3)` on a 10-row batch produces `[4, 4, 2]`-sized pieces, matching
`torch.Tensor.chunk`'s own algorithm, which this port's C++ `chunk` also
uses directly via a dummy `torch::arange` index tensor — see
`tensor_batch.h`'s doc comment on `chunk`).

**`concat` requires an exact matching key set across all batches** (a
`TORCH_CHECK`, not silently padding missing keys with anything). This
matches `tensordict`'s own `torch.cat`-on-`TensorDict` requirement (it
raises on mismatched keys); "differently-shaped-but-compatible batches" in
the validation bar is read here as "same keys, but the batches may
otherwise differ in dim0 extent" (the `concat_differently_shaped` fixture:
three batches of sizes 3/5/2 with the same two keys `x`/`y`, same shapes
beyond dim0).

**`check_consistency` is validated with direct C++ assertions, not a Python
fixture.** It's a validator (asserts invariants, no numeric output to
compare), so there's no "reference output" beyond the pass/fail boundary
already encoded in its own conditions — `tests/validate_data_proto.cpp`'s
`check_consistency_invariant()` constructs both a consistent and an
inconsistent `TensorBatch` directly in C++ and asserts the expected
throw/no-throw behavior, which is the C++-native equivalent of what a
fixture-based check would otherwise be verifying.

**`union`'s conflicting-value-throws case is also a direct C++ assertion,
not a Python fixture**, for the same reason: the point of that branch is
exception behavior (Python's `assert tensor_dict1[key].equal(tensor_dict2[key])`
raising `AssertionError`), not a numeric value to diff against a tolerance.
`check_union`'s third block in `validate_data_proto.cpp` constructs two
`TensorBatch`es with a conflicting shared key directly and asserts
`union_with` throws (`c10::Error`, this port's `TORCH_CHECK` failure mode).

## CUDA-safety audit

Per the task's explicit ask (the `Tensor::accessor<T,N>()` CPU-only-API bug
class, originally found and fixed in `csrc/core_algos`'s GRPO-advantage
port — see that module's README's "CUDA fix (post-review)" section):

```sh
$ grep -n '\.accessor<\|\.data_ptr<' csrc/data_proto/src/*.cpp
# (no output)
```

**Zero hits.** Every raw-value extraction in `tensor_batch.cpp` is one of:

- A standard device-dispatched ATen op (`index_select`, `index` with a
  bool-mask `TensorIndex`, `narrow`, `reshape`, `cat`, `chunk`,
  `repeat_interleave`, `unsqueeze`/`expand`, `to`, etc.) — these dispatch
  correctly on CUDA tensors by construction.
- `.item<int64_t>()` on a 0-dim reduction tensor (`select_idxs`'s bool-mask
  branch: `idxs.sum().item<int64_t>()` to compute the output `batch_size`)
  — explicitly safe on any device (performs the device-to-host copy
  itself), the same safe pattern `core_algos.cpp`'s post-fix code and
  `rollout_corr`'s own audit rely on.
- `torch::equal(a, b)` (`union_with`'s conflicting-key check) — a
  dispatched op that performs the comparison (including any
  device-to-host synchronization) internally, not a raw accessor/pointer
  read.
- `tensor.size(0)`/`tensor.dim()` (`check_consistency`, both `from_dict`
  overloads) — tensor **metadata** queries (shape/rank), never touching
  device memory at all, safe on any device unconditionally.

`tests/cuda_smoke_test.cpp` exercises `to(device)` plus a representative
subset of every other ported operation (`select`, `select_idxs` — both the
int-index and bool-mask branches, specifically because the bool-mask branch
is the one `.item<int64_t>()` call in this port — `slice`, `pop`, `rename`,
`union_with`, `concat`, `chunk`, `reorder`, `repeat` — both `interleave`
branches, `sample_level_repeat`, `unfold_column_chunks`) against real CUDA
tensors, cross-checked against an identical-input CPU run.

## Build + run

Same no-cmake sandbox constraint as `csrc/core_algos`/`csrc/rollout_corr`
(see those modules' READMEs): the build actually exercised here is
`build.sh` (direct `clang++` invocation against a pip-installed libtorch).
This module has no dependency on either sibling `csrc/` module (per the
task's own expectation — "you likely won't need to depend on anything,
since this task is more foundational"), so there's no `add_subdirectory()`
wiring to mirror from `rollout_corr/CMakeLists.txt`'s sibling-dependency
pattern.

```sh
cd csrc/data_proto
TORCH_PYTHON=python3 ./build.sh
./build/validate_data_proto ./tests/fixtures   # CPU fixture comparisons
./build/cuda_smoke_test                        # CUDA regression test; exit 77/SKIP without a GPU
```

`CMakeLists.txt` mirrors `build.sh`'s flags/file list. **Not exercised by
the author of this port** — no `cmake` binary was available in this
environment (confirmed: `which cmake` → not found), same as both prior
modules at the time their own `CMakeLists.txt` files were first written;
please open an issue/PR if it doesn't configure cleanly as-is, same ask
those files made (and, per `core_algos/README.md`, that file's own
`CMakeLists.txt` has since been built and GPU-verified by a reviewer, so
this is a real, previously-successful path for this project, not a
hypothetical one).

To regenerate fixtures from a fresh Python environment (needs the real
`verl` package importable):

```sh
python3 csrc/data_proto/tests/gen_fixtures.py
```

### Actual run output (this environment, arm64 macOS, CPU libtorch 2.8.0, pip wheel)

Zero warnings from this port's own source files (`src/tensor_batch.cpp`,
`tests/validate_data_proto.cpp`, `tests/cuda_smoke_test.cpp`) under
`-Wall -Wextra` — verified explicitly (not just the default `build.sh`
flags, which don't pass `-Wall -Wextra`):

```sh
$ c++ -std=c++17 -O2 -Wall -Wextra ... -c src/tensor_batch.cpp -o /tmp/out.o 2>&1 \
    | grep 'data_proto/\(src\|tests\)'
# (no output, for all three files)
```

`./build/validate_data_proto ./tests/fixtures` (111 individual `[PASS]`
lines across 25 fixture files + 3 direct-C++-assertion checks
(`check_consistency` x2, `union` conflicting-value-throws x1), 0 `[FAIL]`):

```
[PASS] check_consistency_rejects_mismatched_batch_dims
[PASS] check_consistency_accepts_matching_batch_dims
[PASS] select_subset (batch_size)  got=6 want=6
[PASS] select_subset [mask]  max_abs_diff=0 (tol=1e-05)
[PASS] select_idxs_empty (batch_size)  got=0 want=0
[PASS] slice_negative_start [obs]  max_abs_diff=0 (tol=1e-05)
[PASS] pop_subset (popped) [mask]  max_abs_diff=0 (tol=1e-05)
[PASS] rename_basic [attn_mask]  max_abs_diff=0 (tol=1e-05)
[PASS] union_conflicting_values_throws
[PASS] concat_differently_shaped (batch_size)  got=10 want=10
[PASS] chunk_uneven[2] (batch_size)  got=2 want=2
[PASS] split_uneven[3] (batch_size)  got=1 want=1
[PASS] sample_level_repeat_with_zero (batch_size)  got=7 want=7
[PASS] unfold_column_chunks_basic [grouped]  max_abs_diff=0 (tol=1e-05)
...
ALL SCENARIOS PASSED
```

Every observed numeric diff was exactly `0` (not just within tolerance) —
unlike `core_algos`/`rollout_corr`, which have real float32-arithmetic
computations (means, variances, clipping, log-sum-exp, etc.) and therefore
observe `~1e-7`-`1e-6` text-round-trip noise, every operation ported here is
pure **indexing/reshaping/concatenation** (`index_select`, `narrow`, `cat`,
`reshape`, `repeat_interleave`, ...) with no arithmetic on tensor values at
all, so bit-identical float32 values are the expected (and observed) result,
not a coincidence.

`./build/cuda_smoke_test`:

```
[SKIP] CUDA not available in this environment -- this test must be run on a
CUDA-enabled libtorch + real GPU to actually validate CUDA execution (see file header).
```

(exit code 77, correctly self-reporting skip rather than a false pass — no
GPU was available in the environment this port was developed in.)

## Validation methodology

`tests/gen_fixtures.py` imports the **real, unmodified** `verl.protocol`
(specifically `DataProto`) from this checkout and runs it across labeled
scenarios, dumping each operation's input batch (or batches), any
non-tensor parameters (index lists, key lists, split sizes), and Python's
own reference output batch (or batches) to `tests/fixtures/*.txt`.
`tests/validate_data_proto.cpp` reads those back and cross-checks the C++
port — both the resulting tensor *values* and the resulting *key set*
(catching a port that silently drops/adds/misnames a key, not just one that
gets values wrong) and `batch_size()`.

**Scenario coverage** (25 fixture files, mapping directly to the task's own
validation-bar checklist):

- **Normal multi-key batches**: `select_subset`/`select_idxs_*`/`slice_*`/
  `pop_subset`/`rename_basic`/`chunk_even`/`reorder_with_repeats`/
  `repeat_*`/`sample_level_repeat_with_zero`/`unfold_column_chunks_basic`/
  `to_cpu_noop` all operate on a 3-key, 6-row batch (`obs`/`logprob`/`mask`,
  mixed shapes) built via `make_base()`.
- **Single-key batches**: `select_single_key`, `rename_single_string`,
  `concat_single_key` all operate on a 1-key batch built via
  `make_single_key()`.
- **Edge cases**:
  - **Empty selection**: `select_idxs_empty` (an empty `int64` index
    tensor → a real, correctly-shaped 0-row `TensorBatch`, not a crash or a
    missing-key result).
  - **Single-row batch**: `slice_single_row` (`slice(2, 3)` on a 6-row
    batch → exactly 1 row).
  - **Chunk/split with sizes that don't divide evenly**: `chunk_uneven`
    (10 rows into 3 chunks, `auto_padding=True` on the Python side to reach
    the real uneven-split code path — see "Judgment calls" above) and
    `split_uneven` (10 rows, `split_size=3` → sizes `[3, 3, 3, 1]`).
  - **`union` with disjoint vs. overlapping keys**: `union_disjoint` (two
    batches with entirely separate key sets) and `union_overlap_equal` (one
    shared key with identical values, plus one key unique to each side) —
    both are real success-path Python fixtures; the conflicting-overlap
    failure path is a direct C++ assertion (see "Judgment calls" above).
  - **`concat` of differently-shaped-but-compatible batches**:
    `concat_differently_shaped` (three batches of sizes 3/5/2, two shared
    keys `x`/`y` with matching shapes beyond dim0).
- **Every other ported operation gets at least one dedicated scenario**:
  `pop_subset` (remaining batch *and* popped batch both checked),
  `rename_basic` (two independent single-key renames combined — see
  "Judgment calls"), `reorder_with_repeats` (a permutation that also
  repeats an index, `[2, 0, 0, 4, 1, 3]`), `repeat_interleave`/`repeat_tile`
  (both branches of the `interleave` flag), `sample_level_repeat_with_zero`
  (per-row counts including an actual `0`, matching a real dropped-row
  code path), `unfold_column_chunks_basic` (one split key reshaped, one
  non-split key repeat-interleaved, in the same call), `to_cpu_noop`
  (device-transfer value-preservation on CPU; real device-transfer coverage
  is `cuda_smoke_test.cpp`'s job).

### Honesty disclosure: Python environment

Same situation `csrc/rollout_corr/README.md` already documented: this
checkout's system Python is 3.9, but the repo requires `>=3.10`. Fixtures
for this PR were generated with the **same** fresh Python 3.11 venv
recipe `rollout_corr` used: `torch==2.15.0` (CPU wheel, via `--index-url
https://download.pytorch.org/whl/cpu`), `numpy==2.4.6`, `omegaconf==2.4.0`,
`tensordict==0.15.5`, `transformers==5.22.2`, `ray==2.60.0`,
`pillow==13.0.0` — installed to make `import verl.protocol` succeed
end-to-end (its own import chain pulls in `verl.utils.transferqueue_utils`,
`verl.utils.device`, etc., even though none of their APIs are touched by
`DataProto`'s tensor-batch methods under test). The C++ build itself
(`build.sh`) still uses the repo's originally-available `torch==2.8.0` CPU
wheel for arm64 macOS (Python 3.9) — the fixture-generation Python version
and the C++-build libtorch version are independent of each other by design
(the fixtures are plain text files with no Python/torch-version-specific
encoding).

## What was, and wasn't, verified

- **Verified for real (author, CPU, arm64 macOS)**: the port compiles
  cleanly (zero warnings from this port's own files under `-Wall -Wextra`)
  against a real libtorch (pip `torch==2.8.0`, CPU) via direct `clang++`
  invocation, links, runs, and exactly matches the real (unmodified) Python
  `DataProto` across 25 fixture files (111 individual `[PASS]` comparisons:
  tensor values — all exactly `0` max-abs-diff, batch sizes, and key sets)
  plus 3 direct-C++-assertion checks for exception-only behavior
  (`check_consistency` x2, conflicting-value `union` x1).
- **Not verified (by the author of this port)**: CUDA execution of any
  kind — no GPU was available in the environment this port was developed
  in (confirmed: `torch.cuda.is_available()` → `False`).
  `tests/cuda_smoke_test.cpp` was written and compile/link-checked against
  the same CPU-only libtorch (correctly self-reports `[SKIP]`/exit 77
  there), but has not been run on real CUDA hardware by the author. Per
  this project's established process (see `core_algos.cpp`'s own PR, where
  running on real GPU hardware is exactly how the `Tensor::accessor` bug
  was actually found, not just inspected away), **the coordinator should
  run `./build/cuda_smoke_test` on real GPU hardware (e.g. IP66) before
  merging** — the audit above found no CPU-only-accessor-style bugs by
  inspection, but inspection is not a substitute for actually running it on
  a GPU.
- **`CMakeLists.txt` build path**: not exercised (no `cmake` binary in this
  environment), same as both sibling modules' own `CMakeLists.txt` files at
  the time they were first written — the coordinator's independent rebuild
  is the natural place to also exercise this path for the first time, same
  precedent `core_algos/README.md` and `rollout_corr/README.md` already
  record.
- Integration into the actual Python/Ray call sites that construct/consume
  real `DataProto` objects today (the rest of `verl/protocol.py`'s own
  `non_tensor_batch`/`meta_info`/Ray-serialization machinery, and every
  other subsystem in `verl/` that passes `DataProto` objects around) is out
  of scope for this PR, same as both sibling ports' own scope boundary —
  `PLAN.md`'s own sequencing (section 6) treats that kind of integration
  work as a later step once the self-contained pieces (`core_algos`,
  `rollout_corr`, and now this tensor-batch container) exist and are
  validated standalone.
