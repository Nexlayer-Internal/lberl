# `csrc/continuous_token` — C++/Rust port of verl's online prompt tokenization

`PLAN.md` table row #9b / section 6 item 7: the one Python-elimination
candidate that is "a bounded library-binding problem" rather than tensor
math, and the first C++ port in this repo with no libtorch/CUDA dependency
at all. Binds the real [HF `tokenizers`](https://github.com/huggingface/tokenizers)
Rust crate (BPE encode) plus [`minijinja`](https://github.com/mitsuhiko/minijinja)
(Jinja2-compatible chat-template rendering) behind a thin Rust FFI shim and
an ergonomic C++ wrapper, for `verl/utils/tokenizer/continuous_token.py`'s
`ContinuousTokenBuilder.build_initial_tokens` -- and only that method, for
the validated Qwen3-4B / GSM8K single-turn, text-only workload. This is
**not** a CUDA/tensor port: correctness here means exact token-ID sequence
match, a hard-failure comparison, not a float tolerance one (per the task's
own framing) -- so there is no CUDA smoke test in this subdirectory, unlike
the other three ports.

## Scoping trace (read the actual call path, not assumed)

`examples/grpo_trainer/run_qwen3_4b_fsdp.sh` runs GSM8K through
`SingleTurnAgentLoop` (single-turn, text-only, no tools). Reading
`verl/experimental/agent_loop/single_turn_agent_loop.py`'s `run()` line by
line, it calls exactly two Continuous Token entry points, in this order:

1. `self.ct_build_initial_tokens(messages, images=images, videos=videos, audios=audios, mm_inputs_out=mm_inputs)`
   -- note: **no `tools` kwarg is passed**, so `tools` is always `None` on
   this path.
2. `self.ct_merge_assistant_token(prompt_ids, output.token_ids, [], [] if output.log_probs else None, assistant_logprobs=...)`
   -- note: the `response_mask` argument is a literal `[]` at this call
   site, always.

It never calls `ct_merge_context_msg` (`verl/experimental/agent_loop/agent_loop.py`'s
other CT entry point, which wraps `merge_context_tokens` +
`align_response_metadata`). Tracing those two methods down into
`verl/experimental/agent_loop/agent_loop.py`:

- `ct_build_initial_tokens` -> `self.continuous_token_builder.build_initial_tokens(...)`
- `ct_merge_assistant_token` -> `self.continuous_token_builder.merge_assistant_tokens(...)`
  followed by `self.continuous_token_builder.align_response_metadata(...)`

`verl/utils/tokenizer/continuous_token_wiring.py` maps HF `model_type ==
"qwen3"`/`"qwen3_moe"` to `QwenContinuousTokenBuilder`. Reading that class
(`verl/utils/tokenizer/continuous_token.py:540-578`) in full: it overrides
exactly two things -- `__init__` (computes `_newline_id`/`_im_end_id`) and
`_merge_context_token_ids` (used only by `merge_context_tokens`, which
`SingleTurnAgentLoop` never calls). **It does not override
`build_initial_tokens`, `merge_assistant_tokens`, or
`align_response_metadata`** -- those are inherited unchanged from the base
`ContinuousTokenBuilder`.

**Conclusion, narrower than the task's own framing suggested**: for this
exact validated workload, none of `QwenContinuousTokenBuilder`'s Qwen-specific
boundary-handling logic is reachable at all. The base class's behavior for
these three methods *is* the Qwen behavior here. `gen_fixtures.py` asserts
this explicitly at fixture-generation time (`type(builder).build_initial_tokens
is ContinuousTokenBuilder.build_initial_tokens`, etc.) so a future verl change
that makes this false fails loudly instead of silently invalidating this
port's scope claim.

Of those three methods, reading their bodies shows only one touches a
tokenizer at all:

- **`build_initial_tokens`** -> `_render_tokens` -> `verl.utils.tokenizer.chat_template.apply_chat_template`
  -> `tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=..., tools=None, **chat_template_kwargs)`.
  This is the *only* place real tokenizer/chat-template state is touched.
- **`merge_assistant_tokens`** is `list(runtime_token_ids) + list(assistant_token_ids)`
  wrapped in a `MergeResult` -- pure list concatenation, no tokenizer.
- **`align_response_metadata`** is pure integer/list bookkeeping on masks/logprobs
  (trim/pad by recorded counts) -- no tokenizer.

So **only `build_initial_tokens` needed the Rust shim.**
`merge_assistant_tokens`/`align_response_metadata` are ported directly to
C++ in `src/continuous_token.cpp` and never call into Rust at all -- see
that file's comments for the line-for-line correspondence to the Python
source.

What's deliberately **out of scope** (per the task instructions, and
confirmed unreachable above): `GptOssContinuousTokenBuilder`,
`MiniMaxContinuousTokenBuilder`, `GLMContinuousTokenBuilder`,
`Gemma4ContinuousTokenBuilder`, `DeepSeekContinuousTokenBuilder`,
`VLContinuousTokenMixin` and all VL/multimodal subclasses, tool-calling
(`tool_calls`/`reasoning_content` message fields, `_tokenize_tool_group`,
`merge_context_tokens`/`ct_merge_context_msg`), and the `tools` chat-template
branch (`{% if tools %}` / `tojson`) -- `SingleTurnAgentLoop` never reaches
any of it on the validated GSM8K path.

## The real open question: does the Rust `tokenizers` crate render chat templates?

**No, not at this repo's pinned version, confirmed empirically, not
assumed.** `requirements.txt` pins `transformers>=5.5.3,!=5.6.0,<5.13`;
`uv.lock` resolves that to `transformers==5.12.1` -> `tokenizers==0.22.2`.
Downloaded the real `tokenizers-0.22.2.tar.gz` sdist from PyPI and grepped
the *entire* source tree (`grep -ril "minijinja\|chat_template\|jinja" .`):
**zero hits.** `tokenizers/Cargo.toml`'s dependency list has no Jinja/template
crate of any kind. The core Rust crate this project binds to, at this
version, has no chat-template capability whatsoever -- it is purely a
tokenization (BPE/WordPiece/Unigram + pre-tokenization) library. (This
matches reality on the Python side too: `verl/utils/tokenizer/chat_template.py`
calls `tokenizer.apply_chat_template`, which is a `transformers`-side method
that uses real Python `jinja2`, not anything from the `tokenizers` crate --
confirmed by reading `transformers.utils.chat_template_utils._cached_compile_jinja_template`
in the pinned `transformers==5.12.1`: `ImmutableSandboxedEnvironment(trim_blocks=True,
lstrip_blocks=True, extensions=[AssistantTracker, jinja2.ext.loopcontrols])`,
plain Python `jinja2`.)

So the task's anticipated gap is real, not hypothetical. The task's
instruction was explicit: don't paper over it with a hand-rolled renderer
*unless proven byte-identical* against the real Python path. Rather than
writing a custom Jinja-subset interpreter (hand-rolled, exactly what was
to be avoided), this port uses [`minijinja`](https://crates.io/crates/minijinja)
+ [`minijinja-contrib`](https://crates.io/crates/minijinja-contrib)
(`pycompat` feature) -- a real, independently-maintained, actively-developed
Jinja2-compatible template engine used by other Rust LLM-serving projects
for exactly this purpose (HF chat-template rendering), not written for this
task. It is configured to match `transformers`' exact Jinja2 environment
settings (`trim_blocks=True`, `lstrip_blocks=True`) and its
`unknown_method_callback` is wired to `minijinja_contrib::pycompat::unknown_method_callback`,
which is what makes the Python string methods the real Qwen3 template uses
throughout (`.startswith()`, `.endswith()`, `.split()`, `.strip()`,
`.lstrip()`, `.rstrip()`) resolve at all.

### Empirical parity check (before any shim code was written)

Downloaded the real `Qwen/Qwen3-4B` tokenizer files
(`tokenizer.json`, `tokenizer_config.json`) from the HF Hub (not the full
~8GB of model weights -- just the tokenizer), extracted the real
`chat_template` Jinja2 source (4168 bytes -- the well-known Qwen3 template:
`<think>` reasoning-content handling, multi-step tool-call bookkeeping via
`namespace()`, `messages[::-1]` reversed iteration, `tojson`, etc.), and
rendered it two ways:

1. **Python**, via `transformers==5.12.1`'s real `apply_chat_template`
   (the actual production code path).
2. **Rust**, via a standalone `minijinja` + `minijinja-contrib(pycompat)`
   test harness, with the same `trim_blocks`/`lstrip_blocks` settings.

Across 7 representative scenarios -- a real GSM8K question (matching
`examples/data_preprocess/gsm8k.py`'s exact prompt format), a longer/noisy
GSM8K-shaped question (embedded quotes, apostrophes, a literal newline),
system+user message history, multi-turn (user/assistant/user) history, an
empty-content edge case, non-ASCII/unicode content, and `enable_thinking=False`
(a real documented Qwen3 kwarg that switches a different template branch) --
**the rendered text matched byte-for-byte in all 7 cases.** This is the
single most important empirical result in this port; see
`tests/gen_fixtures.py` for how the fixtures were (re-)generated and
`tests/validate_continuous_token.cpp` for the harness that re-checks this
(via full tokenization, not just template rendering) on every run.

This is **not** a claim of universal Jinja2/minijinja equivalence -- it is
an empirical result over the representative inputs this workload actually
produces, same honesty standard as the other three ports' fixture-based
validation. `tools`/`tojson` and the tool-call template branches are
untested (and, per the scoping trace above, unreachable by this workload,
so this is a documented non-goal, not a silent gap).

### Why encoding the rendered text reproduces the exact real token ids

`apply_chat_template(tokenize=True)` must call the tokenizer's encode step
on the rendered string without re-adding special tokens (the template
already emits `<|im_start|>`/`<|im_end|>` etc. as literal text). Verified
empirically against all 4 non-trivial fixtures:
`tokenizer.encode(rendered_text, add_special_tokens=False) == token_ids`
in every case (also true for `add_special_tokens=True` here, since Qwen's
tokenizer config has no extra auto-added special tokens -- but the shim
uses `add_special_tokens=False` since that's what actually matches the real
`apply_chat_template` internals). The Rust shim therefore just needs to
(1) render the template via `minijinja`, then (2) call
`tokenizers::Tokenizer::encode(rendered, false)` on the real HF tokenizer
loaded from the exact same `tokenizer.json` the Qwen3 checkpoint ships --
same BPE vocab/merges/pre-tokenizer regex as the Python bindings, since it's
the same underlying Rust crate either way.

## Layout

```
csrc/continuous_token/
  rust/                        Cargo crate: the only code that touches
                                tokenizers/minijinja directly.
    Cargo.toml / Cargo.lock
    cbindgen.toml               cbindgen config (see below)
    src/lib.rs                  extern "C" FFI shim
  include/verl_continuous_token/
    continuous_token.h          hand-written, ergonomic C++ API (RAII)
    ffi/tokenizer_shim.h        cbindgen-GENERATED C header -- never hand-edited
  src/continuous_token.cpp      C++ wrapper impl + pure-C++ merge/align port
  tests/
    fetch_tokenizer.sh           downloads the real Qwen3-4B tokenizer files
                                  (not committed -- see .gitignore)
    gen_fixtures.py               generates tests/fixtures/*.txt from the
                                  REAL verl Python source (see below)
    fixtures/*.txt                 checked-in reference fixtures
    validate_continuous_token.cpp   exact-match C++ validation harness
  build.sh / CMakeLists.txt
```

## Why this crosses languages (Rust shim + C++ wrapper), unlike the other three ports

This was a decision already made before this port started (not re-litigated
here): bind to HF's real `tokenizers` Rust crate via its C API/FFI rather
than reimplementing BPE + chat-template rendering from scratch in C++. The
`tokenizers` crate has years of specific performance engineering
(priority-queue BPE merging, parallelized batch encoding, optimized
pre-tokenization regex) a fresh C++ port would not match without the same
investment, and token-ID correctness bugs are harder to catch than
float-tolerance bugs -- discrete token IDs are a hard-failure comparison.
There is no supported C API exposed directly by the `tokenizers` crate
itself (it's a Rust library, not a C library), so this port adds a thin
`#[no_mangle] extern "C"` shim crate (`rust/src/lib.rs`) rather than trying
to link `libtokenizers` directly.

## Build integration ("seamlessly connects")

Two build paths, both exercised for real by the author of this port (see
"What was, and wasn't, verified" below):

```sh
cd csrc/continuous_token
./build.sh                       # cargo build (release) -> cbindgen -> c++ build+link
./tests/fetch_tokenizer.sh       # one-time, needs network: real Qwen3-4B tokenizer files
./build/validate_continuous_token ./tests/fixtures \
    ./tests/fixtures/qwen3_tokenizer/tokenizer.json \
    ./tests/fixtures/qwen3_tokenizer/chat_template.jinja
```

or the conventional CMake path:

```sh
cd csrc/continuous_token
cmake -S . -B cmake_build -DCMAKE_BUILD_TYPE=Release
cmake --build cmake_build -j
./cmake_build/validate_continuous_token ./tests/fixtures \
    ./tests/fixtures/qwen3_tokenizer/tokenizer.json \
    ./tests/fixtures/qwen3_tokenizer/chat_template.jinja
```

Both `build.sh` and `CMakeLists.txt` invoke `cargo build` for `rust/` and
`cbindgen` to regenerate `include/verl_continuous_token/ffi/tokenizer_shim.h`
as part of the normal build -- "building the C++ library normally also
builds the Rust shim" is true for both paths, not just documented as a
manual prerequisite step. `CMakeLists.txt` uses `add_custom_command`
(`OUTPUT`/`DEPENDS` on `rust/src/lib.rs`/`Cargo.toml`) so incremental
rebuilds correctly re-invoke `cargo build` only when the Rust source
actually changed, same as any other CMake target.

The one new environment prerequisite this port has that `core_algos`/
`rollout_corr`/`data_proto` didn't: a Rust toolchain (`cargo`/`rustc`) and,
for header regeneration, `cbindgen` (`cargo install cbindgen`; both
`build.sh` and `CMakeLists.txt` degrade gracefully -- falling back to the
checked-in header with a warning -- if `cbindgen` isn't on `PATH`, but
`cargo` itself is a hard requirement). No libtorch/CUDA dependency at all
(this port has no tensor math).

The C header is **generated, not hand-written** (`cbindgen.toml` +
`cbindgen --crate verl_continuous_token_shim`), so it structurally cannot
drift from the real `extern "C"` signatures in `rust/src/lib.rs`. The C++
wrapper (`include/verl_continuous_token/continuous_token.h` +
`src/continuous_token.cpp`) is hand-written on top of that generated header,
giving callers an RAII `ContinuousTokenBuilder` class (move-only, no raw
pointer leaks across the FFI boundary -- builder handles and token-id
buffers are always freed through the matching `vct_*_free` function, never
`free()`/`delete` directly) instead of raw `extern "C"` calls.

## Validation methodology

`tests/gen_fixtures.py` imports the **real, unmodified**
`verl/utils/tokenizer/{tokenizer,chat_template,continuous_token}.py` from
this checkout via `importlib` (registered under their real dotted module
names so internal relative imports resolve), **not a reimplementation**,
and calls the real `QwenContinuousTokenBuilder` (the exact class
`create_continuous_token_builder` selects for `model_type="qwen3"`) against
the real downloaded Qwen3-4B tokenizer. It asserts at generation time that
this builder doesn't override the three in-scope methods (see "Scoping
trace" above) -- the scope claim this whole port depends on.

**Why `importlib` instead of `import verl...`/a full package install**:
`verl/__init__.py` imports `.protocol`, which pulls in Ray/torch/tensordict
-- none of which the three leaf modules this port touches actually need
(confirmed by reading them: stdlib + `transformers` only). Loading the
three files directly, under their real dotted names, calls the literal same
source without a multi-GB ML dependency install (`pip install -r
requirements.txt` pulls in `ray[default]`, `accelerate`, `torchdata`,
`TransferQueue`, etc.) just to validate token-id bookkeeping and
chat-template rendering. Disclosed here, not hidden.

`tests/validate_continuous_token.cpp` reads the generated `tests/fixtures/*.txt`,
runs the same inputs through the C++/Rust port, and compares **exact
token-ID sequences** -- not a tolerance comparison (the task's own framing:
discrete token IDs are a hard-failure comparison). Any mismatch anywhere is
reported as `[FAIL]` with the first differing index and both values.

Scenarios (11 total, `tests/fixtures/*.txt`, 38 individual checks):

- **`build_initial_tokens`** (7 scenarios, exact token-ID sequence against
  the real tokenizer): `gsm8k_basic` (the real GSM8K prompt format),
  `gsm8k_long_edge_case` (embedded quotes/apostrophes/newline),
  `system_plus_user`, `multi_turn_history`, `empty_user_content`,
  `unicode_content`, `enable_thinking_false`.
- **`merge_assistant_tokens` + `align_response_metadata`** (4 scenarios,
  all fields of `MergeResult` plus the aligned mask/logprobs): `basic`,
  `empty_assistant` (zero appended tokens), `no_logprobs` (response_logprobs
  absent end-to-end), `nonempty_prior_response_mask` (a non-empty
  `response_mask` going in, for thoroughness -- the real
  `SingleTurnAgentLoop` call site always passes `[]`, see the scoping
  trace above).

### Actual run output (this environment, arm64 macOS, CPU-only, no GPU)

```
[PASS] build_initial_tokens:build_initial_tokens_gsm8k_basic
[PASS] build_initial_tokens:build_initial_tokens_gsm8k_long_edge_case
[PASS] build_initial_tokens:build_initial_tokens_system_plus_user
[PASS] build_initial_tokens:build_initial_tokens_multi_turn_history
[PASS] build_initial_tokens:build_initial_tokens_empty_user_content
[PASS] build_initial_tokens:build_initial_tokens_unicode_content
[PASS] build_initial_tokens:build_initial_tokens_enable_thinking_false
... (31 more PASS lines: merge_assistant_tokens / align_response_metadata)

38 checks, 0 failures
ALL SCENARIOS PASSED
```

Reproducible via the build.sh/CMake commands above (after
`./tests/fetch_tokenizer.sh`); not all 38 lines reproduced here for
brevity.

## What was, and wasn't, verified

- **Verified for real, by the author of this port, on this machine (CPU
  only, arm64 macOS, no GPU -- correctly so, since this task has no CUDA
  dependency at all)**:
  - A real Rust toolchain was available (`brew install rust cmake`;
    `rustc 1.99.1`, `cargo 1.99.1`) -- so, per the task's own instruction,
    this was actually built and run, not just written and left untested.
  - `rust/` builds cleanly with **zero warnings** against the real,
    published `tokenizers = "=0.22.2"` + `minijinja`/`minijinja-contrib`
    crates (not vendored/modified copies).
  - `cbindgen` was installed and actually used to generate
    `include/verl_continuous_token/ffi/tokenizer_shim.h` from the real
    `rust/src/lib.rs` -- the header in this PR is the generated artifact,
    not hand-written-then-claimed-generated.
  - Both build paths (`build.sh` direct-compiler and `CMakeLists.txt`)
    were built end-to-end for real and produce working binaries.
  - The real Qwen/Qwen3-4B tokenizer files were downloaded from the HF Hub
    and used for every check below -- not synthetic/toy vocab data.
  - The chat-template-parity claim above (minijinja vs. real Python
    Jinja2, 7 scenarios) was checked with a standalone Rust test harness
    *before* `rust/src/lib.rs` was written, specifically to resolve the
    task's "real open question" with evidence rather than assumption.
  - All 38 fixture checks pass with **exact token-ID match** (not a
    tolerance) against fixtures generated by calling the real, unmodified
    Python `QwenContinuousTokenBuilder`.
  - `continuous_token.cpp` + `validate_continuous_token.cpp` were also
    built and run under `-fsanitize=address,undefined` (AddressSanitizer +
    UndefinedBehaviorSanitizer); this caught one real bug (a signed-integer
    left-shift overflow -- undefined behavior, not a correctness bug in the
    sense of wrong output, but UB nonetheless) in the test harness's own
    base64 decoder (`tests/validate_continuous_token.cpp`, not in the
    production `continuous_token.cpp`/Rust shim), which was fixed
    (`uint32_t` accumulator instead of `int`, well-defined wraparound) and
    re-verified clean under the same sanitizers, all 38 checks still
    passing. Leak detection (`ASAN_OPTIONS=detect_leaks=1`) is not
    supported on this platform (macOS) -- the RAII design (every
    Rust-allocated buffer is freed exactly once, through the matching
    `vct_*_free` function, in `continuous_token.cpp`'s destructors) was
    reviewed by hand instead, but not leak-checked under an instrumented
    allocator; that would need a Linux environment with `detect_leaks`
    support (or Valgrind) to actually exercise.
- **Not verified / explicitly out of scope** (see "Scoping trace" above
  for why each of these is believed unreachable by the validated workload,
  not merely unported):
  - `tools`/tool-calling (the `{% if tools %}` chat-template branch,
    `tojson`, `merge_context_tokens`/`ct_merge_context_msg`,
    `_tokenize_tool_group`) -- untested, and per the call-path trace,
    unreachable by `SingleTurnAgentLoop`.
  - Every other model-family `ContinuousTokenBuilder` subclass
    (`GptOssContinuousTokenBuilder`, `MiniMaxContinuousTokenBuilder`,
    `GLMContinuousTokenBuilder`, `Gemma4ContinuousTokenBuilder`,
    `DeepSeekContinuousTokenBuilder`) and all VL/multimodal subclasses --
    explicitly out of scope per the task instructions.
  - Multi-turn/tool-agent-loop usage (`ToolAgentLoop`, which does call
    `ct_merge_context_msg`) -- not this task's validated workload.
  - Integration into the actual Python/Ray/SGLang-or-vLLM rollout loop --
    this port is a standalone, C++-callable library exercised by its own
    test harness, same scope boundary as the other three ports (the
    plan's own sequencing treats runtime integration as a later step).
  - GPU/CUDA: not applicable to this task at all (no tensor math, no
    CUDA dependency) -- this is the first port in `csrc/` where that
    section is "N/A" rather than "not yet verified."
  - Performance: no benchmarking was done against the Python path (the
    task's correctness bar is exact-match, not speed; `tokenizers`' own
    performance engineering is the reason this task binds to it rather
    than reimplementing, per the decision already made before this port
    started, but this port did not independently measure a speedup).
  - Minijinja/Python Jinja2 parity beyond the 7 tested scenarios: not a
    claim of universal equivalence, an empirical result over this
    workload's representative inputs -- see the parity section above.

## Why `tokenizers = "=0.22.2"` exactly (not a version range)

Pinned to the exact version this repo's own dependency resolution produces
today (`uv.lock`: `transformers==5.12.1` -> `tokenizers==0.22.2`), not a
range, because the central finding of this port (no chat-template
capability in the Rust crate) was verified against that *exact* version's
source, not a range of versions that might behave differently. If this
repo's `transformers`/`tokenizers` pin moves, re-run the sdist grep this
README describes (`grep -ril "minijinja\|chat_template\|jinja"` over the
new version's extracted sdist) before assuming the finding still holds --
a future `tokenizers` release could add native chat-template support (the
task's own framing anticipated this: "Some versions of the Rust `tokenizers`
crate have added `minijinja`-based chat-template rendering; others haven't"),
which would be worth revisiting this port's reliance on a separate
`minijinja`/`minijinja-contrib` dependency for.
