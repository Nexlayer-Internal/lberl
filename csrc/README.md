# `csrc/` — C++ ports of this repo's Python-elimination migration (`PLAN.md`)

First C++ code added to this repository; see `PLAN.md` at the repo root (as
merged into `codex/fix-nvshmem-image-20261023` — see
`core_algos/README.md`'s correction note for why that caveat is here) for the
overall survey and prioritized migration order.

Each ported subsystem gets its own subdirectory:

- `core_algos/` — GRPO advantage computation, PPO-clip/dual-clip policy
  loss, and KL penalty math, ported from `verl/trainer/ppo/core_algos.py`.
  See `core_algos/README.md` for exact scope, build/test instructions, and
  validation results.
- `rollout_corr/` — off-policy / rollout-correction math (truncated
  importance sampling, rejection sampling, training-inference-mismatch
  score centering), ported from `verl/trainer/ppo/rollout_corr_helper.py`
  and `score_centering.py`. See `rollout_corr/README.md`.
- `data_proto/` — the tensor-batch half of `verl/protocol.py`'s `DataProto`
  (PLAN.md table item #7): a C++ named-tensor-map struct
  (`verl_data_proto::TensorBatch`) covering `check_consistency`, `select`/
  `select_idxs`/`slice`, `pop`, `rename`, `union`, `concat`, `chunk`/`split`,
  `reorder`, `repeat`/`sample_level_repeat`, `unfold_column_chunks`, and
  `to(device)` -- just enough of `DataProto.batch` to let `core_algos`/
  `rollout_corr` run standalone against real batched tensor data, without
  solving the dynamic `non_tensor_batch`/`meta_info` problem. See
  `data_proto/README.md` for exact scope, build/test instructions, and
  validation results.
- `continuous_token/` — online prompt tokenization (PLAN.md table item
  #9b): a Rust FFI shim binding the real HF `tokenizers` crate (BPE encode)
  + `minijinja` (chat-template rendering) behind an RAII C++ wrapper,
  ported from `verl/utils/tokenizer/continuous_token.py`'s
  `ContinuousTokenBuilder.build_initial_tokens` for the validated
  single-turn, text-only Qwen3-4B/GSM8K workload. Unlike the three ports
  above, this one is tokenization, not tensor math — no libtorch/CUDA
  dependency, and correctness means exact token-ID match rather than a
  float tolerance. See `continuous_token/README.md` for the scoping trace,
  the chat-template-parity investigation, build/test instructions, and
  validation results.

`core_algos/`, `rollout_corr/`, and `data_proto/` target
[libtorch](https://pytorch.org/cppdocs/) (`torch::Tensor` C++ API) directly
— no dependency on the sibling `lbtorch` project's own in-progress work; see
`PLAN.md` section 7 for the longer-term note on where this project is
expected to eventually depend on `lbtorch` for tensor/autograd/optimizer
internals instead of upstream libtorch. `continuous_token/` has no tensor
dependency at all (see its own README for why).
