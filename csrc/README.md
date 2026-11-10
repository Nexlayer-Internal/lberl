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

Code here targets [libtorch](https://pytorch.org/cppdocs/) (`torch::Tensor`
C++ API) directly — no dependency on the sibling `lbtorch` project's own
in-progress work; see `PLAN.md` section 7 for the longer-term note on where
this project is expected to eventually depend on `lbtorch` for
tensor/autograd/optimizer internals instead of upstream libtorch.
