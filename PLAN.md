# verl Python-Elimination Survey

**Scope:** Survey only. No migration code. This document inventories `lbverl`
(forked from `ETOgaosion/verl`, branch `codex/fix-nvshmem-image-20261023`) and
proposes a prioritized order for eliminating Python from its RL/GRPO
orchestration layer. It does **not** touch `lbtorch`, `liquid-brain-native`,
`liquid-brain-native-training`, GCP, or any running infrastructure.

**Validated workload used as the grounding reference throughout:**
`examples/grpo_trainer/run_qwen3_4b_fsdp.sh` — Qwen3-4B, GSM8K
(`$HOME/data/gsm8k/{train,test}.parquet`), `algorithm.adv_estimator=grpo`,
2-GPU FSDP, real completed training steps at ~100s/step, run both via Docker
and natively. Every claim below is checked against this configuration's
actual code path, not against verl's full feature surface (Megatron,
multi-modal, SGLang/TRT-LLM backends, etc. exist in this repo but are **not**
what was validated).

One load-bearing correction up front: `config.trainer.use_v1=True` is the
shipped default (`verl/trainer/config/ppo_trainer.yaml:232`), and
`TaskRunnerV1.run()` **unconditionally forces** `config.transfer_queue.enable
= True` (`verl/trainer/main_ppo.py:147`), regardless of
`transfer_queue.yaml`'s own `enable: False` default. So the "legacy"
`RayPPOTrainer`/`ray_trainer.py` path and the `v1/trainer_base.py` +
TransferQueue path are not equally-weighted alternatives — **v1 +
TransferQueue is the real default control flow** for the validated run, and
this plan is scoped to that path, not the deprecated `main_ppo_v0.py` path.

---

## 1. Architecture, as actually exercised by the validated run

```
TaskRunnerV1 (verl/trainer/main_ppo.py:106)
  -> PPOTrainer subclass (verl/trainer/ppo/v1/trainer_base.py:114, .fit() at :413)
       -> replay_buffer.sample()            [verl/trainer/ppo/v1/replay_buffer.py]
       -> AgentLoopManagerTQ.generate_sequences()  [verl/trainer/ppo/v1/agent_loop_tq.py:235]
            -> AgentLoopWorkerTQ (Ray actor) [agent_loop_tq.py:55]
                 -> rollout backend (vLLM today; RolloutReplica abstraction)
                 -> reward scoring (sync or RewardLoopWorker)
            -> results land in TransferQueue (external pkg, KV store)
       -> WorkerDict / FSDP actor workers (Ray actors) [verl/single_controller/ray/base.py:1032]
            -> verl/workers/engine_workers.py, verl/trainer/ppo/core_algos.py (the GRPO math)
       -> checkpointing, metrics, validation loop
```

Worker roles are all Ray actors (`@ray.remote`), dynamically composed via
`create_colocated_worker_cls` (`verl/single_controller/ray/base.py:1012`),
which monkey-patches multiple role implementations (actor, rollout, ref)
onto one `WorkerDict` class so they can be colocated in a single process /
GPU set. Data moves between roles via `verl.protocol.DataProto`
(`verl/protocol.py:318`) for in-Ray-call-arg data and via **TransferQueue**
(external pip package `TransferQueue==0.1.11`, not vendored in this repo) for
cross-worker trajectory storage — `verl/utils/transferqueue_utils.py` is the
~480-line glue (`tqbridge` decorator at line 347) that converts between
`BatchMeta` handles and live `TensorDict` data.

---

## 2. Prioritized subsystem table

| # | Subsystem | What it does | Category | Recommendation | Relative effort | Why it matters for Qwen3-4B/GSM8K/2-GPU-FSDP specifically |
|---|---|---|---|---|---|---|
| 1 | **GRPO advantage + loss math** — `verl/trainer/ppo/core_algos.py` (2559 lines): `compute_grpo_outcome_advantage` (L268), `kl_penalty`/`kl_penalty_forward` (L2188, L2216, k1/k2/k3 estimators), `compute_policy_loss_vanilla` (L1286, PPO-clip + dual-clip), 15+ other advantage/loss variants (GSPO L1546, DAPO-style clip-cov L1744, CISPO L2048, etc.) | **Genuine RL logic** | **Port first** | Low–Medium | This is the literal GRPO algorithm running every one of the ~100s training steps. Confirmed framework-agnostic: imports are only `torch`, `numpy`, `omegaconf.DictConfig` (for scalar hyperparams like `clip_ratio`, accessed via `.get()`/attribute access, not deep Python logic) — **no HF/vLLM/Ray imports at all**. Pure tensor-in/tensor-out math; exactly the shape `lbtorch`'s validated C++ autograd + `torch::optim::AdamW` can execute directly. |
| 2 | **Off-policy / rollout-correction math** — `verl/trainer/ppo/rollout_corr_helper.py` (1151 lines) + `score_centering.py` (554 lines) | Truncated importance-sampling correction for rollout-vs-train policy mismatch (vLLM BF16 vs FSDP FP32 today), rejection sampling, training-inference-mismatch head/tail score centering (arXiv:2609.20807) | **Genuine RL logic** | Port alongside/after #1 | Medium | Larger and more intricate than vanilla GRPO (multi-granularity IS, log-space numerics, ESS/rejection diagnostics), but still pure `torch.Tensor` math (`verl/workers/utils/losses.py`, `verl/utils/ulysses` for sequence-parallel slicing are its only non-trivial deps). **Directly relevant to the lbx rollout decision** (see §5): any precision/implementation gap between lbx-served logits and libtorch-trained logits reintroduces exactly the mismatch this code already exists to correct — do not treat this as legacy cruft to drop. |
| 3 | **Reward scoring for simple/deterministic rewards** — `verl/utils/reward_score/gsm8k.py` (regex string match, the actual GSM8K scorer used), `math_reward.py`, etc. | Checks final-answer correctness via regex/string compare | **Acceptable boundary, but keep scriptable** | Keep as a Python (or embeddable-scripting) boundary | N/A (keep) | Trivially simple for GSM8K, but `verl/trainer/ppo/reward.py:50` (`get_custom_reward_fn`) and `verl/utils/reward_score/sandbox_fusion/` show the *general* case: dynamically-loaded arbitrary user Python, including sandboxed arbitrary code execution for programming-task verifiers. See §4 — this is a real, defensible exception to "zero Python," not laziness. |
| 4 | **Reward dispatch/aggregation** — `verl/workers/reward_manager/naive.py` (serial per-sample Python loop calling `compute_score`, with a `SIGALRM`-based per-sample timeout at line 28), `dapo.py`, `prime.py`, `batch.py` | Iterates the batch, decodes tokens (HF tokenizer), calls the reward fn, assembles a reward tensor | Mostly plumbing around #3 | Keep as boundary (follows #3) | N/A | Confirms reward computation is driver-process-serial and Python-native by design (timeout guard exists specifically because `compute_score` can hang on pathological regex backtracking) — not something meaningfully portable independent of #3. |
| 5 | **Rollout/generation path — vLLM today, `lbx` is the decided direction** — `verl/workers/rollout/vllm_rollout/*`, `replica.py`, `router.py`, `llm_server.py` | Generates responses for each GRPO group (`rollout.n` samples/prompt); today via vLLM's async HTTP server + Python bindings | **Infrastructure, with a defined pluggable extension point** | Build `lbx` `RolloutReplica` backend (settled direction, not evaluate-later) | **High** | `verl/workers/rollout/replica.py:65` already defines an abstract `RolloutReplica` + `RolloutReplicaRegistry` (vLLM/SGLang/TRT-LLM are registered backends today) — adding `lbx` is architecturally a new registered backend, not a new abstraction. But the real work is non-trivial and genuinely new: (a) lbx's production path serves NVFP4-quantized weights; GRPO training needs a BF16/FP8 serving path alongside it, since the training policy isn't NVFP4; (b) weight sync currently goes through a vLLM-specific adapter (`verl/workers/rollout/vllm_rollout/delta_weight_transfer.py` — ZMQ/CUDA-IPC delta payloads decoded via `vllm.model_executor` internals, with TP-slicing and tied-embedding-alias handling, `weight_update_utils.py:20`) that reloads weights **every GRPO step**; lbx needs an equivalent live-reload path, not the static-checkpoint serving its production path assumes; (c) a generation API the `AgentLoopWorker`-equivalent calls instead of vLLM's OpenAI-style HTTP server. None of this is free; budget it as real integration engineering. |
| 6 | **Data-passing layer — TransferQueue integration** — `verl/utils/transferqueue_utils.py`, `verl/trainer/ppo/v1/replay_buffer.py` (613 lines), `verl/trainer/ppo/v1/agent_loop_tq.py` | External KV-store package providing metadata tracking (`TransferQueueController`) + data plane (`SimpleStorage`/other backends); verl's `ReplayBuffer`/`ReplayBufferAsync` implement real GRPO-group eviction/refill/DAPO-filter/off-policy-staleness policy on top of it | **Mixed**: the eviction/staleness/DAPO-filter *policy* is genuine orchestration logic; the KV-store substrate underneath is infrastructure scaled for a problem this workload doesn't have | Port the policy logic; treat TransferQueue itself as a boundary to simplify, not replicate | Medium (policy) / Low priority (substrate) | TransferQueue's own docs target "128×H100," "64 nodes/1024 cards," multi-modal payloads with SSD/RDMA offload (`docs/data/transfer_queue.md`). At 2-GPU/Qwen3-4B scale, the default `SimpleStorage` backend still spins up `num_data_storage_units: 8` round-robined "across eligible alive Ray nodes" (`verl/trainer/config/transfer_queue/transfer_queue.yaml:22`) — real overhead for a problem that doesn't exist at this scale. The *policy* (which groups are stale, DAPO reward-variance filtering, sync-vs-async sampling) is real GRPO-adjacent logic worth preserving conceptually; the KV-store itself is not worth re-implementing in C++ as a goal — a direct in-process/shared-memory handoff would do the same job at this scale. |
| 7 | **`DataProto` tensor-batch container** — `verl/protocol.py` (1342 lines, class at L318) | Batch container: `{batch: TensorDict, non_tensor_batch: dict[str,np.ndarray], meta_info: dict}`, with custom pickling-adjacent (de)serialization for Ray transport (L241-293) | Infrastructure (data structure) | Port the tensor part; keep the dynamic part as a boundary | Medium | The tensor half (`batch`) maps cleanly to a C++ named-tensor-map struct. `non_tensor_batch`/`meta_info` carry real Python dynamism (arbitrary numpy object arrays holding strings/nested dicts — e.g. multi-turn chat messages, reward metadata) that is genuinely awkward to express in C++ without either a variant/any type or dropping flexibility the orchestration logic currently relies on. |
| 8 | **Ray actor orchestration** — `verl/single_controller/ray/base.py` (1151 lines: `RayWorkerGroup` L430, `create_colocated_worker_cls` L1012, dynamic `WorkerDict` L1032), `verl/single_controller/base/decorator.py` (Dispatch modes, 444 lines) | Actor lifecycle, placement groups, resource pools, dynamic multi-role class composition via monkey-patching, argument dispatch/collection across data-parallel ranks | **Infrastructure** | **Keep Ray; do not attempt to replace it now** | N/A (explicit non-goal for this phase) | See §3 — detailed honest assessment below. |
| 9 | **Dataset loading/preprocessing** — `verl/utils/dataset/rl_dataset.py` (imports `datasets`, `transformers.PreTrainedTokenizer`), `examples/data_preprocess/gsm8k.py` | GSM8K parquet loading, prompt templating, tokenization | **Acceptable boundary** | Leave as Python/HF boundary | N/A (keep) | Same conclusion as the sibling `lbtorch` survey reached for its own dataset path, and for the same reason: this is one-shot preprocessing before the hot training loop, not per-step logic, and depends on the HF `datasets`/`transformers` ecosystem which has no equivalent value in being ported. No different considerations apply here. |
| 10 | **Training-loop control flow** — `verl/trainer/ppo/v1/trainer_base.py` (`PPOTrainer.fit()` L413, `.step()` L534, `._step_once()` L563) | Epoch/step loop, checkpoint cadence, validation cadence, metrics aggregation, parameter-sync-step batching | Orchestration logic (real policy, not boilerplate, but not GRPO math either) | Defer — port after #1/#2 are proven standalone | Medium-High (entangled with Ray/TQ/config) | This is the "FSDP distributed logic" analog from the `lbtorch` survey: real control-flow logic, but deeply entangled with Ray actor handles, Hydra/OmegaConf config objects, and TransferQueue KV-batch metadata (`KVBatchMeta`) at nearly every line. Following the same "biggest win for least entangled effort first" logic that put `lbtorch`'s optimizer math before its FSDP work, this is the equivalent of the "FSDP" side here — valuable, but should come after the self-contained math (#1/#2), not before. |

---

## 3. Ray: honest assessment, explicit recommendation

**Does Ray's complexity earn its keep at this workload's actual scale (2
GPUs, a handful of long-lived worker roles)?** Partially, and the answer is
not simply "replace it" or "keep it forever" — it's **keep it now, revisit
later, and don't confuse "replace Ray" with "port the algorithm."**

What Ray is actually doing here, concretely: actor supervision/placement
across GPUs (`RayResourcePool`/`ResourcePoolManager`,
`verl/single_controller/ray/base.py:113,193`), dynamic multi-role class
composition so actor + rollout + ref can share a process
(`create_colocated_worker_cls`, L1012), and argument dispatch/collection
across data-parallel ranks via a `Dispatch` enum
(`verl/single_controller/base/decorator.py`). None of this is "thousands of
ephemeral tasks" — it's exactly the "handful of long-lived worker roles with
well-defined data exchange" pattern the task brief anticipated, which is
precisely the pattern Ray's actor model is *least* over-engineered for
relative to Ray's more exotic features (dynamic task graphs, autoscaling,
multi-tenant scheduling) that this workload doesn't use.

**Is there a lighter-weight C++-native alternative?** Ray does ship a C++
API, but it is explicitly marked experimental, has nowhere near Python-API
feature parity (placement groups, runtime_env, async actors, and the
dynamic-class-composition pattern this repo relies on all assume the Python
API), and is not what Ray itself recommends for application development.
Porting this orchestration layer onto Ray's C++ API would mean fighting an
immature API to reimplement features the Python API already gives for free
— that is not a port, it is a rewrite onto worse tooling.

**The honest alternative is not "Ray's C++ API" but "build a from-scratch
replacement"** — a custom process-group/RPC layer (gRPC or raw ZMQ, direct
shared-memory or CUDA-IPC tensor transfer) covering: actor supervision and
restart (fault tolerance Ray gives for free today), GPU placement/scheduling,
and the data-exchange patterns TransferQueue and `DataProto` currently
handle. That is a lot of new infrastructure engineering, not a port of
existing logic, and for a 2-GPU workload it buys determinism and a smaller
dependency footprint at the cost of reimplementing operationally-proven
plumbing.

**Recommendation:** Do not attempt to replace Ray in this phase. Treat Ray
(and, honestly, TransferQueue's control-plane concept, if not its
scaled-for-1024-GPUs substrate) as the orchestration boundary to keep while
the self-contained algorithmic core (#1/#2 in the table) is ported to
C++/libtorch as a library that the existing Python/Ray workers call into.
Only once that core exists and is validated standalone does it become
worth asking whether a lighter custom orchestration layer is justified for
the now-smaller remaining Python surface — and even then, budget fault
tolerance and GPU placement as real work, not assumed-free.

---

## 4. Reward functions: honest assessment, explicit recommendation

**Do reward functions need to support arbitrary Python in general?** Yes,
and this is a second real, defensible exception to "zero Python anywhere" —
not a workaround chosen for convenience.

Evidence: `verl/trainer/ppo/reward.py:50` (`get_custom_reward_fn`) dynamically
imports a user-specified Python callable by file path + function name
(`load_extern_object`), with arbitrary `reward_kwargs` merged in, and
explicitly supports both sync and async functions. `verl/utils/reward_score/`
includes `sandbox_fusion/` (executes untrusted model-generated code in a
sandboxed subprocess to verify programming-task solutions) and `prime_code`/
`prime_math` (symbolic/code-execution-based verifiers) — these are not
edge cases in real RLVR practice, they are standard reward-function shapes
for code and math tasks, and inherently require executing arbitrary logic
(or, in the sandbox case, arbitrary *code*) per sample.

The GSM8K reward actually used in the validated run
(`verl/utils/reward_score/gsm8k.py:compute_score`) is deceptively simple —
pure regex string matching, no model calls, no external execution — so the
validated run itself does not exercise this complexity. But a plan that
assumes "reward is always this simple" because that's what GSM8K needed
would be wrong for the system's real intended use.

**Recommendation:** Keep reward computation behind a Python (or embeddable
scripting language, e.g. an embedded Python interpreter or a sandboxed
scripting layer) boundary deliberately, not provisionally. Do not plan to
port `compute_score` logic into the C++ core. The call boundary — "C++
orchestration core calls out to a scoring function, waits for a scalar/dict
back" — is cheap to keep Python-shaped regardless of what the rest of the
system becomes, and reward-manager dispatch (`naive.py`,
`verl/workers/reward_manager/`) already treats this as an external callback,
not inline logic, which is the right shape to preserve.

---

## 5. Rollout/generation: `lbx` is the decided direction (not an open question)

verl's existing `RolloutReplica` abstraction (`verl/workers/rollout/replica.py:65`,
registry at line 311) already supports multiple interchangeable generation
backends (vLLM, SGLang, TRT-LLM are registered today) — adding `lbx` fits
this as a new registered backend, architecturally. That is the good news.
The real scope of work, assessed honestly rather than assumed trivial:

1. **BF16/FP8 serving path.** lbx's production path serves NVFP4-quantized
   weights (per the task brief); the GRPO training policy is not quantized
   that way. This is a second serving path in lbx, not a config flag on the
   existing one.
2. **Dynamic weight reload every GRPO step.** Today, weight sync from
   FSDP-trained weights to the vLLM rollout workers goes through a
   vLLM-specific adapter: `verl/workers/rollout/vllm_rollout/delta_weight_transfer.py`
   decodes ZMQ/CUDA-IPC "delta" payloads and calls **vLLM's own internal
   checkpoint-patch loader** (handles checkpoint names, packed weights, TP
   slicing); `weight_update_utils.py:20` additionally handles
   tied-embedding-alias deduplication specific to vLLM's
   `VocabParallelEmbedding`. None of this transfers to lbx for free — lbx
   needs an equivalent "accept a live weight delta, apply it correctly under
   whatever sharding lbx uses, every step" path, which is new integration
   work, not a config change. This is the single largest piece of genuinely
   new (not ported) engineering in this whole plan.
3. **A generation API call site.** The `AgentLoopWorker`-equivalent
   (`verl/experimental/agent_loop/agent_loop.py`, `verl/trainer/ppo/v1/agent_loop_tq.py`)
   needs to call an lbx generation endpoint instead of vLLM's async HTTP
   server (`verl/workers/rollout/vllm_rollout/vllm_async_server.py`). Shape
   of integration is well understood from the existing multi-backend
   pattern; this is the smallest part of the three.
4. **A side effect worth flagging, not solving now:** `rollout_corr_helper.py`
   and `score_centering.py` (table rows #2) exist specifically to correct
   for rollout-engine-vs-training-engine numerical mismatch (today: vLLM
   BF16 vs FSDP FP32). Moving rollout to lbx changes *what* that mismatch
   looks like (different kernels, different precision characteristics) but
   does not remove the need for this correction code — if anything, a new
   inference engine makes validating that correction logic more important,
   not less.

**Framing for the plan:** this is scoped, real integration work (item #5 in
the priority table, "High" effort) that should be sequenced after the
self-contained GRPO math (#1/#2) is ported and validated standalone, because
the weight-reload adapter work in particular will want to interoperate with
whatever the new C++ training core's weight representation looks like.

---

## 6. Ordered recommendation (what to port first, and why)

Following the same "biggest win for least entangled effort first" logic the
`lbtorch` survey used — there, self-contained optimizer math came before
deeply-entangled FSDP work:

1. **`core_algos.py` GRPO advantage + policy-loss math** (table #1). Smallest
   dependency footprint of anything in this repo that matters for training
   correctness (`torch`/`numpy`/`omegaconf` scalars only), directly
   expressible against `lbtorch`'s already-validated C++ autograd +
   `torch::optim::AdamW`. Highest confidence, lowest entanglement.
2. **`rollout_corr_helper.py` / `score_centering.py`** (table #2). Same
   "pure tensor math" profile as #1, larger and more intricate, and becomes
   load-bearing the moment lbx rollout lands (§5) — port it while the
   vLLM-based mismatch patterns are still available to validate against.
3. **`DataProto`'s tensor half + a minimal C++-callable boundary** (table
   #7) — just enough of a tensor-batch struct to let #1/#2 run standalone
   against real GSM8K data, without yet solving the dynamic
   `non_tensor_batch`/`meta_info` problem in full generality.
4. **`lbx` rollout backend** (table #5, §5) — sequenced after #1-3 are
   validated standalone, since the weight-reload adapter wants a stable
   picture of what the new training core's weights look like.
5. **TransferQueue-backed replay-buffer policy logic** (table #6) — port the
   eviction/staleness/DAPO-filter *policy*, but do not treat replicating
   TransferQueue's distributed KV-store substrate as a goal at this scale.
6. **Training-loop control flow** (table #10) — the "FSDP-equivalent" of
   this project: real, necessary, but the most Ray/TQ/config-entangled
   piece, so it comes last, once the pieces it orchestrates already exist
   in C++.

Explicitly **not sequenced / explicit non-goals for this phase**: replacing
Ray (§3), porting reward-function logic into C++ (§4), touching dataset/HF
tokenizer preprocessing (table #9).

---

## 7. Dependency direction: where this project attaches to `lbtorch`

This project's eventual C++ core (starting with table items #1-#3) should
depend on `lbtorch` for actual tensor/autograd/optimizer operations, not
reimplement them. The attachment point is exactly the boundary `core_algos.py`
already exposes today: functions taking/returning plain `torch.Tensor` with
no framework-specific state threaded through (confirmed by the import list
at the top of `verl/trainer/ppo/core_algos.py` — `torch`, `numpy`,
`omegaconf`, and verl's own config/utility modules only). A C++ port of
these functions against `lbtorch`'s C++ tensor/autograd API is a drop-in
replacement for this file's role, callable from whatever remains of the
Python/Ray orchestration layer during the transition, and eventually from
this project's own C++ orchestration core once that exists. No integration
work between the two projects is being started now — this is purely a note
on where the seam is, for whoever picks up table item #1.
