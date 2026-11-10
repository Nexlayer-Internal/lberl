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
// C++/libtorch port of the tensor-batch half of `verl/protocol.py`'s
// `DataProto` class (PLAN.md table item #7 / section 6, item 3).
//
// Scope (see csrc/data_proto/README.md for the full rationale): this file
// ports ONLY the operations whose semantics are purely about
// reshaping/selecting/combining a batch of named tensors sharing a leading
// "batch size" dimension -- i.e. the `DataProto.batch` (a `TensorDict`) half
// of the class. It deliberately does NOT port `non_tensor_batch` (numpy
// object arrays holding arbitrary Python objects -- e.g. multi-turn chat
// message dicts), `meta_info` (a plain Python dict), Ray-serialization
// machinery (`__getstate__`/`__setstate__`, `DataProtoFuture`),
// `save_to_disk`/`load_from_disk`, `make_iterator` (wraps a PyTorch
// `DataLoader` with Hydra config), `auto_padding`/`padding`
// (tokenizer-padding-policy logic), or diagnostics (`print_size`,
// `get_data_info`, `_get_type_info`). Those remain a Python-side boundary,
// per the task's own scope instructions and PLAN.md's framing of this item
// as "just enough of a tensor-batch struct to let [core_algos /
// rollout_corr] run standalone against real GSM8K data, without yet solving
// the dynamic non_tensor_batch/meta_info problem in full generality."
//
// `TensorBatch` is the C++ named-tensor-map struct standing in for
// `DataProto.batch` (a `TensorDict`): a `std::unordered_map<std::string,
// torch::Tensor>` plus an explicit `batch_size` (the shared leading-dim
// extent across every entry), with `check_consistency()` enforcing the same
// "num_batch_dims == 1, every tensor's dim0 matches" invariant
// `DataProto.check_consistency()` enforces on its own `batch`/`non_tensor_batch`
// pair (here reduced to just the tensor side, since there is no
// `non_tensor_batch` in this struct).
//
// Mutating vs. value-returning methods deliberately mirror the Python
// source's own mutate-self-vs-return-new-object split exactly (see each
// method's doc comment below) rather than picking one convention
// uniformly, since that split is part of the semantics being ported, not an
// implementation detail: `to`, `rename`, `union_with`, and `reorder` mutate
// `*this` (matching `DataProto.to`/`.rename`/`.union`/`.reorder`, which all
// mutate `self` -- the first three also `return self` in Python, so the C++
// versions return `*this` by reference for chaining); `pop` mutates `*this`
// (removes keys) AND returns a new `TensorBatch` of the popped entries
// (matching `DataProto.pop`); everything else (`select`, `select_idxs`,
// `slice`, `concat`, `chunk`, `split`, `reorder`'s sibling `repeat`,
// `sample_level_repeat`, `unfold_column_chunks`) returns a new `TensorBatch`
// without touching `*this`.
#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

namespace verl_data_proto {

// C++ stand-in for `DataProto.batch` (a `TensorDict`): a dict of named
// tensors all sharing a leading batch-size dimension. `non_tensor_batch` and
// `meta_info` are NOT modeled here -- see the file header and
// csrc/data_proto/README.md for why.
class TensorBatch {
 public:
  TensorBatch() = default;

  // The real boundary-crossing constructor: build directly from a map of
  // named tensors (the C++-side equivalent of `DataProto.from_dict`, with
  // `non_tensors`/`meta_info`/`num_batch_dims` dropped since those are out of
  // scope here). Infers `batch_size` from the first tensor's dim0 (the
  // "pivot key", matching `from_dict`'s own pivot-key error-reporting
  // pattern) and runs `check_consistency()` before returning, matching
  // `DataProto.__post_init__`'s unconditional `self.check_consistency()`
  // call. Throws (`TORCH_CHECK`) if any tensor's dim0 disagrees with the
  // pivot, or if any tensor is 0-dimensional (no batch dim to speak of).
  static TensorBatch from_dict(std::unordered_map<std::string, torch::Tensor> tensors);

  // Same, but for the empty-map case where `batch_size` can't be inferred
  // (e.g. an intentionally-empty batch of a known size, or a batch that is
  // all non_tensor_batch on the Python side with zero tensor keys --
  // `DataProto.from_dict` represents that case as `batch=None`, which this
  // tensor-only struct instead represents as a zero-key map with an
  // explicit size). `tensors` must be non-empty for this to be unambiguous
  // with the single-arg overload's inference; use `from_dict(tensors)`
  // instead when `tensors` is non-empty.
  static TensorBatch from_dict(std::unordered_map<std::string, torch::Tensor> tensors, int64_t batch_size);

  // DataProto.check_consistency(): asserts every tensor has a dim0 equal to
  // `batch_size()`. Exposed as public per the Python source's own comment
  // ("We expose this function as a public one so that user can call
  // themselves directly").
  void check_consistency() const;

  // DataProto.__len__: the shared leading-dim extent.
  int64_t batch_size() const { return batch_size_; }

  bool contains(const std::string& key) const { return tensors_.count(key) > 0; }
  const torch::Tensor& at(const std::string& key) const;
  const std::unordered_map<std::string, torch::Tensor>& tensors() const { return tensors_; }

  // DataProto.select(batch_keys=...): returns a new TensorBatch containing
  // only `keys` (a strict subset -- throws if a key is missing, matching
  // TensorDict.select()'s default strict=True). Note the Python source's
  // `deepcopy` flag only ever affects `non_tensor_batch`/`meta_info`, never
  // the tensors themselves (`self.batch.select(...)` is unconditional) --
  // there is therefore no `deepcopy` parameter here; pass the result through
  // an explicit per-tensor `.clone()` yourself if you need a true copy.
  TensorBatch select(const std::vector<std::string>& keys) const;

  // DataProto.select_idxs(idxs): `idxs` may be an int64 index tensor or a
  // bool mask tensor (1-D), matching the Python source's dispatch on
  // `idxs.dtype == bool` vs. int. Returns a new TensorBatch.
  TensorBatch select_idxs(const torch::Tensor& idxs) const;

  // DataProto.slice(start, end, step): Python-slice semantics for `start`/
  // `end` (negative indices count from the end, `std::nullopt` stands in
  // for Python's `None`), matching `slice(start, end, step).indices(len(self))`.
  // `step` must be positive (`std::nullopt` means 1) -- verified directly
  // against the real Python source: `self.batch[slice_obj]` delegates to
  // TensorDict's own indexing, which raises `"step must be greater than
  // zero"` for `step <= 0` (confirmed by running `DataProto.slice(None,
  // None, -1)` against the real tensordict dependency in this checkout, see
  // README.md); this port matches that restriction with a `TORCH_CHECK`
  // rather than independently supporting negative-step reversal Python
  // itself doesn't allow here. Returns a new TensorBatch.
  TensorBatch slice(std::optional<int64_t> start, std::optional<int64_t> end,
                     std::optional<int64_t> step = std::nullopt) const;

  // DataProto.pop(batch_keys=...): mutates `*this` (removes `keys`) and
  // returns a new TensorBatch of exactly the popped entries, at the same
  // `batch_size()`.
  TensorBatch pop(const std::vector<std::string>& keys);

  // DataProto.rename(old_keys, new_keys) + its validate_input helper
  // (accepts a single key as a convenience in Python via `str`; here a
  // single-element vector plays that role, so there's no separate overload).
  // `old_keys`/`new_keys` must have equal length. Mutates `*this` in place
  // (via an erase + re-insert per pair, standing in for TensorDict's
  // `rename_key_`) and returns `*this` by reference, matching
  // `DataProto.rename`'s `return self`.
  TensorBatch& rename(const std::vector<std::string>& old_keys, const std::vector<std::string>& new_keys);

  // DataProto.union(other): per the Python docstring, throws if batch sizes
  // differ, or if a key present in both `*this` and `other` holds unequal
  // tensor values (`tensor_dict1[key].equal(tensor_dict2[key])` -- exact
  // value equality, not just matching shape/dtype). Keys present only in
  // `other` are added to `*this`. Mutates `*this` and returns `*this` by
  // reference, matching `DataProto.union`'s `return self`.
  TensorBatch& union_with(const TensorBatch& other);

  // DataProto.concat (static): combine a list of TensorBatches into one via
  // `torch::cat` along dim 0. Every batch must have the exact same key set
  // (matching tensordict's own `cat` requirement) -- this is the "exact
  // same keys, but the batches may otherwise differ in dim0 extent (and, if
  // shapes beyond dim0 agree, in dim0-extent alone)" reading of "concat of
  // differently-shaped-but-compatible batches" from the task's validation
  // bar. Throws on an empty `batches` list (there is no well-defined
  // "concat of zero batches" result without knowing the key set).
  static TensorBatch concat(const std::vector<TensorBatch>& batches);

  // DataProto.chunk(chunks): splits along dim 0 into exactly `chunks`
  // pieces using the same split-size algorithm `torch::Tensor::chunk` uses
  // (ceil-division sizes, with the last piece possibly smaller) -- computed
  // once via a dummy `torch::arange(batch_size())` so behavior is identical
  // whether or not `*this` has any tensor keys at all. The Python source
  // additionally asserts `len(self) % chunks == 0` unless
  // `is_padding_enabled()` (an auto-padding config flag this port does not
  // model, see the file header) -- `require_equal_chunks` (default `true`)
  // stands in for that assertion; pass `false` to exercise the uneven-split
  // code path real `auto_padding=True` callers rely on. Returns a new
  // vector of TensorBatch; does not mutate `*this`.
  std::vector<TensorBatch> chunk(int64_t chunks, bool require_equal_chunks = true) const;

  // DataProto.split(split_size): `[self[i:i+split_size] for i in
  // range(0, len(self), split_size)]` -- implemented directly in terms of
  // `slice`. Returns a new vector of TensorBatch; does not mutate `*this`.
  std::vector<TensorBatch> split(int64_t split_size) const;

  // DataProto.reorder(indices): "Note that this operation is in-place" --
  // matches exactly: mutates `*this` via `index_select(0, indices)` per
  // tensor (an arbitrary int64 index tensor, not necessarily a permutation
  // -- repeats are allowed, matching plain tensor advanced indexing) and
  // updates `batch_size()` to `indices.numel()`. No return value, matching
  // the Python source (which has no `return` statement either).
  void reorder(const torch::Tensor& indices);

  // DataProto.repeat(repeat_times, interleave=True): `interleave=true`
  // matches `tensor.repeat_interleave(repeat_times, dim=0)`;
  // `interleave=false` matches the "stack" branch
  // (`tensor.unsqueeze(0).expand(repeat_times, *shape).reshape(-1, *shape[1:])`,
  // i.e. tile the whole batch `repeat_times` times rather than repeating
  // each row). Returns a new TensorBatch; does not mutate `*this`.
  TensorBatch repeat(int64_t repeat_times = 2, bool interleave = true) const;

  // DataProto.sample_level_repeat(repeat_times): per-row repeat counts (one
  // count per existing row, `repeat_times.size() == batch_size()` required)
  // via `torch::repeat_interleave`'s per-element-count overload. Returns a
  // new TensorBatch; does not mutate `*this`.
  TensorBatch sample_level_repeat(const std::vector<int64_t>& repeat_times) const;

  // DataProto.unfold_column_chunks(n_split, split_keys): for keys in
  // `split_keys`, reshape dim1 (`shape[1] % n_split == 0` required) into
  // `n_split` and unfold it onto a widened dim0 (`bsz * n_split`); every
  // other key (including every key at all, if `split_keys` is `std::nullopt`,
  // matching the Python source's `if key in split_keys if split_keys is not
  // None else False` guard) is instead `repeat_interleave(n_split, dim=0)`'d
  // to match the new, wider dim0. Returns a new TensorBatch; does not
  // mutate `*this`.
  TensorBatch unfold_column_chunks(int64_t n_split,
                                    std::optional<std::vector<std::string>> split_keys = std::nullopt) const;

  // DataProto.to(device): moves every tensor to `device`. Mutates `*this`
  // and returns `*this` by reference, matching `DataProto.to`'s `return
  // self`.
  TensorBatch& to(torch::Device device);

 private:
  std::unordered_map<std::string, torch::Tensor> tensors_;
  int64_t batch_size_ = 0;
};

}  // namespace verl_data_proto
