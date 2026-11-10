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

#include "verl_data_proto/tensor_batch.h"

#include <algorithm>
#include <unordered_set>
#include <utility>

namespace verl_data_proto {

namespace {

// CPython's slice.indices(length) algorithm (Objects/sliceobject.c) for the
// step>0 case, ported directly so TensorBatch::slice matches Python's
// `DataProto.slice` (`self.batch[slice(start, end, step)]`) exactly,
// including negative start/end and out-of-range clamping. `step <= 0` is
// rejected with TORCH_CHECK rather than handled: verified directly against
// the real Python source (see tensor_batch.h's doc comment on `slice`) that
// `DataProto.slice(None, None, -1)` itself raises ("step must be greater
// than zero") on the real `tensordict` dependency in this checkout, so
// there is no real negative-step behavior to match here.
struct NormalizedSlice {
  int64_t start;
  int64_t stop;
  int64_t step;
};

NormalizedSlice normalize_python_slice(int64_t length, std::optional<int64_t> start_opt,
                                        std::optional<int64_t> stop_opt, std::optional<int64_t> step_opt) {
  int64_t step = step_opt.value_or(1);
  TORCH_CHECK(step > 0, "slice: step must be greater than zero (got ", step, ")");

  int64_t lower = 0;
  int64_t upper = length;

  int64_t start;
  if (!start_opt.has_value()) {
    start = lower;
  } else {
    start = *start_opt;
    if (start < 0) {
      start += length;
      if (start < lower) start = lower;
    } else if (start > upper) {
      start = upper;
    }
  }

  int64_t stop;
  if (!stop_opt.has_value()) {
    stop = upper;
  } else {
    stop = *stop_opt;
    if (stop < 0) {
      stop += length;
      if (stop < lower) stop = lower;
    } else if (stop > upper) {
      stop = upper;
    }
  }

  return {start, stop, step};
}

std::vector<int64_t> slice_row_indices(int64_t length, std::optional<int64_t> start, std::optional<int64_t> end,
                                        std::optional<int64_t> step) {
  NormalizedSlice ns = normalize_python_slice(length, start, end, step);
  std::vector<int64_t> idx;
  for (int64_t i = ns.start; i < ns.stop; i += ns.step) idx.push_back(i);
  return idx;
}

}  // namespace

TensorBatch TensorBatch::from_dict(std::unordered_map<std::string, torch::Tensor> tensors) {
  TORCH_CHECK(!tensors.empty(),
              "TensorBatch::from_dict: tensors must be non-empty to infer batch_size; use the "
              "(tensors, batch_size) overload for an empty map");
  int64_t batch_size = -1;
  std::string pivot_key;
  for (const auto& kv : tensors) {
    TORCH_CHECK(kv.second.dim() >= 1, "TensorBatch::from_dict: tensor '", kv.first,
                "' has no batch dimension (0-dim)");
    if (batch_size < 0) {
      batch_size = kv.second.size(0);
      pivot_key = kv.first;
    } else {
      TORCH_CHECK(kv.second.size(0) == batch_size,
                  "TensorBatch::from_dict: not all tensors have the same batch size. Got '", pivot_key, "' has ",
                  batch_size, ", '", kv.first, "' has ", kv.second.size(0));
    }
  }
  TensorBatch out;
  out.tensors_ = std::move(tensors);
  out.batch_size_ = batch_size;
  out.check_consistency();
  return out;
}

TensorBatch TensorBatch::from_dict(std::unordered_map<std::string, torch::Tensor> tensors, int64_t batch_size) {
  TensorBatch out;
  out.tensors_ = std::move(tensors);
  out.batch_size_ = batch_size;
  out.check_consistency();
  return out;
}

void TensorBatch::check_consistency() const {
  for (const auto& kv : tensors_) {
    TORCH_CHECK(kv.second.dim() >= 1, "check_consistency: tensor '", kv.first,
                "' has no batch dimension (0-dim); only num_batch_dims=1 is supported");
    TORCH_CHECK(kv.second.size(0) == batch_size_, "check_consistency: key '", kv.first, "' length ",
                kv.second.size(0), " is not equal to batch size ", batch_size_);
  }
}

const torch::Tensor& TensorBatch::at(const std::string& key) const {
  auto it = tensors_.find(key);
  TORCH_CHECK(it != tensors_.end(), "key not found in batch: ", key);
  return it->second;
}

TensorBatch TensorBatch::select(const std::vector<std::string>& keys) const {
  TensorBatch out;
  out.batch_size_ = batch_size_;
  for (const auto& k : keys) {
    auto it = tensors_.find(k);
    TORCH_CHECK(it != tensors_.end(), "select: key not found in batch: ", k);
    out.tensors_[k] = it->second;
  }
  return out;
}

TensorBatch TensorBatch::select_idxs(const torch::Tensor& idxs) const {
  TensorBatch out;
  if (idxs.scalar_type() == torch::kBool) {
    // `.item<int64_t>()` on a 0-dim reduction tensor: explicitly safe on any
    // device (performs the device-to-host copy itself), matching the same
    // safe pattern csrc/core_algos and csrc/rollout_corr rely on -- never
    // `.accessor<T,N>()`/raw `.data_ptr<T>()` assuming host memory.
    out.batch_size_ = idxs.sum().item<int64_t>();
    for (const auto& kv : tensors_) {
      out.tensors_[kv.first] = kv.second.index({idxs.to(kv.second.device())});
    }
  } else {
    torch::Tensor idxs_long = idxs.to(torch::kLong);
    out.batch_size_ = idxs_long.numel();
    for (const auto& kv : tensors_) {
      out.tensors_[kv.first] = kv.second.index_select(0, idxs_long.to(kv.second.device()));
    }
  }
  return out;
}

TensorBatch TensorBatch::slice(std::optional<int64_t> start, std::optional<int64_t> end,
                                std::optional<int64_t> step) const {
  std::vector<int64_t> idx = slice_row_indices(batch_size_, start, end, step);
  torch::Tensor idx_tensor = torch::tensor(idx, torch::kLong);

  TensorBatch out;
  out.batch_size_ = static_cast<int64_t>(idx.size());
  for (const auto& kv : tensors_) {
    out.tensors_[kv.first] = kv.second.index_select(0, idx_tensor.to(kv.second.device()));
  }
  return out;
}

TensorBatch TensorBatch::pop(const std::vector<std::string>& keys) {
  std::unordered_map<std::string, torch::Tensor> popped;
  for (const auto& k : keys) {
    auto it = tensors_.find(k);
    TORCH_CHECK(it != tensors_.end(), "pop: key not found in batch: ", k);
    popped[k] = it->second;
    tensors_.erase(it);
  }
  TensorBatch out;
  out.tensors_ = std::move(popped);
  out.batch_size_ = batch_size_;
  return out;
}

TensorBatch& TensorBatch::rename(const std::vector<std::string>& old_keys, const std::vector<std::string>& new_keys) {
  TORCH_CHECK(old_keys.size() == new_keys.size(),
              "rename: new_keys and old_keys must have the same length, but got ", new_keys.size(), " and ",
              old_keys.size());

  // Look up every old key before mutating anything, matching rename_key_'s
  // "validate first" spirit and avoiding a partially-renamed batch if one
  // of the requested keys doesn't exist.
  std::vector<torch::Tensor> values;
  values.reserve(old_keys.size());
  for (const auto& k : old_keys) {
    auto it = tensors_.find(k);
    TORCH_CHECK(it != tensors_.end(), "rename: key not found in batch: ", k);
    values.push_back(it->second);
  }
  for (const auto& k : old_keys) tensors_.erase(k);
  for (size_t i = 0; i < old_keys.size(); ++i) tensors_[new_keys[i]] = values[i];
  return *this;
}

TensorBatch& TensorBatch::union_with(const TensorBatch& other) {
  TORCH_CHECK(batch_size_ == other.batch_size_,
              "union: two tensor batches must have identical batch size. Got ", batch_size_, " and ",
              other.batch_size_);
  for (const auto& kv : other.tensors_) {
    auto it = tensors_.find(kv.first);
    if (it == tensors_.end()) {
      tensors_[kv.first] = kv.second;
    } else {
      // torch::equal is a dispatched op safe on any device (it performs the
      // comparison, including any device-to-host sync, internally) -- not a
      // raw accessor/data_ptr call.
      TORCH_CHECK(torch::equal(it->second, kv.second), "'", kv.first,
                  "' in this batch and other batch are not the same object");
    }
  }
  return *this;
}

TensorBatch TensorBatch::concat(const std::vector<TensorBatch>& batches) {
  TORCH_CHECK(!batches.empty(), "concat: batches must be non-empty");

  const auto& first_keys = batches[0].tensors_;
  for (size_t i = 1; i < batches.size(); ++i) {
    TORCH_CHECK(batches[i].tensors_.size() == first_keys.size(),
                "concat: all batches must have the same set of keys (batch 0 has ", first_keys.size(),
                " keys, batch ", i, " has ", batches[i].tensors_.size(), ")");
  }

  int64_t total_batch_size = 0;
  for (const auto& b : batches) total_batch_size += b.batch_size_;

  std::unordered_map<std::string, torch::Tensor> result;
  for (const auto& kv : first_keys) {
    std::vector<torch::Tensor> pieces;
    pieces.reserve(batches.size());
    for (const auto& b : batches) {
      auto it = b.tensors_.find(kv.first);
      TORCH_CHECK(it != b.tensors_.end(), "concat: key '", kv.first,
                  "' missing from one of the batches being concatenated");
      pieces.push_back(it->second);
    }
    result[kv.first] = torch::cat(pieces, 0);
  }

  TensorBatch out;
  out.tensors_ = std::move(result);
  out.batch_size_ = total_batch_size;
  return out;
}

std::vector<TensorBatch> TensorBatch::chunk(int64_t chunks, bool require_equal_chunks) const {
  TORCH_CHECK(chunks > 0, "chunk: chunks must be positive, got ", chunks);
  if (require_equal_chunks) {
    TORCH_CHECK(batch_size_ % chunks == 0, "chunk: only support equal chunk. Got size of TensorBatch ", batch_size_,
                " and chunk ", chunks, ".");
  }

  // Compute split boundaries once via a dummy index tensor so the result is
  // identical to torch::Tensor::chunk's own ceil-division algorithm,
  // whether or not this batch actually has any tensor keys.
  torch::Tensor idx = torch::arange(batch_size_, torch::kLong);
  std::vector<torch::Tensor> idx_chunks = idx.chunk(chunks, 0);

  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;
  int64_t offset = 0;
  for (const auto& c : idx_chunks) {
    offsets.push_back(offset);
    sizes.push_back(c.size(0));
    offset += c.size(0);
  }

  std::vector<TensorBatch> output;
  output.reserve(idx_chunks.size());
  for (size_t i = 0; i < idx_chunks.size(); ++i) {
    TensorBatch piece;
    piece.batch_size_ = sizes[i];
    for (const auto& kv : tensors_) {
      piece.tensors_[kv.first] = kv.second.narrow(0, offsets[i], sizes[i]);
    }
    output.push_back(std::move(piece));
  }
  return output;
}

std::vector<TensorBatch> TensorBatch::split(int64_t split_size) const {
  TORCH_CHECK(split_size > 0, "split: split_size must be positive, got ", split_size);
  std::vector<TensorBatch> output;
  for (int64_t i = 0; i < batch_size_; i += split_size) {
    int64_t end = std::min(i + split_size, batch_size_);
    output.push_back(slice(i, end, std::nullopt));
  }
  return output;
}

void TensorBatch::reorder(const torch::Tensor& indices) {
  torch::Tensor idx_long = indices.to(torch::kLong);
  for (auto& kv : tensors_) {
    kv.second = kv.second.index_select(0, idx_long.to(kv.second.device()));
  }
  batch_size_ = idx_long.numel();
}

TensorBatch TensorBatch::repeat(int64_t repeat_times, bool interleave) const {
  TORCH_CHECK(repeat_times > 0, "repeat: repeat_times must be positive, got ", repeat_times);
  TensorBatch out;
  out.batch_size_ = batch_size_ * repeat_times;
  for (const auto& kv : tensors_) {
    const torch::Tensor& t = kv.second;
    if (interleave) {
      out.tensors_[kv.first] = t.repeat_interleave(repeat_times, /*dim=*/0);
    } else {
      std::vector<int64_t> expand_shape;
      expand_shape.push_back(repeat_times);
      for (int64_t i = 0; i < t.dim(); ++i) expand_shape.push_back(t.size(i));

      std::vector<int64_t> reshape_shape;
      reshape_shape.push_back(-1);
      for (int64_t i = 1; i < t.dim(); ++i) reshape_shape.push_back(t.size(i));

      out.tensors_[kv.first] = t.unsqueeze(0).expand(expand_shape).reshape(reshape_shape);
    }
  }
  return out;
}

TensorBatch TensorBatch::sample_level_repeat(const std::vector<int64_t>& repeat_times) const {
  TORCH_CHECK(static_cast<int64_t>(repeat_times.size()) == batch_size_,
              "sample_level_repeat: repeat_times must have one entry per row, got ", repeat_times.size(),
              " for batch_size ", batch_size_);
  torch::Tensor counts = torch::tensor(repeat_times, torch::kLong);
  int64_t total = 0;
  for (int64_t c : repeat_times) total += c;

  TensorBatch out;
  out.batch_size_ = total;
  for (const auto& kv : tensors_) {
    out.tensors_[kv.first] = kv.second.repeat_interleave(counts.to(kv.second.device()), /*dim=*/0);
  }
  return out;
}

TensorBatch TensorBatch::unfold_column_chunks(int64_t n_split,
                                               std::optional<std::vector<std::string>> split_keys) const {
  TORCH_CHECK(n_split > 0, "unfold_column_chunks: n_split must be positive, got ", n_split);

  bool has_split_keys = split_keys.has_value();
  std::unordered_set<std::string> split_set;
  if (has_split_keys) split_set = std::unordered_set<std::string>(split_keys->begin(), split_keys->end());

  TensorBatch out;
  out.batch_size_ = batch_size_ * n_split;
  for (const auto& kv : tensors_) {
    const std::string& key = kv.first;
    const torch::Tensor& t = kv.second;
    if (has_split_keys && split_set.count(key) > 0) {
      TORCH_CHECK(t.dim() >= 2, "unfold_column_chunks: key '", key, "' must have at least 2 dims to split along dim1");
      TORCH_CHECK(t.size(1) % n_split == 0, "unfold_column_chunks: key '", key, "' dim1 (", t.size(1),
                  ") is not divisible by n_split (", n_split, ")");
      std::vector<int64_t> shape(t.sizes().begin(), t.sizes().end());
      shape[0] = t.size(0) * n_split;
      shape[1] = t.size(1) / n_split;
      out.tensors_[key] = t.reshape(shape);
    } else {
      out.tensors_[key] = torch::repeat_interleave(t, n_split, /*dim=*/0);
    }
  }
  return out;
}

TensorBatch& TensorBatch::to(torch::Device device) {
  for (auto& kv : tensors_) {
    kv.second = kv.second.to(device);
  }
  return *this;
}

}  // namespace verl_data_proto
