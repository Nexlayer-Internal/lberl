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
// CUDA-tensor smoke test for verl_data_proto::TensorBatch.
//
// Background (see csrc/core_algos/README.md's "CUDA fix (post-review)" and
// csrc/rollout_corr/README.md's "CUDA-safety audit"): a previous port in
// this project called `Tensor::accessor<T,N>()` directly on a tensor that
// could be a CUDA tensor, which is a CPU-only API -- it segfaulted (or
// silently corrupted memory) the first time it actually ran against a GPU,
// despite passing all CPU fixture comparisons. This file exists specifically
// so that bug class gets a standing regression test for this port too: it
// constructs CUDA tensors, runs them through every ported operation that
// constructs or returns a TensorBatch, and cross-checks the result against
// an identical-input CPU run.
//
// CUDA-safety audit for this port (see also README.md): `grep -n
// '\.accessor<\|\.data_ptr<' csrc/data_proto/src/*.cpp` has zero hits. Every
// raw-value extraction in tensor_batch.cpp is either (a) a standard
// device-dispatched ATen op (`index_select`, `index`, `narrow`, `reshape`,
// `cat`, `chunk`, `repeat_interleave`, `to`, etc. -- these dispatch
// correctly on CUDA tensors by construction), or (b) `.item<int64_t>()` on a
// 0-dim reduction tensor (`select_idxs`'s bool-mask branch sums the mask to
// get the output batch_size) or `torch::equal(...)` (`union_with`'s
// conflicting-key check) -- both are dispatched ops that perform their own
// device-to-host synchronization internally, the same safe pattern
// core_algos.cpp's post-fix code and rollout_corr's own audit rely on.
// `check_consistency`/`from_dict`'s `tensor.size(0)` calls are tensor
// *metadata* queries (shape, not data), which are always safe regardless of
// device -- they never touch device memory at all.
//
// IMPORTANT / honesty note (see also csrc/data_proto/README.md): this file
// was written and syntax/API-checked by compiling it against a CPU-only
// libtorch build -- no GPU was available in the environment this port was
// developed in. On that CPU-only build, `torch::cuda::is_available()`
// correctly reports `false` and the test exits early with code 77 (skipped)
// rather than silently reporting success -- this is NOT a substitute for
// running it on real CUDA hardware, only a guard against misreporting a
// skip as a pass. Per this project's established process, the coordinator
// should run this on real GPU hardware (e.g. IP66) before merging.
//
// Exit codes: 0 = pass, 1 = fail (mismatch or crash-adjacent assertion
// failure), 2 = usage error, 77 = skipped (no CUDA available).

#include <iostream>

#include "verl_data_proto/tensor_batch.h"

using verl_data_proto::TensorBatch;

namespace {

constexpr double kTol = 1e-5;

bool g_any_failed = false;

void report(const std::string& name, double diff, double tol = kTol) {
  bool ok = std::isfinite(diff) && diff <= tol;
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "  max_abs_diff=" << diff << " (tol=" << tol << ")"
            << std::endl;
  if (!ok) g_any_failed = true;
}

double max_abs_diff(const torch::Tensor& a, const torch::Tensor& b) {
  if (a.numel() == 0) return 0.0;  // torch::max() requires a dim for empty tensors
  return (a.to(torch::kCPU).to(torch::kFloat64) - b.to(torch::kCPU).to(torch::kFloat64)).abs().max().item<double>();
}

void check_same_batch(const std::string& name, const TensorBatch& cuda_batch, const TensorBatch& cpu_batch) {
  if (cuda_batch.batch_size() != cpu_batch.batch_size()) {
    std::cout << "[FAIL] " << name << " batch_size mismatch: cuda=" << cuda_batch.batch_size()
              << " cpu=" << cpu_batch.batch_size() << std::endl;
    g_any_failed = true;
    return;
  }
  for (const auto& kv : cpu_batch.tensors()) {
    TORCH_CHECK(cuda_batch.at(kv.first).device().is_cuda(), "expected '", kv.first, "' to stay on a CUDA device");
    report(name + " [" + kv.first + "]", max_abs_diff(cuda_batch.at(kv.first), kv.second));
  }
}

}  // namespace

int main() {
  if (!torch::cuda::is_available()) {
    std::cout << "[SKIP] CUDA not available in this environment -- this test must be run on a "
                 "CUDA-enabled libtorch + real GPU to actually validate CUDA execution (see file header)."
              << std::endl;
    return 77;
  }

  const int64_t bsz = 8;
  const torch::Device cuda_device(torch::kCUDA, 0);

  std::unordered_map<std::string, torch::Tensor> cpu_tensors;
  cpu_tensors["obs"] = torch::randn({bsz, 5});
  cpu_tensors["logprob"] = torch::randn({bsz, 8});
  cpu_tensors["mask"] = (torch::rand({bsz, 8}) > 0.3).to(torch::kFloat);

  TensorBatch cpu_batch = TensorBatch::from_dict(cpu_tensors);

  // Build an independent CUDA copy from the same host-side values (not a
  // `.to()` view of the CPU batch), mirroring core_algos's own smoke-test
  // pattern of a fully independent reference pass.
  std::unordered_map<std::string, torch::Tensor> cuda_tensors;
  for (const auto& kv : cpu_tensors) cuda_tensors[kv.first] = kv.second.to(cuda_device);
  TensorBatch cuda_batch = TensorBatch::from_dict(cuda_tensors);

  std::cout << "[INFO] TensorBatch constructed from real CUDA tensors without crashing." << std::endl;

  // to(device): round-trip CPU -> CUDA -> CPU should reproduce the
  // original CPU values exactly (float32, no compute in between).
  {
    TensorBatch moved = TensorBatch::from_dict(cpu_tensors);
    moved.to(cuda_device);
    for (const auto& kv : moved.tensors()) {
      TORCH_CHECK(kv.second.device().is_cuda(), "to(cuda): expected '", kv.first, "' to be on a CUDA device");
    }
    moved.to(torch::kCPU);
    for (const auto& kv : cpu_batch.tensors()) {
      report("to_roundtrip [" + kv.first + "]", max_abs_diff(moved.at(kv.first), kv.second), /*tol=*/0.0);
    }
  }

  // select
  {
    TensorBatch cuda_out = cuda_batch.select({"obs", "mask"});
    TensorBatch cpu_out = cpu_batch.select({"obs", "mask"});
    check_same_batch("select", cuda_out, cpu_out);
  }

  // select_idxs (int, with a repeated index -- the exact shape of indexing
  // that would have hit an accessor<T,N>()-style CPU-only bug if this port
  // had one)
  {
    torch::Tensor idxs_cpu = torch::tensor({0, 2, 2, 5}, torch::kLong);
    torch::Tensor idxs_cuda = idxs_cpu.to(cuda_device);
    TensorBatch cuda_out = cuda_batch.select_idxs(idxs_cuda);
    TensorBatch cpu_out = cpu_batch.select_idxs(idxs_cpu);
    check_same_batch("select_idxs_int", cuda_out, cpu_out);
  }

  // select_idxs (bool mask) -- exercises the `.item<int64_t>()` scalar-sum
  // path for computing batch_size from a CUDA bool tensor.
  {
    torch::Tensor mask_cpu = torch::tensor({true, false, true, false, true, true, false, true});
    torch::Tensor mask_cuda = mask_cpu.to(cuda_device);
    TensorBatch cuda_out = cuda_batch.select_idxs(mask_cuda);
    TensorBatch cpu_out = cpu_batch.select_idxs(mask_cpu);
    check_same_batch("select_idxs_bool", cuda_out, cpu_out);
  }

  // slice
  {
    TensorBatch cuda_out = cuda_batch.slice(2, 6);
    TensorBatch cpu_out = cpu_batch.slice(2, 6);
    check_same_batch("slice", cuda_out, cpu_out);
  }

  // pop (mutates in place)
  {
    TensorBatch cuda_copy = TensorBatch::from_dict(cuda_tensors);
    TensorBatch cpu_copy = TensorBatch::from_dict(cpu_tensors);
    TensorBatch cuda_popped = cuda_copy.pop({"mask"});
    TensorBatch cpu_popped = cpu_copy.pop({"mask"});
    check_same_batch("pop (remaining)", cuda_copy, cpu_copy);
    check_same_batch("pop (popped)", cuda_popped, cpu_popped);
  }

  // rename (mutates in place)
  {
    TensorBatch cuda_copy = TensorBatch::from_dict(cuda_tensors);
    TensorBatch cpu_copy = TensorBatch::from_dict(cpu_tensors);
    cuda_copy.rename({"obs"}, {"observation"});
    cpu_copy.rename({"obs"}, {"observation"});
    check_same_batch("rename", cuda_copy, cpu_copy);
  }

  // union_with -- exercises the `torch::equal(...)` conflicting-key check
  // on CUDA tensors.
  {
    torch::Tensor shared_cpu = torch::randn({bsz, 3});
    torch::Tensor shared_cuda = shared_cpu.to(cuda_device);
    TensorBatch a_cpu = TensorBatch::from_dict({{"a", torch::randn({bsz, 2})}, {"shared", shared_cpu}});
    TensorBatch b_cpu = TensorBatch::from_dict({{"shared", shared_cpu.clone()}, {"c", torch::randn({bsz, 1})}});
    TensorBatch a_cuda = TensorBatch::from_dict(
        {{"a", a_cpu.at("a").to(cuda_device)}, {"shared", shared_cuda}});
    TensorBatch b_cuda = TensorBatch::from_dict(
        {{"shared", shared_cuda.clone()}, {"c", b_cpu.at("c").to(cuda_device)}});
    a_cpu.union_with(b_cpu);
    a_cuda.union_with(b_cuda);
    check_same_batch("union_with", a_cuda, a_cpu);
  }

  // concat
  {
    TensorBatch cuda_out = TensorBatch::concat({cuda_batch, cuda_batch});
    TensorBatch cpu_out = TensorBatch::concat({cpu_batch, cpu_batch});
    check_same_batch("concat", cuda_out, cpu_out);
  }

  // chunk
  {
    auto cuda_chunks = cuda_batch.chunk(4);
    auto cpu_chunks = cpu_batch.chunk(4);
    for (size_t i = 0; i < cpu_chunks.size(); ++i) {
      check_same_batch("chunk[" + std::to_string(i) + "]", cuda_chunks[i], cpu_chunks[i]);
    }
  }

  // reorder (mutates in place, with a repeated index)
  {
    torch::Tensor idx_cpu = torch::tensor({3, 0, 0, 5, 1, 2, 7, 4}, torch::kLong);
    torch::Tensor idx_cuda = idx_cpu.to(cuda_device);
    TensorBatch cuda_copy = TensorBatch::from_dict(cuda_tensors);
    TensorBatch cpu_copy = TensorBatch::from_dict(cpu_tensors);
    cuda_copy.reorder(idx_cuda);
    cpu_copy.reorder(idx_cpu);
    check_same_batch("reorder", cuda_copy, cpu_copy);
  }

  // repeat (interleave and tile)
  {
    TensorBatch cuda_out = cuda_batch.repeat(2, /*interleave=*/true);
    TensorBatch cpu_out = cpu_batch.repeat(2, /*interleave=*/true);
    check_same_batch("repeat_interleave", cuda_out, cpu_out);
  }
  {
    TensorBatch cuda_out = cuda_batch.repeat(2, /*interleave=*/false);
    TensorBatch cpu_out = cpu_batch.repeat(2, /*interleave=*/false);
    check_same_batch("repeat_tile", cuda_out, cpu_out);
  }

  // sample_level_repeat (per-row counts including a zero)
  {
    std::vector<int64_t> counts = {0, 1, 2, 0, 3, 1, 1, 0};
    TensorBatch cuda_out = cuda_batch.sample_level_repeat(counts);
    TensorBatch cpu_out = cpu_batch.sample_level_repeat(counts);
    check_same_batch("sample_level_repeat", cuda_out, cpu_out);
  }

  // unfold_column_chunks
  {
    TensorBatch cuda_out = cuda_batch.unfold_column_chunks(2, std::vector<std::string>{"mask"});
    TensorBatch cpu_out = cpu_batch.unfold_column_chunks(2, std::vector<std::string>{"mask"});
    check_same_batch("unfold_column_chunks", cuda_out, cpu_out);
  }

  if (g_any_failed) {
    std::cout << "\nSOME CUDA SMOKE CHECKS FAILED" << std::endl;
    return 1;
  }
  std::cout << "\nALL CUDA SMOKE CHECKS PASSED" << std::endl;
  return 0;
}
