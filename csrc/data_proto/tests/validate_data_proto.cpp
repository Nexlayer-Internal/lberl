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
// Reads the text fixtures written by gen_fixtures.py (real Python
// `verl.protocol.DataProto` inputs + reference outputs) and checks the
// libtorch port in verl_data_proto::TensorBatch reproduces them within a
// tight numerical tolerance.
//
// Usage: validate_data_proto <fixtures_dir>
// Exit code 0 if every scenario passes, nonzero otherwise. Prints
// max-abs-diff per scenario either way.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "verl_data_proto/tensor_batch.h"

namespace fs = std::filesystem;
using verl_data_proto::TensorBatch;

namespace {

constexpr double kTol = 1e-5;  // float32 reference vs libtorch float32 compute

bool g_any_failed = false;

struct Fixture {
  std::unordered_map<std::string, torch::Tensor> tensors;
  std::unordered_map<std::string, double> scalars;
  std::unordered_map<std::string, std::vector<int64_t>> ints;
  std::unordered_map<std::string, std::vector<std::string>> keys;

  const torch::Tensor& t(const std::string& name) const { return tensors.at(name); }
  double s(const std::string& name) const { return scalars.at(name); }
  const std::vector<int64_t>& i(const std::string& name) const { return ints.at(name); }
  const std::vector<std::string>& k(const std::string& name) const { return keys.at(name); }
};

Fixture load_fixture(const fs::path& path) {
  std::ifstream in(path);
  TORCH_CHECK(in.is_open(), "failed to open fixture: ", path.string());
  Fixture fx;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream header(line);
    std::string kind;
    header >> kind;
    if (kind == "TENSOR") {
      std::string name;
      int ndim;
      header >> name >> ndim;
      std::vector<int64_t> shape(ndim);
      for (auto& d : shape) header >> d;
      int64_t numel;
      header >> numel;
      std::string data_line;
      std::getline(in, data_line);
      std::istringstream data(data_line);
      std::vector<float> values(numel);
      for (int64_t k = 0; k < numel; ++k) {
        double v;
        data >> v;
        values[k] = static_cast<float>(v);
      }
      torch::Tensor t = torch::from_blob(values.data(), shape, torch::kFloat).clone();
      fx.tensors[name] = t;
    } else if (kind == "SCALAR") {
      std::string name;
      double v;
      header >> name >> v;
      fx.scalars[name] = v;
    } else if (kind == "INTS") {
      std::string name;
      int64_t n;
      header >> name >> n;
      std::string data_line;
      std::getline(in, data_line);
      std::istringstream data(data_line);
      std::vector<int64_t> values(n);
      for (auto& v : values) data >> v;
      fx.ints[name] = values;
    } else if (kind == "KEYS") {
      std::string name;
      int64_t n;
      header >> name >> n;
      std::vector<std::string> values(n);
      if (n > 0) {
        std::string data_line;
        std::getline(in, data_line);
        std::istringstream data(data_line);
        for (auto& v : values) data >> v;
      } else {
        // gen_fixtures.py still writes an (empty) data line for n==0.
        std::string data_line;
        std::getline(in, data_line);
      }
      fx.keys[name] = values;
    } else {
      TORCH_CHECK(false, "unknown fixture record kind: ", kind);
    }
  }
  return fx;
}

// Reconstruct a TensorBatch from a `dump_batch(f, name, data)`-style
// fixture block: a KEYS record at `{name}_keys`, a SCALAR at
// `{name}_batch_size`, and one TENSOR per key at `{name}__{key}`.
TensorBatch load_batch(const Fixture& fx, const std::string& name) {
  const auto& keys = fx.k(name + "_keys");
  int64_t batch_size = static_cast<int64_t>(fx.s(name + "_batch_size"));
  std::unordered_map<std::string, torch::Tensor> tensors;
  for (const auto& key : keys) {
    tensors[key] = fx.t(name + "__" + key);
  }
  return TensorBatch::from_dict(std::move(tensors), batch_size);
}

double max_abs_diff(const torch::Tensor& a, const torch::Tensor& b) {
  TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch: ", a.sizes(), " vs ", b.sizes());
  if (a.numel() == 0) return 0.0;  // torch::max() requires a dim for empty tensors
  return (a.to(torch::kFloat64) - b.to(torch::kFloat64)).abs().max().item<double>();
}

void report(const std::string& scenario, double diff, double tol = kTol) {
  bool ok = std::isfinite(diff) && diff <= tol;
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << scenario << "  max_abs_diff=" << diff << " (tol=" << tol << ")"
            << std::endl;
  if (!ok) g_any_failed = true;
}

void report_bool(const std::string& scenario, bool ok, const std::string& detail = "") {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << scenario << (detail.empty() ? "" : ("  " + detail)) << std::endl;
  if (!ok) g_any_failed = true;
}

// Checks a loaded TensorBatch against the fixture's reference batch at
// `ref_name`: same key set, same batch_size, and every tensor within
// tolerance.
void check_batch(const std::string& scenario, const TensorBatch& actual, const Fixture& fx,
                  const std::string& ref_name) {
  const auto& ref_keys_vec = fx.k(ref_name + "_keys");
  std::set<std::string> ref_keys(ref_keys_vec.begin(), ref_keys_vec.end());
  std::set<std::string> actual_keys;
  for (const auto& kv : actual.tensors()) actual_keys.insert(kv.first);

  if (ref_keys != actual_keys) {
    std::cout << "[FAIL] " << scenario << " KEY MISMATCH (expected " << ref_keys_vec.size() << " keys)" << std::endl;
    g_any_failed = true;
    return;
  }

  int64_t ref_batch_size = static_cast<int64_t>(fx.s(ref_name + "_batch_size"));
  report_bool(scenario + " (batch_size)", actual.batch_size() == ref_batch_size,
              "got=" + std::to_string(actual.batch_size()) + " want=" + std::to_string(ref_batch_size));

  for (const auto& key : ref_keys_vec) {
    double diff = max_abs_diff(actual.at(key), fx.t(ref_name + "__" + key));
    report(scenario + " [" + key + "]", diff);
  }
}

// ---------------------------------------------------------------------------

void check_select(const fs::path& dir) {
  {
    Fixture fx = load_fixture(dir / "select_subset.txt");
    TensorBatch in = load_batch(fx, "in");
    TensorBatch out = in.select(fx.k("select_keys"));
    check_batch("select_subset", out, fx, "out");
  }
  {
    Fixture fx = load_fixture(dir / "select_single_key.txt");
    TensorBatch in = load_batch(fx, "in");
    TensorBatch out = in.select(fx.k("select_keys"));
    check_batch("select_single_key", out, fx, "out");
  }
}

void check_select_idxs(const fs::path& dir) {
  {
    Fixture fx = load_fixture(dir / "select_idxs_int_with_repeat.txt");
    TensorBatch in = load_batch(fx, "in");
    torch::Tensor idxs = torch::tensor(fx.i("idxs"), torch::kLong);
    TensorBatch out = in.select_idxs(idxs);
    check_batch("select_idxs_int_with_repeat", out, fx, "out");
  }
  {
    Fixture fx = load_fixture(dir / "select_idxs_bool_mask.txt");
    TensorBatch in = load_batch(fx, "in");
    std::vector<int64_t> mask_ints = fx.i("mask");
    std::vector<bool> mask_bools(mask_ints.size());
    for (size_t i = 0; i < mask_ints.size(); ++i) mask_bools[i] = mask_ints[i] != 0;
    torch::Tensor mask = torch::empty({(int64_t)mask_bools.size()}, torch::kBool);
    for (size_t i = 0; i < mask_bools.size(); ++i) mask[i] = static_cast<bool>(mask_bools[i]);
    TensorBatch out = in.select_idxs(mask);
    check_batch("select_idxs_bool_mask", out, fx, "out");
  }
  {
    Fixture fx = load_fixture(dir / "select_idxs_empty.txt");
    TensorBatch in = load_batch(fx, "in");
    torch::Tensor idxs = torch::tensor(fx.i("idxs"), torch::kLong);
    TensorBatch out = in.select_idxs(idxs);
    check_batch("select_idxs_empty", out, fx, "out");
  }
}

void check_slice(const fs::path& dir) {
  {
    Fixture fx = load_fixture(dir / "slice_basic.txt");
    TensorBatch in = load_batch(fx, "in");
    check_batch("slice_basic", in.slice(1, 4), fx, "out");
  }
  {
    Fixture fx = load_fixture(dir / "slice_negative_start.txt");
    TensorBatch in = load_batch(fx, "in");
    check_batch("slice_negative_start", in.slice(-3, std::nullopt), fx, "out");
  }
  {
    Fixture fx = load_fixture(dir / "slice_step2.txt");
    TensorBatch in = load_batch(fx, "in");
    check_batch("slice_step2", in.slice(0, std::nullopt, 2), fx, "out");
  }
  {
    Fixture fx = load_fixture(dir / "slice_single_row.txt");
    TensorBatch in = load_batch(fx, "in");
    check_batch("slice_single_row", in.slice(2, 3), fx, "out");
  }
}

void check_pop(const fs::path& dir) {
  Fixture fx = load_fixture(dir / "pop_subset.txt");
  TensorBatch in = load_batch(fx, "in");
  TensorBatch popped = in.pop({"mask"});
  check_batch("pop_subset (remaining)", in, fx, "remaining");
  check_batch("pop_subset (popped)", popped, fx, "popped");
}

void check_rename(const fs::path& dir) {
  {
    Fixture fx = load_fixture(dir / "rename_basic.txt");
    TensorBatch in = load_batch(fx, "in");
    in.rename({"obs", "mask"}, {"observation", "attn_mask"});
    check_batch("rename_basic", in, fx, "out");
  }
  {
    Fixture fx = load_fixture(dir / "rename_single_string.txt");
    TensorBatch in = load_batch(fx, "in");
    in.rename({"x"}, {"y"});
    check_batch("rename_single_string", in, fx, "out");
  }
}

void check_union(const fs::path& dir) {
  {
    Fixture fx = load_fixture(dir / "union_disjoint.txt");
    TensorBatch a = load_batch(fx, "a");
    TensorBatch b = load_batch(fx, "b");
    a.union_with(b);
    check_batch("union_disjoint", a, fx, "out");
  }
  {
    Fixture fx = load_fixture(dir / "union_overlap_equal.txt");
    TensorBatch a = load_batch(fx, "a");
    TensorBatch b = load_batch(fx, "b");
    a.union_with(b);
    check_batch("union_overlap_equal", a, fx, "out");
  }
  // Not a Python fixture (the point of this check is exception behavior --
  // see README.md "Deviations from literal Python behavior"): a conflicting
  // overlapping key must make union_with throw, matching
  // `assert tensor_dict1[key].equal(tensor_dict2[key])` in
  // `union_tensor_dict`.
  {
    TensorBatch a = TensorBatch::from_dict({{"shared", torch::ones({3, 2})}});
    TensorBatch b = TensorBatch::from_dict({{"shared", torch::zeros({3, 2})}});
    bool threw = false;
    try {
      a.union_with(b);
    } catch (const c10::Error&) {
      threw = true;
    }
    report_bool("union_conflicting_values_throws", threw);
  }
}

void check_concat(const fs::path& dir) {
  {
    Fixture fx = load_fixture(dir / "concat_differently_shaped.txt");
    TensorBatch a = load_batch(fx, "a");
    TensorBatch b = load_batch(fx, "b");
    TensorBatch c = load_batch(fx, "c");
    TensorBatch out = TensorBatch::concat({a, b, c});
    check_batch("concat_differently_shaped", out, fx, "out");
  }
  {
    Fixture fx = load_fixture(dir / "concat_single_key.txt");
    TensorBatch a = load_batch(fx, "a");
    TensorBatch b = load_batch(fx, "b");
    TensorBatch out = TensorBatch::concat({a, b});
    check_batch("concat_single_key", out, fx, "out");
  }
}

void check_chunk(const fs::path& dir) {
  {
    Fixture fx = load_fixture(dir / "chunk_even.txt");
    TensorBatch in = load_batch(fx, "in");
    auto chunks = in.chunk(3);
    report_bool("chunk_even (count)", (int64_t)chunks.size() == (int64_t)fx.s("num_chunks"));
    for (size_t i = 0; i < chunks.size(); ++i) {
      check_batch("chunk_even[" + std::to_string(i) + "]", chunks[i], fx, "chunk" + std::to_string(i));
    }
  }
  {
    Fixture fx = load_fixture(dir / "chunk_uneven.txt");
    TensorBatch in = load_batch(fx, "in");
    auto chunks = in.chunk(3, /*require_equal_chunks=*/false);
    report_bool("chunk_uneven (count)", (int64_t)chunks.size() == (int64_t)fx.s("num_chunks"));
    for (size_t i = 0; i < chunks.size(); ++i) {
      check_batch("chunk_uneven[" + std::to_string(i) + "]", chunks[i], fx, "chunk" + std::to_string(i));
    }
  }
}

void check_split(const fs::path& dir) {
  Fixture fx = load_fixture(dir / "split_uneven.txt");
  TensorBatch in = load_batch(fx, "in");
  auto pieces = in.split(3);
  report_bool("split_uneven (count)", (int64_t)pieces.size() == (int64_t)fx.s("num_pieces"));
  for (size_t i = 0; i < pieces.size(); ++i) {
    check_batch("split_uneven[" + std::to_string(i) + "]", pieces[i], fx, "piece" + std::to_string(i));
  }
}

void check_reorder(const fs::path& dir) {
  Fixture fx = load_fixture(dir / "reorder_with_repeats.txt");
  TensorBatch in = load_batch(fx, "in");
  torch::Tensor indices = torch::tensor(fx.i("indices"), torch::kLong);
  in.reorder(indices);
  check_batch("reorder_with_repeats", in, fx, "out");
}

void check_repeat(const fs::path& dir) {
  {
    Fixture fx = load_fixture(dir / "repeat_interleave.txt");
    TensorBatch in = load_batch(fx, "in");
    check_batch("repeat_interleave", in.repeat(3, /*interleave=*/true), fx, "out");
  }
  {
    Fixture fx = load_fixture(dir / "repeat_tile.txt");
    TensorBatch in = load_batch(fx, "in");
    check_batch("repeat_tile", in.repeat(3, /*interleave=*/false), fx, "out");
  }
}

void check_sample_level_repeat(const fs::path& dir) {
  Fixture fx = load_fixture(dir / "sample_level_repeat_with_zero.txt");
  TensorBatch in = load_batch(fx, "in");
  TensorBatch out = in.sample_level_repeat(fx.i("repeat_times"));
  check_batch("sample_level_repeat_with_zero", out, fx, "out");
}

void check_unfold_column_chunks(const fs::path& dir) {
  Fixture fx = load_fixture(dir / "unfold_column_chunks_basic.txt");
  TensorBatch in = load_batch(fx, "in");
  int64_t n_split = static_cast<int64_t>(fx.s("n_split"));
  TensorBatch out = in.unfold_column_chunks(n_split, fx.k("split_keys"));
  check_batch("unfold_column_chunks_basic", out, fx, "out");
}

void check_to(const fs::path& dir) {
  Fixture fx = load_fixture(dir / "to_cpu_noop.txt");
  TensorBatch in = load_batch(fx, "in");
  in.to(torch::kCPU);
  check_batch("to_cpu_noop", in, fx, "out");
}

// Not Python-fixture-based (see tensor_batch.h: check_consistency is a
// validator, not a value-producing transform -- there is no "reference
// output" beyond the pass/fail boundary already encoded in its own assert
// conditions, which this directly exercises).
void check_consistency_invariant() {
  bool threw_on_mismatch = false;
  try {
    TensorBatch::from_dict({{"a", torch::zeros({4, 2})}, {"b", torch::zeros({5, 2})}});
  } catch (const c10::Error&) {
    threw_on_mismatch = true;
  }
  report_bool("check_consistency_rejects_mismatched_batch_dims", threw_on_mismatch);

  bool ok_on_match = true;
  try {
    TensorBatch ok = TensorBatch::from_dict({{"a", torch::zeros({4, 2})}, {"b", torch::zeros({4, 3})}});
    ok.check_consistency();
  } catch (const c10::Error&) {
    ok_on_match = false;
  }
  report_bool("check_consistency_accepts_matching_batch_dims", ok_on_match);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: validate_data_proto <fixtures_dir>" << std::endl;
    return 2;
  }
  fs::path dir(argv[1]);

  check_consistency_invariant();
  check_select(dir);
  check_select_idxs(dir);
  check_slice(dir);
  check_pop(dir);
  check_rename(dir);
  check_union(dir);
  check_concat(dir);
  check_chunk(dir);
  check_split(dir);
  check_reorder(dir);
  check_repeat(dir);
  check_sample_level_repeat(dir);
  check_unfold_column_chunks(dir);
  check_to(dir);

  if (g_any_failed) {
    std::cout << "\nSOME SCENARIOS FAILED" << std::endl;
    return 1;
  }
  std::cout << "\nALL SCENARIOS PASSED" << std::endl;
  return 0;
}
