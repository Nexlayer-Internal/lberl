#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate validation fixtures for the C++ port of the tensor-batch half of
`verl/protocol.py`'s `DataProto` (PLAN.md table item #7 / section 6, item 3).

Runs the REAL `verl.protocol.DataProto` (imported directly from this
checkout, no reimplementation) across labeled scenarios covering every
ported operation (`check_consistency` excepted -- see README.md for why that
one is validated with direct C++ assertions instead of a Python fixture) and
dumps inputs + reference outputs as plain text, which the C++ validation
harness (validate_data_proto.cpp) reads back and compares against the
libtorch port bit-for-bit (within float tolerance).

Only the tensor half (`batch`) of each DataProto is ever touched here --
every scenario constructs batches via `DataProto.from_dict(tensors=...)`
with no `non_tensors`/`meta_info` beyond the one scenario that needs
`auto_padding=True` to reach the real uneven-chunk code path (see
`chunk_uneven` below), matching this port's documented scope boundary.
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from verl.protocol import DataProto  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "fixtures"
OUT_DIR.mkdir(exist_ok=True)


def dump_tensor(f, name, t):
    t = t.detach().to(torch.float64).contiguous().cpu()
    flat = t.reshape(-1).tolist()
    shape = list(t.shape)
    f.write(f"TENSOR {name} {len(shape)} {' '.join(str(s) for s in shape)} {len(flat)}\n")
    f.write(" ".join(repr(v) for v in flat) + "\n")


def dump_scalar(f, name, v):
    f.write(f"SCALAR {name} {repr(float(v))}\n")


def dump_ints(f, name, values):
    f.write(f"INTS {name} {len(values)}\n")
    f.write(" ".join(str(int(v)) for v in values) + "\n")


def dump_keys(f, name, keys):
    f.write(f"KEYS {name} {len(keys)}\n")
    f.write(" ".join(keys) + "\n")


def dump_batch(f, name, data: DataProto):
    """Dump a DataProto's tensor batch: its key list, batch size, and every
    tensor, keyed as `{name}__{key}` so the C++ harness can reconstruct the
    full `std::unordered_map<std::string, torch::Tensor>`."""
    keys = sorted(data.batch.keys()) if data.batch is not None else []
    dump_keys(f, f"{name}_keys", keys)
    dump_scalar(f, f"{name}_batch_size", len(data))
    for key in keys:
        dump_tensor(f, f"{name}__{key}", data.batch[key])


def make_base(bsz=6):
    """Multi-key realistic batch: 'obs' (bsz,5), 'logprob' (bsz,8), 'mask' (bsz,8)."""
    torch.manual_seed(0)
    return DataProto.from_dict(
        tensors={
            "obs": torch.randn(bsz, 5),
            "logprob": torch.randn(bsz, 8),
            "mask": (torch.rand(bsz, 8) > 0.3).float(),
        }
    )


def make_single_key(bsz=5):
    torch.manual_seed(1)
    return DataProto.from_dict(tensors={"x": torch.randn(bsz, 4)})


def write_scenario(filename, writer_fn):
    path = OUT_DIR / filename
    with open(path, "w") as f:
        writer_fn(f)
    print(f"wrote {path.name}")


def main():
    # --- select ---
    def w(f):
        base = make_base()
        dump_batch(f, "in", base)
        out = base.select(batch_keys=["obs", "mask"])
        dump_keys(f, "select_keys", ["obs", "mask"])
        dump_batch(f, "out", out)

    write_scenario("select_subset.txt", w)

    def w(f):
        single = make_single_key()
        dump_batch(f, "in", single)
        out = single.select(batch_keys=["x"])
        dump_keys(f, "select_keys", ["x"])
        dump_batch(f, "out", out)

    write_scenario("select_single_key.txt", w)

    # --- select_idxs ---
    def w(f):
        base = make_base()
        dump_batch(f, "in", base)
        idxs = torch.tensor([0, 2, 2, 5], dtype=torch.int64)
        dump_ints(f, "idxs", idxs.tolist())
        out = base.select_idxs(idxs)
        dump_batch(f, "out", out)

    write_scenario("select_idxs_int_with_repeat.txt", w)

    def w(f):
        base = make_base()
        dump_batch(f, "in", base)
        mask = torch.tensor([True, False, True, False, True, True])
        dump_ints(f, "mask", [1 if m else 0 for m in mask.tolist()])
        out = base.select_idxs(mask)
        dump_batch(f, "out", out)

    write_scenario("select_idxs_bool_mask.txt", w)

    def w(f):
        base = make_base()
        dump_batch(f, "in", base)
        idxs = torch.tensor([], dtype=torch.int64)
        dump_ints(f, "idxs", [])
        out = base.select_idxs(idxs)
        dump_batch(f, "out", out)

    write_scenario("select_idxs_empty.txt", w)

    # --- slice ---
    def w(f):
        base = make_base()
        dump_batch(f, "in", base)
        out = base.slice(1, 4)
        dump_batch(f, "out", out)

    write_scenario("slice_basic.txt", w)

    def w(f):
        base = make_base()
        dump_batch(f, "in", base)
        out = base.slice(-3, None)
        dump_batch(f, "out", out)

    write_scenario("slice_negative_start.txt", w)

    def w(f):
        base = make_base()
        dump_batch(f, "in", base)
        out = base.slice(0, None, 2)
        dump_batch(f, "out", out)

    write_scenario("slice_step2.txt", w)

    def w(f):
        base = make_base()
        dump_batch(f, "in", base)
        out = base.slice(2, 3)  # single-row result
        dump_batch(f, "out", out)

    write_scenario("slice_single_row.txt", w)

    # --- pop ---
    def w(f):
        base = make_base()
        dump_batch(f, "in", base)
        popped = base.pop(batch_keys=["mask"])
        dump_batch(f, "remaining", base)
        dump_batch(f, "popped", popped)

    write_scenario("pop_subset.txt", w)

    # --- rename ---
    # Honesty note (see README.md "Deviations from literal Python behavior,
    # documented"): `DataProto.rename(old_keys=[...], new_keys=[...])` with
    # 2+ keys in one call is actually broken in the real Python source --
    # `self.batch.rename_key_(tuple(old_keys), tuple(new_keys))` passes a
    # multi-element tuple as a single `NestedKey`, which tensordict
    # interprets as a *nested* key path (e.g. "look inside the 'obs' tensor
    # for an entry named 'mask'"), not as "rename these two keys
    # independently" -- confirmed by actually running
    # `base.rename(old_keys=["obs", "mask"], new_keys=[...])` against the
    # real tensordict dependency in this checkout, which raises
    # `_KeyThroughLeafError`. Only the single-key case
    # (`old_keys=[single_key]`) actually works. This fixture therefore
    # renames two keys via two separate, individually-valid single-key
    # `rename` calls (the only way to actually achieve a two-key rename
    # against the real Python source today) and dumps the combined final
    # state, which is what this port's one-call multi-pair `rename` is
    # checked against.
    def w(f):
        base = make_base()
        dump_batch(f, "in", base)
        base.rename(old_keys=["obs"], new_keys=["observation"])
        base.rename(old_keys=["mask"], new_keys=["attn_mask"])
        dump_batch(f, "out", base)

    write_scenario("rename_basic.txt", w)

    def w(f):
        single = make_single_key()
        dump_batch(f, "in", single)
        single.rename(old_keys="x", new_keys="y")
        dump_batch(f, "out", single)

    write_scenario("rename_single_string.txt", w)

    # --- union ---
    def w(f):
        torch.manual_seed(2)
        bsz = 4
        a = DataProto.from_dict(tensors={"a": torch.randn(bsz, 3)})
        b = DataProto.from_dict(tensors={"b": torch.randn(bsz, 2)})
        dump_batch(f, "a", a)
        dump_batch(f, "b", b)
        a.union(b)
        dump_batch(f, "out", a)

    write_scenario("union_disjoint.txt", w)

    def w(f):
        torch.manual_seed(3)
        bsz = 4
        shared = torch.randn(bsz, 2)
        a = DataProto.from_dict(tensors={"a": torch.randn(bsz, 3), "shared": shared})
        b = DataProto.from_dict(tensors={"shared": shared.clone(), "c": torch.randn(bsz, 1)})
        dump_batch(f, "a", a)
        dump_batch(f, "b", b)
        a.union(b)
        dump_batch(f, "out", a)

    write_scenario("union_overlap_equal.txt", w)

    # --- concat ---
    def w(f):
        torch.manual_seed(4)
        a = DataProto.from_dict(tensors={"x": torch.randn(3, 4), "y": torch.randn(3, 2)})
        b = DataProto.from_dict(tensors={"x": torch.randn(5, 4), "y": torch.randn(5, 2)})
        c = DataProto.from_dict(tensors={"x": torch.randn(2, 4), "y": torch.randn(2, 2)})
        dump_batch(f, "a", a)
        dump_batch(f, "b", b)
        dump_batch(f, "c", c)
        out = DataProto.concat([a, b, c])
        dump_batch(f, "out", out)

    write_scenario("concat_differently_shaped.txt", w)

    def w(f):
        torch.manual_seed(5)
        a = DataProto.from_dict(tensors={"x": torch.randn(2, 4)})
        b = DataProto.from_dict(tensors={"x": torch.randn(3, 4)})
        dump_batch(f, "a", a)
        dump_batch(f, "b", b)
        out = DataProto.concat([a, b])
        dump_batch(f, "out", out)

    write_scenario("concat_single_key.txt", w)

    # --- chunk ---
    def w(f):
        base = make_base()  # bsz=6
        dump_batch(f, "in", base)
        chunks = base.chunk(3)
        dump_scalar(f, "num_chunks", len(chunks))
        for i, c in enumerate(chunks):
            dump_batch(f, f"chunk{i}", c)

    write_scenario("chunk_even.txt", w)

    def w(f):
        torch.manual_seed(6)
        bsz = 10
        base = DataProto.from_dict(tensors={"x": torch.randn(bsz, 3)}, auto_padding=True)
        assert base.is_padding_enabled()
        dump_batch(f, "in", base)
        chunks = base.chunk(3)
        dump_scalar(f, "num_chunks", len(chunks))
        for i, c in enumerate(chunks):
            dump_batch(f, f"chunk{i}", c)

    write_scenario("chunk_uneven.txt", w)

    # --- split ---
    def w(f):
        torch.manual_seed(7)
        bsz = 10
        base = DataProto.from_dict(tensors={"x": torch.randn(bsz, 3)})
        dump_batch(f, "in", base)
        pieces = base.split(3)
        dump_scalar(f, "num_pieces", len(pieces))
        for i, p in enumerate(pieces):
            dump_batch(f, f"piece{i}", p)

    write_scenario("split_uneven.txt", w)

    # --- reorder ---
    def w(f):
        base = make_base()  # bsz=6
        dump_batch(f, "in", base)
        indices = torch.tensor([2, 0, 0, 4, 1, 3], dtype=torch.int64)
        dump_ints(f, "indices", indices.tolist())
        base.reorder(indices)
        dump_batch(f, "out", base)

    write_scenario("reorder_with_repeats.txt", w)

    # --- repeat ---
    def w(f):
        base = make_base()
        dump_batch(f, "in", base)
        out = base.repeat(repeat_times=3, interleave=True)
        dump_batch(f, "out", out)

    write_scenario("repeat_interleave.txt", w)

    def w(f):
        base = make_base()
        dump_batch(f, "in", base)
        out = base.repeat(repeat_times=3, interleave=False)
        dump_batch(f, "out", out)

    write_scenario("repeat_tile.txt", w)

    # --- sample_level_repeat ---
    def w(f):
        base = make_base()  # bsz=6
        dump_batch(f, "in", base)
        repeat_times = [0, 1, 2, 3, 1, 0]
        dump_ints(f, "repeat_times", repeat_times)
        out = base.sample_level_repeat(repeat_times)
        dump_batch(f, "out", out)

    write_scenario("sample_level_repeat_with_zero.txt", w)

    # --- unfold_column_chunks ---
    def w(f):
        torch.manual_seed(8)
        bsz = 4
        base = DataProto.from_dict(
            tensors={
                "grouped": torch.randn(bsz, 8),
                "meta": torch.randn(bsz, 2),
            }
        )
        dump_batch(f, "in", base)
        n_split = 2
        dump_scalar(f, "n_split", n_split)
        dump_keys(f, "split_keys", ["grouped"])
        out = base.unfold_column_chunks(n_split=n_split, split_keys=["grouped"])
        dump_batch(f, "out", out)

    write_scenario("unfold_column_chunks_basic.txt", w)

    # --- to(device) --- (CPU no-op: values must survive the round trip
    # unchanged; real device-transfer coverage is in cuda_smoke_test.cpp)
    def w(f):
        base = make_base()
        dump_batch(f, "in", base)
        base.to("cpu")
        dump_batch(f, "out", base)

    write_scenario("to_cpu_noop.txt", w)

    print(f"\nAll fixtures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
