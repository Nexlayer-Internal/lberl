#!/usr/bin/env bash
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
#
# Downloads the real Qwen/Qwen3-4B tokenizer files (tokenizer.json,
# tokenizer_config.json) -- just the tokenizer, not the ~8GB of model
# weights -- into tests/fixtures/qwen3_tokenizer/. Needed by
# gen_fixtures.py (to build real reference fixtures) and by
# validate_continuous_token.cpp (to exercise the real C++/Rust port against
# the same files). Not committed to the repo (11MB+ binary tokenizer.json;
# see .gitignore) -- run this once before running the tests.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$HERE/fixtures/qwen3_tokenizer"
mkdir -p "$OUT_DIR"

for f in tokenizer.json tokenizer_config.json; do
  echo "Fetching $f ..."
  curl -sL -o "$OUT_DIR/$f" "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/$f"
done

python3 - "$OUT_DIR" <<'EOF'
import json
import sys

out_dir = sys.argv[1]
with open(f"{out_dir}/tokenizer_config.json") as f:
    cfg = json.load(f)
with open(f"{out_dir}/chat_template.jinja", "w") as f:
    f.write(cfg["chat_template"])
print(f"Wrote {out_dir}/chat_template.jinja ({len(cfg['chat_template'])} bytes)")
EOF

echo "Done. Tokenizer files in: $OUT_DIR"
