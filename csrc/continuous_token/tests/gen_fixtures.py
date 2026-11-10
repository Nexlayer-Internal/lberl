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
"""Generate reference fixtures by calling the REAL, unmodified
verl/utils/tokenizer/{tokenizer,chat_template,continuous_token}.py against the
REAL downloaded Qwen/Qwen3-4B tokenizer files (see fetch_tokenizer.sh).

This does NOT reimplement any tokenization/chat-template logic -- it loads
the exact repo files via importlib and calls the exact, unmodified
`QwenContinuousTokenBuilder` (the class `create_continuous_token_builder`
actually selects for Qwen3 -- see
`verl/utils/tokenizer/continuous_token_wiring.py`). Two assertions below
confirm, at fixture-generation time, that this builder does NOT override
`build_initial_tokens` / `merge_assistant_tokens` / `align_response_metadata`
from the base `ContinuousTokenBuilder` -- the scope claim this whole C++ port
depends on (see ../README.md's scoping trace).

Why importlib instead of `import verl...`: `verl/__init__.py` imports
`.protocol`, which pulls in Ray/torch/tensordict -- none of which the three
leaf modules actually used here need (confirmed by reading them: stdlib +
`transformers` only). Loading the three files directly, registered under
their real dotted module names so their internal relative imports resolve,
calls the literal same source without requiring a multi-GB ML dependency
install just to validate token-id bookkeeping and chat-template rendering.
This is disclosed here, not hidden -- see README.md's honesty section.

Usage:
    ./fetch_tokenizer.sh   # one-time: downloads the real Qwen3-4B tokenizer
    python3 gen_fixtures.py
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TOK_DIR = os.path.join(HERE, "fixtures", "qwen3_tokenizer")
OUT_DIR = os.path.join(HERE, "fixtures")


def _install_pkg(name: str, path: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    mod.__package__ = name
    sys.modules[name] = mod
    return mod


def _load_real_module(dotted_name: str, file_path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(dotted_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


_install_pkg("verl", os.path.join(REPO_ROOT, "verl"))
_install_pkg("verl.utils", os.path.join(REPO_ROOT, "verl", "utils"))
_install_pkg("verl.utils.tokenizer", os.path.join(REPO_ROOT, "verl", "utils", "tokenizer"))

_load_real_module("verl.utils.tokenizer.tokenizer", os.path.join(REPO_ROOT, "verl", "utils", "tokenizer", "tokenizer.py"))
_load_real_module(
    "verl.utils.tokenizer.chat_template", os.path.join(REPO_ROOT, "verl", "utils", "tokenizer", "chat_template.py")
)
continuous_token_mod = _load_real_module(
    "verl.utils.tokenizer.continuous_token", os.path.join(REPO_ROOT, "verl", "utils", "tokenizer", "continuous_token.py")
)

QwenContinuousTokenBuilder = continuous_token_mod.QwenContinuousTokenBuilder
ContinuousTokenBuilder = continuous_token_mod.ContinuousTokenBuilder

try:
    from transformers import AutoTokenizer
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This script needs `transformers` (matching this repo's pin -- see requirements.txt: "
        "transformers>=5.5.3,!=5.6.0,<5.13) importable, e.g.:\n"
        "  uv venv --python 3.12 && source .venv/bin/activate && uv pip install 'transformers==5.12.1'"
    ) from e

if not os.path.isdir(TOK_DIR):
    raise SystemExit(f"{TOK_DIR} not found -- run ./fetch_tokenizer.sh first.")

tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
builder = QwenContinuousTokenBuilder(tokenizer)

# The scope claim this entire C++ port depends on: for the single-turn,
# text-only, no-tools Qwen3-4B/GSM8K workload, QwenContinuousTokenBuilder's
# only overrides (__init__, _merge_context_token_ids) are unreachable
# (SingleTurnAgentLoop never calls ct_merge_context_msg/merge_context_tokens).
# So the base ContinuousTokenBuilder's behavior for these three methods IS
# the Qwen behavior. Fail loudly at fixture-generation time if a future verl
# change makes this false, rather than silently generating fixtures for
# behavior the C++ port no longer matches.
assert type(builder).build_initial_tokens is ContinuousTokenBuilder.build_initial_tokens, (
    "QwenContinuousTokenBuilder now overrides build_initial_tokens -- the C++ port's scope claim "
    "(see README.md) no longer holds; the port needs to be revisited, not just these fixtures."
)
assert type(builder).merge_assistant_tokens is ContinuousTokenBuilder.merge_assistant_tokens, (
    "QwenContinuousTokenBuilder now overrides merge_assistant_tokens -- see README.md scope note above."
)
assert type(builder).align_response_metadata is ContinuousTokenBuilder.align_response_metadata, (
    "QwenContinuousTokenBuilder now overrides align_response_metadata -- see README.md scope note above."
)

GSM8K_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def b64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def write_lines(path: str, lines: list[str]) -> None:
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# === build_initial_tokens scenarios ===
# (messages, add_generation_prompt, enable_thinking_kwargs_or_None)
BUILD_INITIAL_TOKENS_SCENARIOS: dict[str, tuple[list[dict], dict]] = {
    # 1. Real GSM8K-shaped single-turn prompt: matches
    #    examples/data_preprocess/gsm8k.py exactly (one user message,
    #    question + " " + instruction_following).
    "gsm8k_basic": (
        [
            {
                "role": "user",
                "content": (
                    "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
                    "clips in May. How many clips did Natalia sell altogether in April and May? " + GSM8K_INSTRUCTION
                ),
            }
        ],
        {},
    ),
    # 2. Longer/edge-case GSM8K-shaped question: multi-sentence, numbers,
    #    punctuation, quotes, embedded newline (plausible raw-dataset noise).
    "gsm8k_long_edge_case": (
        [
            {
                "role": "user",
                "content": (
                    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of "
                    'babysitting. She said, "I need to calculate this carefully!"\n'
                    "Her rates are: $12/hr standard, $18/hr after 9pm, and a flat $5 bonus if she helps "
                    "with homework (which she did, for 'algebra' review).\n"
                    "How much did she earn? " + GSM8K_INSTRUCTION
                ),
            }
        ],
        {},
    ),
    # 3. System + user message history (not produced by gsm8k.py itself, but
    #    representative of the general chat-template-rendering path).
    "system_plus_user": (
        [
            {"role": "system", "content": "You are a helpful math tutor. Show your work."},
            {"role": "user", "content": "If a train travels 60 miles in 1.5 hours, what is its average speed?"},
        ],
        {},
    ),
    # 4. Multi-message history (user/assistant/user).
    "multi_turn_history": (
        [
            {"role": "user", "content": "What is 7 * 8?"},
            {"role": "assistant", "content": "7 * 8 = 56."},
            {"role": "user", "content": "Now add 10 to that."},
        ],
        {},
    ),
    # 5. Empty-content edge case.
    "empty_user_content": ([{"role": "user", "content": ""}], {}),
    # 6. Unicode / non-ASCII content.
    "unicode_content": (
        [
            {
                "role": "user",
                "content": "François has 3 pommes. Il en achète 2 de plus. Combien a-t-il de pommes au total? 😀",
            }
        ],
        {},
    ),
    # 7. enable_thinking=False (GSM8K's own launch script doesn't set this,
    #    but it's a real, documented Qwen3 chat-template kwarg; included for
    #    thoroughness of the chat-template-parity check).
    "enable_thinking_false": ([{"role": "user", "content": "What is 2+2?"}], {"enable_thinking": False}),
}

for name, (messages, kwargs) in BUILD_INITIAL_TOKENS_SCENARIOS.items():
    scenario_builder = QwenContinuousTokenBuilder(tokenizer, chat_template_kwargs=kwargs) if kwargs else builder
    token_ids = scenario_builder.build_initial_tokens(messages, tools=None)

    lines = [
        "ADD_GENERATION_PROMPT 1",
        f"ENABLE_THINKING_DEFINED {1 if 'enable_thinking' in kwargs else 0}",
        f"ENABLE_THINKING_VALUE {1 if kwargs.get('enable_thinking') else 0}",
        f"MESSAGES {len(messages)}",
    ]
    for m in messages:
        lines.append(f"MESSAGE_ROLE {m['role']}")
        lines.append(f"MESSAGE_CONTENT_B64 {b64(m['content'])}")
    lines.append(f"TOKEN_IDS {len(token_ids)}")
    lines.append(" ".join(str(t) for t in token_ids))

    write_lines(os.path.join(OUT_DIR, f"build_initial_tokens_{name}.txt"), lines)
    print(f"[build_initial_tokens] {name}: {len(token_ids)} tokens")

# === merge_assistant_tokens + align_response_metadata scenarios ===
# response_mask is always [] at the real call site (SingleTurnAgentLoop.run
# always calls ct_merge_assistant_token(prompt_ids, output.token_ids, [], ...)
# -- see README.md), but a couple of non-empty-response_mask cases are
# included anyway to exercise the general port more thoroughly.
BOOKKEEPING_SCENARIOS = {
    "basic": dict(
        runtime_token_ids=[1, 2, 3, 4, 5],
        assistant_token_ids=[10, 11, 12],
        response_mask=[],
        response_logprobs=[],
        assistant_logprobs=[-0.1, -0.2, -0.3],
    ),
    "empty_assistant": dict(
        runtime_token_ids=[1, 2, 3],
        assistant_token_ids=[],
        response_mask=[],
        response_logprobs=[],
        assistant_logprobs=[],
    ),
    "no_logprobs": dict(
        runtime_token_ids=[7, 8],
        assistant_token_ids=[9, 10, 11, 12],
        response_mask=[],
        response_logprobs=None,
        assistant_logprobs=None,
    ),
    "nonempty_prior_response_mask": dict(
        runtime_token_ids=[100, 101, 102],
        assistant_token_ids=[200, 201],
        response_mask=[1, 1],
        response_logprobs=[-0.5, -0.6],
        assistant_logprobs=[-0.05, -0.07],
    ),
}

for name, case in BOOKKEEPING_SCENARIOS.items():
    merge_result = builder.merge_assistant_tokens(case["runtime_token_ids"], case["assistant_token_ids"])
    aligned_mask, aligned_logprobs = builder.align_response_metadata(
        merge_result,
        case["response_mask"],
        case["response_logprobs"],
        assistant_logprobs=case["assistant_logprobs"],
    )

    def ints(xs):
        return [str(x) for x in xs]

    def floats(xs):
        return [repr(float(x)) for x in xs]

    lines = [
        f"RUNTIME_TOKEN_IDS {len(case['runtime_token_ids'])}",
        " ".join(ints(case["runtime_token_ids"])),
        f"ASSISTANT_TOKEN_IDS {len(case['assistant_token_ids'])}",
        " ".join(ints(case["assistant_token_ids"])),
        f"RESPONSE_MASK {len(case['response_mask'])}",
        " ".join(ints(case["response_mask"])),
        f"HAS_RESPONSE_LOGPROBS {1 if case['response_logprobs'] is not None else 0}",
        f"RESPONSE_LOGPROBS {len(case['response_logprobs'] or [])}",
        " ".join(floats(case["response_logprobs"] or [])),
        f"HAS_ASSISTANT_LOGPROBS {1 if case['assistant_logprobs'] is not None else 0}",
        f"ASSISTANT_LOGPROBS {len(case['assistant_logprobs'] or [])}",
        " ".join(floats(case["assistant_logprobs"] or [])),
        f"EXPECTED_MERGED_TOKEN_IDS {len(merge_result.token_ids)}",
        " ".join(ints(merge_result.token_ids)),
        f"EXPECTED_APPENDED_TOKEN_COUNT {merge_result.appended_token_count}",
        f"EXPECTED_KIND {merge_result.kind}",
        f"EXPECTED_INSERTED_TOKEN_IDS {len(merge_result.inserted_token_ids)}",
        " ".join(ints(merge_result.inserted_token_ids)),
        f"EXPECTED_REMOVED_PREFIX_COUNT {merge_result.removed_prefix_token_count}",
        f"EXPECTED_ALIGNED_MASK {len(aligned_mask)}",
        " ".join(ints(aligned_mask)),
        f"HAS_EXPECTED_ALIGNED_LOGPROBS {1 if aligned_logprobs is not None else 0}",
        f"EXPECTED_ALIGNED_LOGPROBS {len(aligned_logprobs or [])}",
        " ".join(floats(aligned_logprobs or [])),
    ]
    write_lines(os.path.join(OUT_DIR, f"merge_assistant_{name}.txt"), lines)
    print(f"[merge_assistant_tokens] {name}: appended={merge_result.appended_token_count} kind={merge_result.kind}")

print("\nFixtures written to", OUT_DIR)
