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

#include "verl_continuous_token/continuous_token.h"

#include <cstdio>
#include <sstream>
#include <stdexcept>

#include "verl_continuous_token/ffi/tokenizer_shim.h"

namespace verl_continuous_token {

namespace {

// Minimal JSON string escaper. Only needs to handle what `ChatMessage.role`/
// `.content` can contain: arbitrary UTF-8 text. JSON permits literal UTF-8
// bytes >= 0x80 inside strings (same as Python's `json.dumps(...,
// ensure_ascii=False)`, which is what HF's `tojson` override and verl's own
// JSON round-trips use) -- only `"`, `\`, and C0 control characters need
// escaping. This is deliberately not a general JSON writer: it exists only
// to serialize the two-field `{role, content}` shape this FFI boundary
// actually needs (see continuous_token.h's `ChatMessage` doc comment for why
// that shape is sufficient for this workload).
void AppendJsonEscapedString(std::string& out, const std::string& s) {
  out.push_back('"');
  for (unsigned char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  out.push_back('"');
}

std::string MessagesToJson(const std::vector<ChatMessage>& messages) {
  std::string out;
  out.push_back('[');
  for (size_t i = 0; i < messages.size(); ++i) {
    if (i) out.push_back(',');
    out += "{\"role\":";
    AppendJsonEscapedString(out, messages[i].role);
    out += ",\"content\":";
    AppendJsonEscapedString(out, messages[i].content);
    out.push_back('}');
  }
  out.push_back(']');
  return out;
}

// RAII helper for the Rust-owned `char*` error strings `vct_*` functions
// populate on failure.
class RustErrorString {
 public:
  RustErrorString() = default;
  ~RustErrorString() {
    if (ptr_ != nullptr) {
      vct_free_string(ptr_);
    }
  }
  RustErrorString(const RustErrorString&) = delete;
  RustErrorString& operator=(const RustErrorString&) = delete;

  char** out_param() { return &ptr_; }
  bool has_error() const { return ptr_ != nullptr; }
  std::string message() const { return ptr_ != nullptr ? std::string(ptr_) : std::string(); }

 private:
  char* ptr_ = nullptr;
};

}  // namespace

struct ContinuousTokenBuilder::Impl {
  VctBuilder* handle = nullptr;

  ~Impl() {
    if (handle != nullptr) {
      vct_builder_free(handle);
    }
  }
};

ContinuousTokenBuilder::ContinuousTokenBuilder(const std::string& tokenizer_json_path,
                                               const std::string& chat_template)
    : impl_(std::make_unique<Impl>()) {
  RustErrorString err;
  VctBuilder* handle = vct_builder_new(tokenizer_json_path.c_str(),
                                       reinterpret_cast<const uint8_t*>(chat_template.data()),
                                       chat_template.size(), err.out_param());
  if (handle == nullptr) {
    throw ContinuousTokenError(err.has_error() ? err.message()
                                                : "vct_builder_new failed with no error message");
  }
  impl_->handle = handle;
}

ContinuousTokenBuilder::~ContinuousTokenBuilder() = default;
ContinuousTokenBuilder::ContinuousTokenBuilder(ContinuousTokenBuilder&&) noexcept = default;
ContinuousTokenBuilder& ContinuousTokenBuilder::operator=(ContinuousTokenBuilder&&) noexcept = default;

std::vector<int64_t> ContinuousTokenBuilder::build_initial_tokens(const std::vector<ChatMessage>& messages,
                                                                   bool add_generation_prompt,
                                                                   std::optional<bool> enable_thinking) const {
  const std::string messages_json = MessagesToJson(messages);

  VctTokenIds out{};
  RustErrorString err;
  const int32_t rc = vct_build_initial_tokens(
      impl_->handle, reinterpret_cast<const uint8_t*>(messages_json.data()), messages_json.size(),
      add_generation_prompt, enable_thinking.has_value(), enable_thinking.value_or(false), &out, err.out_param());

  if (rc != 0) {
    throw ContinuousTokenError(err.has_error() ? err.message()
                                                : "vct_build_initial_tokens failed with no error message");
  }

  std::vector<int64_t> result(out.data, out.data + out.len);
  vct_free_token_ids(out);
  return result;
}

// === Pure C++ ports below: no tokenizer/Rust FFI involvement at all. ===
// These are direct, line-for-line ports of
// `ContinuousTokenBuilder.merge_assistant_tokens` and
// `ContinuousTokenBuilder.align_response_metadata`
// (`verl/utils/tokenizer/continuous_token.py`, base class, not overridden by
// `QwenContinuousTokenBuilder` -- see continuous_token.h).

MergeResult merge_assistant_tokens(const std::vector<int64_t>& runtime_token_ids,
                                   const std::vector<int64_t>& assistant_token_ids) {
  MergeResult result;
  result.token_ids = runtime_token_ids;
  result.token_ids.insert(result.token_ids.end(), assistant_token_ids.begin(), assistant_token_ids.end());
  result.appended_token_count = static_cast<int64_t>(assistant_token_ids.size());
  result.kind = "assistant";
  // inserted_token_ids / removed_prefix_token_count default-initialize to
  // empty/0, matching the Python dataclass defaults used by this method.
  return result;
}

AlignedResponseMetadata align_response_metadata(
    const MergeResult& merge_result, const std::vector<int64_t>& response_mask,
    const std::optional<std::vector<double>>& response_logprobs,
    const std::optional<std::vector<double>>& assistant_logprobs) {
  AlignedResponseMetadata out;
  out.response_mask = response_mask;
  out.response_logprobs = response_logprobs;

  if (!out.response_logprobs.has_value() && assistant_logprobs.has_value()) {
    throw std::invalid_argument("response_logprobs is required when assistant_logprobs is provided");
  }

  // Python: `aligned_mask[: -merge_result.removed_prefix_token_count]` guarded by
  // `if merge_result.removed_prefix_token_count:` (skipped entirely when 0 --
  // Python's `a[:-0]` would otherwise mean `a[:0]`, i.e. empty, which is NOT
  // "no trim"; the `if` guard is what actually makes 0 a no-op in the source).
  if (merge_result.removed_prefix_token_count != 0) {
    const auto n = static_cast<size_t>(merge_result.removed_prefix_token_count);
    // Python list slicing clamps out-of-range trims to empty rather than
    // throwing; mirror that for robustness even though this workload never
    // exercises a trim larger than the mask (SingleTurnAgentLoop's
    // merge_assistant_tokens call always produces removed_prefix_token_count
    // == 0 -- see README.md's scoping trace).
    const size_t keep = n >= out.response_mask.size() ? 0 : out.response_mask.size() - n;
    out.response_mask.resize(keep);
    if (out.response_logprobs.has_value()) {
      const size_t keep_lp = n >= out.response_logprobs->size() ? 0 : out.response_logprobs->size() - n;
      out.response_logprobs->resize(keep_lp);
    }
  }

  const auto inserted_token_count = static_cast<size_t>(merge_result.inserted_token_ids.size());
  out.response_mask.insert(out.response_mask.end(), inserted_token_count, 0);
  if (out.response_logprobs.has_value()) {
    out.response_logprobs->insert(out.response_logprobs->end(), inserted_token_count, 0.0);
  }

  if (merge_result.kind == "assistant") {
    out.response_mask.insert(out.response_mask.end(), static_cast<size_t>(merge_result.appended_token_count), 1);
    if (out.response_logprobs.has_value()) {
      std::vector<double> effective_assistant_logprobs;
      if (!assistant_logprobs.has_value()) {
        if (merge_result.appended_token_count != 0) {
          throw std::invalid_argument("assistant_logprobs is required for assistant Continuous Token alignment");
        }
        // effective_assistant_logprobs stays empty, matching Python's `assistant_logprobs = []`.
      } else {
        effective_assistant_logprobs = *assistant_logprobs;
      }
      if (static_cast<int64_t>(effective_assistant_logprobs.size()) != merge_result.appended_token_count) {
        std::ostringstream oss;
        oss << "assistant_logprobs length must match appended assistant token count, got "
            << effective_assistant_logprobs.size() << " and " << merge_result.appended_token_count;
        throw std::invalid_argument(oss.str());
      }
      out.response_logprobs->insert(out.response_logprobs->end(), effective_assistant_logprobs.begin(),
                                     effective_assistant_logprobs.end());
    }
  } else if (merge_result.kind == "context") {
    out.response_mask.insert(out.response_mask.end(), static_cast<size_t>(merge_result.appended_token_count), 0);
    if (out.response_logprobs.has_value()) {
      out.response_logprobs->insert(out.response_logprobs->end(),
                                     static_cast<size_t>(merge_result.appended_token_count), 0.0);
    }
  } else {
    throw std::invalid_argument("Unknown Continuous Token merge kind: " + merge_result.kind);
  }

  return out;
}

}  // namespace verl_continuous_token
