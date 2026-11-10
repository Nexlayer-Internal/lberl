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
// Validation harness: reads the fixtures `tests/gen_fixtures.py` generated
// from the REAL Python `QwenContinuousTokenBuilder` (base-class behavior;
// see README.md's scoping trace) and checks the C++/Rust port's output
// against them with EXACT token-ID sequence comparison -- not a tolerance
// comparison. A single mismatched token id anywhere is a real bug and this
// harness reports it as a hard failure, per this task's correctness bar
// (discrete token ids, unlike the other three ports' tensor-math float
// tolerances).
//
// Usage:
//   ./validate_continuous_token <fixtures_dir> <tokenizer_json_path> <chat_template_path>
// e.g.:
//   ./build/validate_continuous_token ./tests/fixtures \
//       ./tests/fixtures/qwen3_tokenizer/tokenizer.json \
//       ./tests/fixtures/qwen3_tokenizer/chat_template.jinja

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "verl_continuous_token/continuous_token.h"

namespace fs = std::filesystem;
using verl_continuous_token::ChatMessage;
using verl_continuous_token::ContinuousTokenBuilder;
using verl_continuous_token::MergeResult;

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool ok, const std::string& label, const std::string& detail = "") {
  ++g_checks;
  if (ok) {
    std::cout << "[PASS] " << label << "\n";
  } else {
    ++g_failures;
    std::cout << "[FAIL] " << label;
    if (!detail.empty()) std::cout << "  " << detail;
    std::cout << "\n";
  }
}

template <typename T>
std::string FormatMismatch(const std::vector<T>& expected, const std::vector<T>& actual) {
  std::ostringstream oss;
  oss << "expected.size()=" << expected.size() << " actual.size()=" << actual.size();
  size_t n = std::min(expected.size(), actual.size());
  for (size_t i = 0; i < n; ++i) {
    if (expected[i] != actual[i]) {
      oss << " first diff at index " << i << ": expected=" << expected[i] << " actual=" << actual[i];
      break;
    }
  }
  return oss.str();
}

template <typename T>
bool VectorsEqual(const std::vector<T>& a, const std::vector<T>& b) {
  return a == b;
}

// Minimal fixture-file reader matching the TAG/COUNT/[values] text format
// gen_fixtures.py writes (see that file's header comment for the exact
// format; this is a line-based reader, not a general parser).
class FixtureReader {
 public:
  explicit FixtureReader(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open fixture file: " + path);
    std::string line;
    while (std::getline(f, line)) lines_.push_back(line);
  }

  // Reads a line "TAG <count>", asserts the tag matches, returns count.
  int64_t ReadCountLine(const std::string& expected_tag) {
    std::string line = NextLine();
    std::istringstream iss(line);
    std::string tag;
    int64_t count;
    iss >> tag >> count;
    if (tag != expected_tag) {
      throw std::runtime_error("expected tag '" + expected_tag + "' but got '" + tag + "' in line: " + line);
    }
    return count;
  }

  // Reads a line "TAG <int>", returns the int.
  int64_t ReadIntLine(const std::string& expected_tag) { return ReadCountLine(expected_tag); }

  // Reads a line "TAG <token>" (token may be a word like a role name or kind).
  std::string ReadWordLine(const std::string& expected_tag) {
    std::string line = NextLine();
    std::istringstream iss(line);
    std::string tag, word;
    iss >> tag >> word;
    if (tag != expected_tag) {
      throw std::runtime_error("expected tag '" + expected_tag + "' but got '" + tag + "' in line: " + line);
    }
    return word;
  }

  // Reads the next raw line (a whitespace-separated values line), splitting
  // into `count` int64 values (0 values is a valid, possibly-blank line).
  std::vector<int64_t> ReadIntValues(int64_t count) {
    std::string line = NextLine();
    std::istringstream iss(line);
    std::vector<int64_t> values;
    int64_t v;
    while (iss >> v) values.push_back(v);
    if (static_cast<int64_t>(values.size()) != count) {
      throw std::runtime_error("expected " + std::to_string(count) + " int values, got " +
                                std::to_string(values.size()));
    }
    return values;
  }

  std::vector<double> ReadDoubleValues(int64_t count) {
    std::string line = NextLine();
    std::istringstream iss(line);
    std::vector<double> values;
    double v;
    while (iss >> v) values.push_back(v);
    if (static_cast<int64_t>(values.size()) != count) {
      throw std::runtime_error("expected " + std::to_string(count) + " double values, got " +
                                std::to_string(values.size()));
    }
    return values;
  }

  bool AtEnd() const { return pos_ >= lines_.size(); }

 private:
  std::string NextLine() {
    if (pos_ >= lines_.size()) throw std::runtime_error("unexpected end of fixture file");
    return lines_[pos_++];
  }

  std::vector<std::string> lines_;
  size_t pos_ = 0;
};

std::string Base64Decode(const std::string& encoded) {
  static const std::string kChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::vector<int> reverse(256, -1);
  for (size_t i = 0; i < kChars.size(); ++i) reverse[static_cast<unsigned char>(kChars[i])] = static_cast<int>(i);

  std::string out;
  // uint32_t (not int): well-defined wraparound on overflow. Only the low
  // byte is ever read back out (masked with & 0xFF below), and unsigned
  // modulo-2^32 arithmetic preserves those low bits exactly the same as
  // unbounded-precision arithmetic would, so wraparound here is harmless --
  // unlike plain `int`, whose signed-overflow UB UBSan correctly flagged in
  // an earlier version of this helper on long inputs.
  uint32_t val = 0;
  int bits = -8;
  for (unsigned char c : encoded) {
    if (c == '=' || c == '\n' || c == '\r') continue;
    if (reverse[c] == -1) continue;
    val = (val << 6) + static_cast<uint32_t>(reverse[c]);
    bits += 6;
    if (bits >= 0) {
      out.push_back(static_cast<char>((val >> bits) & 0xFF));
      bits -= 8;
    }
  }
  return out;
}

void RunBuildInitialTokensFixture(const ContinuousTokenBuilder& builder, const std::string& path) {
  FixtureReader r(path);
  const std::string name = fs::path(path).stem().string();

  const int64_t add_gen_prompt_int = r.ReadIntLine("ADD_GENERATION_PROMPT");
  const int64_t enable_thinking_defined = r.ReadIntLine("ENABLE_THINKING_DEFINED");
  const int64_t enable_thinking_value = r.ReadIntLine("ENABLE_THINKING_VALUE");
  const int64_t num_messages = r.ReadCountLine("MESSAGES");

  std::vector<ChatMessage> messages;
  for (int64_t i = 0; i < num_messages; ++i) {
    std::string role = r.ReadWordLine("MESSAGE_ROLE");
    std::string content_b64 = r.ReadWordLine("MESSAGE_CONTENT_B64");
    messages.push_back(ChatMessage{role, Base64Decode(content_b64)});
  }

  const int64_t num_tokens = r.ReadCountLine("TOKEN_IDS");
  std::vector<int64_t> expected_ids = r.ReadIntValues(num_tokens);

  std::optional<bool> enable_thinking;
  if (enable_thinking_defined) enable_thinking = static_cast<bool>(enable_thinking_value);

  try {
    std::vector<int64_t> actual_ids =
        builder.build_initial_tokens(messages, static_cast<bool>(add_gen_prompt_int), enable_thinking);
    Check(VectorsEqual(expected_ids, actual_ids), "build_initial_tokens:" + name,
          FormatMismatch(expected_ids, actual_ids));
  } catch (const std::exception& e) {
    Check(false, "build_initial_tokens:" + name, std::string("threw: ") + e.what());
  }
}

void RunMergeAssistantFixture(const std::string& path) {
  FixtureReader r(path);
  const std::string name = fs::path(path).stem().string();

  auto runtime_ids = r.ReadIntValues(r.ReadCountLine("RUNTIME_TOKEN_IDS"));
  auto assistant_ids = r.ReadIntValues(r.ReadCountLine("ASSISTANT_TOKEN_IDS"));
  auto response_mask = r.ReadIntValues(r.ReadCountLine("RESPONSE_MASK"));
  const int64_t has_response_logprobs = r.ReadIntLine("HAS_RESPONSE_LOGPROBS");
  auto response_logprobs_vals = r.ReadDoubleValues(r.ReadCountLine("RESPONSE_LOGPROBS"));
  const int64_t has_assistant_logprobs = r.ReadIntLine("HAS_ASSISTANT_LOGPROBS");
  auto assistant_logprobs_vals = r.ReadDoubleValues(r.ReadCountLine("ASSISTANT_LOGPROBS"));

  auto expected_merged = r.ReadIntValues(r.ReadCountLine("EXPECTED_MERGED_TOKEN_IDS"));
  const int64_t expected_appended_count = r.ReadIntLine("EXPECTED_APPENDED_TOKEN_COUNT");
  const std::string expected_kind = r.ReadWordLine("EXPECTED_KIND");
  auto expected_inserted = r.ReadIntValues(r.ReadCountLine("EXPECTED_INSERTED_TOKEN_IDS"));
  const int64_t expected_removed_prefix = r.ReadIntLine("EXPECTED_REMOVED_PREFIX_COUNT");
  auto expected_aligned_mask = r.ReadIntValues(r.ReadCountLine("EXPECTED_ALIGNED_MASK"));
  const int64_t has_expected_aligned_logprobs = r.ReadIntLine("HAS_EXPECTED_ALIGNED_LOGPROBS");
  auto expected_aligned_logprobs = r.ReadDoubleValues(r.ReadCountLine("EXPECTED_ALIGNED_LOGPROBS"));

  std::optional<std::vector<double>> response_logprobs;
  if (has_response_logprobs) response_logprobs = response_logprobs_vals;
  std::optional<std::vector<double>> assistant_logprobs;
  if (has_assistant_logprobs) assistant_logprobs = assistant_logprobs_vals;

  MergeResult merge_result = verl_continuous_token::merge_assistant_tokens(runtime_ids, assistant_ids);

  Check(VectorsEqual(expected_merged, merge_result.token_ids), "merge_assistant_tokens:" + name + " (token_ids)",
        FormatMismatch(expected_merged, merge_result.token_ids));
  Check(merge_result.appended_token_count == expected_appended_count,
        "merge_assistant_tokens:" + name + " (appended_token_count)");
  Check(merge_result.kind == expected_kind, "merge_assistant_tokens:" + name + " (kind)");
  Check(VectorsEqual(expected_inserted, merge_result.inserted_token_ids),
        "merge_assistant_tokens:" + name + " (inserted_token_ids)");
  Check(merge_result.removed_prefix_token_count == expected_removed_prefix,
        "merge_assistant_tokens:" + name + " (removed_prefix_token_count)");

  auto aligned = verl_continuous_token::align_response_metadata(merge_result, response_mask, response_logprobs,
                                                                 assistant_logprobs);
  Check(VectorsEqual(expected_aligned_mask, aligned.response_mask), "align_response_metadata:" + name + " (mask)",
        FormatMismatch(expected_aligned_mask, aligned.response_mask));
  Check(static_cast<bool>(has_expected_aligned_logprobs) == aligned.response_logprobs.has_value(),
        "align_response_metadata:" + name + " (logprobs presence)");
  if (has_expected_aligned_logprobs && aligned.response_logprobs.has_value()) {
    bool eq = expected_aligned_logprobs.size() == aligned.response_logprobs->size();
    if (eq) {
      for (size_t i = 0; i < expected_aligned_logprobs.size(); ++i) {
        // Exact match expected: these are pure pass-through values from the
        // fixture, not computed floating point -- no tolerance needed.
        if (expected_aligned_logprobs[i] != (*aligned.response_logprobs)[i]) {
          eq = false;
          break;
        }
      }
    }
    Check(eq, "align_response_metadata:" + name + " (logprobs values)");
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <fixtures_dir> <tokenizer_json_path> <chat_template_path>\n";
    return 2;
  }
  const std::string fixtures_dir = argv[1];
  const std::string tokenizer_json_path = argv[2];
  const std::string chat_template_path = argv[3];

  std::ifstream tmpl_file(chat_template_path);
  if (!tmpl_file) {
    std::cerr << "cannot open chat template file: " << chat_template_path << "\n";
    return 2;
  }
  std::ostringstream tmpl_stream;
  tmpl_stream << tmpl_file.rdbuf();
  const std::string chat_template = tmpl_stream.str();

  std::unique_ptr<ContinuousTokenBuilder> builder;
  try {
    builder = std::make_unique<ContinuousTokenBuilder>(tokenizer_json_path, chat_template);
  } catch (const std::exception& e) {
    std::cerr << "Failed to construct ContinuousTokenBuilder: " << e.what() << "\n";
    return 1;
  }

  std::vector<std::string> build_initial_fixtures;
  std::vector<std::string> merge_assistant_fixtures;
  for (const auto& entry : fs::directory_iterator(fixtures_dir)) {
    if (!entry.is_regular_file()) continue;
    const std::string filename = entry.path().filename().string();
    if (filename.rfind("build_initial_tokens_", 0) == 0) {
      build_initial_fixtures.push_back(entry.path().string());
    } else if (filename.rfind("merge_assistant_", 0) == 0) {
      merge_assistant_fixtures.push_back(entry.path().string());
    }
  }
  std::sort(build_initial_fixtures.begin(), build_initial_fixtures.end());
  std::sort(merge_assistant_fixtures.begin(), merge_assistant_fixtures.end());

  if (build_initial_fixtures.empty() && merge_assistant_fixtures.empty()) {
    std::cerr << "No fixtures found in " << fixtures_dir << "\n";
    return 2;
  }

  for (const auto& path : build_initial_fixtures) RunBuildInitialTokensFixture(*builder, path);
  for (const auto& path : merge_assistant_fixtures) RunMergeAssistantFixture(path);

  std::cout << "\n" << g_checks << " checks, " << g_failures << " failures\n";
  if (g_failures > 0) {
    std::cout << "SOME SCENARIOS FAILED\n";
    return 1;
  }
  std::cout << "ALL SCENARIOS PASSED\n";
  return 0;
}
