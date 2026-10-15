#pragma once

#include <algorithm>
#include <cstdint>
#include <charconv>
#include <cstdlib>
#include <string_view>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace splash::model {

// Startup-only, independently opt-in. Direct RuntimeContext callers can
// provide these explicitly without reading process-global configuration.
struct OptionalFeatures final {
  bool promptLookup = false;
  bool compactDraftHead = false;
  bool fastTopK = false;
  std::filesystem::path compactRanking;
  uint32_t lookupMinimumMatch = 2;
  uint32_t compactCapacity = 98304;

  static OptionalFeatures fromEnvironment() {
    const auto flag = [](const char *name) {
      const char *value = std::getenv(name);
      if (!value || std::string_view(value) == "0") return false;
      if (std::string_view(value) == "1") return true;
      throw std::invalid_argument(std::string(name) + " must be 0 or 1");
    };
    OptionalFeatures result{flag("SPLASH_PROMPT_LOOKUP"),
                            flag("SPLASH_COMPACT_DRAFT_HEAD"),
                            flag("SPLASH_FAST_TOPK"), {}};
    const auto integer = [](const char *name, uint32_t fallback) {
      const char *value = std::getenv(name);
      if (!value) return fallback;
      const std::string_view text(value);
      uint32_t number = 0;
      const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), number);
      if (error != std::errc{} || end != text.data() + text.size())
        throw std::invalid_argument(std::string(name) + " must be an unsigned integer");
      return number;
    };
    if (result.promptLookup) {
      result.lookupMinimumMatch = integer("SPLASH_PROMPT_LOOKUP_MIN_MATCH", 2);
      if (result.lookupMinimumMatch < 2 || result.lookupMinimumMatch > 4)
        throw std::invalid_argument("SPLASH_PROMPT_LOOKUP_MIN_MATCH must be 2, 3 or 4");
    }
    if (result.compactDraftHead) {
      result.compactCapacity = integer("SPLASH_COMPACT_DRAFT_CAPACITY", 98304);
      if (result.compactCapacity <= 256 || result.compactCapacity > 98304 || result.compactCapacity % 256)
        throw std::invalid_argument("SPLASH_COMPACT_DRAFT_CAPACITY must be a multiple of256 in512..98304");
    }
    if (result.compactDraftHead)
      if (const char *path = std::getenv("SPLASH_COMPACT_DRAFT_RANKING"))
        result.compactRanking = path;
    return result;
  }
};

struct OptionalFeatureReport final {
  OptionalFeatures configured;
  uint64_t lookupAttempts = 0, lookupHits = 0, lookupAccepted = 0;
  uint64_t compactCycles = 0, compactFallbacks = 0;
  uint64_t fastTopKSelections = 0, fastTopKFallbacks = 0;
  uint32_t compactRows = 0, compactCapacity = 0;
  uint64_t extraMetalBytes = 0;
  std::string lookupReason, compactReason, fastTopKReason;
};

// A deterministic proposal from an earlier occurrence of the current suffix.
// History includes the pending anchor. All proposed tokens must already be
// in history; a short continuation is a miss, not padded or invented tokens.
inline std::vector<uint32_t> lookupProposal(std::span<const uint32_t> history,
                                            uint32_t count, uint32_t minimumMatch = 2) {
  if (minimumMatch < 2 || minimumMatch > 4)
    throw std::invalid_argument("lookup minimum match must be 2, 3 or 4");
  if (!count || history.size() < count + minimumMatch) return {};
  for (size_t n = std::min<size_t>(4, history.size()); n >= minimumMatch; --n) {
    const size_t suffix = history.size() - n;
    if (history.size() < n + count) continue;
    for (size_t end = history.size() - count; end >= n; --end) {
      const size_t start = end - n;
      if (start >= suffix) continue;
      if (std::equal(history.begin() + start, history.begin() + end,
                     history.begin() + suffix))
        return {history.begin() + end, history.begin() + end + count};
    }
  }
  return {};
}

} // namespace splash::model
