#pragma once

#include <algorithm>
#include <cstdint>
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
  uint32_t compactRows = 0;
  uint64_t extraMetalBytes = 0;
  std::string lookupReason, compactReason, fastTopKReason;
};

// A deterministic proposal from an earlier occurrence of the current suffix.
// History includes the pending anchor. All proposed tokens must already be
// in history; a short continuation is a miss, not padded or invented tokens.
inline std::vector<uint32_t> lookupProposal(std::span<const uint32_t> history,
                                            uint32_t count) {
  if (!count || history.size() < count + 2) return {};
  for (size_t n = std::min<size_t>(4, history.size()); n >= 2; --n) {
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
