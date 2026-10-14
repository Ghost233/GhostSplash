#pragma once

#include <algorithm>
#include <cstdint>
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

  void validate() const {
    if (compactDraftHead && compactRanking.empty())
      throw std::invalid_argument("--compact-draft-head on requires --compact-draft-ranking FILE");
    if (lookupMinimumMatch < 2 || lookupMinimumMatch > 4)
      throw std::invalid_argument("--prompt-lookup-min-match must be 2, 3 or 4");
    if (compactCapacity <= 256 || compactCapacity > 98304 || compactCapacity % 256)
      throw std::invalid_argument("--compact-draft-capacity must be a multiple of 256 in 512..98304");
  }
};

struct OptionalFeatureReport final {
  OptionalFeatures configured;
  uint64_t lookupAttempts = 0, lookupHits = 0, lookupAccepted = 0;
  uint64_t compactCycles = 0, compactFallbacks = 0, compactCoverageFallbacks = 0, compactRequestFallbacks = 0;
  bool compactValidated = false;
  uint64_t compactValidationComparisons = 0;
  float compactValidationMaximumError = 0;
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
