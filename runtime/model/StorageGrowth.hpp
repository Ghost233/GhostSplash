#pragma once

#include <cstdint>
#include <functional>

namespace splash::model {

// Released storage holds its helpers already. Required is the missing
// floor; peak also includes temporary payload, never those retained bytes.
struct StorageRestorePlan final {
  uint64_t targetBytes = 0, requiredBytes = 0, peakBytes = 0;
};

// One immutable store-layout proposal. Peak includes temporary payload;
// auxiliary is a retained subset of contribution, never an extra charge.
struct StorageGrowthPlan final {
  uint64_t revision = 0, candidate = 0, beforeBytes = 0;
  uint64_t contributionBytes = 0, peakBytes = 0, auxiliaryBytes = 0;
  uint32_t group = 0;
  bool operator==(const StorageGrowthPlan &) const = default;
};

struct StorageGrowthReceipt final {
  uint64_t contributionBytes = 0, grossBlocks = 0;
};

using StorageGrowthGuard = std::function<bool(const StorageGrowthPlan &)>;

} // namespace splash::model
