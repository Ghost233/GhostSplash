#pragma once

#include <compare>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

namespace splash::model {

struct WeightShardIdentity {
  std::string headerSha256;
  uint64_t dataOffset = 0, fileBytes = 0, requiredFileBytes = 0;
  uint64_t device = 0, inode = 0;
  int64_t modifiedSeconds = 0, modifiedNanoseconds = 0;
  bool complete = false;
};

struct WeightRange {
  std::filesystem::path path;
  uint64_t offset = 0;
  uint64_t bytes = 0;
  uint64_t tensorBytes = 0; // whole tensor bytes, before selecting rows/an expert
  std::shared_ptr<const WeightShardIdentity> source;
  bool operator==(const WeightRange &) const = default;
};

enum class WeightUnitKind { Tensor, Expert, EmbeddingRow };
struct WeightPageKey {
  WeightUnitKind kind = WeightUnitKind::Tensor;
  std::string tensor;
  uint64_t unit = 0;
  auto operator<=>(const WeightPageKey &) const = default;
};

// Copying a lease pins the page. Its bytes remain valid until the last lease
// goes away, including after the cache itself is destroyed.
class WeightPageLease {
public:
  [[nodiscard]] const uint8_t *data() const noexcept;
  [[nodiscard]] uint64_t bytes() const noexcept;
  [[nodiscard]] std::span<const uint8_t> view() const noexcept { return {data(), bytes()}; }
private:
  friend class WeightPageCache;
  struct Page;
  explicit WeightPageLease(std::shared_ptr<const Page> page) : page_(std::move(page)) {}
  std::shared_ptr<const Page> page_;
};

struct WeightCacheTelemetry {
  uint64_t hits = 0, misses = 0, bytesRead = 0, residentBytes = 0, evictions = 0;
};

// Host bytes only: the caller accounts for any Metal staging separately.
// Concurrent acquire calls serialize the exact reads, so a miss never creates
// duplicate resident pages. A unit must fit whole; pinned pages are never evicted.
class WeightPageCache {
public:
  explicit WeightPageCache(uint64_t byteBudget);
  ~WeightPageCache();
  WeightPageCache(const WeightPageCache &) = delete;
  WeightPageCache &operator=(const WeightPageCache &) = delete;
  [[nodiscard]] WeightPageLease acquire(const WeightPageKey &key, const WeightRange &range);
  [[nodiscard]] WeightCacheTelemetry telemetry() const;
  [[nodiscard]] uint64_t byteBudget() const noexcept;
private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace splash::model
