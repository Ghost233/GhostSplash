#include "model/SparseRowStore.hpp"
#include "model/WeightSource.hpp"
#include "AwakeClock.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <unordered_set>
#include <utility>
#include <vector>

namespace splash::model {

void SparseRowTable::validate() const {
  if (!rows || !rowBytes) throw std::invalid_argument("streamed table " + name + " has no rows");
  if (rowBytes > kSparseRowMaximumBytes)
    throw std::invalid_argument("streamed table " + name + " has rows wider than the store takes");
  // Row ids travel as 32-bit values (the GPU's index buffers and the
  // staging slots); a wider table needs a wider ABI.
  if (rows > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("streamed table " + name + " has more rows than 32-bit ids address");
  if (rows > (std::numeric_limits<uint64_t>::max() - offset) / rowBytes)
    throw std::invalid_argument("streamed table " + name + " ends past the largest file offset");
}

namespace {

// Latencies count the time the Mac is awake, as the runtime's every duration.
using Clock = AwakeClock;
constexpr uint32_t kNoSlot = std::numeric_limits<uint32_t>::max();

// Reads the source a table's metadata was parsed from, as it was opened
// (WeightSource): a new file at its path changes nothing, and a write to it
// is a change.
class SourceRowReader final : public RowReader {
public:
  explicit SourceRowReader(std::shared_ptr<const WeightSource> source) : source_(std::move(source)) {}
  void read(uint64_t offset, std::span<uint8_t> bytes) override { source_->readFile(offset, bytes); }
  void checkUnchanged() override { source_->checkUnchanged(); }
  [[nodiscard]] uint64_t size() const override { return source_->bytes(); }

private:
  std::shared_ptr<const WeightSource> source_;
};

uint64_t nanosecondsSince(Clock::time_point start) {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

uint64_t indexBuckets(uint64_t rows) noexcept { return std::bit_ceil(std::max<uint64_t>(2 * rows, 2)); }

// Id's bucket of a table of 2^(64 - shift) buckets: the top bits of its
// product with 2^64 over the golden ratio (Fibonacci hashing).
uint64_t hashBucket(uint32_t id, uint32_t shift) noexcept { return (uint64_t{id} * 0x9E3779B97F4A7C15ull) >> shift; }

// The distinct ids of a stage or a prefetch, each with the order it came
// in: an open-addressed table of at least twice as many buckets as ids,
// whose entries carry the epoch they were written in, so begin() empties it
// without clearing it.
class DistinctIds final {
public:
  [[nodiscard]] uint64_t allocatedBytes() const noexcept {
    return (ids_.capacity() + orders_.capacity() + epochs_.capacity()) * sizeof(uint32_t);
  }
  // Empties the set for up to `count` ids.
  void begin(size_t count) {
    const size_t buckets = std::bit_ceil(std::max<size_t>(2 * count, 2));
    if (ids_.size() < buckets) {
      ids_.assign(buckets, 0);
      orders_.assign(buckets, 0);
      epochs_.assign(buckets, 0);
      epoch_ = 0;
    }
    if (++epoch_ == 0) {
      std::fill(epochs_.begin(), epochs_.end(), 0);
      epoch_ = 1;
    }
    mask_ = buckets - 1;
    shift_ = 64 - uint32_t(std::countr_zero(buckets));
    size_ = 0;
  }
  // The order `id` was first added in, and whether this call added it.
  std::pair<uint32_t, bool> add(uint32_t id) {
    size_t b = size_t(hashBucket(id, shift_));
    while (epochs_[b] == epoch_ && ids_[b] != id) b = (b + 1) & mask_;
    if (epochs_[b] == epoch_) return {orders_[b], false};
    epochs_[b] = epoch_;
    ids_[b] = id;
    orders_[b] = size_++;
    return {orders_[b], true};
  }

private:
  std::vector<uint32_t> ids_, orders_, epochs_;
  uint32_t epoch_ = 0, size_ = 0, shift_ = 0;
  size_t mask_ = 0;
};

} // namespace

uint64_t SparseRowStore::cacheBytesFor(uint64_t rows, uint32_t rowBytes) noexcept {
  if (!rows) return 0;
  // Row bytes, its id, two recency links; buckets of 32-bit slot numbers.
  const uint64_t perRow = uint64_t{rowBytes} + 3 * sizeof(uint32_t);
  if (rows > std::numeric_limits<uint64_t>::max() / 4 / perRow) return std::numeric_limits<uint64_t>::max();
  return rows * perRow + indexBuckets(rows) * sizeof(uint32_t);
}

uint64_t SparseRowStore::workingBytes(uint32_t rowBytes, uint32_t prefetchRows, uint32_t stageRows) noexcept {
  // A prefetched row's bytes, id and destination; a stage's dedupe map (three
  // words in each of twice as many buckets), distinct and missing ids, and
  // its read job's id and destination per row.
  const uint64_t prefetch = uint64_t{prefetchRows} * (uint64_t{rowBytes} + sizeof(uint32_t) + sizeof(uint8_t *));
  const uint64_t stage = uint64_t{stageRows} * (6 * sizeof(uint32_t) + 2 * sizeof(uint32_t) +
                                                sizeof(uint32_t) + sizeof(uint8_t *));
  return prefetch + stage;
}

uint64_t SparseRowStore::cacheRowsFor(uint64_t cacheBytes, uint32_t rowBytes) noexcept {
  if (!rowBytes) return 0;
  // Slot numbers are 32-bit, one of them reserved.
  uint64_t low = 0, high = std::min<uint64_t>(cacheBytes / rowBytes, std::numeric_limits<uint32_t>::max() - 1);
  while (low < high) {
    const uint64_t middle = low + (high - low + 1) / 2;
    if (cacheBytesFor(middle, rowBytes) <= cacheBytes) low = middle;
    else high = middle - 1;
  }
  return low;
}

// Fixed-size chunks preserve every retained row's address while the cache
// grows. Shrinking destroys allocations instead of retaining vector capacity.
template <class T> class RowChunks final {
public:
  void allocate(uint64_t count, uint64_t quantum, HostMemoryAdmission admit = {}) {
    quantum_ = quantum ? quantum : count;
    admit_ = std::move(admit);
    resize(count);
  }
  void resize(uint64_t count) {
    const uint64_t blocks = (count + quantum_ - 1) / quantum_;
    const auto old = chunks_.size();
    try {
      while (chunks_.size() < blocks) {
        Chunk chunk;
        const auto allocate = [&] { chunk.memory = std::make_unique_for_overwrite<T[]>(quantum_); };
        if (admit_) chunk.allocation = admit_(quantum_ * sizeof(T), allocate, [&] { return quantum_ * sizeof(T); });
        else allocate();
        chunks_.push_back(std::move(chunk));
      }
    } catch (...) { chunks_.resize(old); throw; }
    chunks_.resize(blocks);
  }
  T &operator[](uint64_t i) const noexcept { return chunks_[i / quantum_].memory[i % quantum_]; }
private:
  struct Chunk final { HostMemoryAllocation allocation; std::unique_ptr<T[]> memory; };
  uint64_t quantum_ = 1;
  HostMemoryAdmission admit_;
  std::vector<Chunk> chunks_;
};

constexpr uint64_t kRowChunkRows = 4096;
constexpr uint64_t kEvictedRows = 4096;

struct ResizableRowGeometry final {
  uint64_t chunkRows = kRowChunkRows, fullRows = UINT64_MAX, buckets = 0, fixedBytes = 0;
};

ResizableRowGeometry resizableRowGeometry(uint64_t maximumBytes, uint32_t rowBytes,
                                         uint32_t prefetchRows, uint32_t stageRows,
                                         std::optional<uint64_t> modelRows) {
  const uint64_t working = SparseRowStore::workingBytes(rowBytes, prefetchRows, stageRows);
  if (!rowBytes || maximumBytes <= working || (modelRows && !*modelRows))
    throw std::invalid_argument("adaptive PLE maximum is below its working memory or has no model rows");
  ResizableRowGeometry result;
  if (modelRows) {
    result.chunkRows = std::min(kRowChunkRows, *modelRows);
    result.fullRows = std::min((*modelRows + result.chunkRows - 1) / result.chunkRows * result.chunkRows,
                              (uint64_t{kNoSlot} - 1) / result.chunkRows * result.chunkRows);
  }
  result.buckets = indexBuckets(std::min(result.fullRows,
      SparseRowStore::cacheRowsFor(maximumBytes - working, rowBytes)));
  result.fixedBytes = working + (result.buckets + kEvictedRows) * sizeof(uint32_t);
  return result;
}

struct SparseRowStore::Impl {
  StorageGrowthGuard growthGuard;
  uint64_t capacityRevision = 1;
  // Payloads below die before these accounting owners.
  std::vector<HostMemoryAllocation> workingAllocations;
  void allocatePayload(uint64_t bytes, const std::function<void()> &allocate,
                       const std::function<uint64_t()> &held = {}) {
    if (bytes && options.hostAllocation)
      workingAllocations.push_back(options.hostAllocation(bytes, allocate, held ? held : [bytes] { return bytes; }));
    else allocate();
  }
  // A prefetch: ids[i]'s row into its own rows, read on the readers
  // (`batch`); `ended` once its rows were cached.
  struct Prefetch final {
    std::vector<uint32_t> ids;
    std::vector<uint8_t> rows;
    std::vector<uint32_t> places;
    std::shared_ptr<SsdReaders::Batch> batch;
    bool ended = false; // under prefetchMutex
  };

  SparseRowTable table;
  std::unique_ptr<RowReader> reader;
  SparseRowStoreOptions options;

  // The cache: slots of rows with their ids, a recency list (head most
  // recent) and an open-addressed index of slot + 1 (0 empty).
  mutable std::mutex cacheMutex;
  uint64_t capacity = 0;
  uint64_t chunkRows = kRowChunkRows, maximumRows = UINT64_MAX;
  uint64_t used = 0;
  RowChunks<uint8_t> rows, prefetched;
  RowChunks<uint32_t> ids, previous, next;
  std::unique_ptr<uint32_t[]> buckets, evicted;
  std::atomic<uint64_t> committed{0};
  uint64_t bucketMask = 0;
  uint32_t bucketShift = 0;
  uint32_t head = kNoSlot, tail = kNoSlot;
  uint64_t evictions = 0;

  // Stages, which are serialized, and prefetches: the distinct ids of the
  // last of each, and the prefetches not yet stopped or finished, whose rows
  // prefetchingRows count and prefetchingIds hold until they end.
  std::mutex stageMutex;
  DistinctIds distinct;
  std::vector<uint32_t> unique, missing;
  std::vector<uint8_t *> destinations;
  std::mutex prefetchMutex;
  DistinctIds prefetchDistinct;
  std::vector<std::shared_ptr<Prefetch>> prefetching;
  uint64_t prefetchingRows = 0;
  std::vector<uint8_t> prefetchPayload;
  std::vector<uint32_t> freePlaces;
  uint64_t boundedWorkingHeld = 0;
  [[nodiscard]] uint8_t *prefetchRow(Prefetch &each, size_t index) {
    return options.boundedWorkingMemory ? prefetchPayload.data() + uint64_t{each.places[index]} * table.rowBytes
                                        : each.rows.data() + index * table.rowBytes;
  }
  std::unordered_set<uint32_t> prefetchingIds;

  // Failure.
  std::atomic<bool> poisoned{false};
  mutable std::mutex failureMutex;
  std::string failure;

  // Statistics.
  struct UseCounters final {
    std::atomic<uint64_t> stages{0}, requested{0}, unique{0}, hits{0}, read{0};
    LatencyHistogram latency;
    std::atomic<uint64_t> reused{0}, waitNanoseconds{0}, prefetchedRows{0};
  };
  std::array<UseCounters, kSparseRowUses> uses;
  std::atomic<uint64_t> readBytes{0}, readRetries{0}, readErrors{0};
  std::atomic<uint64_t> prefetchRequested{0}, prefetchRead{0}, prefetchDropped{0};
  LatencyHistogram readLatency;

  Impl(SparseRowTable table, std::unique_ptr<RowReader> reader, SparseRowStoreOptions options)
      : table(std::move(table)), reader(std::move(reader)), options(std::move(options)) {}

  [[nodiscard]] uint8_t *slotRow(uint32_t slot) const noexcept { return &rows[uint64_t{slot} * table.rowBytes]; }
  [[nodiscard]] uint64_t bucketOf(uint32_t id) const noexcept { return hashBucket(id, bucketShift); }

  void allocate(uint64_t cacheBytes) {
    const uint64_t working = workingBytes(table.rowBytes, options.prefetchRows, options.stageRows);
    if (options.maximumCacheBytes && (cacheBytes > options.maximumCacheBytes || options.maximumCacheBytes <= working))
      throw std::invalid_argument("adaptive PLE cache exceeds its maximum or cannot hold its working memory");
    capacity = cacheBytes > working ? cacheRowsFor(cacheBytes - working, table.rowBytes) : 0;
    if (!capacity) throw std::invalid_argument("the row cache holds no row of " + table.name);
    uint64_t bucketCount = indexBuckets(options.maximumCacheBytes
        ? cacheRowsFor(options.maximumCacheBytes - working, table.rowBytes) : capacity);
    if (options.maximumCacheBytes) {
      const auto geometry = resizableRowGeometry(options.maximumCacheBytes, table.rowBytes,
          options.prefetchRows, options.stageRows,
          options.boundedWorkingMemory ? std::optional(table.rows) : std::nullopt);
      bucketCount = geometry.buckets;
      chunkRows = geometry.chunkRows;
      maximumRows = geometry.fullRows;
      const uint64_t fixed = geometry.fixedBytes;
      if (cacheBytes <= fixed) throw std::invalid_argument("adaptive PLE cache is below its index reservation");
      capacity = std::min(maximumRows,
          (cacheBytes - fixed) / (table.rowBytes + 3 * sizeof(uint32_t) + 1) / chunkRows * chunkRows);
      if (!capacity) throw std::invalid_argument("adaptive PLE cache holds no row chunk");
      allocatePayload(kEvictedRows * sizeof(uint32_t), [&] { evicted = std::make_unique<uint32_t[]>(kEvictedRows); });
      std::fill_n(evicted.get(), kEvictedRows, kNoSlot);
    }
    bucketMask = bucketCount - 1;
    bucketShift = 64 - uint32_t(std::countr_zero(bucketCount));
    const uint64_t quantum = options.maximumCacheBytes ? chunkRows : 0;
    rows.allocate(capacity * table.rowBytes, quantum * table.rowBytes, options.hostAllocation);
    if (options.maximumCacheBytes) prefetched.allocate(capacity, quantum, options.hostAllocation);
    ids.allocate(capacity, quantum, options.hostAllocation);
    previous.allocate(capacity, quantum, options.hostAllocation);
    next.allocate(capacity, quantum, options.hostAllocation);
    allocatePayload(bucketCount * sizeof(uint32_t), [&] { buckets = std::make_unique<uint32_t[]>(bucketCount); });
    if (options.boundedWorkingMemory) {
      allocatePayload(3 * indexBuckets(options.stageRows) * sizeof(uint32_t), [&] { distinct.begin(options.stageRows); },
                      [&] { return distinct.allocatedBytes(); });
      allocatePayload(uint64_t{options.stageRows} * sizeof(uint32_t), [&] { unique.reserve(options.stageRows); }, [&] { return unique.capacity() * sizeof(uint32_t); });
      allocatePayload(uint64_t{options.stageRows} * sizeof(uint32_t), [&] { missing.reserve(options.stageRows); }, [&] { return missing.capacity() * sizeof(uint32_t); });
      allocatePayload(uint64_t{options.stageRows} * sizeof(uint8_t *), [&] { destinations.reserve(options.stageRows); }, [&] { return destinations.capacity() * sizeof(uint8_t *); });
      allocatePayload(3 * indexBuckets(options.prefetchRows) * sizeof(uint32_t), [&] { prefetchDistinct.begin(options.prefetchRows); },
                      [&] { return prefetchDistinct.allocatedBytes(); });
      allocatePayload(uint64_t{options.prefetchRows} * table.rowBytes, [&] { prefetchPayload.resize(uint64_t{options.prefetchRows} * table.rowBytes); },
                      [&] { return prefetchPayload.capacity(); });
      allocatePayload(uint64_t{options.prefetchRows} * sizeof(uint32_t), [&] { freePlaces.reserve(options.prefetchRows); }, [&] { return freePlaces.capacity() * sizeof(uint32_t); });
      for (uint32_t row = 0; row < options.prefetchRows; ++row) freePlaces.push_back(row);
      boundedWorkingHeld = distinct.allocatedBytes() + prefetchDistinct.allocatedBytes() +
          (unique.capacity() + missing.capacity() + freePlaces.capacity()) * sizeof(uint32_t) +
          destinations.capacity() * sizeof(uint8_t *) + prefetchPayload.capacity();
    }
    publishBytes();
  }

  void publishBytes() {
    committed.store(workingBytes(table.rowBytes, options.prefetchRows, options.stageRows) +
        (options.maximumCacheBytes ? capacity * (table.rowBytes + 3 * sizeof(uint32_t) + 1) +
             (bucketMask + 1 + kEvictedRows) * sizeof(uint32_t) : cacheBytesFor(capacity, table.rowBytes)),
        std::memory_order_release);
  }

  // Removes the coldest row and compacts its hole from the final occupied
  // slot. At most one row is copied per eviction, inside the donor's memory.
  void discardTail() {
    const uint32_t slot = tail, last = static_cast<uint32_t>(used - 1);
    if (evicted) evicted[ids[slot] % kEvictedRows] = ids[slot];
    erase(ids[slot]);
    unlink(slot);
    ++evictions;
    if (slot != last) {
      std::memcpy(slotRow(slot), slotRow(last), table.rowBytes);
      ids[slot] = ids[last];
      if (evicted) prefetched[slot] = prefetched[last];
      previous[slot] = previous[last];
      next[slot] = next[last];
      (previous[slot] == kNoSlot ? head : next[previous[slot]]) = slot;
      (next[slot] == kNoSlot ? tail : previous[next[slot]]) = slot;
      uint64_t b = bucketOf(ids[slot]);
      while (buckets[b] != last + 1) b = (b + 1) & bucketMask;
      buckets[b] = slot + 1;
    }
    --used;
  }

  // Cache operations, under cacheMutex.
  [[nodiscard]] uint32_t find(uint32_t id) const noexcept {
    for (uint64_t b = bucketOf(id);; b = (b + 1) & bucketMask) {
      const uint32_t entry = buckets[b];
      if (!entry) return kNoSlot;
      if (ids[entry - 1] == id) return entry - 1;
    }
  }
  void unlink(uint32_t slot) noexcept {
    const uint32_t before = previous[slot], after = next[slot];
    (before == kNoSlot ? head : next[before]) = after;
    (after == kNoSlot ? tail : previous[after]) = before;
  }
  void pushFront(uint32_t slot) noexcept {
    previous[slot] = kNoSlot;
    next[slot] = head;
    (head == kNoSlot ? tail : previous[head]) = slot;
    head = slot;
  }
  void touch(uint32_t slot) noexcept {
    if (head == slot) return;
    unlink(slot);
    pushFront(slot);
  }
  // Removes `id`'s bucket, shifting the probe run after it back.
  void erase(uint32_t id) noexcept {
    uint64_t hole = bucketOf(id);
    while (ids[buckets[hole] - 1] != id) hole = (hole + 1) & bucketMask;
    for (uint64_t probe = (hole + 1) & bucketMask;; probe = (probe + 1) & bucketMask) {
      const uint32_t entry = buckets[probe];
      if (!entry) break;
      const uint64_t home = bucketOf(ids[entry - 1]);
      // The entry may move to the hole when its home is not in (hole, probe].
      if (((probe - home) & bucketMask) >= ((probe - hole) & bucketMask)) {
        buckets[hole] = entry;
        hole = probe;
      }
    }
    buckets[hole] = 0;
  }
  void insert(uint32_t id, const uint8_t *bytes, bool fromPrefetch = false) noexcept {
    if (const uint32_t slot = find(id); slot != kNoSlot) {
      touch(slot);
      return;
    }
    uint32_t slot;
    if (used < capacity) {
      slot = uint32_t(used++);
    } else {
      slot = tail;
      unlink(slot);
      erase(ids[slot]);
      if (evicted) evicted[ids[slot] % kEvictedRows] = ids[slot];
      ++evictions;
    }
    ids[slot] = id;
    if (evicted) prefetched[slot] = fromPrefetch;
    std::memcpy(slotRow(slot), bytes, table.rowBytes);
    uint64_t b = bucketOf(id);
    while (buckets[b]) b = (b + 1) & bucketMask;
    buckets[b] = slot + 1;
    pushFront(slot);
  }

  void poison(const std::string &message) {
    {
      const std::lock_guard lock(failureMutex);
      if (failure.empty()) failure = message;
    }
    poisoned.store(true, std::memory_order_release);
  }
  [[noreturn]] void throwPoisoned() const {
    const std::lock_guard lock(failureMutex);
    throw SparseRowError("rows of " + table.name + " are unavailable: " + failure);
  }

  void readRow(uint32_t id, uint8_t *destination) {
    const auto started = Clock::now();
    reader->read(table.offset + uint64_t{id} * table.rowBytes, {destination, table.rowBytes});
    readLatency.add(nanosecondsSince(started));
    readBytes.fetch_add(table.rowBytes, std::memory_order_relaxed);
  }
  void count(const SsdReaders::Tally &tally) {
    readRetries.fetch_add(tally.retries, std::memory_order_relaxed);
    readErrors.fetch_add(tally.failures, std::memory_order_relaxed);
  }

  // Runs on the thread that read a prefetch's last row: its rows enter the
  // cache once the source is unchanged since they were read, but a row that
  // could not be read, a hint, is dropped; a written source poisons the
  // store for the next stage.
  void endPrefetch(Prefetch &prefetch, const SsdReaders::Batch &batch) {
    count(batch.tally());
    try {
      reader->checkUnchanged();
    } catch (const std::exception &error) {
      poison(error.what());
    }
    uint64_t read = 0;
    if (!poisoned.load(std::memory_order_acquire)) {
      const std::lock_guard lock(cacheMutex);
      for (size_t index = 0; index < prefetch.ids.size(); ++index)
        if (!batch.failed(index)) {
          insert(prefetch.ids[index], prefetchRow(prefetch, index), true);
          ++read;
        }
    }
    prefetchRead.fetch_add(read, std::memory_order_relaxed);
    prefetchDropped.fetch_add(prefetch.ids.size() - read, std::memory_order_relaxed);
    const std::lock_guard lock(prefetchMutex);
    prefetch.ended = true;
    prefetchingRows -= prefetch.ids.size();
    for (const uint32_t id : prefetch.ids) prefetchingIds.erase(id);
    if (options.boundedWorkingMemory)
      freePlaces.insert(freePlaces.end(), prefetch.places.begin(), prefetch.places.end());
  }

  // The prefetches in flight; forgets those that ended, whose readers no
  // longer run them.
  std::vector<std::shared_ptr<Prefetch>> inFlight() {
    const std::lock_guard lock(prefetchMutex);
    std::erase_if(prefetching, [](const std::shared_ptr<Prefetch> &prefetch) { return prefetch->ended; });
    return prefetching;
  }

  // Finishes the prefetches in flight here and on the readers.
  void adoptPrefetches() {
    for (const std::shared_ptr<Prefetch> &prefetch : inFlight()) options.readers->finish(*prefetch->batch);
  }
};

SparseRowStore::SparseRowStore(SparseRowTable table, std::unique_ptr<RowReader> reader,
                               SparseRowStoreOptions options) {
  table.validate();
  if (!reader) throw std::invalid_argument("streamed table " + table.name + " has no reader");
  if (table.offset + table.rows * table.rowBytes > reader->size())
    throw std::invalid_argument("streamed table " + table.name + " ends past its file");
  if (!options.readers) throw std::invalid_argument("streamed table " + table.name + " has no readers");
  if (options.hostAllocation && !options.boundedWorkingMemory)
    throw std::invalid_argument("host admission requires bounded PLE working memory");
  const uint64_t cacheBytes = options.cacheBytes;
  impl_ = std::make_unique<Impl>(std::move(table), std::move(reader), std::move(options));
  impl_->allocate(cacheBytes);
}

SparseRowStore::~SparseRowStore() {
  // A prefetch's rows enter the cache, which goes with the store.
  for (const std::shared_ptr<Impl::Prefetch> &prefetch : impl_->inFlight())
    impl_->options.readers->stop(*prefetch->batch);
}

const SparseRowTable &SparseRowStore::table() const noexcept { return impl_->table; }

uint64_t SparseRowStore::committedBytes() const noexcept {
  return impl_->committed.load(std::memory_order_acquire);
}

uint64_t SparseRowStore::boundedWorkingBytes(uint32_t rowBytes, uint32_t prefetchRows, uint32_t stageRows) noexcept {
  return uint64_t{prefetchRows} * (uint64_t{rowBytes} + sizeof(uint32_t)) +
      3 * (indexBuckets(prefetchRows) + indexBuckets(stageRows)) * sizeof(uint32_t) +
      uint64_t{stageRows} * (2 * sizeof(uint32_t) + sizeof(uint8_t *));
}

uint64_t SparseRowStore::nonBackendHostBytes() const noexcept {
  // No off->on migration exists: every claimed cache/working payload of an
  // admitted store has a lease from construction, including partial growth.
  return impl_->options.hostAllocation ? 0 : hostAllocatedBytes();
}

uint64_t SparseRowStore::hostAllocatedBytes() const noexcept {
  const auto &s = *impl_;
  if (!s.options.boundedWorkingMemory) return committedBytes();
  return committedBytes() - workingBytes(s.table.rowBytes, s.options.prefetchRows, s.options.stageRows) + s.boundedWorkingHeld;
}

uint64_t SparseRowStore::minimumResizableBytes(uint64_t maximumBytes, uint32_t rowBytes,
                                               uint32_t prefetchRows, uint32_t stageRows,
                                               std::optional<uint64_t> modelRows) {
  const auto geometry = resizableRowGeometry(maximumBytes, rowBytes, prefetchRows, stageRows, modelRows);
  return geometry.fixedBytes + geometry.chunkRows * (rowBytes + 3 * sizeof(uint32_t) + 1);
}

uint64_t SparseRowStore::maximumResizableBytes(uint64_t modelRows, uint32_t rowBytes,
                                               uint32_t prefetchRows, uint32_t stageRows) {
  if (!modelRows || !rowBytes || modelRows > kNoSlot)
    throw std::invalid_argument("invalid PLE model row geometry");
  const uint64_t chunk = std::min(kRowChunkRows, modelRows);
  const uint64_t rows = std::min((modelRows + chunk - 1) / chunk * chunk,
                               (uint64_t{kNoSlot} - 1) / chunk * chunk);
  return workingBytes(rowBytes, prefetchRows, stageRows) +
      (indexBuckets(rows) + kEvictedRows) * sizeof(uint32_t) + rows * (rowBytes + 3 * sizeof(uint32_t) + 1);
}

void SparseRowStore::resize(uint64_t cacheBytes) { resizeImpl(cacheBytes, false); }

void SparseRowStore::resizeImpl(uint64_t cacheBytes, bool planned) {
  Impl &s = *impl_;
  if (s.growthGuard && !planned && cacheBytes > committedBytes())
    throw std::invalid_argument("PLE receiving requires an authorized single-block plan");
  const std::lock_guard stageLock(s.stageMutex);
  if (!s.options.maximumCacheBytes || cacheBytes > s.options.maximumCacheBytes)
    throw std::invalid_argument("PLE resize is disabled or exceeds its maximum");
  s.adoptPrefetches();
  const std::lock_guard lock(s.cacheMutex);
  const uint64_t fixed = workingBytes(s.table.rowBytes, s.options.prefetchRows, s.options.stageRows) +
      (s.bucketMask + 1 + kEvictedRows) * sizeof(uint32_t);
  if (cacheBytes < fixed + s.chunkRows * (s.table.rowBytes + 3 * sizeof(uint32_t) + 1))
    throw std::invalid_argument("PLE resize is below its physical chunk floor");
  const uint64_t target = std::min(s.maximumRows,
      (cacheBytes - fixed) / (s.table.rowBytes + 3 * sizeof(uint32_t) + 1) / s.chunkRows * s.chunkRows);
  while (s.used > target) s.discardTail();
  const uint64_t old = s.capacity;
  if (target != old) ++s.capacityRevision;
  try {
    s.rows.resize(target * s.table.rowBytes);
    s.prefetched.resize(target);
    s.ids.resize(target);
    s.previous.resize(target);
    s.next.resize(target);
  } catch (...) {
    s.rows.resize(old * s.table.rowBytes);
    s.prefetched.resize(old);
    s.ids.resize(old);
    s.previous.resize(old);
    s.next.resize(old);
    throw;
  }
  s.capacity = target;
  s.publishBytes();
}

void SparseRowStore::setGrowthGuard(StorageGrowthGuard guard) {
  if (!guard) throw std::invalid_argument("PLE growth guard is empty");
  impl_->growthGuard = std::move(guard);
}

std::optional<StorageGrowthPlan> SparseRowStore::planGrowOne(uint64_t maximumBytes) const {
  auto &s = *impl_;
  const std::lock_guard stageLock(s.stageMutex);
  const std::lock_guard cacheLock(s.cacheMutex);
  if (!s.options.maximumCacheBytes) return std::nullopt;
  const uint64_t before = committedBytes();
  const uint64_t wanted = std::min(maximumBytes, s.options.maximumCacheBytes);
  if (wanted <= before) return std::nullopt;
  const uint64_t fixed = before - s.capacity * (s.table.rowBytes + 13);
  const uint64_t rows = s.capacity + s.chunkRows;
  if (rows > s.maximumRows || fixed + rows * (s.table.rowBytes + 13) > wanted) return std::nullopt;
  const uint64_t contribution = (rows - s.capacity) * (s.table.rowBytes + 13);
  return StorageGrowthPlan{s.capacityRevision, s.capacityRevision, before, contribution, contribution,
                           (rows - s.capacity) * 13, 0};
}

StorageGrowthReceipt SparseRowStore::growOne(const StorageGrowthPlan &plan) {
  auto &s = *impl_;
  if (plan.revision != s.capacityRevision || plan.candidate != s.capacityRevision ||
      plan.beforeBytes != committedBytes() || plan.contributionBytes != s.chunkRows * (s.table.rowBytes + 13) ||
      plan.peakBytes != plan.contributionBytes || plan.auxiliaryBytes != s.chunkRows * 13 || plan.group ||
      (s.growthGuard && !s.growthGuard(plan)))
    throw std::invalid_argument("PLE single-block plan is stale or not authorized");
  const uint64_t beforeRows = s.capacity;
  resizeImpl(plan.beforeBytes + plan.contributionBytes, true);
  return {committedBytes() - plan.beforeBytes, (s.capacity - beforeRows) / s.chunkRows};
}

SparseRowStageResult SparseRowStore::stage(std::span<const uint32_t> ids, std::span<uint8_t> rows,
                                           std::span<uint32_t> slots, SparseRowUse use, bool adoptPrefetch) {
  Impl &s = *impl_;
  const std::lock_guard stageLock(s.stageMutex);
  const auto started = Clock::now();
  const uint32_t rowBytes = s.table.rowBytes;
  if (s.options.boundedWorkingMemory && ids.size() > s.options.stageRows)
    throw SparseRowError("PLE stage exceeds its admitted working-memory plan");
  if (slots.size() < ids.size() || rows.size() / rowBytes < ids.size())
    throw std::invalid_argument("staging of " + s.table.name + " is smaller than its rows");
  if (ids.size() > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("too many rows staged at once");
  if (s.poisoned.load(std::memory_order_acquire)) s.throwPoisoned();
  uint64_t readWait = 0, reused = 0, prefetchHits = 0;
  if (adoptPrefetch) {
    const auto wait = Clock::now();
    s.adoptPrefetches();
    readWait += nanosecondsSince(wait);
  }
  if (s.poisoned.load(std::memory_order_acquire)) s.throwPoisoned();
  try {
    s.reader->checkUnchanged();
  } catch (const std::exception &error) {
    s.poison(error.what());
    s.throwPoisoned();
  }

  // Distinct ids, in order of first use.
  s.distinct.begin(ids.size());
  s.unique.clear();
  for (size_t i = 0; i < ids.size(); ++i) {
    const uint32_t id = ids[i];
    if (id >= s.table.rows)
      throw SparseRowError("row " + std::to_string(id) + " is past the end of " + s.table.name);
    const auto [order, added] = s.distinct.add(id);
    if (added) s.unique.push_back(id);
    slots[i] = order;
  }

  // Cached rows are copied out while the cache cannot evict them.
  s.missing.clear();
  {
    const std::lock_guard lock(s.cacheMutex);
    for (uint32_t u = 0; u < s.unique.size(); ++u) {
      const uint32_t slot = s.find(s.unique[u]);
      // A completed prefetch may have reinserted an evicted row before this
      // demand arrived. Attribute that source reread to the consuming phase.
      if (s.evicted && s.evicted[s.unique[u] % kEvictedRows] == s.unique[u]) {
        ++reused;
        s.evicted[s.unique[u] % kEvictedRows] = kNoSlot;
      }
      if (slot == kNoSlot) { s.missing.push_back(u); continue; }
      if (s.evicted && s.prefetched[slot]) { ++prefetchHits; s.prefetched[slot] = false; }
      std::memcpy(rows.data() + uint64_t{u} * rowBytes, s.slotRow(slot), rowBytes);
      s.touch(slot);
    }
  }

  if (!s.missing.empty()) {
    s.destinations.clear();
    for (const uint32_t u : s.missing) s.destinations.push_back(rows.data() + uint64_t{u} * rowBytes);
    // No reader writes into `rows` once the reads returned.
    SsdReaders::Tally tally;
    const auto readStarted = Clock::now();
    try {
      s.options.readers->run(
          s.missing.size(), [&](size_t index) { s.readRow(s.unique[s.missing[index]], s.destinations[index]); },
          &tally);
      s.count(tally);
      s.reader->checkUnchanged();
    } catch (const std::exception &error) {
      s.count(tally);
      s.poison(error.what());
      s.throwPoisoned();
    }
    readWait += nanosecondsSince(readStarted);
    const std::lock_guard lock(s.cacheMutex);
    for (size_t index = 0; index < s.missing.size(); ++index)
      s.insert(s.unique[s.missing[index]], s.destinations[index]);
  }

  SparseRowStageResult result;
  result.uniqueRows = uint32_t(s.unique.size());
  result.readRows = uint32_t(s.missing.size());
  result.hitRows = result.uniqueRows - result.readRows;
  result.nanoseconds = nanosecondsSince(started);
  Impl::UseCounters &counters = s.uses[static_cast<uint32_t>(use)];
  counters.stages.fetch_add(1, std::memory_order_relaxed);
  counters.requested.fetch_add(ids.size(), std::memory_order_relaxed);
  counters.unique.fetch_add(result.uniqueRows, std::memory_order_relaxed);
  counters.hits.fetch_add(result.hitRows, std::memory_order_relaxed);
  counters.read.fetch_add(result.readRows, std::memory_order_relaxed);
  counters.latency.add(result.nanoseconds);
  counters.reused.fetch_add(reused);
  counters.waitNanoseconds.fetch_add(readWait);
  counters.prefetchedRows.fetch_add(prefetchHits);
  return result;
}

void SparseRowStore::prefetch(std::span<const uint32_t> ids) {
  Impl &s = *impl_;
  if (ids.empty() || s.poisoned.load(std::memory_order_acquire)) return;
  const std::lock_guard prefetchLock(s.prefetchMutex);
  s.prefetchRequested.fetch_add(ids.size(), std::memory_order_relaxed);
  // Distinct ids within the table, neither cached nor being read ahead, up
  // to the bound.
  if (s.options.boundedWorkingMemory && ids.size() > s.options.prefetchRows) {
    s.prefetchDropped.fetch_add(ids.size() - s.options.prefetchRows, std::memory_order_relaxed);
    ids = ids.first(s.options.prefetchRows);
  }
  s.prefetchDistinct.begin(ids.size());
  auto prefetch = std::make_shared<Impl::Prefetch>();
  {
    const std::lock_guard lock(s.cacheMutex);
    for (const uint32_t id : ids)
      if (id < s.table.rows && s.prefetchDistinct.add(id).second && s.find(id) == kNoSlot &&
          !s.prefetchingIds.contains(id))
        prefetch->ids.push_back(id);
  }
  std::erase_if(s.prefetching, [](const std::shared_ptr<Impl::Prefetch> &each) { return each->ended; });
  const uint64_t room = s.options.prefetchRows > s.prefetchingRows ? s.options.prefetchRows - s.prefetchingRows : 0;
  if (prefetch->ids.size() > room) {
    s.prefetchDropped.fetch_add(prefetch->ids.size() - room, std::memory_order_relaxed);
    prefetch->ids.resize(room);
  }
  if (prefetch->ids.empty()) return;
  if (s.options.boundedWorkingMemory) {
    prefetch->places.reserve(prefetch->ids.size());
    for (size_t row = 0; row < prefetch->ids.size(); ++row) {
      prefetch->places.push_back(s.freePlaces.back());
      s.freePlaces.pop_back();
    }
  } else prefetch->rows.resize(prefetch->ids.size() * s.table.rowBytes);
  s.prefetchingRows += prefetch->ids.size();
  s.prefetchingIds.insert(prefetch->ids.begin(), prefetch->ids.end());
  s.prefetching.push_back(prefetch);
  // The batch ends its prefetch, which the store holds until then.
  Impl::Prefetch *const reading = prefetch.get();
  prefetch->batch = s.options.readers->start(
      reading->ids.size(),
      [&s, reading](size_t index) {
        s.readRow(reading->ids[index], s.prefetchRow(*reading, index));
      },
      std::max(1u, s.options.readers->readers()),
      [&s, reading](const SsdReaders::Batch &batch) { s.endPrefetch(*reading, batch); });
}

void SparseRowStore::drainPrefetch() { impl_->adoptPrefetches(); }

SparseRowStats SparseRowStore::stats() const {
  const Impl &s = *impl_;
  SparseRowStats result;
  result.table = s.table.name;
  result.tableBytes = s.table.rows * s.table.rowBytes;
  result.cacheBytes = committedBytes();
  {
    const std::lock_guard lock(s.cacheMutex);
    result.cacheCapacityRows = s.capacity;
    result.cachedRows = s.used;
    result.evictions = s.evictions;
  }
  for (uint32_t use = 0; use < kSparseRowUses; ++use) {
    const Impl::UseCounters &counters = s.uses[use];
    result.uses[use] = {counters.stages.load(), counters.requested.load(), counters.unique.load(),
                        counters.hits.load(),   counters.read.load(),      counters.latency.snapshot(),
                        counters.reused.load(), counters.waitNanoseconds.load(), counters.prefetchedRows.load()};
  }
  result.readBytes = s.readBytes.load();
  result.readRetries = s.readRetries.load();
  result.readErrors = s.readErrors.load();
  result.prefetchRequestedRows = s.prefetchRequested.load();
  result.prefetchReadRows = s.prefetchRead.load();
  result.prefetchDroppedRows = s.prefetchDropped.load();
  result.read = s.readLatency.snapshot();
  result.poisoned = s.poisoned.load();
  {
    const std::lock_guard lock(s.failureMutex);
    result.failure = s.failure;
  }
  return result;
}

std::unique_ptr<RowReader> openRowReader(std::shared_ptr<const WeightSource> source) {
  return std::make_unique<SourceRowReader>(std::move(source));
}

} // namespace splash::model
