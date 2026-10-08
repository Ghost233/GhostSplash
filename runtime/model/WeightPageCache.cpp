#include "model/WeightPageCache.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace splash::model {
struct WeightPageLease::Page {
  WeightRange range;
  std::vector<uint8_t> bytes;
};
struct WeightPageCache::State {
  struct Entry {
    std::shared_ptr<WeightPageLease::Page> page;
    std::list<WeightPageKey>::iterator recency;
  };
  uint64_t budget;
  mutable std::mutex mutex;
  std::map<WeightPageKey, Entry> pages;
  std::list<WeightPageKey> recency; // newest first
  WeightCacheTelemetry stats;
  explicit State(uint64_t bytes) : budget(bytes) {}
};
namespace {
class File {
public:
  explicit File(const std::filesystem::path &path) {
    descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) throw std::runtime_error("cannot open weight shard " + path.string() + ": " + std::strerror(errno));
    if (fcntl(descriptor, F_NOCACHE, 1) != 0) {
      const std::string reason = std::strerror(errno); close(descriptor);
      throw std::runtime_error("cannot disable weight shard filesystem cache: " + reason);
    }
  }
  ~File() { close(descriptor); }
  int descriptor;
};
void checkSource(int descriptor, const WeightRange &range) {
  struct stat info{};
  if (fstat(descriptor, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size < 0 ||
      range.offset > uint64_t(info.st_size) || range.bytes > uint64_t(info.st_size) - range.offset)
    throw std::runtime_error("weight shard is truncated: " + range.path.string());
  if (range.source) {
    const auto &s = *range.source;
    if (uint64_t(info.st_size) != s.fileBytes || uint64_t(info.st_dev) != s.device ||
        uint64_t(info.st_ino) != s.inode || info.st_mtimespec.tv_sec != s.modifiedSeconds ||
        info.st_mtimespec.tv_nsec != s.modifiedNanoseconds)
      throw std::runtime_error("weight shard changed since its header was indexed: " + range.path.string());
  }
}
}
const uint8_t *WeightPageLease::data() const noexcept { return page_->bytes.data(); }
uint64_t WeightPageLease::bytes() const noexcept { return page_->bytes.size(); }
WeightPageCache::WeightPageCache(uint64_t byteBudget) : state_(std::make_unique<State>(byteBudget)) {
  if (!byteBudget) throw std::invalid_argument("weight cache budget must be positive");
}
WeightPageCache::~WeightPageCache() = default;
uint64_t WeightPageCache::byteBudget() const noexcept { return state_->budget; }
WeightCacheTelemetry WeightPageCache::telemetry() const {
  std::lock_guard lock(state_->mutex);
  return state_->stats;
}
WeightPageLease WeightPageCache::acquire(const WeightPageKey &key, const WeightRange &range) {
  std::lock_guard lock(state_->mutex);
  auto &s = *state_;
  if (!range.bytes || range.bytes > s.budget || range.bytes > std::numeric_limits<size_t>::max())
    throw std::runtime_error("weight unit exceeds cache budget or is empty: " + key.tensor);
  if (range.offset > uint64_t(std::numeric_limits<off_t>::max()) ||
      range.bytes > uint64_t(std::numeric_limits<off_t>::max()) - range.offset)
    throw std::runtime_error("weight range exceeds file offset bounds: " + key.tensor);
  if (range.source) {
    if (!range.source->complete)
      throw std::runtime_error("cannot load weights from an incomplete GGUF shard: " + range.path.string());
    if (range.offset < range.source->dataOffset || range.offset > range.source->requiredFileBytes ||
        range.bytes > range.source->requiredFileBytes - range.offset || !range.tensorBytes || range.bytes > range.tensorBytes)
      throw std::runtime_error("weight range is outside the indexed GGUF tensor payload: " + key.tensor);
  }
  File file(range.path);
  checkSource(file.descriptor, range);
  auto found = s.pages.find(key);
  if (found != s.pages.end()) {
    if (found->second.page->range != range) throw std::runtime_error("weight cache key reused for a different range: " + key.tensor);
    ++s.stats.hits;
    s.recency.splice(s.recency.begin(), s.recency, found->second.recency);
    return WeightPageLease(found->second.page);
  }
  ++s.stats.misses;
  // Fail without evicting anything when the in-flight working set cannot fit.
  uint64_t pinned = 0;
  for (const auto &[unused, entry] : s.pages)
    if (entry.page.use_count() > 1) pinned += entry.page->bytes.size();
  if (range.bytes > s.budget - pinned)
    throw std::runtime_error("weight cache has insufficient space while pages are pinned: " + key.tensor);
  while (range.bytes > s.budget - s.stats.residentBytes) {
    auto victim = s.recency.end();
    while (victim != s.recency.begin()) {
      --victim;
      if (s.pages.at(*victim).page.use_count() == 1) break;
    }
    auto entry = s.pages.find(*victim);
    s.stats.residentBytes -= entry->second.page->bytes.size();
    s.pages.erase(entry);
    s.recency.erase(victim);
    ++s.stats.evictions;
  }
  auto page = std::make_shared<WeightPageLease::Page>();
  page->range = range;
  page->bytes.resize(range.bytes);
  uint64_t done = 0;
  while (done < range.bytes) {
    const size_t part = std::min<uint64_t>(range.bytes - done, 1u << 20);
    const ssize_t count = pread(file.descriptor, page->bytes.data() + done, part, off_t(range.offset + done));
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) throw std::runtime_error("cannot read weight shard " + range.path.string() +
        (count == 0 ? ": truncated during read" : ": " + std::string(std::strerror(errno))));
    done += uint64_t(count);
    s.stats.bytesRead += uint64_t(count);
  }
  checkSource(file.descriptor, range);
  s.recency.push_front(key);
  try {
    s.pages.emplace(key, State::Entry{page, s.recency.begin()});
  } catch (...) {
    s.recency.pop_front();
    throw;
  }
  s.stats.residentBytes += range.bytes;
  return WeightPageLease(std::move(page));
}
} // namespace splash::model
