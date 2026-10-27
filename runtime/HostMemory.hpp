#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

namespace splash {

// Payload capacity only. Allocator/container overhead remains the runtime's
// separate allowance. Host storage is never added to Metal buffer statistics.
struct HostMemoryStats final {
  uint64_t fixedBytes = 0, temporaryBytes = 0, peakBytes = 0;
};
struct HostMemoryAccounting final {
  std::atomic<uint64_t> fixedBytes{0}, temporaryBytes{0}, peakBytes{0};
};

// Declared before the payload it accounts for, so destruction frees payload
// first and retires the account afterwards. Moves transfer one physical owner.
class HostMemoryAllocation final {
public:
  HostMemoryAllocation() noexcept = default;
  HostMemoryAllocation(std::shared_ptr<HostMemoryAccounting> owner, uint64_t bytes, bool temporary) noexcept
      : owner_(std::move(owner)), bytes_(bytes), temporary_(temporary) {
    counter().fetch_add(bytes_, std::memory_order_relaxed);
    const uint64_t total = owner_->fixedBytes.load(std::memory_order_relaxed) +
                           owner_->temporaryBytes.load(std::memory_order_relaxed);
    uint64_t peak = owner_->peakBytes.load(std::memory_order_relaxed);
    while (peak < total && !owner_->peakBytes.compare_exchange_weak(peak, total, std::memory_order_relaxed)) {}
  }
  ~HostMemoryAllocation() { release(); }
  HostMemoryAllocation(const HostMemoryAllocation &) = delete;
  HostMemoryAllocation &operator=(const HostMemoryAllocation &) = delete;
  HostMemoryAllocation(HostMemoryAllocation &&other) noexcept
      : owner_(std::move(other.owner_)), bytes_(std::exchange(other.bytes_, 0)), temporary_(other.temporary_) {}
  HostMemoryAllocation &operator=(HostMemoryAllocation &&other) noexcept {
    if (this != &other) {
      release(); owner_ = std::move(other.owner_); bytes_ = std::exchange(other.bytes_, 0); temporary_ = other.temporary_;
    }
    return *this;
  }
  [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }
private:
  std::atomic<uint64_t> &counter() noexcept { return temporary_ ? owner_->temporaryBytes : owner_->fixedBytes; }
  void release() noexcept {
    if (owner_) { counter().fetch_sub(bytes_, std::memory_order_relaxed); owner_.reset(); }
    bytes_ = 0;
  }
  std::shared_ptr<HostMemoryAccounting> owner_;
  uint64_t bytes_ = 0;
  bool temporary_ = false;
};

using HostMemoryAdmission = std::function<HostMemoryAllocation(
    uint64_t bytes, const std::function<void()> &allocate, const std::function<uint64_t()> &held)>;

} // namespace splash
