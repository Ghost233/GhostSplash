#pragma once

#include <algorithm>
#include <array>
#include <optional>
#include <cstdint>
#include <stdexcept>

namespace splash::engine {

struct ExpertPleSample final {
  uint64_t units = 0;
  uint64_t expertMisses = 0, expertGhostHits = 0, expertWaitNanoseconds = 0;
  uint64_t pleReads = 0, pleReused = 0, pleWaitNanoseconds = 0;
  uint64_t pleCachedRows = 0, pleCapacityRows = 0;
  uint64_t workRows = 0;
};

// Runtime store budgets. The managed total is frozen at startup, so changing
// this split cannot borrow from KV or from any other category of the plan.
class ExpertPlePolicy final {
public:
  ExpertPlePolicy(uint64_t pool, uint64_t managed, uint64_t expertFloor, uint64_t pleFloor,
                  uint32_t denominator)
      : minimumPle_(pleFloor), expertFloor_(expertFloor), managedBytes_(managed) {
    if (!pool || managed > pool || expertFloor > managed ||
        pleFloor > managed - expertFloor || (denominator != 10 && denominator != 20))
      throw std::invalid_argument("invalid adaptive expert/PLE budget");
    configuredMaximumPle_ = pool / (denominator + 1);
    maximumPle_ = std::min(configuredMaximumPle_, managed - expertFloor);
    if (minimumPle_ > maximumPle_)
      throw std::invalid_argument("PLE minimum exceeds --expert-ple-ratio ceiling");
    initialPle_ = std::clamp<uint64_t>(512ULL << 20, minimumPle_, maximumPle_);
    target_ = initialPle_;
  }

  [[nodiscard]] uint64_t initialPleBytes() const noexcept {
    return initialPle_;
  }
  [[nodiscard]] uint64_t maximumPleBytes() const noexcept { return maximumPle_; }
  [[nodiscard]] uint64_t minimumPleBytes() const noexcept { return minimumPle_; }

  // A changed local quota is an external capacity change, not an applied
  // experiment. A shrink requires new observations and discards the old
  // target; recovery keeps measured windows without reviving that target.
  // Returns whether an outstanding transfer or trial was invalidated.
  bool constrainManagedBytes(uint64_t managed, uint64_t currentPle) {
    if (managed < expertFloor_ || minimumPle_ > managed - expertFloor_)
      throw std::invalid_argument("local adaptive quota is below the execution floors");
    if (localQuota_ && managed == managedBytes_) return false;
    const bool invalidate = !localQuota_ || managed < managedBytes_;
    localQuota_ = true;
    managedBytes_ = managed;
    maximumPle_ = std::min(configuredMaximumPle_, managed - expertFloor_);
    if (!invalidate) return false;
    abandonTransfer(currentPle);
    reason_ = "quota_changed";
    return true;
  }

  [[nodiscard]] std::optional<uint64_t> pendingTarget() const noexcept {
    return proposalPending_ ? std::optional<uint64_t>(target_) : std::nullopt;
  }
  void abandonTransfer(uint64_t currentPle) noexcept {
    target_ = std::clamp(currentPle, minimumPle_, maximumPle_);
    oldPle_ = target_;
    proposalPending_ = trial_ = rollbackPending_ = false;
    for (auto &w : windows_) { w.valid = false; w.streak = 0; }
    reason_ = "transfer_unavailable";
  }

  // Cumulative counters, sampled separately for prefill and decode. Timers
  // measure wall-clock blocking, never the sum of parallel worker durations.
  [[nodiscard]] std::optional<uint64_t> observe(double now, bool decode, const ExpertPleSample &s,
                                               uint64_t currentPle, std::optional<uint64_t> managed = std::nullopt) {
    if (managed) constrainManagedBytes(*managed, currentPle);
    if (stopped_) return std::nullopt;
    if (proposalPending_) return target_;
    Window &w = windows_[decode];
    if (!w.valid || s.units < w.sample.units) {
      w = {s, now, true, 0, 0};
      if (!trial_) reason_ = now < cooldownUntil_ ? "cooldown" : "warming";
      return std::nullopt;
    }
    if (now - w.since < 10 || s.units - w.sample.units < 32) return std::nullopt;
    const auto previous = w.sample;
    w.sample = s;
    w.since = now;
    const auto delta = [](uint64_t a, uint64_t b) { return a >= b ? a - b : uint64_t{0}; };
    const double expertWait = double(delta(s.expertWaitNanoseconds, previous.expertWaitNanoseconds)) / 1e6;
    const double pleWait = double(delta(s.pleWaitNanoseconds, previous.pleWaitNanoseconds)) / 1e6;
    const double units = double(s.workRows > previous.workRows ? s.workRows - previous.workRows : s.units - previous.units);
    const double cost = (expertWait + pleWait) / units;
    lastCost_ = cost;
    costReady_ = true;
    lastCostDecode_ = decode;
    if (trial_ && decode != trialDecode_) {
      // A new phase cannot validate the previous phase's cost. End that
      // experiment safely instead of blocking the new phase indefinitely.
      trial_ = false;
      rollbackPending_ = true;
      target_ = oldPle_;
      w.streak = 0;
      reason_ = lastDecision_ = "phase_changed_restore";
      proposalPending_ = target_ != currentPle;
      return pendingTarget();
    }
    if (now < cooldownUntil_) { reason_ = "cooldown"; return std::nullopt; }
    if (trial_) {
      // Amortize the resize itself over this evaluation window too. A change
      // that does not demonstrably repay its cost is undone once.
      const bool improved = cost + resizeMilliseconds_ / units < baselineCost_ * 0.95;
      trial_ = false;
      w.streak = 0;
      if (improved) { reason_ = "trial_improved"; return std::nullopt; }
      reason_ = lastDecision_ = "trial_no_gain_restore";
      rollbackPending_ = true;
      target_ = oldPle_;
      proposalPending_ = target_ != currentPle;
      return pendingTarget();
    }
    const auto expertMisses = delta(s.expertMisses, previous.expertMisses);
    const auto reused = delta(s.pleReused, previous.pleReused);
    const auto pleReads = delta(s.pleReads, previous.pleReads);
    const double pleGain = pleReads ? pleWait * double(reused) / double(pleReads) : 0;
    const double expertPressure = expertMisses
        ? expertWait * double(delta(s.expertGhostHits, previous.expertGhostHits)) / double(expertMisses) : 0;
    int direction = 0;
    if (reused >= 8 && pleGain > std::max(1.0, expertPressure * 1.1)) direction = 1;
    else if (expertMisses && expertWait > 1 &&
             ((s.pleCapacityRows && s.pleCachedRows < s.pleCapacityRows / 2) ||
              (!reused && pleWait < expertWait * 0.1))) direction = -1;
    if (!direction) { w.streak = 0; reason_ = "no_capacity_benefit"; return std::nullopt; }
    w.streak = w.direction == direction ? w.streak + 1 : 1;
    w.direction = direction;
    if (w.streak < 2) { reason_ = "confirming_pressure"; return std::nullopt; }
    constexpr uint64_t step = 128ULL << 20;
    target_ = direction > 0 ? std::min(maximumPle_, currentPle + std::min(step, maximumPle_ - currentPle))
                            : std::max(minimumPle_, currentPle - std::min(step, currentPle - minimumPle_));
    if (target_ == currentPle) { reason_ = "at_capacity_bound"; return std::nullopt; }
    oldPle_ = currentPle;
    baselineCost_ = cost;
    trialDecode_ = decode;
    reason_ = lastDecision_ = direction > 0 ? "trial_grow_ple" : "trial_grow_experts";
    w.streak = 0;
    proposalPending_ = true;
    return target_;
  }

  void applied(double now, uint64_t ple, double resizeMilliseconds) noexcept {
    target_ = ple;
    if (proposalPending_) reason_ = lastDecision_;
    proposalPending_ = false;
    resizeMilliseconds_ = resizeMilliseconds;
    cooldownUntil_ = now + (rollbackPending_ ? 60 : 30);
    trial_ = !rollbackPending_;
    if (rollbackPending_) ++rollbacks_;
    rollbackPending_ = false;
    ++adjustments_;
    // A fresh window after the transfer excludes the resize operation.
    for (auto &w : windows_) { w.valid = false; w.streak = 0; }
  }
  void pause() noexcept { for (auto &w : windows_) { w.valid = false; w.streak = 0; } reason_ = "host_growth_paused"; }
  void stop() noexcept { stopped_ = true; reason_ = "resize_failed"; }
  [[nodiscard]] const char *lastDecision() const noexcept { return lastDecision_; }
  [[nodiscard]] bool costReady() const noexcept { return costReady_; }
  [[nodiscard]] bool costIsDecode() const noexcept { return lastCostDecode_; }
  [[nodiscard]] bool trialPending() const noexcept { return trial_; }
  [[nodiscard]] double lastCost() const noexcept { return lastCost_; }
  [[nodiscard]] double referenceCost() const noexcept { return baselineCost_; }
  [[nodiscard]] double resizeMilliseconds() const noexcept { return resizeMilliseconds_; }
  [[nodiscard]] bool stopped() const noexcept { return stopped_; }
  [[nodiscard]] const char *reason() const noexcept { return reason_; }
  [[nodiscard]] uint64_t adjustments() const noexcept { return adjustments_; }
  [[nodiscard]] uint64_t rollbacks() const noexcept { return rollbacks_; }
  [[nodiscard]] uint64_t targetPleBytes() const noexcept { return target_; }
  [[nodiscard]] double cooldownUntil() const noexcept { return cooldownUntil_; }

private:
  struct Window final {
    ExpertPleSample sample;
    double since = 0;
    bool valid = false;
    int direction = 0;
    uint32_t streak = 0;
  };
  std::array<Window, 2> windows_{};
  uint64_t minimumPle_, maximumPle_, expertFloor_, managedBytes_;
  uint64_t configuredMaximumPle_ = 0, initialPle_ = 0, oldPle_ = 0, target_ = 0;
  uint64_t adjustments_ = 0, rollbacks_ = 0;
  double baselineCost_ = 0, resizeMilliseconds_ = 0, cooldownUntil_ = 0, lastCost_ = 0;
  bool localQuota_ = false, proposalPending_ = false;
  bool trial_ = false, trialDecode_ = false, rollbackPending_ = false, stopped_ = false;
  bool costReady_ = false, lastCostDecode_ = false;
  const char *reason_ = "warming", *lastDecision_ = "none";
};

} // namespace splash::engine
