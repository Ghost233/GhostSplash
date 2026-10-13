#pragma once
#include "ane/Power.hpp"
#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
namespace splash::ops::ane_ffn {
struct RecoverySetting {
  bool enabled = false;
  double maximumIdleWatts = 0.05;
  double cooldownSeconds = 60;
  double quietSeconds = 10;
  std::optional<double> maximumPowerPercent = std::nullopt;
  std::optional<double> referenceWatts = std::nullopt;
  [[nodiscard]] std::optional<double> powerThreshold() const noexcept {
    if (!maximumPowerPercent) return maximumIdleWatts;
    if (!std::isfinite(*maximumPowerPercent) || *maximumPowerPercent < 0 || *maximumPowerPercent > 100 ||
        !referenceWatts || !std::isfinite(*referenceWatts) || *referenceWatts <= 0) return std::nullopt;
    return *referenceWatts * (*maximumPowerPercent / 100.0);
  }

};
struct RecoveryReport {
  std::string phase, reason;
  std::optional<double> watts;
  double cooldownRemaining = 0, quietSeconds = 0;
  uint64_t attempts = 0, restored = 0;
  std::optional<double> relativePowerPercent = std::nullopt;
};
// Engine-thread policy; distinct, fresh one-second samples must cover a quiet window.
class RecoveryGate final {
public:
  explicit RecoveryGate(RecoverySetting setting) : setting_(setting) {}
  void stopped(double now) { next_ = now + delay_; quiet_ = 0; sequence_ = 0; last_ = 0; }
  void failed(double now) {
    delay_ = std::min(300.0, delay_ * 2);
    stopped(now);
  }
  void stable() noexcept { delay_ = setting_.cooldownSeconds; }
  [[nodiscard]] bool ready(double now, const ane::PowerSample &sample) {
    const auto threshold = setting_.powerThreshold();
    if (!setting_.enabled || !threshold) return false;
    if (!sample.watts || !std::isfinite(*sample.watts) || *sample.watts < 0 ||
        !std::isfinite(sample.measuredAtSeconds) || !std::isfinite(sample.intervalSeconds) ||
        sample.measuredAtSeconds > now || now - sample.measuredAtSeconds > 2 ||
        sample.intervalSeconds <= 0 || sample.intervalSeconds > 2 ||
        *sample.watts > *threshold) {
      quiet_ = 0;
      return false;
    }
    if (sample.quietSeconds) {
      if (!std::isfinite(*sample.quietSeconds) || *sample.quietSeconds < 0) { quiet_ = 0; return false; }
      quiet_ = *sample.quietSeconds;
    } else if (sample.sequence != sequence_) {
      if (last_ && (sample.measuredAtSeconds <= last_ || sample.measuredAtSeconds - last_ > 2)) quiet_ = 0;
      quiet_ += sample.intervalSeconds;
      sequence_ = sample.sequence;
      last_ = sample.measuredAtSeconds;
    }
    return now >= next_ && quiet_ >= setting_.quietSeconds;
  }
  [[nodiscard]] double nextAttempt() const noexcept { return next_; }
  [[nodiscard]] double quietSeconds() const noexcept { return quiet_; }
private:
  RecoverySetting setting_;
  double delay_ = setting_.cooldownSeconds, next_ = 0, quiet_ = 0, last_ = 0;
  uint64_t sequence_ = 0;
};
} // namespace splash::ops::ane_ffn
