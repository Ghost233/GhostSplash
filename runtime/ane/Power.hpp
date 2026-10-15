#pragma once
#include <cstdint>
#include <algorithm>
#include <charconv>
#include <span>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <string_view>
#include <memory>
#include <optional>
#include <string>
namespace splash::ane {
inline std::optional<double> powerWatts(int64_t previous, int64_t current,
                                      std::string_view unit, double seconds) {
  if (previous < 0 || current < previous || !std::isfinite(seconds) || seconds <= 0)
    return std::nullopt;
  const double scale = unit == "J" ? 1 : unit == "mJ" ? 1e-3 :
                       unit == "uJ" ? 1e-6 : unit == "nJ" ? 1e-9 : 0;
  if (!scale) return std::nullopt;
  return double(current - previous) * scale / seconds;
}
inline std::optional<double> relativePowerPercent(double watts, double reference) {
  if (!std::isfinite(watts) || watts < 0 || !std::isfinite(reference) || reference <= 0) return std::nullopt;
  const double ratio = watts / reference * 100;
  return std::isfinite(ratio) ? std::optional<double>(ratio) : std::nullopt;
}
struct PowerSample {
  std::optional<double> watts;
  double measuredAtSeconds = 0;
  uint64_t sequence = 0;
  double intervalSeconds = 0;
  std::optional<double> quietSeconds = std::nullopt;
};
struct PowerReferenceSetting {
  enum class Mode { Off, Automatic, Manual };
  Mode mode = Mode::Off;
  double manualWatts = 0;
  static PowerReferenceSetting parse(std::string_view value) {
    PowerReferenceSetting result;
    if (value == "off") return result;
    if (value == "auto") { result.mode = Mode::Automatic; return result; }
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result.manualWatts);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
        !std::isfinite(result.manualWatts) || result.manualWatts <= 0)
      throw std::invalid_argument("--ane-power-reference must be off, auto or positive watts");
    result.mode = Mode::Manual;
    return result;
  }
};
struct PowerReference {
  bool valid = false;
  std::string source, reason;
  double referenceWatts = 0;
  std::optional<double> peakWatts = std::nullopt, steadyWatts = std::nullopt, backgroundWatts = std::nullopt;
  double loadSeconds = 0, share = 0;
  uint32_t samples = 0, rows = 0;
};
// Only complete loaded intervals count. This is a workload reference, never a hardware rating.
inline PowerReference summarizePowerReference(std::span<const PowerSample> samples,
                                              double begin, double end, double background) {
  PowerReference result;
  result.source = "auto_ffn";
  result.loadSeconds = end - begin;
  if (std::isfinite(background) && background >= 0) result.backgroundWatts = background;
  std::vector<double> watts;
  uint64_t sequence = 0;
  for (const auto &s : samples) {
    if (!s.watts || !std::isfinite(*s.watts) || *s.watts < 0 || s.sequence <= sequence ||
        !std::isfinite(s.intervalSeconds) || s.intervalSeconds <= 0 || s.intervalSeconds > 2 ||
        !std::isfinite(s.measuredAtSeconds) || s.measuredAtSeconds > end ||
        s.measuredAtSeconds - s.intervalSeconds < begin) continue;
    watts.push_back(*s.watts); sequence = s.sequence;
  }
  result.samples = static_cast<uint32_t>(watts.size());
  if (!watts.empty()) {
    std::sort(watts.begin(), watts.end());
    result.peakWatts = watts.back(); result.steadyWatts = watts[watts.size() / 2];
  }
  if (!result.backgroundWatts || background > 0.05) result.reason = "background ANE load was not quiet";
  else if (watts.size() < 3) result.reason = "fewer than three complete valid high-load samples";
  else if (!(watts.back() > background + 0.01)) result.reason = "high-load power did not rise above background";
  else { result.valid = true; result.referenceWatts = watts.back(); }
  return result;
}
// Passive whole-device telemetry. Missing or invalid readings stay unknown.
class PowerMonitor final {
public:
  explicit PowerMonitor(double maximumIdleWatts = 0.05);
  ~PowerMonitor();
  void active(bool value);
  [[nodiscard]] PowerSample latest() const;
  [[nodiscard]] std::vector<PowerSample> samples() const;
  [[nodiscard]] std::string reason() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace splash::ane
