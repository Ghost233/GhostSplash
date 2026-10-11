#pragma once

#include "metal/MetalBackend.hpp"
#include <map>
#include <string>
#include <tuple>

namespace splash::bench {
// Updated only by the backend's successful-submission callback, on its submitter.
// This records execution routing, not per-dispatch GPU durations.
class FfnTrace final {
public:
  using Key = std::tuple<std::string, std::string, uint32_t>;
  using Counts = std::map<Key, uint64_t>;
  void submitted(const metal::Command &command) {
    for (const auto &dispatch : command.dispatches)
      if (!dispatch.attribution.empty())
        ++counts_[{dispatch.attribution, dispatch.pipelineName, dispatch.logicalRows}];
  }
  struct Stage final { uint64_t dispatches = 0, ticks = 0; double estimatedSeconds = 0; };
  void timed(const metal::CommandTimestampProfile &profile) {
    if (!profile.valid) { timingError_ = profile.error; return; }
    if (profile.dispatches.empty()) return;
    const uint64_t first = profile.dispatches.front().beginTicks, last = profile.dispatches.back().endTicks;
    if (last <= first) { timingError_ = "empty timestamp interval"; return; }
    const double scale = profile.gpuSpanSeconds / double(last-first);
    gpuSpanSeconds_ += profile.gpuSpanSeconds;
    encoderBoundary_ |= profile.encoderBoundarySamples;
    for (size_t i=0; i<profile.dispatches.size(); ++i) {
      const auto &item = profile.dispatches[i];
      auto &stage = stages_[item.attribution.empty() ? "other_gpu_work" : item.attribution];
      ++stage.dispatches; stage.ticks += item.endTicks-item.beginTicks;
      stage.estimatedSeconds += double(item.endTicks-item.beginTicks)*scale;
      if (i && item.followsEventWait) {
        auto &wait = stages_["event_wait_and_encoder_gap"];
        const uint64_t ticks = item.beginTicks-profile.dispatches[i-1].endTicks;
        ++wait.dispatches; wait.ticks += ticks; wait.estimatedSeconds += double(ticks)*scale;
      }
    }
  }
  [[nodiscard]] const std::map<std::string,Stage> &stages() const noexcept { return stages_; }
  [[nodiscard]] const std::string &timingError() const noexcept { return timingError_; }
  [[nodiscard]] bool encoderBoundarySamples() const noexcept { return encoderBoundary_; }
  [[nodiscard]] double gpuSpanSeconds() const noexcept { return gpuSpanSeconds_; }
  [[nodiscard]] Counts snapshot() const { return counts_; }
  [[nodiscard]] Counts since(const Counts &before) const {
    Counts result;
    for (const auto &[key, count] : counts_) {
      const auto old = before.find(key);
      const uint64_t delta = count - (old == before.end() ? 0 : old->second);
      if (delta) result[key] = delta;
    }
    return result;
  }
private:
  Counts counts_;
  std::map<std::string,Stage> stages_;
  std::string timingError_;
  double gpuSpanSeconds_ = 0;
  bool encoderBoundary_ = false;
};
} // namespace splash::bench
