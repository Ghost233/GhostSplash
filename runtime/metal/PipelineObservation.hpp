#pragma once

#include "AwakeClock.hpp"

// Test-only observation of the synchronous Metal pipeline creation call.
// Serving builds never construct this scope.
#include <chrono>
#include <cstdio>
#include <exception>
#include <functional>
#include <string>
#include <string_view>
#include <utility>

namespace splash::metal {
class PipelineObservation final {
public:
  using Clock = AwakeClock;
  using Result = std::function<void(double, bool)>;
  explicit PipelineObservation(std::string_view kernel, Result result = {})
      : kernel_(kernel), result_(std::move(result)), exceptions_(std::uncaught_exceptions()) {
    const auto unixMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::fprintf(stderr, "PIPELINE_BEGIN kernel=%s cache=miss system_cache=unknown start_unix_ms=%lld\n",
                 kernel_.c_str(), static_cast<long long>(unixMs));
    std::fflush(stderr);
    started_ = Clock::now();
  }
  ~PipelineObservation() {
    const double seconds = std::chrono::duration<double>(Clock::now() - started_).count();
    const bool success = std::uncaught_exceptions() == exceptions_;
    const auto unixMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::fprintf(stderr, "PIPELINE_END kernel=%s status=%s end_unix_ms=%lld elapsed_ms=%.6f\n",
                 kernel_.c_str(), success ? "ok" : "error", static_cast<long long>(unixMs), seconds * 1000);
    std::fflush(stderr);
    if (result_) result_(seconds, success);
  }
  PipelineObservation(const PipelineObservation &) = delete;
  PipelineObservation &operator=(const PipelineObservation &) = delete;
private:
  std::string kernel_;
  Result result_;
  int exceptions_;
  Clock::time_point started_;
};
} // namespace splash::metal
