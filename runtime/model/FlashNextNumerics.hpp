#pragma once
#include "model/Model.hpp"
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
namespace splash::model {
// Numeric output failures affect a lane. IO, GPU and contract failures are
// different exception types and must reach the engine's fatal-error handler.
class FlashNextNumericalFailure final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};
inline void requireFiniteFlashNextLogit(float value, const char *message) {
  if (!std::isfinite(value)) throw FlashNextNumericalFailure(message);
}
template<class Run>
ModelStepResult flashNextLaneResult(uint64_t requestId, Run run) {
  try { return run(); }
  catch (const FlashNextNumericalFailure &error) {
    ModelStepResult result; result.requestId=requestId; result.failure=error.what(); return result;
  }
}
} // namespace splash::model
