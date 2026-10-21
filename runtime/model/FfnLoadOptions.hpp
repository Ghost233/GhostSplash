#pragma once

#include "ops/Weights.hpp"

#include <cstdint>
#include <stdexcept>

namespace splash::model {

enum class FfnWeightMode { FullFp16, Quantized, LimitedFp16 };
struct FfnLoadOptions final {
  FfnWeightMode mode = FfnWeightMode::FullFp16;
  uint64_t budgetBytes = 0;
  ops::FfnPrefillVariant variant = ops::FfnPrefillVariant::Baseline;
  bool trace = false;
};

inline void requireFfnLoadOptions(const FfnLoadOptions &options) {
  if ((options.mode == FfnWeightMode::LimitedFp16) != (options.budgetBytes > 0))
    throw std::invalid_argument("limited FFN cache requires a positive budget, other modes require zero");
}

} // namespace splash::model
