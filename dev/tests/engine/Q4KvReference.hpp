#pragma once

// Independent scalar reference for standard GGML Q4_0 KV blocks. The
// convention is quantize_row_q4_0_ref in llama.cpp (MIT; THIRD_PARTY_NOTICES).
#include "metal/abi/KvExtent.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <span>

namespace splash::test {

inline SplashQ4Block q4KvReference(std::span<const float, 32> values) {
  float maximum = 0, signedMaximum = 0;
  for (float value : values)
    if (std::abs(value) > maximum) {
      maximum = std::abs(value);
      signedMaximum = value;
    }
  const float d = signedMaximum / -8.0f;
  const float inverse = d != 0 ? 1.0f / d : 0;
  SplashQ4Block block{};
  const __fp16 half = static_cast<__fp16>(d);
  std::memcpy(&block.d, &half, 2);
  for (uint32_t j = 0; j < 16; ++j) {
    const float first = values[j] * inverse;
    const float second = values[j + 16] * inverse;
    const int low = std::clamp(int(first + 8.5f), 0, 15);
    const int high = std::clamp(int(second + 8.5f), 0, 15);
    block.qs[j] = uint8_t(low | (high << 4));
  }
  return block;
}

inline float q4KvReferenceValue(const SplashQ4Block &block, uint32_t dimension) {
  __fp16 half;
  std::memcpy(&half, &block.d, 2);
  const uint32_t code = (block.qs[dimension % 16] >> (dimension >= 16 ? 4 : 0)) & 15;
  return float(half) * (int(code) - 8);
}

} // namespace splash::test
