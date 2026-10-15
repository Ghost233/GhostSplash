#pragma once
#include "metal/CommandGraph.hpp"
#include "metal/abi/ModernBert.h"
#include <vector>

namespace splash::ops {
// Validated buffer contracts and all dispatch/tile policy for the decision encoder.
class ModernBert final {
public:
  static constexpr uint32_t keyRows(uint32_t rows) { return (rows + 127) / 128 * 128; }
  static constexpr uint32_t queryRows(uint32_t rows) { return (rows + 63) / 64 * 64; }
  static void embedding(metal::CommandGraph &, std::vector<metal::MetalBuffer>, ModernRowsParams);
  static void normalization(metal::CommandGraph &, std::vector<metal::MetalBuffer>, ModernNormParams);
  static void typeEmbedding(metal::CommandGraph &, std::vector<metal::MetalBuffer>, ModernRowsParams);
  static void prepareQkv(metal::CommandGraph &, std::vector<metal::MetalBuffer>, ModernQkvParams);
  static void attention(metal::CommandGraph &, std::vector<metal::MetalBuffer>, ModernAttentionParams);
  static void gatherAttention(metal::CommandGraph &, std::vector<metal::MetalBuffer>, ModernRowsParams);
  static void activation(metal::CommandGraph &, std::vector<metal::MetalBuffer>, ModernActivationParams);
  static void selectFloat(metal::CommandGraph &, std::vector<metal::MetalBuffer>, ModernRowsParams);
  static void selectBfloat(metal::CommandGraph &, std::vector<metal::MetalBuffer>, ModernRowsParams);
  static void classifier(metal::CommandGraph &, std::vector<metal::MetalBuffer>, ModernRowsParams);
  static void repackQ8(metal::MetalBackend &, const metal::MetalBuffer &native,
                       const metal::MetalBuffer &planes, uint32_t rows, uint32_t columns);
};
} // namespace splash::ops
