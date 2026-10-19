#pragma once
#include "metal/CommandGraph.hpp"
#include "metal/abi/D3.h"

namespace splash::ops {
class D3 final {
public:
  static uint32_t storageRows(uint32_t rows) { return (rows + 255) / 128 * 128; }
  static void attention(metal::CommandGraph &, const metal::MetalBuffer &queries,
                        const metal::MetalBuffer &keys, const metal::MetalBuffer &values,
                        const metal::MetalBuffer &output, D3AttentionParams,
                        const metal::MetalBuffer &scales = {});
  static void zero(metal::CommandGraph &, const metal::MetalBuffer &);
  static void transposeValues(metal::CommandGraph &, const metal::MetalBuffer &,
                               const metal::MetalBuffer &, D3AttentionParams);
  static void quantizeKv(metal::CommandGraph &, const metal::MetalBuffer &keys,
                         const metal::MetalBuffer &values, const metal::MetalBuffer &decodedValues,
                         const metal::MetalBuffer &codes, const metal::MetalBuffer &scales,
                         D3AttentionParams);
};
}
