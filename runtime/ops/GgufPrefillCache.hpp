#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/Gguf.h"
#include "ops/Weights.hpp"

namespace splash::ops {

// Cache values are the staged dequantizer's single FP16 rounding of the
// FP32 coefficients, in row-major [output][input] order.
[[nodiscard]] uint64_t ggufPrefillCacheBytes(const QuantizedSegment &segment);
void writeGgufPrefillCache(metal::MetalBackend &backend, const QuantizedSegment &segment,
                           const metal::MetalBuffer &destination);

// M64N32 is an explicit candidate with the same accumulator element count
// per simdgroup as M32N64; it currently supports only the 64-input step.
enum class GgufHalfPrefillTile : uint8_t { M32N64, M64N32 };

// The persistent cache is read directly by TensorOps, with the existing
// prefill geometry and epilogues: a plain, r residual, g up with BF16 gate.
// inputStep is one of 64, 128, 256, 512, 2560 or 9216 and must divide K.
void addGgufHalfPrefill(metal::CommandGraph &graph, metal::MetalBuffer input,
                        const QuantizedSegment &segment, metal::MetalBuffer output,
                        const GgufPrefillParams &params, char epilogue = 'a', metal::MetalBuffer auxiliary = {},
                        uint32_t inputStep = GGUF_PREFILL_STEP,
                        GgufHalfPrefillTile tile = GgufHalfPrefillTile::M32N64);

} // namespace splash::ops
