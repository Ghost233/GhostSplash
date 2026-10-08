#pragma once
#include "metal/abi/KvExtent.h"
#include "ops/Normalization.hpp"
#include "ops/PagedKv.hpp"
namespace splash::ops {
// Hyper-connection rows are [token][stream][channel], in bf16. Projection
// weights and linear operations use the existing Linear interface. Norm
// multipliers are the GGUF's already folded (1 + trained weight).
class FlashNext final {
public:
  static void addGroupedRms(metal::CommandGraph &, metal::MetalBuffer input,
                            const NormWeights &, metal::MetalBuffer output,
                            uint32_t width, uint32_t streams, uint32_t rows,
                            float epsilon = 1e-6f);
  static void addHcSilu(metal::CommandGraph &, metal::MetalBuffer input,
                        metal::MetalBuffer output, uint32_t width,
                        uint32_t streams, uint32_t rows);
  static void addHcMix(metal::CommandGraph &, metal::MetalBuffer normalized,
                       metal::MetalBuffer gateLogits, metal::MetalBuffer output,
                       uint32_t width, uint32_t streams, uint32_t rows);
  // injectLogits is [token][stream], output may alias residual.
  static void addHcScatter(metal::CommandGraph &, metal::MetalBuffer residual,
                           metal::MetalBuffer block,
                           metal::MetalBuffer injectLogits,
                           metal::MetalBuffer output, uint32_t width,
                           uint32_t streams, uint32_t rows);
  static void addPleGate(metal::CommandGraph &,
                         metal::MetalBuffer normalizedKey,
                         metal::MetalBuffer normalizedQuery,
                         metal::MetalBuffer value, metal::MetalBuffer output,
                         uint32_t width, uint32_t streams, uint32_t rows);
  // State stores the last (taps-1)*dilation normalized rows in chronological
  // order, initially zeros. Conv weights are F32 [stream*width][tap], with
  // the last tap multiplying the current row. State input and output may
  // alias: the operator encodes convolution before a separate state commit.
  static void
  addPleConv(metal::CommandGraph &, metal::MetalBuffer gated,
             metal::MetalBuffer normalized, metal::MetalBuffer state,
             metal::MetalBuffer convWeights, metal::MetalBuffer output,
             metal::MetalBuffer nextState, uint32_t width, uint32_t streams,
             uint32_t rows, uint32_t taps, uint32_t dilation);
  // Pool contiguous raw indexer key blocks before norm and RoPE. firstBlock
  // addresses both source and output so completed older blocks stay cached.
  static void addQsaPool(metal::CommandGraph &, metal::MetalBuffer rawKeys,
                         metal::MetalBuffer pooledKeys, uint32_t width,
                         uint32_t ratio, uint32_t blocks,
                         uint32_t firstBlock = 0);
  // Split-half partial RoPE; F32 cos/sin [table row][rotaryDimension/2],
  // positions uint32 [rows]. Each pair is read before either side is written,
  // so output may alias input. Unrotated trailing channels are copied.
  static void addRope(metal::CommandGraph &, metal::MetalBuffer input,
                      metal::MetalBuffer cosine, metal::MetalBuffer sine,
                      metal::MetalBuffer positions, metal::MetalBuffer output,
                      uint32_t width, uint32_t heads, uint32_t rows,
                      uint32_t rotaryDimension, uint32_t tableRows);
  // query: BF16 [row][head][width], keys: BF16 [block][width], scores:
  // F32 [row][block]. queryPositions are uint32 absolute sequence indices.
  // Only blocks whose last member is <= query position are scored.
  static void addQsaScores(metal::CommandGraph &, metal::MetalBuffer query,
                           metal::MetalBuffer keys,
                           metal::MetalBuffer queryPositions,
                           metal::MetalBuffer scores, uint32_t width,
                           uint32_t heads, uint32_t rows, uint32_t blocks,
                           uint32_t ratio);
  // Contiguous causal sequence selection: highest-score complete blocks plus
  // every incomplete tail member. Output int32 [row][budget+ratio-1], -1
  // padded. Equal scores select the lower block index deterministically.
  static void addQsaSelect(metal::CommandGraph &, metal::MetalBuffer scores,
                           metal::MetalBuffer queryPositions,
                           metal::MetalBuffer selected, uint32_t rows,
                           uint32_t blocks, uint32_t ratio, uint32_t budget);
  static void addGdnGate(metal::CommandGraph &, metal::MetalBuffer recurrent,
                         metal::MetalBuffer gate, const NormWeights &norm,
                         metal::MetalBuffer output, uint32_t width,
                         uint32_t keyHeads, uint32_t valueHeads,
                         uint32_t tokens, bool tiledOutput,
                         uint32_t gateStride = 0);
  // KV2 Page32, head dimension256. Already normalized/rotated query and
  // unactivated sigmoid gate are [row][query head][dimension]. Uses the
  // pool's resident extent page addresses; never duplicates the KV cache.
  static void addQsaAttention(metal::CommandGraph &, metal::MetalBuffer query,
                              metal::MetalBuffer gate,
                              metal::MetalBuffer pageTable,
                              metal::MetalBuffer selected,
                              metal::MetalBuffer output, SplashKvLayer layer,
                              uint32_t queryHeads, uint32_t rows,
                              uint32_t selectionStride,
                              uint32_t pageTableEntries, kv::Format format);
  static void addBroadcast(metal::CommandGraph &, metal::MetalBuffer input,
                           metal::MetalBuffer output, uint32_t width,
                           uint32_t streams, uint32_t rows);
  static void addAdd(metal::CommandGraph &, metal::MetalBuffer a,
                     metal::MetalBuffer b, metal::MetalBuffer output,
                     uint32_t width, uint32_t rows);
  static void addSiluMul(metal::CommandGraph &, metal::MetalBuffer a,
                         metal::MetalBuffer b, metal::MetalBuffer output,
                         uint32_t width, uint32_t rows);
  static void addSigmoidMul(metal::CommandGraph &, metal::MetalBuffer a,
                            metal::MetalBuffer gate, metal::MetalBuffer output,
                            uint32_t width, uint32_t rows);
  // BF16 vector times fp32 coefficient, added into fp32 accumulation.
  static void addScaledAccum(metal::CommandGraph &, metal::MetalBuffer input,
                             metal::MetalBuffer accumulation, uint32_t width,
                             uint32_t rows, float scale);
  static void addFloatToBf16(metal::CommandGraph &, metal::MetalBuffer input,
                             metal::MetalBuffer output, uint32_t width,
                             uint32_t rows);
};
} // namespace splash::ops
