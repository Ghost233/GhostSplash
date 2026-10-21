#pragma once

#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/kv_codec.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/kv_paging.h"
#include <metal_stdlib>

using namespace metal;

constant uint SplashChunkMaximumRows = SPLASH_PREFILL_TOKEN_BUDGET;
constant uint SplashChunkMaximumPhysicalTokens =
    SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS;

inline bool
splash_chunk_contract_valid(constant SplashChunkedPrefillParams &params) {
  uint required_pages = (params.committed_tokens + params.chunk_tokens +
                         SplashKvPageTokens - 1) /
                        SplashKvPageTokens;
  return params.committed_tokens <= SplashChunkMaximumPhysicalTokens &&
         params.chunk_tokens > 0 &&
         params.chunk_tokens <= SplashChunkMaximumRows &&
         params.committed_tokens + params.chunk_tokens <=
             SplashChunkMaximumPhysicalTokens &&
         params.chunk_stride >= params.chunk_tokens &&
         params.chunk_stride <= SplashChunkMaximumRows &&
         params.chunk_stride % SPLASH_TARGET_KV_BLOCK_TOKENS == 0 &&
         params.page_table_entries >= required_pages &&
         params.kv.extent_pages > 0;
}

inline ulong splash_current_key_index(uint stride, uint head, uint token,
                                        uint dimension) {
  return (ulong(head) * stride + token) * SplashKvHeadDimension + dimension;
}

inline ulong splash_current_value_index(uint stride, uint head, uint token,
                                          uint dimension) {
  return (ulong(head) * SplashKvHeadDimension + dimension) * stride + token;
}

// The scratch of a store entry's scale reduction, by CacheElement: eight
// simdgroup maxima for INT8; BF16 copies the source bits and has none.
#define SPLASH_STORE_MAXIMA_int8_t threadgroup float maxima[8];
#define SPLASH_STORE_MAXIMA_bfloat threadgroup float *maxima = nullptr;
#define SPLASH_STORE_MAXIMA_uchar threadgroup float maxima[8];
#define SPLASH_STORE_MAXIMA_SplashQ4Block threadgroup float *maxima = nullptr;

// One lane per dimension stores a current row in its final page slot.
// INT8 derives a per-row scale; BF16 copies the original bits. Slots are
// addressed inside the head's slab of the page: the page index functions at
// page zero and head zero.
template <uint KVHeads, typename CacheElement, bool ValuesTokenMajor = false>
__attribute__((always_inline)) inline void splash_store_kv_row(
    device const bfloat *chunk_keys, device const bfloat *chunk_values,
    device const SplashKvPage *page_table,
    constant SplashChunkedPrefillParams &params, threadgroup float *maxima,
    bool value_tensor, uint head, uint chunk_token, uint dimension,
    uint simd_lane, uint simd_group) {
  uint logical_token = params.committed_tokens + chunk_token;
  const SplashKvPage page = page_table[logical_token / SplashKvPageTokens];
  const SplashKvPageTensors<CacheElement> slab =
      SplashKvAddressing<KVHeads, CacheElement>(params.kv, head).page(page);
  uint page_token = logical_token % SplashKvPageTokens;
  ulong source_index =
      value_tensor ? splash_current_value_index(params.chunk_stride, head,
                                                  chunk_token, dimension)
                   : splash_current_key_index(params.chunk_stride, head,
                                                chunk_token, dimension);
  bfloat source = value_tensor ? chunk_values[source_index] : chunk_keys[source_index];

  splash_kv_store_row<CacheElement, ValuesTokenMajor>(slab, source, maxima, value_tensor, page_token,
                      dimension, simd_lane, simd_group);
}
