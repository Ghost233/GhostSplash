#include "metal/kernels/common/paged_attention_tile.h"

// Prefill runs the shared device-operand page loop
// (splash_paged_attention_tile) over the selected query rows of one KV head.

// Prefill split: KV heads vary first, then query tiles, then history splits.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint TileRows, typename CacheElement>
inline void splash_prefill_attention_split_phase(
    device bfloat *queries, device float *partials,
    device float *statistics, device const SplashKvPage *page_table,
    constant SplashPrefillAttentionParams &params,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale, uint3 group,
    uint thread_index, threadgroup bfloat *stage) {
  constexpr uint D = SplashKvHeadDimension;
  uint kv_head = group.x;
  uint split = group.z;
  uint tile = group.y;
  uint tile_start = tile * TileRows;
  if (!splash_prefill_attention_contract_valid(params) ||
      kv_head >= KVHeads || split >= params.split_count ||
      tile_start >= params.rows)
    return;
  uint active_rows = min(TileRows, params.rows - tile_start);
  ulong tile_offset = (ulong(kv_head) * params.chunk_stride + tile_start) *
                      QueryHeadsPerKVHead * D;
  ulong slot = (ulong(tile) * KVHeads + kv_head) * params.split_count + split;
  splash_paged_attention_tile<KVHeads, QueryHeadsPerKVHead,
                              TileRows, CacheElement>(
      queries + tile_offset, page_table, params.kv, kv_head,
      params.committed_tokens + tile_start, active_rows, params.split_count, split,
      partials, statistics, slot, scores, probabilities, row_max, row_sum,
      previous_scale, rescale, thread_index, stage);
}

template <uint KVHeads, uint QueryHeadsPerKVHead, uint TileRows, bool Rotate = false>
inline void splash_prefill_attention_reduce_phase(
    device const float *partials, device const float *statistics,
    device bfloat *output,
    constant SplashPrefillAttentionParams &params, uint3 group,
    uint thread_index, threadgroup float *weights,
    threadgroup float *group_values, threadgroup float *rotation) {
  constexpr ushort M = TileRows * QueryHeadsPerKVHead;
  constexpr ushort D = SplashKvHeadDimension;
  uint kv_head = group.x;
  uint fused_row = group.y;
  uint tile = group.z;
  uint tile_start = tile * TileRows;
  if (!splash_prefill_attention_contract_valid(params) ||
      kv_head >= KVHeads || fused_row >= M || thread_index >= D ||
      tile_start >= params.rows)
    return;
  uint active_rows = min(TileRows, params.rows - tile_start);
  ulong tile_offset = (ulong(kv_head) * params.chunk_stride + tile_start) *
                      QueryHeadsPerKVHead * D;
  splash_attention_reduce_row<QueryHeadsPerKVHead,
                              TileRows, Rotate>(
      partials, statistics, output + tile_offset,
      params.committed_tokens + tile_start, active_rows, params.split_count,
      (ulong(tile) * KVHeads + kv_head) * params.split_count, fused_row,
      thread_index, weights, group_values, rotation);
}

#define PAGED_PREFILL_SPLIT(Name, Heads, Group, TileRows, CacheElement)                  \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const SplashKvPage *page_table [[buffer(3)]],                     \
      constant SplashPrefillAttentionParams &params [[buffer(4)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint M = Group * TileRows;             \
    constexpr uint N = SplashKvPageTokens;                                     \
    alignas(16) threadgroup float scores[M * N];                               \
    alignas(16) threadgroup bfloat probabilities[M * N];                       \
    threadgroup float row_max[M];                                              \
    threadgroup float row_sum[M];                                              \
    threadgroup float previous_scale[M];                                       \
    threadgroup atomic_uint rescale;                                           \
    threadgroup bfloat stage[is_same<CacheElement, SplashQ4Block>::value ?      \
                            SplashKvPageTokens * SplashKvHeadDimension : 1];  \
    splash_prefill_attention_split_phase<Heads, Group, TileRows, CacheElement>(          \
        queries, partials, statistics, page_table, params, scores,             \
        probabilities, row_max, row_sum, previous_scale, &rescale, group,      \
        thread_index, stage);                                                  \
  }

PAGED_PREFILL_SPLIT(prefill_attention_q8_split, 4, 6, SPLASH_PREFILL_ATTENTION_TILE_ROWS, int8_t)
PAGED_PREFILL_SPLIT(prefill_attention_q8_split_kv2_g8, 2, 8, SPLASH_PREFILL_ATTENTION_TILE_ROWS, int8_t)
PAGED_PREFILL_SPLIT(prefill_attention_q8_split_kv4_g4, 4, 4, SPLASH_PREFILL_ATTENTION_TILE_ROWS, int8_t)
PAGED_PREFILL_SPLIT(prefill_attention_q8_split_kv4_g4_r2, 4, 4, 2, int8_t)
PAGED_PREFILL_SPLIT(prefill_attention_q8_split_kv4_g4_r16, 4, 4, 16, int8_t)
// BF16 shares the page loop and reduction, without quantization scales.
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split, 4, 6, SPLASH_PREFILL_ATTENTION_TILE_ROWS, bfloat)
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split_kv2_g8, 2, 8, SPLASH_PREFILL_ATTENTION_TILE_ROWS, bfloat)
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split_kv4_g4, 4, 4, SPLASH_PREFILL_ATTENTION_TILE_ROWS, bfloat)
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split_kv4_g4_r2, 4, 4, 2, bfloat)
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split_kv4_g4_r16, 4, 4, 16, bfloat)
PAGED_PREFILL_SPLIT(prefill_attention_q4h_split, 4, 6, SPLASH_PREFILL_ATTENTION_TILE_ROWS, uchar)
PAGED_PREFILL_SPLIT(prefill_attention_q4h_split_kv2_g8, 2, 8, SPLASH_PREFILL_ATTENTION_TILE_ROWS, uchar)
PAGED_PREFILL_SPLIT(prefill_attention_q4h_split_kv4_g4, 4, 4, SPLASH_PREFILL_ATTENTION_TILE_ROWS, uchar)
PAGED_PREFILL_SPLIT(prefill_attention_q4h_split_kv4_g4_r2, 4, 4, 2, uchar)
PAGED_PREFILL_SPLIT(prefill_attention_q4h_split_kv4_g4_r16, 4, 4, 16, uchar)
PAGED_PREFILL_SPLIT(prefill_attention_q4_0_split, 4, 6, SPLASH_PREFILL_ATTENTION_TILE_ROWS, SplashQ4Block)
PAGED_PREFILL_SPLIT(prefill_attention_q4_0_split_kv2_g8, 2, 8, SPLASH_PREFILL_ATTENTION_TILE_ROWS, SplashQ4Block)
PAGED_PREFILL_SPLIT(prefill_attention_q4_0_split_kv4_g4, 4, 4, SPLASH_PREFILL_ATTENTION_TILE_ROWS, SplashQ4Block)
PAGED_PREFILL_SPLIT(prefill_attention_q4_0_split_kv4_g4_r2, 4, 4, 2, SplashQ4Block)
PAGED_PREFILL_SPLIT(prefill_attention_q4_0_split_kv4_g4_r16, 4, 4, 16, SplashQ4Block)
#undef PAGED_PREFILL_SPLIT

#define PAGED_PREFILL_REDUCE(Name, Heads, Group, TileRows, Rotate)                               \
  kernel void Name(                                                            \
      device const float *partials [[buffer(0)]],                              \
      device const float *statistics [[buffer(1)]],                            \
      device bfloat *output [[buffer(2)]],                                     \
      constant SplashPrefillAttentionParams &params [[buffer(3)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    threadgroup float weights[SplashPrefillMaximumSplits];                     \
    threadgroup float group_values[8];                                         \
    threadgroup float rotation[Rotate ? 256 : 1];                                         \
    splash_prefill_attention_reduce_phase<Heads, Group, TileRows, Rotate>(                       \
        partials, statistics, output, params, group, thread_index, weights,    \
        group_values, rotation);                                               \
  }
PAGED_PREFILL_REDUCE(prefill_attention_reduce, 4, 6, SPLASH_PREFILL_ATTENTION_TILE_ROWS, false)
PAGED_PREFILL_REDUCE(prefill_attention_reduce_kv2_g8, 2, 8, SPLASH_PREFILL_ATTENTION_TILE_ROWS, false)
PAGED_PREFILL_REDUCE(prefill_attention_reduce_kv4_g4, 4, 4, SPLASH_PREFILL_ATTENTION_TILE_ROWS, false)
PAGED_PREFILL_REDUCE(prefill_attention_reduce_kv4_g4_r2, 4, 4, 2, false)
PAGED_PREFILL_REDUCE(prefill_attention_reduce_kv4_g4_r16, 4, 4, 16, false)
PAGED_PREFILL_REDUCE(prefill_attention_q4h_reduce, 4, 6, SPLASH_PREFILL_ATTENTION_TILE_ROWS, true)
PAGED_PREFILL_REDUCE(prefill_attention_q4h_reduce_kv2_g8, 2, 8, SPLASH_PREFILL_ATTENTION_TILE_ROWS, true)
PAGED_PREFILL_REDUCE(prefill_attention_q4h_reduce_kv4_g4, 4, 4, SPLASH_PREFILL_ATTENTION_TILE_ROWS, true)
PAGED_PREFILL_REDUCE(prefill_attention_q4h_reduce_kv4_g4_r2, 4, 4, 2, true)
PAGED_PREFILL_REDUCE(prefill_attention_q4h_reduce_kv4_g4_r16, 4, 4, 16, true)
#undef PAGED_PREFILL_REDUCE
