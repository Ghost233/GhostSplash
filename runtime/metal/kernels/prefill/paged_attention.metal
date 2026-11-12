#include "metal/kernels/common/paged_attention_tile.h"

// Prefill runs the shared device-operand page loop
// (splash_paged_attention_tile) over the selected query rows of one KV head.

// Prefill split: KV heads (and each one's tile parts,
// splash_attention_tile_part) vary first, then query tiles, then history
// splits.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint HeadDim, uint Window, uint TileRows, typename CacheElement, bool Int8Queries>
inline void splash_prefill_attention_split_phase(
    device bfloat *queries, device float *partials,
    device float *statistics, device const SplashKvPage *page_table,
    constant SplashPrefillAttentionParams &params,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale, uint3 group,
    uint thread_index, threadgroup bfloat *stage) {
  constexpr uint Parts = SPLASH_ATTENTION_TILE_PARTS(QueryHeadsPerKVHead);
  const uint split = group.z;
  const uint kv_head = group.x / Parts;
  const SplashAttentionTile tile =
      splash_attention_tile_part<QueryHeadsPerKVHead, TileRows, Parts, HeadDim>(
          splash_prefill_attention_tile<KVHeads, QueryHeadsPerKVHead, HeadDim, TileRows>(params, kv_head, group.y),
          group.x % Parts, params.split_count);
  if (!tile.active || split >= tile.splits)
    return;
  splash_paged_attention_tile<KVHeads, QueryHeadsPerKVHead,
                              TileRows / Parts, HeadDim, Window, CacheElement, Int8Queries>(
      queries + tile.offset, page_table, params.kv, kv_head, tile.first_position,
      tile.active_rows, tile.splits, split, partials, statistics,
      tile.first_slot + split, scores, probabilities, row_max, row_sum,
      previous_scale, rescale, thread_index, stage);
}

// Prefill reduce: a threadgroup of HeadDim threads per fused row. A reduce
// that adds learned sinks passes the layer's, one per query head, last.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint HeadDim, uint Window, uint TileRows, bool Rotate = false, class... Sinks>
inline void splash_prefill_attention_reduce_phase(
    device const float *partials, device const float *statistics,
    device bfloat *output,
    constant SplashPrefillAttentionParams &params, uint3 group,
    uint thread_index, threadgroup float *weights,
    threadgroup float *group_values, threadgroup float *rotation, Sinks... sinks) {
  constexpr ushort M = TileRows * QueryHeadsPerKVHead;
  constexpr ushort D = HeadDim;
  const uint fused_row = group.y;
  const SplashAttentionTile tile =
      splash_prefill_attention_tile<KVHeads, QueryHeadsPerKVHead, HeadDim, TileRows>(params, group.x, group.z);
  if (!tile.active || fused_row >= M || thread_index >= D)
    return;
  splash_attention_reduce_row<QueryHeadsPerKVHead,
                              TileRows,
                              SPLASH_ATTENTION_TILE_PARTS(QueryHeadsPerKVHead), HeadDim, Window, Rotate>(
      partials, statistics, output, tile, params.split_count, fused_row,
      thread_index, weights, group_values,
      rotation, splash_group_sinks<QueryHeadsPerKVHead>(group.x, sinks)...);
}

#define PAGED_PREFILL_SPLIT(Name, Heads, Group, HeadDim, Window, TileRows, CacheElement, Int8Queries) \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const SplashKvPage *page_table [[buffer(3)]],                     \
      constant SplashPrefillAttentionParams &params [[buffer(4)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint M = Group * TileRows /            \
                       SPLASH_ATTENTION_TILE_PARTS(Group);                     \
    constexpr uint N = SplashKvPageTokens;                                     \
    alignas(16) threadgroup float scores[M * N];                               \
    alignas(16) threadgroup bfloat probabilities[M * N];                       \
    threadgroup float row_max[M];                                              \
    threadgroup float row_sum[M];                                              \
    threadgroup float previous_scale[M];                                       \
    threadgroup atomic_uint rescale;                                           \
    threadgroup bfloat stage[is_same<CacheElement, SplashQ4Block>::value ?      \
                            SplashKvPageTokens * HeadDim : 1];  \
    splash_prefill_attention_split_phase<Heads, Group, HeadDim, Window, TileRows,        \
                                         CacheElement, Int8Queries>(                        \
        queries, partials, statistics, page_table, params, scores,             \
        probabilities, row_max, row_sum, previous_scale, &rescale, group,      \
        thread_index, stage);                                                  \
  }

PAGED_PREFILL_SPLIT(prefill_attention_q8_split, 4, 6, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, int8_t, false)
PAGED_PREFILL_SPLIT(prefill_attention_q8_split_kv2_g8, 2, 8, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, int8_t, false)
PAGED_PREFILL_SPLIT(prefill_attention_q8_split_kv2_g16, 2, 16, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, int8_t, false)
// Four KV heads of 128 dimensions over the whole history, and over a window
// of the 512 keys before each row.
PAGED_PREFILL_SPLIT(prefill_attention_q8_split_kv4_g8_d128, 4, 8, 128, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, int8_t, false)
PAGED_PREFILL_SPLIT(prefill_attention_q8_split_kv4_g8_d128_w512, 4, 8, 128, 512, SPLASH_PREFILL_ATTENTION_TILE_ROWS, int8_t, false)
// INT8 query codes against the INT8 keys, where a split reads enough pages
// (AttentionCalibration::int8QueriesFromPages).
PAGED_PREFILL_SPLIT(prefill_attention_q8q_split, 4, 6, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, int8_t, true)
PAGED_PREFILL_SPLIT(prefill_attention_q8q_split_kv2_g8, 2, 8, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, int8_t, true)
PAGED_PREFILL_SPLIT(prefill_attention_q8q_split_kv2_g16, 2, 16, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, int8_t, true)
PAGED_PREFILL_SPLIT(prefill_attention_q8q_split_kv4_g8_d128, 4, 8, 128, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, int8_t, true)
PAGED_PREFILL_SPLIT(prefill_attention_q8q_split_kv4_g8_d128_w512, 4, 8, 128, 512, SPLASH_PREFILL_ATTENTION_TILE_ROWS, int8_t, true)
// BF16 shares the page loop and reduction, without quantization scales.
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split, 4, 6, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, bfloat, false)
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split_kv2_g8, 2, 8, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, bfloat, false)
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split_kv2_g16, 2, 16, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, bfloat, false)
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split_kv4_g8_d128, 4, 8, 128, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, bfloat, false)
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split_kv4_g8_d128_w512, 4, 8, 128, 512, SPLASH_PREFILL_ATTENTION_TILE_ROWS, bfloat, false)
PAGED_PREFILL_SPLIT(prefill_attention_q8_split_kv2_g4, 2, 4, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, int8_t, false)
PAGED_PREFILL_SPLIT(prefill_attention_q8_split_kv4_g4, 4, 4, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, int8_t, false)
PAGED_PREFILL_SPLIT(prefill_attention_q8_split_kv4_g4_r2, 4, 4, 256, 0, 2, int8_t, false)
PAGED_PREFILL_SPLIT(prefill_attention_q8_split_kv4_g4_r16, 4, 4, 256, 0, 16, int8_t, false)
// BF16 shares the page loop and reduction, without quantization scales.
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split_kv2_g4, 2, 4, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, bfloat, false)
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split_kv4_g4, 4, 4, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, bfloat, false)
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split_kv4_g4_r2, 4, 4, 256, 0, 2, bfloat, false)
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split_kv4_g4_r16, 4, 4, 256, 0, 16, bfloat, false)
PAGED_PREFILL_SPLIT(prefill_attention_q4h_split, 4, 6, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, uchar, false)
PAGED_PREFILL_SPLIT(prefill_attention_q4h_split_kv2_g8, 2, 8, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, uchar, false)
PAGED_PREFILL_SPLIT(prefill_attention_q4h_split_kv2_g4, 2, 4, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, uchar, false)
PAGED_PREFILL_SPLIT(prefill_attention_q4h_split_kv4_g4, 4, 4, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, uchar, false)
PAGED_PREFILL_SPLIT(prefill_attention_q4h_split_kv4_g4_r2, 4, 4, 256, 0, 2, uchar, false)
PAGED_PREFILL_SPLIT(prefill_attention_q4h_split_kv4_g4_r16, 4, 4, 256, 0, 16, uchar, false)
PAGED_PREFILL_SPLIT(prefill_attention_q4_0_split, 4, 6, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, SplashQ4Block, false)
PAGED_PREFILL_SPLIT(prefill_attention_q4_0_split_kv2_g8, 2, 8, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, SplashQ4Block, false)
PAGED_PREFILL_SPLIT(prefill_attention_q4_0_split_kv2_g4, 2, 4, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, SplashQ4Block, false)
PAGED_PREFILL_SPLIT(prefill_attention_q4_0_split_kv4_g4, 4, 4, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, SplashQ4Block, false)
PAGED_PREFILL_SPLIT(prefill_attention_q4_0_split_kv4_g4_r2, 4, 4, 256, 0, 2, SplashQ4Block, false)
PAGED_PREFILL_SPLIT(prefill_attention_q4_0_split_kv4_g4_r16, 4, 4, 256, 0, 16, SplashQ4Block, false)
PAGED_PREFILL_SPLIT(prefill_attention_q4h_split_kv2_g16, 2, 16, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, uchar, false)
PAGED_PREFILL_SPLIT(prefill_attention_q4_0_split_kv2_g16, 2, 16, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, SplashQ4Block, false)
#undef PAGED_PREFILL_SPLIT

#define PAGED_PREFILL_REDUCE(Name, Heads, Group, HeadDim, Window, TileRows, Rotate) \
  kernel void Name(                                                            \
      device const float *partials [[buffer(0)]],                              \
      device const float *statistics [[buffer(1)]],                            \
      device bfloat *output [[buffer(2)]],                                     \
      constant SplashPrefillAttentionParams &params [[buffer(3)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    threadgroup float weights[SplashPrefillMaximumSplits];                     \
    threadgroup float group_values[HeadDim / 32];                              \
    threadgroup float rotation[Rotate ? HeadDim : 1]; \
    splash_prefill_attention_reduce_phase<Heads, Group, HeadDim, Window, TileRows, Rotate>(      \
        partials, statistics, output, params, group, thread_index, weights,    \
        group_values, rotation);                                               \
  }

// A reduce with learned sinks binds the layer's, an F32 per query head,
// before its parameters.
#define PAGED_PREFILL_SINK_REDUCE(Name, Heads, Group, HeadDim, Window)         \
  kernel void Name(                                                            \
      device const float *partials [[buffer(0)]],                              \
      device const float *statistics [[buffer(1)]],                            \
      device bfloat *output [[buffer(2)]],                                     \
      device const float *sinks [[buffer(3)]],                                 \
      constant SplashPrefillAttentionParams &params [[buffer(4)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    threadgroup float weights[SplashPrefillMaximumSplits];                     \
    threadgroup float group_values[HeadDim / 32];                              \
    splash_prefill_attention_reduce_phase<Heads, Group, HeadDim, Window, SPLASH_PREFILL_ATTENTION_TILE_ROWS>(      \
        partials, statistics, output, params, group, thread_index, weights,    \
        group_values, nullptr, sinks);                                                  \
  }
PAGED_PREFILL_REDUCE(prefill_attention_reduce, 4, 6, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, false)
PAGED_PREFILL_REDUCE(prefill_attention_reduce_kv2_g8, 2, 8, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, false)
PAGED_PREFILL_REDUCE(prefill_attention_reduce_kv2_g16, 2, 16, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, false)
PAGED_PREFILL_SINK_REDUCE(prefill_attention_reduce_kv4_g8_d128_sinks, 4, 8, 128, 0)
PAGED_PREFILL_SINK_REDUCE(prefill_attention_reduce_kv4_g8_d128_w512_sinks, 4, 8, 128, 512)
#undef PAGED_PREFILL_SINK_REDUCE
PAGED_PREFILL_REDUCE(prefill_attention_reduce_kv2_g4, 2, 4, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, false)
PAGED_PREFILL_REDUCE(prefill_attention_reduce_kv4_g4, 4, 4, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, false)
PAGED_PREFILL_REDUCE(prefill_attention_reduce_kv4_g4_r2, 4, 4, 256, 0, 2, false)
PAGED_PREFILL_REDUCE(prefill_attention_reduce_kv4_g4_r16, 4, 4, 256, 0, 16, false)
PAGED_PREFILL_REDUCE(prefill_attention_q4h_reduce, 4, 6, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, true)
PAGED_PREFILL_REDUCE(prefill_attention_q4h_reduce_kv2_g8, 2, 8, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, true)
PAGED_PREFILL_REDUCE(prefill_attention_q4h_reduce_kv2_g4, 2, 4, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, true)
PAGED_PREFILL_REDUCE(prefill_attention_q4h_reduce_kv4_g4, 4, 4, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, true)
PAGED_PREFILL_REDUCE(prefill_attention_q4h_reduce_kv4_g4_r2, 4, 4, 256, 0, 2, true)
PAGED_PREFILL_REDUCE(prefill_attention_q4h_reduce_kv4_g4_r16, 4, 4, 256, 0, 16, true)
PAGED_PREFILL_REDUCE(prefill_attention_q4h_reduce_kv2_g16, 2, 16, 256, 0, SPLASH_PREFILL_ATTENTION_TILE_ROWS, true)
#undef PAGED_PREFILL_REDUCE
