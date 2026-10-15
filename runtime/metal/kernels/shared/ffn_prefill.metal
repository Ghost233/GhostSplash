// Opt-in target FFN candidates. Activations remain BF16; accumulated dots remain FP32.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_staged_tile.h"

#define FFN_FORMATS(X) X(FmtQ4K, q4k) X(FmtQ5K, q5k) X(FmtQ6K, q6k) X(FmtIQ4XS, iq4xs)
#define FFN_BUFFERS device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], \
  device uchar *w1 [[buffer(2)]], device uchar *meta [[buffer(3)]], device bfloat *output [[buffer(4)]]
#define FFN_THREADS uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
  uint sg [[simdgroup_index_in_threadgroup]]
#define FFN_STAGE(F) threadgroup half stage[2 * 32 * 64]; \
  threadgroup half2 table[F::Kind == QuantCodebook ? kQuantPairTableEntries : 1]; \
  quant_pair_table<F>(table, sg * 32 + lane, 128)
#define FFN_PLAIN(F, f) \
  kernel void ffn_quant_m64n32_##f##_a(FFN_BUFFERS, constant GgufPrefillParams &p [[buffer(5)]], FFN_THREADS) { \
    FFN_STAGE(F); \
    const uint first = group.x * 256, rows = p.rows > first ? p.rows - first : 0; \
    gguf_prefill_tile<F, 64, 4, 32, 64>(input + ulong(first) * p.input_size, w0, w1, meta, \
      output + ulong(first) * p.out_stride, p.input_size, group.y * 32, rows, stage, table, lane, sg, p.out_stride, p.out_offset); \
  }
#define FFN_EP(F, f, suffix, ep) \
  kernel void ffn_quant_m64n32_##f##_##suffix(FFN_BUFFERS, device bfloat *aux [[buffer(5)]], \
      constant GgufPrefillParams &p [[buffer(6)]], FFN_THREADS) { \
    FFN_STAGE(F); \
    const uint first = group.x * 256, rows = p.rows > first ? p.rows - first : 0; \
    gguf_prefill_tile<F, 64, 4, 32, 64, ep>(input + ulong(first) * p.input_size, w0, w1, meta, \
      output + ulong(first) * p.out_stride, p.input_size, group.y * 32, rows, stage, table, lane, sg, \
      p.out_stride, p.out_offset, aux + ulong(first) * p.out_stride); \
  }
#define FFN_LEADING(F, f) \
  kernel void ffn_quant_m64n32_##f##_r_leading_inputs(FFN_BUFFERS, device bfloat *aux [[buffer(5)]], \
      constant GgufPrefillLeadingParams &leading [[buffer(6)]], FFN_THREADS) { \
    FFN_STAGE(F); \
    constant GgufPrefillParams &p = leading.prefill; \
    const uint first = group.x * 256, rows = p.rows > first ? p.rows - first : 0; \
    const uint groups = p.input_size / 32, physical = leading.plane_input_size / 32; \
    const ulong before = ulong(group.y * 32 / QUANT_TILE_ROWS) * QUANT_TILE_ROWS; \
    const ulong unread = before * (physical - groups), units = before * (physical / F::MetaGroups - groups / F::MetaGroups); \
    gguf_prefill_tile<F, 64, 4, 32, 64, EpResidual>(input + ulong(first) * p.input_size, \
      w0 + unread * F::P0, w1 + unread * F::P1, meta + units * F::MetaBytes, output + ulong(first) * p.out_stride, \
      p.input_size, group.y * 32, rows, stage, table, lane, sg, p.out_stride, p.out_offset, aux + ulong(first) * p.out_stride); \
  }
#define FFN_M64(F, f) FFN_PLAIN(F, f) FFN_EP(F, f, r, EpResidual) FFN_EP(F, f, g, EpUpWithGate) FFN_LEADING(F, f)
FFN_FORMATS(FFN_M64)

// A first fused candidate shares one lookup table and one double-buffer stage.
// It removes gate scratch traffic but preserves both K walks and BF16 rounding points.
#define FFN_FUSED(F, f) \
  kernel void ffn_quant_fused_m64n32_##f(device bfloat *input [[buffer(0)]], \
      device uchar *gw0 [[buffer(1)]], device uchar *gw1 [[buffer(2)]], device uchar *gm [[buffer(3)]], \
      device uchar *uw0 [[buffer(4)]], device uchar *uw1 [[buffer(5)]], device uchar *um [[buffer(6)]], \
      device bfloat *output [[buffer(7)]], constant GgufPrefillParams &p [[buffer(8)]], FFN_THREADS) { \
    FFN_STAGE(F); \
    const uint first = group.x * 256, rows = p.rows > first ? p.rows - first : 0, origin = group.y * 32; \
    const bool active = sg * 64 < rows; \
    device bfloat *x = input + ulong(first + sg * 64) * p.input_size; \
    auto gate = staged_accumulator<64, 32, 64>(x, p.input_size, stage); \
    auto up = staged_accumulator<64, 32, 64>(x, p.input_size, stage); \
    gguf_zero(gate); gguf_zero(up); \
    gguf_staged_steps<F, 64, 32, 64, 128>(x, gw0, gw1, gm, p.input_size, origin, stage, table, \
      sg * 32 + lane, 0, p.input_size / 64, 0, active, gate); \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    gguf_staged_steps<F, 64, 32, 64, 128>(x, uw0, uw1, um, p.input_size, origin, stage, table, \
      sg * 32 + lane, 0, p.input_size / 64, 0, active, up); \
    if (!active) return; \
    for (ushort i = 0; i < gate.get_capacity(); ++i) { \
      if (!gate.is_valid_element(i)) continue; \
      const auto index = gate.get_multidimensional_index(i); \
      const ulong at = ulong(first + sg * 64 + index[1]) * p.out_stride + p.out_offset + origin + index[0]; \
      output[at] = bfloat(float(bfloat(up[i])) * splash_silu(float(bfloat(gate[i])))); \
    } \
  }
FFN_FORMATS(FFN_FUSED)

// Apply group scale/offset after reduction, and restore the original half weights.
// A 16-code rounding-residual table stays in SIMD registers; no full FP16 image.
template <GgufEpilogue Ep>
inline void q4k_group_scale(device bfloat *input, device uchar *w0, device uchar *meta, device bfloat *output,
    device bfloat *aux, constant GgufQ4KGroupParams &range, uint2 group, uint lane) {
  constant GgufPrefillLeadingParams &leading = range.leading;
  constant GgufPrefillParams &p = leading.prefill;
  const uint column = group.x, first = group.y * GGUF_Q4K_GROUP_ROWS + range.row_offset;
  const uint physical = leading.plane_input_size / 32;
  const uint tile = column / QUANT_TILE_ROWS, local = column % QUANT_TILE_ROWS;
  float sum[GGUF_Q4K_GROUP_ROWS] = {};
  for (uint g = 0; g < p.input_size / 32; ++g) {
    const ulong at = (ulong(tile) * physical * QUANT_TILE_ROWS + ulong(g) * QUANT_TILE_ROWS + local) * FmtQ4K::P0;
    const uint chunk = (lane & 15) / 4, pair = (lane >= 16 ? 2 : 0) + (lane & 3) / 2;
    const uint word = FmtQ4K::loadChunk(w0 + at, w0 + at, ushort(chunk));
    const uint code = (word >> (pair * 4 + (lane & 1) * 16)) & 15u;
    float scale = 0, offset = 0;
    if (!lane) {
      const ulong mt = (ulong(tile) * (physical / FmtQ4K::MetaGroups) * QUANT_TILE_ROWS +
          ulong(g / FmtQ4K::MetaGroups) * QUANT_TILE_ROWS + local) * FmtQ4K::MetaBytes;
      const QuantCoef coef = FmtQ4K::coef(FmtQ4K::loadMeta(meta + mt), ushort(g % FmtQ4K::MetaGroups));
      scale = coef.s.x;
      offset = coef.m.x;
    }
    scale = simd_shuffle(scale, 0);
    offset = simd_shuffle(offset, 0);
    float rounding = 0;
    if (lane < 16) {
      const float affine = fma(scale, float(lane), offset);
      rounding = float(half(affine)) - affine;
    }
    // Reuse the same weight code, coefficients and residual for four rows.
    const float residual = simd_shuffle(rounding, code);
    for (uint r = 0; r < GGUF_Q4K_GROUP_ROWS; ++r) {
      const uint row = first + r;
      if (row >= p.rows) continue;
      const float x = float(input[ulong(row) * p.input_size + g * 32 + lane]);
      const float correction = simd_sum(x * residual);
      const float dotCodes = simd_sum(x * float(code)), dotOnes = simd_sum(x);
      if (!lane) {
        sum[r] = fma(scale, dotCodes, sum[r]);
        sum[r] = fma(offset, dotOnes, sum[r]);
        sum[r] += correction;
      }
    }
  }
  if (!lane) for (uint r = 0; r < GGUF_Q4K_GROUP_ROWS; ++r) {
    const uint row = first + r;
    if (row >= p.rows) continue;
    const ulong at = ulong(row) * p.out_stride + p.out_offset + column;
    output[at] = gguf_epilogue<Ep>(sum[r], aux, at);
  }
}
#define FFN_SCALE_A \
  kernel void ffn_q4k_group_scale_a(FFN_BUFFERS, constant GgufQ4KGroupParams &p [[buffer(5)]], \
      uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    (void)w1; \
    q4k_group_scale<EpNone>(input, w0, meta, output, nullptr, p, group, lane); \
  }
#define FFN_SCALE_EP(suffix, ep) \
  kernel void ffn_q4k_group_scale_##suffix(FFN_BUFFERS, device bfloat *aux [[buffer(5)]], \
      constant GgufQ4KGroupParams &p [[buffer(6)]], \
      uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    (void)w1; \
    q4k_group_scale<ep>(input, w0, meta, output, aux, p, group, lane); \
  }
FFN_SCALE_A
FFN_SCALE_EP(r, EpResidual)
FFN_SCALE_EP(g, EpUpWithGate)
