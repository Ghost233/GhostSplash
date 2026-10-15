// Persistent FP16 weights: reuse exactly the staged GGUF dequantization,
// then read the row-major device-half matrix directly in the prefill GEMM.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_staged_tile.h"

#define GGUF_CACHE(F, name) \
kernel void gguf_prefill_cache_##name(device uchar *w0 [[buffer(0)]], \
    device uchar *w1 [[buffer(1)]], device uchar *meta [[buffer(2)]], \
    device half *output [[buffer(3)]], constant uint &input_size [[buffer(4)]], \
    uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) { \
  threadgroup half stage[32 * 32]; \
  threadgroup half2 lut[F::Kind == QuantCodebook ? kQuantPairTableEntries : 1]; \
  quant_pair_table<F>(lut, lane, 32); \
  const uint groups = input_size / 32, row = tid / groups, g = tid % groups; \
  const ulong payload = quant_tile_index(row, g, groups); \
  const ulong header = quant_tile_index(row, g / F::MetaGroups, groups / F::MetaGroups); \
  dequant32<F>(F::load(w0 + payload * F::P0, w1 + payload * F::P1), \
               F::loadMeta(meta + header * F::MetaBytes), g % F::MetaGroups, lut, stage + lane * 32); \
  threadgroup_barrier(mem_flags::mem_threadgroup); \
  for (uint i = 0; i < 32; ++i) output[ulong(tid) * 32 + i] = stage[lane * 32 + i]; \
}
QUANT_FORMATS(GGUF_CACHE)
#undef GGUF_CACHE

template <ushort Rows, ushort Cols, ushort KS, GgufEpilogue Ep>
inline void half_prefill(device bfloat *input, device half *weights, device bfloat *output,
                         constant GgufPrefillParams &p, uint2 group, uint simd_group,
                         device bfloat *aux = nullptr) {
  const uint row = group.x * (Rows * GGUF_PREFILL_SIMDGROUPS) + simd_group * Rows, col = group.y * Cols;
  if (row >= p.rows) return;
  // Storage rows are padded to the prefill tile, as in the quantized GEMM.
  auto a = tensor(input + ulong(row) * p.input_size, dextents<int, 2>{int(p.input_size), Rows},
                  array<int, 2>{1, int(p.input_size)});
  auto b = tensor(weights + ulong(col) * p.input_size, dextents<int, 2>{int(p.input_size), Cols},
                  array<int, 2>{1, int(p.input_size)});
  constexpr auto descriptor = matmul2d_descriptor(Rows, Cols, KS, false, true, false,
                                                   matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  auto a0 = a.template slice<KS, Rows>(0, 0);
  auto b0 = b.template slice<KS, Cols>(0, 0);
  auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
  gguf_zero(acc);
  for (uint step = 0; step < p.input_size; step += KS) {
    auto as = a.template slice<KS, Rows>(step, 0);
    auto bs = b.template slice<KS, Cols>(step, 0);
    operation.run(as, bs, acc);
  }
  gguf_elements(acc, [&](uint r, uint c, float value) {
    const ulong at = ulong(row + r) * p.out_stride + p.out_offset + col + c;
    output[at] = gguf_epilogue<Ep>(value, aux, at);
  });
}

#define HALF_PREFILL_THREAD uint2 group [[threadgroup_position_in_grid]], \
    uint simd_group [[simdgroup_index_in_threadgroup]]
#define HALF_PREFILL_EP(Rows, Cols, Step, suffix, name, Ep) \
kernel void gguf_half_prefill##suffix##_##name(device bfloat *input [[buffer(0)]], device half *weights [[buffer(1)]], \
    device bfloat *output [[buffer(2)]], device bfloat *aux [[buffer(3)]], \
    constant GgufPrefillParams &p [[buffer(4)]], HALF_PREFILL_THREAD) { \
  half_prefill<Rows, Cols, Step, Ep>(input, weights, output, p, group, simd_group, aux); \
}
#define HALF_PREFILL(Rows, Cols, Step, suffix) \
kernel void gguf_half_prefill##suffix##_a(device bfloat *input [[buffer(0)]], device half *weights [[buffer(1)]], \
    device bfloat *output [[buffer(2)]], constant GgufPrefillParams &p [[buffer(3)]], HALF_PREFILL_THREAD) { \
  half_prefill<Rows, Cols, Step, EpNone>(input, weights, output, p, group, simd_group); \
} \
HALF_PREFILL_EP(Rows, Cols, Step, suffix, r, EpResidual) \
HALF_PREFILL_EP(Rows, Cols, Step, suffix, g, EpUpWithGate)
HALF_PREFILL(32, 64, 64, )
HALF_PREFILL(32, 64, 128, _k128)
HALF_PREFILL(32, 64, 256, _k256)
HALF_PREFILL(32, 64, 512, _k512)
HALF_PREFILL(32, 64, 2560, _k2560)
HALF_PREFILL(32, 64, 9216, _k9216)
HALF_PREFILL(64, 32, 64, _m64n32)
#undef HALF_PREFILL
#undef HALF_PREFILL_EP
#undef HALF_PREFILL_THREAD
