// Single-row GGUF GEMV: lanes own adjacent output columns of MDGG0001;
// four simdgroups divide K, then combine FP32 sums in a fixed order. Every
// weight rounds exactly once to FP16, matching the staged GEMM operands.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_staged.h"
#include "metal/kernels/common/gguf_tile.h"

struct SingleTokenParams {
  uint inputs, planeInputs, columns, stride, offset, splits;
};

// The staged dequantizer's operands, kept in registers rather than a shared
// tile. BF16 inputs remain BF16, including values above FP16's finite range.
template <class F>
inline float4 single_chunk_dot(typename F::Chunk q, QuantCoef k, threadgroup half2 *table,
                                device const bfloat *input, ushort chunk) {
  half4 lo, hi;
  if constexpr (F::Kind == QuantLinear) {
    const uint4 codes = F::codes(q);
    lo = half4(staged_linear<F>(codes.x, k.s.x, k.m.x), staged_linear<F>(codes.y, k.s.x, k.m.x));
    hi = half4(staged_linear<F>(codes.z, k.s.y, k.m.y), staged_linear<F>(codes.w, k.s.y, k.m.y));
  } else if constexpr (F::Kind == QuantCodebook) {
    typedef typename F::Scale S;
    const uchar4 codes = as_type<uchar4>(F::indices(q));
    lo = half4(half2(vec<S, 2>(table[codes.x]) * S(k.s.x)), half2(vec<S, 2>(table[codes.y]) * S(k.s.x)));
    hi = half4(half2(vec<S, 2>(table[codes.z]) * S(k.s.y)), half2(vec<S, 2>(table[codes.w]) * S(k.s.y)));
  } else if constexpr (F::Kind == QuantInt8) {
    typedef typename F::Scale S;
    const uint2 values = F::values(q);
    lo = half4(vec<S, 4>(as_type<char4>(values.x)) * S(k.s.x));
    hi = half4(vec<S, 4>(as_type<char4>(values.y)) * S(k.s.y));
  } else if constexpr (F::Kind == QuantFloat8) {
    const uint2 values = F::values(q);
    lo = half4(float4(quant_e4m3_word(values.x)) * k.s.x);
    hi = half4(float4(quant_e4m3_word(values.y)) * k.s.y);
  } else if constexpr (F::Kind == QuantFloat16) {
    typedef vec<typename F::Value, 2> V2;
    const uint4 values = F::values(q);
    lo = half4(float4(float2(as_type<V2>(values.x)), float2(as_type<V2>(values.y))) * k.s.x);
    hi = half4(float4(float2(as_type<V2>(values.z)), float2(as_type<V2>(values.w))) * k.s.y);
  } else {
    const uint2 grid = F::grid(q); const uint signs = F::signs(q);
    lo = half4(float4(as_type<uchar4>(grid.x)) * k.s.x);
    hi = half4(float4(as_type<uchar4>(grid.y)) * k.s.y);
    lo = select(lo, -lo, bool4(signs & 1, signs & 2, signs & 4, signs & 8));
    hi = select(hi, -hi, bool4(signs & 16, signs & 32, signs & 64, signs & 128));
  }
  const float4 xlo = float4(*((device const bfloat4 *)(input + 4 * chunk)));
  const float4 xhi = float4(*((device const bfloat4 *)(input + 16 + 4 * chunk)));
  return fma(float4(lo), xlo, float4(hi) * xhi);
}

template <class F, uint Partitions>
inline float single_quant_sum(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta,
                               constant SingleTokenParams &p, uint column, uint split, uint partition,
                               threadgroup half2 *table) {
  const uint groups = p.planeInputs / 32, units = groups / F::MetaGroups;
  const uint per = p.inputs / 32 / p.splits, begin = split * per, end = begin + per;
  const uint tile = column / QUANT_TILE_ROWS, row = column % QUANT_TILE_ROWS;
  device uchar *a = w0 + (ulong(tile) * groups * QUANT_TILE_ROWS + row) * F::P0;
  device uchar *b = w1 + (ulong(tile) * groups * QUANT_TILE_ROWS + row) * F::P1;
  device uchar *m = meta + (ulong(tile) * units * QUANT_TILE_ROWS + row) * F::MetaBytes;
  uint previous = ~0u;
  typename F::Meta header;
  float4 sum = 0.0f;
  for (uint group = begin + partition; group < end; group += Partitions) {
    const uint unit = group / F::MetaGroups;
    if (unit != previous) { header = F::loadMeta(m + ulong(unit) * QUANT_TILE_ROWS * F::MetaBytes); previous = unit; }
    const auto payload = F::load(a + ulong(group) * QUANT_TILE_ROWS * F::P0,
                                b + ulong(group) * QUANT_TILE_ROWS * F::P1);
    QuantCoef coefficients;
    if constexpr (F::ScaleInChunk) coefficients = F::coef(header, F::chunk(payload, 0));
    else coefficients = F::coef(header, ushort(group % F::MetaGroups));
#pragma unroll
    for (ushort chunk = 0; chunk < 4; ++chunk)
      sum += single_chunk_dot<F>(F::chunk(payload, chunk), coefficients, table, input + group * 32, chunk);
  }
  return ((sum.x + sum.y) + sum.z) + sum.w;
}

template <GgufEpilogue Ep, class Out, bool Partial>
inline void single_store(float sum, device Out *output, device bfloat *aux, device float *partials,
                          constant SingleTokenParams &p, uint column, uint split) {
  const uint at = p.offset + column;
  if constexpr (Partial) partials[ulong(split) * p.stride + at] = sum;
  else output[at] = gguf_epilogue<Ep, Out>(sum, aux, at);
}

template <class F, GgufEpilogue Ep, class Out, bool Partial, uint Columns = 32>
kernel void gguf_single(device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]],
                         device uchar *w1 [[buffer(2)]], device uchar *meta [[buffer(3)]],
                         device Out *output [[buffer(4)]], device bfloat *aux [[buffer(5)]],
                         device float *partials [[buffer(6)]], constant SingleTokenParams &p [[buffer(7)]],
                         uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                         uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  threadgroup half2 table[F::Kind == QuantCodebook ? kQuantPairTableEntries : 1];
  quant_pair_table<F>(table, simdgroup * 32 + lane, 128);
  constexpr uint ColumnSimdgroups = Columns / 32, Partitions = 4 / ColumnSimdgroups;
  static_assert(Columns == 32 || Columns == 64, "single-token column tile");
  const uint localColumn = (simdgroup % ColumnSimdgroups) * 32 + lane;
  const uint partition = simdgroup / ColumnSimdgroups;
  const uint column = group.x * Columns + localColumn;
  const float sum = column < p.columns ?
      single_quant_sum<F, Partitions>(input, w0, w1, meta, p, column, group.y, partition, table) : 0.0f;
  threadgroup float sums[128];
  sums[partition * Columns + localColumn] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (partition == 0 && column < p.columns) {
    float total = sums[localColumn];
#pragma unroll
    for (uint part = 1; part < Partitions; ++part) total += sums[part * Columns + localColumn];
    single_store<Ep, Out, Partial>(total, output, aux, partials, p, column, group.y);
  }
}

// F32 alpha/beta segments have row-major weights, preserved without half
// conversion. Their small output width benefits from distributing K.
template <GgufEpilogue Ep, class Out, bool Partial>
kernel void gguf_single_float(device bfloat *input [[buffer(0)]], device float *weights [[buffer(1)]],
                               device uchar *unused1 [[buffer(2)]], device uchar *unused2 [[buffer(3)]],
                               device Out *output [[buffer(4)]], device bfloat *aux [[buffer(5)]],
                               device float *partials [[buffer(6)]], constant SingleTokenParams &p [[buffer(7)]],
                               uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                               uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  (void)unused1; (void)unused2;
  const uint column = group.x * 32 + lane, per = p.inputs / p.splits;
  float sum = 0.0f;
  if (column < p.columns)
    for (uint k = group.y * per + simdgroup; k < (group.y + 1) * per; k += 4)
      sum = fma(float(input[k]), weights[ulong(column) * p.planeInputs + k], sum);
  threadgroup float sums[128];
  sums[simdgroup * 32 + lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup == 0 && column < p.columns)
    single_store<Ep, Out, Partial>(((sums[lane] + sums[32 + lane]) + sums[64 + lane]) + sums[96 + lane],
                                  output, aux, partials, p, column, group.y);
}

template <GgufEpilogue Ep, class Out>
kernel void gguf_single_reduce(device float *partials [[buffer(0)]], device Out *output [[buffer(1)]],
                                device bfloat *aux [[buffer(2)]], constant SingleTokenParams &p [[buffer(3)]],
                                uint column [[thread_position_in_grid]]) {
  if (column >= p.columns) return;
  const uint at = p.offset + column;
  float sum = partials[at];
  for (uint split = 1; split < p.splits; ++split) sum += partials[ulong(split) * p.stride + at];
  output[at] = gguf_epilogue<Ep, Out>(sum, aux, at);
}

#define SINGLE_BINDINGS \
  (device bfloat *, device uchar *, device uchar *, device uchar *, device bfloat *, device bfloat *, \
   device float *, constant SingleTokenParams &, uint2, uint, uint)
#define SINGLE_FLOAT_BINDINGS \
  (device bfloat *, device float *, device uchar *, device uchar *, device bfloat *, device bfloat *, \
   device float *, constant SingleTokenParams &, uint2, uint, uint)
#define SINGLE_F32_BINDINGS \
  (device bfloat *, device uchar *, device uchar *, device uchar *, device float *, device bfloat *, \
   device float *, constant SingleTokenParams &, uint2, uint, uint)
#define SINGLE_INSTANCES(F, f) \
  template [[host_name("gguf_single_" #f "_a")]] kernel void gguf_single<F, EpNone, bfloat, false> SINGLE_BINDINGS; \
  template [[host_name("gguf_single_" #f "_r")]] kernel void gguf_single<F, EpResidual, bfloat, false> SINGLE_BINDINGS; \
  template [[host_name("gguf_single_" #f "_g")]] kernel void gguf_single<F, EpUpWithGate, bfloat, false> SINGLE_BINDINGS; \
  template [[host_name("gguf_single_" #f "_a_f32")]] kernel void gguf_single<F, EpNone, float, false> SINGLE_F32_BINDINGS; \
  template [[host_name("gguf_single_" #f "_partial")]] kernel void gguf_single<F, EpNone, bfloat, true> SINGLE_BINDINGS;
QUANT_FORMATS(SINGLE_INSTANCES)
// A bounded alternative: twice as many output columns and two K partitions
// per threadgroup. The baseline instances and their arithmetic stay intact.
#define SINGLE_N64_INSTANCES(F, f) \
  template [[host_name("gguf_single_" #f "_n64_a")]] kernel void gguf_single<F, EpNone, bfloat, false, 64> SINGLE_BINDINGS; \
  template [[host_name("gguf_single_" #f "_n64_r")]] kernel void gguf_single<F, EpResidual, bfloat, false, 64> SINGLE_BINDINGS; \
  template [[host_name("gguf_single_" #f "_n64_g")]] kernel void gguf_single<F, EpUpWithGate, bfloat, false, 64> SINGLE_BINDINGS; \
  template [[host_name("gguf_single_" #f "_n64_a_f32")]] kernel void gguf_single<F, EpNone, float, false, 64> SINGLE_F32_BINDINGS; \
  template [[host_name("gguf_single_" #f "_n64_partial")]] kernel void gguf_single<F, EpNone, bfloat, true, 64> SINGLE_BINDINGS;
QUANT_FORMATS(SINGLE_N64_INSTANCES)
#define SINGLE_FLOAT_INSTANCE(name, ep, out, partial) \
  template [[host_name("gguf_single_float_" name)]] kernel void gguf_single_float<ep, out, partial> \
  (device bfloat *, device float *, device uchar *, device uchar *, device out *, device bfloat *, \
   device float *, constant SingleTokenParams &, uint2, uint, uint);
SINGLE_FLOAT_INSTANCE("a", EpNone, bfloat, false)
SINGLE_FLOAT_INSTANCE("r", EpResidual, bfloat, false)
SINGLE_FLOAT_INSTANCE("g", EpUpWithGate, bfloat, false)
SINGLE_FLOAT_INSTANCE("a_f32", EpNone, float, false)
SINGLE_FLOAT_INSTANCE("partial", EpNone, bfloat, true)
#define SINGLE_REDUCE(name, ep, out) \
  template [[host_name("gguf_single_reduce_" name)]] kernel void gguf_single_reduce<ep, out> \
  (device float *, device out *, device bfloat *, constant SingleTokenParams &, uint);
SINGLE_REDUCE("a", EpNone, bfloat)
SINGLE_REDUCE("r", EpResidual, bfloat)
SINGLE_REDUCE("g", EpUpWithGate, bfloat)
SINGLE_REDUCE("a_f32", EpNone, float)
