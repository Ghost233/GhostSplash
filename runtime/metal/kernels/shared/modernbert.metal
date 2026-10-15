#include "metal/abi/ModernBert.h"
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Same erf approximation as the production vision GEMM; error < 1.5e-7.
inline float modern_erf(float x) {
  float sign = x < 0 ? -1.0f : 1.0f;
  x = fabs(x);
  float t = 1.0f / (1.0f + 0.3275911f * x);
  float p = t * (0.254829592f + t * (-0.284496736f +
      t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  return sign * (1.0f - p * precise::exp(-x * x));
}
inline float modern_gelu(float x) {
  return 0.5f * x * (1.0f + modern_erf(x * 0.7071067811865476f));
}

kernel void modern_embed_q8(
    device const uint *tokens [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *hidden [[buffer(2)]], constant ModernRowsParams &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  for (uint c = tid; c < p.stride; c += 256) {
    float v = 0;
    if (c < p.width) {
      device const uchar *b = weights + (ulong(tokens[row]) * (p.width / 32) + c / 32) * 34;
      half s = *reinterpret_cast<device const half *>(b);
      v = float(s) * float(reinterpret_cast<device const char *>(b + 2)[c % 32]);
    }
    hidden[ulong(row) * p.stride + c] = v;
  }
}

// Residual + bias + LayerNorm, keeping the residual stream FP32. Padding does
// not participate in mean or variance and is zeroed for the quantized GEMM.
kernel void modern_norm(
    device float *hidden [[buffer(0)]], device const float *delta [[buffer(1)]],
    device const float *delta_bias [[buffer(2)]], device const float *weight [[buffer(3)]],
    device const float *bias [[buffer(4)]], device bfloat *output [[buffer(5)]],
    constant ModernNormParams &p [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[8];
  float sum = 0;
  for (uint c = tid; c < p.width; c += 256) {
    ulong at = ulong(row) * p.stride + c;
    float x = hidden[at];
    if (p.add_delta) x += delta[ulong(row) * p.delta_stride + c] + delta_bias[c];
    hidden[at] = x;
    sum += x;
  }
  sum = simd_sum(sum);
  if (lane == 0) sums[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mean = 0;
  for (uint i = 0; i < 8; ++i) mean += sums[i];
  mean /= float(p.width);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float square = 0;
  for (uint c = tid; c < p.width; c += 256) {
    float x = hidden[ulong(row) * p.stride + c] - mean;
    square += x * x;
  }
  square = simd_sum(square);
  if (lane == 0) sums[sg] = square;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float variance = 0;
  for (uint i = 0; i < 8; ++i) variance += sums[i];
  float inverse = rsqrt(variance / float(p.width) + p.epsilon);
  for (uint c = tid; c < p.stride; c += 256) {
    float x = c < p.width ? (hidden[ulong(row) * p.stride + c] - mean) * inverse * weight[c] + bias[c] : 0;
    output[ulong(row) * p.stride + c] = bfloat(x);
    if (p.normalize_hidden) hidden[ulong(row) * p.stride + c] = x;
  }
}

kernel void modern_type_add(device float *hidden [[buffer(0)]],
    device const float *types [[buffer(1)]], constant ModernRowsParams &p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  uint row = index / p.stride, c = index % p.stride;
  if (row < p.rows && c < p.width) hidden[index] += types[p.kind * p.width + c];
}

// Q uses only selected option positions in the final decision block. K/V
// keep the entire sequence; the encoder uses NEOX RoPE, the head uses none.
kernel void modern_qkv_prepare(
    device const float *packed [[buffer(0)]], device const float *bias [[buffer(1)]],
    device const uint *positions [[buffer(2)]], device bfloat *queries [[buffer(3)]],
    device bfloat *keys [[buffer(4)]], device bfloat *values [[buffer(5)]],
    constant ModernQkvParams &p [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
    uint c [[thread_index_in_threadgroup]]) {
  if (c >= 64) return;
  uint token = group.x, head = group.y;
  uint other = c < 32 ? c + 32 : c - 32;
  float angle = p.theta ? float(token) * pow(p.theta, -float(c % 32) / 32.0f) : 0;
  float co = cos(angle), si = sin(angle), sign = c < 32 ? -1.0f : 1.0f;
  float k = 0, v = 0;
  if (token < p.tokens) {
    ulong base = ulong(token) * p.source_stride;
    k = packed[base + p.width + head * 64 + c] + bias[p.width + head * 64 + c];
    float k2 = packed[base + p.width + head * 64 + other] + bias[p.width + head * 64 + other];
    k = k * co + sign * k2 * si;
    v = packed[base + 2 * p.width + head * 64 + c] + bias[2 * p.width + head * 64 + c];
  }
  keys[(ulong(head) * p.padded_tokens + token) * 64 + c] = bfloat(k);
  values[(ulong(head) * p.padded_tokens + token) * 64 + c] = bfloat(v);
  if (token < p.padded_queries) {
    float q = 0;
    if (token < p.query_rows) {
      uint source = p.selected ? positions[token] : token;
      ulong base = ulong(source) * p.source_stride;
      q = packed[base + head * 64 + c] + bias[head * 64 + c];
      float q2 = packed[base + head * 64 + other] + bias[head * 64 + other];
      float a = p.theta ? float(source) * pow(p.theta, -float(c % 32) / 32.0f) : 0;
      q = q * cos(a) + sign * q2 * sin(a);
    }
    queries[(ulong(head) * p.padded_queries + token) * 64 + c] = bfloat(q);
  }
}

// MPP flash attention derived from production vision_attention. Scores and
// running softmax are FP32; only one 64x128 score tile is materialized.
kernel void modern_attention(
    device bfloat *queries [[buffer(0)]], device bfloat *keys [[buffer(1)]],
    device bfloat *values [[buffer(2)]], device bfloat *output [[buffer(3)]],
    constant ModernAttentionParams &p [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr ushort M = 64, N = 128, D = 64, R = 8;
  threadgroup float storage[M * N];
  threadgroup bfloat *probs = reinterpret_cast<threadgroup bfloat *>(storage);
  threadgroup float *rowSum = storage + M * N / 2;
  threadgroup float *previous = rowSum + M;
  float maxima[R], sums[R];
  for (ushort i = 0; i < R; ++i) { maxima[i] = -INFINITY; sums[i] = 0; }
  uint head = group.y, start = group.x * M;
  device bfloat *q = queries + (ulong(head) * p.padded_queries + start) * D;
  device bfloat *k = keys + ulong(head) * p.padded_tokens * D;
  device bfloat *v = values + ulong(head) * p.padded_tokens * D;
  auto qt = tensor(q, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  constexpr auto qkDescriptor = matmul2d_descriptor(M, N, D, false, true, false);
  matmul2d<qkDescriptor, execution_simdgroups<8>> qk;
  constexpr auto pvDescriptor = matmul2d_descriptor(M, D, N, false, false, false);
  matmul2d<pvDescriptor, execution_simdgroups<8>> pv;
  auto q0 = qt.slice<D, M>(0, 0);
  auto pt = tensor(probs, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto p0 = pt.slice<N, M>(0, 0);
  // V follows the production vision attention layout: [head, token, channel].
  // MPP consumes contiguous channels of each key token.
  auto vTensor = tensor(v, dextents<int, 2>{D, N}, array<int, 2>{1, D});
  auto firstV = vTensor.slice<D, N>(0, 0);
  auto running = pv.template get_destination_cooperative_tensor<decltype(p0), decltype(firstV), float>();
  for (ushort i = 0; i < running.get_capacity(); ++i) running[i] = 0;
  uint first = p.window && start > p.window / 2 ? (start - p.window / 2) / N : 0;
  uint last = p.window ? min((p.tokens + N - 1) / N, (start + M + p.window / 2 + N - 1) / N) : (p.tokens + N - 1) / N;
  for (uint tile = first; tile < last; ++tile) {
    auto kt = tensor(k + ulong(tile) * N * D, dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto k0 = kt.slice<D, N>(0, 0);
    auto scores = qk.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();
    auto st = tensor(storage, dextents<int, 2>{N, M}, array<int, 2>{1, N});
    qk.run(q0, k0, scores); scores.store(st.slice<N, M>(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float exponents[R][4], scales[R];
    for (ushort i = 0; i < R; ++i) {
      uint row = sg * R + i, query = start + row;
      float local[4], tileMax = -INFINITY;
      for (ushort j = 0; j < 4; ++j) {
        uint column = lane + j * 32, key = tile * N + column;
        bool visible = key < p.tokens && (!p.window || abs(int(query) - int(key)) <= int(p.window / 2));
        local[j] = visible ? storage[row * N + column] * 0.125f : -INFINITY;
        tileMax = max(tileMax, local[j]);
      }
      tileMax = simd_max(tileMax);
      float next = max(maxima[i], tileMax);
      float scale = isfinite(next) ? precise::exp(maxima[i] - next) : 1.0f;
      float sum = 0;
      for (ushort j = 0; j < 4; ++j) {
        exponents[i][j] = isfinite(local[j]) ? precise::exp(local[j] - next) : 0;
        sum += exponents[i][j];
      }
      scales[i] = scale; sums[i] = sums[i] * scale + simd_sum(sum); maxima[i] = next;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort i = 0; i < R; ++i) {
      ushort row = sg * R + i;
      if (!lane) { previous[row] = scales[i]; rowSum[row] = sums[i]; }
      for (ushort j = 0; j < 4; ++j) probs[row * N + lane + j * 32] = bfloat(exponents[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tileV = tensor(v + ulong(tile) * N * D, dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto v0 = tileV.slice<D, N>(0, 0);
    auto partial = pv.template get_destination_cooperative_tensor<decltype(p0), decltype(v0), float>();
    pv.run(p0, v0, partial);
    for (ushort i = 0; i < running.get_capacity(); ++i) {
      auto ix = running.get_multidimensional_index(i);
      running[i] = running[i] * previous[ix[1]] + partial[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  device bfloat *out = output + (ulong(head) * p.padded_queries + start) * D;
  // Float and BF16 cooperative tensors may have different element mappings.
  // Scatter the FP32 accumulator by its logical coordinates, as the GGUF tiles do.
  for (ushort i = 0; i < running.get_capacity(); ++i) {
    if (!running.is_valid_element(i)) continue;
    auto ix = running.get_multidimensional_index(i);
    out[ulong(ix[1]) * D + ix[0]] = bfloat(rowSum[ix[1]] > 0 ? running[i] / rowSum[ix[1]] : 0);
  }
}

kernel void modern_attention_gather(device const bfloat *attention [[buffer(0)]],
    device bfloat *output [[buffer(1)]], constant ModernRowsParams &p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  uint row = index / p.stride, c = index % p.stride;
  if (row >= p.rows) return;
  output[index] = c < p.width ? attention[(ulong(c / 64) * p.kind + row) * 64 + c % 64] : bfloat(0);
}

kernel void modern_ffn_activate(device const float *input [[buffer(0)]],
    device const float *bias [[buffer(1)]], device bfloat *output [[buffer(2)]],
    constant ModernActivationParams &p [[buffer(3)]], uint index [[thread_position_in_grid]]) {
  uint row = index / p.destination_stride, c = index % p.destination_stride;
  if (row >= p.rows) return;
  float v = 0;
  if (c < p.width) {
    v = input[ulong(row) * p.source_stride + c] + bias[c];
    v = p.head ? max(v, 0.0f) : modern_gelu(v) * input[ulong(row) * p.source_stride + p.width + c];
  }
  output[index] = bfloat(v);
}

kernel void modern_select_float(device const float *input [[buffer(0)]],
    device const uint *positions [[buffer(1)]], device float *output [[buffer(2)]],
    constant ModernRowsParams &p [[buffer(3)]], uint index [[thread_position_in_grid]]) {
  uint row = index / p.stride;
  if (row < p.rows) output[index] = input[ulong(positions[row]) * p.stride + index % p.stride];
}
kernel void modern_select_bfloat(device const bfloat *input [[buffer(0)]],
    device const uint *positions [[buffer(1)]], device bfloat *output [[buffer(2)]],
    constant ModernRowsParams &p [[buffer(3)]], uint index [[thread_position_in_grid]]) {
  uint row = index / p.stride;
  if (row < p.rows) output[index] = input[ulong(positions[row]) * p.stride + index % p.stride];
}
kernel void modern_classifier(device const float *input [[buffer(0)]],
    device const float *bias [[buffer(1)]], device const bfloat *weight [[buffer(2)]],
    device const float *output_bias [[buffer(3)]], device float *output [[buffer(4)]],
    constant ModernRowsParams &p [[buffer(5)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[8]; float sum = 0;
  for (uint c = tid; c < p.width; c += 256)
    sum += modern_gelu(input[ulong(row) * p.stride + c] + bias[c]) * float(weight[c]);
  sum = simd_sum(sum); if (!lane) sums[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!tid) { float total = output_bias[0]; for (uint i = 0; i < 8; ++i) total += sums[i]; output[row] = total; }
}
