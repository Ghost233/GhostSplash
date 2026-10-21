#include "metal/abi/ExecutionGeometry.h"
#include "metal/kernels/common/gdn_primitives.h"
#include "metal/kernels/common/rms_inverse.h"

// One ordinary decode token, one value head per threadgroup. This preserves
// the prefill recurrence's eight lanes per state row and four float4 terms;
// the shared operands avoid round trips between three separate dispatches.
template <class W>
kernel void gdn_single_vh32(device const bfloat *packed [[buffer(0)]],
                            device const bfloat *convWeights [[buffer(1)]],
                            device const bfloat *convIn [[buffer(2)]], device bfloat *convOut [[buffer(3)]],
                            device bfloat *queries [[buffer(4)]], device bfloat *keys [[buffer(5)]],
                            device bfloat *values [[buffer(6)]], device const float *aScale [[buffer(7)]],
                            device const bfloat *timeBias [[buffer(8)]], device float *decay [[buffer(9)]],
                            device bfloat *beta [[buffer(10)]], device const float *stateIn [[buffer(11)]],
                            device float *stateOut [[buffer(12)]], device bfloat *recurrent [[buffer(13)]],
                            device const W *norm [[buffer(14)]], device bfloat *hidden [[buffer(15)]],
                            constant GDNGatePrefillParams &params [[buffer(16)]],
                            uint head [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr uint KeyHeads = 16, ValueHeads = 32, HeadDim = 128, ConvDim = 8192, PackedWidth = 12544;
  constexpr uint KeyWidth = KeyHeads * HeadDim, ValueWidth = ValueHeads * HeadDim;
  const uint keyHead = head / 2, qChannel = keyHead * HeadDim + tid,
             kChannel = KeyWidth + qChannel, vChannel = 2 * KeyWidth + head * HeadDim + tid;
  const bool sharedWriter = (head & 1) == 0;
  const GdnConvHistory history{convIn, packed, ConvDim, PackedWidth, 0};
  const GdnConvHistory carried{convIn, packed, ConvDim, PackedWidth, 1};
  const float query = float(gdn_conv_silu(packed, history, convWeights, PackedWidth, 0, qChannel));
  const float key = float(gdn_conv_silu(packed, history, convWeights, PackedWidth, 0, kChannel));
  const bfloat value = gdn_conv_silu(packed, history, convWeights, PackedWidth, 0, vChannel);
  threadgroup float reductions[8];
  const float qSum = simd_sum(query * query), kSum = simd_sum(key * key);
  if (lane == 0) { reductions[simdgroup] = qSum; reductions[4 + simdgroup] = kSum; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    reductions[0] = rsqrt((reductions[0] + reductions[1] + reductions[2] + reductions[3]) / HeadDim + kRmsEpsilon);
    reductions[4] = rsqrt((reductions[4] + reductions[5] + reductions[6] + reductions[7]) / HeadDim + kRmsEpsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup bfloat q[HeadDim], k[HeadDim], v[HeadDim], rows[HeadDim];
  q[tid] = bfloat(float(bfloat(query * reductions[0])) * 0.0078125f);
  k[tid] = bfloat(float(bfloat(key * reductions[4])) * 0.08838834765f);
  v[tid] = value;
  if (sharedWriter) { queries[qChannel] = q[tid]; keys[qChannel] = k[tid]; }
  values[head * HeadDim + tid] = value;
  threadgroup float gates[2];
  if (tid == 0) {
    const GdnGates gate = gdn_gates(packed, timeBias, aScale, ConvDim + ValueWidth,
                                   ConvDim + ValueWidth + ValueHeads, head);
    gates[0] = gate.decay; gates[1] = float(gate.beta);
    decay[head] = gate.decay; beta[head] = gate.beta;
  }
  for (uint row = 0; row < SPLASH_GDN_CONVOLUTION_TAPS - 1; ++row) {
    convOut[row * ConvDim + vChannel] = carried.at(row, vChannel);
    if (sharedWriter) {
      convOut[row * ConvDim + qChannel] = carried.at(row, qChannel);
      convOut[row * ConvDim + kChannel] = carried.at(row, kChannel);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint row0 = simdgroup * 4 + lane / 8, column = (lane % 8) * 16;
  float4 key4[4], query4[4];
  for (uint c = 0; c < 4; ++c) {
    key4[c] = float4(*((threadgroup bfloat4 *)(k + column + c * 4)));
    query4[c] = float4(*((threadgroup bfloat4 *)(q + column + c * 4)));
  }
  for (uint batch = 0; batch < HeadDim / 16; ++batch) {
    const uint row = row0 + batch * 16;
    const ulong base = (ulong(head) * HeadDim + row) * HeadDim + column;
    float4 state[4], partial = 0.0f;
    for (uint c = 0; c < 4; ++c) {
      state[c] = *((device const float4 *)(stateIn + base + c * 4));
      state[c] *= gates[0];
      partial = fma(state[c], key4[c], partial);
    }
    float memory = (partial.x + partial.y) + (partial.z + partial.w);
    for (uint shift = 1; shift < 8; shift <<= 1) memory += simd_shuffle_xor(memory, shift);
    const float delta = (float(v[row]) - memory) * gates[1];
    partial = 0.0f;
    for (uint c = 0; c < 4; ++c) {
      state[c] = fma(key4[c], delta, state[c]);
      partial = fma(state[c], query4[c], partial);
      *((device float4 *)(stateOut + base + c * 4)) = state[c];
    }
    float result = (partial.x + partial.y) + (partial.z + partial.w);
    for (uint shift = 1; shift < 8; shift <<= 1) result += simd_shuffle_xor(result, shift);
    if (lane % 8 == 0) { rows[row] = bfloat(result); recurrent[head * HeadDim + row] = bfloat(result); }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float out = float(rows[tid]);
  const float inverse = rms_inverse_of_sums<4>(out * out, HeadDim, reductions, tid, lane, simdgroup);
  const bfloat normalized = bfloat(out * inverse * float(norm[tid]));
  const float silu = splash_silu(float(packed[ConvDim + head * HeadDim + tid]));
  const uint outputHead = gdn_output_head<KeyHeads, ValueHeads>(head, (params.flags & GDN_TILED_HEADS) != 0);
  hidden[outputHead * HeadDim + tid] = bfloat(float(normalized) * silu);
}

#define GDN_SINGLE_INSTANCE(name, W) \
  template [[host_name(name)]] kernel void gdn_single_vh32<W> \
  (device const bfloat *, device const bfloat *, device const bfloat *, device bfloat *, device bfloat *, \
   device bfloat *, device bfloat *, device const float *, device const bfloat *, device float *, device bfloat *, \
   device const float *, device float *, device bfloat *, device const W *, device bfloat *, \
   constant GDNGatePrefillParams &, uint, uint, uint, uint);
GDN_SINGLE_INSTANCE("gdn_single_vh32", bfloat)
GDN_SINGLE_INSTANCE("gdn_single_vh32_f32", float)
