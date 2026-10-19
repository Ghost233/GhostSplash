#include "metal/abi/D3.h"
#include "metal/abi/KvCodec.h"
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Online softmax and MPP QK/PV, matching the encoder arithmetic with 256-channel GQA.
kernel void d3_attention(
    device bfloat *queries [[buffer(0)]], device bfloat *keys [[buffer(1)]],
    device bfloat *values [[buffer(2)]], device bfloat *output [[buffer(3)]],
    device const float *kv_scales [[buffer(4)]],
    constant D3AttentionParams &p [[buffer(5)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr ushort M = 64, N = 128, D = 256, R = 8;
  threadgroup float storage[M * N];
  threadgroup bfloat *probs = reinterpret_cast<threadgroup bfloat *>(storage);
  threadgroup float *rowSum = storage + M * N / 2;
  threadgroup float *previous = rowSum + M;
  float maxima[R], sums[R];
  for (ushort i = 0; i < R; ++i) { maxima[i] = -INFINITY; sums[i] = 0; }
  uint head = group.y, start = group.x * M;
  uint gqa = p.query_heads / p.kv_heads, kv = head / gqa, sub = head % gqa;
  device bfloat *q = queries + ((ulong(kv) * p.stride + start + p.query_start) * gqa + sub) * D;
  device bfloat *k = keys + ulong(kv) * p.stride * D;
  device bfloat *v = values + ulong(kv) * p.stride * D;
  auto qt = tensor(q, dextents<int, 2>{D, M}, array<int, 2>{1, int(gqa * D)});
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
  uint first = 0, last = (p.tokens + N - 1) / N;
  for (uint tile = first; tile < last; ++tile) {
    auto kt = tensor(k + ulong(tile) * N * D, dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto k0 = kt.slice<D, N>(0, 0);
    auto scores = qk.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();
    auto st = tensor(storage, dextents<int, 2>{N, M}, array<int, 2>{1, N});
    qk.run(q0, k0, scores); scores.store(st.slice<N, M>(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float exponents[R][4], scales[R];
    for (ushort i = 0; i < R; ++i) {
      uint row = sg * R + i;
      float local[4], tileMax = -INFINITY;
      for (ushort j = 0; j < 4; ++j) {
        uint column = lane + j * 32, key = tile * N + column;
        bool visible = key < p.tokens;
        local[j] = visible ? storage[row * N + column] * 0.0625f * (p.q4h ? kv_scales[kv*p.stride+key] : 1.0f) : -INFINITY;
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
      for (ushort j = 0; j < 4; ++j) {
        uint key=tile*N+lane+j*32;
        float scale=p.q4h && key<p.tokens ? kv_scales[(p.kv_heads+kv)*p.stride+key] : 1.0f;
        probs[row * N + lane + j * 32] = bfloat(exponents[i][j]*scale);
      }
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
  device bfloat *out = output + ((ulong(kv) * p.stride + start) * gqa + sub) * D;
  if (p.q4h) {
    uint tid = sg * 32 + lane;
    // Reuse score storage after PV. Invert H64 in FP32 before BF16 output.
    for (uint block=0;block<4;++block) {
      for (ushort i=0;i<running.get_capacity();++i) {
        if (!running.is_valid_element(i)) continue;
        auto ix=running.get_multidimensional_index(i);
        if (uint(ix[0])/64==block) storage[ix[1]*64+ix[0]%64]=running[i]/rowSum[ix[1]];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint step=1;step<64;step*=2) {
        float local[16];
        for(uint i=0;i<16;++i) {
          uint at=tid+i*256;
          local[i]=splash_kv_hadamard_step(storage[at],storage[at^step],(at&step)!=0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=0;i<16;++i) storage[tid+i*256]=local[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      for(uint at=tid;at<4096;at+=256) out[ulong(at/64)*gqa*D+block*64+at%64]=bfloat(storage[at]*0.125f);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else {
    for (ushort i = 0; i < running.get_capacity(); ++i) {
      if (!running.is_valid_element(i)) continue;
      auto ix = running.get_multidimensional_index(i);
      out[ulong(ix[1]) * gqa * D + ix[0]] = bfloat(rowSum[ix[1]] > 0 ? running[i] / rowSum[ix[1]] : 0);
    }
  }
}


kernel void d3_zero(device uint *data [[buffer(0)]], constant uint &words [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i < words) data[i] = 0;
}

kernel void d3_values(device const bfloat *input [[buffer(0)]], device bfloat *output [[buffer(1)]], constant D3AttentionParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < p.kv_heads * p.stride * 256) {
    uint c=i%256, token=(i/256)%p.stride, head=i/(256*p.stride);
    output[i]=input[(head*256+c)*p.stride+token];
  }
}

// Q/K H256 and V H64 were BF16-rounded by the shared Qwen QKV prepare.
// Match the existing Q4H per-token/head quantizer; stage its decoded values for MPP.
kernel void d3_quantize_kv(device const bfloat *keys [[buffer(0)]], device const bfloat *values [[buffer(1)]],
    device uchar *codes [[buffer(2)]], device float *scales [[buffer(3)]],
    constant D3AttentionParams &p [[buffer(4)]],uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float maxima[8];
  uint tensor=group/(p.tokens*p.kv_heads), row=group%(p.tokens*p.kv_heads), head=row/p.tokens, token=row%p.tokens;
  float x=tensor?float(values[(head*256+tid)*p.stride+token]):float(keys[(head*p.stride+token)*256+tid]);
  float maximum=simd_max(fabs(x));
  if(!lane)maxima[sg]=maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maximum=0;for(uint i=0;i<8;++i)maximum=max(maximum,maxima[i]);
  int code=splash_q4h_quantize(x,maximum), other=simd_shuffle_xor(code,1);
  ulong destination=(ulong(tensor)*p.kv_heads*p.stride+head*p.stride+token);
  if(!(tid&1))codes[destination*128+tid/2]=splash_kv_pack_nibbles(code,other);
  float scale=splash_q4h_scale(maximum);
  if(!tid)scales[destination]=scale;
}
// MPP operands are read back from the packed cache, not source activations.
kernel void d3_decode_kv(device const uchar *codes [[buffer(0)]],device bfloat *keys [[buffer(1)]],
    device bfloat *decoded [[buffer(2)]],constant D3AttentionParams &p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
  uint tensor=group/(p.tokens*p.kv_heads),row=group%(p.tokens*p.kv_heads),head=row/p.tokens,token=row%p.tokens;
  ulong destination=ulong(tensor)*p.kv_heads*p.stride+head*p.stride+token;
  bfloat value=bfloat(splash_q4h_unpack(codes[destination*128+tid/2],tid));
  if(tensor)decoded[(head*p.stride+token)*256+tid]=value;
  else keys[(head*p.stride+token)*256+tid]=value;
}
