#pragma once

#include "metal/abi/KvCodec.h"
#include "metal/kernels/common/kv_extent.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

// Dense attention uses the intrinsic signed-nibble decoder directly as a
// tensor operand. Selected attention uses splash_q4h_decode on the same
// bytes. K and V have the same adjacent-nibble, token-major orientation.
inline auto splash_q4h_page_operand(device uchar *data) {
  constexpr uint D = SPLASH_KV_HEAD_DIMENSION, N = SPLASH_TARGET_KV_BLOCK_TOKENS;
  return tensor<device int4b_format, dextents<int, 2>, tensor_inline>(
      data, dextents<int, 2>{D, N}, array<int, 2>{1, D});
}

inline bfloat2 splash_q4_0_pair(device const SplashQ4Block *block, uint lane) {
  return bfloat2(bfloat(splash_q4_0_decode(block->qs[lane], lane, float(block->d))),
                 bfloat(splash_q4_0_decode(block->qs[lane], lane + 16, float(block->d))));
}

template <class Cache>
inline auto splash_kv_key_operand(device Cache *data, threadgroup bfloat *stage) {
  constexpr uint D = SPLASH_KV_HEAD_DIMENSION, N = SPLASH_TARGET_KV_BLOCK_TOKENS;
  if constexpr (is_same<Cache, SplashQ4Block>::value)
    return tensor(stage, dextents<int,2>{D,N}, array<int,2>{1,D});
  else if constexpr (is_same<Cache, uchar>::value)
    return splash_q4h_page_operand(data);
  else
    return tensor<device Cache, dextents<int,2>, tensor_inline>(data, dextents<int,2>{D,N}, array<int,2>{1,D});
}

template <class Cache>
inline auto splash_kv_value_operand(device Cache *data, threadgroup bfloat *stage) {
  constexpr uint D = SPLASH_KV_HEAD_DIMENSION, N = SPLASH_TARGET_KV_BLOCK_TOKENS;
  if constexpr (is_same<Cache, SplashQ4Block>::value)
    return tensor(stage, dextents<int,2>{D,N}, array<int,2>{1,D});
  else if constexpr (is_same<Cache, uchar>::value)
    return splash_q4h_page_operand(data);
  else
    return tensor<device Cache, dextents<int,2>, tensor_inline>(data, dextents<int,2>{N,D}, array<int,2>{1,N});
}

template <class Cache>
inline float splash_kv_read(SplashKvPageTensors<Cache> slab, uint token, uint dimension, bool valueTensor) {
  const auto data = valueTensor ? slab.values : slab.keys;
  const uint element = token * SPLASH_KV_HEAD_DIMENSION + dimension;
  if constexpr (is_same<Cache, SplashQ4Block>::value) {
    const auto block = data + element / 32;
    return float(bfloat(splash_q4_0_decode(block->qs[dimension % 16], dimension, float(block->d))));
  } else if constexpr (is_same<Cache, uchar>::value) {
    return splash_q4h_decode(data[element / 2], dimension,
                            (valueTensor ? slab.value_scales : slab.key_scales)[token]);
  } else {
    const uint at = valueTensor ? dimension * SPLASH_TARGET_KV_BLOCK_TOKENS + token : element;
    if constexpr (is_same<Cache, bfloat>::value) return float(data[at]);
    else return splash_q8_decode(data[at], (valueTensor ? slab.value_scales : slab.key_scales)[token]);
  }
}

// Every GPU store uses the same codec after its caller selects a head slab
// and loads one BF16 dimension. Only the parallel reductions are GPU-specific.
template <class Cache>
inline void splash_kv_store_row(SplashKvPageTensors<Cache> slab, bfloat source,
                               threadgroup float *maxima, bool valueTensor,
                               uint token, uint dimension, uint lane, uint group) {
  const auto data = valueTensor ? slab.values : slab.keys;
  const uint element = token * SPLASH_KV_HEAD_DIMENSION + dimension;
  if constexpr (is_same<Cache, SplashQ4Block>::value) {
    const float value = float(source), maximum = simd_max(abs(value));
    const uint first = simd_min(abs(value) == maximum ? lane : 32u);
    const float signedMaximum = maximum == 0.0f ? 0.0f : simd_shuffle(value, first);
    const float scale = splash_q4_0_scale(signedMaximum);
    const int code = splash_q4_0_quantize(value, splash_q4_0_inverse(scale));
    const int other = simd_shuffle(code, (lane + 16) % 32);
    const auto block = data + element / 32;
    if (lane == 0) block->d = maximum == 0.0f ? as_type<half>(ushort(0x8000)) : half(scale);
    if (lane < 16) block->qs[lane] = splash_kv_pack_nibbles(code, other);
  } else if constexpr (is_same<Cache, bfloat>::value) {
    data[valueTensor ? dimension * SPLASH_TARGET_KV_BLOCK_TOKENS + token : element] = source;
  } else {
    const float value = float(source), local = simd_max(abs(value));
    if (lane == 0) maxima[group] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (group == 0) for (uint offset = 4; offset; offset >>= 1) {
      if (lane < offset) maxima[lane] = max(maxima[lane], maxima[lane + offset]);
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float maximum = maxima[0];
    const auto scales = valueTensor ? slab.value_scales : slab.key_scales;
    if constexpr (is_same<Cache, uchar>::value) {
      const int code = splash_q4h_quantize(value, maximum);
      const int other = simd_shuffle_xor(code, 1);
      if (!(dimension & 1)) data[element / 2] = splash_kv_pack_nibbles(code, other);
      if (dimension == 0) scales[token] = splash_q4h_scale(maximum);
    } else {
      data[valueTensor ? dimension * SPLASH_TARGET_KV_BLOCK_TOKENS + token : element] = int8_t(splash_q8_quantize(value, maximum));
      if (dimension == 0) scales[token] = splash_q8_scale(maximum);
    }
  }
}
