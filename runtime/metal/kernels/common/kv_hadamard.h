#pragma once
#include "metal/abi/KvCodec.h"
#include <metal_stdlib>
using namespace metal;

// Normalized Walsh-Hadamard, H^2=I, independent blocks in a 256-thread group.
// The order is fixed; arithmetic is FP32 and the caller chooses BF16 rounding.
template <uint Block>
inline float splash_kv_hadamard(float value, threadgroup float *scratch,
                                uint thread_index) {
  static_assert(Block == 64 || Block == 256);
  for (uint stride = 1; stride < 32; stride <<= 1) {
    float other = simd_shuffle_xor(value, stride);
    value = splash_kv_hadamard_step(value, other, (thread_index & stride) != 0);
  }
  for (uint stride = 32; stride < Block; stride <<= 1) {
    scratch[thread_index] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float other = scratch[thread_index ^ stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value = splash_kv_hadamard_step(value, other, (thread_index & stride) != 0);
  }
  return splash_kv_hadamard_normalize(value, Block);
}

inline bfloat splash_kv_rotate_query_key(float value, threadgroup float *scratch,
                                         uint thread_index) {
  return bfloat(splash_kv_hadamard<SPLASH_KV_KEY_ROTATION_BLOCK>(value, scratch, thread_index));
}

inline bfloat splash_kv_rotate_value(float value, threadgroup float *scratch,
                                     uint thread_index) {
  return bfloat(splash_kv_hadamard<SPLASH_KV_VALUE_ROTATION_BLOCK>(value, scratch, thread_index));
}

inline float splash_kv_unrotate_value(float value, threadgroup float *scratch,
                                      uint thread_index) {
  return splash_kv_hadamard<SPLASH_KV_VALUE_ROTATION_BLOCK>(value, scratch, thread_index);
}
