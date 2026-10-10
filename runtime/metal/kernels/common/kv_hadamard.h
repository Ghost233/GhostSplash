#pragma once
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
    value = (thread_index & stride) ? other - value : value + other;
  }
  for (uint stride = 32; stride < Block; stride <<= 1) {
    scratch[thread_index] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float other = scratch[thread_index ^ stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value = (thread_index & stride) ? other - value : value + other;
  }
  return value * (Block == 256 ? 0.0625f : 0.125f);
}
