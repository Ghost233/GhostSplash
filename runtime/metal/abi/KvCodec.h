#pragma once

#include "metal/abi/KvExtent.h"
#ifndef __METAL_VERSION__
#include <bit>
#include <cmath>
#endif

// INT8 uses a symmetric per-head FP32 scale. BF16 is copied as stored.
// Standard Q4_0 uses the first signed absolute maximum, inline FP16 scales
// and split-half nibbles in 32-value blocks (GGML; THIRD_PARTY_NOTICES).
// Q4H v1: normalized H256 Q/K and independent H64 V blocks, BF16-rounded
// before quantization. Both tensors store adjacent signed nibbles in
// token-major order with one FP32 scale per (token, head).
#define SPLASH_KV_KEY_ROTATION_BLOCK SPLASH_KV_HEAD_DIMENSION
#define SPLASH_KV_VALUE_ROTATION_BLOCK 64u
#define SPLASH_Q4H_VALUES_PER_BYTE 2u
#define SPLASH_Q4_0_BLOCK_VALUES 32u

inline float splash_kv_hadamard_step(float value, float other, bool high) {
  return high ? other - value : value + other;
}

inline float splash_kv_hadamard_normalize(float value, uint32_t block) {
  return value * (block == SPLASH_KV_KEY_ROTATION_BLOCK ? 0.0625f : 0.125f);
}

#ifndef __METAL_VERSION__
inline float splash_kv_bf16_to_float(uint16_t bits) {
  return std::bit_cast<float>(uint32_t{bits} << 16);
}

inline uint16_t splash_kv_float_to_bf16(float value) {
  const auto bits = std::bit_cast<uint32_t>(value);
  return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
#endif

inline float splash_q8_scale(float maximum) { return maximum / 127.0f; }

inline int32_t splash_q8_quantize(float value, float maximum) {
  if (maximum == 0.0f) return 0;
#ifdef __METAL_VERSION__
  const int32_t code = int32_t(metal::rint(value * 127.0f / maximum));
#else
  const int32_t code = int32_t(std::nearbyint(value * 127.0f / maximum));
#endif
  return code < -127 ? -127 : code > 127 ? 127 : code;
}

inline float splash_q8_decode(int8_t code, float scale) { return float(code) * scale; }

inline float splash_q4_0_scale(float signedMaximum) { return signedMaximum * -0.125f; }

inline float splash_q4_0_inverse(float scale) { return scale != 0.0f ? 1.0f / scale : 0.0f; }

inline int32_t splash_q4_0_quantize(float value, float inverse) {
  const int32_t code = int32_t(value * inverse + 8.5f);
  return code < 0 ? 0 : code > 15 ? 15 : code;
}

inline float splash_q4h_scale(float maximum) {
#ifdef __METAL_VERSION__
  return metal::precise::divide(maximum, 7.0f);
#else
  return maximum / 7.0f;
#endif
}

inline int32_t splash_q4h_quantize(float value, float maximum) {
  if (maximum == 0.0f) return 0;
#ifdef __METAL_VERSION__
  const int32_t code = int32_t(metal::rint(metal::precise::divide(value * 7.0f, maximum)));
#else
  const int32_t code = int32_t(std::nearbyint(value * 7.0f / maximum));
#endif
  return code < -7 ? -7 : code > 7 ? 7 : code;
}

inline uint8_t splash_kv_pack_nibbles(int32_t low, int32_t high) {
  return uint8_t((low & 15) | ((high & 15) << 4));
}

inline int32_t splash_q4h_unpack(uint8_t packed, uint32_t dimension) {
  const int32_t code = (packed >> ((dimension & 1) * 4)) & 15;
  return code >= 8 ? code - 16 : code;
}

inline float splash_q4h_decode(uint8_t packed, uint32_t dimension, float scale) {
  return float(splash_q4h_unpack(packed, dimension)) * scale;
}

inline float splash_q4_0_decode(uint8_t packed, uint32_t dimension, float scale) {
  const int32_t code = (packed >> (dimension % 32 >= 16 ? 4 : 0)) & 15;
  return float(code - 8) * scale;
}
