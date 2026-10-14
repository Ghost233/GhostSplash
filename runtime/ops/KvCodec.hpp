#pragma once

#include "metal/abi/KvCodec.h"
#include "ops/PagedKv.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <span>
#include <stdexcept>

namespace splash::kv {

struct Q4hRow final {
  std::array<uint8_t, SPLASH_KV_HEAD_DIMENSION / SPLASH_Q4H_VALUES_PER_BYTE> codes{};
  float scale = 0;
};

inline std::array<uint16_t, SPLASH_KV_HEAD_DIMENSION> rotateRow(
    std::span<const uint16_t, SPLASH_KV_HEAD_DIMENSION> source, bool valueTensor) {
  std::array<float, SPLASH_KV_HEAD_DIMENSION> rotated{};
  for (uint32_t d = 0; d < rotated.size(); ++d)
    rotated[d] = splash_kv_bf16_to_float(source[d]);
  const uint32_t block = valueTensor ? SPLASH_KV_VALUE_ROTATION_BLOCK : SPLASH_KV_KEY_ROTATION_BLOCK;
  for (uint32_t start = 0; start < rotated.size(); start += block)
    for (uint32_t stride = 1; stride < block; stride <<= 1)
      for (uint32_t at = 0; at < block; at += 2 * stride)
        for (uint32_t lane = 0; lane < stride; ++lane) {
          auto &first = rotated[start + at + lane];
          auto &second = rotated[start + at + lane + stride];
          const float a = first, b = second;
          first = splash_kv_hadamard_step(a, b, false);
          second = splash_kv_hadamard_step(b, a, true);
        }
  std::array<uint16_t, SPLASH_KV_HEAD_DIMENSION> result{};
  for (uint32_t d = 0; d < rotated.size(); ++d) {
    result[d] = splash_kv_float_to_bf16(splash_kv_hadamard_normalize(rotated[d], block));
  }
  return result;
}

// The ordinary GPU store also accepts already rotated BF16 rows; the ABI
// helpers define its identical scale, rounding and nibble representation.
inline Q4hRow quantizeQ4h(std::span<const uint16_t, SPLASH_KV_HEAD_DIMENSION> source) {
  std::array<float, SPLASH_KV_HEAD_DIMENSION> values{};
  float maximum = 0;
  for (uint32_t d = 0; d < values.size(); ++d) {
    values[d] = splash_kv_bf16_to_float(source[d]);
    maximum = std::max(maximum, std::abs(values[d]));
  }
  Q4hRow row;
  row.scale = splash_q4h_scale(maximum);
  for (uint32_t d = 0; d < values.size(); d += SPLASH_Q4H_VALUES_PER_BYTE)
    row.codes[d / SPLASH_Q4H_VALUES_PER_BYTE] = splash_kv_pack_nibbles(
        splash_q4h_quantize(values[d], maximum), splash_q4h_quantize(values[d + 1], maximum));
  return row;
}

inline SplashQ4Block quantizeQ4_0(std::span<const uint16_t, SPLASH_Q4_0_BLOCK_VALUES> source) {
  float maximum = 0, signedMaximum = 0;
  for (auto bits : source) {
    const float value = splash_kv_bf16_to_float(bits);
    if (std::abs(value) > maximum) { maximum = std::abs(value); signedMaximum = value; }
  }
  const float scale = splash_q4_0_scale(signedMaximum);
  const float inverse = splash_q4_0_inverse(scale);
  SplashQ4Block block{};
  const __fp16 half = static_cast<__fp16>(scale);
  std::memcpy(&block.d, &half, sizeof(half));
  for (uint32_t d = 0; d < 16; ++d)
    block.qs[d] = splash_kv_pack_nibbles(
        splash_q4_0_quantize(splash_kv_bf16_to_float(source[d]), inverse),
        splash_q4_0_quantize(splash_kv_bf16_to_float(source[d + 16]), inverse));
  return block;
}

// The source is one post-RoPE BF16 head. PageStorage supplies the tensor
// spans of a page; this layer owns every format's rotation and placement.
inline void writeRow(Format format, std::span<const std::span<std::byte>> page,
                     uint32_t layer, uint32_t head, uint32_t token, bool valueTensor,
                     std::span<const uint16_t, SPLASH_KV_HEAD_DIMENSION> source) {
  if (!validFormat(format)) throw std::invalid_argument("invalid KV row format");
  const bool external = format == Format::Int8 || format == Format::Int4Hadamard;
  auto data = page[layer * (external ? 4 : 2) + uint32_t(valueTensor) * (external ? 2 : 1)];
  if (format == Format::BFloat16) {
    for (uint32_t d = 0; d < source.size(); ++d) {
      const auto at = valueTensor ? splash_kv_value_element(head, token, d) : splash_kv_key_element(head, token, d);
      std::memcpy(data.data() + at * 2, source.data() + d, 2);
    }
    return;
  }
  std::array<uint16_t, SPLASH_KV_HEAD_DIMENSION> rotated{};
  if (usesHadamard(format)) { rotated = rotateRow(source, valueTensor); source = rotated; }
  const auto first = splash_kv_key_element(head, token, 0);
  if (isQ4_0(format)) {
    for (uint32_t d = 0; d < source.size(); d += SPLASH_Q4_0_BLOCK_VALUES) {
      const auto block = quantizeQ4_0(std::span<const uint16_t,32>(source.data() + d, 32));
      std::memcpy(data.data() + (first + d) / 32 * sizeof(block), &block, sizeof(block));
    }
    return;
  }
  auto scales = page[layer * 4 + uint32_t(valueTensor) * 2 + 1];
  float scale = 0;
  if (format == Format::Int4Hadamard) {
    const auto row = quantizeQ4h(source);
    std::memcpy(data.data() + first / SPLASH_Q4H_VALUES_PER_BYTE, row.codes.data(), row.codes.size());
    scale = row.scale;
  } else {
    float maximum = 0;
    for (auto bits : source) maximum = std::max(maximum, std::abs(splash_kv_bf16_to_float(bits)));
    scale = splash_q8_scale(maximum);
    for (uint32_t d = 0; d < source.size(); ++d) {
      const auto at = valueTensor ? splash_kv_value_element(head, token, d) : splash_kv_key_element(head, token, d);
      const int8_t code = int8_t(splash_q8_quantize(splash_kv_bf16_to_float(source[d]), maximum));
      std::memcpy(data.data() + at, &code, 1);
    }
  }
  std::memcpy(scales.data() + splash_kv_scale_element(head, token) * 4, &scale, 4);
}

} // namespace splash::kv
