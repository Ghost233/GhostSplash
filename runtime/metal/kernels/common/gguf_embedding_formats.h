#pragma once

#include "metal/abi/QuantTables.h"
#include <metal_stdlib>

// The native GGUF blocks of the token tables (ops/Embedding.cpp), one struct
// per format: a row is hidden / Weights blocks of Bytes bytes, each laid out as
// the format's ggml block_* at the byte offsets its struct names, and value is
// element dim as bf16. Every format's value keeps the source order of its
// float operations (reassociate(off)), as the GGUF GEMMs do.

// The little-endian half at byte `at` of a block.
inline half gguf_half(device const uchar *block, uint at) {
  return as_type<half>(ushort(block[at] | (block[at + 1] << 8)));
}

// block_q4_K: half d | half dmin | uchar scales[12] | uchar qs[128], eight
// 32-weight groups with 6-bit scales and mins; block_q5_K (Fifth) holds
// uchar qh[32] before qs, a fifth bit per weight.
template <bool Fifth> struct GgufEmbedK {
  enum : uint { Weights = 256, Bytes = Fifth ? 176 : 144, D = 0, DMin = 2, Scales = 4, High = 16, Codes = Fifth ? 48 : 16 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint j = (dim % Weights) / 32, l = dim % 32;
    const half d = gguf_half(block, D), dmin = gguf_half(block, DMin);
    device const uchar *sc = block + Scales; uchar m, s;
    if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; } else { s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
    uchar q = (block[Codes + (j / 2) * 32 + l] >> ((j % 2) * 4)) & 15;
    if (Fifth) q |= ((block[High + l] >> j) & 1) << 4;
    return bfloat(float(d) * float(s) * float(q) - float(dmin) * float(m));
  }
};
using GgufEmbedQ4K = GgufEmbedK<false>;
using GgufEmbedQ5K = GgufEmbedK<true>;
// block_q6_K: uchar ql[128] | uchar qh[64] | int8 scales[16] | half d; codes
// are 6-bit with zero point 32.
struct GgufEmbedQ6K {
  enum : uint { Weights = 256, Bytes = 210, Low = 0, High = 128, Scales = 192, D = 208, Zero = 32 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights, n = l / 128, r = l % 128, quarter = r / 32, pos = r % 32;
    const uchar lo = (block[Low + n * 64 + (quarter & 1) * 32 + pos] >> ((quarter >> 1) * 4)) & 15;
    const uchar hi = (block[High + n * 32 + pos] >> (2 * quarter)) & 3;
    const char sc = as_type<char>(block[Scales + n * 8 + 2 * quarter + pos / 16]);
    const half d = gguf_half(block, D);
    return bfloat(float(d) * float(sc) * float(int(lo | (hi << 4)) - int(Zero)));
  }
};
// block_q8_0: half d | int8 qs[32].
struct GgufEmbedQ80 {
  enum : uint { Weights = 32, Bytes = 34, D = 0, Codes = 2 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const half d = gguf_half(block, D);
    return bfloat(float(d) * float(as_type<char>(block[Codes + dim % Weights])));
  }
};
// The 2-bit code of weight l of a block_q3_K or block_q2_K's qs[64]: bits 2j of qs[32n + pos] for l = 128n + 32j + pos.
inline uchar gguf_k2_code(device const uchar *qs, uint l) { return (qs[32 * (l / 128) + l % 32] >> (2 * (l % 128 / 32))) & 3; }
// block_q3_K: uchar hmask[32] | uchar qs[64] | uchar scales[12] | half d; 16-element groups with a 6-bit scale
// (offset 32), codes with zero point 4 when their hmask bit is clear.
struct GgufEmbedQ3K {
  enum : uint { Weights = 256, Bytes = 110, High = 0, Codes = 32, Scales = 96, D = 108 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights, n = l / 128, j = (l % 128) / 32, pos = l % 32, is = l / 16;
    const uchar q = gguf_k2_code(block + Codes, l), h = (block[High + pos] >> (4 * n + j)) & 1;
    const uchar sc = (is < 8 ? block[Scales + is] & 0xF : block[Scales + is - 8] >> 4) |
                     ((block[Scales + 8 + is % 4] >> (2 * (is / 4))) & 3) << 4;
    const half d = gguf_half(block, D);
    return bfloat(float(d) * float(int(sc) - 32) * float(int(q) - (h ? 0 : 4)));
  }
};
// block_q2_K: uchar scales[16] | uchar qs[64] | half d | half dmin; 16-element groups whose scale byte holds a
// 4-bit scale and min.
struct GgufEmbedQ2K {
  enum : uint { Weights = 256, Bytes = 84, Scales = 0, Codes = 16, D = 80, DMin = 82 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights;
    const uchar q = gguf_k2_code(block + Codes, l), sc = block[Scales + l / 16];
    const half d = gguf_half(block, D), dmin = gguf_half(block, DMin);
    return bfloat(float(d) * float(sc & 0xF) * float(q) - float(dmin) * float(sc >> 4));
  }
};
// block_q4_0: half d | uchar qs[16], element j in the low nibble of qs[j] and j + 16 in the high one, with zero
// point 8; block_q4_1 (Min) holds half m after d, the value's offset.
template <bool Min> struct GgufEmbedQ4 {
  enum : uint { Weights = 32, Bytes = Min ? 20 : 18, D = 0, M = 2, Codes = Min ? 4 : 2 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights;
    const uchar q = (block[Codes + l % 16] >> (4 * (l / 16))) & 15;
    if (Min) return bfloat(float(q) * float(gguf_half(block, D)) + float(gguf_half(block, M)));
    return bfloat(float(int(q) - 8) * float(gguf_half(block, D)));
  }
};
using GgufEmbedQ40 = GgufEmbedQ4<false>;
using GgufEmbedQ41 = GgufEmbedQ4<true>;

// block_pq2_0: half d | uchar qs[32], element l in bits 2 (l % 4) of qs[l / 4]; zero point 1. decode is the
// element in fp32, which the rotated gather (shared/gguf_rotation.metal) transforms before it rounds.
struct GgufEmbedPQ20 {
  enum : uint { Weights = 128, Bytes = 34, D = 0, Codes = 2 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights;
    const uchar q = (block[Codes + l / 4] >> (2 * (l % 4))) & 3;
    return float(int(q) - 1) * float(gguf_half(block, D));
  }
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
    return bfloat(decode(block, dim));
  }
};
// block_iq4_nl: block_q4_0's layout, whose codes index the IQ4_NL values.
struct GgufEmbedIQ4NL {
  enum : uint { Weights = 32, Bytes = 18, D = 0, Codes = 2 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights;
    const uchar q = (block[Codes + l % 16] >> (4 * (l / 16))) & 15;
    return bfloat(float(gguf_half(block, D)) * float(kIQ4NLValues[q]));
  }
};
// block_iq4_xs: half d | ushort scales_h | uchar scales_l[4] | uchar qs[128], eight 32-weight groups of IQ4_NL codes
// with a 6-bit scale (offset 32): group j's element l in nibble l / 16 of qs[16 j + l % 16].
struct GgufEmbedIQ4XS {
  enum : uint { Weights = 256, Bytes = 136, D = 0, High = 2, Low = 4, Codes = 8 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint j = (dim % Weights) / 32, l = dim % 32;
    const uint high = block[High] | (block[High + 1] << 8);
    const int ls = int((block[Low + j / 2] >> (4 * (j % 2))) & 15) | int(((high >> (2 * j)) & 3) << 4);
    const uchar q = (block[Codes + 16 * j + l % 16] >> (4 * (l / 16))) & 15;
    return bfloat(float(gguf_half(block, D)) * float(ls - 32) * float(kIQ4NLValues[q]));
  }
};
// block_iq3_s: half d | uchar qs[64] | uchar qh[8] | uchar signs[32] | uchar scales[4], eight 32-weight groups with
// a 4-bit scale s worth 1 + 2 s: group j's element l is IQ3_S grid entry qs[8 j + l / 4] (ninth index bit l / 4 of
// qh[j]) at byte l % 4, negated by bit l % 8 of signs[4 j + l / 8].
struct GgufEmbedIQ3S {
  enum : uint { Weights = 256, Bytes = 110, D = 0, Codes = 2, High = 66, Signs = 74, Scales = 106 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint j = (dim % Weights) / 32, l = dim % 32;
    const uint index = block[Codes + 8 * j + l / 4] | (((block[High + j] >> (l / 4)) & 1) << 8);
    const uint magnitude = (kIQ3SGrid[index] >> (8 * (l % 4))) & 0xFF;
    const uint s = (block[Scales + j / 2] >> (4 * (j % 2))) & 15;
    const float sign = (block[Signs + 4 * j + l / 8] >> (l % 8)) & 1 ? -1.0f : 1.0f;
    return bfloat(float(gguf_half(block, D)) * float(1 + 2 * s) * float(magnitude) * sign);
  }
};

// block_iq2_s: half d | uchar qs[64] | uchar qh[8] | uchar scales[8]. Each 32-weight group has four
// 10-bit grid indices (the low bytes in qs, the high pairs in qh), four sign bytes in qs[32..63], and
// two 16-weight scales. The scale is d * (0.5 + s) * 0.25, in GGML's fp32 operation order.
struct GgufEmbedIQ2S {
  enum : uint { Weights = 256, Bytes = 82, D = 0, Codes = 2, Signs = 34, High = 66, Scales = 74 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint j = (dim % Weights) / 32, l = dim % 32, entry = l / 8;
    const uint index = block[Codes + 4 * j + entry] | (((block[High + j] >> (2 * entry)) & 3) << 8);
    const uint magnitude = uint((kIQ2SGrid[index] >> (8 * (l % 8))) & 0xFF);
    const uint s = (block[Scales + j] >> (4 * (l / 16))) & 15;
    const float scale = float(gguf_half(block, D)) * (0.5f + float(s)) * 0.25f;
    const float sign = (block[Signs + 4 * j + entry] >> (l % 8)) & 1 ? -1.0f : 1.0f;
    return bfloat(scale * float(magnitude) * sign);
  }
};

// Unaligned little-endian words in the native IQ blocks.
inline uint gguf_u16(device const uchar *block, uint at) {
  return uint(block[at]) | (uint(block[at + 1]) << 8);
}
inline uint gguf_u32(device const uchar *block, uint at) {
  return gguf_u16(block, at) | (gguf_u16(block, at + 2) << 16);
}
inline uint gguf_signs7(uint index) { return index | ((metal::popcount(index) & 1) << 7); }

// block_iq3_xxs: half d | uchar qs[64] | packed signs and scales[32]. A 32-weight group's word
// contains four 7-bit sign indices and a 4-bit scale; each grid entry contains four magnitudes.
struct GgufEmbedIQ3XXS {
  enum : uint { Weights = 256, Bytes = 98, D = 0, Codes = 2, Signs = 66 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint j = (dim % Weights) / 32, l = dim % 32;
    const uint aux = gguf_u32(block, Signs + 4 * j);
    const uint grid = kIQ3XXSGrid[block[Codes + 8 * j + l / 4]];
    const uint signs = gguf_signs7((aux >> (7 * (l / 8))) & 127);
    const float scale = float(gguf_half(block, D)) * (0.5f + float(aux >> 28)) * 0.5f;
    return bfloat(scale * float((grid >> (8 * (l % 4))) & 0xFF) * ((signs >> (l % 8)) & 1 ? -1.0f : 1.0f));
  }
};
// block_iq2_xxs: half d | uchar qs[64]. Each 32-weight group contains four grid index bytes
// followed by four packed 7-bit sign indices and a 4-bit scale.
struct GgufEmbedIQ2XXS {
  enum : uint { Weights = 256, Bytes = 66, D = 0, Codes = 2 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint j = (dim % Weights) / 32, l = dim % 32, entry = l / 8;
    const uint aux = gguf_u32(block, Codes + 8 * j + 4);
    const uint magnitude = uint((kIQ2XXSGrid[block[Codes + 8 * j + entry]] >> (8 * (l % 8))) & 0xFF);
    const uint signs = gguf_signs7((aux >> (7 * entry)) & 127);
    const float scale = float(gguf_half(block, D)) * (0.5f + float(aux >> 28)) * 0.25f;
    return bfloat(scale * float(magnitude) * ((signs >> (l % 8)) & 1 ? -1.0f : 1.0f));
  }
};
// block_iq2_xs: half d | ushort qs[32] | uchar scales[8]. Each entry has a 9-bit grid index
// and a 7-bit sign index; each scale byte holds two 16-weight scales.
struct GgufEmbedIQ2XS {
  enum : uint { Weights = 256, Bytes = 74, D = 0, Codes = 2, Scales = 66 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint j = (dim % Weights) / 32, l = dim % 32;
    const uint code = gguf_u16(block, Codes + 8 * j + 2 * (l / 8));
    const uint magnitude = uint((kIQ2XSGrid[code & 511] >> (8 * (l % 8))) & 0xFF);
    const uint signs = gguf_signs7(code >> 9), s = (block[Scales + j] >> (4 * (l / 16))) & 15;
    const float scale = float(gguf_half(block, D)) * (0.5f + float(s)) * 0.25f;
    return bfloat(scale * float(magnitude) * ((signs >> (l % 8)) & 1 ? -1.0f : 1.0f));
  }
};
// IQ1 grid entries pack grid value + 1 in the low/high nibble of four bytes.
inline float gguf_iq1_value(uint index, uint element) {
  return float(int((kIQ1SGrid[index] >> (8 * (element % 4) + 4 * (element / 4))) & 15) - 1);
}
// block_iq1_s: half d | uchar qs[32] | ushort qh[8], a 32-weight group's high index bits,
// 3-bit scale and delta sign. Its grid values are offset by +1/8 or -1/8.
struct GgufEmbedIQ1S {
  enum : uint { Weights = 256, Bytes = 50, D = 0, Codes = 2, High = 34 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint j = (dim % Weights) / 32, l = dim % 32, entry = l / 8;
    const uint high = gguf_u16(block, High + 2 * j);
    const uint index = block[Codes + 4 * j + entry] | (((high >> (3 * entry)) & 7) << 8);
    const float scale = float(gguf_half(block, D)) * float(2 * ((high >> 12) & 7) + 1);
    return bfloat(scale * (gguf_iq1_value(index, l % 8) + (high & 0x8000 ? -0.125f : 0.125f)));
  }
};
// block_iq1_m: uchar qs[32] | uchar qh[16] | ushort scales[4]. The top nibble of each scale
// word carries a quarter of half d; the low 12 bits hold four 3-bit scales.
struct GgufEmbedIQ1M {
  enum : uint { Weights = 256, Bytes = 56, Codes = 0, High = 32, Scales = 48 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint j = (dim % Weights) / 32, l = dim % 32, entry = l / 8;
    const uint d = (gguf_u16(block, Scales) >> 12) | ((gguf_u16(block, Scales + 2) >> 8) & 0xF0) |
                   ((gguf_u16(block, Scales + 4) >> 4) & 0xF00) | (gguf_u16(block, Scales + 6) & 0xF000);
    const uint high = (block[High + 2 * j + entry / 2] >> (4 * (entry % 2))) & 15;
    const uint index = block[Codes + 4 * j + entry] | ((high & 7) << 8);
    const uint s = (gguf_u16(block, Scales + 2 * (j / 2)) >> (6 * (j % 2) + 3 * (l / 16))) & 7;
    const float scale = float(as_type<half>(ushort(d))) * float(2 * s + 1);
    return bfloat(scale * (gguf_iq1_value(index, l % 8) + (high & 8 ? -0.125f : 0.125f)));
  }
};
// block_mxfp4: uchar e | uchar qs[16], E8M0 scale times E2M1 values (kFP4Values stores twice
// those values, so the exponent is reduced by one). e = 0 or 1 denotes an fp32 subnormal scale.
struct GgufEmbedMXFP4 {
  enum : uint { Weights = 32, Bytes = 17, E = 0, Codes = 1 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights, e = block[E];
    const uint code = (block[Codes + l % 16] >> (4 * (l / 16))) & 15;
    const float scale = as_type<float>(e < 2 ? 0x00200000u << e : (e - 1) << 23);
    return bfloat(float(kFP4Values[code]) * scale);
  }
};
