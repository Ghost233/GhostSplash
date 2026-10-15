#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct ModernNormParams {
  uint32_t rows, width, stride, delta_stride, add_delta, normalize_hidden;
  float epsilon;
};
struct ModernQkvParams {
  uint32_t tokens, padded_tokens, heads, width, query_rows, padded_queries, selected, source_stride;
  float theta;
};
// Q/K/V and output are [head, padded token, 64 channels].
struct ModernAttentionParams {
  uint32_t tokens, padded_tokens, heads, query_rows, padded_queries, window;
};
struct ModernActivationParams {
  uint32_t rows, width, source_stride, destination_stride, head;
};
struct ModernRowsParams {
  uint32_t rows, width, stride, kind;
};
static_assert(sizeof(ModernNormParams) == 28);
static_assert(sizeof(ModernQkvParams) == 36);
static_assert(sizeof(ModernAttentionParams) == 24);
static_assert(sizeof(ModernActivationParams) == 20);
static_assert(sizeof(ModernRowsParams) == 16);
