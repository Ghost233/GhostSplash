#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif
// Q/output [KV head, padded token, GQA group, 256]; K/V [KV, token, 256].
struct D3AttentionParams {
  uint32_t tokens, stride, query_rows, query_start, query_heads, kv_heads;
  uint32_t q4h = 0;
};
static_assert(sizeof(D3AttentionParams) == 28);
