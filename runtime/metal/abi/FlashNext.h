#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif
struct FlashNextParams {
  uint32_t width;
  uint32_t streams;
  uint32_t rows;
  float epsilon;
};
static_assert(sizeof(FlashNextParams) == 16, "Flash Next params are 16 bytes");
struct FlashNextConvParams {
  uint32_t width;
  uint32_t streams;
  uint32_t rows;
  uint32_t taps;
  uint32_t dilation;
};
static_assert(sizeof(FlashNextConvParams) == 20,
              "Flash Next conv params are 20 bytes");

struct FlashNextPoolParams {
  uint32_t width;
  uint32_t ratio;
  uint32_t blocks;
  uint32_t first_block;
};
struct FlashNextRowsParams {
  uint32_t width;
  uint32_t valid_rows;
  uint32_t storage_rows;
  uint32_t input_stride;
  uint32_t output_stride;
};
static_assert(sizeof(FlashNextRowsParams) == 20, "Flash Next row copy params are 20 bytes");
static_assert(sizeof(FlashNextPoolParams) == 16,
              "Flash Next pool params are 16 bytes");
struct FlashNextRopeParams {
  uint32_t width;
  uint32_t heads;
  uint32_t rows;
  uint32_t rotary_dimension;
  uint32_t table_rows;
};
static_assert(sizeof(FlashNextRopeParams) == 20,
              "Flash Next RoPE params are 20 bytes");
struct FlashNextQsaParams {
  uint32_t width;
  uint32_t heads;
  uint32_t rows;
  uint32_t blocks;
  uint32_t ratio;
  uint32_t budget;
};
static_assert(sizeof(FlashNextQsaParams) == 24,
              "Flash Next QSA params are 24 bytes");

#include "metal/abi/KvExtent.h"
struct FlashNextGdnParams {
  uint32_t width;
  uint32_t key_heads;
  uint32_t value_heads;
  uint32_t tokens;
  uint32_t gate_stride;
  uint32_t tiled;
};
static_assert(sizeof(FlashNextGdnParams) == 24,
              "Flash Next GDN params are 24 bytes");
struct FlashNextAttentionParams {
  SplashKvLayer kv;
  uint32_t query_heads;
  uint32_t rows;
  uint32_t selection_stride;
  uint32_t page_table_entries;
};
static_assert(sizeof(FlashNextAttentionParams) == 24,
              "Flash Next attention params are 24 bytes");
