#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct CompactGatherParams {
  uint32_t rows, offset, units, unit_bytes;
};
static_assert(sizeof(CompactGatherParams) == 16);
struct CompactMapParams {
  uint32_t rows, capacity, valid, sampled;
};
static_assert(sizeof(CompactMapParams) == 16);
