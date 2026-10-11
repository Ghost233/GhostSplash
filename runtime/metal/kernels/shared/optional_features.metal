#include "metal/abi/KernelABI.h"
#include "metal/abi/OptionalFeatures.h"
#include "metal/abi/QuantFormat.h"

// The source and destination use [tile][unit][256 rows][unit bytes].
// Copy the codes and scale metadata byte-for-byte, including lookup formats.
kernel void compact_gather_plane(device const uchar *source [[buffer(0)]],
                                 device uchar *destination [[buffer(1)]],
                                 device const uint *ids [[buffer(2)]],
                                 constant CompactGatherParams &p [[buffer(3)]],
                                 uint i [[thread_position_in_grid]]) {
  const ulong bytes = ulong(p.rows) * p.units * p.unit_bytes;
  if (i >= bytes) return;
  const uint byte = uint(i % p.unit_bytes);
  const ulong unitRow = i / p.unit_bytes;
  const uint unit = uint(unitRow % p.units);
  const uint row = p.offset + uint(unitRow / p.units);
  const uint original = ids[row];
  const ulong from = ((ulong(original / QUANT_TILE_ROWS) * p.units + unit) *
                      QUANT_TILE_ROWS + original % QUANT_TILE_ROWS) * p.unit_bytes + byte;
  const ulong to = ((ulong(row / QUANT_TILE_ROWS) * p.units + unit) *
                    QUANT_TILE_ROWS + row % QUANT_TILE_ROWS) * p.unit_bytes + byte;
  destination[to] = source[from];
}

kernel void compact_mask_logits(device float *logits [[buffer(0)]],
                                constant CompactMapParams &p [[buffer(1)]],
                                uint i [[thread_position_in_grid]]) {
  if (i < p.rows * p.capacity && i % p.capacity >= p.valid) logits[i] = -INFINITY;
}

kernel void compact_map_proposals(device uint *proposals [[buffer(0)]],
                                  device uint *candidates [[buffer(1)]],
                                  device const uint *ids [[buffer(2)]],
                                  constant CompactMapParams &p [[buffer(3)]],
                                  uint i [[thread_position_in_grid]]) {
  if (i < SPLASH_DRAFT_PROPOSAL_TOKENS) {
    const uint id = proposals[i];
    proposals[i] = id < p.valid ? ids[id] : 0xffffffffu;
  }
  if (p.sampled && i < SPLASH_DRAFT_PROPOSAL_TOKENS * SPLASH_DRAFT_CANDIDATES) {
    const uint id = candidates[i];
    candidates[i] = id < p.valid ? ids[id] : 0xffffffffu;
  }
}

kernel void compact_map_partials(device uint *partials [[buffer(0)]],
                                 device const uint *ids [[buffer(1)]],
                                 constant CompactMapParams &p [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
  if (i < p.rows) {
    const uint id = partials[i];
    partials[i] = id < p.valid ? ids[id] : 0xffffffffu;
  }
}
