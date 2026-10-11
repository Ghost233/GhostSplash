#pragma once

#include "ops/Weights.hpp"
#include "metal/CommandGraph.hpp"

#include <span>

namespace splash::ops {

// Gather arbitrary rows of a single quantized tensor in its native planes.
// Allocation/quantization remain weight-only; no FP16 copy is produced.
[[nodiscard]] Projection gatherProjection(metal::MetalBackend &backend,
                                          const Projection &source,
                                          std::span<const uint32_t> ids);
void updateProjectionRows(metal::MetalBackend &backend, const Projection &source,
                          const Projection &destination, metal::MetalBuffer ids,
                          uint32_t offset, uint32_t rows);

void addCompactLogitMask(metal::CommandGraph &graph, metal::MetalBuffer logits,
                         uint32_t rows, uint32_t capacity, uint32_t valid);
void addCompactProposalMap(metal::CommandGraph &graph, metal::MetalBuffer proposals,
                           metal::MetalBuffer candidates, metal::MetalBuffer mapping,
                           uint32_t capacity, uint32_t valid, bool sampled);

} // namespace splash::ops
