#pragma once

#include "ops/Linear.hpp"

namespace splash::ops {

// One BF16 input row and Block32 planes. Unlike the verification path, this
// dispatch computes no padded rows. K splits are reduced in order, with no
// counters; partials may be reused by serial dispatches in one graph.
// columnsPerThreadgroup selects the 32-column baseline or the 64-column
// candidate, both 128 threads. Float segments retain their 32-column tile.
[[nodiscard]] uint64_t singleTokenLinearScratchBytes(const Projection &projection, uint32_t splits = 1);
void addSingleTokenLinear(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &projection,
                          metal::MetalBuffer output, metal::MetalBuffer auxiliary = {},
                          LinearEpilogue epilogue = LinearEpilogue::None,
                          metal::MetalBuffer partials = {}, uint32_t splits = 1,
                          uint32_t columnsPerThreadgroup = 32);

} // namespace splash::ops
