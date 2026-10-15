#include "ops/CompactProjection.hpp"
#include "ops/Linear.hpp"
#include "ops/BufferExtent.hpp"
#include "metal/abi/OptionalFeatures.h"
#include "metal/abi/QuantFormat.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace splash::ops {

void updateProjectionRows(metal::MetalBackend &backend, const Projection &source,
                          const Projection &destination, metal::MetalBuffer ids,
                          uint32_t offset, uint32_t rows) {
  if (!source.takesPlaneViews() || !destination.takesPlaneViews() ||
      source.inputSize != destination.inputSize || !rows ||
      offset > destination.outputSize || rows > destination.outputSize - offset ||
      ids.sizeBytes() < uint64_t{destination.outputSize} * sizeof(uint32_t))
    throw std::invalid_argument("invalid compact projection gather");
  const auto &from = source.blocks().segments.front();
  const auto &to = destination.blocks().segments.front();
  if (from.formatId != to.formatId) throw std::invalid_argument("compact gather format mismatch");
  if (offset > to.outputSize || rows > to.outputSize - offset || !ids.contents())
    throw std::invalid_argument("compact mapping must be readable and destination rows covered");
  const auto *mapping = static_cast<const uint32_t *>(ids.contents());
  for (uint32_t row = offset; row < offset + rows; ++row)
    if (mapping[row] >= from.outputSize)
      throw std::invalid_argument("compact row id outside source tensor");
  requireSegmentPlanes(from, "compact source");
  requireSegmentPlanes(to, "compact destination");
  const auto &format = from.format();
  const uint32_t groups = source.inputSize / 32;
  metal::CommandGraph graph;
  const auto gather = [&](metal::MetalBuffer input, metal::MetalBuffer output,
                          uint32_t units, uint32_t unitBytes) {
    if (!unitBytes) return;
    const CompactGatherParams params{rows, offset, units, unitBytes};
    graph.add("compact_gather_plane", {input, output, ids}, params,
              {(uint64_t{rows} * units * unitBytes + 255) / 256, 1, 1}, {256, 1, 1});
  };
  gather(from.plane0, to.plane0, groups, format.plane0_bytes);
  gather(from.plane1, to.plane1, groups, format.plane1_bytes);
  gather(from.meta, to.meta, groups / format.meta_groups, format.meta_bytes);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
}

Projection gatherProjection(metal::MetalBackend &backend, const Projection &source,
                            std::span<const uint32_t> ids) {
  if (!source.takesPlaneViews() || ids.empty() || ids.size() % QUANT_TILE_ROWS)
    throw std::invalid_argument("compact head requires one unrotated quantized tensor and tile-aligned rows");
  for (uint32_t id : ids)
    if (id >= source.outputSize) throw std::invalid_argument("compact row id outside vocabulary");
  const auto &segment = source.blocks().segments.front();
  const auto &format = segment.format();
  const uint32_t rows = static_cast<uint32_t>(ids.size()), groups = source.inputSize / 32;
  const auto allocate = [&](uint64_t bytes) {
    return bytes ? backend.allocateBuffer(bytes, metal::BufferStorage::Private, "compact-draft-head")
                 : metal::MetalBuffer{};
  };
  Projection result(rows, source.inputSize,
      BlockWeights{{QuantizedSegment::planes(segment.formatId, rows, source.inputSize,
          allocate(uint64_t{rows} * groups * format.plane0_bytes),
          allocate(uint64_t{rows} * groups * format.plane1_bytes),
          allocate(uint64_t{rows} * (groups / format.meta_groups) * format.meta_bytes))}});
  result.destination = source.destination;
  auto mapping = backend.allocateBuffer(ids.size_bytes(), metal::BufferStorage::Shared, "compact-gather-ids");
  std::memcpy(mapping.contents(), ids.data(), ids.size_bytes());
  updateProjectionRows(backend, source, result, mapping, 0, rows);
  return result;
}


CompactProjectionValidation validateCompactProjection(
    metal::MetalBackend &backend, const Projection &source, const Projection &compact,
    metal::MetalBuffer mapping) {
  if (!source.takesPlaneViews() || !compact.takesPlaneViews() ||
      source.destination != FloatOutput::Float32 || compact.destination != FloatOutput::Float32 ||
      source.inputSize != compact.inputSize || !mapping.contents())
    throw std::invalid_argument("compact projection validation has incompatible head layout");
  requireBytes(mapping, uint64_t{compact.outputSize} * 4, "compact validation mapping");
  const auto *ids = static_cast<const uint32_t *>(mapping.contents());
  for (uint32_t i = 0; i < compact.outputSize; ++i)
    if (ids[i] >= source.outputSize)
      throw std::invalid_argument("compact validation mapping ID is outside the full vocabulary");
  constexpr uint32_t rows = SPLASH_DRAFT_QUERY_ROWS;
  auto input = backend.allocateBuffer(uint64_t{rows} * source.inputSize * 2,
                                      metal::BufferStorage::Shared, "compact-validation-input");
  auto fullOutput = backend.allocateBuffer(uint64_t{rows} * source.outputSize * 4,
                                           metal::BufferStorage::Shared, "compact-validation-full");
  auto compactOutput = backend.allocateBuffer(uint64_t{rows} * compact.outputSize * 4,
                                              metal::BufferStorage::Shared, "compact-validation-selected");
  auto *values = static_cast<uint16_t *>(input.contents());
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t k = 0; k < source.inputSize; ++k) {
      const float value = float(int((uint64_t{row} * 131 + uint64_t{k} * 37 + 17) % 257) - 128) / 128;
      // These dyadic probe values are exactly representable in BF16.
      values[uint64_t{row} * source.inputSize + k] = uint16_t(std::bit_cast<uint32_t>(value) >> 16);
    }
  const Linear linear(backend.capabilities());
  metal::CommandGraph graph;
  for (const auto &[projection, output] :
       std::array<std::pair<const Projection *, metal::MetalBuffer>, 2>{{{&source, fullOutput}, {&compact, compactOutput}}}) {
    const auto plan = Linear::plan({{projection->outputSize, projection->inputSize}, rows,
                                    LinearPhase::Decode, LinearEpilogue::None},
                                   {LinearTile::GgufStaged, 1, false, false}, FloatOutput::Float32);
    static_cast<void>(linear.add(graph, {.input = input, .output = output}, *projection, plan));
  }
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  const auto *full = static_cast<const float *>(fullOutput.contents());
  const auto *selected = static_cast<const float *>(compactOutput.contents());
  CompactProjectionValidation result;
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t i = 0; i < compact.outputSize; ++i) {
      const float expected = full[uint64_t{row} * source.outputSize + ids[i]];
      const float actual = selected[uint64_t{row} * compact.outputSize + i];
      if (!std::isfinite(expected) || !std::isfinite(actual) || actual != expected)
        throw std::runtime_error("compact projection numerical validation failed at row " +
            std::to_string(i) + ", token " + std::to_string(ids[i]) + ", probe " + std::to_string(row));
      result.maximumAbsoluteError = std::max(result.maximumAbsoluteError, std::fabs(actual - expected));
      ++result.comparisons;
    }
  return result;
}

void addCompactLogitMask(metal::CommandGraph &graph, metal::MetalBuffer logits,
                         uint32_t rows, uint32_t capacity, uint32_t valid) {
  if (!rows || !capacity || !valid || valid > capacity)
    throw std::invalid_argument("invalid compact logit mask");
  requireBytes(logits, uint64_t{rows} * capacity * 4, "compact logits");
  const CompactMapParams params{rows, capacity, valid, 0};
  graph.add("compact_mask_logits", {logits}, params,
            {(uint64_t{rows} * capacity + 255) / 256, 1, 1}, {256, 1, 1});
}

void addCompactProposalMap(metal::CommandGraph &graph, metal::MetalBuffer proposals,
                           metal::MetalBuffer candidates, metal::MetalBuffer mapping,
                           uint32_t capacity, uint32_t valid, bool sampled) {
  if (!capacity || !valid || valid > capacity)
    throw std::invalid_argument("invalid compact proposal map");
  requireBytes(mapping, uint64_t{valid} * 4, "compact mapping");
  requireBytes(proposals, uint64_t{SPLASH_DRAFT_PROPOSAL_TOKENS} * 4, "compact proposals");
  if (sampled) requireBytes(candidates,
      uint64_t{SPLASH_DRAFT_PROPOSAL_TOKENS} * SPLASH_DRAFT_CANDIDATES * 4, "compact candidates");
  const CompactMapParams params{SPLASH_DRAFT_QUERY_ROWS, capacity, valid, sampled};
  graph.add("compact_map_proposals", {proposals, candidates, mapping}, params, {1, 1, 1}, {256, 1, 1});
}

} // namespace splash::ops
