#include "SingleTokenLinear.hpp"

#include "ops/BufferExtent.hpp"

#include <stdexcept>
#include <string>

namespace splash::ops {
namespace {

struct SingleTokenParams final {
  uint32_t inputs, planeInputs, columns, stride, offset, splits;
};

void requireSplits(uint32_t inputs, uint32_t splits) {
  if (!splits || splits > 8 || (splits & (splits - 1)) || !inputs || inputs % (32 * splits))
    throw std::invalid_argument("single-token K splits must be 1, 2, 4 or 8 and divide whole input groups");
}

const char *suffix(LinearEpilogue epilogue, FloatOutput destination) {
  if (destination == FloatOutput::Float32) {
    if (epilogue != LinearEpilogue::None)
      throw std::invalid_argument("single-token FP32 output requires a plain epilogue");
    return "a_f32";
  }
  switch (epilogue) {
    case LinearEpilogue::None: return "a";
    case LinearEpilogue::Residual: return "r";
    case LinearEpilogue::UpWithGate: return "g";
    case LinearEpilogue::GateUp:
      throw std::invalid_argument("single-token gate/up requires a gate dispatch followed by up-with-gate");
  }
  throw std::invalid_argument("unknown single-token epilogue");
}

} // namespace

uint64_t singleTokenLinearScratchBytes(const Projection &projection, uint32_t splits) {
  requireSplits(projection.inputSize, splits);
  return splits > 1 ? uint64_t{projection.outputSize} * splits * sizeof(float) : 0;
}

void addSingleTokenLinear(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &projection,
                          metal::MetalBuffer output, metal::MetalBuffer auxiliary, LinearEpilogue epilogue,
                          metal::MetalBuffer partials, uint32_t splits, uint32_t columnsPerThreadgroup) {
  if (projection.rotation)
    throw std::invalid_argument("single-token projection requires unrotated block weights");
  requireSplits(projection.inputSize, splits);
  if (columnsPerThreadgroup != 32 && columnsPerThreadgroup != 64)
    throw std::invalid_argument("single-token column tile must be 32 or 64");
  const std::string ep = suffix(epilogue, projection.destination);
  requireBytes(input, uint64_t{projection.inputSize} * 2, "single-token input");
  requireBytes(output, uint64_t{projection.outputSize} * elementBytes(projection.destination), "single-token output");
  if (epilogue != LinearEpilogue::None)
    requireBytes(auxiliary, uint64_t{projection.outputSize} * 2, "single-token auxiliary");
  requireBytes(partials, singleTokenLinearScratchBytes(projection, splits), "single-token partials");
  // Validate every segment before recording a dispatch.
  for (const QuantizedSegment &segment : projection.blocks().segments) {
    if (segment.isFloat())
      requireBytes(segment.plane0, uint64_t{segment.outputSize} * projection.planeInputSize() * sizeof(float),
                   "single-token float weights");
    else
      requireSegmentPlanes(segment, "single-token weights", projection.planeInputs());
  }
  if (!auxiliary) auxiliary = output;
  if (!partials) partials = output;
  for (const QuantizedSegment &segment : projection.blocks().segments) {
    const SingleTokenParams p{projection.inputSize, projection.planeInputSize(), segment.outputSize,
                              projection.outputSize, segment.columnOffset, splits};
    const uint32_t tile = segment.isFloat() ? 32 : columnsPerThreadgroup;
    const std::string name = std::string("gguf_single_") + (segment.isFloat() ? "float" : segment.name()) +
                              (tile == 64 ? "_n64" : "");
    const auto plane1 = segment.isFloat() ? segment.plane0 : segment.plane1Slot();
    const auto meta = segment.isFloat() ? segment.plane0 : segment.meta;
    graph.add(name + "_" + (splits > 1 ? "partial" : ep),
              {input, segment.plane0, plane1, meta, output, auxiliary, partials}, p,
              {(segment.outputSize + tile - 1) / tile, splits, 1}, {128, 1, 1});
    if (splits > 1)
      graph.add("gguf_single_reduce_" + ep, {partials, output, auxiliary}, p,
                {(segment.outputSize + 127) / 128, 1, 1}, {128, 1, 1});
  }
}

} // namespace splash::ops
