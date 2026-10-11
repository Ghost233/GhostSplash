#include "ops/GgufPrefillCache.hpp"
#include "ops/Linear.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::ops {

uint64_t ggufPrefillCacheBytes(const QuantizedSegment &segment) {
  if (segment.isFloat()) throw std::invalid_argument("FP16 prefill cache requires quantized weights");
  return uint64_t{segment.outputSize} * segment.inputSize * sizeof(uint16_t);
}

void writeGgufPrefillCache(metal::MetalBackend &backend, const QuantizedSegment &segment,
                           const metal::MetalBuffer &destination, bool trace) {
  const uint64_t bytes = ggufPrefillCacheBytes(segment);
  requireSegmentPlanes(segment, "FP16 prefill cache");
  if (!destination || destination.sizeBytes() != bytes)
    throw std::invalid_argument("FP16 prefill cache buffer has the wrong size");
  const uint64_t groups = uint64_t{segment.outputSize} * (segment.inputSize / 32);
  metal::CommandGraph graph;
  graph.add(std::string("gguf_prefill_cache_") + segment.name(),
            {segment.plane0, segment.plane1Slot(), segment.meta, destination}, segment.inputSize,
            {(groups + 31) / 32, 1, 1}, {32, 1, 1});
  if (trace) graph.attributeFrom(0, "target_ffn/cache_build", 0);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
}

void addGgufHalfPrefill(metal::CommandGraph &graph, metal::MetalBuffer input,
                        const QuantizedSegment &segment, metal::MetalBuffer output,
                        const GgufPrefillParams &params, char epilogue, metal::MetalBuffer auxiliary,
                        uint32_t inputStep, GgufHalfPrefillTile tile) {
  const bool supportedStep = inputStep == 64 || inputStep == 128 || inputStep == 256 || inputStep == 512 ||
                             inputStep == 2560 || inputStep == 9216;
  const bool alternate = tile == GgufHalfPrefillTile::M64N32;
  const uint32_t tileRows = alternate ? 256 : GGUF_PREFILL_ROWS, tileColumns = alternate ? 32 : GGUF_TILE_COLUMNS;
  if ((!alternate && tile != GgufHalfPrefillTile::M32N64) || (alternate && inputStep != 64) ||
      !supportedStep || !segment.prefillHalf || segment.prefillHalf.sizeBytes() != ggufPrefillCacheBytes(segment) ||
      params.input_size != segment.inputSize || params.input_size % inputStep ||
      segment.outputSize % tileColumns || !params.rows ||
      (epilogue != 'a' && epilogue != 'r' && epilogue != 'g') || (epilogue != 'a' && !auxiliary))
    throw std::invalid_argument("invalid FP16 prefill cache projection");
  std::vector<metal::MetalBuffer> buffers{std::move(input), segment.prefillHalf, std::move(output)};
  if (epilogue != 'a') buffers.push_back(std::move(auxiliary));
  const std::string stepSuffix = alternate ? "m64n32_" :
      inputStep == GGUF_PREFILL_STEP ? "" : "k" + std::to_string(inputStep) + "_";
  graph.add("gguf_half_prefill_" + stepSuffix + epilogue, std::move(buffers), params,
            {(params.rows + tileRows - 1) / tileRows, segment.outputSize / tileColumns, 1}, {GGUF_PREFILL_THREADS, 1, 1});
}

} // namespace splash::ops
