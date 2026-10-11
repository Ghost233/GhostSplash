#include "Qwen3_5_4B.hpp"
#include "QwenTargetLoader.hpp"
#include "ops/GgufPrefillCache.hpp"

namespace splash::model {

Qwen3_5_4BWeights loadQwen3_5_4BWeights(metal::MetalBackend &backend, Qwen3_5_4BLayout layout,
                                      const QwenTargetFiles &files) {
  const auto readFfn = [&](WeightFile &file, Qwen3_8LayerWeights &layer, const auto &format) {
    layer.gateProjection = format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
    layer.upProjection = format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-up");
    layer.downProjection = format.projection(file, layout.hiddenSize, layout.intermediateSize, "mlp-down");
  };
  return loadQwenTarget<Qwen3_5_4BWeights>(backend, layout, files, readFfn);
}

Qwen3_5_4BDFlashWeights loadQwen3_5_4BDFlashWeights(metal::MetalBackend &backend,
    Qwen3_5_4BDFlashLayout layout, const QwenTargetFiles &files) {
  const auto readFfn = [&](WeightFile &file, Qwen3_8LayerWeights &layer, const auto &format) {
    layer.gateProjection = format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
    layer.upProjection = format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-up");
    layer.downProjection = format.projection(file, layout.hiddenSize, layout.intermediateSize, "mlp-down");
  };
  return loadQwenTarget<Qwen3_5_4BDFlashWeights>(backend, layout, files, readFfn);
}

template <class Weights>
void buildPrefillCache(metal::MetalBackend &backend, WeightImages &images, Weights &weights, uint64_t budgetBytes) {
  const uint64_t baseline = backend.memoryStats().allocatedBytes;
  const auto cache = [&](ops::Projection &projection, uint32_t layer, const char *name) {
    ops::BlockWeights blocks = projection.blocks();
    for (ops::QuantizedSegment &segment : blocks.segments) {
      if (segment.isFloat() || segment.prefillHalf) continue;
      const uint64_t bytes = ops::ggufPrefillCacheBytes(segment);
      if (bytes > budgetBytes - weights.ffnCacheAllocatedBytes) continue;
      const ops::QuantizedSegment source = segment;
      segment.prefillHalf = images.loadAuxiliary(
          "target/layer-" + std::to_string(layer) + "/" + name + "-prefill-fp16",
          ops::ggufPrefillCacheBytes(source),
          [&backend, source, trace = projection.ffn.trace](std::span<uint8_t>, const metal::MetalBuffer &destination) {
            ops::writeGgufPrefillCache(backend, source, destination, trace);
          });
      weights.ffnCacheAllocatedBytes += segment.prefillHalf.allocatedBytes();
      if (weights.ffnCacheAllocatedBytes > budgetBytes)
        throw std::runtime_error("actual FFN cache allocation exceeds its limit");
    }
    ops::Projection cached(projection.outputSize, projection.inputSize, std::move(blocks));
    cached.destination = projection.destination;
    cached.rotation = projection.rotation;
    cached.ffn = projection.ffn;
    projection = std::move(cached);
  };
  for (uint32_t i = 0; i < weights.layers.size(); ++i) {
    cache(weights.layers[i].gateProjection, i, "mlp-gate");
    cache(weights.layers[i].upProjection, i, "mlp-up");
    cache(weights.layers[i].downProjection, i, "mlp-down");
  }
  const uint64_t bytes = metal::allocationDelta(baseline, backend.memoryStats().allocatedBytes);
  weights.actualAllocatedBytes += bytes;
}

void buildQwen3_5_4BPrefillCache(metal::MetalBackend &backend, WeightImages &images, Qwen3_5_4BWeights &weights, uint64_t budgetBytes) {
  buildPrefillCache(backend, images, weights, budgetBytes);
}
void buildQwen3_5_4BPrefillCache(metal::MetalBackend &backend, WeightImages &images, Qwen3_5_4BDFlashWeights &weights, uint64_t budgetBytes) {
  buildPrefillCache(backend, images, weights, budgetBytes);
}

} // namespace splash::model
