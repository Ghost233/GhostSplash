#include "model/Whittle27.hpp"
#include "model/QwenTargetLoader.hpp"

namespace splash::model {

Whittle27Weights loadWhittle27Weights(metal::MetalBackend &backend, Whittle27Layout layout,
                                      const QwenTargetFiles<Whittle27Layout> &files) {
  return loadQwenTarget<Whittle27Weights>(backend, layout, files,
      [](WeightFile &file, Whittle27LayerWeights &layer, const auto &format) {
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(format)>, BlockTargetFormat>) {
          ops::BlockMoeWeights ffn;
          ffn.router = readQuantizedSegment(file, "router");
          ffn.gate.routed = readQuantizedSegment(file, "experts-gate");
          ffn.up.routed = readQuantizedSegment(file, "experts-up");
          ffn.down.routed = readQuantizedSegment(file, "experts-down");
          ffn.gate.shared = readQuantizedSegment(file, "shared-expert-gate");
          ffn.up.shared = readQuantizedSegment(file, "shared-expert-up");
          ffn.down.shared = readQuantizedSegment(file, "shared-expert-down");
          ffn.sharedScalarGate = readQuantizedSegment(file, "shared-expert-scalar-gate");
          layer.ffn = std::move(ffn);
          for (ops::Projection *projection : {&layer.sharedGate, &layer.sharedUp, &layer.sharedDown}) {
            ops::BlockWeights weights;
            weights.segments.push_back(readQuantizedSegment(file, "Whittle shared expert"));
            const auto &segment = weights.segments.front();
            *projection = ops::Projection(segment.outputSize, segment.inputSize, std::move(weights));
          }
        } else {
          throw WeightStoreError("Whittle supports GGUF text targets only");
        }
      });
}

} // namespace splash::model
