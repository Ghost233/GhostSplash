// CPU-only: registering dense 4B FFNs must not require allocating a model,
// starting Metal, or calibrating an ANE split.
#include "model/Runtime.hpp"
#include "ops/AneFfn.hpp"
#include "metal/abi/QuantFormat.h"

#include <iostream>
#include <string>
#include <type_traits>

using namespace splash;

namespace {
int failures = 0;
void require(bool condition, const std::string &what) {
  if (!condition) { ++failures; std::cerr << "FAIL " << what << '\n'; }
}

template <class Weights>
void dense(const char *label, uint32_t expectedLayers) {
  Weights weights;
  weights.layers.resize(weights.layout.layers);
  const auto projection = [](uint32_t n, uint32_t k) {
    return ops::Projection(n, k, ops::BlockWeights{{
        ops::QuantizedSegment::planes(GGUF_FMT_AF4G64, n, k, {}, {}, {})}});
  };
  for (auto &layer : weights.layers) {
    layer.gateProjection = projection(weights.layout.intermediateSize, weights.layout.hiddenSize);
    layer.upProjection = projection(weights.layout.intermediateSize, weights.layout.hiddenSize);
    layer.downProjection = projection(weights.layout.hiddenSize, weights.layout.intermediateSize);
  }
  model::LoadedModel loaded;
  loaded.target = std::move(weights);
  const auto layers = model::aneFfnLayers(loaded);
  require(layers.size() == expectedLayers, std::string(label) + " FFN layer count=" + std::to_string(layers.size()));
  const auto &stored = std::get<Weights>(loaded.target);
  for (size_t i = 0; i < layers.size(); ++i)
    require(layers[i].gate == &stored.layers[i].gateProjection &&
            layers[i].up == &stored.layers[i].upProjection &&
            layers[i].down == &stored.layers[i].downProjection,
            std::string(label) + " registered the wrong projection at layer " + std::to_string(i));
  // Shape acceptance is tested separately even when registration fails.
  const ops::SwiGluProjections first{&stored.layers[0].gateProjection, &stored.layers[0].upProjection,
                                    &stored.layers[0].downProjection};
  const char *reason = ops::AneFfn::unsupported(std::span(&first, 1));
  require(!reason, std::string(label) + " shape rejected: " + (reason ? reason : ""));
  if (!reason)
    require(ops::AneFfn::units(std::span(&first, 1)) == stored.layout.intermediateSize / 512,
            std::string(label) + " channel units changed");
}
} // namespace

int main() {
  dense<model::Qwen3_5_4BWeights>("4B NoDraft", 32);
  dense<model::Qwen3_5_4BDFlashWeights>("4B Standard", 32);
  dense<model::Qwen3_8Weights>("27B dense", 64);
  model::LoadedModel sparse;
  sparse.target = model::Qwen3_6MoeWeights{};
  require(model::aneFfnLayers(sparse).empty(), "sparse MoE was registered as a dense FFN");
  if (failures) return 1;
  std::cout << "ANE FFN dense registration and 2560/5120 shape checks passed\n";
  return 0;
}
