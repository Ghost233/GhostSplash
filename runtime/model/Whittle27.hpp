#pragma once

#include "model/Qwen3_8.hpp"
#include "model/Qwen3_6Moe.hpp"

namespace splash::model {

struct Whittle27Layout final : QwenHybridLayout<5> {
  static constexpr std::string_view family = "Whittle-MoE-27B-A17.8B";
  static constexpr std::string_view layerMagic = Qwen3_6MoeLayout::layerMagic;
  static constexpr std::string_view headMagic = Qwen3_6MoeLayout::headMagic;
  constexpr Whittle27Layout() : QwenHybridLayout<5>(Qwen3_8Layout{}) {
    ffnKind = QwenFfnKind::SparseMoe;
    experts = 64;
    expertsPerToken = 16;
    sourceExpertIntermediateSize = 192;
    expertIntermediateSize = 256;
    intermediateSize = 5120; // The separate, always-on shared expert.
  }
  bool operator==(const Whittle27Layout &) const = default;
};

struct Whittle27LayerWeights final {
  ops::NormWeights inputNorm;
  QwenMixerWeights mixer;
  ops::NormWeights postAttentionNorm;
  ops::MoeWeights ffn;
  ops::Projection sharedGate, sharedUp, sharedDown;
};
using Whittle27Weights = QwenTargetWeights<Whittle27Layout, Whittle27LayerWeights>;
[[nodiscard]] Whittle27Weights loadWhittle27Weights(
    metal::MetalBackend &backend, Whittle27Layout layout,
    const QwenTargetFiles<Whittle27Layout> &files);

} // namespace splash::model
