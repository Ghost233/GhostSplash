#pragma once

#include "Qwen3_8.hpp"

namespace splash::model {

// Text-only Qwen3.5-4B, without a speculative draft. Its GDN geometry uses
// the existing 32-value-head kernels; full attention is 16Q/4KV.
struct Qwen3_5_4BLayout final : QwenHybridLayout<0> {
  static constexpr std::string_view family = "Qwen3.5-4B";
  static constexpr std::string_view layerMagic = "MDFL0006";
  static constexpr std::string_view headMagic = "MDFL0002";

  constexpr Qwen3_5_4BLayout()
      : QwenHybridLayout{{.maximumContextTokens = 262'144,
                          .layers = 32,
                          .hiddenSize = 2560,
                          .vocabularySize = 248320,
                          .packedGdnWidth = 12544,
                          .packedFullWidth = 10240,
                          .convolutionDimension = 8192,
                          .gdnKeyHeads = 16,
                          .gdnValueHeads = 32,
                          .gdnHeadDimension = 128,
                          .attentionWidth = 4096,
                          .attentionQueryHeads = 16,
                          .attentionKvHeads = 4,
                          .attentionHeadDimension = 256,
                          .rotaryPairs = SPLASH_TARGET_ROPE_PAIRS,
                          .rotaryTheta = 10'000'000.0F,
                          .fullAttentionPeriod = 4,
                          .maskToken = 248077,
                          .stopTokens = {248044, 248046},
                          .ffnKind = QwenFfnKind::Dense,
                          .intermediateSize = 9216,
                          .tiedWordEmbeddings = true}, {}} {}
  bool operator==(const Qwen3_5_4BLayout &) const = default;
};

struct Qwen3_5_4BDFlashLayout final : QwenHybridLayout<8> {
  static constexpr std::string_view family = Qwen3_5_4BLayout::family;
  static constexpr std::string_view layerMagic = Qwen3_5_4BLayout::layerMagic;
  static constexpr std::string_view headMagic = Qwen3_5_4BLayout::headMagic;
  constexpr Qwen3_5_4BDFlashLayout()
      : QwenHybridLayout{static_cast<const QwenTargetDimensions &>(Qwen3_5_4BLayout{}),
                         {1, 5, 9, 13, 17, 21, 25, 29}} {}
  bool operator==(const Qwen3_5_4BDFlashLayout &) const = default;
};

inline constexpr DFlashDraftLayout kQwen3_5_4BDFlashDraftLayout{
    .layers = 6, .hiddenSize = 2560, .vocabularySize = 248320,
    .dynamicSize = 0, .qkvSize = 6144, .attentionSize = 4096,
    .intermediateSize = 9216, .attentionHeadDimension = 128,
    .rotaryTheta = 10'000'000.0F, .targetHiddenSize = 20480,
    .selectorRank = 0, .kvHeads = 8, .kind = DraftKind::Standard};

using Qwen3_5_4BWeights = QwenTargetWeights<Qwen3_5_4BLayout, Qwen3_8LayerWeights>;
using Qwen3_5_4BDFlashWeights = QwenTargetWeights<Qwen3_5_4BDFlashLayout, Qwen3_8LayerWeights>;
[[nodiscard]] Qwen3_5_4BDFlashWeights
loadQwen3_5_4BDFlashWeights(metal::MetalBackend &backend, Qwen3_5_4BDFlashLayout layout,
                           const QwenTargetFiles<Qwen3_5_4BDFlashLayout> &files);
void buildQwen3_5_4BPrefillCache(metal::MetalBackend &backend, WeightImages &images,
                                Qwen3_5_4BDFlashWeights &weights);


[[nodiscard]] Qwen3_5_4BWeights
loadQwen3_5_4BWeights(metal::MetalBackend &backend, Qwen3_5_4BLayout layout,
                    const QwenTargetFiles<Qwen3_5_4BLayout> &files);

// Called after every source image is loaded, so restore rebuilds caches
// only after all quantized source planes are available.
void buildQwen3_5_4BPrefillCache(metal::MetalBackend &backend, WeightImages &images,
                               Qwen3_5_4BWeights &weights);

} // namespace splash::model
