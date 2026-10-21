#pragma once

#include "Qwen3_8.hpp"
#include "DFlashDraft.hpp"

namespace splash::model {

class WeightImages;

// Text-only Qwen3.5-4B, without a speculative draft. Its GDN geometry uses
// the existing 32-value-head kernels; full attention is 16Q/4KV.
struct Qwen3_5_4BLayout final : QwenHybridLayout {
  static constexpr std::string_view family = "Qwen3.5-4B";
  static constexpr std::string_view modelType = "qwen3_5_text";
  static constexpr std::string_view ggufArchitecture = "qwen35";
  static constexpr bool loadsSafetensors = false;
  static constexpr std::string_view layerMagic = "MDFL0006";
  static constexpr std::string_view headMagic = "MDFL0002";

  constexpr Qwen3_5_4BLayout()
      : QwenHybridLayout{{.nativeContextTokens = 262'144,
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
                          .tiedWordEmbeddings = true}} {}
  bool operator==(const Qwen3_5_4BLayout &) const = default;
};

// Kev-0.8B uses the same hybrid and pointer-head semantics with 16 GDN value heads.
struct Qwen3_5_0_8BLayout final : QwenHybridLayout {
  static constexpr std::string_view family = "Kev-0.8B";
  static constexpr std::string_view modelType = Qwen3_5_4BLayout::modelType;
  static constexpr std::string_view ggufArchitecture = Qwen3_5_4BLayout::ggufArchitecture;
  static constexpr bool loadsSafetensors = false;
  static constexpr std::string_view layerMagic = Qwen3_5_4BLayout::layerMagic;
  static constexpr std::string_view headMagic = Qwen3_5_4BLayout::headMagic;
  constexpr Qwen3_5_0_8BLayout()
      : QwenHybridLayout{static_cast<const QwenTargetDimensions &>(Qwen3_5_4BLayout{})} {
    layers = 24;
    hiddenSize = 1024;
    packedGdnWidth = 8448;
    packedFullWidth = 5120;
    convolutionDimension = 6144;
    gdnValueHeads = 16;
    attentionWidth = 2048;
    attentionQueryHeads = 8;
    attentionKvHeads = 2;
    intermediateSize = 3584;
    decisionHeadSize = 512;
  }
  bool operator==(const Qwen3_5_0_8BLayout &) const = default;
};

struct Qwen3_5_4BDFlashLayout final : QwenHybridLayout {
  static constexpr std::string_view family = Qwen3_5_4BLayout::family;
  static constexpr std::string_view modelType = Qwen3_5_4BLayout::modelType;
  static constexpr std::string_view ggufArchitecture = Qwen3_5_4BLayout::ggufArchitecture;
  static constexpr bool loadsSafetensors = false;
  static constexpr std::string_view layerMagic = Qwen3_5_4BLayout::layerMagic;
  static constexpr std::string_view headMagic = Qwen3_5_4BLayout::headMagic;
  constexpr Qwen3_5_4BDFlashLayout()
      : QwenHybridLayout{static_cast<const QwenTargetDimensions &>(Qwen3_5_4BLayout{})} {}
  bool operator==(const Qwen3_5_4BDFlashLayout &) const = default;
};

inline constexpr DFlashDraftLayout kQwen3_5_4BDFlashDraftLayout{
    .layers = 6, .hiddenSize = 2560, .vocabularySize = 248320,
    .dynamicSize = 0, .qkvSize = 6144, .attentionSize = 4096,
    .intermediateSize = 9216, .attentionHeadDimension = 128,
    .rotaryTheta = 10'000'000.0F,
    .selectorRank = 0, .kvHeads = 8, .kind = DraftKind::Standard,
    .targetLayers = {1, 5, 9, 13, 17, 21, 25, 29}, .targetLayerCount = 8};

using Qwen3_5_0_8BWeights = QwenTargetWeights<Qwen3_5_0_8BLayout, Qwen3_8LayerWeights>;
[[nodiscard]] Qwen3_5_0_8BWeights loadQwen3_5_0_8BWeights(metal::MetalBackend &, Qwen3_5_0_8BLayout, const QwenTargetFiles &);
void buildQwen3_5_0_8BPrefillCache(metal::MetalBackend &, WeightImages &, Qwen3_5_0_8BWeights &, uint64_t budgetBytes = UINT64_MAX);

using Qwen3_5_4BWeights = QwenTargetWeights<Qwen3_5_4BLayout, Qwen3_8LayerWeights>;
using Qwen3_5_4BDFlashWeights = QwenTargetWeights<Qwen3_5_4BDFlashLayout, Qwen3_8LayerWeights>;
[[nodiscard]] Qwen3_5_4BDFlashWeights
loadQwen3_5_4BDFlashWeights(metal::MetalBackend &backend, Qwen3_5_4BDFlashLayout layout,
                           const QwenTargetFiles &files);
void buildQwen3_5_4BPrefillCache(metal::MetalBackend &backend, WeightImages &images,
                                Qwen3_5_4BDFlashWeights &weights, uint64_t budgetBytes = UINT64_MAX);


[[nodiscard]] Qwen3_5_4BWeights
loadQwen3_5_4BWeights(metal::MetalBackend &backend, Qwen3_5_4BLayout layout,
                    const QwenTargetFiles &files);

// Called after every source image is loaded, so restore rebuilds caches
// only after all quantized source planes are available.
void buildQwen3_5_4BPrefillCache(metal::MetalBackend &backend, WeightImages &images,
                               Qwen3_5_4BWeights &weights, uint64_t budgetBytes = UINT64_MAX);

} // namespace splash::model
