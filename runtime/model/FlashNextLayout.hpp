#pragma once
#include "model/Model.hpp"
#include "model/StateLayout.hpp"
#include "ops/GDN.hpp"
#include <string_view>
namespace splash::model {
struct FlashNextLayout final {
  static constexpr std::string_view family = "Qwen3.8-Flash-Next";
  uint32_t maximumContextTokens = 262144, layers = 48, hiddenSize = 2560,
           vocabularySize = 248320, streams = 4, lowRank = 320,
           experts = 512, expertsPerToken = 10, intermediateSize = 640,
           queryHeads = 24, kvHeads = 2, headDimension = 256,
           rotaryDimension = 64, fullAttentionPeriod = 4,
           indexHeads = 4, indexWidth = 128, indexRatio = 4, indexBudget = 2048,
           pleLayer = 1, pleHeads = 16, pleRowWidth = 160, pleTaps = 4, pleDilation = 3;
  float epsilon = 1e-6f, rotaryTheta = 10000000.f;
  uint32_t stopTokens[2] = {248044, 248046};
  uint64_t expertCacheBytes = 20ull << 30, hostCacheBytes = 64ull << 20;
  [[nodiscard]] constexpr uint32_t hyperWidth() const noexcept { return hiddenSize * streams; }
  [[nodiscard]] constexpr bool isAttention(uint32_t layer) const noexcept {
    return (layer + 1) % fullAttentionPeriod == 0;
  }
  [[nodiscard]] constexpr kv::Layout kvLayout() const noexcept {
    return {layers / fullAttentionPeriod, kvHeads, headDimension};
  }
  [[nodiscard]] constexpr ops::GdnShape gdnShape() const noexcept { return {16, 48, 128, 10240, 16640}; }
  [[nodiscard]] constexpr GdnStateLayout gdnStateLayout() const noexcept {
    return {layers - layers / fullAttentionPeriod, 3, 10240, 48, 128, 128};
  }
  [[nodiscard]] constexpr uint64_t stateBytes() const noexcept {
    const auto gdn = gdnStateLayout();
    return 2 * gdn.cellBytes() + uint64_t{pleTaps - 1} * pleDilation * hyperWidth() * 2 +
           uint64_t{kvLayout().attentionLayers} * maximumContextTokens * indexWidth * 2 * 5 / 4;
  }
  [[nodiscard]] bool valid() const noexcept;
  bool operator==(const FlashNextLayout &) const = default;
};
} // namespace splash::model
