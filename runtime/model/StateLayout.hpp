#pragma once

#include "Checked.hpp"
#include "metal/abi/DraftAttention.h"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GDN.h"

#include <cstdint>
#include <array>
#include <algorithm>

namespace splash::model {

// The taps of the GDN short convolution the kernels compute; a state cell
// keeps the inputs of all taps but the current token's.
inline constexpr uint32_t kGdnConvolutionTaps = SPLASH_GDN_CONVOLUTION_TAPS;

// Physical state geometry is supplied by the paired target and draft models.
// The engine sees only opaque CompositeState handles and byte accounting.
struct GdnStateLayout final {
  static constexpr uint32_t bfloat16Bytes = 2;

  uint32_t layers = 0;
  uint32_t convolutionHistory = 0;
  uint32_t convolutionChannels = 0;
  uint32_t recurrentGroups = 0;
  uint32_t recurrentRows = 0;
  uint32_t recurrentColumns = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return layers && convolutionHistory && convolutionChannels &&
           recurrentGroups && recurrentRows && recurrentColumns;
  }
  [[nodiscard]] constexpr uint64_t convolutionLayerBytes() const noexcept {
    return alignUp(uint64_t{convolutionHistory} * convolutionChannels *
                   bfloat16Bytes);
  }
  [[nodiscard]] constexpr uint64_t convolutionBytes() const noexcept {
    return uint64_t{layers} * convolutionLayerBytes();
  }
  [[nodiscard]] constexpr uint64_t recurrentLayerBytes() const noexcept {
    return alignUp(uint64_t{recurrentGroups} * recurrentRows *
                   recurrentColumns * sizeof(float));
  }
  [[nodiscard]] constexpr uint64_t recurrentBytes() const noexcept {
    return uint64_t{layers} * recurrentLayerBytes();
  }
  [[nodiscard]] constexpr uint64_t cellBytes() const noexcept {
    return convolutionBytes() + recurrentBytes();
  }

  bool operator==(const GdnStateLayout &) const = default;
};

// One ring of SPLASH_DRAFT_SLIDING_WINDOW slots per KV head for the keys and
// one for the values of every draft layer.
struct DraftStateLayout final {
  static constexpr uint32_t bfloat16Bytes = 2;

  uint32_t layers = 0;
  uint32_t kvHeads = 0;
  uint32_t headDimension = 0;
  uint32_t contextWidth = 0;
  bool fullHistory = false;
  std::array<uint32_t, 8> layerContextTokens{};

  [[nodiscard]] constexpr uint32_t contextTokens(uint32_t layer = 0) const noexcept {
    return layer < layerContextTokens.size() && layerContextTokens[layer]
               ? layerContextTokens[layer] : SPLASH_DRAFT_SLIDING_WINDOW;
  }
  [[nodiscard]] constexpr uint32_t historyCapacity() const noexcept {
    uint32_t tokens = SPLASH_DRAFT_SLIDING_WINDOW;
    if (fullHistory)
      for (uint32_t i = 0; i < layers; ++i) tokens = std::max(tokens, contextTokens(i));
    return tokens;
  }
  [[nodiscard]] constexpr bool usesContextWindow() const noexcept {
    return layers && !fullHistory;
  }
  [[nodiscard]] constexpr bool valid() const noexcept {
    if (!layers || layers > layerContextTokens.size() || !kvHeads || !headDimension) return false;
    if (usesContextWindow() && (!contextWidth || contextWidth % SPLASH_DRAFT_CONTEXT_GROUP)) return false;
    for (uint32_t i = 0; i < layers; ++i)
      if (contextTokens(i) % 128) return false;
    return true;
  }
  [[nodiscard]] constexpr uint64_t tensorBytes(uint32_t layer = 0) const noexcept {
    return uint64_t{kvHeads} * contextTokens(layer) * headDimension *
           bfloat16Bytes;
  }
  [[nodiscard]] constexpr uint64_t ringBytes() const noexcept {
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < layers; ++i) bytes += 2 * tensorBytes(i);
    return bytes;
  }

  [[nodiscard]] constexpr uint64_t windowBytes() const noexcept {
    return usesContextWindow() ? draft_context_window_bytes(contextWidth) : 0;
  }
  [[nodiscard]] constexpr uint64_t cachedBytes() const noexcept {
    return fullHistory ? ringBytes() : windowBytes();
  }

  bool operator==(const DraftStateLayout &) const = default;
};

struct CompositeStateLayout final {
  // The GDN cells a lane holds, with one draft ring.
  static constexpr uint32_t kLaneGdnCells = 2;

  GdnStateLayout target;
  DraftStateLayout draft;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return target.valid() && (draft.valid() || draft == DraftStateLayout{});
  }
  [[nodiscard]] constexpr uint64_t laneBytes() const noexcept {
    return kLaneGdnCells * target.cellBytes() + draft.ringBytes() + draft.windowBytes();
  }
  [[nodiscard]] constexpr uint64_t cachedBytes() const noexcept {
    return target.cellBytes() + draft.cachedBytes();
  }

  bool operator==(const CompositeStateLayout &) const = default;
};

} // namespace splash::model
