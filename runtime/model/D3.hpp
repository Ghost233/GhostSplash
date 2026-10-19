#pragma once
#include "model/ModernBert.hpp"

namespace splash::model {
// Noncausal full-attention layers with causal GDN recurrence. Only answer-code head rows are resident.
class D3 final {
public:
  D3(metal::MetalBackend &, const std::filesystem::path &gguf, uint32_t context, uint64_t maximumBytes,
      bool q4h = true);
  ~D3();
  void validate(const ModernBertInput &) const;
  std::vector<float> score(std::span<const uint32_t>, std::span<const uint32_t>, uint32_t kind, bool selected = true);
  std::vector<std::vector<float>> scoreBatch(std::span<const ModernBertInput>, bool selected = true);
  uint32_t vocabularySize() const noexcept { return 248320; }
  uint64_t weightBytes() const noexcept;
  uint64_t quantizedWeightBytes() const noexcept;
  uint64_t workspaceBytes() const noexcept;
  metal::CommandTiming lastTiming() const noexcept;
  std::span<const uint32_t> codeTokens() const noexcept;
  uint64_t kvBytes() const noexcept;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
