#pragma once
#include "metal/MetalBackend.hpp"
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace splash::model {
struct ModernBertInput {
  std::span<const uint32_t> tokens, options;
  uint32_t kind = 0;
};

// Bidirectional decision encoder. No causal KV or recurrent prefix state.
class ModernBert final {
public:
  ModernBert(metal::MetalBackend &, const std::filesystem::path &gguf,
             uint32_t context, uint64_t maximumBytes);
  ~ModernBert();
  ModernBert(const ModernBert &) = delete;
  ModernBert &operator=(const ModernBert &) = delete;
  std::vector<float> score(std::span<const uint32_t> tokens,
                           std::span<const uint32_t> options, uint32_t kind,
                           bool optionQueries = true);
  void validate(const ModernBertInput &) const;
  std::vector<std::vector<float>> scoreBatch(std::span<const ModernBertInput>, bool optionQueries = true);
  uint32_t vocabularySize() const noexcept;
  uint32_t maskToken() const noexcept;
  uint32_t bosToken() const noexcept;
  uint32_t sepToken() const noexcept;
  uint64_t weightBytes() const noexcept;
  uint64_t quantizedWeightBytes() const noexcept;
  uint64_t workspaceBytes() const noexcept;
  metal::CommandTiming lastTiming() const noexcept;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace splash::model
