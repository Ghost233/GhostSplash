#pragma once

#include "ops/CompactProjection.hpp"
#include <filesystem>
#include <memory>
#include <string>

namespace splash::model {

struct CompactTokenRows final {
  std::vector<uint32_t> ids;
  uint32_t valid = 0;
  std::string reason;
};

// Ranked static vocabulary plus at most 256 distinct dynamic prompt/history
// tokens. A rare-token ratio above 1% falls back to the complete head.
[[nodiscard]] CompactTokenRows compactTokenRows(std::span<const uint32_t> ranking,
                                               std::span<const uint32_t> history,
                                               uint32_t capacity);

[[nodiscard]] std::vector<uint32_t> readCompactRanking(const std::filesystem::path &path, uint32_t vocabulary);

class CompactDraftHead final {
public:
  [[nodiscard]] static std::unique_ptr<CompactDraftHead> createIfEnabled(
      bool enabled, metal::MetalBackend &backend, const ops::Projection &source,
      const std::filesystem::path &rankingFile, uint32_t requestedCapacity = 98304);
  CompactDraftHead(metal::MetalBackend &backend, const ops::Projection &source,
                   const std::filesystem::path &rankingFile, uint32_t requestedCapacity = 98304);
  [[nodiscard]] bool prepare(std::span<const uint32_t> history);
  [[nodiscard]] const ops::Projection &projection() const noexcept { return projection_; }
  [[nodiscard]] const metal::MetalBuffer &mapping() const noexcept { return mapping_; }
  [[nodiscard]] uint32_t validRows() const noexcept { return valid_; }
  [[nodiscard]] uint32_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] uint64_t bytes() const noexcept;
  [[nodiscard]] const ops::CompactProjectionValidation &validation() const noexcept { return validation_; }
  [[nodiscard]] const std::string &reason() const noexcept { return reason_; }

private:
  metal::MetalBackend &backend_;
  const ops::Projection &source_;
  std::vector<uint32_t> ranking_;
  ops::Projection projection_;
  metal::MetalBuffer mapping_;
  uint32_t capacity_ = 0, valid_ = 0;
  std::string reason_;
  ops::CompactProjectionValidation validation_;
};

} // namespace splash::model
