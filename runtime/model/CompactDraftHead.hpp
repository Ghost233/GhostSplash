#pragma once

#include "ops/CompactProjection.hpp"
#include <filesystem>
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

class CompactDraftHead final {
public:
  CompactDraftHead(metal::MetalBackend &backend, const ops::Projection &source,
                   const std::filesystem::path &rankingFile, std::string_view manifest);
  [[nodiscard]] bool prepare(std::span<const uint32_t> history);
  [[nodiscard]] const ops::Projection &projection() const noexcept { return projection_; }
  [[nodiscard]] const metal::MetalBuffer &mapping() const noexcept { return mapping_; }
  [[nodiscard]] uint32_t validRows() const noexcept { return valid_; }
  [[nodiscard]] uint64_t bytes() const noexcept;
  [[nodiscard]] const std::string &reason() const noexcept { return reason_; }

private:
  metal::MetalBackend &backend_;
  const ops::Projection &source_;
  std::vector<uint32_t> ranking_;
  ops::Projection projection_;
  metal::MetalBuffer mapping_;
  uint32_t capacity_ = 0, valid_ = 0;
  std::string reason_;
};

} // namespace splash::model
