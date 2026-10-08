#pragma once

#include "model/GgufFile.hpp"
#include "model/WeightPageCache.hpp"

#include <filesystem>
#include <map>
#include <string_view>
#include <vector>

namespace splash::model {

struct GgufShardTensor {
  GgufTensor tensor;
  std::filesystem::path path;
  uint64_t offset = 0; // absolute file offset, tensor.offset is relative
  std::shared_ptr<const WeightShardIdentity> source;
};

// Reads just headers with exact pread calls: no mmap or payload prefetch.
// HeaderOnly is for inspecting downloads; its incomplete shards cannot be
// used by WeightPageCache. All listed shard headers must be available.
class GgufShardIndex {
public:
  enum class Mode { Complete, HeaderOnly };
  explicit GgufShardIndex(const std::vector<std::filesystem::path> &paths, Mode mode = Mode::Complete);
  [[nodiscard]] const GgufMetadata &metadata() const noexcept { return metadata_; }
  // Integer arrays retain all64bits (PLE hash multipliers must not go through double).
  [[nodiscard]] std::optional<std::span<const uint64_t>> unsignedArray(std::string_view name) const;
  [[nodiscard]] const std::vector<GgufShardTensor> &tensors() const noexcept { return tensors_; }
  [[nodiscard]] const GgufShardTensor &require(std::string_view name) const;
  [[nodiscard]] WeightRange rows(std::string_view name, uint64_t first, uint64_t count) const;
  [[nodiscard]] WeightRange expert(std::string_view name, uint64_t index) const;
private:
  GgufMetadata metadata_;
  std::map<std::string, std::vector<uint64_t>, std::less<>> arrays_;
  std::vector<GgufShardTensor> tensors_;
  std::map<std::string, size_t, std::less<>> lookup_;
};

} // namespace splash::model
