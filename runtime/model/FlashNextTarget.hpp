#pragma once
#include "model/FlashNextLayout.hpp"
#include "model/GgufShardIndex.hpp"
#include "model/WeightStore.hpp"
#include "ops/Normalization.hpp"
#include <map>
#include <memory>
#include <optional>
namespace splash::model {
struct FlashNextWeightSpec final {
  std::string name;
  std::vector<uint64_t> dimensions;
  bool diskOnly = false;
};
// Public metadata validator; it reads no payload and rejects every missing,
// wrongly shaped or unsupported tensor before any Metal allocation.
[[nodiscard]] std::vector<FlashNextWeightSpec> flashNextWeightSpecs(const FlashNextLayout &);
void requireFlashNextMetadata(const GgufMetadata &, const FlashNextLayout &);
void requireFlashNextWeights(const GgufShardIndex &, const FlashNextLayout &);
[[nodiscard]] std::vector<std::filesystem::path> findFlashNextShards(const std::filesystem::path &);
struct FlashNextWeights final {
  FlashNextLayout layout;
  std::shared_ptr<GgufShardIndex> index;
  std::shared_ptr<WeightPageCache> hostPages;
  std::map<std::string, ops::Projection, std::less<>> projections;
  std::map<std::string, std::pair<uint32_t,uint32_t>, std::less<>> mtpShapes;
  bool hasMtp = false;
  std::map<std::string, metal::MetalBuffer, std::less<>> tensors;
  uint64_t actualAllocatedBytes = 0;
  std::string manifestFingerprintSha256;
  std::vector<WeightFileRecord> files;
  // A lease owns repacked Metal planes until the last graph consuming them
  // completes; resident cache entries are never evicted while leased.
  struct ExpertPage {
    ops::Projection projection;
    uint64_t bytes = 0;
  };
  class ExpertCache;
  std::shared_ptr<ExpertCache> expertPages;
  [[nodiscard]] const ops::Projection &projection(std::string_view name) const;
  [[nodiscard]] ops::NormWeights norm(std::string_view name) const;
};
[[nodiscard]] uint64_t flashNextStaticWeightBytes(const GgufShardIndex &, const FlashNextLayout &);
[[nodiscard]] std::shared_ptr<FlashNextWeights> loadFlashNextWeights(
    metal::MetalBackend &, const std::filesystem::path &, FlashNextLayout, std::string sourceIdentity);
[[nodiscard]] std::shared_ptr<const FlashNextWeights::ExpertPage> acquireFlashNextExpert(
    FlashNextWeights &, metal::MetalBackend &, std::string_view tensor, uint32_t expert);
[[nodiscard]] WeightCacheTelemetry flashNextExpertTelemetry(const FlashNextWeights &);
[[nodiscard]] uint64_t flashNextExpertBudget(const FlashNextWeights &) noexcept;
void setFlashNextExpertBudget(FlashNextWeights &,uint64_t bytes);
[[nodiscard]] uint64_t releaseFlashNextExpertPage(FlashNextWeights &) noexcept;
} // namespace splash::model
