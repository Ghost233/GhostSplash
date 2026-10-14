#include "model/CompactDraftHead.hpp"
#include "StderrLine.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>

namespace splash::model {

CompactTokenRows compactTokenRows(std::span<const uint32_t> ranking,
                                  std::span<const uint32_t> history, uint32_t capacity) {
  if (capacity <= 256 || capacity > ranking.size() || capacity % 256)
    throw std::invalid_argument("invalid compact capacity");
  const uint32_t fixed = capacity - 256;
  std::vector<bool> included(ranking.size());
  CompactTokenRows result{{ranking.begin(), ranking.begin() + fixed}, fixed, {}};
  for (uint32_t id : result.ids) included[id] = true;
  uint64_t rare = 0;
  std::vector<uint32_t> dynamic;
  for (uint32_t id : history) {
    if (id >= included.size()) throw std::invalid_argument("compact history token outside vocabulary");
    if (!included[id]) {
      ++rare;
      if (std::find(dynamic.begin(), dynamic.end(), id) == dynamic.end()) dynamic.push_back(id);
    }
  }
  if (dynamic.size() > 256) return {{}, 0, "more than 256 distinct rare tokens"};
  if (rare * 100 > history.size()) return {{}, 0, "rare-token ratio exceeds 1 percent"};
  result.ids.insert(result.ids.end(), dynamic.begin(), dynamic.end());
  result.valid = static_cast<uint32_t>(result.ids.size());
  // Masked rows only pad the native 256-row storage tiles.
  result.ids.resize(capacity, ranking.front());
  return result;
}

std::vector<uint32_t> readCompactRanking(const std::filesystem::path &path, uint32_t expectedVocabulary) {
  if (path.empty() || !std::filesystem::is_regular_file(path))
    throw std::invalid_argument("compact ranking file is missing or not a regular file: " + path.string());
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  const auto size = stream.tellg();
  if (!stream || size < 12)
    throw std::invalid_argument("compact ranking header is truncated or unreadable");
  stream.seekg(0);
  char magic[8]; uint32_t vocabulary = 0;
  stream.read(magic, 8);
  stream.read(reinterpret_cast<char *>(&vocabulary), 4);
  if (std::string_view(magic, 8) == "SPLCHD01")
    throw std::invalid_argument("legacy compact ranking requires one-time --import-legacy conversion");
  if (!stream || std::string_view(magic, 8) != "SPLCHD02")
    throw std::invalid_argument("compact ranking format or version is invalid");
  if (!vocabulary || vocabulary != expectedVocabulary)
    throw std::invalid_argument("compact ranking vocabulary size does not match the current head");
  if (static_cast<uint64_t>(size) != 12 + uint64_t{vocabulary} * 4)
    throw std::invalid_argument("compact ranking size is invalid (truncated or trailing IDs)");
  std::vector<uint32_t> ranking(vocabulary);
  stream.read(reinterpret_cast<char *>(ranking.data()), uint64_t{vocabulary} * 4);
  std::vector<bool> seen(vocabulary);
  for (uint32_t id : ranking) {
    if (!stream || id >= vocabulary || seen[id])
      throw std::invalid_argument("compact ranking is not a complete vocabulary permutation (invalid or duplicate ID)");
    seen[id] = true;
  }
  return ranking;
}

std::unique_ptr<CompactDraftHead> CompactDraftHead::createIfEnabled(
    bool enabled, metal::MetalBackend &backend, const ops::Projection &source,
    const std::filesystem::path &rankingFile, uint32_t requestedCapacity) {
  if (!enabled) return {};
  return std::make_unique<CompactDraftHead>(backend, source, rankingFile, requestedCapacity);
}

CompactDraftHead::CompactDraftHead(metal::MetalBackend &backend, const ops::Projection &source,
                                   const std::filesystem::path &rankingFile, uint32_t requestedCapacity)
    : backend_(backend), source_(source) {
  if (requestedCapacity <= 256 || requestedCapacity > 98304 || requestedCapacity % 256)
    throw std::invalid_argument("compact capacity must be a multiple of 256 in 512..98304");
  if (!source.takesPlaneViews() || source.destination != ops::FloatOutput::Float32 ||
      source.blocks().segments.front().outputSize != source.outputSize)
    throw std::invalid_argument("compact head layout is not one unrotated quantized FP32 projection");
  ranking_ = readCompactRanking(rankingFile, source.outputSize);
  capacity_ = std::min<uint32_t>(requestedCapacity, source.outputSize / 2 / 256 * 256);
  if (capacity_ <= 256)
    throw std::invalid_argument("vocabulary is too small for compact head");
  std::vector<uint32_t> ids(ranking_.begin(), ranking_.begin() + capacity_);
  projection_ = ops::gatherProjection(backend_, source_, ids);
  mapping_ = backend_.allocateBuffer(uint64_t{capacity_} * 4,
                                     metal::BufferStorage::Shared, "compact-draft-token-map");
  const uint32_t fixed = capacity_ - 256;
  for (uint32_t i = 0; i < 256; ++i)
    ids[fixed + i] = ranking_[fixed + uint64_t{i} * (ranking_.size() - fixed) / 256];
  std::memcpy(mapping_.contents(), ids.data(), uint64_t{capacity_} * 4);
  // Validate both the initial gather and the dynamic-row update path once.
  ops::updateProjectionRows(backend_, source_, projection_, mapping_, fixed, 256);
  validation_ = ops::validateCompactProjection(backend_, source_, projection_, mapping_);
  logLine("compact head enabled: public ranking=", rankingFile.string(), "; vocabulary=", source_.outputSize,
          "; capacity=", capacity_, "; numerical_validation=passed; comparisons=", validation_.comparisons);
}

bool CompactDraftHead::prepare(std::span<const uint32_t> history) {
  if (ranking_.empty()) return false;
  const auto selected = compactTokenRows(ranking_, history, capacity_);
  if (!selected.reason.empty()) { reason_ = selected.reason; return false; }
  const uint32_t fixed = capacity_ - 256;
  if (std::memcmp(static_cast<uint32_t *>(mapping_.contents()) + fixed,
                  selected.ids.data() + fixed, 256 * 4)) {
    std::memcpy(static_cast<uint32_t *>(mapping_.contents()) + fixed,
                selected.ids.data() + fixed, 256 * 4);
    ops::updateProjectionRows(backend_, source_, projection_, mapping_, fixed, 256);
  }
  valid_ = selected.valid;
  reason_.clear();
  return true;
}

uint64_t CompactDraftHead::bytes() const noexcept {
  uint64_t bytes = mapping_.sizeBytes();
  for (const auto &s : projection_.blocks().segments)
    bytes += s.plane0.sizeBytes() + s.plane1.sizeBytes() + s.meta.sizeBytes();
  return bytes;
}

} // namespace splash::model
