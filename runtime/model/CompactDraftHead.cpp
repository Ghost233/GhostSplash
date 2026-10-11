#include "model/CompactDraftHead.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>

namespace splash::model {

CompactTokenRows compactTokenRows(std::span<const uint32_t> ranking,
                                  std::span<const uint32_t> history, uint32_t capacity) {
  if (capacity <= 256 || capacity > ranking.size() || capacity % 256)
    return {{}, 0, "invalid compact capacity"};
  const uint32_t fixed = capacity - 256;
  std::vector<bool> included(ranking.size());
  CompactTokenRows result{{ranking.begin(), ranking.begin() + fixed}, fixed, {}};
  for (uint32_t id : result.ids) included[id] = true;
  uint64_t rare = 0;
  std::vector<uint32_t> dynamic;
  for (uint32_t id : history) {
    if (id >= included.size()) return {{}, 0, "history token outside vocabulary"};
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

CompactDraftHead::CompactDraftHead(metal::MetalBackend &backend, const ops::Projection &source,
                                   const std::filesystem::path &rankingFile,
                                   std::string_view manifest)
    : backend_(backend), source_(source) {
  if (!source.takesPlaneViews() || source.destination != ops::FloatOutput::Float32 ||
      source.blocks().segments.front().outputSize != source.outputSize) {
    reason_ = "head is not one unrotated quantized FP32 projection"; return;
  }
  std::ifstream stream(rankingFile, std::ios::binary | std::ios::ate);
  if (!stream) { reason_ = "model-bound compact ranking is missing"; return; }
  const auto size = stream.tellg();
  const uint64_t expected = 76 + uint64_t{source.outputSize} * 4;
  if (size < 0 || static_cast<uint64_t>(size) != expected) {
    reason_ = "compact ranking size does not match vocabulary"; return;
  }
  stream.seekg(0);
  char magic[8], fingerprint[64];
  uint32_t vocabulary = 0;
  stream.read(magic, 8); stream.read(fingerprint, 64);
  stream.read(reinterpret_cast<char *>(&vocabulary), 4);
  if (!stream || std::string_view(magic, 8) != "SPLCHD01" ||
      std::string_view(fingerprint, 64) != manifest || vocabulary != source.outputSize) {
    reason_ = "compact ranking model identity mismatch"; return;
  }
  ranking_.resize(vocabulary);
  stream.read(reinterpret_cast<char *>(ranking_.data()), uint64_t{vocabulary} * 4);
  std::vector<bool> seen(vocabulary);
  for (uint32_t id : ranking_) {
    if (!stream || id >= vocabulary || seen[id]) {
      ranking_.clear(); reason_ = "compact ranking is not a vocabulary permutation"; return;
    }
    seen[id] = true;
  }
  capacity_ = std::min<uint32_t>(98304, vocabulary / 2 / 256 * 256);
  if (capacity_ <= 256) { ranking_.clear(); reason_ = "vocabulary too small for compact head"; return; }
}

bool CompactDraftHead::prepare(std::span<const uint32_t> history) {
  if (ranking_.empty()) return false;
  const auto selected = compactTokenRows(ranking_, history, capacity_);
  if (!selected.reason.empty()) { reason_ = selected.reason; return false; }
  const uint32_t fixed = capacity_ - 256;
  if (!mapping_) {
    projection_ = ops::gatherProjection(backend_, source_, selected.ids);
    mapping_ = backend_.allocateBuffer(uint64_t{capacity_} * 4,
                                        metal::BufferStorage::Shared, "compact-draft-token-map");
    std::memcpy(mapping_.contents(), selected.ids.data(), uint64_t{capacity_} * 4);
  } else if (std::memcmp(static_cast<uint32_t *>(mapping_.contents()) + fixed,
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
