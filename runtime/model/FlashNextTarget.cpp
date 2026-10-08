#include "model/FlashNextTarget.hpp"
#include "model/FlashNextNgram.hpp"
#include "Checked.hpp"
#include "model/GgufImageLayout.hpp"
#include "metal/abi/GgufRepack.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
namespace splash::model {
namespace {
float sourceFloat(const uint8_t *bytes, uint32_t type, uint64_t index) {
  float f;
  if (type == 0) std::memcpy(&f, bytes + 4 * index, 4);
  else if (type == 30) { uint16_t b; std::memcpy(&b, bytes + 2 * index, 2); uint32_t v = uint32_t{b} << 16; std::memcpy(&f, &v, 4); }
  else if (type == 1) { __fp16 h; std::memcpy(&h, bytes + 2 * index, 2); f = h; }
  else throw GgufError("Flash Next float tensor has unsupported type");
  return f;
}
bool floatType(uint32_t type) { return type == 0 || type == 1 || type == 30; }
uint64_t preparedBytes(const GgufTensor &t, uint64_t rows) {
  if (floatType(t.type)) return ((rows + 7) / 8 * 8) * t.columns() * 4;
  const uint32_t id = gguf_format_of(t.type);
  if (id == GGUF_FMT_COUNT) throw GgufError("unsupported Flash Next projection type");
  const auto planes = ggufPlaneBytes(kQuantFormats[id], (rows + 255) / 256 * 256, (t.columns() + 255) / 256 * 256);
  return planes.plane0 + planes.plane1 + planes.meta;
}
ops::Projection prepareProjection(metal::MetalBackend &backend, WeightPageCache &host,
                                  const GgufShardTensor &tensor, WeightRange range,
                                  WeightPageKey key, uint32_t rows, uint32_t columns) {
  const auto tile = [&](uint32_t first, uint32_t count, uint64_t rowBytes) {
    WeightRange part = range; part.offset += uint64_t{first} * rowBytes; part.bytes = uint64_t{count} * rowBytes;
    return host.acquire({key.kind, key.tensor, (key.unit << 32) | first}, part);
  };
  if (floatType(tensor.tensor.type)) {
    const uint32_t padded = (rows + 7) / 8 * 8;
    auto values = backend.allocateBuffer(uint64_t{padded} * columns * 4, metal::BufferStorage::Shared, key.tensor);
    auto *out = static_cast<float *>(values.contents());
    std::memset(out, 0, values.sizeBytes());
    const uint64_t sourceRowBytes = uint64_t{columns} * (tensor.tensor.type == 0 ? 4 : 2);
    for (uint32_t first = 0; first < rows; first += 256) {
      const uint32_t count = std::min(256u, rows - first);
      auto native = tile(first, count, sourceRowBytes);
      for (uint64_t i = 0; i < uint64_t{count} * columns; ++i) out[uint64_t{first} * columns + i] = sourceFloat(native.data(), tensor.tensor.type, i);
    }
    return {padded, columns, ops::BlockWeights{{ops::QuantizedSegment::floats(padded, columns, values)}}};
  }
  const uint32_t id = gguf_format_of(tensor.tensor.type);
  const auto &format = kQuantFormats[id];
  const uint32_t padded = (rows + 255) / 256 * 256;
  const uint32_t physicalColumns = (columns + 255) / 256 * 256;
  const uint64_t sourceRowBytes = ggufRowBytes(format, columns);
  const uint64_t rowBytes = ggufRowBytes(format, physicalColumns);
  const auto p = ggufPlaneBytes(format, padded, physicalColumns);
  auto packed = backend.allocateBuffer(p.plane0 + p.plane1 + p.meta, metal::BufferStorage::Shared, key.tensor);
  // Staging contains at most one tile (up to 256 native rows), never a second
  // full expert cache. Padding rows are true zero-scaled native blocks.
  auto staging = backend.allocateBuffer(256 * rowBytes, metal::BufferStorage::Shared, "flash-weight-staging");
  for (uint32_t first = 0; first < padded; first += 256) {
    std::memset(staging.contents(), 0, staging.sizeBytes());
    const uint32_t count = first < rows ? std::min(256u, rows - first) : 0;
    if (count) {
      auto native = tile(first, count, sourceRowBytes);
      for (uint32_t row = 0; row < count; ++row)
        std::memcpy(static_cast<uint8_t *>(staging.contents()) + row * rowBytes,
                    native.data() + row * sourceRowBytes, sourceRowBytes);
    }
    const uint64_t groups = physicalColumns / 32;
    const auto above = ggufPlaneBytes(format, first, physicalColumns);
    const uint64_t begin = above.plane0;
    const uint64_t metaAt = p.plane0 + p.plane1 + above.meta;
    const uint64_t plane1At = p.plane0 + above.plane1;
    GgufRepackParams params{256, physicalColumns, id, uint32_t(rowBytes),
                            p.plane1 ? uint32_t(plane1At - begin) : 0, uint32_t(metaAt - begin)};
    static_cast<void>(backend.submit({"gguf_repack", {{0, staging}, {1, backend.view(packed, begin, packed.sizeBytes() - begin)}},
      {{2, &params, sizeof(params)}}, {groups, 1, 1}, {256, 1, 1}}));
  }
  auto segment = ops::QuantizedSegment::planes(id, rows, physicalColumns,
      backend.view(packed, 0, p.plane0), p.plane1 ? backend.view(packed, p.plane0, p.plane1) : metal::MetalBuffer{},
      backend.view(packed, p.plane0 + p.plane1, p.meta));
  return {padded, physicalColumns, ops::BlockWeights{{segment}}};
}
} // namespace
bool FlashNextLayout::valid() const noexcept {
  const FlashNextLayout expected;
  return maximumContextTokens && maximumContextTokens <= expected.maximumContextTokens &&
         layers == 48 && hiddenSize == 2560 && vocabularySize == 248320 && streams == 4 && lowRank == 320 &&
         experts == 512 && expertsPerToken == 10 && intermediateSize == 640 &&
         queryHeads == 24 && kvHeads == 2 && headDimension == 256 && rotaryDimension == 64 &&
         fullAttentionPeriod == 4 && indexHeads == 4 && indexWidth == 128 && indexRatio == 4 && indexBudget == 2048 &&
         pleLayer == 1 && pleHeads == 16 && pleRowWidth == 160 && pleTaps == 4 && pleDilation == 3 &&
         epsilon == expected.epsilon && rotaryTheta == expected.rotaryTheta && expertCacheBytes && hostCacheBytes;
}
std::vector<FlashNextWeightSpec> flashNextWeightSpecs(const FlashNextLayout &l) {
  if (!l.valid()) throw GgufError("unsupported Flash Next geometry");
  std::vector<FlashNextWeightSpec> out;
  auto add = [&](std::string name, std::initializer_list<uint64_t> dims, bool disk = false) {
    out.push_back({std::move(name), dims, disk});
  };
  auto hc = [&](const std::string &p, bool inject) {
    add(p + "norm.weight", {l.hyperWidth()}); add(p + "down.weight", {l.hyperWidth(), l.lowRank});
    add(p + "up.weight", {l.lowRank, l.hyperWidth()});
    if (inject) add(p + "inject.weight", {l.hyperWidth(), l.streams});
  };
  add("token_embd.weight", {l.hiddenSize, l.vocabularySize}, true);
  add("output.weight", {l.hiddenSize, l.vocabularySize}); hc("output_hc_", false);
  // Number of rows is validated using the PLE offsets and vocab-size arrays.
  add("per_layer_token_embd.weight", {l.pleRowWidth, 0}, true);
  for (uint32_t i = 0; i < l.layers; ++i) {
    const std::string p = "blk." + std::to_string(i) + ".";
    hc(p + "hc_attn_", true); hc(p + "hc_ffn_", true);
    add(p + "ffn_gate_inp.weight", {l.hiddenSize, l.experts});
    add(p + "ffn_gate_inp_shexp.weight", {l.hiddenSize});
    for (const char *role : {"gate", "up"}) {
      add(p + "ffn_" + role + "_exps.weight", {l.hiddenSize, l.intermediateSize, l.experts}, true);
      add(p + "ffn_" + role + "_shexp.weight", {l.hiddenSize, l.intermediateSize});
    }
    add(p + "ffn_down_exps.weight", {l.intermediateSize, l.hiddenSize, l.experts}, true);
    add(p + "ffn_down_shexp.weight", {l.intermediateSize, l.hiddenSize});
    if (l.isAttention(i)) {
      add(p + "attn_q.weight", {l.hiddenSize, 2ull * l.queryHeads * l.headDimension});
      for (const char *role : {"k", "v"}) add(p + "attn_" + role + ".weight", {l.hiddenSize, l.kvHeads * l.headDimension});
      add(p + "attn_output.weight", {l.queryHeads * l.headDimension, l.hiddenSize});
      for (const char *role : {"q", "k"}) add(p + "attn_" + role + "_norm.weight", {l.headDimension});
      add(p + "indexer.q_proj.weight", {l.hiddenSize, l.indexHeads * l.indexWidth});
      add(p + "indexer.k_proj.weight", {l.hiddenSize, l.indexWidth});
      for (const char *role : {"q", "k"}) add(p + "indexer." + role + "_norm.weight", {l.indexWidth});
    } else {
      add(p + "attn_qkv.weight", {l.hiddenSize, 10240}); add(p + "attn_gate.weight", {l.hiddenSize, 6144});
      for (const char *role : {"alpha", "beta"}) add(p + "ssm_" + role + ".weight", {l.hiddenSize, 48});
      add(p + "ssm_conv1d.weight", {4, 10240}); add(p + "ssm_dt.bias", {48}); add(p + "ssm_a", {48});
      add(p + "ssm_norm.weight", {128}); add(p + "ssm_out.weight", {6144, l.hiddenSize});
    }
    if (i == l.pleLayer) {
      add(p + "ple_key.weight", {l.hiddenSize, l.hyperWidth()});
      add(p + "ple_value.weight", {l.hiddenSize, l.hiddenSize});
      for (const char *role : {"key", "query", "conv"}) add(p + "ple_norm_" + role + ".weight", {l.hyperWidth()});
      add(p + "ple_conv1d.weight", {l.pleTaps, l.hyperWidth()});
    }
  }
  return out;
}
void requireFlashNextMetadata(const GgufMetadata &m, const FlashNextLayout &l) {
  if (!l.valid() || m.architecture() != "qwen4exp") throw GgufError("unsupported Flash Next GGUF architecture");
  for (const auto &[key, expected] : std::initializer_list<std::pair<const char *, uint64_t>>{
      {"block_count", l.layers}, {"embedding_length", l.hiddenSize}, {"expert_count", l.experts},
      {"expert_used_count", l.expertsPerToken}, {"expert_feed_forward_length", l.intermediateSize},
      {"attention.head_count", l.queryHeads}, {"attention.head_count_kv", l.kvHeads},
      {"attention.key_length", l.headDimension}, {"attention.value_length", l.headDimension},
      {"ssm.conv_kernel", 4}, {"ssm.inner_size", 6144}, {"ssm.state_size", 128},
      {"ssm.time_step_rank", 48}, {"ssm.group_count", 16},
      {"expert_shared_feed_forward_length", l.intermediateSize}, {"hyper_connection.count", l.streams},
      {"hyper_connection.low_rank", l.lowRank}, {"full_attention_interval", l.fullAttentionPeriod},
      {"rope.dimension_count", l.rotaryDimension}, {"attention.indexer.head_count", l.indexHeads},
      {"attention.indexer.key_length", l.indexWidth}, {"attention.indexer.top_k", l.indexBudget},
      {"ple.ngram_size", 3}, {"ple.heads_per_ngram", 8}, {"ple.conv_kernel", l.pleTaps},
      {"ple.eos_token_id", 248044}, {"embedding_length_per_layer_input", l.pleRowWidth}}) {
    if (m.unsignedValue(std::string("qwen4exp.") + key) != expected)
      throw GgufError(std::string("Flash Next metadata mismatch: ") + key);
  }
}
void requireFlashNextWeights(const GgufShardIndex &index, const FlashNextLayout &l) {
  requireFlashNextMetadata(index.metadata(), l);
  auto ple = index.unsignedArray("qwen4exp.ple.layers");
  if (!ple || ple->size() != 1 || (*ple)[0] != l.pleLayer) throw GgufError("unsupported Flash Next PLE layer arrangement");
  const auto recurrent = index.unsignedArray("qwen4exp.attention.recurrent_layers");
  const auto ratios = index.unsignedArray("qwen4exp.attention.compress_ratios");
  if (!recurrent || !ratios || recurrent->size() != l.layers || ratios->size() != l.layers)
    throw GgufError("invalid Flash Next attention layer arrays");
  for (uint32_t i = 0; i < l.layers; ++i)
    if ((*recurrent)[i] != uint64_t(!l.isAttention(i)) || (*ratios)[i] != (l.isAttention(i) ? l.indexRatio : 0))
      throw GgufError("unsupported Flash Next attention layer arrangement");
  FlashNextNgramConfig config;
  config.embeddingRows = index.require("per_layer_token_embd.weight").tensor.rows();
  const auto copy = [&](const char *key, auto &to) {
    auto values = index.unsignedArray(key);
    if (!values || values->size() != to.size()) throw GgufError(std::string("invalid Flash Next PLE array: ") + key);
    std::copy(values->begin(), values->end(), to.begin());
  };
  copy("qwen4exp.ple.layer_multipliers", config.multipliers);
  copy("qwen4exp.ple.head_offsets", config.offsets);
  copy("qwen4exp.ple.head_vocab_sizes", config.vocabularies);
  static_cast<void>(FlashNextNgram(config));
  for (const auto &spec : flashNextWeightSpecs(l)) {
    const auto &t = index.require(spec.name).tensor;
    if (spec.name == "per_layer_token_embd.weight") {
      if (t.dims.size() != 2 || t.columns() != l.pleRowWidth || !t.rows()) throw GgufError("invalid PLE embedding shape");
    } else if (t.dims != spec.dimensions) throw GgufError("Flash Next tensor shape mismatch: " + spec.name);
    if (!floatType(t.type) && gguf_format_of(t.type) == GGUF_FMT_COUNT)
      throw GgufError("unsupported Flash Next tensor type: " + spec.name);
    if ((spec.name == "token_embd.weight" || spec.name == "per_layer_token_embd.weight") && gguf_format_of(t.type) == GGUF_FMT_COUNT)
      throw GgufError("Flash Next embedding must use a native quantized format");
    if (!spec.diskOnly && (spec.dimensions.size() == 1 || spec.name.ends_with("conv1d.weight")) && !floatType(t.type))
      throw GgufError("Flash Next vector or convolution must be float: " + spec.name);
    if (spec.name.ends_with("norm.weight") && t.type != 0) throw GgufError("Flash Next norm must be F32: " + spec.name);
  }
}
std::vector<std::filesystem::path> findFlashNextShards(const std::filesystem::path &directory) {
  std::vector<std::filesystem::path> paths;
  for (const auto &entry : std::filesystem::directory_iterator(directory))
    if (entry.is_regular_file() && entry.path().extension() == ".gguf") paths.push_back(entry.path());
  std::sort(paths.begin(), paths.end());
  if (paths.empty()) throw GgufError("Flash Next model directory has no GGUF shards");
  return paths;
}
const ops::Projection &FlashNextWeights::projection(std::string_view name) const { return projections.at(std::string(name)); }
ops::NormWeights FlashNextWeights::norm(std::string_view name) const { return {tensors.at(std::string(name)), true}; }
uint64_t flashNextStaticWeightBytes(const GgufShardIndex &index, const FlashNextLayout &l) {
  requireFlashNextWeights(index, l);
  uint64_t bytes = 0;
  for (const auto &spec : flashNextWeightSpecs(l)) if (!spec.diskOnly) {
    const auto &t = index.require(spec.name).tensor;
    bytes += spec.dimensions.size() == 2 && !spec.name.ends_with("conv1d.weight")
      ? preparedBytes(t, t.rows()) : t.elements() * 4;
  }
  return bytes;
}
// LRU entries hold immutable repacked Metal weights. A live shared_ptr is a
// pin; misses fail rather than exceed the budget if every entry is pinned.
class FlashNextWeights::ExpertCache {
public:
  struct Entry { std::shared_ptr<ExpertPage> page; uint64_t last; };
  explicit ExpertCache(uint64_t budget) : budget_(budget) {}
  std::map<WeightPageKey, Entry> entries;
  uint64_t clock = 0;
  WeightCacheTelemetry telemetry;
  uint64_t budget_;
  std::mutex mutex;
};
std::shared_ptr<const FlashNextWeights::ExpertPage> acquireFlashNextExpert(
    FlashNextWeights &w, metal::MetalBackend &backend, std::string_view name, uint32_t expert) {
  const WeightPageKey key{WeightUnitKind::Expert, std::string(name), expert};
  auto &cache = *w.expertPages;
  std::lock_guard lock(cache.mutex);
  auto found = cache.entries.find(key);
  if (found != cache.entries.end()) { ++cache.telemetry.hits; found->second.last = ++cache.clock; return found->second.page; }
  const auto &tensor = w.index->require(name);
  const auto range = w.index->expert(name, expert);
  const uint32_t columns = tensor.tensor.dims.at(0), rows = tensor.tensor.dims.at(1);
  const uint64_t needed = preparedBytes(tensor.tensor, rows);
  if (needed > cache.budget_) throw GgufError("Flash Next expert exceeds resident cache budget");
  while (cache.telemetry.residentBytes + needed > cache.budget_) {
    auto victim = cache.entries.end();
    for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it)
      if (it->second.page.use_count() == 1 && (victim == cache.entries.end() || it->second.last < victim->second.last)) victim = it;
    if (victim == cache.entries.end()) throw GgufError("Flash Next resident cache budget is pinned");
    cache.telemetry.residentBytes -= victim->second.page->bytes; cache.entries.erase(victim); ++cache.telemetry.evictions;
  }
  ++cache.telemetry.misses;
  auto page = std::make_shared<FlashNextWeights::ExpertPage>();
  const auto bytesBefore = w.hostPages->telemetry().bytesRead;
  page->projection = prepareProjection(backend, *w.hostPages, tensor, range, key, rows, columns);
  page->bytes = needed;
  cache.telemetry.bytesRead += w.hostPages->telemetry().bytesRead - bytesBefore; cache.telemetry.residentBytes += needed;
  cache.entries.emplace(key, FlashNextWeights::ExpertCache::Entry{page, ++cache.clock});
  return page;
}
uint64_t releaseFlashNextExpertPage(FlashNextWeights &w) noexcept {
  auto &cache = *w.expertPages;
  std::lock_guard lock(cache.mutex);
  auto victim = cache.entries.end();
  for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it)
    if (it->second.page.use_count() == 1 && (victim == cache.entries.end() || it->second.last < victim->second.last)) victim = it;
  if (victim == cache.entries.end()) return 0;
  const auto bytes = victim->second.page->bytes;
  cache.telemetry.residentBytes -= bytes; cache.entries.erase(victim); ++cache.telemetry.evictions; return bytes;
}
WeightCacheTelemetry flashNextExpertTelemetry(const FlashNextWeights &w) {
  std::lock_guard lock(w.expertPages->mutex); return w.expertPages->telemetry;
}
std::shared_ptr<FlashNextWeights> loadFlashNextWeights(metal::MetalBackend &backend,
    const std::filesystem::path &root, FlashNextLayout layout, std::string identity) {
  auto w = std::make_shared<FlashNextWeights>(); w->layout = layout;
  w->index = std::make_shared<GgufShardIndex>(findFlashNextShards(root / "target"));
  requireFlashNextWeights(*w->index, layout);
  w->hostPages = std::make_shared<WeightPageCache>(layout.hostCacheBytes);
  w->expertPages = std::make_shared<FlashNextWeights::ExpertCache>(layout.expertCacheBytes);
  for (const auto &spec : flashNextWeightSpecs(layout)) if (!spec.diskOnly) {
    const auto &t = w->index->require(spec.name);
    const WeightPageKey key{WeightUnitKind::Tensor, spec.name, 0};
    const auto range = w->index->rows(spec.name, 0, t.tensor.rows());
    if (spec.dimensions.size() == 2 && !spec.name.ends_with("conv1d.weight")) {
      w->projections.emplace(spec.name, prepareProjection(backend, *w->hostPages, t, range, key, t.tensor.rows(), t.tensor.columns()));
      w->actualAllocatedBytes += preparedBytes(t.tensor, t.tensor.rows());
    } else {
      auto native = w->hostPages->acquire(key, range);
      auto buffer = backend.allocateBuffer(t.tensor.elements() * 4, metal::BufferStorage::Shared, spec.name);
      auto *f = static_cast<float *>(buffer.contents());
      for (uint64_t i = 0; i < t.tensor.elements(); ++i) f[i] = sourceFloat(native.data(), t.tensor.type, i);
      w->actualAllocatedBytes += buffer.sizeBytes(); w->tensors.emplace(spec.name, buffer);
    }
  }
  for (const auto &path : findFlashNextShards(root / "target"))
    w->files.push_back({path.filename().string(), "GGUF", 0, 0, std::filesystem::file_size(path), identity});
  w->manifestFingerprintSha256 = weightManifestFingerprint(w->files);
  return w;
}
} // namespace splash::model
