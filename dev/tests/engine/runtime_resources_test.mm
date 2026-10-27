#include "GgufFixtures.hpp"
#include "ProtocolPeer.hpp"
#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "TestCheckpoint.hpp"
#include "TestDrafts.hpp"
#include "TestStderr.hpp"
#include "TestAdmission.hpp"
#include "Checked.hpp"
#include "engine/Bootstrap.hpp"
#include "engine/RuntimeResources.hpp"
#include "engine/Engine.hpp"
#include "engine/MemoryPlan.hpp"
#include "model/ModelSizing.hpp"
#include "model/RuntimeArenas.hpp"
#include "model/SsdReaders.hpp"

#import <Foundation/Foundation.h>

#include <sys/time.h>

#include <algorithm>
#include <atomic>
#include <future>
#include <thread>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

std::atomic<bool> observeLargeHostGrowth{false};
std::atomic<uint64_t> largeHostGrowths{0}, largeHostBytes{0}, firstLargeHostBytes{0}, backgroundHostGrowths{0};
std::thread::id observedHostOwner;
thread_local size_t failingMetadataBytes = 0;
thread_local int matchingMetadataAllocations = -1;
void *operator new(size_t bytes) {
  if (bytes == failingMetadataBytes && matchingMetadataAllocations >= 0 && matchingMetadataAllocations-- == 0)
    throw std::bad_alloc();
  if (observeLargeHostGrowth.load(std::memory_order_acquire) && bytes >= (1ULL << 20)) {
    if (largeHostGrowths.fetch_add(1, std::memory_order_relaxed) == 0) firstLargeHostBytes = bytes;
    largeHostBytes.fetch_add(bytes, std::memory_order_relaxed);
    if (std::this_thread::get_id() != observedHostOwner) backgroundHostGrowths.fetch_add(1, std::memory_order_relaxed);
  }
  if (void *memory = std::malloc(bytes ? bytes : 1)) return memory;
  throw std::bad_alloc();
}
void operator delete(void *memory) noexcept { std::free(memory); }

namespace {

using namespace splash;
using namespace splash::engine;

using splash::test::rejects;
using splash::test::require;

struct AutomaticStorageClock final {
  std::shared_ptr<std::atomic<uint64_t>> now = std::make_shared<std::atomic<uint64_t>>(0);
  explicit AutomaticStorageClock(RuntimeResourcesConfig &config) {
    auto events = std::make_shared<std::atomic<uint64_t>>(0);
    config.memoryPressure = [source = config.memoryPressure, events] {
      const auto level = source ? source() : MemoryPressure::Normal;
      if (level == MemoryPressure::Critical) events->fetch_add(1);
      return level;
    };
    config.memoryPolicy = {[clock = now] { return clock->load(); }, [source = config.memoryPressure, events] {
      const auto level = source();
      return MemoryPolicyObservation{MemoryPolicyObservation::Pressure(uint32_t(level) + 1), events->load()};
    }};
  }
  void advance(const MemoryGovernorSnapshot &group) const {
    if (!group.pressureControlEnabled || (group.automaticGrowthReason != "headroom_observation" &&
        group.automaticGrowthReason != "ready" && group.automaticGrowthReason != "block_cooldown" &&
        group.automaticGrowthReason != "automatic_outstanding")) return;
    const uint64_t stable = 30'000 - std::min<uint64_t>(30'000, group.normalStableMilliseconds);
    const auto target = now->load() + std::max(stable, group.blockCooldownMilliseconds);
    auto before = now->load();
    while (before < target && !now->compare_exchange_weak(before, target)) {}
  }
};

metal::AllocationResult growAutomaticTo(RuntimeResources &resources, const AutomaticStorageClock &clock,
                                       bool table, uint64_t target, std::chrono::steady_clock::time_point deadline) {
  // This bounded owner acts for an admitted request; Ready alone is idle.
  auto *experts = resources.modelContext().model.expertStore();
  auto *ple = resources.modelContext().model.streamedTable();
  for (;;) {
    const uint64_t actual = table ? ple->committedBytes() : experts->allocatedBytes();
    if (actual == target) return {};
    require(actual < target && std::chrono::steady_clock::now() < deadline,
            "paced seed did not reach its original target within the original grow deadline");
    clock.advance(resources.memoryGovernor().evaluate());
    const auto result = resources.growAutomaticStorageOne(table ? experts->allocatedBytes() : target,
        table ? target : ple ? ple->committedBytes() : 0);
    if (!result && result.failure != metal::AllocationFailure::GrowthDeferred) return result;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void testLoadedDraftIsRequiredOnlyWithDraft() {
  model::LoadedModel loaded;
  model::Qwen3_5_4BWeights target;
  target.actualAllocatedBytes = 1;
  target.manifestFingerprintSha256 = "target";
  loaded.descriptor = model::makeModelDescriptor("target-only", target.layout, {}, {},
      model::TargetSource::Gguf, model::VisionSource::None);
  loaded.target = std::move(target);
  loaded.manifestFingerprintSha256 = "model";
  requireLoadedModel(loaded);
  loaded.draft.actualAllocatedBytes = 1;
  rejects([&] { requireLoadedModel(loaded); }, "incomplete allocation accounting",
          "target-only loader reported nonexistent draft weights");
}

// A small model whose weights load and whose geometry the runtime plans:
// four target and two draft layers of the 2048-wide hidden state a compiled
// draft attention takes, the draft reading a GDN layer and the attention
// layer, with Qwen3.8-27B's attention and GDN heads and a two-block vision
// tower.
struct PlannedModel final {
  model::Qwen3_8Layout target;
  model::DFlashDraftLayout draft;
  ops::VisionLayout vision;
};

PlannedModel plannedModel() {
  PlannedModel result;
  model::Qwen3_8Layout &target = result.target;
  target.layers = 4;
  target.hiddenSize = 2048;
  target.vocabularySize = 1024;
  target.intermediateSize = 1024;
  constexpr std::array<uint32_t, 2> draftReads{1, 3};
  result.draft = model::draftLayout(target.hiddenSize, target.vocabularySize, 2, 1024, 10'000'000.0F, draftReads);
  ops::VisionLayout &vision = result.vision;
  vision.depth = 2;
  vision.hiddenSize = 128;
  vision.patchDimension = 1536;
  vision.intermediateSize = 200;
  vision.paddedIntermediateSize = 256;
  vision.mergedHiddenSize = 512;
  vision.outputHiddenSize = target.hiddenSize;
  vision.heads = 2;
  vision.headDimension = 64;
  vision.positionGridSide = 4;
  return result;
}

// The target and draft checkpoints of `model` under root, which weight
// planning reads; its vision tower stays absent, so loading stops there.
void writeCheckpoints(const std::filesystem::path &root, const model::ModelDescriptor &model) {
  std::visit([&](const auto &target) { test::writeSyntheticCheckpoints(root, target, model.draft); },
             model.target);
}

RuntimeResourcesConfig budgetConfig(const char *metallibPath, const std::filesystem::path &root) {
  const PlannedModel planned = plannedModel();
  RuntimeResourcesConfig config;
  config.metallibPath = metallibPath;
  config.modelRoot = root;
  config.model = model::makeModelDescriptor("budget-test", planned.target, planned.draft, planned.vision,
                                            model::TargetSource::Safetensors, model::VisionSource::Safetensors);
  config.buildId = "budget-test";
  writeCheckpoints(root, config.model);
  return config;
}

// A model under root whose weights load, which nothing but the host
// refuses: `name` names its descriptor and build.
RuntimeResourcesConfig loadableConfig(const char *metallibPath, const std::filesystem::path &root,
                                      const std::string &name) {
  const PlannedModel planned = plannedModel();
  static_cast<void>(test::writeSyntheticModel(root, planned.target, planned.draft, planned.vision));
  RuntimeResourcesConfig config;
  config.metallibPath = metallibPath;
  config.modelRoot = root;
  config.model = model::makeModelDescriptor(name, planned.target, planned.draft, planned.vision,
                                            model::TargetSource::Safetensors,
                                            model::VisionSource::Safetensors);
  config.model.sourceIdentity = "synthetic";
  config.buildId = name;
  config.hostMemory = [] { return std::optional<HostMemory>(HostMemory{64 * kGiB, 64 * kGiB}); };
  return config;
}

// A local volume of 1000 GiB with `availableBytes` free, which leaves the
// automatic SSD cache room from 12 GiB on.
DiskVolume localVolume(uint64_t availableBytes) {
  return {"/Volumes/Test", availableBytes, 1000 * kGiB, true};
}

// A volume probe that reads `volume` for every directory.
std::function<std::optional<DiskVolume>(const std::filesystem::path &)>
reading(std::optional<DiskVolume> volume) {
  return [volume](const std::filesystem::path &) { return volume; };
}

// The routed experts of each layer of the MoE model below.
constexpr uint32_t kMoeExperts = 128;
constexpr uint32_t kMoeLayers = 8;

// A MoE model under root whose weights load: the planned model's draft and
// vision tower beside kMoeLayers layers of Qwen3.6-35B-A3B's geometry, each
// with kMoeExperts routed experts, eight a token. Its experts above their
// smallest store, a layer's, take more than the arenas of its prefill groups
// and the first 64K tokens of its KV.
RuntimeResourcesConfig moeConfig(const char *metallibPath, const std::filesystem::path &root) {
  const PlannedModel planned = plannedModel();
  model::Qwen3_6MoeLayout target;
  target.layers = kMoeLayers;
  target.vocabularySize = planned.target.vocabularySize;
  target.experts = kMoeExperts;
  static_cast<void>(test::writeSyntheticModel(root, target, planned.draft, planned.vision));
  RuntimeResourcesConfig config;
  config.metallibPath = metallibPath;
  config.modelRoot = root;
  config.model = model::makeModelDescriptor("moe-test", target, planned.draft, planned.vision,
                                            model::TargetSource::Safetensors, model::VisionSource::Safetensors);
  config.model.sourceIdentity = "synthetic";
  config.buildId = "moe-test";
  config.hostMemory = [] { return std::optional<HostMemory>(HostMemory{64 * kGiB, 64 * kGiB}); };
  return config;
}

// A Qwen3.8-Flash-Next model under root whose weights load, a block-sparse
// target whose KV spills: a GGUF of its family's tensors
// (gguf_fixtures::targetTensors) with its attention, GDN, indexer and PLE
// heads but a hidden state of 256 values, a vocabulary of 256 tokens, eight
// layers, every second one full attention, and four routed experts a layer,
// and a draft of one layer that reads four of them; no vision tower.
RuntimeResourcesConfig flashNextConfig(const char *metallibPath, const std::filesystem::path &root, bool mixed = false,
                                      uint32_t tableRows = gguf_fixtures::kPleTableRows, bool executable = false) {
  model::Qwen4ExpLayout target;
  target.layers = executable ? 4 : 8;
  target.hiddenSize = executable ? HYPER_CONNECTION_NORM_HIDDEN : 256;
  target.vocabularySize = 256;
  target.fullAttentionPeriod = 2;
  target.maskToken = 252;
  target.stopTokens = {250, 251};
  target.pleEosToken = 250;
  target.pleImageToken = 253;
  target.experts = executable ? 8 : 4;
  target.expertsPerToken = 2;
  target.expertIntermediateSize = 256;
  target.pleLayer = 0;
  gguf_fixtures::TensorTypes types = gguf_fixtures::qwen4ExpTypes();
  for (const char *name : {"hc_attn_down.weight", "hc_attn_up.weight", "hc_ffn_down.weight", "hc_ffn_up.weight",
                           "output_hc_down.weight", "output_hc_up.weight", "ffn_gate_exps.weight",
                           "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_shexp.weight",
                           "ffn_up_shexp.weight", "ffn_down_shexp.weight"})
    types[name] = gguf_fixtures::kQ8_0;
  if (mixed)
    for (const char *name : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"})
      types["blk.0." + std::string(name)] = gguf_fixtures::kQ4_K;
  // The PLE hash, its heads an equal share of the table's rows each.
  std::vector<test::gguf::Key> keys = gguf_fixtures::metadata(target);
  const std::string ple = std::string(model::Qwen4ExpLayout::ggufArchitecture) + ".ple.";
  const uint32_t heads = target.pleHeads(), headRows = tableRows / heads;
  std::vector<int32_t> offsets, rows(heads, int32_t(headRows));
  for (uint32_t head = 0; head < heads; ++head)
    offsets.push_back(int32_t(head * headRows));
  keys.push_back(test::gguf::int32ArrayKey(ple + "layer_multipliers", {1, 31, 961}));
  keys.push_back(test::gguf::int32ArrayKey(ple + "head_offsets", offsets));
  keys.push_back(test::gguf::int32ArrayKey(ple + "head_vocab_sizes", rows));
  keys.push_back(test::gguf::uint32Key(ple + "eos_token_id", target.pleEosToken));
  if (executable) {
    uint64_t payload = 0, maximumTensor = 0;
    for (auto tensor : gguf_fixtures::targetTensors(target, types, false)) {
      if (tensor.name == "per_layer_token_embd.weight") tensor.dims[1] = tableRows;
      uint64_t elements = 1;
      for (const auto dimension : tensor.dims) elements = checkedMultiply(elements, dimension, "fixture tensor");
      const auto &format = *model::ggmlTypeTraits(tensor.type);
      const uint64_t bytes = checkedMultiply(elements / format.blockElements, format.blockBytes, "fixture tensor");
      payload = checkedAdd(payload, bytes, "fixture payload");
      maximumTensor = std::max(maximumTensor, bytes);
    }
    require(payload < (512ULL << 20), "executable PLE fixture exceeds its pre-allocation payload bound");
    // Values are encoded directly in each tensor: no F32 reference or
    // conversion payload. Serialization holds the tensor vectors plus the
    // GGUF vector's old and new storage.
    const uint64_t serializationPeak = checkedAdd(checkedMultiply(payload, 4, "fixture peak"), 16ULL << 20, "fixture peak");
    std::cout << "PLE fixture target_payload_bytes=" << payload << " largest_tensor_bytes=" << maximumTensor
              << " float_reference_bytes=0 construction_peak_bound_bytes=" << serializationPeak
              << " layers=" << target.layers << " hidden=" << target.hiddenSize << " experts=" << target.experts
              << " qkv=" << target.packedGdnWidth << " full_qkv=" << target.packedFullWidth
              << " expert_width=" << target.expertIntermediateSize << " draft_layers=1 draft_reads=1,3\n";
  }
  std::filesystem::create_directories(root / "target");
  auto tensors = gguf_fixtures::targetTensors(target, types);
  if (executable) {
    for (auto &tensor : tensors) {
      if (tensor.type == gguf_fixtures::kQ8_0) {
        const auto &format = *model::ggmlTypeTraits(tensor.type);
        const uint64_t rowBytes = tensor.dims[0] / format.blockElements * format.blockBytes;
        for (uint64_t byte = 0; byte < tensor.data.size(); byte += format.blockBytes) {
          const uint64_t row = byte / rowBytes, block = byte % rowBytes / format.blockBytes;
          tensor.data[byte] = 0; tensor.data[byte + 1] = 0x18; // fp16 scale 1/512
          for (uint32_t value = 2; value < format.blockBytes; ++value)
            tensor.data[byte + value] = ((row >> ((block * format.blockElements + value - 2) % 19)) & 1) ? 1 : 255;
        }
      } else {
        require(tensor.type == model::ggml::kF32, "executable fixture expects Q8_0 or F32 source tensors");
        for (uint64_t element = 0; element < tensor.data.size() / sizeof(float); ++element) {
          // GGUF stores the already negative GDN -exp(A_log), not A_log.
          const float value = tensor.name.ends_with("ssm_a") ? -1.0F : gguf_fixtures::isNorm(tensor.name) ? 1.0F :
              float(int((element + element / tensor.dims[0]) % 11) - 5) / 512.0F;
          std::memcpy(tensor.data.data() + element * sizeof(float), &value, sizeof(value));
        }
      }
    }
  }
  if (tableRows != gguf_fixtures::kPleTableRows) {
    auto &table = gguf_fixtures::tensorNamed(tensors, "per_layer_token_embd.weight");
    const uint64_t rowBytes = table.data.size() / table.dims[1];
    table.dims[1] = tableRows;
    table.data.resize(rowBytes * tableRows);
    if (executable) {
      require(table.type == gguf_fixtures::kQ8_0, "executable PLE source must use finite Q8_0 blocks");
      const auto &format = *model::ggmlTypeTraits(table.type);
      for (uint64_t byte = 0; byte < table.data.size(); byte += format.blockBytes) {
        table.data[byte] = 0; table.data[byte + 1] = 0x24; // fp16 scale 1/64
        const uint64_t row = byte / rowBytes, block = byte % rowBytes / format.blockBytes;
        for (uint32_t value = 2; value < format.blockBytes; ++value)
          table.data[byte + value] = ((row >> ((block * format.blockElements + value - 2) % 19)) & 1) ? 1 : 255;
      }
    } else {
      for (uint64_t byte = 0; byte < table.data.size(); ++byte) table.data[byte] = uint8_t(byte * 7 + 11);
    }
  }
  test::writeFile(root / "target" / "model.gguf", test::gguf::file(keys, tensors));
  constexpr std::array<uint32_t, 4> draftReads{1, 3, 5, 7};
  const model::DFlashDraftLayout draft =
      model::draftLayout(target.hiddenSize, target.vocabularySize, 1, 512, 1'000'000.0F,
                         std::span(draftReads).first(executable ? 2 : 4));
  test::writeSyntheticShard(root / "draft" / "model.safetensors", test::draftTensors(draft));
  RuntimeResourcesConfig config;
  config.metallibPath = metallibPath;
  config.modelRoot = root;
  config.model = model::makeModelDescriptor("flash-next-test", target, draft,
                                            {.outputHiddenSize = target.hiddenSize}, model::TargetSource::Gguf,
                                            model::VisionSource::None);
  config.model.sourceIdentity = "synthetic";
  config.buildId = "flash-next-test";
  config.hostMemory = [] { return std::optional<HostMemory>(HostMemory{64 * kGiB, 64 * kGiB}); };
  return config;
}

// A bounded native MoE fixture, with its own target and draft source files.
RuntimeResourcesConfig nativeMoeConfig(const char *metallibPath, const std::filesystem::path &root,
                                     bool visionEnabled = false, bool compact = false) {
  model::Qwen3_6MoeLayout target;
  target.layers = 4; target.hiddenSize = 2048; target.vocabularySize = 256;
  if (compact) target.hiddenSize = 256;
  target.maskToken = 252; target.stopTokens = {250, 251};
  target.experts = 16; target.expertsPerToken = 4; target.expertIntermediateSize = 256;
  constexpr std::array<uint32_t, 2> reads{1, 3};
  const auto draft = model::draftLayout(target.hiddenSize, target.vocabularySize, 1, 1024, 1'000'000.0F, reads);
  test::writeSyntheticCheckpoints(root / "model", target, draft);
  RuntimeResourcesConfig config;
  config.modelRoot = root / "model";
  config.metallibPath = metallibPath;
  const auto vision = visionEnabled ? plannedModel().vision : ops::VisionLayout{.outputHiddenSize = target.hiddenSize};
  if (visionEnabled) test::writeSyntheticShard(root / "model/vision/model.safetensors", test::visionTensors(vision));
  config.model = model::makeModelDescriptor("local-native", target, draft, vision,
      model::TargetSource::Safetensors, visionEnabled ? model::VisionSource::Safetensors : model::VisionSource::None);
  config.model.sourceIdentity = "synthetic";
  config.buildId = "local-native";
  config.hostMemory = [] { return std::optional<HostMemory>(HostMemory{64 * kGiB, 64 * kGiB}); };
  config.maximumMemoryBytes = 4 * kGiB;
  config.idleReleaseSeconds = 1;
  return config;
}

void testLocalReleasedCurrentLayout(const char *metallibPath, bool withPle, bool manualPle = false) {
  const test::TemporaryDirectory root("sm-current-restore");
  auto config = withPle ? flashNextConfig(metallibPath, root.path() / "model", false, 96 * 4096, true) :
      nativeMoeConfig(metallibPath, root.path(), false, true);
  config.maximumMemoryBytes = 4 * kGiB;
  config.moeMemoryPressure = true;
  config.expertPleMemoryBytes = withPle ? kGiB : 512ULL << 20;
  config.expertPleAdaptive = withPle && !manualPle;
  if (config.expertPleAdaptive) config.memoryPressure = [] { return MemoryPressure::Warning; };
  if (withPle) {
    const auto metadata = model::streamableWeights(config.modelRoot, config.model,
        {.streamExperts = true, .streamTable = true}, config.ffnLoading, true, true);
    require(metadata.loadedBytes < (512ULL << 20), "synthetic PLE restore fixture exceeded its fixed-weight bound");
    if (manualPle) config.pleMemory = {StreamedSetting::Mode::Size, metadata.tableFloorBytes};
  }
  AutomaticStorageClock policyClock(config);
  RuntimeMetrics metrics;
  RuntimeBootstrapConfig start;
  start.resources = config;
  start.nativeLoop.metrics = &metrics;
  start.nativeLoop.engine.maxContext = 128;
  start.nativeLoop.engine.resourceWaitTimeoutMilliseconds = 4000;
  double now = 0;
  start.nativeLoop.monotonicMilliseconds = [&] { return now; };
  start.nativeLoop.unixMicros = [] { return uint64_t{1'000'000}; };
  std::vector<uint8_t> output;
  auto runtime = RuntimeBootstrap::start(start, [&](std::span<const uint8_t> bytes) {
    output.insert(output.end(), bytes.begin(), bytes.end());
  }, [] { return "{}"; });
  auto &resources = runtime->resources();
  auto *experts = resources.modelContext().model.expertStore();
  auto *ple = resources.modelContext().model.streamedTable();
  const auto range = resources.expertStorageSnapshot()->resizable;
  require(experts && range.chunks.size() == 1 && bool(ple) == withPle,
          "bounded restore fixture lost its intended public store configuration");
  const uint64_t floor = range.minimumBytes, chunk = range.chunks.front().allocationBytes;
  const uint64_t pleHeld = ple ? ple->committedBytes() : 0;
  const uint64_t automaticPleHeld = withPle && !manualPle ? pleHeld : 0;
  if (config.expertPleAdaptive) {
    const auto measured = runtime->modelRuntime().actualRuntimeMemory();
    require(measured.sharedPrefillActualAllocatedBytes && measured.sharedDecodeActualAllocatedBytes,
            "executable adaptive fixture omitted its actual warmup arenas");
    static_cast<void>(resources.rebalanceExpertPle(0, measured));
    const auto observed = resources.expertPleStatusJson();
    require(observed.find("\"reason\":\"host_growth_paused\"") == std::string::npos &&
                observed.find("\"target_ple_bytes\":" + std::to_string(pleHeld)) != std::string::npos &&
                observed.find("\"ple_maximum_bytes\":" + std::to_string(pleHeld)) != std::string::npos &&
                observed.find("\"runtime_audit\":{\"scope\":\"runtime_store_epoch\",\"valid\":true") != std::string::npos,
            "measured Warning local adaptive policy was paused or ignored the current floor quota: " + observed);
  }
  const auto checkPleRows = [&] {
    if (!ple) return;
    const std::array<uint32_t, 4> ids{1, 7, 18, 31};
    std::array<uint32_t, 4> slots{};
    const uint64_t rowBytes = ple->table().rowBytes;
    std::vector<uint8_t> rows(ids.size() * rowBytes);
    ple->stage(ids, rows, slots, model::SparseRowUse::Prefill);
    const auto &format = *model::ggmlTypeTraits(gguf_fixtures::kQ8_0);
    for (size_t row = 0; row < ids.size(); ++row)
      for (uint64_t column = 0; column < rowBytes; ++column) {
        const uint64_t within = column % format.blockBytes, block = column / format.blockBytes;
        const uint8_t expected = within == 0 ? 0 : within == 1 ? 0x24 :
            ((ids[row] >> ((block * format.blockElements + within - 2) % 19)) & 1) ? 1 : 255;
        require(rows[slots[row] * rowBytes + column] == expected,
                "PLE release/restore changed distinct finite source rows or their slot mapping");
      }
  };
  checkPleRows();
  require(static_cast<bool>(growAutomaticTo(resources, policyClock, false, floor + 3 * chunk,
      std::chrono::steady_clock::now() + std::chrono::seconds(3))), "old larger layout seed was refused");
  const auto originalPlan = resources.memoryPlan().toStatusJson();
  const auto generate = [&](uint64_t id) {
    protocol::RequestFrame frame;
    frame.requestId = id; frame.promptTokens = {1, 2, 3, 4}; frame.logicalMaxOutputTokens = 4;
    frame.cacheSalt = {id, 0}; frame.absoluteDeadlineUnixMicros = 601'000'000;
    frame.remainingDeadlineMicros = 600'000'000;
    require(runtime->nativeLoop().receive(protocol::peer::serialize(frame)), "current-layout generation frame failed");
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(12);
    while (runtime->nativeLoop().snapshot().completed < id) {
      for (const auto &event : protocol::peer::decodeEvents(output))
        if (const auto *error = std::get_if<protocol::ErrorEvent>(&event))
          throw std::runtime_error("current restore request failed: " + error->code + ": " + error->message);
      require(runtime->nativeLoop().engineHealthy() && std::chrono::steady_clock::now() < until,
              "current-layout generation exceeded its original deadline or stopped the engine");
      static_cast<void>(runtime->nativeLoop().tick());
      now += 1;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::vector<uint32_t> tokens;
    for (const auto &event : protocol::peer::decodeEvents(output))
      if (const auto *emitted = std::get_if<protocol::TokensEvent>(&event); emitted && emitted->requestId == id)
        tokens.insert(tokens.end(), emitted->tokens.begin(), emitted->tokens.end());
    require(!tokens.empty(), "current-layout fixture emitted no tokens");
    return tokens;
  };
  const auto reference = generate(1);
  require(runtime->nativeLoop().receive(protocol::peer::serialize(protocol::ReleaseRequestFrame{91})) &&
              runtime->nativeLoop().weightsSnapshot().released && !(!runtime->nativeLoop().weightsSnapshot().released && !runtime->nativeLoop().weightsSnapshot().restoring),
          "public release kept model Ready or refused its idle weights");
  const uint64_t retained = experts->allocatedBytes();
  auto &governor = resources.memoryGovernor();
  governor.setAutomaticStorageQuota(floor + chunk + automaticPleHeld, governor.automaticStorageMaximumBytes(),
      [&] { return experts->allocatedBytes() + (withPle && !manualPle ? ple->committedBytes() : 0); });
  bool sawExpertRestore = false;
  governor.setAllocationDriver([&](std::string_view purpose, uint64_t bytes, const std::function<void()> &allocate) {
    if (purpose == "restore_weights" && !resources.modelContext().model.images->released()) {
      // The accepted request still waits for its weights: T7 assigns it
      // the current floor, even though the local cap retained one extra
      // chunk. The active request may receive that chunk only afterwards.
      require(experts->released() && experts->allocatedBytes() == retained &&
                  experts->stats().budgetBytes == floor && bytes == floor - retained &&
                  resources.expertPleStatusJson().find("\"expert_restore_target_bytes\":" + std::to_string(floor)) != std::string::npos,
              "first expert payload preceded current target selection or charged retained helpers again");
      sawExpertRestore = true;
    }
    allocate();
  });
  const auto changed = generate(2);
  checkPleRows();
  governor.setAllocationDriver({});
  require(sawExpertRestore && experts->allocatedBytes() == floor && changed == reference &&
              (!ple || ple->committedBytes() == pleHeld) && resources.memoryPlan().toStatusJson() == originalPlan &&
              runtime->nativeLoop().weightsSnapshot().restores == 1 && (!runtime->nativeLoop().weightsSnapshot().released && !runtime->nativeLoop().weightsSnapshot().restoring),
          "smaller current restore changed numerical output, retained PLE, plan or lifecycle state");
  if (manualPle)
    require(resources.expertPleStatusJson().find("\"ple_budget_bytes\":" + std::to_string(config.pleMemory.bytes)) != std::string::npos &&
                resources.expertPleStatusJson().find("\"automatic_ple\":false") != std::string::npos,
            "manual PLE cap changed or was deducted from automatic experts a second time");
  require(static_cast<bool>(growAutomaticTo(resources, policyClock, false, floor + chunk,
      std::chrono::steady_clock::now() + std::chrono::seconds(3))) && experts->allocatedBytes() == floor + chunk,
          "selected current target stayed permanently at its execution floor");
}

void testImagePayloadIsPreparedByOwner(const char *metallibPath) {
  const test::TemporaryDirectory root("sm-writer");
  auto config = nativeMoeConfig(metallibPath, root.path(), true);
  config.moeMemoryPressure = true;
  config.allocationDriver = [&](std::string_view purpose, uint64_t, const std::function<void()> &allocate) {
    if (purpose != "startup_weights") { allocate(); return; }
    backgroundHostGrowths = 0;
    observedHostOwner = std::this_thread::get_id();
    observeLargeHostGrowth.store(true, std::memory_order_release);
    try { allocate(); } catch (...) { observeLargeHostGrowth = false; throw; }
    observeLargeHostGrowth = false;
  };
  auto resources = RuntimeResources::create(config, 128);
  require(backgroundHostGrowths == 0, "image reader workers allocated new large payloads without an owner gate: count=" +
          std::to_string(backgroundHostGrowths.load()));
  const auto host = resources->backend().hostMemoryStats();
  require(host.temporaryBytes == 0 && host.fixedBytes == resources->modelContext().model.hostHeldBytes() &&
              resources->memoryGovernor().actualBytes() == resources->backend().memoryStats().allocatedBytes + host.fixedBytes,
          "completed load retained temporary payload or counted fixed host capacity twice");
}

void testLocalReleaseInvalidatesModelReadiness(const char *metallibPath) {
  const test::TemporaryDirectory root("sm-release-ready");
  auto config = nativeMoeConfig(metallibPath, root.path());
  config.maximumMemoryBytes = 4 * kGiB;
  config.moeMemoryPressure = true;
  config.idleReleaseSeconds = std::numeric_limits<double>::infinity();
  RuntimeMetrics metrics;
  RuntimeBootstrapConfig start;
  start.resources = config;
  start.nativeLoop.metrics = &metrics;
  start.nativeLoop.engine.maxContext = 2048;
  RuntimeBootstrap *published = nullptr;
  std::vector<uint8_t> output;
  auto runtime = RuntimeBootstrap::start(start, [&](std::span<const uint8_t> bytes) {
    output.insert(output.end(), bytes.begin(), bytes.end());
  }, [&] { return published->statusJson(metrics.snapshot(), {}); });
  published = runtime.get();
  require((!runtime->nativeLoop().weightsSnapshot().released && !runtime->nativeLoop().weightsSnapshot().restoring),
          "completed native warmup did not publish model readiness");
  require(runtime->nativeLoop().receive(protocol::peer::serialize(protocol::ReleaseRequestFrame{91})),
          "native release request was refused");
  bool answered = false;
  for (const auto &event : protocol::peer::decodeEvents(output))
    if (const auto *status = std::get_if<protocol::StatusJsonEvent>(&event); status && status->correlationId == 91) {
      answered = true;
      require(status->json.find("\"released\":true") != std::string::npos &&
                  status->json.find("\"model_ready\":false") != std::string::npos,
              "released native weights still reported model_ready: " + status->json);
    }
  require(answered && runtime->nativeLoop().weightsSnapshot().released &&
              !(!runtime->nativeLoop().weightsSnapshot().released && !runtime->nativeLoop().weightsSnapshot().restoring),
          "native release did not invalidate the model lifecycle before its answer");
}

void testNoPlePressureReturnsExpertChunks(const char *metallibPath) {
  for (const bool critical : {false, true}) {
    const test::TemporaryDirectory root("sm-shrink");
    auto config = nativeMoeConfig(metallibPath, root.path());
    std::atomic<MemoryPressure> pressure{MemoryPressure::Normal};
    std::atomic<uint64_t> spare{64 * kGiB};
    std::atomic<bool> measured{true};
    std::atomic<uint64_t> hostSamples{0};
    config.moeMemoryPressure = true;
    config.hostMemoryReserveBytes = 2 * kGiB;
      config.idleReleaseSeconds = std::numeric_limits<double>::infinity();
    config.memoryPressure = [&] { return pressure.load(); };
    config.hostMemory = [&]() -> std::optional<HostMemory> {
      ++hostSamples;
      if (!measured) return std::nullopt;
      return HostMemory{64 * kGiB, spare.load()};
    };
    AutomaticStorageClock policyClock(config);
      RuntimeMetrics metrics;
    RuntimeBootstrapConfig start;
    start.resources = config;
    start.nativeLoop.metrics = &metrics;
    start.nativeLoop.engine.maxContext = 2048;
    start.nativeLoop.engine.resourceWaitTimeoutMilliseconds = 4000;
    double now = 0;
    start.nativeLoop.monotonicMilliseconds = [&] { return now; };
    start.nativeLoop.unixMicros = [] { return uint64_t{1'000'000}; };
    std::vector<uint8_t> output;
    auto runtime = RuntimeBootstrap::start(start, [&](std::span<const uint8_t> bytes) {
      output.insert(output.end(), bytes.begin(), bytes.end());
    }, [] { return "{}"; });
    auto &resources = runtime->resources();
    auto *store = resources.modelContext().model.expertStore();
    const auto range = resources.expertStorageSnapshot()->resizable;
    require(store && range.chunks.size() == 1 && !resources.streamedTableStats(),
            "pressure fixture requires a single-format MoE without PLE");
    const uint64_t floor = range.minimumBytes, chunk = range.chunks.front().allocationBytes;
    require(store->allocatedBytes() == floor, "local start did not retain the admitted expert floor");
    const auto growUntil = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    const auto admitted = growAutomaticTo(resources, policyClock, false, floor + chunk, growUntil);
    require(static_cast<bool>(admitted), "optional chunk precondition did not obtain healthy local admission: " + admitted.detail);
    require(store->allocatedBytes() == floor + chunk, "optional precondition did not allocate exactly one real chunk");
    const auto generate = [&](uint64_t id) {
      protocol::RequestFrame request;
      request.requestId = id; request.promptTokens = {1, 2, 3, 4}; request.logicalMaxOutputTokens = 4;
      request.cacheSalt = {id, 0};
      request.absoluteDeadlineUnixMicros = 601'000'000; request.remainingDeadlineMicros = 600'000'000;
      require(runtime->nativeLoop().receive(protocol::peer::serialize(request)), "shrink generation refused its frame");
      const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(12);
      while (runtime->nativeLoop().snapshot().completed < id) {
        require(runtime->nativeLoop().engineHealthy() && std::chrono::steady_clock::now() < until,
                "generation did not finish around expert shrink");
        for (const auto &event : protocol::peer::decodeEvents(output))
          if (const auto *error = std::get_if<protocol::ErrorEvent>(&event))
            throw std::runtime_error("shrink generation: " + error->code + ": " + error->message);
        static_cast<void>(runtime->nativeLoop().tick());
        if (!runtime->nativeLoop().commandInFlight())
          static_cast<void>(runtime->controlPass(pressure));
        now += 1;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      std::vector<uint32_t> tokens;
      for (const auto &event : protocol::peer::decodeEvents(output))
        if (const auto *emitted = std::get_if<protocol::TokensEvent>(&event); emitted && emitted->requestId == id)
          tokens.insert(tokens.end(), emitted->tokens.begin(), emitted->tokens.end());
      require(!tokens.empty(), "shrink generation emitted no tokens");
      return tokens;
    };
    const auto reference = generate(1);
    const auto originalPlan = resources.memoryPlan().breakdown().toStatusJson();
    const auto capacity = resources.expertStorageSnapshot()->capacityWeightBytes;

    pressure = MemoryPressure::Warning;
    spare = 3 * kGiB + chunk;
    for (unsigned pass = 0; pass < 3; ++pass) {
      now += 1000;
      static_cast<void>(runtime->controlPass(pressure));
      const auto allocated = store->allocatedBytes();
      const auto quota = store->stats().budgetBytes;
      if (allocated != floor + chunk || quota != floor + chunk)
        require(false, "repeated Warning with no reserve deficit discarded expert capacity: pass=" + std::to_string(pass) +
              " floor=" + std::to_string(floor) + " chunk=" + std::to_string(chunk) +
              " actual=" + std::to_string(allocated) + " quota=" + std::to_string(quota) +
              " released=" + std::to_string(resources.memoryGovernor().snapshot().automaticStorageReleasedBytes) +
              " withheld=" + std::to_string(resources.memoryGovernor().snapshot().automaticStorageWithheldBytes) +
              " group=" + runtime->statusJson({}, {}));
    }
    measured = false;
    now += 1000;
    static_cast<void>(runtime->controlPass(pressure));
    require(store->allocatedBytes() == floor + chunk, "missing telemetry was treated as freeable expert capacity");
    if (critical) pressure = MemoryPressure::Critical;
    else { measured = true; spare = 3 * kGiB - chunk / 2; }
    now += 1000;
    const auto beforeShrink = store->stats();
    const auto beforeBackend = resources.backend().memoryStats();
    const auto beforeActual = resources.actualMemoryReport(runtime->modelRuntime().actualRuntimeMemory(), false);
    static_cast<void>(runtime->controlPass(pressure));
    const auto shrunk = resources.expertStorageSnapshot();
    const auto governor = resources.memoryGovernor().snapshot();
    const auto afterBackend = resources.backend().memoryStats();
    const auto afterActual = resources.actualMemoryReport(runtime->modelRuntime().actualRuntimeMemory(), false);
    const auto currentPlan = resources.memoryPlan().breakdown().toStatusJson();
    // Quota is the allowed byte ceiling; actual allocation rounds to real
    // chunks. A half-chunk Warning deficit removes the whole optional chunk.
    const uint64_t expectedQuota = critical ? floor : floor + chunk / 2;
    const bool auxiliaryUnchanged =
        beforeActual.draftWeightsBytes == afterActual.draftWeightsBytes &&
        beforeActual.visionWeightsBytes == afterActual.visionWeightsBytes &&
        beforeActual.sharedPrefillBytes == afterActual.sharedPrefillBytes &&
        beforeActual.sharedDecodeBytes == afterActual.sharedDecodeBytes &&
        beforeActual.kvBlockCacheBytes == afterActual.kvBlockCacheBytes &&
        beforeActual.kvHistoryStagingBytes == afterActual.kvHistoryStagingBytes &&
        beforeActual.stateStagingBytes == afterActual.stateStagingBytes &&
        beforeActual.aneFfnBytes == afterActual.aneFfnBytes;
    const auto known = [](const std::optional<uint64_t> &value) {
      return value ? std::to_string(*value) : std::string("unknown");
    };
    const std::string shrinkDiagnostic =
        " critical=" + std::to_string(critical) + " command=" + std::to_string(runtime->nativeLoop().commandInFlight()) +
        " chunk=" + std::to_string(chunk) + " geometry_floor=" + std::to_string(floor) +
        " before_allocated=" + std::to_string(beforeShrink.allocatedBytes) +
        " before_quota=" + std::to_string(beforeShrink.budgetBytes) +
        " before_floor=" + std::to_string(beforeShrink.minimumBytes) +
        " after_store_allocated=" + std::to_string(store->allocatedBytes()) +
        " after_snapshot_allocated=" + known(shrunk->allocatedBytes) +
        " after_quota=" + known(shrunk->budgetBytes) + " after_floor=" + known(shrunk->currentFloorBytes) +
        " before_capacity=" + known(capacity) + " after_capacity=" + known(shrunk->capacityWeightBytes) +
        " backend_before=" + std::to_string(beforeBackend.allocatedBytes) +
        " backend_after=" + std::to_string(afterBackend.allocatedBytes) +
        " kv_before=" + std::to_string(beforeActual.kvAllocatedBytes) +
        " kv_after=" + std::to_string(afterActual.kvAllocatedBytes) +
        " state_before=" + std::to_string(beforeActual.stateAllocatedBytes) +
        " state_after=" + std::to_string(afterActual.stateAllocatedBytes) +
        " auxiliary_unchanged=" + std::to_string(auxiliaryUnchanged) +
        " released=" + std::to_string(governor.automaticStorageReleasedBytes) +
        " withheld=" + std::to_string(governor.automaticStorageWithheldBytes) +
        " measured=" + std::to_string(governor.hostMeasurementValid) +
        " local_spare=" + std::to_string(governor.hostSpareBytes) +
        " local_base=" + std::to_string(governor.hostSpareReserveBytes) +
        " system_pressure=" + memoryPressureName(governor.systemPressure) +
        " effective_pressure=" + memoryPressureName(governor.pressure) +
        " reason=" + std::string(governor.automaticStorageReason) +
        " plan_unchanged=" + std::to_string(currentPlan == originalPlan) +
        (currentPlan == originalPlan ? "" : " original_plan=" + originalPlan + " current_plan=" + currentPlan);
    require(store->allocatedBytes() == floor && shrunk->allocatedBytes == floor &&
                shrunk->budgetBytes == expectedQuota &&
                shrunk->currentFloorBytes == floor && shrunk->capacityWeightBytes < capacity &&
                governor.automaticStorageReleasedBytes == chunk && governor.automaticStorageWithheldBytes == chunk &&
                governor.automaticStorageReason == "minimum_insufficient" &&
                resources.memoryPlan().breakdown().toStatusJson() == originalPlan,
            "pressure did not return real expert chunks at the geometry floor while retaining KV/state limits" + shrinkDiagnostic);
    require(auxiliaryUnchanged && beforeActual.kvAllocatedBytes >= afterActual.kvAllocatedBytes &&
                beforeActual.stateAllocatedBytes >= afterActual.stateAllocatedBytes &&
                beforeActual.targetWeightsBytes - afterActual.targetWeightsBytes == chunk &&
                beforeBackend.allocatedBytes >= afterBackend.allocatedBytes &&
                beforeBackend.allocatedBytes - afterBackend.allocatedBytes == chunk +
                    (beforeActual.kvAllocatedBytes - afterActual.kvAllocatedBytes) +
                    (beforeActual.stateAllocatedBytes - afterActual.stateAllocatedBytes) &&
                (critical || (beforeActual.kvAllocatedBytes == afterActual.kvAllocatedBytes &&
                              beforeActual.stateAllocatedBytes == afterActual.stateAllocatedBytes)),
            "backend release did not equal exact expert plus independent KV/state release" + shrinkDiagnostic);
    const auto status = runtime->statusJson({}, {});
    require(status.find("\"automatic_storage_released_bytes\":" + std::to_string(chunk)) != std::string::npos &&
                status.find("\"automatic_storage_reason\":") != std::string::npos,
            "public status omitted physical release or pressure decision reason");
    measured = true; spare = 64 * kGiB; pressure = MemoryPressure::Normal;
    now += 1000;
    static_cast<void>(runtime->controlPass(pressure));
    require(generate(2) == reference && runtime->nativeLoop().engineHealthy(),
            "expert shrink changed deterministic tokens or invalidated bound expert mappings");
  }
}

void testCriticalStopsHostWorkspaceBeforeAllocation(const char *metallibPath) {
  const test::TemporaryDirectory root("sm-critical");
  std::atomic<MemoryPressure> pressure{MemoryPressure::Normal};
  auto config = nativeMoeConfig(metallibPath, root.path());
  config.moeMemoryPressure = true;
  config.memoryPressure = [&] { return pressure.load(); };
  config.allocationDriver = [&](std::string_view purpose, uint64_t, const std::function<void()> &allocate) {
    if (purpose != "startup_weights") { allocate(); return; }
    pressure = MemoryPressure::Critical;
    largeHostGrowths = 0; largeHostBytes = 0; firstLargeHostBytes = 0;
    observedHostOwner = std::this_thread::get_id();
    observeLargeHostGrowth.store(true, std::memory_order_release);
    try { allocate(); } catch (...) { observeLargeHostGrowth = false; throw; }
    observeLargeHostGrowth = false;
  };
  try { auto resources = RuntimeResources::create(config, 128); throw std::runtime_error("Critical admitted loading"); }
  catch (const RuntimeResourcesError &error) {
    require(error.failure() == RuntimeResourceFailure::HostCapacity,
            std::string("Critical lost its local refusal cause: ") + error.what());
  }
  require(largeHostGrowths == 0, "Critical allowed host payload growth after local begin: count=" +
      std::to_string(largeHostGrowths.load()) + " first_bytes=" + std::to_string(firstLargeHostBytes.load()) +
      " payload_bytes=" + std::to_string(largeHostBytes.load()));

}

void testLocalPleChunkCriticalAndActual(const char *metallibPath) {
  const test::TemporaryDirectory root("sm-ple");
  std::atomic<MemoryPressure> pressure{MemoryPressure::Normal};
  auto config = flashNextConfig(metallibPath, root.path() / "model", false, 96 * 4096);
  config.maximumMemoryBytes = 4 * kGiB;
  config.moeMemoryPressure = true;
  config.expertPleAdaptive = true; config.expertPleMemoryBytes = kGiB;
  config.memoryPressure = [&] { return pressure.load(); };
  bool injectCritical = true;
  config.allocationDriver = [&](std::string_view purpose, uint64_t, const std::function<void()> &allocate) {
    if (purpose == "ple_optional" && injectCritical) pressure = MemoryPressure::Critical;
    allocate();
  };
  AutomaticStorageClock policyClock(config);
  auto resources = RuntimeResources::create(config, 128);
  resources->memoryGovernor().setServing(false);
  auto *ple = resources->modelContext().model.streamedTable();
  require(ple != nullptr, "valid PLE storage fixture did not stream its table");
  const uint64_t beforeCommitted = ple->committedBytes(), beforeHost = ple->hostAllocatedBytes();
  const uint64_t beforeFixed = resources->backend().hostMemoryStats().fixedBytes;
  const auto checkRows = [&] {
    const std::array<uint32_t, 4> ids{1, 7, 1, 9};
    std::array<uint32_t, 4> slots{};
    const uint32_t rowBytes = ple->table().rowBytes;
    std::vector<uint8_t> rows(uint64_t{ids.size()} * rowBytes);
    ple->stage(ids, rows, slots, model::SparseRowUse::Prefill);
    for (size_t index = 0; index < ids.size(); ++index)
      for (uint32_t byte = 0; byte < rowBytes; ++byte)
        require(rows[uint64_t{slots[index]} * rowBytes + byte] == uint8_t((uint64_t{ids[index]} * rowBytes + byte) * 7 + 11),
                "preallocated PLE working memory changed source rows");
  };
  checkRows();
  ple->prefetch(std::array<uint32_t, 3>{7, 9, 11}); ple->drainPrefetch(); checkRows();
  const uint64_t growth = uint64_t{4096} * (ple->table().rowBytes + 13);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
  const auto result = growAutomaticTo(*resources, policyClock, true, beforeCommitted + growth, deadline);
  require(result.failure == metal::AllocationFailure::HostPressure && ple->committedBytes() == beforeCommitted &&
              ple->hostAllocatedBytes() == beforeHost && resources->backend().hostMemoryStats().fixedBytes == beforeFixed,
          std::string("Critical failed to veto actual PLE chunk growth: ") + metal::allocationFailureName(result.failure) +
          " detail=" + result.detail +
          " committed_before=" + std::to_string(beforeCommitted) + " committed_after=" + std::to_string(ple->committedBytes()) +
          " host_before=" + std::to_string(beforeHost) + " host_after=" + std::to_string(ple->hostAllocatedBytes()));
  pressure = MemoryPressure::Normal; injectCritical = false;
  checkRows();
  const auto host = resources->backend().hostMemoryStats();
  require(host.temporaryBytes == 0 && host.fixedBytes == resources->modelContext().model.hostHeldBytes() &&
              resources->memoryGovernor().actualBytes() == resources->backend().memoryStats().allocatedBytes + host.fixedBytes,
          "PLE cache/working payload was lost or charged twice after Critical rollback");
  resources.reset();
}

void testLocalPlePressureTotalStorageOnly(const char *metallibPath, bool verifyStatus = false,
                                         bool adaptive = true, bool warning = false) {
  const test::TemporaryDirectory root("sm-ple-pressure");
  std::atomic<MemoryPressure> pressure{MemoryPressure::Normal};
  auto config = flashNextConfig(metallibPath, root.path() / "model", false, 96 * 4096);
  config.maximumMemoryBytes = 4 * kGiB;
  config.moeMemoryPressure = true;
  config.expertPleAdaptive = adaptive;
  config.expertPleMemoryBytes = kGiB;
  config.memoryPressure = [&] { return pressure.load(); };
  std::atomic<uint64_t> spare{64 * kGiB};
  std::atomic<bool> measured{true};
  if (warning) {
    config.hostMemoryReserveBytes = 2 * kGiB;
    config.hostMemory = [&]() -> std::optional<HostMemory> {
      return measured ? std::optional(HostMemory{64 * kGiB, spare.load()}) : std::nullopt;
    };
  }
  AutomaticStorageClock policyClock(config);
  auto resources = RuntimeResources::create(config, 128);
  auto &governor = resources->memoryGovernor();
  governor.setServing(false);
  auto *experts = resources->modelContext().model.expertStore();
  auto *ple = resources->modelContext().model.streamedTable();
  require(experts && ple, "local PLE pressure fixture did not stream both stores");
  if (adaptive) {
    require(!resources->rebalanceExpertPle(0, {}),
            "storage-only adaptive fixture applied a transfer without runtime measurements");
    const auto observed = resources->expertPleStatusJson();
    require(observed.find("\"reason\":\"host_growth_paused\"") != std::string::npos &&
                observed.find("\"adjustments\":0") != std::string::npos &&
                observed.find("\"runtime_audit\":{\"scope\":\"runtime_store_epoch\",\"valid\":false,\"error\":\"missing_measurement\"") != std::string::npos,
            "storage-only adaptive fixture accepted an incomplete runtime measurement: " + observed);
  }
  const uint64_t eFloor = experts->allocatedBytes(), pFloor = ple->committedBytes();
  const uint64_t eChunk = resources->expertStorageSnapshot()->resizable.chunks.front().allocationBytes;
  const uint64_t pChunk = 4096 * (ple->table().rowBytes + 13);
  const auto grow = [&](bool table, uint64_t target) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    const auto result = growAutomaticTo(*resources, policyClock, table, target, until);
    require(static_cast<bool>(result), "PLE total fixture did not receive healthy local optional admission: " + result.detail);
  };
  grow(false, eFloor + eChunk);
  grow(true, pFloor + pChunk);
  require(experts->allocatedBytes() == eFloor + eChunk && ple->committedBytes() == pFloor + pChunk,
          "local PLE fixture did not allocate exactly one optional chunk per store");
  const std::vector<uint32_t> ids{4, 17};
  std::vector<uint8_t> rows(ids.size() * ple->table().rowBytes);
  std::vector<uint32_t> slots(ids.size());
  ple->stage(ids, rows, slots, model::SparseRowUse::Decode);
  const auto expected = rows;
  ple->prefetch(ids);
  if (warning) {
    pressure = MemoryPressure::Warning;
    governor.setPressure(pressure);
    spare = 3 * kGiB + eChunk + pChunk;
    for (unsigned pass = 0; pass < 3; ++pass) {
      governor.controlAutomaticStorage();
      require(experts->allocatedBytes() == eFloor + eChunk && ple->committedBytes() == pFloor + pChunk &&
                  governor.snapshot().automaticStorageReleasedBytes == 0,
              "combined Warning without a reserve deficit discarded automatic storage");
    }
    measured = false;
    governor.controlAutomaticStorage();
    require(experts->allocatedBytes() == eFloor + eChunk && ple->committedBytes() == pFloor + pChunk,
            "missing combined-store telemetry was treated as a request to donate");
    measured = true;
    spare = 3 * kGiB - (eChunk + pChunk) / 2;
  }
  const auto beforeMetal = resources->backend().memoryStats().allocatedBytes;
  const auto beforeHost = resources->backend().hostMemoryStats().fixedBytes;
  const auto planBefore = resources->memoryPlan().toStatusJson();
  pressure = warning ? MemoryPressure::Warning : MemoryPressure::Critical;
  governor.setPressure(pressure);
  governor.controlAutomaticStorage();
  require(experts->allocatedBytes() == eFloor && ple->committedBytes() == pFloor &&
              beforeMetal - resources->backend().memoryStats().allocatedBytes == eChunk &&
              beforeHost - resources->backend().hostMemoryStats().fixedBytes == pChunk &&
              governor.snapshot().automaticStorageReleasedBytes == eChunk + pChunk &&
              governor.snapshot().automaticStorageWithheldBytes == eChunk + pChunk &&
              resources->memoryPlan().toStatusJson() == planBefore,
          "combined expert/PLE pressure did not donate both real storage chunks while preserving the dynamic plan");
  ple->stage(ids, rows, slots, model::SparseRowUse::Decode);
  require(rows == expected && !ple->stats().poisoned, "combined pressure changed PLE source rows or poisoned retained state");
  if (verifyStatus) {
    // A half-total deficit assigns half of each optional chunk. The actual
    // stores round down independently and release both complete chunks.
    const uint64_t eQuota = eFloor + (warning ? eChunk / 2 : 0);
    const uint64_t pQuota = pFloor + (warning ? pChunk / 2 : 0);
    const auto status = resources->expertPleStatusJson();
    require(status.find("\"pressure_managed\":true") != std::string::npos &&
                status.find("\"current_quota_bytes\":" + std::to_string(eQuota + pQuota)) != std::string::npos &&
                status.find("\"expert_budget_bytes\":" + std::to_string(eQuota)) != std::string::npos &&
                status.find("\"ple_budget_bytes\":" + std::to_string(pQuota)) != std::string::npos &&
                status.find("\"automatic_actual_bytes\":" + std::to_string(eFloor + pFloor)) != std::string::npos &&
                status.find("\"receiving\":\"paced_grant_required\"") != std::string::npos,
            "combined pressure status omitted current category quotas, actual total or closed receiver: " + status);
  }
}

void testFixedPlePressureStorageOnly(const char *metallibPath, bool resident) {
  const test::TemporaryDirectory root("sm-fixed-ple-pressure");
  std::atomic<MemoryPressure> pressure{MemoryPressure::Normal};
  auto config = flashNextConfig(metallibPath, root.path() / "model", false, 96 * 4096);
  config.maximumMemoryBytes = 4 * kGiB;
  config.moeMemoryPressure = true;
  config.expertPleMemoryBytes = kGiB;
  const uint64_t manualLimit = (48ULL << 20) + 123;
  config.pleMemory = resident ? StreamedSetting{StreamedSetting::Mode::Off, 0} :
      StreamedSetting{StreamedSetting::Mode::Size, manualLimit};
  config.memoryPressure = [&] { return pressure.load(); };
  const auto metadata = model::streamableWeights(config.modelRoot, config.model,
      {.streamExperts = true, .streamTable = true}, config.ffnLoading, true, true);
  AutomaticStorageClock policyClock(config);
  auto resources = RuntimeResources::create(config, 128);
  auto &governor = resources->memoryGovernor();
  governor.setServing(false);
  auto *experts = resources->modelContext().model.expertStore();
  auto *ple = resources->modelContext().model.streamedTable();
  require(experts && bool(ple) != resident, "fixed PLE fixture changed the requested residency mode");
  const uint64_t floor = experts->allocatedBytes();
  const uint64_t chunk = resources->expertStorageSnapshot()->resizable.chunks.front().allocationBytes;
  const uint64_t fixedPle = ple ? ple->committedBytes() : metadata.tableBytes;
  const uint64_t pleQuota = resident ? metadata.tableBytes : manualLimit;
  const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  const auto admitted = growAutomaticTo(*resources, policyClock, false, floor + chunk, until);
  require(static_cast<bool>(admitted), "fixed PLE fixture did not obtain optional expert admission: " + admitted.detail);
  const auto plan = resources->memoryPlan().toStatusJson();
  const auto beforeMetal = resources->backend().memoryStats().allocatedBytes;
  const auto beforeHost = resources->backend().hostMemoryStats().fixedBytes;
  pressure = MemoryPressure::Critical;
  governor.setPressure(pressure);
  governor.controlAutomaticStorage();
  resources->publishMemoryAccounting();
  const auto status = resources->expertPleStatusJson();
  require(experts->allocatedBytes() == floor && (!ple || ple->committedBytes() == fixedPle) &&
              beforeMetal - resources->backend().memoryStats().allocatedBytes == chunk &&
              resources->backend().hostMemoryStats().fixedBytes == beforeHost &&
              governor.snapshot().automaticStorageReleasedBytes == chunk &&
              governor.snapshot().automaticStorageWithheldBytes == chunk &&
              resources->memoryPlan().toStatusJson() == plan &&
              status.find("\"automatic_ple\":false") != std::string::npos &&
              status.find("\"current_quota_bytes\":" + std::to_string(floor)) != std::string::npos &&
              status.find("\"automatic_actual_bytes\":" + std::to_string(floor)) != std::string::npos &&
              status.find("\"ple_budget_bytes\":" + std::to_string(pleQuota)) != std::string::npos &&
              status.find("\"ple_fixed_bytes\":" + std::to_string(fixedPle)) != std::string::npos &&
              status.find("\"ple_resident_bytes\":" + std::to_string(resident ? fixedPle : 0)) != std::string::npos,
          "fixed PLE changed size, was charged twice, or lost its logical cap/actual category: " + status);
  require(resources->memoryGovernor().actualBytes() ==
              resources->backend().memoryStats().allocatedBytes + resources->backend().hostMemoryStats().fixedBytes,
          "fixed PLE storage-only accounting omitted/duplicated actual payload or fabricated Ready");
}

void testPlePressureDisabledStorageOnly(const char *metallibPath) {
  const test::TemporaryDirectory root("sm-ple-pressure-off");
  auto config = flashNextConfig(metallibPath, root.path() / "model", false, 96 * 4096);
  config.maximumMemoryBytes = 4 * kGiB;
  config.expertPleMemoryBytes = kGiB;
  config.pleMemory = {StreamedSetting::Mode::Size, 48ULL << 20};
  auto resources = RuntimeResources::create(config, 128);
  auto *ple = resources->modelContext().model.streamedTable();
  require(ple != nullptr, "disabled pressure fixture lost its fixed streamed PLE cache");
  const auto beforeMetal = resources->backend().memoryStats().allocatedBytes;
  const auto beforeHost = resources->backend().hostMemoryStats().fixedBytes;
  const auto beforePle = ple->committedBytes();
  const auto plan = resources->memoryPlan().toStatusJson();
  resources->memoryGovernor().setPressure(MemoryPressure::Critical);
  resources->memoryGovernor().controlAutomaticStorage();
  require(resources->backend().memoryStats().allocatedBytes == beforeMetal &&
              resources->backend().hostMemoryStats().fixedBytes == beforeHost && ple->committedBytes() == beforePle &&
              resources->memoryPlan().toStatusJson() == plan &&
              resources->memoryGovernor().automaticStorageMaximumBytes() == 0 &&
              resources->memoryGovernor().snapshot().automaticStorageReleasedBytes == 0 &&
              !resources->memoryGovernor().snapshot().pressureControlEnabled &&
              resources->expertPleStatusJson().find("\"pressure_managed\":true") == std::string::npos,
          "default-off PLE acquired automatic pressure storage behavior");
}

void testPartialPlePressureStorageOnly(const char *metallibPath) {
  const test::TemporaryDirectory root("sm-ple-pressure-partial");
  std::atomic<MemoryPressure> pressure{MemoryPressure::Normal};
  std::atomic<bool> measured{true};
  std::atomic<uint64_t> spare{64 * kGiB};
  auto config = flashNextConfig(metallibPath, root.path() / "model", true, 96 * 4096);
  config.maximumMemoryBytes = 4 * kGiB;
  config.moeMemoryPressure = true;
  config.expertPleMemoryBytes = kGiB;
  config.hostMemoryReserveBytes = 2 * kGiB;
  config.memoryPressure = [&] { return pressure.load(); };
  config.hostMemory = [&]() -> std::optional<HostMemory> {
    return measured ? std::optional(HostMemory{64 * kGiB, spare.load()}) : std::nullopt;
  };
  AutomaticStorageClock policyClock(config);
  auto resources = RuntimeResources::create(config, 128);
  auto &governor = resources->memoryGovernor();
  governor.setServing(false);
  auto *experts = resources->modelContext().model.expertStore();
  auto *ple = resources->modelContext().model.streamedTable();
  require(experts && ple, "partial pressure fixture lost a streamed store");
  const auto geometry = resources->expertStorageSnapshot()->resizable;
  const uint64_t eFloor = experts->allocatedBytes(), pFloor = ple->committedBytes();
  const uint64_t pChunk = 4096 * (ple->table().rowBytes + 13);
  const uint64_t eGrowth = geometry.fullBytes - eFloor;
  require(eGrowth && geometry.chunks.size() == 2, "partial fixture has no optional mixed expert storage");
  const auto grow = [&](bool table, uint64_t target) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    const auto admitted = growAutomaticTo(*resources, policyClock, table, target, until);
    require(static_cast<bool>(admitted), "partial fixture optional admission failed: " + admitted.detail);
  };
  grow(false, geometry.fullBytes);
  grow(true, pFloor + pChunk);
  const auto beforeMetal = resources->backend().memoryStats().allocatedBytes;
  const auto beforeHost = resources->backend().hostMemoryStats().fixedBytes;
  const auto plan = resources->memoryPlan().toStatusJson();
  pressure = MemoryPressure::Critical;
  governor.setPressure(pressure);
  measured = false;
  // Seven Q8 blocks have 28 owner entries (112 bytes). The Q4 block
  // already needs its complete minimum. Reject the Q8 donor metadata copy;
  // PLE must still donate, and the retained expert bytes remain visible.
  failingMetadataBytes = 7 * 4 * sizeof(uint32_t);
  matchingMetadataAllocations = 0;
  try { governor.controlAutomaticStorage(); }
  catch (...) { matchingMetadataAllocations = -1; throw; }
  matchingMetadataAllocations = -1;
  const auto partial = governor.snapshot();
  const auto status = resources->expertPleStatusJson();
  require(experts->allocatedBytes() == geometry.fullBytes && ple->committedBytes() == pFloor &&
              resources->backend().memoryStats().allocatedBytes == beforeMetal &&
              beforeHost - resources->backend().hostMemoryStats().fixedBytes == pChunk &&
              partial.automaticStorageReleasedBytes == pChunk && partial.automaticStorageWithheldBytes == pChunk &&
              partial.automaticStorageReason == "shrink_allocation_failed" && !partial.hostMeasurementValid &&
              status.find("\"current_quota_bytes\":" + std::to_string(eFloor + pFloor)) != std::string::npos &&
              status.find("\"expert_store_budget_bytes\":" + std::to_string(geometry.fullBytes)) != std::string::npos &&
              status.find("\"automatic_actual_bytes\":" + std::to_string(geometry.fullBytes + pFloor)) != std::string::npos &&
              resources->memoryPlan().toStatusJson() == plan,
          "partial donation fabricated rollback, lost actual/unknown accounting, or refilled storage: " + status);
  measured = true;
  const auto optionalChunk = std::find_if(geometry.chunks.begin(), geometry.chunks.end(), [](const auto &chunk) {
    return chunk.maximumChunks > chunk.minimumChunks;
  })->allocationBytes;
  pressure = MemoryPressure::Warning;
  governor.setPressure(pressure);
  spare = 3 * kGiB - optionalChunk / 2;
  governor.controlAutomaticStorage();
  require(experts->allocatedBytes() == geometry.fullBytes - optionalChunk &&
              governor.automaticStorageQuotaBytes() == eFloor + pFloor &&
              governor.snapshot().automaticStorageReleasedBytes == pChunk + optionalChunk &&
              governor.snapshot().automaticStorageWithheldBytes == pChunk + optionalChunk,
          "small Warning after partial Critical raised the old quota or donated beyond one needed real chunk");
  pressure = MemoryPressure::Critical;
  governor.setPressure(pressure);
  governor.controlAutomaticStorage();
  require(experts->allocatedBytes() == eFloor && ple->committedBytes() == pFloor &&
              governor.snapshot().automaticStorageReleasedBytes == eGrowth + pChunk &&
              governor.snapshot().automaticStorageWithheldBytes == eGrowth + pChunk &&
              resources->memoryPlan().toStatusJson() == plan,
          "pressure retry did not account only the remaining real expert donation");
}

void testExpertStorageMetadata(const char *metallibPath) {
  const test::TemporaryDirectory root("expert-storage-metadata");
  for (const bool mixed : {false, true}) {
    const auto config = flashNextConfig(metallibPath, root.path() / (mixed ? "mixed" : "q8"), mixed);
    const auto weights = model::streamableWeights(config.modelRoot, config.model,
        {.streamExperts = true, .streamTable = true});
    require(weights.expertStorage.has_value(), "MoE metadata omitted expert storage range");
    const auto &e = *weights.expertStorage;
    // Eight layers of four experts, three 256x256 projections. Q8_0
    // stores 34 bytes/32 weights; Q4_K stores 144/256, in layer zero only.
    require(e.weightBytes == (mixed ? 6'291'456 : 6'684'672) &&
            e.residentLayoutBytes == weights.expertBytes &&
            e.contiguous.minimumBytes == weights.expertFloorBytes &&
            e.resizable.minimumBytes > e.contiguous.minimumBytes &&
            e.resizable.fullBytes >= e.weightBytes && e.resizable.chunks.size() == (mixed ? 2 : 1) &&
            !e.minimumRequiredBytes && !e.allocatedBytes && !e.capacityWeightBytes,
            "expert metadata conflated weight formats, layout overhead or unknown loaded requirements");
    for (const auto &chunk : e.resizable.chunks)
      require(chunk.minimumChunks == 1 && chunk.slots == 4 &&
                  ((chunk.weightBytes == 835'584 && chunk.maximumChunks == (mixed ? 7 : 8)) ||
                   (mixed && chunk.weightBytes == 442'368 && chunk.maximumChunks == 1)),
              "mixed quantization did not retain each real format-group block");
  }
  const auto config = nativeMoeConfig(metallibPath, root.path() / "mlx", false, true);
  const auto weights = model::streamableWeights(config.modelRoot, config.model,
      {.streamExperts = true, .streamTable = true}, config.ffnLoading, true);
  // Four layers, sixteen experts, three 256x256 affine4 projections;
  // each 64 weights stores 32 code bytes and two fp16 scale/bias pairs.
  require(weights.expertStorage && weights.expertStorage->weightBytes == 7'077'888 &&
              weights.expertStorage->resizable.chunks.size() == 1 &&
              weights.expertStorage->resizable.chunks[0].weightBytes == 1'769'472 &&
              weights.expertStorage->resizable.chunks[0].maximumChunks == 4 &&
              !weights.tableBytes && !weights.tableFloorBytes,
          "MLX model without PLE did not expose its actual affine expert storage boundary");
}

void testExpertStorageLifecycle(const char *metallibPath) {
  const test::TemporaryDirectory root("expert-storage-lifecycle");
  auto config = flashNextConfig(metallibPath, root.path());
  const auto metadata = model::streamableWeights(config.modelRoot, config.model,
      {.streamExperts = true, .streamTable = true});
  config.expertMemory = {StreamedSetting::Mode::Size, metadata.expertFloorBytes};
  auto resources = RuntimeResources::create(config, 32);
  const auto allocated = resources->expertStorageSnapshot();
  require(allocated && allocated->layout == "contiguous" &&
          allocated->weightBytes == 6'684'672 && allocated->capacityWeightBytes == 835'584 &&
          allocated->residentWeightBytes == 0 && allocated->budgetBytes == metadata.expertFloorBytes &&
          allocated->minimumRequiredBytes == resources->memoryPlan().breakdown().minimumRequiredBytes &&
          *allocated->allocatedBytes == *allocated->capacityWeightBytes + *allocated->auxiliaryBytes,
          "loaded expert report lost the real weight capacity, auxiliary bytes or runtime requirement");
  resources->releasableMemory().release();
  const auto released = resources->expertStorageSnapshot();
  require(released && released->released && released->capacityWeightBytes == 0 &&
          released->residentWeightBytes == 0 && *released->allocatedBytes < *allocated->allocatedBytes &&
          released->allocatedBytes == released->auxiliaryBytes,
          "released expert report retained allocated weight capacity or dropped retained auxiliaries");
  while (resources->releasableMemory().released())
    static_cast<void>(resources->releasableMemory().restore(test::admitAll));
  const auto restored = resources->expertStorageSnapshot();
  require(restored && !restored->released && restored->allocatedBytes == allocated->allocatedBytes &&
          restored->capacityWeightBytes == allocated->capacityWeightBytes,
          "expert restore did not restore reported physical capacity");
}

void testMixedExpertStorage(const char *metallibPath) {
  for (const bool ple : {false, true}) {
    const test::TemporaryDirectory root("mixed-expert-storage");
    RuntimeResourcesConfig config;
    if (ple) config = flashNextConfig(metallibPath, root.path(), true);
    else {
      model::Qwen3_6MoeLayout target;
      target.layers = 4; target.hiddenSize = 256; target.vocabularySize = 256;
      target.experts = 4; target.expertsPerToken = 2; target.expertIntermediateSize = 256;
      auto types = gguf_fixtures::qwen4ExpTypes();
      for (const char *name : {"ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"})
        types[name] = gguf_fixtures::kQ8_0;
      for (const char *name : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
        types[name] = gguf_fixtures::kQ8_0;
        types["blk.0." + std::string(name)] = gguf_fixtures::kQ4_K;
      }
      std::filesystem::create_directories(root.path() / "target");
      test::writeFile(root.path() / "target/model.gguf", test::gguf::file(
          gguf_fixtures::metadata(target), gguf_fixtures::targetTensors(target, types)));
      constexpr std::array<uint32_t, 2> reads{1, 3};
      const auto draft = model::draftLayout(target.hiddenSize, target.vocabularySize, 1, 512, 1'000'000.0F, reads);
      test::writeSyntheticShard(root.path() / "draft/model.safetensors", test::draftTensors(draft));
      config.metallibPath = metallibPath; config.modelRoot = root.path();
      config.model = model::makeModelDescriptor("mixed-moe", target, draft,
          {.outputHiddenSize = target.hiddenSize}, model::TargetSource::Gguf, model::VisionSource::None);
      config.model.sourceIdentity = "synthetic";
      config.buildId = "mixed-moe";
      config.hostMemory = [] { return std::optional<HostMemory>(HostMemory{64 * kGiB, 64 * kGiB}); };
    }
    const auto metadata = model::streamableWeights(config.modelRoot, config.model,
        {.streamExperts = true, .streamTable = true});
    require(bool(metadata.tableBytes) == ple, "mixed fixture has the wrong PLE contract");
    config.expertMemory = {StreamedSetting::Mode::Size, metadata.expertFloorBytes};
    auto resources = RuntimeResources::create(config, 32);
    const auto e = resources->expertStorageSnapshot();
    // One singleton layer of Q4_K and the other layers Q8_0. A singleton
    // is retained at load; each streamed size holds one layer's experts.
    require(e && e->weightBytes == (ple ? 6'291'456 : 2'949'120) &&
            e->capacityWeightBytes == 1'277'952 && e->residentWeightBytes == 442'368 &&
            e->resizable.chunks.size() == 2 &&
            *e->allocatedBytes == *e->capacityWeightBytes + *e->auxiliaryBytes,
            "mixed quantization capacity used slots or source-file bytes instead of stored weight planes");
    config.expertMemory = {StreamedSetting::Mode::Off};
    resources.reset();
    resources = RuntimeResources::create(config, 32);
    const auto full = resources->expertStorageSnapshot();
    require(full && full->layout == "resident" && full->capacityWeightBytes == full->weightBytes &&
            full->residentWeightBytes == full->weightBytes && !resources->streamedExpertStats(),
            "whole resident experts did not report full byte capacity");
    resources->releasableMemory().release();
    const auto released = resources->expertStorageSnapshot();
    require(released->capacityWeightBytes == 0 && released->residentWeightBytes == 0 &&
            released->allocatedBytes == released->residentSlotTableBytes,
            "resident expert release omitted its retained slot table");
  }
}

// What a start sized: its context, the least memory its plan needs and its
// arenas, its routed experts' store, whether its SSD cache started, its
// backend's memory, its log and how many times it loaded the weights.
struct StreamedStart final {
  uint32_t context = 0;
  uint64_t minimumRequired = 0, arenaBytes = 0;
  std::optional<model::ExpertStoreStats> experts;
  bool tier = false;
  metal::MetalMemoryStats memory;
  std::string log;
  size_t loads = 0;
};

StreamedStart startStreamed(const RuntimeResourcesConfig &config) {
  StreamedStart started;
  started.log = test::capturedStderr([&] {
    auto resources = RuntimeResources::create(config, 0);
    const EngineMemoryBreakdown &budget = resources->memoryPlan().breakdown();
    started.context = resources->memoryPlan().maximumContextTokens();
    started.minimumRequired = budget.minimumRequiredBytes;
    started.arenaBytes = budget.sharedPrefillBytes + budget.sharedDecodeBytes;
    started.experts = resources->streamedExpertStats();
    started.tier = resources->cache().snapshot().kvTier.capacityBytes != 0;
    started.memory = resources->backend().refreshMemoryStats();
  });
  for (size_t at = started.log.find("Weights loaded"); at != std::string::npos;
       at = started.log.find("Weights loaded", at + 1))
    ++started.loads;
  return started;
}

// The sizes a model's weights take, planned from its files before it loads,
// are what it loads: the images without its routed experts and those with
// all of them, and the smallest store, which holds a layer's experts and
// refuses a byte less. One load plans the arenas of either: those of streamed
// experts planned from a model that loads them whole are a streaming model's,
// and the reverse.
void testStreamableWeightsAreWhatLoads(const RuntimeResourcesConfig &config) {
  const std::filesystem::path &root = config.modelRoot;
  const model::StreamableWeights weights =
      model::streamableWeights(root, config.model, {.streamExperts = true, .streamTable = true});
  require(weights.expertBytes > weights.expertFloorBytes && weights.expertFloorBytes && !weights.tableBytes &&
              !weights.tableFloorBytes && !weights.tableRowBytes && weights.tableRefusal.empty(),
          "the streamable weights of a MoE without a table are not its experts'");
  const auto allocated = [](const model::LoadedModel &loaded) {
    return loaded.targetActualAllocatedBytes() + loaded.draft.actualAllocatedBytes +
           loaded.vision.actualAllocatedBytes;
  };
  metal::MetalBackend backend(config.metallibPath.string());
  const ops::ExecutionPlans operators(backend.capabilities());
  const auto arenas = [&](const model::LoadedModel &loaded, model::StreamedParts parts) {
    const model::ModelMemoryPlan plan =
        model::plannedRuntimeMemory(loaded, operators, kv::Format::Int8, ops::GdnStateFormat::Float32, parts);
    return plan.sharedPrefillPlannedAllocatedBytes + plan.sharedDecodePlannedAllocatedBytes;
  };
  const model::LoadedModel whole = model::loadModel(backend, root, config.model);
  require(allocated(whole) == weights.loadedBytes + weights.expertBytes,
          "the experts loaded with their layers did not take what was planned");
  // Streamed experts read on a pool of SSD readers, as the engine's do.
  const auto readers = std::make_shared<model::SsdReaders>(model::kSsdReaders);
  const model::LoadedModel streamed = model::loadModel(
      backend, root, config.model, {.expertMemoryBytes = weights.expertFloorBytes, .readers = readers});
  require(streamed.expertStore() && streamed.expertStore()->stats().slots == kMoeExperts &&
              allocated(streamed) <= weights.loadedBytes + weights.expertFloorBytes,
          "the smallest store did not hold a layer's experts within its bytes");
  require(arenas(whole, {.experts = true}) == arenas(streamed, model::StreamedParts::loaded(streamed)) &&
              arenas(streamed, {}) == arenas(whole, model::StreamedParts::loaded(whole)) &&
              arenas(whole, {.experts = true}) > arenas(whole, {}),
          "the arenas of what streams are not those of a model loaded so, or prefill groups took no more");
  rejects([&] {
    static_cast<void>(
        model::loadModel(backend, root, config.model,
                         {.expertMemoryBytes = weights.expertFloorBytes - 1, .readers = readers}));
  }, "holds fewer slots", "a store a byte below the smallest loaded");
}

// --expert-memory auto loads a MoE target's routed experts where the plan
// holds them, in one load as main does. Where it does not, they stream
// through a store of all that the first 64K tokens of KV leave, in two
// loads: the first plans the arenas of either, and the second loads that
// store; the backend's peaks, which the memory audit holds to the plan,
// count only the second. The automatic SSD cache gives its state staging up
// where it alone would stream them. Off loads them whatever the budget, a
// budget without room for their arenas stops the start with its breakdown,
// and a size below the smallest store is refused by name before anything
// loads.
void testAutomaticExpertMemory(RuntimeResourcesConfig config) {
  const model::StreamableWeights weights =
      model::streamableWeights(config.modelRoot, config.model, {.streamExperts = true});
  const StreamedStart whole = startStreamed(config);
  require(!whole.experts && whole.loads == 1 &&
              whole.log.find("Routed experts load with their layers") != std::string::npos,
          "routed experts memory holds did not load with their layers in one load");
  const auto streams = [&](const StreamedStart &start, const char *message) {
    require(start.experts && start.experts->slots > kMoeExperts &&
                start.experts->slots < kMoeLayers * kMoeExperts && start.loads == 2 &&
                start.log.find("--expert-memory auto leaves it beside the floors") != std::string::npos &&
                start.context >= 65'536,
            message);
  };
  // A byte short of the budget that holds every expert, which the plan
  // without the arenas holds: whole, then through the store. The first
  // load's experts do not count toward the peaks.
  config.maximumMemoryBytes = whole.minimumRequired - 1;
  const StreamedStart reloaded = startStreamed(config);
  streams(reloaded,
          "experts that fit beside everything but the arenas did not stream through the store the plan sizes");
  const uint64_t firstLoad = weights.loadedBytes + weights.expertBytes;
  require(reloaded.memory.peakAllocatedBytes < firstLoad && reloaded.memory.devicePeakAllocatedBytes < firstLoad,
          "the backend's peaks counted the load the start discarded");
  // Short of that too: at the floor, then through the store.
  config.maximumMemoryBytes = whole.minimumRequired - whole.arenaBytes - 1;
  streams(startStreamed(config),
          "routed experts memory cannot hold did not stream through the store the plan sizes");

  // A byte short of every expert beside the automatic SSD cache's staging:
  // the cache gives it up, and they load whole in one load.
  config.cacheDisk = {.automatic = true};
  config.diskVolume = reading(localVolume(500 * kGiB));
  config.maximumMemoryBytes = whole.minimumRequired + config.model.stateLayout.cachedBytes() - 1;
  const StreamedStart yielded = startStreamed(config);
  require(!yielded.experts && !yielded.tier && yielded.loads == 1 &&
              yielded.log.find("SSD cache off: its " + std::to_string(config.model.stateLayout.cachedBytes() >> 20) +
                               " MiB of state staging would stream the routed experts") != std::string::npos,
          "the automatic SSD cache's staging streamed experts that load whole without it");
  config.maximumMemoryBytes += 1;
  const StreamedStart kept = startStreamed(config);
  require(!kept.experts && kept.tier && kept.loads == 1, "the automatic SSD cache yielded room it had");
  config.cacheDisk = {};
  config.diskVolume = RuntimeResourcesConfig{}.diskVolume;

  config.expertMemory = {StreamedSetting::Mode::Off, 0};
  config.maximumMemoryBytes = 0;
  require(!startStreamed(config).experts, "--expert-memory off streamed the experts");
  // A budget that holds the weights but not their arenas: the start loads
  // them, and stops with the engine plan's breakdown.
  config.maximumMemoryBytes = whole.minimumRequired - whole.arenaBytes / 2;
  try {
    static_cast<void>(RuntimeResources::create(config, 0));
    throw std::runtime_error("a plan without room for its arenas started");
  } catch (const RuntimeResourcesError &error) {
    require(error.stage() == RuntimeResourceStage::MemoryPlanning &&
                error.budgetDescription().find("deficit: " + std::to_string(whole.arenaBytes / 2) + " bytes") !=
                    std::string::npos,
            "a start without room for its arenas was refused without its breakdown");
  }
  config.maximumMemoryBytes = 0;
  config.expertMemory = {StreamedSetting::Mode::Size, weights.expertFloorBytes};
  const StreamedStart floor = startStreamed(config);
  require(floor.experts && floor.experts->slots == kMoeExperts && floor.loads == 1,
          "an explicit store did not load once at its size");
  config.expertMemory = {StreamedSetting::Mode::Size, weights.expertFloorBytes - 1};
  try {
    static_cast<void>(RuntimeResources::create(config, 0));
    throw std::runtime_error("a store below its floor was accepted");
  } catch (const RuntimeResourcesError &error) {
    require(error.stage() == RuntimeResourceStage::Configuration &&
                std::string_view(error.what()).find("is less than its smallest store") != std::string_view::npos,
            "a store below its floor was not refused by name before loading");
  }
}

// --verify-rows: by default a MoE target's decode cycles verify the rows the
// verify budget chooses, whether its routed experts load with their layers or
// stream from the SSD, and a dense target's every row; `all` verifies every
// row. The start says which, and the runtime built on what it assembled runs
// it.
void testVerifyRows(const char *metallibPath, RuntimeResourcesConfig moe) {
  struct Started final {
    std::string log;
    bool budget = false;
  };
  const auto start = [](const RuntimeResourcesConfig &config) {
    Started started;
    started.log = test::capturedStderr([&] {
      const auto resources = RuntimeResources::create(config, 0);
      const EngineMemoryBreakdown &budget = resources->memoryPlan().breakdown();
      std::unique_ptr<model::RuntimeModel> runtime;
      require(static_cast<bool>(resources->memoryGovernor().allocationAdmission()(
                  budget.sharedPrefillBytes + budget.sharedDecodeBytes,
                  [&] { runtime = model::createRuntime(resources->modelContext()); })),
              "a runtime's arenas were refused");
      started.budget = runtime->telemetry().verifyBudget;
    });
    return started;
  };
  const auto requireStart = [&](const RuntimeResourcesConfig &config, std::string_view line, bool budget,
                                const char *message) {
    const Started started = start(config);
    require(started.log.find(line) != std::string::npos && started.budget == budget, message);
  };
  const std::string_view budgeted = "Decode cycles verify the rows the verify budget chooses (--verify-rows auto, as "
                                    "moe-test is a MoE target): greedy and seeded outputs can differ from run to run.";
  requireStart(moe, budgeted, true, "a MoE target with resident experts did not budget its rows");
  moe.expertMemory = {StreamedSetting::Mode::Size,
                      model::streamableWeights(moe.modelRoot, moe.model, {.streamExperts = true}).expertFloorBytes};
  requireStart(moe, budgeted, true, "a MoE target whose experts stream did not budget its rows");
  moe.verifyRows = model::VerifyRows::All;
  requireStart(moe, "Decode cycles verify every row of the draft's block (--verify-rows all).", false,
               "--verify-rows all did not verify every row of a MoE target");
  const test::TemporaryDirectory root("splash-verify-rows");
  requireStart(loadableConfig(metallibPath, root.path(), "verify-rows-test"),
               "Decode cycles verify every row of the draft's block (--verify-rows auto, as verify-rows-test is "
               "dense).",
               false, "a dense target did not verify every row");
}

// Startup fails at the loader, whose vision weight files are deliberately
// absent. Reaching it is the assertion: everything the engine checks before
// loading the weights let this configuration through.
void requireReachesModelLoader(RuntimeResourcesConfig config,
                               const std::filesystem::path &root,
                               const char *message) {
  try {
    auto resources = RuntimeResources::create(config, 0);
    throw std::runtime_error("a model without weights unexpectedly loaded");
  } catch (const RuntimeResourcesError &error) {
    const std::string_view text = error.what();
    require(error.failure() == RuntimeResourceFailure::Other &&
                error.stage() == RuntimeResourceStage::ModelLoading &&
                text.find(root.string()) != std::string_view::npos,
            message);
  }
}

// Beside its weights a model needs at least the runtime reserves, one lane's
// state and the KV runway.
uint64_t minimumBytes(const RuntimeResourcesConfig &config) {
  return minimumRequiredBytes(model::streamableWeights(config.modelRoot, config.model, {}).loadedBytes +
                                  model::kPipelineReserveBytes + model::kRuntimeOverheadReserveBytes,
                              config.model.stateLayout.laneBytes(),
                              config.model.targetKvLayout)
      .value();
}

// A persistent cache's directory is named for what its copies hold: the same
// models and layouts name the same one under any build, and a change to any
// of them, or to the RoPE scaling its keys went through, names another.
void testPersistentCacheNamespace() {
  const model::ModelDescriptor model = model::makeModelDescriptor(
      "namespace-test", model::Qwen3_8Layout{}, test::qwen3_8Draft(),
      test::qwen3_8Vision(), model::TargetSource::Safetensors,
      model::VisionSource::Safetensors);
  const auto identity = [&](char models, std::string_view build, kv::Format format) {
    kv::Layout layout = model.targetKvLayout;
    layout.format = format;
    return makeRuntimeCacheIdentity(std::string(64, models), std::string(64, 'b'), build,
                                    layout, {});
  };
  const std::string name =
      persistentCacheNamespace(identity('a', "build", kv::Format::Int8), model.stateLayout);
  require(name.size() == 32 && name.find_first_not_of("0123456789abcdef") == std::string::npos,
          "a cache namespace is not 32 lowercase hex digits");
  require(persistentCacheNamespace(identity('a', "another build", kv::Format::Int8),
                                   model.stateLayout) == name,
          "the build changed the cache namespace");
  model::CompositeStateLayout states = model.stateLayout, bfloatStates = model.stateLayout;
  ++states.draft.layers;
  bfloatStates.target.format = ops::GdnStateFormat::BFloat16;
  require(persistentCacheNamespace(identity('c', "build", kv::Format::Int8), model.stateLayout) !=
                  name &&
              persistentCacheNamespace(identity('a', "build", kv::Format::BFloat16),
                                       model.stateLayout) != name &&
              persistentCacheNamespace(identity('a', "build", kv::Format::Int8), states) != name &&
              persistentCacheNamespace(identity('a', "build", kv::Format::Int8), bfloatStates) != name,
          "other models, layouts or state formats shared a cache namespace");
  auto standard = model.stateLayout;
  standard.draft.fullHistory = true;
  standard.draft.layerContextTokens[0] = 8192;
  const auto shortHistory = persistentCacheNamespace(identity('a', "build", kv::Format::Int8), standard);
  standard.draft.layerContextTokens[0] = 32768;
  require(shortHistory != name &&
              shortHistory != persistentCacheNamespace(identity('a', "build", kv::Format::Int8), standard),
          "standard draft history capacities shared a persistent namespace");
  const auto rotatedQ4 = persistentCacheNamespace(identity('a', "build", kv::Format::Q4_0Hadamard), model.stateLayout);
  const auto plainQ4 = persistentCacheNamespace(identity('a', "build", kv::Format::Q4_0Unrotated), model.stateLayout);
  const auto q4h = persistentCacheNamespace(identity('a', "build", kv::Format::Int4Hadamard), model.stateLayout);
  require(rotatedQ4 != plainQ4 && rotatedQ4 != q4h && plainQ4 != name,
          "different Q4 quantizers or rotation bases shared a persistent namespace");
  // A dense layout's namespace is the one main (be3b083c) names, with FP32
  // and BF16 states: a change of its lines would leave every 27B's and 35B's
  // persistent cache cold.
  require(name == "6365d687c8be103a3bbcc6d2a83b1543" &&
              persistentCacheNamespace(identity('a', "build", kv::Format::Int8), bfloatStates) ==
                  "172879058b5e2bcc713e129a2b62393a",
          "a dense layout's cache namespace is not the one main names");
  // A QSA layout's pages hold token-major values and the index's pooled
  // keys, and its states the PLE history and the index's pending keys.
  RuntimeCacheIdentity tokenMajor = identity('a', "build", kv::Format::Int8), indexed = tokenMajor;
  tokenMajor.kvLayout.valuesTokenMajor = true;
  indexed.kvLayout.indexDimension = 128;
  model::CompositeStateLayout auxiliary = model.stateLayout;
  auxiliary.target.auxiliaryBytes = 9216;
  require(persistentCacheNamespace(tokenMajor, model.stateLayout) != name &&
              persistentCacheNamespace(indexed, model.stateLayout) != name &&
              persistentCacheNamespace(identity('a', "build", kv::Format::Int8), auxiliary) != name,
          "token-major values, an index or auxiliary state shared a dense layout's cache namespace");
  const auto scaled = [&](double factor) {
    kv::Layout layout = model.targetKvLayout;
    layout.format = kv::Format::Int8;
    return persistentCacheNamespace(makeRuntimeCacheIdentity(std::string(64, 'a'), std::string(64, 'b'), "build",
                                                             layout, {factor, 262144}),
                                    model.stateLayout);
  };
  require(scaled(1.0) == name && scaled(4.0) != name && scaled(2.0) != name && scaled(2.0) != scaled(4.0),
          "YaRN's keys shared a cache namespace with another scaling's");
  // Factors that print alike to six digits still rotate keys differently.
  require(scaled(3.9999999) != scaled(4.0) && scaled(1.0000001) != scaled(1.0000004),
          "factors alike to six digits shared a cache namespace");
}

void testWeightBudgetBeforeLoading(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());

  {
    config.memoryPressure = [] { return MemoryPressure::Critical; };
    try {
      auto resources = RuntimeResources::create(config, 0);
      throw std::runtime_error("model load ignored system pressure");
    } catch (const RuntimeResourcesError &error) {
      require(error.failure() == RuntimeResourceFailure::HostCapacity,
              "startup pressure did not remain retryable: stage=" + std::to_string(static_cast<int>(error.stage())) +
                  " failure=" + std::to_string(static_cast<int>(error.failure())) + " what=" + error.what());
      require(std::string_view(error.what()).find("not enough free memory") !=
                  std::string_view::npos,
              "startup pressure reached the weight loader");
    }
  }
  config.memoryPressure = [] { return MemoryPressure::Warning; };

  // The low ceiling is one byte short of the minimum, so admission must
  // charge the weights and all a model needs beside them. The other ceilings
  // must reach the real loader, whose expected weight files are deliberately
  // absent.
  const uint64_t minimum = minimumBytes(config);
  for (uint64_t ceiling : {minimum - 1, minimum, uint64_t{0}}) {
    config.maximumMemoryBytes = ceiling;
    try {
      auto resources = RuntimeResources::create(config, 0);
      throw std::runtime_error("a model without weights unexpectedly loaded");
    } catch (const RuntimeResourcesError &error) {
      if (ceiling == minimum - 1) {
        require(error.failure() == RuntimeResourceFailure::EngineCapacity,
                "hard weight budget lost its engine-capacity classification");
        const std::string_view text = error.what();
        require(error.stage() == RuntimeResourceStage::MemoryPlanning &&
                    text.find("require " + std::to_string(minimum) + " bytes") !=
                        std::string_view::npos &&
                    text.find("budget is " + std::to_string(minimum - 1) +
                              " bytes") != std::string_view::npos,
                "weight loading began before checking the memory ceiling");
      } else {
        require(error.failure() == RuntimeResourceFailure::Other,
                "missing model file was misclassified as allocation pressure");
      }
    }
  }
  config.maximumMemoryBytes = 0;
  requireReachesModelLoader(config, root.path(),
                            "a sufficient weight budget did not reach the "
                            "model loader");
}

// A state's write to the disk tier stages through a state-sized buffer the
// plan sets aside only when the tier starts: a quota that holds no state
// leaves the tier off and the budget to KV. An explicit quota's staging is
// planned whatever it costs, and the automatic one never costs a start. The
// notices and the refusal name the tier as users know it, the SSD cache.
void testStateStagingNeedsAStartedTier(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());
  const uint64_t minimum = minimumBytes(config);
  const uint64_t stateBytes = config.model.stateLayout.cachedBytes();
  config.maximumMemoryBytes = minimum;
  config.cacheDisk = {.bytes = stateBytes - 1};
  const std::string notice = test::capturedStderr([&] {
    requireReachesModelLoader(config, root.path(),
                              "a quota that holds no state set staging aside");
  });
  require(notice.find("SSD cache off: its quota holds no state of " +
                      std::to_string(stateBytes >> 20) + " MiB") != std::string::npos,
          "the notice of a quota that holds no state did not name the SSD cache");

  // The automatic tier lets a budget with no room for its staging start,
  // and says why it is off.
  config.cacheDisk = {.automatic = true};
  const std::string automatic = test::capturedStderr([&] {
    requireReachesModelLoader(config, root.path(),
                              "the automatic tier's staging refused a start");
  });
  require(automatic.find("SSD cache off: the memory budget has no room for its " +
                         std::to_string(stateBytes >> 20) +
                         " MiB of state staging beside the model") != std::string::npos,
          "the automatic tier did not say the memory budget turned it off");

  config.cacheDisk = {.bytes = stateBytes};
  try {
    auto resources = RuntimeResources::create(config, 0);
    throw std::runtime_error("a model without weights unexpectedly loaded");
  } catch (const RuntimeResourcesError &error) {
    require(error.failure() == RuntimeResourceFailure::EngineCapacity &&
                error.stage() == RuntimeResourceStage::MemoryPlanning &&
                std::string_view(error.what())
                        .find("SSD cache state staging require " +
                              std::to_string(minimum + stateBytes) +
                              " bytes") != std::string_view::npos,
            "a started tier's staging was not counted by name before loading");
  }
  config.maximumMemoryBytes = minimum + stateBytes;
  requireReachesModelLoader(config, root.path(),
                            "a budget with room for the staging did not "
                            "reach the model loader");
}

// The image patch limit is a positive multiple of four up to the
// protocol's per-image ceiling; anything else is refused before the backend
// is created.
void testImagePatchCapIsBounded(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());
  for (const uint32_t patches : {ops::kMaximumImagePatches + 4, 6U}) {
    config.maximumImagePatches = patches;
    try {
      auto resources = RuntimeResources::create(config, 0);
      throw std::runtime_error("an image patch cap outside the protocol's was accepted");
    } catch (const RuntimeResourcesError &error) {
      require(error.stage() == RuntimeResourceStage::Configuration,
              "an invalid image patch cap was not a configuration error");
    }
  }
  config.maximumImagePatches = 4096;
  requireReachesModelLoader(config, root.path(),
                            "a smaller image patch cap did not reach the model loader");
}

// Startup and the governor sample host memory through the config's probe,
// and the automatic SSD cache its volume through another: without either,
// the config is refused before the backend is created.
void testHostMemoryProbeIsRequired(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  for (const bool memory : {true, false}) {
    RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());
    if (memory)
      config.hostMemory = {};
    else
      config.diskVolume = {};
    try {
      auto resources = RuntimeResources::create(config, 0);
      throw std::runtime_error("resources were assembled without a host memory or volume probe");
    } catch (const RuntimeResourcesError &error) {
      require(error.stage() == RuntimeResourceStage::Configuration,
              memory ? "a missing host memory probe was not a configuration error"
                     : "a missing volume probe was not a configuration error");
    }
  }
}

// The 27B under a budget half a GiB above its weights: the weights alone fit,
// but not with what the runtime needs beside them. Startup refuses it before
// any weight is loaded.
void testModelBeyondBudgetIsRefusedBeforeLoading(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());
  config.model = model::makeModelDescriptor("budget-test", model::Qwen3_8Layout{}, test::qwen3_8Draft(),
                                            test::qwen3_8Vision(), model::TargetSource::Safetensors,
                                            model::VisionSource::Safetensors);
  writeCheckpoints(root.path(), config.model);
  config.maximumMemoryBytes = model::streamableWeights(root.path(), config.model, {}).loadedBytes + kGiB / 2;
  try {
    auto resources = RuntimeResources::create(config, 0);
    throw std::runtime_error("a model that cannot fit unexpectedly loaded");
  } catch (const RuntimeResourcesError &error) {
    require(error.failure() == RuntimeResourceFailure::EngineCapacity &&
                error.stage() == RuntimeResourceStage::MemoryPlanning,
            "a model that cannot fit reached the weight loader");
  }
}

// The rule that keeps users off the startup floor: admission weighs
// reclaimable memory against the macOS reserve, never against the model.
// A model far larger than everything reclaimable still starts, because the
// operation guard checks each image's allocation as loading builds the
// residency up. A quarter of the 27B's layers keeps it within the automatic
// budget of a 24 GB Mac.
void testStartupAdmissionIgnoresModelSize(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());
  model::Qwen3_8Layout target;
  target.layers = 16;
  model::DFlashDraftLayout draft = test::qwen3_8Draft();
  draft.targetLayers = {1, 4, 7, 10, 13};
  config.model = model::makeModelDescriptor("budget-test", target, draft, test::qwen3_8Vision(),
                                            model::TargetSource::Safetensors, model::VisionSource::Safetensors);
  writeCheckpoints(root.path(), config.model);
  require(model::streamableWeights(root.path(), config.model, {}).loadedBytes > 3 * kGiB,
          "the model must exceed the sample");
  config.hostMemory = [] { return std::optional<HostMemory>(HostMemory{3 * kGiB, 3 * kGiB}); };
  requireReachesModelLoader(config, root.path(),
                            "a model larger than reclaimable host memory "
                            "refused to start");

  // Below the reserve macOS is the one at risk, so startup waits instead.
  // Unmeasurable telemetry waits the same way.
  for (std::optional<uint64_t> available :
       {std::optional<uint64_t>(64 * kMiB), std::optional<uint64_t>()}) {
    config.hostMemory = [available] {
      return available ? std::optional<HostMemory>(HostMemory{*available, *available}) : std::nullopt;
    };
    try {
      auto resources = RuntimeResources::create(config, 0);
      throw std::runtime_error("model load ignored the macOS reserve");
    } catch (const RuntimeResourcesError &error) {
      require(error.failure() == RuntimeResourceFailure::HostCapacity,
              "exhausted host memory did not remain retryable");
      require(std::string_view(error.what()).find("not enough free memory") !=
                  std::string_view::npos,
              "exhausted host memory reached the weight loader");
    }
  }
}

// The memory plan takes the vision category from what loaded, so a model
// with vision whose loader produced no vision bytes must stop here.
void testLoadedVisionIsRequiredOnlyWithVision() {
  model::LoadedModel loaded;
  loaded.descriptor = model::makeModelDescriptor(
      "loaded-test", model::Qwen3_8Layout{}, test::qwen3_8Draft(),
      test::qwen3_8Vision(), model::TargetSource::Safetensors,
      model::VisionSource::Safetensors);
  model::Qwen3_8Weights target;
  target.actualAllocatedBytes = 1;
  target.manifestFingerprintSha256 = "target";
  loaded.target = std::move(target);
  loaded.draft.actualAllocatedBytes = 1;
  loaded.manifestFingerprintSha256 = "model";
  require(loaded.descriptor.hasVision(), "the test model has no vision");
  rejects([&] { requireLoadedModel(loaded); },
          "loaded model has incomplete allocation accounting",
          "a multimodal model without loaded vision weights was accepted");
  loaded.vision.actualAllocatedBytes = 1;
  requireLoadedModel(loaded);
  loaded.vision.actualAllocatedBytes = 0;
  loaded.descriptor.visionSource = model::VisionSource::None;
  requireLoadedModel(loaded);
}

// Resource assembly names the step it fails at, at its real throw site: a
// host with less than its reserve refuses the start before the weights are
// opened, and the startup retry window retries it. Past the weights the host
// refuses only what loading would (testStartGrowsThroughHostPressure).
void testFailedStepIsNamed(const char *metallibPath) {
  using namespace std::chrono_literals;
  const test::TemporaryDirectory root("splash-assembly");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "assembly-test");
  const auto host = [&](uint64_t available) {
    config.hostMemory = [available] { return std::optional<HostMemory>(HostMemory{available, available}); };
  };
  // Nothing but the host refuses this model. The suite's one complete start
  // also reports the SSD cache a disk quota gives it, by that name: a size of
  // its temporary files' whole volume, which the room the volume has beyond
  // its floor always limits, as that room.
  host(64 * kGiB);
  const std::optional<DiskVolume> temporary = volumeOf(std::filesystem::temp_directory_path());
  require(temporary && temporary->local, "the temporary files' volume is not a local volume measured");
  config.cacheDisk = {.bytes = temporary->sizeBytes};
  const std::string started =
      test::capturedStderr([&] { static_cast<void>(RuntimeResources::create(config, 0)); });
  const size_t at = started.find("SSD cache: up to ");
  const std::string line =
      at == std::string::npos ? std::string() : started.substr(at, started.find('\n', at) - at);
  require(line.find(" of temporary files in ") != std::string::npos &&
              line.find(" (the ") != std::string::npos &&
              line.find(" its volume has beyond the ") != std::string::npos &&
              line.find("; KV pages of ") != std::string::npos &&
              line.find("--max-cache-disk 0 turns it off.") != std::string::npos,
          "a start did not report its SSD cache by that name, or a size its volume limits as the "
          "volume's room");
  config.cacheDisk = {};
  const auto failure = [&](uint64_t available) {
    host(available);
    try {
      static_cast<void>(RuntimeResources::create(config, 0));
    } catch (const RuntimeResourcesError &error) {
      return RuntimeBootstrapError(error).report();
    }
    throw std::runtime_error("resources were assembled on a host short of memory");
  };
  const uint64_t reserve = EngineMemoryPolicy::hostAvailableReserveBytes(
      metal::probeDeviceCapabilities().physicalMemoryBytes);
  const RuntimeBootstrapReport before = failure(reserve / 2);
  require(before.resourceStage == RuntimeResourceStage::ModelLoading &&
              before.resourceFailure == RuntimeResourceFailure::HostCapacity &&
              before.message.find("not enough free memory to start") != std::string::npos,
          "a start the host refused before the weights loaded named another step");
  require(before.message.find("--max-memory") != std::string::npos,
          "a start the host refused did not name --max-memory as a remedy");
  const auto first = StartupRetryWindow::Clock::time_point{};
  require(StartupRetryWindow(30s).retryUntil(before, first) == first + 30s,
          "a start the host refused was not retried");
}

// What a start allocates, the KV runway and then the arenas and the warmup as
// RuntimeBootstrap::start allocates them, is memory a request in service
// needs: neither the warning margin above the spare reserve nor macOS's
// warning refuses it, only what refuses loading, and an allocation the
// engine marks meanwhile leaves the start's mark in place. Once serving
// begins, growth that no request in service needs keeps to both again.
void testStartGrowsThroughHostPressure(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-start-pressure");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "start-pressure-test");
  const uint64_t reserve = EngineMemoryPolicy::hostAvailableReserveBytes(
      metal::probeDeviceCapabilities().physicalMemoryBytes);
  // A host with its reserve but not the warning margin beyond it, then one
  // with ample room while macOS warns.
  for (const bool warning : {false, true}) {
    const uint64_t available = warning ? 64 * kGiB : reserve + kHostWarningMarginBytes / 2;
    config.hostMemory = [available] {
      return std::optional<HostMemory>(HostMemory{available, available});
    };
    config.memoryPressure = [warning] {
      return warning ? MemoryPressure::Warning : MemoryPressure::Normal;
    };
    const std::string host = warning ? "under macOS's warning" : "short of the warning margin";
    std::unique_ptr<RuntimeResources> resources;
    static_cast<void>(
        test::capturedStderr([&] { resources = RuntimeResources::create(config, 0); }));
    const metal::AllocationAdmission admit = resources->memoryGovernor().allocationAdmission();
    const EngineMemoryBreakdown &budget = resources->memoryPlan().breakdown();
    std::unique_ptr<model::RuntimeModel> runtime;
    require(static_cast<bool>(admit(budget.sharedPrefillBytes + budget.sharedDecodeBytes, [&] {
              runtime = model::createRuntime(resources->modelContext());
            })),
            "the start's arenas waited " + host);
    EngineConfig engine;
    connectToGovernor(engine, resources->memoryGovernor());
    {
      const Serving allocation(engine.serving, true);
    }
    // The warmup's lane state; it throws if refused.
    static_cast<void>(runtime->warmupPrefill(model::ExecutionLimits::prefillTokenBudget));
    static_cast<void>(runtime->warmupDecodeBatch(1));
    resources->beginServing();
    const metal::AllocationResult grown = admit(kMiB, [] {});
    require(!grown && grown.failure == metal::AllocationFailure::HostPressure,
            "growth that no request in service needs went on " + host + " once serving began");
  }
}

// A model served with YaRN starts and runs as any other, with YaRN all the
// way into its runtime: the start names its factor and the context it
// serves, the runtime's geometry serves that window, its arenas hold YaRN's
// rotary frequencies, and its prefill and decode build the target's RoPE
// tables with the attention factor.
void testYarnRuntime(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-yarn");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "yarn-test");
  config.model = model::servedFor(std::move(config.model), 1'010'000);
  std::unique_ptr<RuntimeResources> resources;
  const std::string started =
      test::capturedStderr([&] { resources = RuntimeResources::create(config, 0); });
  require(started.find("YaRN is on: the target's RoPE is scaled by 3.85284 for 1010000 tokens of context, past the "
                       "native window of 262144. Static YaRN changes answers at every length, so pass a "
                       "--max-context above 262144 only when a request needs it.") != std::string::npos,
          "a start with YaRN did not name its factor and the context it serves");
  const model::RuntimeContext context = resources->modelContext();
  const auto geometry = model::RuntimeGeometry::from(context.model, kv::Format::Int8, ops::GdnStateFormat::Float32);
  require(geometry.target.ropeScaling == config.model.ropeScaling && geometry.target.contextTokens() == 1'010'000,
          "the runtime's geometry lost YaRN or its window");
  std::array<float, SPLASH_TARGET_ROPE_PAIRS> expected{};
  model::rotaryInverseFrequencies(geometry.target.rotaryTheta, geometry.target.rotaryPairs,
                                  geometry.target.ropeScaling, expected);
  const metal::AllocationAdmission admit = resources->memoryGovernor().allocationAdmission();
  const EngineMemoryBreakdown &budget = resources->memoryPlan().breakdown();
  std::unique_ptr<model::RuntimeModel> runtime;
  require(static_cast<bool>(admit(budget.sharedPrefillBytes + budget.sharedDecodeBytes, [&] {
            const model::PrefillArena arena(context.backend, geometry, context.operators);
            require(std::memcmp(arena.get(model::PrefillTensor::TargetInverseFrequencies).contents(),
                                expected.data(), sizeof(expected)) == 0,
                    "the prefill arena does not hold YaRN's frequencies");
          })),
          "the YaRN prefill arena was refused");
  require(static_cast<bool>(admit(budget.sharedPrefillBytes + budget.sharedDecodeBytes,
                                  [&] { runtime = model::createRuntime(context); })),
          "the YaRN runtime's arenas were refused");
  static_cast<void>(runtime->warmupPrefill(model::ExecutionLimits::prefillTokenBudget));
  static_cast<void>(runtime->warmupDecodeBatch(1));
}

// The SSD cache holds no write back unless a write budget is set; one that
// is set starts with a quarter of a day's.
void testWriteBudgetOnlyWhenSet(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-write-budget");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "write-budget-test");
  config.cacheDisk = {.bytes = kGiB};
  const auto tier = [&] {
    KvTierSnapshot snapshot;
    static_cast<void>(test::capturedStderr([&] {
      snapshot = RuntimeResources::create(config, 0)->cache().snapshot().kvTier;
    }));
    return snapshot;
  };
  const KvTierSnapshot unset = tier();
  require(unset.capacityBytes == kGiB && !unset.writeBudgetBytes &&
              unset.writesGated == DiskWritesGate::No,
          "the SSD cache had a write budget nobody set");
  config.cacheDiskWritesPerDay = 8 * kGiB;
  const KvTierSnapshot set = tier();
  require(set.writeBudgetBytes == 2 * kGiB && set.writesGated == DiskWritesGate::No,
          "a write budget that was set did not start with a quarter of a day's");
}

// Every start removes the caches under its root that nobody used for two
// weeks and keeps the others, one without --persistent-cache or the SSD
// cache too.
void testStaleCachesGoAtEveryStart(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-stale-model"), caches("splash-stale");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "stale-test");
  config.cacheRoot = caches.path() / "prefix-cache";
  const std::string stale(CacheDirectory::kNamespaceDigits, 'a'),
      fresh(CacheDirectory::kNamespaceDigits, 'b');
  for (const std::string &name : {stale, fresh}) {
    auto directory = CacheDirectory::open(config.cacheRoot, name, std::chrono::milliseconds(0));
    std::ofstream(directory->kvSlots()) << "cache";
  }
  const timeval old[2] = {{0, 0}, {1000000000, 0}};
  for (const char *file : {"lock", "kv.slots"})
    require(::utimes((config.cacheRoot / stale / file).c_str(), old) == 0,
            "a cache file could not be aged");
  static_cast<void>(test::capturedStderr([&] { static_cast<void>(RuntimeResources::create(config, 0)); }));
  require(!std::filesystem::exists(config.cacheRoot / stale) &&
              std::filesystem::exists(config.cacheRoot / fresh / "kv.slots"),
          "a start without a persistent cache kept a stale cache under its root, or removed another");
}

// A cache root the engine makes for a persistent cache stays out of Time
// Machine backups; one that was there keeps its own setting.
void testMadeCacheRootStaysOutOfBackups(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-backups-model"), caches("splash-backups");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "backups-test");
  config.cacheDisk = {.bytes = kGiB};
  config.persistentCache = true;
  const auto excluded = [](const std::filesystem::path &path) {
    NSURL *url = [NSURL fileURLWithPath:@(path.c_str()) isDirectory:YES];
    NSNumber *value = nil;
    return [url getResourceValue:&value forKey:NSURLIsExcludedFromBackupKey error:nil] &&
           value.boolValue;
  };
  config.cacheRoot = caches.path() / "made" / "prefix-cache";
  static_cast<void>(test::capturedStderr([&] { static_cast<void>(RuntimeResources::create(config, 0)); }));
  require(std::filesystem::is_directory(config.cacheRoot) && excluded(config.cacheRoot) &&
              !excluded(caches.path() / "made"),
          "the cache root the engine made was not excluded from backups, or its parent was");
  config.cacheRoot = caches.path() / "there";
  std::filesystem::create_directory(config.cacheRoot);
  static_cast<void>(test::capturedStderr([&] { static_cast<void>(RuntimeResources::create(config, 0)); }));
  require(!excluded(config.cacheRoot), "a cache root that was there was excluded from backups");
}

// The automatic tier keeps its staging only where the plan holds the
// requested context beside it. Asked for the context the plan holds without
// the staging but not with it, a start turns the tier off with one line and
// plans as without it.
void testAutomaticTierYieldsToTheContext(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-automatic");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "automatic-test");
  config.diskVolume = reading(localVolume(500 * kGiB));
  uint64_t minimumRequired = 0;
  const auto start = [&](uint32_t requestedContextTokens, uint32_t &contextTokens,
                         bool &tier) {
    return test::capturedStderr([&] {
      auto resources = RuntimeResources::create(config, requestedContextTokens);
      contextTokens = resources->memoryPlan().maximumContextTokens();
      minimumRequired = resources->memoryPlan().breakdown().minimumRequiredBytes;
      tier = resources->cache().snapshot().kvTier.capacityBytes != 0;
    });
  };
  uint32_t without = 0, with = 0, yielded = 0;
  bool tier = false;
  static_cast<void>(start(0, without, tier));
  // A budget whose KV the staging shortens: the plan's least, the staging
  // and a little KV.
  config.maximumMemoryBytes =
      minimumRequired + config.model.stateLayout.cachedBytes() + 64 * kMiB;
  static_cast<void>(start(0, without, tier));
  require(!tier && without, "a start without the SSD cache had one, or no context");
  config.cacheDisk = {.automatic = true};
  static_cast<void>(start(0, with, tier));
  require(tier && with < without,
          "the automatic tier did not start, or its staging cost the plan no context");
  const std::string off = start(without, yielded, tier);
  require(!tier && yielded == without &&
              off.find("SSD cache off: the memory plan cannot hold a " + std::to_string(without) +
                       "-token context beside its state staging") != std::string::npos,
          "the automatic tier kept staging the requested context needs");
}

// The automatic tier's staging is decided by memory alone, so a start serves
// the same context whatever room the volume its files would go to has: a
// relaunch after something else filled the disk announces the context the
// server serves. The volume decides only whether the files open. A volume
// without room, one that cannot be measured or one that is not local leaves
// the tier off with one line and the staging the plan set aside unused; a
// persistent cache whose root has no room keeps temporary files.
void testAutomaticContextIgnoresTheVolume(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-volume"), caches("splash-volume-root");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "volume-test");
  struct Started {
    uint32_t context = 0, planContext = 0;
    uint64_t staging = 0, minimumRequired = 0;
    bool tier = false, persistent = false;
    std::string log;
  };
  const auto start = [&] {
    Started started;
    started.log = test::capturedStderr([&] {
      auto resources = RuntimeResources::create(config, 0);
      started.context = resources->aneFfnOutcome().context;
      started.planContext = resources->memoryPlan().maximumContextTokens();
      started.staging = resources->memoryPlan().breakdown().stateStagingBytes;
      started.minimumRequired = resources->memoryPlan().breakdown().minimumRequiredBytes;
      const CacheSnapshot cache = resources->cache().snapshot();
      started.tier = cache.kvTier.capacityBytes != 0;
      started.persistent = cache.persistent;
    });
    return started;
  };
  const auto offOnce = [](const Started &started, std::string_view reason) {
    const size_t at = started.log.find("SSD cache off: ");
    return at != std::string::npos && started.log.find("SSD cache off", at + 1) == std::string::npos &&
           started.log.find(reason, at) != std::string::npos &&
           started.log.find("; the state staging the plan set aside stays unused.", at) !=
               std::string::npos;
  };
  // A budget whose KV the staging shortens: the plan's least, the staging
  // and a little KV.
  config.maximumMemoryBytes =
      start().minimumRequired + config.model.stateLayout.cachedBytes() + 64 * kMiB;
  const Started off = start();
  config.cacheDisk = {.automatic = true};
  config.diskVolume = reading(localVolume(500 * kGiB));
  const Started room = start();
  require(!off.tier && room.tier && room.staging == config.model.stateLayout.cachedBytes() &&
              room.context == room.planContext && room.context < off.context,
          "the automatic tier did not start beside its staging, or the staging cost no context");
  const std::array<std::pair<std::optional<DiskVolume>, std::string>, 3> refusals{{
      {localVolume(11 * kGiB), "11.0 GiB available on /Volumes/Test is too little beyond the 10.0 GiB"},
      {std::nullopt, "its volume cannot be measured"},
      {DiskVolume{"/Volumes/Share", 500 * kGiB, 1000 * kGiB, false}, "/Volumes/Share is not a local volume"},
  }};
  for (const auto &[volume, reason] : refusals) {
    config.diskVolume = reading(volume);
    const Started refused = start();
    require(!refused.tier && refused.staging == room.staging && refused.context == room.context &&
                refused.planContext == room.planContext && offOnce(refused, reason),
            "a volume without room changed the automatic context, or did not leave the tier off "
            "once (" + reason + ")");
  }
  // A persistent cache's root without room leaves the tier temporary files;
  // with no room for those either, it is off.
  config.persistentCache = true;
  config.cacheRoot = caches.path() / "prefix-cache";
  config.diskVolume = [&](const std::filesystem::path &directory) {
    return std::optional(localVolume(directory == config.cacheRoot ? 11 * kGiB : 500 * kGiB));
  };
  const Started temporary = start();
  require(temporary.tier && !temporary.persistent && temporary.context == room.context &&
              temporary.log.find("Persistent cache off: 11.0 GiB available on /Volumes/Test") !=
                  std::string::npos &&
              temporary.log.find("SSD cache off") == std::string::npos,
          "a persistent cache whose root had no room did not keep temporary files");
  config.diskVolume = reading(localVolume(11 * kGiB));
  const Started neither = start();
  require(!neither.tier && neither.context == room.context &&
              offOnce(neither, "for its root, 11.0 GiB available on /Volumes/Test"),
          "a persistent cache with no room anywhere did not leave the tier off once");
}

// Exercise the production assembly and runtime with each optional allocation.
// A category's fixed plan must cover the enabled buffers before KV growth.
void testOptionalFeatureMemoryPlan(const char *metallibPath) {
  const test::TemporaryDirectory root("optional-memory-plan");
  const PlannedModel planned = plannedModel();
  static_cast<void>(test::writeSyntheticModel(root.path(), planned.target, planned.draft, planned.vision));
  const auto rankingPath = root.path() / "public.splash";
  const uint32_t vocabulary = planned.target.vocabularySize;
  std::vector<uint8_t> ranking(12 + uint64_t{vocabulary} * 4);
  std::memcpy(ranking.data(), "SPLCHD02", 8);
  std::memcpy(ranking.data() + 8, &vocabulary, 4);
  for (uint32_t id = 0; id < vocabulary; ++id)
    std::memcpy(ranking.data() + 12 + uint64_t{id} * 4, &id, 4);
  test::writeFile(rankingPath, ranking);
  for (uint32_t switches = 0; switches < 4; ++switches) {
    RuntimeResourcesConfig config;
    config.metallibPath = metallibPath;
    config.modelRoot = root.path();
    config.model = model::makeModelDescriptor("optional-memory-test", planned.target, planned.draft,
        planned.vision, model::TargetSource::Safetensors, model::VisionSource::Safetensors);
    config.model.sourceIdentity = "synthetic";
    config.buildId = "optional-memory-test";
    config.maximumMemoryBytes = 4ULL << 30;
    config.aneFfn.enabled = false;
    config.hostMemory = [] { return std::optional<HostMemory>(HostMemory{64ULL << 30, 64ULL << 30}); };
    config.optionalFeatures.compactDraftHead = switches & 1;
    config.optionalFeatures.fastTopK = switches & 2;
    config.optionalFeatures.compactCapacity = 512;
    config.optionalFeatures.compactRanking = switches & 1 ? rankingPath : root.path() / "missing-unused";
    auto resources = RuntimeResources::create(config, 0);
    auto runtime = model::createRuntime(resources->modelContext());
    const auto actual = runtime->actualRuntimeMemory();
    const uint64_t reserved = resources->memoryPlan().breakdown().sharedDecodeBytes;
    require(actual.sharedDecodeActualAllocatedBytes <= reserved,
        "shared decode actual exceeds plan with optional switches=" + std::to_string(switches) +
        ": actual=" + std::to_string(actual.sharedDecodeActualAllocatedBytes) +
        ", planned=" + std::to_string(reserved));
  }
}

// A block-sparse target's context past what memory holds spills to the SSD
// cache, so where the automatic cache's volume has no room for it, a first
// start serves only what memory holds. A relaunch keeps its first engine's
// context whatever room the volume has left since: its spilling plan opens
// the cache's temporary files however little room their volume has.
void testRelaunchKeepsTheSpilledContext(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-relaunch");
  RuntimeResourcesConfig config = flashNextConfig(metallibPath, root.path());
  config.cacheDisk = {.automatic = true};
  config.diskVolume = reading(localVolume(500 * kGiB));
  // A budget whose memory holds less of the window than spilling does.
  config.maximumMemoryBytes = startStreamed(config).minimumRequired + 512 * kMiB;
  const StreamedStart first = startStreamed(config);
  config.diskVolume = reading(localVolume(11 * kGiB));
  const StreamedStart refused = startStreamed(config);
  require(first.tier && !refused.tier && refused.context < first.context &&
              refused.log.find("SSD cache off: 11.0 GiB available on /Volumes/Test") != std::string::npos,
          "a block-sparse target did not spill past what its memory holds, or spilled to an SSD cache "
          "whose volume has no room for it");
  config.relaunchContext = first.context;
  const StreamedStart relaunch = startStreamed(config);
  require(relaunch.tier && relaunch.context >= first.context,
          "a relaunch served less than its first engine's context for want of SSD room");
}

// The engine asks the governor whether the host pauses growth, and its
// serving mark lets a request in service grow through that pause.
void testEngineFollowsTheGovernor(const char *metallibPath) {
  metal::MetalBackend backend(metallibPath);
  constexpr uint64_t kGiB = 1ULL << 30;
  constexpr uint64_t hostReserve = 2 * kGiB;
  std::optional<uint64_t> available = hostReserve + 8 * kGiB;
  const metal::MetalMemoryStats memory = backend.memoryStats();
  MemoryGovernor governor(
      backend, std::max(memory.allocatedBytes, memory.deviceCurrentAllocatedBytes) + kGiB,
      hostReserve, hostReserve,
      [&available] { return std::optional<HostMemory>(HostMemory{*available, *available}); }, 0);
  EngineConfig config;
  connectToGovernor(config, governor);
  require(config.growthPaused && config.serving && !config.growthPaused(),
          "the engine was not connected to the governor");
  const auto admits = [admit = governor.allocationAdmission()] {
    return static_cast<bool>(admit(1024, [] {}));
  };
  // Inside the warning margin the host pauses growth that no request in
  // service needs.
  available = hostReserve + kGiB / 2;
  require(config.growthPaused() && !admits(), "the engine did not see the host's pause");
  config.serving(true);
  require(admits(), "the serving mark did not reach the governor");
  config.serving(false);
  require(!admits(), "the serving mark was not cleared");
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      require(argc == 2 || (argc == 3 && (std::string_view(argv[2]) == "--weight-budget" ||
          std::string_view(argv[2]) == "--local-ple" ||
          std::string_view(argv[2]) == "--local-ple-pressure" ||
          std::string_view(argv[2]) == "--local-ple-pressure-status" ||
          std::string_view(argv[2]) == "--local-ple-pressure-static" ||
          std::string_view(argv[2]) == "--local-ple-pressure-warning" ||
          std::string_view(argv[2]) == "--local-ple-pressure-manual" ||
          std::string_view(argv[2]) == "--local-ple-pressure-resident" ||
          std::string_view(argv[2]) == "--local-ple-pressure-off" ||
          std::string_view(argv[2]) == "--local-ple-pressure-partial" ||
          std::string_view(argv[2]) == "--local-writer" ||
          std::string_view(argv[2]) == "--local-critical" ||
          std::string_view(argv[2]) == "--local-release" ||
          std::string_view(argv[2]) == "--local-current-restore" ||
          std::string_view(argv[2]) == "--local-current-restore-ple" ||
          std::string_view(argv[2]) == "--local-shrink" ||
          std::string_view(argv[2]) == "--expert-metadata" ||
          std::string_view(argv[2]) == "--expert-storage" ||
          std::string_view(argv[2]) == "--optional-memory")),
          "expected metallib path and a known test selector");
      if (argc == 3 && std::string_view(argv[2]) == "--expert-metadata") {
        testExpertStorageMetadata(argv[1]);
        std::cout << "expert storage CPU metadata tests passed\n";
        return EXIT_SUCCESS;
      }
      if (argc == 3 && std::string_view(argv[2]) == "--weight-budget") {
        testWeightBudgetBeforeLoading(argv[1]);
        return EXIT_SUCCESS;
      }
      if (argc == 3 && std::string_view(argv[2]) == "--local-ple") {
        testLocalPleChunkCriticalAndActual(argv[1]);
        return EXIT_SUCCESS;
      }
      if (argc == 3 && std::string_view(argv[2]) == "--local-ple-pressure") {
        testLocalPlePressureTotalStorageOnly(argv[1]);
        return EXIT_SUCCESS;
      }
      if (argc == 3 && std::string_view(argv[2]) == "--local-ple-pressure-status") {
        testLocalPlePressureTotalStorageOnly(argv[1], true);
        return EXIT_SUCCESS;
      }
      if (argc == 3 && std::string_view(argv[2]) == "--local-ple-pressure-static") {
        testLocalPlePressureTotalStorageOnly(argv[1], true, false);
        return EXIT_SUCCESS;
      }
      if (argc == 3 && std::string_view(argv[2]) == "--local-ple-pressure-warning") {
        testLocalPlePressureTotalStorageOnly(argv[1], true, true, true);
        return EXIT_SUCCESS;
      }
      if (argc == 3 && std::string_view(argv[2]) == "--local-ple-pressure-manual") {
        testFixedPlePressureStorageOnly(argv[1], false);
        return EXIT_SUCCESS;
      }
      if (argc == 3 && std::string_view(argv[2]) == "--local-ple-pressure-resident") {
        testFixedPlePressureStorageOnly(argv[1], true);
        return EXIT_SUCCESS;
      }
      if (argc == 3 && std::string_view(argv[2]) == "--local-ple-pressure-off") {
        testPlePressureDisabledStorageOnly(argv[1]);
        return EXIT_SUCCESS;
      }
      if (argc == 3 && std::string_view(argv[2]) == "--local-ple-pressure-partial") {
        testPartialPlePressureStorageOnly(argv[1]);
        return EXIT_SUCCESS;
      }
      if (argc == 3 && std::string_view(argv[2]) == "--local-writer") {
        testImagePayloadIsPreparedByOwner(argv[1]);
        return EXIT_SUCCESS;
      }
      if (argc == 3 && std::string_view(argv[2]) == "--local-critical") {
        testCriticalStopsHostWorkspaceBeforeAllocation(argv[1]);
        return EXIT_SUCCESS;
      }
      if (argc == 3 && std::string_view(argv[2]) == "--local-release") {
        testLocalReleaseInvalidatesModelReadiness(argv[1]);
        return EXIT_SUCCESS;
      }
      if (argc == 3 && std::string_view(argv[2]) == "--local-current-restore") {
        testLocalReleasedCurrentLayout(argv[1], false);
        return EXIT_SUCCESS;
      }
      if (argc == 3 && std::string_view(argv[2]) == "--local-current-restore-ple") {
        testLocalReleasedCurrentLayout(argv[1], true); testLocalReleasedCurrentLayout(argv[1], true, true);
        return EXIT_SUCCESS;
      }
      if (argc == 3 && std::string_view(argv[2]) == "--local-shrink") {
        testNoPlePressureReturnsExpertChunks(argv[1]);
        return EXIT_SUCCESS;
      }
      if (argc == 3 && std::string_view(argv[2]) == "--expert-storage") {
        testExpertStorageMetadata(argv[1]);
        testExpertStorageLifecycle(argv[1]);
        testMixedExpertStorage(argv[1]);
        std::cout << "expert storage tests passed\n";
        return EXIT_SUCCESS;
      }
      if (argc == 3) {
        testOptionalFeatureMemoryPlan(argv[1]);
        std::cout << "optional feature memory plan tests passed\n";
        return EXIT_SUCCESS;
      }
      testExpertStorageMetadata(argv[1]);
      testExpertStorageLifecycle(argv[1]);
      testMixedExpertStorage(argv[1]);
      testLoadedDraftIsRequiredOnlyWithDraft();
      testLoadedVisionIsRequiredOnlyWithVision();
      testPersistentCacheNamespace();
      testWeightBudgetBeforeLoading(argv[1]);
      testStateStagingNeedsAStartedTier(argv[1]);
      testImagePatchCapIsBounded(argv[1]);
      testHostMemoryProbeIsRequired(argv[1]);
      testModelBeyondBudgetIsRefusedBeforeLoading(argv[1]);
      testStartupAdmissionIgnoresModelSize(argv[1]);
      testFailedStepIsNamed(argv[1]);
      testStartGrowsThroughHostPressure(argv[1]);
      testYarnRuntime(argv[1]);
      testWriteBudgetOnlyWhenSet(argv[1]);
      testStaleCachesGoAtEveryStart(argv[1]);
      testMadeCacheRootStaysOutOfBackups(argv[1]);
      testAutomaticTierYieldsToTheContext(argv[1]);
      testAutomaticContextIgnoresTheVolume(argv[1]);
      testRelaunchKeepsTheSpilledContext(argv[1]);
      testEngineFollowsTheGovernor(argv[1]);
      testOptionalFeatureMemoryPlan(argv[1]);
      {
        const test::TemporaryDirectory root("splash-moe");
        const RuntimeResourcesConfig moe = moeConfig(argv[1], root.path());
        testStreamableWeightsAreWhatLoads(moe);
        testAutomaticExpertMemory(moe);
        testVerifyRows(argv[1], moe);
      }
      std::cout << "runtime resources tests passed\n";
      return EXIT_SUCCESS;
    } catch (const std::exception &error) {
      std::cerr << "runtime resources tests failed: " << error.what() << '\n';
      return EXIT_FAILURE;
    }
  }
}
