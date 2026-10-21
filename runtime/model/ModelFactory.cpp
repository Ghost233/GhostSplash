#include "ModelFactory.hpp"
#include "AwakeClock.hpp"
#include "model/ModelSizing.hpp"
#include <chrono>
#include <optional>
#include <type_traits>
#include "model/DraftCheckpoint.hpp"
#include "model/GgufTarget.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/SafetensorsTarget.hpp"
#include "model/VisionLoader.hpp"

#include <functional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace splash::model {

void requireCompatibleModel(const LoadedModel &model) {
  if (!model.descriptor.valid() ||
      model.descriptor.draft != model.draft.layout ||
      !std::visit(
          [&](const auto &target) {
            return model.descriptor.target == TargetLayout{target.layout} &&
                   (!model.descriptor.hasDraft() || target.layout.vocabularySize ==
                       model.draft.layout.vocabularySize);
          },
          model.target)) {
    throw std::invalid_argument(
        "target and draft model interfaces are incompatible");
  }
}

std::unique_ptr<VisionLoader> planVisionLoader(const std::filesystem::path &root, const ModelDescriptor &descriptor) {
  if (descriptor.visionSource != VisionSource::Safetensors && descriptor.visionSource != VisionSource::Gguf)
    return nullptr;
  return std::make_unique<VisionLoader>(root / "vision", descriptor.visionSource, descriptor.vision);
}

QwenVisionWeights loadVisionWeights(metal::MetalBackend &backend, WeightImages &images,
                                    const VisionLoader *loader) {
  return loader ? loadQwenVisionWeights(backend, images, *loader) : QwenVisionWeights{};
}

namespace {

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen3_8Layout &layout,
                         const QwenTargetFiles &files) {
  return loadQwen3_8Weights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen3_6MoeLayout &layout,
                         const QwenTargetFiles &files) {
  return loadQwen3_6MoeWeights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen4ExpLayout &layout,
                         const QwenTargetFiles &files) {
  return loadQwen4ExpWeights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen3_5_0_8BLayout &layout, const QwenTargetFiles &files) {
  return loadQwen3_5_0_8BWeights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen3_5_4BLayout &layout,
                         const QwenTargetFiles &files) {
  return loadQwen3_5_4BWeights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen3_5_4BDFlashLayout &layout,
                         const QwenTargetFiles &files) {
  return loadQwen3_5_4BDFlashWeights(backend, layout, files);
}

} // namespace

SparseRowStoreOptions streamedTableRows(const Qwen3_8Layout &) { return {}; }

SparseRowStoreOptions streamedTableRows(const Qwen3_6MoeLayout &) { return {}; }
SparseRowStoreOptions streamedTableRows(const Qwen3_5_4BLayout &) { return {}; }
SparseRowStoreOptions streamedTableRows(const Qwen3_5_4BDFlashLayout &) { return {}; }
SparseRowStoreOptions streamedTableRows(const Qwen3_5_0_8BLayout &) { return {}; }

StreamedParts StreamedParts::loaded(const LoadedModel &model, bool kv) noexcept {
  const SparseRowStore *table = model.streamedTable();
  return {.experts = model.expertStore() != nullptr, .tableRowBytes = table ? table->table().rowBytes : 0, .kv = kv};
}

LoadedModel loadModel(metal::MetalBackend &backend,
                              const std::filesystem::path &root,
                              const ModelDescriptor &descriptor,
                              const ModelLoadOptions &options) {
  LoadedModel result;
  result.descriptor = descriptor;
  const auto started = AwakeClock::now();
  const FfnLoadOptions &ffn = options.ffn;
  requireFfnLoadOptions(ffn);
  result.ffnOptions = ffn;
  if ((ffn.mode != FfnWeightMode::FullFp16 || ffn.variant != ops::FfnPrefillVariant::Baseline) &&
      !std::holds_alternative<Qwen3_5_0_8BLayout>(descriptor.target) &&
      !std::holds_alternative<Qwen3_5_4BLayout>(descriptor.target) &&
      !std::holds_alternative<Qwen3_5_4BDFlashLayout>(descriptor.target))
    throw std::invalid_argument("quantized FFN experiment requires Qwen3.5-4B");
  if (!result.descriptor.valid())
    throw std::invalid_argument("model descriptor is invalid");
  result.images = std::make_shared<WeightImages>(backend, result.descriptor.sourceIdentity);
  WeightImages &images = *result.images;
  // Every source's metadata is checked before the first image is written:
  // the vision tower's and the draft's here, the target's by its loader.
  const auto vision = planVisionLoader(root, result.descriptor);
  std::optional<DraftCheckpointLoader> draft;
  if (result.descriptor.hasDraft())
    draft.emplace(backend, images, root / "draft", result.descriptor.draft);
  // The weights that stream from their files read on the engine's readers.
  const ExpertStoreOptions experts{.memoryBytes = options.expertMemoryBytes, .readers = options.readers};
  result.target = std::visit(
      [&](const auto &layout) -> TargetWeights {
        const std::filesystem::path directory = root / "target";
        switch (result.descriptor.targetSource) {
        case TargetSource::Safetensors: {
          if constexpr (std::remove_cvref_t<decltype(layout)>::loadsSafetensors) {
            SafetensorsTargetLoader loader(backend, images, directory, layout, experts);
            return readTarget(backend, layout, std::ref(loader));
          } else {
            throw std::invalid_argument(std::string(layout.family) + " loads from a GGUF only");
          }
        }
        case TargetSource::Gguf: {
          // A streamed table's store: its memory and readers here, the rows
          // a step stages and prefetches from the family.
          SparseRowStoreOptions table = streamedTableRows(layout);
          table.cacheBytes = options.streamedTableBytes;
          table.readers = options.readers;
          GgufTargetLoader loader(backend, images, findTargetGgufs(directory), layout.ggufArchitecture, layout, table,
                                  experts);
          return readTarget(backend, layout, std::ref(loader));
        }
        }
        throw std::invalid_argument("unknown target source");
      },
      result.descriptor.target);
  if (draft) result.draft = loadDFlashDraftWeights(backend, *draft, result.descriptor.draft);
  result.vision = loadVisionWeights(backend, images, vision.get());

  const auto loadedAt = AwakeClock::now();
  result.loadTimings.sourceSeconds = std::chrono::duration<double>(loadedAt - started).count();
  const auto policy = [&](auto &target) {
    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(target)>, Qwen3_5_0_8BWeights> || std::is_same_v<std::remove_cvref_t<decltype(target)>, Qwen3_5_4BWeights> ||
                  std::is_same_v<std::remove_cvref_t<decltype(target)>, Qwen3_5_4BDFlashWeights>)
      for (auto &layer : target.layers) {
        layer.gateProjection.ffn = {ops::FfnRole::Gate, ffn.variant, ffn.trace};
        layer.upProjection.ffn = {ops::FfnRole::Up, ffn.variant, ffn.trace};
        layer.downProjection.ffn = {ops::FfnRole::Down, ffn.variant, ffn.trace};
      }
  };
  std::visit(policy, result.target);
  if (ffn.mode != FfnWeightMode::Quantized) {
    if (auto *target = std::get_if<Qwen3_5_0_8BWeights>(&result.target))
      buildQwen3_5_0_8BPrefillCache(backend, images, *target, ffn.mode == FfnWeightMode::LimitedFp16 ? ffn.budgetBytes : UINT64_MAX);
    if (auto *target = std::get_if<Qwen3_5_4BWeights>(&result.target))
      buildQwen3_5_4BPrefillCache(backend, images, *target, ffn.mode == FfnWeightMode::LimitedFp16 ? ffn.budgetBytes : UINT64_MAX);
    if (auto *target = std::get_if<Qwen3_5_4BDFlashWeights>(&result.target))
      buildQwen3_5_4BPrefillCache(backend, images, *target, ffn.mode == FfnWeightMode::LimitedFp16 ? ffn.budgetBytes : UINT64_MAX);
  }

  result.loadTimings.ffnCacheSeconds = std::chrono::duration<double>(AwakeClock::now() - loadedAt).count();
  std::vector<WeightFileRecord> records(result.targetFiles().begin(),
                                        result.targetFiles().end());
  records.insert(records.end(), result.draft.files.begin(),
                 result.draft.files.end());
  records.insert(records.end(), result.vision.files.begin(),
                 result.vision.files.end());
  result.manifestFingerprintSha256 = weightManifestFingerprint(records);
  requireCompatibleModel(result);
  return result;
}

uint64_t modelWeightBytes(const std::filesystem::path &root, const ModelDescriptor &descriptor, FfnLoadOptions ffn) {
  const StreamableWeights weights =
      streamableWeights(root, descriptor, {.streamExperts = true, .streamTable = true}, ffn);
  return weights.loadedBytes + weights.expertBytes + weights.tableBytes;
}

} // namespace splash::model
