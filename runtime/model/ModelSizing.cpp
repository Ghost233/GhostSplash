#include "model/ModelSizing.hpp"

#include "model/DraftCheckpoint.hpp"
#include "model/ExpertStore.hpp"
#include "model/GgufTarget.hpp"
#include "model/ModelFactory.hpp"
#include "model/SafetensorsImage.hpp"
#include "model/SparseRowStore.hpp"
#include "model/VisionLoader.hpp"

#include <span>
#include <algorithm>
#include <type_traits>
#include <vector>

namespace splash::model {
namespace {

uint64_t loadedBytes(std::span<const gguf::Image> images) {
  uint64_t bytes = 0;
  for (const gguf::Image &image : images)
    bytes += gguf::loadedBytes(image);
  return bytes;
}

// What a target of `layout` whose images `plan` plans for what streams
// takes, while their sources are open: what loads while every part `streams`
// lets stream does, and what each such part adds where it loads instead,
// the routed experts' slot table with them.
template <class Layout, class Plan>
StreamableWeights targetWeights(const Layout &layout, const Plan &plan, gguf::PlanOptions streams) {
  streams.streamExperts = streams.streamExperts && layout.ffnKind == QwenFfnKind::SparseMoe;
  StreamableWeights result;
  std::vector<gguf::Image> streamed;
  try {
    streamed = plan(streams);
  } catch (const GgufError &error) {
    if (!streams.streamTable)
      throw;
    // A table that cannot stream loads with the weights; whatever else
    // failed, the plan without it fails again.
    result.tableRefusal = error.what();
    streams.streamTable = false;
    streamed = plan(streams);
  }
  result.loadedBytes = loadedBytes(streamed);
  for (const gguf::Image &image : streamed)
    if (image.streamedRows) {
      result.tableBytes = loadedBytes(plan({.streamExperts = streams.streamExperts})) - result.loadedBytes;
      const SparseRowStoreOptions rows = streamedTableRows(layout);
      result.tableRowBytes = image.streamedRows->rowBytes;
      result.tableFloorBytes = SparseRowStore::workingBytes(image.streamedRows->rowBytes, rows.prefetchRows,
                                                            rows.stageRows) +
                               SparseRowStore::cacheBytesFor(1, image.streamedRows->rowBytes);
    }
  if (streams.streamExperts) {
    result.expertBytes = loadedBytes(plan({.streamTable = result.tableBytes != 0})) - result.loadedBytes +
                         residentExpertSlotsBytes(layout.experts);
    StreamedBlocks blocks;
    blocks.blocks = gguf::streamedExperts(std::span(streamed).first(layout.layers));
    result.expertFloorBytes = ExpertStore::minimumBytes(
        blocks, {layout.hiddenSize, layout.experts, layout.expertsPerToken, layout.expertIntermediateSize});
  }
  return result;
}

} // namespace

StreamableWeights streamableWeights(const std::filesystem::path &root, const ModelDescriptor &descriptor,
                                    gguf::PlanOptions streams, FfnLoadOptions ffn) {
  requireFfnLoadOptions(ffn);
  StreamableWeights result = std::visit(
      [&](const auto &layout) {
        const std::filesystem::path directory = root / "target";
        if (descriptor.targetSource == TargetSource::Gguf) {
          const OpenGguf target(findTargetGgufs(directory));
          const auto plan = [&](gguf::PlanOptions options) {
            return gguf::planImages(target.file(), layout.ggufArchitecture, layout, options);
          };
          StreamableWeights weights = targetWeights(layout, plan, streams);
          using Layout = std::remove_cvref_t<decltype(layout)>;
          if constexpr (std::is_same_v<Layout, Qwen3_5_4BLayout> ||
                        std::is_same_v<Layout, Qwen3_5_4BDFlashLayout> ||
                        std::is_same_v<Layout, Qwen3_5_0_8BLayout>) {
            if (ffn.mode != FfnWeightMode::Quantized) {
              const uint64_t limit = ffn.mode == FfnWeightMode::LimitedFp16 ? ffn.budgetBytes : UINT64_MAX;
              uint64_t cache = 0;
              for (const gguf::Image &image : plan({}))
                for (const gguf::Repack &repack : image.repacks)
                  if (std::ranges::any_of(repack.sources, [](const auto &source) {
                        return source.name.ends_with(".ffn_gate.weight") ||
                               source.name.ends_with(".ffn_up.weight") ||
                               source.name.ends_with(".ffn_down.weight");
                      })) {
                    const uint64_t bytes = repack.rows * repack.columns * sizeof(uint16_t);
                    if (bytes <= limit - cache) cache += alignUp(bytes);
                  }
              weights.loadedBytes += cache;
            }
          }
          return weights;
        }
        if constexpr (std::remove_cvref_t<decltype(layout)>::loadsSafetensors) {
          const SafetensorsCheckpoint checkpoint(directory);
          return targetWeights(
              layout, [&](gguf::PlanOptions options) { return safetensors::planImages(checkpoint, layout, options); },
              streams);
        } else {
          throw std::invalid_argument(std::string(layout.family) + " loads from a GGUF only");
        }
      },
      descriptor.target);
  if (descriptor.hasDraft()) result.loadedBytes += draftImageBytes(root / "draft", descriptor.draft);
  if (descriptor.hasVision()) result.loadedBytes += visionImageBytes(descriptor.vision);
  return result;
}

} // namespace splash::model
