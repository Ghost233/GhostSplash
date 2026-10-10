#include "TestChecks.hpp"
#include "model/ModelFactory.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/RuntimeArenas.hpp"

#include "metal/abi/QuantFormat.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

using namespace splash;

using splash::test::rejects;
using splash::test::require;

template <class Weights>
model::LoadedModel package() {
  model::LoadedModel result;
  Weights target;
  constexpr bool targetOnly = std::is_same_v<Weights, model::Qwen3_5_4BWeights>;
  const model::DFlashDraftLayout draft = targetOnly ? model::DFlashDraftLayout{}
                                      : std::is_same_v<Weights, model::Qwen3_6MoeWeights>
                                             ? model::kQwen3_6MoeDraftLayout
                                             : model::kQwen3_8DraftLayout;
  const auto projection = [](uint32_t n, uint32_t k) {
    if constexpr (targetOnly)
      return ops::Projection(n, k, ops::BlockWeights{{ops::QuantizedSegment::planes(GGUF_FMT_Q4K, n, k, {}, {}, {})}});
    else return ops::Projection(n, k, ops::AffineWeights{});
  };
  const auto &layout = target.layout;
  target.logitsProjection = projection(layout.vocabularySize, layout.hiddenSize);
  target.layers.resize(layout.layers);
  for (uint32_t i = 0; i < layout.layers; ++i) {
    auto &layer = target.layers[i];
    if (layout.isFullAttentionLayer(i)) {
      model::QwenAttentionWeights attention;
      attention.inputProjection = projection(layout.packedFullWidth, layout.hiddenSize);
      attention.outputProjection = projection(layout.hiddenSize, layout.attentionWidth);
      layer.mixer = std::move(attention);
    } else {
      model::QwenGdnWeights gdn;
      gdn.inputProjection = projection(layout.packedGdnWidth, layout.hiddenSize);
      gdn.outputProjection = projection(layout.hiddenSize, layout.attentionWidth);
      layer.mixer = std::move(gdn);
    }
    if constexpr (std::is_same_v<Weights, model::Qwen3_8Weights> || targetOnly) {
      layer.gateProjection = projection(layout.intermediateSize, layout.hiddenSize);
      layer.upProjection = layer.gateProjection;
      layer.downProjection = projection(layout.hiddenSize, layout.intermediateSize);
    }
  }
  ops::VisionLayout vision;
  vision.outputHiddenSize = targetOnly ? 0 : target.layout.hiddenSize;
  result.descriptor = model::makeModelDescriptor(
      "operator workspace test", target.layout, draft, vision,
      targetOnly ? model::TargetSource::Gguf : model::TargetSource::Package,
      targetOnly ? model::VisionSource::None : model::VisionSource::Package);
  result.target = std::move(target);
  result.draft.layout = draft;
  return result;
}

// The 4B has no draft. Sizing its arenas must never plan a zero-width
// affine draft projection, and every scratch bound must come from the target.
void checkTargetOnlyWorkspaces() {
  const auto loaded = package<model::Qwen3_5_4BWeights>();
  require(loaded.descriptor.valid() && !loaded.descriptor.hasDraft(), "invalid target-only fixture");
  model::requireCompatibleModel(loaded);
  const auto geometry = model::RuntimeGeometry::from(loaded, kv::Format::Int8);
  for (uint32_t family : {9U, 10U, 11U})
    for (uint32_t cores : {16U, 20U, 40U}) {
      DeviceCapabilities device;
      device.appleGpuFamily = family;
      device.gpuCoreCount = cores;
      const ops::ExecutionPlans plans(device);
      ops::LinearScratchSize expected;
      for (const auto &projection : geometry.target.decodeProjections)
        expected.include(plans.linear().decodeScratchSize(projection));
      const auto actual = model::DecodeArena::linearScratchSize(geometry, plans);
      require(actual.partials == expected.partials && actual.counters == expected.counters &&
                  actual.rotated == expected.rotated,
              "target-only linear scratch included nonexistent draft work");
      uint64_t gateBytes = 0;
      for (const auto &projection : geometry.target.gateUpProjections)
        gateBytes = std::max(gateBytes, plans.gateUpWorkspace(projection));
      require(model::DecodeArena::gateScratchBytes(geometry, plans) == gateBytes,
              "target-only gate scratch included nonexistent draft work");
      require(model::plannedPrefillBytes(geometry, plans) > 0 &&
                  model::plannedDecodeBytes(geometry, plans) > 0,
              "target-only arena planning omitted real target work");
    }
}

void checkMixedLayouts() {
  auto mixed = package<model::Qwen3_8Weights>();
  auto &target = std::get<model::Qwen3_8Weights>(mixed.target);
  auto &up = target.layers.front().upProjection;
  up = ops::Projection(up.outputSize, up.inputSize,
                       ops::BlockWeights{{ops::QuantizedSegment::planes(GGUF_FMT_Q4K, up.outputSize,
                                                                        up.inputSize, {}, {}, {})}});
  rejects([&] { static_cast<void>(model::qwenTargetGeometry(target)); },
          "fused gate/up projections must have matching shapes and layouts",
          "incompatible fused gate/up layouts reached execution");
  target.layers.front().gateProjection = up;
  require(target.logitsProjection.layout() == ops::WeightLayout::Affine64,
          "mixed fixture must keep an affine vocabulary head");
  for (uint32_t family : {9U, 10U, 11U}) {
    DeviceCapabilities device;
    device.appleGpuFamily = family;
    device.gpuCoreCount = 16;
    ops::ExecutionPlans plans(device);
    const auto geometry = model::RuntimeGeometry::from(mixed, kv::Format::Int8);
    const auto head = target.logitsProjection.shape();
    const auto containsHead = [&](const auto &shapes) {
      return std::find(shapes.begin(), shapes.end(), head) != shapes.end();
    };
    require(!containsHead(geometry.target.prefillProjections) &&
                containsHead(geometry.target.decodeProjections),
            "vocabulary head must reserve workspace only in decode");
    const auto scratch = model::DecodeArena::linearScratchSize(geometry, plans);
    for (uint32_t lanes = 1; lanes <= model::kLaneCount; ++lanes) {
      const auto plan = plans.linear().plan({{up.outputSize, up.inputSize}, lanes * model::kDecodeRows,
          ops::LinearPhase::Decode, ops::LinearEpilogue::GateUp}, up);
      const auto required = plan.scratchSize();
      require(scratch.input >= required.input && scratch.sums >= required.sums &&
                  scratch.partials >= required.partials && scratch.counters >= required.counters,
              "affine head hid a block-quantized layer's scratch requirement");
      require(model::DecodeArena::gateScratchBytes(geometry, plans) >= plan.gateScratchBytes(),
              "mixed gate/up workspace is too small");
    }
    const auto sizes = model::prefillTensorBytes(geometry, plans);
    for (uint32_t rows : {1U, 8U, 17U, 32U}) {
      const auto required = plans.linear().plan({{up.outputSize, up.inputSize}, rows,
          ops::LinearPhase::Prefill, ops::LinearEpilogue::None}, up).scratchSize();
      require(sizes[uint32_t(model::PrefillTensor::LinearPartials)] >= required.partials &&
                  sizes[uint32_t(model::PrefillTensor::LinearCounters)] >= required.counters,
              "mixed short-prefill split scratch is too small");
    }
  }
  // Every MoE block of a target shares one layout, which the geometry's one
  // MoE shape records: no source mixes them.
  auto sparse = package<model::Qwen3_6MoeWeights>();
  auto &moe = std::get<model::Qwen3_6MoeWeights>(sparse.target);
  require(model::qwenTargetGeometry(moe).moeShape().weightLayout == ops::WeightLayout::Affine64,
          "the MoE shape lost the blocks' layout");
  for (auto &layer : moe.layers) layer.ffn = ops::BlockMoeWeights{};
  require(model::qwenTargetGeometry(moe).moeShape().weightLayout == ops::WeightLayout::Block32 &&
              moe.logitsProjection.layout() == ops::WeightLayout::Affine64,
          "the MoE shape must follow the expert layers, not the head");
  moe.layers.back().ffn = ops::AffineMoeWeights{};
  rejects([&] { static_cast<void>(model::qwenTargetGeometry(moe)); },
          "the MoE blocks of a target must share one weight layout",
          "a target mixing MoE layouts reached execution");
}

// One decode arena serves every lane count, and on Apple10 and later a
// Split128 plan's partials grow with the rows. The arena must hold every
// lane's plan of every affine target and draft projection at the measured
// core counts.
void checkStandardDraftScratch() {
  auto geometry = model::RuntimeGeometry::from(package<model::Qwen3_5_4BWeights>(), kv::Format::Int8);
  geometry.draft = {.layers=6, .hiddenSize=2560, .vocabularySize=248320, .dynamicSize=0,
      .qkvSize=6144, .attentionSize=4096, .intermediateSize=9216, .attentionHeadDimension=128,
      .rotaryTheta=1000000, .targetHiddenSize=20480, .selectorRank=0, .kvHeads=8,
      .kind=model::DraftKind::Standard};
  DeviceCapabilities device;
  device.appleGpuFamily = 10; device.gpuCoreCount = 20;
  const ops::ExecutionPlans plans(device);
  ops::LinearScratchSize expected;
  for (const auto &projection : geometry.target.decodeProjections)
    expected.include(plans.linear().decodeScratchSize(projection));
  for (const ops::ProjectionShape shape : {ops::ProjectionShape{6144,2560}, {2048,2560},
      {2560,4096}, {9216,2560}, {2560,9216}, {2560,20480}})
    expected.include(plans.linear().decodeScratchSize(shape));
  const auto actual = model::DecodeArena::linearScratchSize(geometry, plans);
  require(actual.input == expected.input && actual.sums == expected.sums && actual.partials == expected.partials &&
              actual.counters == expected.counters && actual.rotated == expected.rotated,
          "standard DFlash scratch planning included nonexistent dynamic/selector projections");
  const auto tensors = model::decodeTensorBytes(geometry, plans);
  require(tensors[static_cast<uint32_t>(model::DecodeTensor::DraftDynamic)] == 0 &&
              tensors[static_cast<uint32_t>(model::DecodeTensor::DraftConvolved)] == 0 &&
              tensors[static_cast<uint32_t>(model::DecodeTensor::SelectorHidden)] == 0,
          "standard DFlash arenas allocated fake dynamic/conv/selector buffers");
  geometry.draft.qkvSize = 0;
  rejects([&] { static_cast<void>(model::DecodeArena::linearScratchSize(geometry, plans)); },
          "invalid linear matrix", "missing required standard QKV matrix was silently skipped");
}

void checkLaneScratch(const model::LoadedModel &package) {
  const auto geometry = model::RuntimeGeometry::from(package, kv::Format::Int8);
  const auto &d = geometry.draft;
  std::vector<ops::LinearMatrix> matrices{
      {d.dynamicSize, d.hiddenSize}, {d.qkvSize, d.hiddenSize}, {d.contextKvSize(), d.hiddenSize},
      {d.hiddenSize, d.attentionSize},
      {d.intermediateSize, d.hiddenSize}, {d.hiddenSize, d.intermediateSize},
      {d.selectorRank, d.hiddenSize}, {d.hiddenSize, d.targetHiddenSize}};
  for (const auto &p : geometry.target.decodeProjections)
    if (p.layout == ops::WeightLayout::Affine64) matrices.push_back({p.outputSize, p.inputSize});
  for (uint32_t family : {10U, 11U})
    for (uint32_t cores : {12U, 20U, 40U}) {
      DeviceCapabilities device;
      device.appleGpuFamily = family;
      device.gpuCoreCount = cores;
      const ops::ExecutionPlans plans(device);
      const auto scratch = model::DecodeArena::linearScratchSize(geometry, plans);
      for (const auto matrix : matrices)
        for (uint32_t lanes = 1; lanes <= model::kLaneCount; ++lanes)
          for (auto epilogue : {ops::LinearEpilogue::None, ops::LinearEpilogue::Residual,
                                ops::LinearEpilogue::GateUp}) {
            const auto need = plans.linear().plan({matrix, lanes * model::kDecodeRows,
                ops::LinearPhase::Decode, epilogue}).scratchSize();
            require(scratch.partials >= need.partials && scratch.counters >= need.counters,
                    "decode arena scratch below a lane's affine plan");
          }
    }
}

// Arenas are sized from the projections the weights hold, so each must have
// sizes; an empty one would drop its workspace from the bound silently.
void checkUnsizedProjection() {
  auto broken = package<model::Qwen3_8Weights>();
  std::get<model::Qwen3_8Weights>(broken.target).layers.back().downProjection = ops::Projection();
  rejects([&] { static_cast<void>(model::RuntimeGeometry::from(broken, kv::Format::Int8)); },
          "invalid model runtime geometry", "a target projection without sizes reached arena sizing");
}

// The GDN value rows are sized with attentionWidth, so a layout whose value
// heads span another width is refused before loading. Arena sizing checks
// the GDN shape, whose packed rows must also hold the two gates of every
// value head.
void checkGdnWidths() {
  const auto sparse = package<model::Qwen3_6MoeWeights>();
  const auto sizeArenas = [&](const model::Qwen3_6MoeLayout &layout) {
    auto candidate = sparse;
    std::get<model::Qwen3_6MoeWeights>(candidate.target).layout = layout;
    static_cast<void>(model::RuntimeGeometry::from(candidate, kv::Format::Int8));
  };
  // The shipped sparse layout passes both checks.
  const model::Qwen3_6MoeLayout shipped;
  model::requireQwenLayout(shipped);
  sizeArenas(shipped);
  auto narrowValues = shipped;
  narrowValues.gdnValueHeads = narrowValues.gdnKeyHeads;
  narrowValues.convolutionDimension = 3 * narrowValues.gdnKeyHeads * narrowValues.gdnHeadDimension;
  rejects([&] { model::requireQwenLayout(narrowValues); }, "Qwen target layout is inconsistent",
          "a GDN value width other than attentionWidth was accepted");
  auto withoutGates = shipped;
  withoutGates.packedGdnWidth = shipped.convolutionDimension + shipped.attentionWidth;
  rejects([&] { sizeArenas(withoutGates); }, "invalid model runtime geometry",
          "packed GDN rows without the gates reached arena sizing");
}

} // namespace

int main() {
  try {
    checkUnsizedProjection();
    checkGdnWidths();
    checkMixedLayouts();
    checkTargetOnlyWorkspaces();
    const auto dense = package<model::Qwen3_8Weights>();
    const auto sparse = package<model::Qwen3_6MoeWeights>();
    checkStandardDraftScratch();
    checkLaneScratch(dense);
    checkLaneScratch(sparse);
    std::cout << "model execution plans: PASS (two paired geometries)\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
