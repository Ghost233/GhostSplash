#include "TestChecks.hpp"
#include "model/ModelFactory.hpp"
#include "model/KevHead.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/RuntimeArenas.hpp"
#include "model/CompactDraftHead.hpp"
#include "ops/Sampling.hpp"

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
model::LoadedModel loadedModel() {
  model::LoadedModel result;
  Weights target;
  constexpr bool targetOnly = std::is_same_v<Weights, model::Qwen3_5_4BWeights>;
  const model::DFlashDraftLayout draft = targetOnly ? model::DFlashDraftLayout{}
                                      : std::is_same_v<Weights, model::Qwen3_6MoeWeights>
                                             ? model::kQwen3_6MoeDraftLayout
                                             : model::kQwen3_8DraftLayout;
  const auto projection = [](uint32_t n, uint32_t k) {
    return ops::Projection(n, k, ops::BlockWeights{{ops::QuantizedSegment::planes(GGUF_FMT_Q4K, n, k, {}, {}, {})}});
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
      targetOnly ? model::TargetSource::Gguf : model::TargetSource::Safetensors,
      targetOnly ? model::VisionSource::None : model::VisionSource::Safetensors);
  result.target = std::move(target);
  result.draft.layout = draft;
  return result;
}

// MLX affine 4-bit uses packed values plus BF16 scale/offset pairs per 64 inputs.
// The selected vocabulary copies and fast top-k must be reserved independently.
void checkOptionalFeatureMemoryPlan() {
  auto loaded = loadedModel<model::Qwen3_8Weights>();
  auto &target = std::get<model::Qwen3_8Weights>(loaded.target);
  target.logitsProjection = ops::Projection(248320, 5120,
      ops::BlockWeights{{ops::QuantizedSegment::planes(GGUF_FMT_AF4G64, 248320, 5120, {}, {}, {})}});
  target.logitsProjection.destination = ops::FloatOutput::Float32;
  DeviceCapabilities device;
  device.appleGpuFamily = 10;
  device.gpuCoreCount = 20;
  const ops::ExecutionPlans operators(device);
  const auto baseline = model::plannedRuntimeMemory(loaded, operators, kv::Format::Int4Hadamard);
  const uint64_t fastBytes = alignUp(ops::Sampling::fastWorkspaceBytes(model::kLaneCount * model::kDecodeRows));
  for (uint32_t capacity : {98304U, 65536U, 32768U})
    for (uint32_t switches = 0; switches < 4; ++switches) {
      model::OptionalFeatures features;
      features.compactDraftHead = switches & 1;
      features.fastTopK = switches & 2;
      features.compactCapacity = capacity;
      features.compactRanking = "/unused-by-size-planning";
      const uint64_t compactBytes = alignUp(uint64_t{capacity} * 5120 / 2) +
          alignUp(uint64_t{capacity} * (5120 / 64) * 4) + alignUp(uint64_t{capacity} * 4);
      const auto planned = model::plannedRuntimeMemory(loaded, operators, kv::Format::Int4Hadamard, features);
      require(planned.sharedPrefillPlannedAllocatedBytes == baseline.sharedPrefillPlannedAllocatedBytes &&
                  planned.laneStatePlannedAllocatedBytes == baseline.laneStatePlannedAllocatedBytes,
              "optional buffers changed unrelated memory categories");
      require(planned.sharedDecodePlannedAllocatedBytes == baseline.sharedDecodePlannedAllocatedBytes +
                  (features.compactDraftHead ? compactBytes : 0) + (features.fastTopK ? fastBytes : 0),
              "optional buffers are missing or duplicated in the fixed decode budget");
      if (capacity == 98304 && switches == 3 && model::kDecodeRows == 8)
        require(planned.sharedDecodePlannedAllocatedBytes - baseline.sharedDecodePlannedAllocatedBytes == 283639808,
                "27B optional memory does not match the real 270.5 MiB regression");
    }
  auto smallHead = ops::Projection(1024, 512,
      ops::BlockWeights{{ops::QuantizedSegment::planes(GGUF_FMT_AF4G64, 1024, 512, {}, {}, {})}});
  smallHead.destination = ops::FloatOutput::Float32;
  require(model::CompactDraftHead::plannedBytes(smallHead) ==
              alignUp(512ULL * 512 / 2) + alignUp(512ULL * (512 / 64) * 4) + alignUp(512ULL * 4),
          "small vocabulary cap or token-map allocation alignment was omitted");
}

// The 4B has no draft. Sizing its arenas must never plan a zero-width
// affine draft projection, and every scratch bound must come from the target.
void checkTargetOnlyWorkspaces() {
  const auto loaded = loadedModel<model::Qwen3_5_4BWeights>();
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

// A layer's gate and up projections run as one gate/up plan, so their shapes
// must match. The vocabulary head reserves workspace only in decode, and the
// arenas hold a gate/up layer's decode and short-prefill plans.
void checkGateUpLayers() {
  auto dense = loadedModel<model::Qwen3_8Weights>();
  auto &target = std::get<model::Qwen3_8Weights>(dense.target);
  const ops::Projection up = target.layers.front().upProjection;
  target.layers.front().upProjection = ops::Projection(
      up.outputSize, up.inputSize + 256,
      ops::BlockWeights{{ops::QuantizedSegment::planes(GGUF_FMT_Q4K, up.outputSize, up.inputSize + 256, {}, {}, {})}});
  rejects([&] { static_cast<void>(model::qwenTargetGeometry(target)); },
          "fused gate/up projections must have matching shapes", "mismatched fused gate/up shapes reached execution");
  target.layers.front().upProjection = up;
  for (uint32_t family : {9U, 10U, 11U}) {
    DeviceCapabilities device;
    device.appleGpuFamily = family;
    device.gpuCoreCount = 16;
    ops::ExecutionPlans plans(device);
    const auto geometry = model::RuntimeGeometry::from(dense, kv::Format::Int8);
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
              "the decode arena is below a gate/up layer's scratch");
      require(model::DecodeArena::gateScratchBytes(geometry, plans) >= plan.gateScratchBytes(),
              "the gate/up workspace is too small");
    }
    const auto sizes = model::prefillTensorBytes(geometry, plans);
    for (uint32_t rows : {1U, 8U, 17U, 32U}) {
      const auto required = plans.linear().plan({{up.outputSize, up.inputSize}, rows,
          ops::LinearPhase::Prefill, ops::LinearEpilogue::None}, up).scratchSize();
      require(sizes[uint32_t(model::PrefillTensor::LinearPartials)] >= required.partials &&
                  sizes[uint32_t(model::PrefillTensor::LinearCounters)] >= required.counters,
              "the short-prefill split scratch is too small");
    }
  }
}

// One decode arena serves every lane count, and on Apple10 and later a
// Split128 plan's partials grow with the rows. The arena must hold every
// lane's plan of every affine target and draft projection at the measured
// core counts.
void checkStandardDraftScratch() {
  auto geometry = model::RuntimeGeometry::from(loadedModel<model::Qwen3_5_4BWeights>(), kv::Format::Int8);
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
  for (const auto &p : geometry.target.decodeProjections) matrices.push_back({p.outputSize, p.inputSize});
  for (uint32_t family : {9U, 10U, 11U})
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
            require(scratch.input >= need.input && scratch.sums >= need.sums && scratch.partials >= need.partials &&
                        scratch.counters >= need.counters,
                    "decode arena scratch below a lane's plan");
          }
    }
}

// Arenas are sized from the projections the weights hold, so each must have
// sizes; an empty one would drop its workspace from the bound silently.
void checkUnsizedProjection() {
  auto broken = loadedModel<model::Qwen3_8Weights>();
  std::get<model::Qwen3_8Weights>(broken.target).layers.back().downProjection = ops::Projection();
  rejects([&] { static_cast<void>(model::RuntimeGeometry::from(broken, kv::Format::Int8)); },
          "invalid model runtime geometry", "a target projection without sizes reached arena sizing");
}

// The GDN value rows are sized with attentionWidth, so a layout whose value
// heads span another width is refused before loading. Arena sizing checks
// the GDN shape, whose packed rows must also hold the two gates of every
// value head.
void checkGdnWidths() {
  const auto sparse = loadedModel<model::Qwen3_6MoeWeights>();
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
  // The dense layout is checked like the sparse one: its capture layers and
  // its convolution width against its GDN heads.
  const model::Qwen3_8Layout dense;
  model::requireQwenLayout(dense);
  auto capturePastLastLayer = dense;
  capturePastLastLayer.hiddenCaptureLayers.back() = dense.layers;
  auto convolutionMismatch = dense;
  convolutionMismatch.convolutionDimension += dense.gdnHeadDimension;
  for (const model::Qwen3_8Layout &broken : {capturePastLastLayer, convolutionMismatch})
    rejects([&] { model::requireQwenLayout(broken); }, "Qwen target layout is inconsistent",
            "an inconsistent dense layout was accepted");
}

} // namespace

int main() {
  try {
    {
      std::vector<float> projected(3 * 512, 0.0F), bias(512, 0.0F);
      projected[2 * 512] = 3.0F;
      projected[256] = 2.0F;
      projected[512 + 256] = -4.0F;
      bias[0] = 1.0F;
      bias[256] = 0.5F;
      const auto scores = model::kevPointerScores(projected, bias);
      require(scores.size() == 2 && scores[0] == 0.625F && scores[1] == -0.875F,
              "Kev must use the final row query, option keys, bias and sqrt(256) scale");
      projected[2 * 512] = std::numeric_limits<float>::infinity();
      bool rejected = false;
      try { static_cast<void>(model::kevPointerScores(projected, bias)); }
      catch (const std::runtime_error &) { rejected = true; }
      require(rejected, "Kev must reject non-finite pointer scores");
    }
    checkOptionalFeatureMemoryPlan();
    checkUnsizedProjection();
    checkGdnWidths();
    checkGateUpLayers();
    const auto dense = loadedModel<model::Qwen3_8Weights>();
    const auto sparse = loadedModel<model::Qwen3_6MoeWeights>();
    checkTargetOnlyWorkspaces();
    checkStandardDraftScratch();
    checkLaneScratch(dense);
    checkLaneScratch(sparse);
    std::cout << "model execution plans: PASS (two paired geometries)\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
