// A standard DFlash Transformer layer through DFlashDraft, against a
// direct FP64 reference. The separable nonzero Q4 weights give exact closed
// form products, so this covers the full 4B dimensions without an enormous
// CPU GEMM. Both causal and bidirectional proposal masks are exercised.
#include "TestBuffers.hpp"
#include "model/DFlashDraft.hpp"
#include "LinearNumerics.hpp"
#include "GgufFormatReference.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

using namespace splash;
using namespace splash::ops;
using namespace splash::model;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::test::require;
using splash::test::bf16ToFloat;
using splash::test::floatToBf16;

namespace {

constexpr uint32_t Rows = SPLASH_DRAFT_QUERY_ROWS, Hidden = 2560, Qkv = 6144, Attention = 4096,
                   Intermediate = 9216, Head = 128, Vocabulary = 256;
constexpr double Scale = 1.0 / 512;
using Values = std::vector<double>;

double bf(double value) { return bf16ToFloat(floatToBf16(static_cast<float>(value))); }
MetalBuffer allocated(MetalBackend &backend, uint64_t bytes) {
  if (!bytes) return {};
  auto buffer = test::sharedBuffer(backend, bytes);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}
MetalBuffer ones(MetalBackend &backend, uint32_t width) {
  auto buffer = allocated(backend, uint64_t{width} * 2);
  std::fill_n(static_cast<uint16_t *>(buffer.contents()), width, floatToBf16(1.0f));
  return buffer;
}

// W[n,k] = Scale * ((n + seed) % 3 - 1) * ((k + seed) % 3 - 1).
// Build independent MLX native rows, then repack their AF4G64 block planes.
// Codes 7/8/9 and these power-of-two scale/bias values dequantize exactly in
// both BF16 and the staged tile's half weights, preserving the closed form.
Projection projection(MetalBackend &backend, uint32_t output, uint32_t input, uint32_t seed) {
  const uint32_t groups = input / 64;
  const uint64_t parameters = uint64_t{output} * groups;
  std::vector<uint8_t> weights(parameters * 32), scales(parameters * 2), biases(parameters * 2);
  const uint16_t scale = floatToBf16(Scale), bias = floatToBf16(-8 * Scale);
  for (uint32_t n = 0; n < output; ++n)
    for (uint32_t group = 0; group < groups; ++group) {
      const uint64_t at = uint64_t{n} * groups + group;
      std::memcpy(scales.data() + at * 2, &scale, 2);
      std::memcpy(biases.data() + at * 2, &bias, 2);
      for (uint32_t pair = 0; pair < 32; ++pair) {
        const auto code = [&](uint32_t k) {
          return uint8_t(8 + (int((n + seed) % 3) - 1) * (int((k + seed) % 3) - 1));
        };
        weights[at * 32 + pair] = code(group * 64 + pair * 2) | (code(group * 64 + pair * 2 + 1) << 4);
      }
    }
  const auto native = gguf_reference::mlxNative(gguf_reference::Fmt(GGUF_FMT_AF4G64), output, input, weights, scales, biases);
  const auto planes = gguf_reference::repack(gguf_reference::Fmt(GGUF_FMT_AF4G64), native, output, input, nullptr);
  const auto upload = [&](const std::vector<uint8_t> &bytes) {
    auto buffer = allocated(backend, bytes.size());
    std::memcpy(buffer.contents(), bytes.data(), bytes.size());
    return buffer;
  };
  return Projection(output, input, BlockWeights{{QuantizedSegment::planes(
      GGUF_FMT_AF4G64, output, input, upload(planes.w0), {}, upload(planes.meta))}});
}

Values project(const Values &input, uint32_t width, uint32_t output, uint32_t seed, bool rounded = true) {
  Values result(Rows * output);
  for (uint32_t row = 0; row < Rows; ++row) {
    double sum = 0;
    for (uint32_t k = 0; k < width; ++k) sum += input[row * width + k] * (int((k + seed) % 3) - 1);
    for (uint32_t n = 0; n < output; ++n) {
      const double value = sum * Scale * (int((n + seed) % 3) - 1);
      result[row * output + n] = rounded ? bf(value) : value;
    }
  }
  return result;
}
Values residualAdd(const Values &projected, const Values &residual) {
  Values result(projected.size());
  for (size_t i = 0; i < result.size(); ++i) result[i] = bf(bf(projected[i]) + residual[i]);
  return result;
}

Values norm(const Values &input, uint32_t width) {
  Values result(input.size());
  for (uint32_t row = 0; row < input.size() / width; ++row) {
    double squares = 0;
    for (uint32_t n = 0; n < width; ++n) squares += input[row * width + n] * input[row * width + n];
    const double inverse = 1 / std::sqrt(squares / width + SPLASH_RMS_EPSILON);
    for (uint32_t n = 0; n < width; ++n) result[row * width + n] = bf(input[row * width + n] * inverse);
  }
  return result;
}
Values attention(const Values &qkv, bool causal, Values *preparedKeys = nullptr) {
  Values queries(Rows * Attention), keys(Rows * 1024), values(Rows * 1024);
  for (uint32_t row = 0; row < Rows; ++row) {
    for (uint32_t head = 0; head < 40; ++head) {
      const uint32_t offset = head < 32 ? head * Head : Attention + (head - 32) * Head;
      double sum = 0;
      for (uint32_t dim = 0; dim < Head; ++dim) sum += qkv[row * Qkv + offset + dim] * qkv[row * Qkv + offset + dim];
      const double inverse = 1 / std::sqrt(sum / Head + SPLASH_RMS_EPSILON);
      for (uint32_t dim = 0; dim < Head; ++dim) {
        const double value = bf(qkv[row * Qkv + offset + dim] * inverse);
        if (head < 32) queries[row * Attention + offset + dim] = value;
        else keys[row * 1024 + (head - 32) * Head + dim] = value;
      }
    }
    for (uint32_t dim = 0; dim < 1024; ++dim) values[row * 1024 + dim] = qkv[row * Qkv + Attention + 1024 + dim];
  }
  if (preparedKeys) *preparedKeys = keys;
  Values result(Rows * Attention);
  for (uint32_t row = 0; row < Rows; ++row)
    for (uint32_t head = 0; head < 32; ++head) {
      const uint32_t kvHead = head / 4, count = causal ? row + 1 : Rows;
      std::array<double, Rows> scores{};
      double maximum = -INFINITY;
      for (uint32_t key = 0; key < count; ++key) {
        for (uint32_t dim = 0; dim < Head; ++dim)
          scores[key] += queries[row * Attention + head * Head + dim] * keys[key * 1024 + kvHead * Head + dim];
        scores[key] /= std::sqrt(double(Head)); maximum = std::max(maximum, scores[key]);
      }
      double mass = 0;
      for (uint32_t key = 0; key < count; ++key) { scores[key] = std::exp(scores[key] - maximum); mass += scores[key]; }
      for (uint32_t dim = 0; dim < Head; ++dim) {
        double sum = 0;
        for (uint32_t key = 0; key < count; ++key) sum += scores[key] * values[key * 1024 + kvHead * Head + dim];
        result[row * Attention + head * Head + dim] = bf(sum / mass);
      }
    }
  return result;
}

void run(MetalBackend &backend, bool causal) {
  const ExecutionPlans operators(backend.capabilities());
  DFlashDraftWeights weights;
  weights.layout = {.layers=1, .hiddenSize=Hidden, .vocabularySize=Vocabulary, .dynamicSize=0,
      .qkvSize=Qkv, .attentionSize=Attention, .intermediateSize=Intermediate,
      .attentionHeadDimension=Head, .rotaryTheta=1000000, .selectorRank=0,
      .kvHeads=8, .kind=DraftKind::Standard};
  weights.layout.targetLayerCount = 8;
  for (uint32_t index = 0; index < weights.layout.targetLayerCount; ++index) weights.layout.targetLayers[index] = 1 + 4 * index;
  DFlashDraftLayerWeights layer;
  layer.inputNorm = {ones(backend, Hidden), false}; layer.queryNorm = ones(backend, Head);
  layer.keyNorm = ones(backend, Head); layer.postAttentionNorm = {ones(backend, Hidden), false};
  layer.qkvProjection = projection(backend, Qkv, Hidden, 1);
  layer.outputProjection = projection(backend, Hidden, Attention, 2);
  layer.gateProjection = projection(backend, Intermediate, Hidden, 3);
  layer.upProjection = projection(backend, Intermediate, Hidden, 4);
  layer.downProjection = projection(backend, Hidden, Intermediate, 6);
  layer.causal = causal; layer.slidingWindow = causal ? 4096 : 0;
  weights.layers.push_back(layer); weights.finalNorm = {ones(backend, Hidden), false};
  auto head = projection(backend, Vocabulary, Hidden, 6); head.destination = FloatOutput::Float32;
  weights.contextProjection = projection(backend, Hidden, weights.layout.targetHiddenSize(), 7);
  weights.hiddenNorm = {ones(backend, Hidden), false};
  const auto &linear = operators.linear();
  LinearScratchSize size;
  for (const auto *p : {&layer.qkvProjection, &layer.outputProjection, &layer.upProjection,
                        &layer.downProjection, &head, &weights.contextProjection})
    size.include(linear.decodeScratchSize(p->shape()));
  const LinearScratch scratch{allocated(backend,size.input), allocated(backend,size.sums),
      allocated(backend,size.partials), allocated(backend,size.counters), allocated(backend,size.rotated)};
  const auto aw = DraftAttention::plan(weights.layout.attentionShape(), 1).workspace();
  const uint32_t storageRows = linear.decodeStorageRows(Rows, layer.qkvProjection.shape());
  const auto hidden = [&] { return allocated(backend, storageRows * Hidden * 2); };
  DFlashDecodeBuffers buffers;
  buffers.linearScratch=scratch; buffers.hidden={hidden(), hidden()}; buffers.normalized=hidden();
  buffers.proposalQkv=allocated(backend, storageRows * Qkv * 2); buffers.attention=allocated(backend, aw.groupedQueriesBytes);
  buffers.residual=hidden(); buffers.projected=hidden(); buffers.intermediate=allocated(backend, storageRows * Intermediate * 2); buffers.finalHidden=hidden();
  buffers.logits=allocated(backend, storageRows * Vocabulary * 4); buffers.gateScratch=allocated(backend, storageRows * Intermediate * 2);
  buffers.queryKeys=allocated(backend,aw.queryKeysBytes); buffers.queryValues=allocated(backend,aw.queryValuesBytes);
  buffers.ropeCos=allocated(backend,Rows * (Head/2) * 4); buffers.ropeSin=allocated(backend,Rows * (Head/2) * 4);
  std::fill_n(static_cast<float *>(buffers.ropeCos.contents()), Rows * (Head/2), 1.0f);
  const uint32_t capacity = causal ? 4096 : 128;
  buffers.persistentKeys.resize(1); buffers.persistentValues.resize(1);
  buffers.persistentKeys[0][0]=allocated(backend,uint64_t{8} * capacity * Head * 2);
  buffers.persistentValues[0][0]=allocated(backend,uint64_t{8} * capacity * Head * 2);
  // The fixed four-lane ABI requires valid bindings even when only one
  // lane is dispatched. Inactive lanes alias its cache; their kernels never
  // execute. The active cache remains independently allocated and checked.
  for (uint32_t lane = 1; lane < ExecutionLimits::maximumBatchWidth; ++lane) {
    buffers.persistentKeys[0][lane] = buffers.persistentKeys[0][0];
    buffers.persistentValues[0][lane] = buffers.persistentValues[0][0];
  }
  {
    auto absent = buffers.persistentKeys[0];
    absent[0] = {};
    metal::CommandGraph refused;
    const std::array<uint32_t, 1> lengths{0};
    splash::test::rejects([&] {
      DraftAttention::addDecode(refused,
          {buffers.attention, absent, buffers.persistentValues[0], buffers.queryKeys, buffers.queryValues},
          lengths, DraftAttention::plan(weights.layout.attentionShape(), 1),
          {capacity, causal ? 4096u : 0u, causal});
    }, "draft key ring buffer holds", "standard attention accepted an absent active cache");
    require(refused.empty(), "absent active draft cache encoded an attention dispatch");
  }
  Values input(Rows * Hidden);
  for (uint32_t row = 0; row < Rows; ++row)
    for (uint32_t n = 0; n < Hidden; ++n) {
      input[row * Hidden + n] = bf(float(int((n * 7 + row * 13) % 29) - 14) * 0.073f +
                                float(int(n % 3) - 1) * 0.23f + row * 0.009f);
      static_cast<uint16_t *>(buffers.hidden[0].contents())[row * Hidden + n] = floatToBf16(input[row * Hidden + n]);
    }
  const DFlashDraft draft(weights, backend, operators);
  require(draft.decodeStorageLanes(1) >= 1 &&
              draft.decodeStorageLanes(1) <= ExecutionLimits::maximumBatchWidth,
          "standard draft storage planning treated absent dynamic projections as matrices");
  {
    // Exercise the actual FC -> hidden norm -> context K/V graph entry.
    // Runtime must bind physical storage capacity while this entry's
    // startPositions keep its normalization and cache writers at Rows.
    DFlashContextBuffers context;
    context.linearScratch = scratch;
    context.capturedTargetHidden = allocated(backend,
        uint64_t{storageRows} * weights.layout.targetHiddenSize() * 2);
    context.projected = hidden(); context.hidden = hidden();
    context.contextKv = allocated(backend,
        uint64_t{storageRows} * weights.layout.contextKvSize() * 2);
    context.ropeCos = buffers.ropeCos; context.ropeSin = buffers.ropeSin;
    context.retainedCounts = allocated(backend, sizeof(uint32_t));
    context.persistentKeys = buffers.persistentKeys;
    context.persistentValues = buffers.persistentValues;
    const std::array<uint32_t, 1> starts{0};
    if (storageRows > Rows) {
      auto shortContext = context;
      shortContext.capturedTargetHidden = backend.view(context.capturedTargetHidden, 0,
          uint64_t{Rows} * weights.layout.targetHiddenSize() * 2);
      metal::CommandGraph refused;
      splash::test::rejects([&] { draft.addContextCommit(refused, shortContext, starts); },
          "projection input buffer holds", "context FC accepted an unpadded captured input view");
      require(refused.empty(), "short context FC input encoded a dispatch before rejection");
    }
    metal::CommandGraph contextGraph;
    draft.addContextCommit(contextGraph, context, starts);
    require(!contextGraph.empty(), "padded standard context commit encoded no FC/normalization graph");
    std::cout << "standard context graph: PASS rows=" << Rows
              << " storage_rows=" << storageRows << '\n';
  }
  metal::CommandGraph graph;
  const std::array<uint32_t,1> lengths{0};
  draft.addDecode(graph,buffers,head,lengths);
  for (const auto &dispatch : graph.dispatches()) {
    require(dispatch.pipelineName.find("convolution") == std::string::npos,
            "standard DFlash incorrectly dispatched a convolution");
    require(dispatch.pipelineName.find("selector") == std::string::npos,
            "standard DFlash incorrectly dispatched an edge selector");
  }
  static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
  const auto mixed = project(norm(input,Hidden),Hidden,Qkv,1);
  Values preparedKeys;
  const auto attended = attention(mixed,causal,&preparedKeys);
  // The standard block projection writes a BF16 result before the residual add, as
  // torch/MLX BF16 projections do. Keep both rounding boundaries here.
  const auto residual=residualAdd(project(attended,Attention,Hidden,2),input);
  const auto normalized=norm(residual,Hidden);
  const auto gate=project(normalized,Hidden,Intermediate,3), up=project(normalized,Hidden,Intermediate,4);
  Values glu(gate.size());
  for (size_t i=0;i<glu.size();++i) glu[i]=bf(up[i] * gate[i] / (1+std::exp(-gate[i])));
  const auto final=residualAdd(project(glu,Intermediate,Hidden,6),residual);
  const auto logits=project(norm(final,Hidden),Hidden,Vocabulary,6,false);
  std::string firstFailure;
  const auto compare = [&](const char *stage, const Values &expected, auto actual,
                            double relative, double absolute) {
    double maxError = 0.0, maxRatio = 0.0;
    size_t worst = 0, failed = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
      const double got = actual(i), error = std::fabs(got - expected[i]);
      const double bound = relative * std::fabs(expected[i]) + absolute;
      const double ratio = std::isfinite(got) ? error / bound : INFINITY;
      if (ratio > maxRatio) { maxRatio = ratio; maxError = error; worst = i; }
      failed += ratio > 1.0;
    }
    std::cout << "standard layer causal=" << causal << " stage=" << stage
              << " failures=" << failed << " max_abs=" << maxError << " bound_ratio=" << maxRatio
              << " index=" << worst << " got=" << actual(worst) << " expected=" << expected[worst] << '\n';
    if (failed && firstFailure.empty()) firstFailure = stage;
  };
  const auto bfValues = [&](const MetalBuffer &buffer) {
    const auto *data = static_cast<const uint16_t *>(buffer.contents());
    return [data](size_t i) { return double(bf16ToFloat(data[i])); };
  };
  Values expectedValues(Rows * 1024);
  for (uint32_t row = 0; row < Rows; ++row)
    for (uint32_t dim = 0; dim < 1024; ++dim)
      expectedValues[row * 1024 + dim] = mixed[row * Qkv + Attention + 1024 + dim];
  const auto *storedValues = static_cast<const uint16_t *>(buffers.queryValues.contents());
  const auto *storedKeys = static_cast<const uint16_t *>(buffers.queryKeys.contents());
  compare("prepared_V", expectedValues, [&](size_t i) {
    const size_t row = i / 1024, dim = i % 1024;
    return double(bf16ToFloat(storedValues[dim * Rows + row]));
  }, 0.025, 0.00025);
  compare("normalized_K", preparedKeys, [&](size_t i) {
    const size_t row = i / 1024, head = (i % 1024) / Head, dim = i % Head;
    return double(bf16ToFloat(storedKeys[(head * Rows + row) * Head + dim]));
  }, 0.025, 0.00025);
  compare("attention_reorder", attended, bfValues(buffers.proposalQkv), 0.025, 0.00025);
  compare("attention_residual", residual, bfValues(buffers.residual), 0.025, 0.00025);
  compare("post_attention_norm", normalized, bfValues(buffers.normalized), 0.025, 0.00025);
  compare("SwiGLU", glu, bfValues(buffers.intermediate), 0.025, 0.00025);
  compare("final_residual", final, bfValues(buffers.hidden[1]), 0.025, 0.00025);
  compare("final_norm", norm(final,Hidden), bfValues(buffers.finalHidden), 0.025, 0.00025);
  const auto *actualLogits = static_cast<const float *>(buffers.logits.contents());
  compare("shared_head_logits", logits, [&](size_t i) { return double(actualLogits[i]); }, 0.03, 0.0005);
  require(firstFailure.empty(), "standard DFlash differs from direct FP64 layer reference; first stage=" + firstFailure);
}

} // namespace

int main(int argc,char **argv) {
  @autoreleasepool {
    try {
      if (argc!=2) throw std::invalid_argument("usage: dflash-standard METALLIB");
      // Cancellation distinguishes BF16 matmul->residual from a fused
      // FP32 matmul+residual. This seam must not disappear from the oracle.
      require(residualAdd({1.003}, {-1.0})[0] == 0.0 && bf(1.003 - 1.0) > 0.0,
              "standard reference lost the BF16 projection rounding before residual addition");
      MetalBackend backend(argv[1]); run(backend,false); run(backend,true);
      std::cout << "dflash-standard: PASS\n"; return 0;
    } catch (const std::exception &error) {
      std::cerr << "dflash-standard: FAIL: " << error.what() << '\n'; return 1;
    }
  }
}
