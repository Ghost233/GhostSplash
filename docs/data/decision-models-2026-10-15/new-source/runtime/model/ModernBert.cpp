#include "model/ModernBert.hpp"
#include "model/GgufFile.hpp"
#include "model/WeightSource.hpp"
#include "model/GgufImageLayout.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/ModernBert.h"
#include "ops/Linear.hpp"
#include "ops/ModernBert.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace splash::model {
namespace {
using metal::MetalBuffer;
uint32_t padded(uint32_t n, uint32_t multiple) { return (n + multiple - 1) / multiple * multiple; }
struct Layer {
  ops::Projection qkv, output, up, down;
  MetalBuffer attnNorm, attnBias, qkvBias, outputBias, ffnNorm, ffnBias, upBias, downBias;
  uint32_t intermediate = 0;
};
}
struct ModernBert::Impl {
  metal::MetalBackend &backend;
  WeightSource source;
  GgufFile file;
  ops::Linear linear;
  uint32_t context, width, stride, heads, vocabulary, mask, sep, bos, encoderLayers;
  float epsilon, theta, localTheta;
  uint64_t budget, weights = 0, quantizedBytes = 0, workspace = 0, baseBytes = 0;
  uint32_t workspaceRows = 0, maximumOutput = 0, maximumInput = 0;
  std::vector<Layer> layers;
  MetalBuffer embedding, embeddingNorm, finalNorm, types, clsNorm, clsNormBias, clsBias, clsOut, clsOutBias;
  ops::Projection cls;
  MetalBuffer zero, inputTokens, positions, hidden, normed, projected, delta, queries, keys, values, attention;
  MetalBuffer optionHidden, optionInput, scores;
  ops::LinearScratch scratch;
  metal::CommandTiming timing;

  MetalBuffer allocate(uint64_t bytes, std::string_view name) {
    // Bound the actual aligned allocations, including temporary weight preparation.
    if (bytes > budget - std::min(budget, backend.memoryStats().allocatedBytes))
      throw std::runtime_error("ModernBERT allocation exceeds --max-memory");
    auto b = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, name);
    if (backend.memoryStats().allocatedBytes > budget)
      throw std::runtime_error("ModernBERT aligned allocations exceed --max-memory");
    std::memset(b.contents(), 0, b.sizeBytes());
    return b;
  }
  const GgufTensor &tensor(std::string_view name, std::span<const uint64_t> shape, uint32_t type) {
    const auto &t = file.require(name);
    if (t.type != type || t.dims.size() != shape.size() || !std::equal(t.dims.begin(), t.dims.end(), shape.begin()))
      throw GgufError("unexpected ModernBERT tensor type or shape: " + std::string(name));
    return t;
  }
  MetalBuffer vector(std::string name, uint32_t n, uint32_t type = ggml::kF32) {
    const uint64_t shape[]{n}; const auto &t = tensor(name, shape, type);
    auto b = allocate(t.bytes, name); source.readData(t.offset, {static_cast<uint8_t *>(b.contents()), static_cast<size_t>(t.bytes)}); return b;
  }
  ops::Projection projection(std::string name, uint32_t n, uint32_t k) {
    const uint64_t shape[]{k, n}; const auto &t = tensor(name, shape, 8);
    uint32_t np = padded(n, 256), kp = padded(k, 256);
    const auto &format = kQuantFormats[GGUF_FMT_Q80];
    auto sizes = ggufPlaneBytes(format, np, kp);
    auto planes = allocate(sizes.plane0 + sizes.meta, name);
    quantizedBytes += planes.allocatedBytes();
    auto native = allocate(uint64_t{np} * (kp / 32) * 34, "modern-q8-repack-staging");
    std::vector<uint8_t> original(t.bytes);
    source.readData(t.offset, original);
    const uint64_t from = uint64_t{k / 32} * 34, to = uint64_t{kp / 32} * 34;
    auto *data = static_cast<uint8_t *>(native.contents());
    for (uint32_t row = 0; row < n; ++row) std::memcpy(data + row * to, original.data() + row * from, from);
    ops::ModernBert::repackQ8(backend, native, planes, np, kp);
    auto segment = ops::QuantizedSegment::planes(GGUF_FMT_Q80, np, kp,
        backend.view(planes, 0, sizes.plane0), {}, backend.view(planes, sizes.plane0, sizes.meta));
    ops::Projection result(np, kp, ops::BlockWeights{{std::move(segment)}});
    result.destination = ops::FloatOutput::Float32;
    return result;
  }

  Impl(metal::MetalBackend &b, const std::filesystem::path &path, uint32_t capacity, uint64_t maximum)
      : backend(b), source(path), file(source), linear(b.capabilities()), context(capacity), budget(maximum) {
    const std::string prefix = "modern-bert.";
    const auto integer = [&](std::string_view name) {
      auto v = file.metadata().unsignedValue(prefix + std::string(name));
      if (!v || !*v || *v > UINT32_MAX) throw GgufError("invalid ModernBERT metadata: " + std::string(name));
      return static_cast<uint32_t>(*v);
    };
    width = integer("embedding_length"); stride = padded(width, 256);
    heads = integer("attention.head_count"); vocabulary = integer("vocab_size");
    uint32_t count = integer("block_count"); encoderLayers = count - 2;
    if (file.metadata().architecture() != "modern-bert" ||
        file.metadata().stringValue(prefix + "decision.type") != "laya" ||
        file.metadata().unsignedValue(prefix + "attention.causal") != 0 ||
        integer("decision.block_count") != 2 || integer("context_length") != 8192 ||
        !context || context > 8192 || width / heads != 64 || width % heads ||
        !((width == 384 && count == 24 && vocabulary == 256000) ||
          (width == 1024 && count == 30 && vocabulary == 50368)) ||
        integer("attention.sliding_window") != 128 || integer("attention.sliding_window_pattern") != 3 ||
        file.metadata().stringValue(prefix + "hidden_activation") != "gelu")
      throw GgufError("unsupported ModernBERT decision configuration");
    epsilon = file.metadata().floatValue(prefix + "attention.layer_norm_epsilon").value_or(0);
    theta = file.metadata().floatValue(prefix + "rope.freq_base").value_or(0);
    localTheta = file.metadata().floatValue(prefix + "rope.freq_base_swa").value_or(0);
    if (!(epsilon > 0 && std::isfinite(epsilon) && theta > 0 && localTheta > 0)) throw GgufError("invalid ModernBERT normalization or RoPE");
    const auto tokenId = [&](const char *name) {
      auto id = file.metadata().unsignedValue(std::string("tokenizer.ggml.") + name + "_token_id");
      if (!id || *id >= vocabulary) throw GgufError("invalid ModernBERT marker token");
      return static_cast<uint32_t>(*id);
    };
    mask = tokenId("mask"); sep = tokenId("seperator"); bos = tokenId("bos");
    auto ffn = file.numericArray(prefix + "feed_forward_length");
    if (!ffn || ffn->size() != count) throw GgufError("invalid ModernBERT FFN width array");
    const uint64_t base = backend.memoryStats().allocatedBytes;
    baseBytes = base;
    zero = allocate(65536 * sizeof(float), "modern-zero-bias");
    const uint64_t embShape[]{width, vocabulary}; const auto &emb = tensor("token_embd.weight", embShape, 8);
    embedding = allocate(emb.bytes, "modern-quantized-embedding");
    quantizedBytes += embedding.allocatedBytes();
    source.readData(emb.offset, {static_cast<uint8_t *>(embedding.contents()), static_cast<size_t>(emb.bytes)});
    embeddingNorm = vector("token_embd_norm.weight", width); finalNorm = vector("output_norm.weight", width);
    const uint64_t typeShape[]{width, 3}; const auto &type = tensor("token_types.weight", typeShape, ggml::kF32);
    types = allocate(type.bytes, "modern-question-type-embedding"); source.readData(type.offset, {static_cast<uint8_t *>(types.contents()), static_cast<size_t>(type.bytes)});
    cls = projection("cls.weight", width, width);
    clsNorm = vector("cls.norm.weight", width); clsNormBias = vector("cls.norm.bias", width);
    clsBias = vector("cls.bias", width); clsOut = vector("cls.output.weight", width, ggml::kBF16);
    clsOutBias = vector("cls.output.bias", 1);
    maximumOutput = 3 * width; maximumInput = stride;
    ops::LinearScratchSize scratchSize;
    const auto include = [&](const ops::Projection &p) { scratchSize.include(linear.prefillScratchSize(p.shape())); };
    include(cls);
    for (uint32_t i = 0; i < count; ++i) {
      const double middle = (*ffn)[i];
      if (!(middle > 0 && middle <= 32768 && std::floor(middle) == middle) || uint32_t(middle) % 32)
        throw GgufError("unsupported ModernBERT FFN width");
      bool head = i >= encoderLayers;
      std::string p = "blk." + std::to_string(i) + ".";
      Layer l; l.intermediate = static_cast<uint32_t>(middle);
      l.qkv = projection(p + "attn_qkv.weight", 3 * width, width);
      l.output = projection(p + "attn_output.weight", width, width);
      l.up = projection(p + "ffn_up.weight", (head ? 1 : 2) * l.intermediate, width);
      l.down = projection(p + "ffn_down.weight", width, l.intermediate);
      l.attnNorm = i ? vector(p + "attn_norm.weight", width) : embeddingNorm;
      l.ffnNorm = vector(p + "ffn_norm.weight", width);
      l.attnBias = head ? vector(p + "attn_norm.bias", width) : zero;
      l.qkvBias = head ? vector(p + "attn_qkv.bias", 3 * width) : zero;
      l.outputBias = head ? vector(p + "attn_output.bias", width) : zero;
      l.ffnBias = head ? vector(p + "ffn_norm.bias", width) : zero;
      l.upBias = head ? vector(p + "ffn_up.bias", l.intermediate) : zero;
      l.downBias = head ? vector(p + "ffn_down.bias", width) : zero;
      for (const auto *projection : {&l.qkv, &l.output, &l.up, &l.down}) {
        include(*projection); maximumOutput = std::max(maximumOutput, projection->outputSize);
        maximumInput = std::max(maximumInput, projection->inputSize);
      }
      layers.push_back(std::move(l));
    }
    weights = backend.memoryStats().allocatedBytes - base - zero.allocatedBytes();
    positions = allocate(4 * 256 * 4, "modern-option-positions");
    optionHidden = allocate(uint64_t{256} * stride * 4, "modern-option-residual");
    optionInput = allocate(uint64_t{256} * stride * 2, "modern-option-classifier-input");
    scores = allocate(4 * 256 * 4, "modern-option-logits");
    const auto scratchBuffer = [&](uint64_t bytes, const char *name) { return bytes ? allocate(bytes, name) : MetalBuffer{}; };
    scratch.input = scratchBuffer(scratchSize.input, "modern-linear-input"); scratch.sums = scratchBuffer(scratchSize.sums, "modern-linear-sums");
    scratch.partials = scratchBuffer(scratchSize.partials, "modern-linear-partials"); scratch.counters = scratchBuffer(scratchSize.counters, "modern-linear-counters");
    scratch.rotated = scratchBuffer(scratchSize.rotated, "modern-linear-rotation");
    // Admit the declared context before Ready, while allocating only the rows
    // actually used. A small budget must not promise an impossible long request.
    const uint64_t fullRows = ops::ModernBert::keyRows(context);
    const std::array<uint64_t,9> fullBuffers{
      fullRows*16, fullRows*stride*4, fullRows*maximumInput*2,
      fullRows*maximumOutput*4, fullRows*padded(width,256)*4,
      fullRows*width*2, fullRows*width*2, fullRows*width*2, fullRows*width*2};
    uint64_t reserve = 0;
    for (uint64_t bytes : fullBuffers) reserve += (bytes + 16383) / 16384 * 16384;
    const uint64_t fixed = backend.memoryStats().allocatedBytes;
    if (fixed > budget || reserve > budget - fixed)
      throw std::invalid_argument("encoder context does not fit --max-memory; lower --max-context or increase the memory limit");
    reserveWorkspace(128);
  }
  void reserveWorkspace(uint32_t rows) {
    rows = ops::ModernBert::keyRows(rows);
    if (rows <= workspaceRows) return;
    inputTokens = {}; hidden = {}; normed = {}; projected = {}; delta = {};
    queries = {}; keys = {}; values = {}; attention = {};

    inputTokens = allocate(uint64_t{4} * rows * 4, "modern-input-tokens");
    hidden = allocate(uint64_t{rows} * stride * 4, "modern-fp32-residual");
    normed = allocate(uint64_t{rows} * maximumInput * 2, "modern-normalized-and-ffn-input");
    projected = allocate(uint64_t{rows} * maximumOutput * 4, "modern-shared-projection-output");
    delta = allocate(uint64_t{rows} * padded(width, 256) * 4, "modern-residual-delta");
    queries = allocate(uint64_t{rows} * width * 2, "modern-queries");
    keys = allocate(uint64_t{rows} * width * 2, "modern-keys");
    values = allocate(uint64_t{rows} * width * 2, "modern-values");
    attention = allocate(uint64_t{rows} * width * 2, "modern-attention-output");
    workspaceRows = rows;
    workspace = backend.memoryStats().allocatedBytes - baseBytes - weights;
  }
  void multiply(metal::CommandGraph &g, const ops::Projection &p, const MetalBuffer &in, const MetalBuffer &out, uint32_t rows) {
    for (uint32_t first = 0; first < rows; first += SPLASH_PREFILL_TOKEN_BUDGET) {
      uint32_t chunk = std::min<uint32_t>(SPLASH_PREFILL_TOKEN_BUDGET, rows - first);
      auto plan = linear.prefillPlan(p, chunk, ops::LinearEpilogue::None);
      uint32_t storage = plan.storageRows();
      ops::LinearBuffers buffers{
        backend.view(in, uint64_t{first} * p.inputSize * 2, uint64_t{storage} * p.inputSize * 2),
        backend.view(out, uint64_t{first} * p.outputSize * 4, uint64_t{storage} * p.outputSize * 4), {}, {}, scratch};
      static_cast<void>(linear.add(g, buffers, p, plan));
    }
  }
  void norm(metal::CommandGraph &g, const MetalBuffer &stream, const MetalBuffer &out, const MetalBuffer &w,
            const MetalBuffer &bias, uint32_t rows, bool add, const MetalBuffer &deltaBias, bool replace = false) {
    ops::ModernBert::normalization(g, {stream, delta, deltaBias, w, bias, out},
          ModernNormParams{rows, width, stride, padded(width, 256), uint32_t(add), uint32_t(replace), epsilon});
  }
  void validate(std::span<const uint32_t> tokens, std::span<const uint32_t> options, uint32_t kind) const {
    if (tokens.empty() || tokens.size() > context || tokens.front() != bos || tokens.back() != sep || kind > 2 ||
        options.size() < 2 || options.size() > 255 || !std::is_sorted(options.begin(), options.end()) ||
        std::adjacent_find(options.begin(), options.end()) != options.end() ||
        std::ranges::any_of(tokens, [&](uint32_t t) { return t >= vocabulary; }) ||
        std::ranges::any_of(options, [&](uint32_t p) { return p >= tokens.size() - 1 || tokens[p] != mask; }))
      throw std::invalid_argument("invalid ModernBERT decision tokens, type or option positions");
  }
  void encode(metal::CommandGraph &g, const ModernBertInput &request, uint32_t slot, bool selected) {
    const auto tokens = request.tokens, options = request.options;
    const uint32_t kind = request.kind;
    auto input = backend.view(inputTokens, uint64_t{slot} * workspaceRows * 4, uint64_t{workspaceRows} * 4);
    auto markers = backend.view(positions, uint64_t{slot} * 256 * 4, 256 * 4);
    auto output = backend.view(scores, uint64_t{slot} * 256 * 4, 256 * 4);
    std::memcpy(input.contents(), tokens.data(), tokens.size_bytes());
    std::memcpy(markers.contents(), options.data(), options.size_bytes());
    uint32_t n = static_cast<uint32_t>(tokens.size()), opts = static_cast<uint32_t>(options.size()), pad = ops::ModernBert::keyRows(n);
    ops::ModernBert::embedding(g, {input, embedding, hidden}, ModernRowsParams{n, width, stride, vocabulary});
    norm(g, hidden, normed, embeddingNorm, zero, n, false, zero, true);
    for (uint32_t i = 0; i < layers.size(); ++i) {
      const auto &l = layers[i];
      bool head = i >= encoderLayers, last = i + 1 == layers.size(), sliced = selected && last;
      uint32_t qrows = sliced ? opts : n, qpad = ops::ModernBert::queryRows(qrows);
      if (i == encoderLayers) {
        ops::ModernBert::typeEmbedding(g, {hidden, types}, ModernRowsParams{n, width, stride, kind});
        norm(g, hidden, normed, l.attnNorm, l.attnBias, n, false, zero);
      }
      multiply(g, l.qkv, normed, projected, n);
      bool local = !head && i % 3 != 0;
      ops::ModernBert::prepareQkv(g, {projected, l.qkvBias, markers, queries, keys, values},
            ModernQkvParams{n, pad, heads, width, qrows, qpad, uint32_t(sliced), l.qkv.outputSize, head ? 0.0f : (local ? localTheta : theta)});
      ops::ModernBert::attention(g, {queries, keys, values, attention}, ModernAttentionParams{n, pad, heads, qrows, qpad, local ? 128U : 0U});
      ops::ModernBert::gatherAttention(g, {attention, normed}, ModernRowsParams{qrows, width, stride, qpad});
      multiply(g, l.output, normed, delta, qrows);
      const auto stream = sliced ? optionHidden : hidden;
      if (sliced) ops::ModernBert::selectFloat(g, {hidden, markers, optionHidden}, ModernRowsParams{opts, width, stride, n});
      norm(g, stream, normed, l.ffnNorm, l.ffnBias, qrows, true, l.outputBias);
      multiply(g, l.up, normed, projected, qrows);
      ops::ModernBert::activation(g, {projected, l.upBias, normed},
            ModernActivationParams{qrows, l.intermediate, l.up.outputSize, l.down.inputSize, uint32_t(head)});
      multiply(g, l.down, normed, delta, qrows);
      if (last) {
        norm(g, stream, sliced ? optionInput : normed, clsNorm, clsNormBias, qrows, true, l.downBias);
        if (!sliced) ops::ModernBert::selectBfloat(g, {normed, markers, optionInput}, ModernRowsParams{opts, width, stride, n});
      } else if (i + 1 == encoderLayers) {
        norm(g, hidden, normed, finalNorm, zero, n, true, l.downBias, true);
      } else {
        norm(g, hidden, normed, layers[i + 1].attnNorm, layers[i + 1].attnBias, n, true, l.downBias);
      }
    }
    multiply(g, cls, optionInput, projected, opts);
    ops::ModernBert::classifier(g, {projected, clsBias, clsOut, clsOutBias, output},
          ModernRowsParams{opts, width, cls.outputSize, 0});
  }
  std::vector<std::vector<float>> scoreBatch(std::span<const ModernBertInput> batch, bool selected) {
    if (batch.empty() || batch.size() > 4) throw std::invalid_argument("encoder submission batch must have 1..4 requests");
    uint32_t maximumRows = 0;
    for (const auto &r : batch) {
      validate(r.tokens, r.options, r.kind);
      maximumRows = std::max(maximumRows, static_cast<uint32_t>(r.tokens.size()));
    }
    source.checkUnchanged(); reserveWorkspace(maximumRows);
    metal::CommandGraph graph;
    // Independent sequences execute in order inside one command. Only immutable
    // input/marker rows and final scores are per-request; all large scratch is shared.
    for (uint32_t i = 0; i < batch.size(); ++i) encode(graph, batch[i], i, selected);
    timing = backend.submitCommandAsync(graph.command()).wait();
    std::vector<std::vector<float>> result;
    for (uint32_t i = 0; i < batch.size(); ++i) {
      const auto *v = static_cast<const float *>(scores.contents()) + i * 256;
      result.emplace_back(v, v + batch[i].options.size());
      if (!std::ranges::all_of(result.back(), [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("ModernBERT decision score is not finite");
    }
    return result;
  }
};
ModernBert::ModernBert(metal::MetalBackend &b, const std::filesystem::path &p, uint32_t c, uint64_t limit)
    : impl_(std::make_unique<Impl>(b, p, c, limit)) {}
ModernBert::~ModernBert() = default;
std::vector<float> ModernBert::score(std::span<const uint32_t> t, std::span<const uint32_t> o, uint32_t kind, bool selected) { const ModernBertInput input{t, o, kind}; return impl_->scoreBatch({&input, 1}, selected).front(); }
void ModernBert::validate(const ModernBertInput &r) const { impl_->validate(r.tokens, r.options, r.kind); }
std::vector<std::vector<float>> ModernBert::scoreBatch(std::span<const ModernBertInput> r, bool selected) { return impl_->scoreBatch(r, selected); }
uint32_t ModernBert::vocabularySize() const noexcept { return impl_->vocabulary; }
uint32_t ModernBert::maskToken() const noexcept { return impl_->mask; }
uint32_t ModernBert::bosToken() const noexcept { return impl_->bos; }
uint32_t ModernBert::sepToken() const noexcept { return impl_->sep; }
uint64_t ModernBert::weightBytes() const noexcept { return impl_->weights; }
uint64_t ModernBert::quantizedWeightBytes() const noexcept { return impl_->quantizedBytes; }
uint64_t ModernBert::workspaceBytes() const noexcept { return impl_->workspace; }
metal::CommandTiming ModernBert::lastTiming() const noexcept { return impl_->timing; }
} // namespace splash::model
