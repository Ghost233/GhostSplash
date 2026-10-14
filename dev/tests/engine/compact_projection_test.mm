#include "TestBuffers.hpp"
#include "GgufFormatReference.hpp"
#include "model/CompactDraftHead.hpp"
#include "TestFiles.hpp"
#include "ops/Linear.hpp"
#include "ops/DraftSelector.hpp"
#include "metal/abi/OptionalFeatures.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

using namespace splash;
using namespace splash::ops;
using splash::test::require;

namespace {
metal::MetalBuffer upload(metal::MetalBackend &backend, const std::vector<uint8_t> &bytes) {
  if (bytes.empty()) return {};
  auto buffer = test::sharedBuffer(backend, bytes.size());
  std::memcpy(buffer.contents(), bytes.data(), bytes.size());
  return buffer;
}

void formatCheck(metal::MetalBackend &backend, uint32_t format) {
  constexpr uint32_t N = 1024, K = 512, C = 512, R = SPLASH_TARGET_VERIFY_ROWS;
  std::mt19937 random(100 + format);
  const auto f = gguf_reference::Fmt(format);
  const auto native = gguf_reference::makeNative(f, N, K, random);
  const auto packed = gguf_reference::repack(f, native, N, K, nullptr);
  Projection source(N, K, BlockWeights{{QuantizedSegment::planes(
      format, N, K, upload(backend, packed.w0), upload(backend, packed.w1), upload(backend, packed.meta))}});
  source.destination = FloatOutput::Float32;
  std::vector<uint32_t> ids(C);
  for (uint32_t i = 0; i < C; ++i) ids[i] = (73 * i + 199) % N;
  const auto compact = gatherProjection(backend, source, ids);
  if (format == GGUF_FMT_Q6K) {
    const test::TemporaryDirectory directory("compact-public-lifecycle");
    const auto path = directory.path() / "ranking.splash";
    test::writeFile(path, "bad file");
    const auto before = backend.memoryStats().allocatedBytes;
    auto disabled = model::CompactDraftHead::createIfEnabled(false, backend, Projection{}, path);
    require(!disabled && backend.memoryStats().allocatedBytes == before,
            "disabled compact must ignore malformed ranking and allocate nothing");
    bool badFile = false;
    try { static_cast<void>(model::CompactDraftHead::createIfEnabled(true, backend, source, path)); }
    catch (const std::invalid_argument &) { badFile = true; }
    require(badFile && backend.memoryStats().allocatedBytes == before,
            "malformed ranking must reject before compact allocation");
    std::vector<uint32_t> ranking(N);
    std::iota(ranking.begin(), ranking.end(), 0);
    std::vector<uint8_t> binary(12 + N * 4);
    std::memcpy(binary.data(), "SPLCHD02", 8);
    std::memcpy(binary.data() + 8, &N, 4);
    std::memcpy(binary.data() + 12, ranking.data(), N * 4);
    test::writeFile(path, binary);
    auto head = model::CompactDraftHead::createIfEnabled(true, backend, source, path, C);
    const auto checks = head->validation().comparisons;
    require(checks == uint64_t{SPLASH_DRAFT_QUERY_ROWS} * C && head->bytes() > 0,
            "explicit compact must validate before enabling");
    std::vector<uint32_t> history(10000, 7); history.push_back(1000);
    require(head->prepare(history) && head->validRows() == 257 &&
                static_cast<uint32_t *>(head->mapping().contents())[256] == 1000,
            "dynamic compact rows preserve original token IDs");
    require(!head->prepare(std::vector<uint32_t>{1000, 7}) &&
                head->reason() == "rare-token ratio exceeds 1 percent", "coverage ratio is normal fallback");
    history.assign(30000, 7);
    for (uint32_t i = 0; i < 257; ++i) history.push_back(256 + i);
    require(!head->prepare(history) && head->reason() == "more than 256 distinct rare tokens",
            "coverage ID budget is normal fallback");
    require(head->validation().comparisons == checks, "generation must not repeat full-head validation");
    bool badLayout = false;
    try { static_cast<void>(model::CompactDraftHead::createIfEnabled(true, backend, Projection{}, path)); }
    catch (const std::invalid_argument &) { badLayout = true; }
    require(badLayout, "unsupported head layout must reject enabling");
  }
  auto validationIds = test::sharedBuffer(backend, ids.size() * 4);
  std::memcpy(validationIds.contents(), ids.data(), ids.size() * 4);
  const auto validation = validateCompactProjection(backend, source, compact, validationIds);
  require(validation.comparisons == uint64_t{SPLASH_DRAFT_QUERY_ROWS} * C &&
              validation.maximumAbsoluteError == 0, "startup compact projection validation");
  std::swap(static_cast<uint32_t *>(validationIds.contents())[0],
            static_cast<uint32_t *>(validationIds.contents())[1]);
  bool rejected = false;
  try { static_cast<void>(validateCompactProjection(backend, source, compact, validationIds)); }
  catch (const std::runtime_error &) { rejected = true; }
  require(rejected, "wrong compact token mapping must fail numerical validation");
  const auto &segment = compact.blocks().segments.front();
  const auto &info = kQuantFormats[format];
  const auto plane = [&](const metal::MetalBuffer &buffer, const std::vector<uint8_t> &original,
                         uint32_t units, uint32_t bytes) {
    if (!bytes) return;
    auto read = test::sharedBuffer(backend, buffer.sizeBytes());
    metal::CommandGraph graph;
    graph.add("test_copy_u32", {buffer, read}, uint32_t(buffer.sizeBytes() / 4),
              {(buffer.sizeBytes() / 4 + 255) / 256, 1, 1});
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    const auto *got = static_cast<const uint8_t *>(read.contents());
    for (uint32_t row = 0; row < C; ++row)
      for (uint32_t unit = 0; unit < units; ++unit) {
        const uint64_t from = ((uint64_t{ids[row] / 256} * units + unit) * 256 + ids[row] % 256) * bytes;
        const uint64_t to = ((uint64_t{row / 256} * units + unit) * 256 + row % 256) * bytes;
        require(!std::memcmp(got + to, original.data() + from, bytes),
                "compact quantized plane differs: " + std::string(info.name));
      }
  };
  plane(segment.plane0, packed.w0, K / 32, info.plane0_bytes);
  plane(segment.plane1, packed.w1, K / 32, info.plane1_bytes);
  plane(segment.meta, packed.meta, K / 32 / info.meta_groups, info.meta_bytes);

  auto input = test::sharedBuffer(backend, R * K * 2);
  std::vector<float> activation(R * K);
  for (uint32_t i = 0; i < R * K; ++i) {
    activation[i] = gguf_reference::bf2f(gguf_reference::f2bf(float(int(random() % 200) - 100) / 128));
    static_cast<uint16_t *>(input.contents())[i] = gguf_reference::f2bf(activation[i]);
  }
  auto originalOutput = test::sharedBuffer(backend, R * N * 4);
  auto compactOutput = test::sharedBuffer(backend, R * C * 4);
  const Linear linear(backend.capabilities());
  metal::CommandGraph graph;
  for (const auto &pair : std::array<std::pair<const Projection *, metal::MetalBuffer>, 2>{{{&source, originalOutput}, {&compact, compactOutput}}}) {
    const auto plan = Linear::plan({{pair.first->outputSize, K}, R, LinearPhase::Decode, LinearEpilogue::None},
                                   {LinearTile::GgufStaged, 1, false, false}, FloatOutput::Float32);
    static_cast<void>(linear.add(graph, {.input = input, .output = pair.second}, *pair.first, plan));
  }
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  const auto *full = static_cast<const float *>(originalOutput.contents());
  const auto *got = static_cast<const float *>(compactOutput.contents());
  std::vector<float> weights(K);
  for (uint32_t n = 0; n < C; ++n) {
    gguf_reference::rowValues(f, native.data() + ids[n] * gguf_reference::rowBytes(f, K), K, weights.data());
    for (uint32_t r = 0; r < R; ++r) {
      const float value = got[r * C + n];
      require(value == full[r * N + ids[n]], "compact selected logit is not bitwise equal");
      const auto dot = gguf_reference::dot(activation.data() + r * K, weights.data(), K);
      require(std::isfinite(value) && std::fabs(value - dot.value) <= gguf_reference::projectionBound(dot, true),
              "compact logit exceeds existing FP64 bound");
    }
  }
}

void publicSharingCheck(metal::MetalBackend &backend) {
  constexpr uint32_t N = 1024, K = 512;
  const test::TemporaryDirectory directory("compact-shared-ranking");
  const auto path = directory.path() / "public.splash";
  std::vector<uint32_t> ranking(N);
  std::iota(ranking.begin(), ranking.end(), 0);
  std::vector<uint8_t> binary(12 + N * 4);
  std::memcpy(binary.data(), "SPLCHD02", 8);
  std::memcpy(binary.data() + 8, &N, 4);
  std::memcpy(binary.data() + 12, ranking.data(), N * 4);
  test::writeFile(path, binary);
  for (uint32_t format : {uint32_t{GGUF_FMT_IQ3S}, uint32_t{GGUF_FMT_Q6K}}) {
    std::mt19937 random(723 + format);
    const auto native = gguf_reference::makeNative(gguf_reference::Fmt(format), N, K, random);
    const auto packed = gguf_reference::repack(gguf_reference::Fmt(format), native, N, K, nullptr);
    Projection source(N, K, BlockWeights{{QuantizedSegment::planes(format, N, K,
        upload(backend, packed.w0), upload(backend, packed.w1), upload(backend, packed.meta))}});
    source.destination = FloatOutput::Float32;
    const auto compact = model::CompactDraftHead::createIfEnabled(true, backend, source, path, 512);
    require(compact->validation().comparisons == uint64_t{SPLASH_DRAFT_QUERY_ROWS} * 512 &&
                compact->validation().maximumAbsoluteError == 0,
            "same public file must validate independently against different model heads");
  }
}

void mappingCheck(metal::MetalBackend &backend) {
  constexpr uint32_t Capacity = 512, Valid = 499;
  auto logits = test::sharedBuffer(backend, SPLASH_DRAFT_QUERY_ROWS * Capacity * 4);
  std::fill_n(static_cast<float *>(logits.contents()), SPLASH_DRAFT_QUERY_ROWS * Capacity, 3.0F);
  auto ids = test::sharedBuffer(backend, Capacity * 4);
  for (uint32_t i = 0; i < Capacity; ++i) static_cast<uint32_t *>(ids.contents())[i] = 1000 + i;
  auto proposed = test::sharedBuffer(backend, SPLASH_DRAFT_PROPOSAL_TOKENS * 4);
  auto candidates = test::sharedBuffer(backend, SPLASH_DRAFT_PROPOSAL_TOKENS * SPLASH_DRAFT_CANDIDATES * 4);
  std::fill_n(static_cast<uint32_t *>(proposed.contents()), SPLASH_DRAFT_PROPOSAL_TOKENS, 17U);
  std::fill_n(static_cast<uint32_t *>(candidates.contents()), SPLASH_DRAFT_PROPOSAL_TOKENS * SPLASH_DRAFT_CANDIDATES, 0xffffffffU);
  static_cast<uint32_t *>(candidates.contents())[0] = 18;
  const CompactMapParams params{SPLASH_DRAFT_QUERY_ROWS, Capacity, Valid, 1};
  metal::CommandGraph graph;
  graph.add("compact_mask_logits", {logits}, params, {SPLASH_DRAFT_QUERY_ROWS * Capacity / 256, 1, 1});
  graph.add("compact_map_proposals", {proposed, candidates, ids}, params, {1, 1, 1});
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  require(static_cast<uint32_t *>(proposed.contents())[0] == 1017, "proposal id mapping");
  require(static_cast<uint32_t *>(candidates.contents())[0] == 1018 &&
          static_cast<uint32_t *>(candidates.contents())[1] == 0xffffffffU, "sampled candidate id mapping");
  for (uint32_t i = 0; i < SPLASH_DRAFT_QUERY_ROWS * Capacity; ++i)
    require(static_cast<float *>(logits.contents())[i] == (i % Capacity >= Valid ? -INFINITY : 3.0F),
            "padding must not enter proposal distribution");
}
}

void dflash2MappingCheck(metal::MetalBackend &backend) {
  constexpr uint32_t Vocabulary = 1024, Capacity = 512, Valid = 496, R = SPLASH_DRAFT_QUERY_ROWS;
  auto ids = test::sharedBuffer(backend, Capacity * 4);
  for (uint32_t i = 0; i < Capacity; ++i) static_cast<uint32_t *>(ids.contents())[i] = (73 * i + 199) % Vocabulary;
  auto predecessor = test::sharedBuffer(backend, Vocabulary * SPLASH_DRAFT_SELECTOR_RANK * 2);
  auto successor = test::sharedBuffer(backend, Vocabulary * SPLASH_DRAFT_SELECTOR_RANK * 2);
  auto hidden = test::sharedBuffer(backend, R * SPLASH_DRAFT_SELECTOR_RANK * 2);
  for (const auto &buffer : {predecessor, successor, hidden})
    for (uint32_t i = 0; i < buffer.sizeBytes() / 2; ++i)
      static_cast<uint16_t *>(buffer.contents())[i] = gguf_reference::f2bf(float(int((i * 37) % 53) - 26) / 128);
  const auto workspace = DraftSelector::workspace(SPLASH_DRAFT_PROPOSAL_TOKENS);
  const auto buffers = [&](uint32_t vocabulary) {
    DraftSelectorBuffers b;
    b.logits = test::sharedBuffer(backend, R * vocabulary * 4);
    b.partialIds = test::sharedBuffer(backend, workspace.partialIdsBytes);
    b.partialValues = test::sharedBuffer(backend, workspace.partialValuesBytes);
    b.candidates = test::sharedBuffer(backend, workspace.candidatesBytes);
    b.unary = test::sharedBuffer(backend, workspace.unaryBytes);
    b.selectorHidden = hidden;
    b.uniforms = test::sharedBuffer(backend, SPLASH_SAMPLING_UNIFORMS * 4);
    std::fill_n(static_cast<float *>(b.uniforms.contents()), SPLASH_SAMPLING_UNIFORMS, 0.37F);
    b.proposedTokens = test::sharedBuffer(backend, SPLASH_DRAFT_PROPOSAL_TOKENS * 4);
    b.proposalProbabilities = test::sharedBuffer(backend, workspace.proposalProbabilitiesBytes);
    return b;
  };
  auto full = buffers(Vocabulary), compact = buffers(Capacity);
  std::fill_n(static_cast<float *>(full.logits.contents()), R * Vocabulary, -INFINITY);
  std::fill_n(static_cast<float *>(compact.logits.contents()), R * Capacity, -INFINITY);
  for (uint32_t r = 0; r < R; ++r)
    for (uint32_t i = 0; i < Valid; ++i) {
      const uint32_t id = static_cast<uint32_t *>(ids.contents())[i];
      const float value = float(int((id * 71 + r * 17) % 1019) - 509) / 128;
      static_cast<float *>(full.logits.contents())[r * Vocabulary + id] = value;
      static_cast<float *>(compact.logits.contents())[r * Capacity + i] = value;
    }
  compact.compactMapping = ids; compact.compactVocabulary = Capacity; compact.compactValidRows = Valid;
  const DraftSelector selector(Vocabulary);
  const uint32_t anchor = 42;
  for (float temperature : {0.0F, 0.8F}) {
    const SamplingPolicy policy{0, temperature};
    metal::CommandGraph graph;
    selector.add(graph, full, {predecessor, successor}, {&anchor, 1}, {&policy, 1});
    selector.add(graph, compact, {predecessor, successor}, {&anchor, 1}, {&policy, 1});
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    require(!std::memcmp(full.proposedTokens.contents(), compact.proposedTokens.contents(),
                         SPLASH_DRAFT_PROPOSAL_TOKENS * 4), "DFlash2 mapped proposals differ");
    require(!std::memcmp(full.candidates.contents(), compact.candidates.contents(),
                         workspace.candidatesBytes), "DFlash2 mapped codebook candidate IDs differ");
    if (temperature > 0)
      require(!std::memcmp(full.proposalProbabilities.contents(), compact.proposalProbabilities.contents(),
                           workspace.proposalProbabilitiesBytes), "DFlash2 mapped proposal probabilities differ");
  }
}


int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: compact-projection TEST_PRODUCTION_METALLIB");
    metal::MetalBackend backend(argv[1]);
    for (uint32_t f = 0; f < GGUF_FMT_COUNT; ++f) formatCheck(backend, f);
    publicSharingCheck(backend);
    mappingCheck(backend);
    dflash2MappingCheck(backend);
    std::cout << "compact projection: " << GGUF_FMT_COUNT
              << " formats, native-plane bytes, selected logits and FP64 bounds PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "compact projection FAIL: " << error.what() << '\n';
    return 1;
  }
}
