// Diagnostic only: original affine Q4 FC versus exact BF16 weights widened
// to the existing F32 projection. Every run owns a fresh model and runtime.
// drafter-fc-precision METALLIB MODEL_ROOT --prompt-token-file PATH
//     [--prompt-tokens 512|32768] [--mode abba|q4|bf16-fc|header]
//     [--prefill-schedule boundary|whole] (boundary matches HTTP replay)
// Context is 33792, greedy, output limit 1024, INT8 target KV. This bypasses
// production startup preflight, but reports actual GPU memory and uses a
// 12-GiB allocation governor. No production loader/config/ABI is modified.
#include "engine/MemoryGovernor.hpp"
#include "engine/Types.hpp"
#include "model/ModelFactory.hpp"
#include "model/QwenState.hpp"
#include "model/Runtime.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightImages.hpp"
#include "ops/PageStorage.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

using Clock = std::chrono::steady_clock;
constexpr uint32_t ContextTokens = 33792, MaximumOutput = 1024;
constexpr uint64_t HardBudgetBytes = 12ULL << 30;
constexpr uint64_t FcElements = uint64_t{2560} * 20480;
constexpr uint64_t FcF32Bytes = FcElements * sizeof(float);

uint32_t count(std::string_view text) {
  uint32_t value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !value)
    throw std::invalid_argument("prompt token count must be positive");
  return value;
}

double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string tokenDigest(std::span<const uint32_t> tokens) {
  return model::weightDigest(std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(tokens.data()), tokens.size_bytes()));
}

std::shared_ptr<model::SafetensorsCheckpoint> fcSource(const std::filesystem::path &root) {
  auto source = std::make_shared<model::SafetensorsCheckpoint>(root / "draft");
  const auto &fc = source->require("fc.weight");
  if (fc.dtype != "BF16" || fc.shape != std::vector<uint64_t>{2560, 20480} ||
      fc.bytes != FcElements * sizeof(uint16_t))
    throw std::invalid_argument("fc.weight must be BF16 [2560,20480]");
  return source;
}

uint64_t exactFc(model::LoadedModel &loaded, metal::MetalBackend &backend,
                 const std::filesystem::path &root) {
  auto source = fcSource(root);
  const model::SourceTensor fc = source->require("fc.weight");
  const uint64_t before = backend.memoryStats().allocatedBytes;
  auto values = loaded.images->loadAuxiliary("draft/diagnostic-exact-bf16-fc-f32", FcF32Bytes,
      [source, fc](std::span<uint8_t> destination, const metal::MetalBuffer &) {
        if (destination.size() != FcF32Bytes)
          throw std::invalid_argument("exact FC destination size mismatch");
        std::vector<uint16_t> staging(model::kLoadStepBytes / sizeof(uint16_t));
        auto *output = reinterpret_cast<uint32_t *>(destination.data());
        for (uint64_t first = 0; first < FcElements; first += staging.size()) {
          const uint64_t elements = std::min<uint64_t>(staging.size(), FcElements - first);
          fc.read(first * sizeof(uint16_t), std::span<uint8_t>(
              reinterpret_cast<uint8_t *>(staging.data()), elements * sizeof(uint16_t)));
          for (uint64_t i = 0; i < elements; ++i) {
            if ((staging[i] & 0x7f80u) == 0x7f80u)
              throw std::runtime_error("non-finite BF16 FC source weight");
            // Widening BF16 to F32 copies its exact representation; no FP16
            // narrowing or quantization participates in this experiment.
            output[first + i] = uint32_t{staging[i]} << 16;
          }
        }
        source->checkUnchanged();
      });
  const auto &old = loaded.draft.contextProjection;
  ops::Projection replacement(old.outputSize, old.inputSize,
      ops::BlockWeights{{ops::QuantizedSegment::floats(old.outputSize, old.inputSize, std::move(values))}});
  replacement.destination = old.destination;
  loaded.draft.contextProjection = std::move(replacement);
  const uint64_t delta = metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
  loaded.draft.actualAllocatedBytes += delta;
  model::requireCompatibleModel(loaded);
  return delta;
}

struct PrefillChunk {
  uint32_t offset = 0, rows = 0;
  double gpuMilliseconds = 0.0;
};

struct Result {
  std::string mode, sourceIdentity, device, prefillSchedule;
  uint32_t replayBoundary = 0;
  std::vector<PrefillChunk> prefillChunks;
  std::vector<uint32_t> output;
  uint64_t cycles = 0, drafted = 0, accepted = 0, committed = 0, prefillProduced = 0;
  uint64_t extraFcBytes = 0, targetWeightBytes = 0, draftWeightBytes = 0;
  uint64_t peakBytes = 0, allocatedBytes = 0, devicePeakBytes = 0;
  double setupWall = 0.0, prefillWall = 0.0, prefillGpu = 0.0;
  double decodeWall = 0.0, decodeGpu = 0.0, requestWall = 0.0;
  bool stopped = false;
};

Result run(const std::filesystem::path &library, const std::filesystem::path &root,
           std::span<const uint32_t> prompt, bool precise, const std::string &schedule) {
  Result result;
  result.mode = precise ? "bf16-fc" : "q4";
  result.prefillSchedule = schedule;
  result.replayBoundary = prompt.empty() ? 0 : uint32_t((prompt.size() - 1) / kv::kPageTokens * kv::kPageTokens);
  const auto setupStart = Clock::now();
  auto descriptor = model::inspectModelRoot(root);
  if (!descriptor.hasDraft() || descriptor.draft.kind != model::DraftKind::Standard ||
      descriptor.draft.hiddenSize != 2560 || descriptor.draft.targetHiddenSize != 20480)
    throw std::invalid_argument("benchmark requires the Qwen3.5-4B standard DFlash assembly");
  std::visit([&](const auto &target) {
    if (target.stopTokens != std::array<uint32_t, 2>{248044, 248046})
      throw std::invalid_argument("unexpected target stop tokens");
    for (const uint32_t token : prompt)
      if (token >= target.vocabularySize) throw std::invalid_argument("prompt token exceeds vocabulary");
  }, descriptor.target);
  descriptor.draft.contextCapacity = ContextTokens;
  descriptor.stateLayout.draft = descriptor.draft.stateLayout();
  metal::MetalBackend backend(library);
  auto loaded = model::loadModel(backend, root, descriptor);
  if (precise) result.extraFcBytes = exactFc(loaded, backend, root);
  result.sourceIdentity = loaded.descriptor.sourceIdentity;
  result.device = backend.capabilities().deviceName;
  result.targetWeightBytes = loaded.targetActualAllocatedBytes();
  result.draftWeightBytes = loaded.draft.actualAllocatedBytes;
  ops::ExecutionPlans operators(backend.capabilities());
  const auto layout = loaded.targetKvLayout(kv::Format::Int8);
  const uint32_t pagesPerLane = (ContextTokens + model::ExecutionLimits::targetVerifyRows +
                                 kv::kPageTokens - 1) / kv::kPageTokens + 1;
  const uint32_t extentPages = layout.maximumExtentPages();
  const uint32_t pageCount = (pagesPerLane + extentPages - 1) / extentPages * extentPages;
  MemoryGovernor governor(backend, HardBudgetBytes, 1, 1, queryHostMemory, 0);
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout, pageCount, extentPages);
  for (uint32_t i = 0; i < pageCount / extentPages; ++i)
    if (!pages.allocateExtent(i)) throw std::runtime_error("could not allocate target KV extent");
  model::QwenStateStorage states(backend, governor.allocationAdmission(), loaded.stateLayout(ops::GdnStateFormat::Float32), nullptr);
  model::Runtime executor({backend, loaded, pages, states, operators});
  if (backend.memoryStats().allocatedBytes > HardBudgetBytes)
    throw std::runtime_error("benchmark runtime allocation exceeds 12 GiB");
  result.setupWall = seconds(setupStart);
  std::vector<uint32_t> pageTable(pagesPerLane);
  for (uint32_t i = 0; i < pagesPerLane; ++i) pageTable[i] = i;
  constexpr uint64_t id = 1;
  EngineRequest request;
  request.id = id;
  request.prompt.assign(prompt.begin(), prompt.end());
  request.maxNewTokens = MaximumOutput;
  request.sampling.temperature = 0.0F;
  const auto requestStart = Clock::now();
  executor.beginColdRequest(request.modelView(), 0);
  try {
    const auto prefillStart = Clock::now();
    uint32_t offset = 0;
    while (offset < prompt.size()) {
      uint32_t rows = std::min<uint32_t>(model::ExecutionLimits::prefillTokenBudget,
                                        uint32_t(prompt.size()) - offset);
      // Match the Engine replay checkpoint before the final Page32 block.
      // A zero/already-reached boundary never introduces an empty chunk.
      if (schedule == "boundary" && offset < result.replayBoundary &&
          rows > result.replayBoundary - offset)
        rows = result.replayBoundary - offset;
      BatchPlan plan{.kind = WorkKind::Prefill, .items = {{id, rows, offset}},
                     .decodeStage = DecodeStage::Regular};
      ModelBatchItem item{id, offset, rows, pageTable, 1};
      item.inputTokens = prompt.subspan(offset, rows);
      const double gpuBefore = executor.telemetry().totalPrefillGpuSeconds;
      auto step = executor.prefill(plan, std::span<const ModelBatchItem>(&item, 1));
      result.prefillChunks.push_back({offset, rows,
          (executor.telemetry().totalPrefillGpuSeconds - gpuBefore) * 1000.0});
      if (step.size() != 1 || step[0].consumedPromptTokens != rows || !step[0].failure.empty())
        throw std::runtime_error(step.size() == 1 && !step[0].failure.empty()
                                     ? step[0].failure : "prefill result size/row mismatch");
      result.output.insert(result.output.end(), step[0].outputTokens.begin(), step[0].outputTokens.end());
      result.stopped = step[0].finished;
      offset += rows;
    }
    result.prefillProduced = result.output.size();
    result.prefillWall = seconds(prefillStart);
    result.prefillGpu = executor.telemetry().totalPrefillGpuSeconds;
    uint64_t position = prompt.size();
    const auto decodeStart = Clock::now();
    while (!result.stopped && result.output.size() < MaximumOutput) {
      BatchPlan plan{.kind = WorkKind::Decode, .items = {{id, 0, 0}}, .decodeStage = DecodeStage::Regular};
      ModelBatchItem item{id, position, 0, pageTable, 1};
      auto step = executor.decode(plan, std::span<const ModelBatchItem>(&item, 1));
      if (step.size() != 1 || !step[0].failure.empty() || step[0].outputTokens.empty())
        throw std::runtime_error(step.size() == 1 && !step[0].failure.empty()
                                     ? step[0].failure : "decode produced no valid result");
      const auto &part = step[0];
      if (part.outputTokensWithoutKv > part.outputTokens.size())
        throw std::runtime_error("decode uncommitted output count is invalid");
      const uint64_t committed = part.outputTokens.size() - part.outputTokensWithoutKv;
      position += committed;
      result.committed += committed;
      result.output.insert(result.output.end(), part.outputTokens.begin(), part.outputTokens.end());
      result.drafted += part.draftedTokens;
      result.accepted += part.acceptedDraftTokens;
      ++result.cycles;
      result.stopped = part.finished;
      if (result.output.size() > MaximumOutput || position != states.metadata(0).lengths.targetTokens)
        throw std::runtime_error("decode output budget or exact KV position diverged");
    }
    result.decodeWall = seconds(decodeStart);
    result.decodeGpu = executor.telemetry().totalDecodeGpuSeconds;
    result.requestWall = seconds(requestStart);
    if (result.output.empty() ||
        (result.stopped && result.output.back() != 248044 && result.output.back() != 248046) ||
        (!result.stopped && result.output.size() != MaximumOutput))
      throw std::runtime_error("generation did not complete at a stop or the output limit");
    const auto memory = backend.memoryStats();
    result.peakBytes = memory.peakAllocatedBytes;
    result.allocatedBytes = memory.allocatedBytes;
    result.devicePeakBytes = memory.devicePeakAllocatedBytes;
    executor.checkHealth();
    executor.end(id);
    if (result.peakBytes > HardBudgetBytes)
      throw std::runtime_error("benchmark peak backend memory exceeds 12 GiB");
  } catch (...) {
    const auto failure = std::current_exception();
    try { executor.end(id); }
    catch (const std::exception &cleanup) { std::cerr << "FC benchmark cleanup: " << cleanup.what() << '\n'; }
    std::rethrow_exception(failure);
  }
  return result;
}

void print(const Result &r, uint32_t index, std::span<const uint32_t> prompt) {
  std::cout << std::setprecision(12)
      << "{\"event\":\"run\",\"run_index\":" << index << ",\"mode\":" << std::quoted(r.mode)
      << ",\"prefill_schedule\":" << std::quoted(r.prefillSchedule)
      << ",\"replay_boundary\":" << r.replayBoundary
      << ",\"production_preflight_bypassed\":true,\"fresh_model_runtime\":true,\"warmup\":false"
      << ",\"context_tokens\":" << ContextTokens << ",\"max_output_tokens\":" << MaximumOutput
      << ",\"kv_format\":\"int8\",\"temperature\":0,\"input_tokens\":" << prompt.size()
      << ",\"input_token_sha256\":" << std::quoted(tokenDigest(prompt))
      << ",\"source_identity\":" << std::quoted(r.sourceIdentity) << ",\"device\":" << std::quoted(r.device)
      << ",\"produced_tokens\":" << r.output.size()
      << ",\"prefill_output_tokens\":" << r.prefillProduced
      << ",\"decode_output_tokens\":" << r.output.size() - r.prefillProduced
      << ",\"committed_tokens\":" << r.committed
      << ",\"drafted_tokens\":" << r.drafted << ",\"accepted_draft_tokens\":" << r.accepted
      << ",\"cycles\":" << r.cycles << ",\"acceptance_rate\":" << (r.drafted ? double(r.accepted) / r.drafted : 0.0)
      << ",\"mean_tokens_per_cycle\":" << (r.cycles ? double(r.output.size() - r.prefillProduced) / r.cycles : 0.0)
      << ",\"setup_wall_seconds\":" << r.setupWall << ",\"prefill_wall_seconds\":" << r.prefillWall
      << ",\"prefill_gpu_seconds\":" << r.prefillGpu
      << ",\"prefill_tok_s_wall\":" << (r.prefillWall ? prompt.size() / r.prefillWall : 0.0)
      << ",\"prefill_tok_s_gpu\":" << (r.prefillGpu ? prompt.size() / r.prefillGpu : 0.0)
      << ",\"decode_wall_seconds\":" << r.decodeWall << ",\"decode_gpu_seconds\":" << r.decodeGpu
      << ",\"decode_tok_s_wall\":" << (r.decodeWall ? (r.output.size() - r.prefillProduced) / r.decodeWall : 0.0)
      << ",\"decode_tok_s_gpu\":" << (r.decodeGpu ? (r.output.size() - r.prefillProduced) / r.decodeGpu : 0.0)
      << ",\"request_wall_seconds\":" << r.requestWall
      << ",\"extra_fc_allocated_bytes\":" << r.extraFcBytes << ",\"target_weight_bytes\":" << r.targetWeightBytes
      << ",\"draft_weight_bytes\":" << r.draftWeightBytes << ",\"backend_allocated_bytes\":" << r.allocatedBytes
      << ",\"backend_peak_bytes\":" << r.peakBytes << ",\"device_peak_bytes\":" << r.devicePeakBytes
      << ",\"finish\":" << std::quoted(r.stopped ? "stop" : "length")
      << ",\"output_token_sha256\":" << std::quoted(tokenDigest(r.output))
      << ",\"prefill_chunks\":[";
  for (size_t i = 0; i < r.prefillChunks.size(); ++i) {
    if (i) std::cout << ',';
    const auto &chunk = r.prefillChunks[i];
    std::cout << "{\"prompt_offset\":" << chunk.offset << ",\"count\":" << chunk.rows
              << ",\"gpu_ms\":" << chunk.gpuMilliseconds << '}';
  }
  std::cout << "],\"output_tokens\":[";
  for (size_t i = 0; i < r.output.size(); ++i) { if (i) std::cout << ','; std::cout << r.output[i]; }
  std::cout << "]}" << std::endl;
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc < 3) throw std::invalid_argument("usage: drafter-fc-precision METALLIB MODEL_ROOT "
          "--prompt-token-file PATH [--prompt-tokens 512|32768] [--mode abba|q4|bf16-fc|header] "
          "[--prefill-schedule boundary|whole]");
      std::filesystem::path promptFile;
      std::string mode = "abba", schedule = "boundary";
      uint32_t expectedTokens = 0;
      for (int i = 3; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::invalid_argument("option needs a value");
        const std::string_view option(argv[i]);
        if (option == "--prompt-token-file") promptFile = argv[i + 1];
        else if (option == "--prompt-tokens") expectedTokens = count(argv[i + 1]);
        else if (option == "--mode") mode = argv[i + 1];
        else if (option == "--prefill-schedule") schedule = argv[i + 1];
        else throw std::invalid_argument("unknown FC benchmark option");
      }
      const std::filesystem::path root(argv[2]);
      if (mode == "header") {
        const auto source = fcSource(root);
        std::cout << "{\"event\":\"fc_header\",\"dtype\":\"BF16\",\"rows\":2560,\"inputs\":20480,"
                     "\"source_bytes\":" << source->require("fc.weight").bytes
                  << ",\"f32_proxy_bytes\":" << FcF32Bytes << ",\"tensor_count\":" << source->tensorCount()
                  << ",\"gpu_used\":false}" << std::endl;
        return 0;
      }
      if (schedule != "boundary" && schedule != "whole")
        throw std::invalid_argument("--prefill-schedule takes boundary or whole");
      if (mode != "abba" && mode != "q4" && mode != "bf16-fc")
        throw std::invalid_argument("--mode takes abba, q4, bf16-fc or header");
      std::ifstream input(promptFile);
      if (!input) throw std::invalid_argument("cannot open prompt token file");
      std::vector<uint32_t> prompt;
      uint32_t token = 0;
      while (input >> token) prompt.push_back(token);
      if (!input.eof() || (prompt.size() != 512 && prompt.size() != 32768) ||
          (expectedTokens && prompt.size() != expectedTokens))
        throw std::invalid_argument("prompt token file must contain exactly 512 or 32768 token IDs");
      std::vector<bool> sequence;
      if (mode == "abba") sequence = {false, true, true, false};
      else sequence = {mode == "bf16-fc"};
      std::vector<uint32_t> reference;
      bool allEqual = true;
      for (size_t i = 0; i < sequence.size(); ++i) {
        std::cout << "{\"event\":\"run_start\",\"run_index\":" << i + 1
                  << ",\"mode\":" << std::quoted(sequence[i] ? "bf16-fc" : "q4") << "}" << std::endl;
        const auto result = run(argv[1], root, prompt, sequence[i], schedule);
        print(result, uint32_t(i + 1), prompt);
        if (!i) reference = result.output;
        else {
          const auto mismatch = std::mismatch(reference.begin(), reference.end(), result.output.begin(), result.output.end());
          const bool equal = reference == result.output;
          allEqual &= equal;
          std::cout << "{\"event\":\"token_comparison\",\"reference_run\":1,\"run_index\":" << i + 1
                    << ",\"tokens_equal\":" << (equal ? "true" : "false")
                    << ",\"first_difference\":" << (equal ? -1 : int64_t(mismatch.first - reference.begin()))
                    << "}" << std::endl;
        }
      }
      std::cout << "{\"event\":\"complete\",\"runs\":" << sequence.size()
                << ",\"comparisons_performed\":" << sequence.size() - 1
                << ",\"all_emitted_token_vectors_equal\":"
                << (sequence.size() == 1 ? "null" : allEqual ? "true" : "false") << "}" << std::endl;
      return allEqual ? 0 : 1;
    } catch (const std::exception &error) {
      std::cerr << "drafter-fc-precision: " << error.what() << '\n';
      return 1;
    }
  }
}
