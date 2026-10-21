// JSON matrix adapter. Correctness mode emits no timings and never calibrates ANE.
#include "AwakeClock.hpp"
#include "engine/Engine.hpp"
#include "engine/RuntimeResources.hpp"
#include "model/Runtime.hpp"
#include "FfnTrace.hpp"
#import <Foundation/Foundation.h>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace splash;
using namespace splash::engine;
namespace {
using Clock = AwakeClock;
uint32_t tailSplits = 1;
uint64_t tailPlanCalls = 0;
double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
std::string string(NSString *value) {
  if (![value isKindOfClass:[NSString class]]) throw std::invalid_argument("missing string");
  return value.UTF8String;
}
kv::Format format(NSString *name) {
  for (kv::Format candidate : {kv::Format::Int8, kv::Format::BFloat16,
      kv::Format::Q4_0Hadamard, kv::Format::Q4_0Unrotated, kv::Format::Int4Hadamard})
    if (kv::formatName(candidate) == string(name)) return candidate;
  throw std::invalid_argument("unknown KV format");
}
NSDictionary *memoryReport(RuntimeResources &resources, const ActualMemoryReport &report) {
  const auto &loaded = resources.modelContext().model;
  const uint64_t cache = loaded.targetFfnCacheAllocatedBytes();
  const uint64_t total = report.targetWeightsBytes + report.draftWeightsBytes + report.visionWeightsBytes +
      report.stateAllocatedBytes + report.sharedPrefillBytes + report.sharedDecodeBytes + report.kvAllocatedBytes +
      report.stateStagingBytes + report.aneFfnBytes;
  if (total > report.backendAllocatedBytes || cache > report.targetWeightsBytes)
    throw std::logic_error("FFN memory categories exceed tracked allocation");
  uint32_t cached = 0;
  uint64_t quantizedPayload = 0;
  NSMutableDictionary *formats = [NSMutableDictionary dictionary];
  std::visit([&](const auto &target) {
    for (const auto &layer : target.layers) {
      if constexpr (requires { layer.gateProjection; }) {
        for (const auto *projection : {&layer.gateProjection, &layer.upProjection, &layer.downProjection}) {
          bool hasCache = false;
          for (const auto &segment : projection->blocks().segments) {
            hasCache |= bool(segment.prefillHalf);
            if (segment.isFloat()) continue;
            quantizedPayload += segment.plane0.sizeBytes() + segment.plane1.sizeBytes() + segment.meta.sizeBytes();
            NSString *name = [NSString stringWithUTF8String:segment.name()];
            formats[name] = @([formats[name] unsignedIntValue] + 1);
          }
          cached += hasCache;
        }
      }
    }
  }, loaded.target);
  const uint64_t draftState = resources.modelContext().stateStorage.draftAllocatedBytes();
  if (draftState > report.stateAllocatedBytes) throw std::logic_error("draft state exceeds total state");
  return @{
    @"target_source_weights_bytes":@(report.targetWeightsBytes - cache),
    @"ffn_fp16_cache_bytes":@(cache), @"ffn_cached_matrices":@(cached),
    @"ffn_quantized_payload_bytes":@(quantizedPayload), @"ffn_formats":formats,
    @"draft_weights_bytes":@(report.draftWeightsBytes), @"vision_weights_bytes":@(report.visionWeightsBytes),
    @"state_bytes":@(report.stateAllocatedBytes), @"draft_state_bytes":@(draftState),
    @"prefill_workspace_bytes":@(report.sharedPrefillBytes), @"decode_workspace_bytes":@(report.sharedDecodeBytes),
    @"kv_bytes":@(report.kvAllocatedBytes), @"state_staging_bytes":@(report.stateStagingBytes),
    @"ane_bytes":@(report.aneFfnBytes), @"unclassified_bytes":@(report.backendAllocatedBytes - total),
    @"backend_current_bytes":@(report.backendAllocatedBytes), @"backend_peak_bytes":@(report.backendPeakAllocatedBytes),
    @"device_current_bytes":@(report.deviceCurrentAllocatedBytes), @"device_peak_bytes":@(report.devicePeakAllocatedBytes)};
}

NSArray *pathReport(const bench::FfnTrace::Counts &counts) {
  NSMutableArray *rows = [NSMutableArray array];
  for (const auto &[key, count] : counts) {
    const auto &[scope, kernel, inputRows] = key;
    [rows addObject:@{@"scope":[NSString stringWithUTF8String:scope.c_str()],
      @"kernel":[NSString stringWithUTF8String:kernel.c_str()], @"rows":@(inputRows), @"submitted_dispatches":@(count)}];
  }
  return rows;
}
model::FfnWeightMode weightMode(NSString *value) {
  if ([value isEqual:@"full_fp16"]) return model::FfnWeightMode::FullFp16;
  if ([value isEqual:@"quantized"]) return model::FfnWeightMode::Quantized;
  if ([value isEqual:@"limited_fp16"]) return model::FfnWeightMode::LimitedFp16;
  throw std::invalid_argument("unknown FFN weight mode");
}
ops::FfnPrefillVariant prefillVariant(NSString *value) {
  if ([value isEqual:@"baseline"]) return ops::FfnPrefillVariant::Baseline;
  if ([value isEqual:@"m64n32"]) return ops::FfnPrefillVariant::M64N32;
  if ([value isEqual:@"fused_m64n32"]) return ops::FfnPrefillVariant::FusedM64N32;
  if ([value isEqual:@"q4k_group_scale"]) return ops::FfnPrefillVariant::Q4KGroupScale;
  throw std::invalid_argument("unknown FFN prefill variant");
}

struct Observation {
  std::vector<uint32_t> output;
  uint32_t cached = 0;
  bool complete = false;
  Clock::time_point submitted{}, prefillEnd{}, first{}, end{};
};
class Events final : public EngineEventSink {
public:
  std::vector<Observation> requests;
  NSMutableArray *widths = [NSMutableArray array];
  double prefill = 0, decode = 0;
  uint64_t drafted = 0, accepted = 0;
  uint32_t promptTokens;
  std::function<void()> prefillMemory;
  Events(uint32_t count, uint32_t input) : requests(count), promptTokens(input) {}
  void batchCompleted(WorkKind kind, uint32_t width, uint32_t, uint32_t,
                      uint32_t proposals, uint32_t accepts, double, double cycle) override {
    if (kind == WorkKind::Decode) {
      [widths addObject:@(width)]; decode += cycle / 1000;
    } else prefill += cycle / 1000;
    drafted += proposals; accepted += accepts;
  }
  void started(uint64_t id, uint32_t matched, uint32_t) override { requests.at(id - 1).cached = matched; }
  void promptProgress(uint64_t id, uint32_t processed) override {
    if (processed == promptTokens) {
      requests.at(id - 1).prefillEnd = Clock::now();
#ifdef SPLASH_MATRIX_SWIFT27B
      std::cerr << "PREFILL_COMPLETE request=" << id << " tokens=" << processed << '\n';
#endif
      if (prefillMemory) prefillMemory();
    }
  }
  void tokens(uint64_t id, std::span<const uint32_t> tokens) override {
    auto &request = requests.at(id - 1);
    if (request.output.empty()) request.first = Clock::now();
#ifdef SPLASH_MATRIX_SWIFT27B
    const size_t previous = request.output.size();
#endif
    request.output.insert(request.output.end(), tokens.begin(), tokens.end());
#ifdef SPLASH_MATRIX_SWIFT27B
    if (request.output.size() / 2048 > previous / 2048)
      std::cerr << "OUTPUT_PROGRESS request=" << id << " tokens=" << request.output.size() << '\n';
#endif
  }
  void maskRequested(uint64_t, std::span<const uint32_t>) override { throw std::logic_error("unexpected mask"); }
  void completed(uint64_t id, EngineFinishReason reason, bool bounded, uint32_t, uint32_t count,
                 uint32_t, std::span<const float>, const SpeculativeCounts &) override {
    auto &request = requests.at(id - 1);
    if (reason != EngineFinishReason::Length || bounded || count != request.output.size())
      throw std::runtime_error("forced generation did not finish at length");
    request.complete = true; request.end = Clock::now();
  }
  void failed(uint64_t, LaneOutcome outcome, std::string message, uint32_t) override {
    throw std::runtime_error(std::string(laneOutcomeWire(outcome).code) + ": " + message);
  }
};
NSDictionary *run(NSDictionary *payload) {
  NSDictionary *parameters = payload[@"job"][@"parameters"];
  NSArray *prompts = payload[@"prompts"];
  if (![payload[@"mode"] isEqual:@"performance"] && ![payload[@"mode"] isEqual:@"correctness"])
    throw std::invalid_argument("explicit performance/correctness mode is required");
  const bool performance = [payload[@"mode"] isEqual:@"performance"];
  const bool featureCorrectness = [payload[@"optional_features_correctness"] boolValue];
#ifdef SPLASH_MATRIX_SWIFT27B
  constexpr uint32_t correctnessOutputLimit = 16384;
#else
  constexpr uint32_t correctnessOutputLimit = 2048;
#endif
  const uint32_t input = [parameters[@"input_tokens"] unsignedIntValue];
  const uint32_t output = [parameters[@"output_tokens"] unsignedIntValue];
  const uint32_t width = [parameters[@"concurrency"] unsignedIntValue];
  if (featureCorrectness && (performance || input > 32768 || output > correctnessOutputLimit))
    throw std::invalid_argument("optional feature correctness requires correctness mode and input<=32768/output<=" + std::to_string(correctnessOutputLimit));
  tailSplits = [parameters[@"tail_prefill_splits"] unsignedIntValue];
  if (!width || width > 4 || prompts.count != width || !tailSplits || tailSplits > 16 ||
      ![parameters[@"ignore_eos"] boolValue] || ![parameters[@"cold_requests"] boolValue] ||
      [parameters[@"auto_calibration"] boolValue] || (!featureCorrectness && [parameters[@"temperature"] doubleValue] != 0))
    throw std::invalid_argument("invalid fixed matrix contract");
  const bool cacheExperiment = parameters[@"ffn_weight_mode"] != nil;
#ifdef SPLASH_MATRIX_SWIFT27B
  constexpr bool reportMemory = true;
#else
  const bool reportMemory = cacheExperiment;
#endif
  const bool stageProfile = [parameters[@"ffn_stage_profile"] boolValue];
  if (stageProfile && (!performance || !cacheExperiment)) throw std::invalid_argument("stage profiling is a separate explicit performance diagnostic");
  const bool longCorrectness = [payload[@"ffn_long_correctness"] boolValue];
  if (longCorrectness && (performance || !cacheExperiment || width != 1 || input != 32768 || (output != 512 && output != 2048)))
    throw std::invalid_argument("long FFN correctness requires correctness mode, C1 and exact32768 input/512 or2048 output");
  if (!performance && !longCorrectness && !featureCorrectness && (input > (cacheExperiment ? 2048U : 128U) || output > 32))
    throw std::invalid_argument("correctness limit is input128/output32, or input2048 with explicit FFN cache mode");
#ifdef SPLASH_MATRIX_SWIFT27B
  if (performance && (input != 32768 || output != 16384))
    throw std::invalid_argument("Swift performance requires exact32768 input and16384 output");
#else
  if (performance && (input != 32768 || (output != 512 && output != 2048)))
    throw std::invalid_argument("performance requires exact32768 input and512/2048 output");
#endif
  const bool draft = [parameters[@"drafter_enabled"] boolValue];
  if (draft && [parameters[@"drafter_block_size"] unsignedIntValue] != model::ExecutionLimits::draftQueryRows)
    throw std::invalid_argument("binary draft block mismatch");
  RuntimeResourcesConfig config;
  NSDictionary *featureSettings = payload[@"optional_features"];
  config.optionalFeatures.promptLookup = [featureSettings[@"prompt_lookup"] boolValue];
  config.optionalFeatures.compactDraftHead = [featureSettings[@"compact_draft_head"] boolValue];
  config.optionalFeatures.fastTopK = [featureSettings[@"fast_topk"] boolValue];
  if (featureSettings[@"compact_draft_ranking"])
    config.optionalFeatures.compactRanking = string(featureSettings[@"compact_draft_ranking"]);
  if (featureSettings[@"lookup_minimum_match"])
    config.optionalFeatures.lookupMinimumMatch = [featureSettings[@"lookup_minimum_match"] unsignedIntValue];
  if (featureSettings[@"compact_capacity"])
    config.optionalFeatures.compactCapacity = [featureSettings[@"compact_capacity"] unsignedIntValue];
  config.optionalFeatures.validate();
  config.modelRoot = string(payload[@"model_root"]);
  config.metallibPath = string(payload[@"metallib"]);
  config.model = model::inspectModelRoot(config.modelRoot);
#ifdef SPLASH_MATRIX_SWIFT27B
  if (!std::holds_alternative<model::Qwen3_8Layout>(config.model.target) ||
      !draft || config.model.draft.kind != model::DraftKind::DFlash2)
    throw std::invalid_argument("Swift matrix requires Qwen3.8-27B with DFlash2");
  if (cacheExperiment)
    throw std::invalid_argument("4B FFN experiment options are unsupported for Swift 27B");
#else
  if ((!std::holds_alternative<model::Qwen3_5_4BLayout>(config.model.target) &&
       !std::holds_alternative<model::Qwen3_5_4BDFlashLayout>(config.model.target)) ||
      (draft && config.model.draft.kind != model::DraftKind::Standard))
    throw std::invalid_argument("matrix requires Qwen3.5-4B and Standard DFlash or NoDraft");
#endif
  if (config.model.hasDraft() != draft) throw std::invalid_argument("model drafter mismatch");
  NSString *ffnMode = parameters[@"ffn_weight_mode"] ?: @"full_fp16";
  auto trace = std::make_shared<bench::FfnTrace>();
  NSString *ffnVariant = parameters[@"ffn_prefill_variant"] ?: @"baseline";
  config.ffnLoading = {weightMode(ffnMode), [parameters[@"ffn_cache_budget_bytes"] unsignedLongLongValue], prefillVariant(ffnVariant), cacheExperiment};
  if (cacheExperiment) config.submissionObserver = [trace](const metal::Command &command) { trace->submitted(command); };
  config.kvFormat = format(parameters[@"kv_format"]);
  config.buildId = string(payload[@"identity"][@"code_sha256"]);
  config.maximumMemoryBytes = [parameters[@"memory_budget_bytes"] unsignedLongLongValue];
  const bool ane = [parameters[@"ane_requested"] boolValue];
  if (ane && ([parameters[@"ane_share"] doubleValue] != 0.5 ||
              [parameters[@"ane_minimum_rows"] unsignedIntValue] != 512))
    throw std::invalid_argument("ANE must use manual share0.5/minimum512");
  config.aneFfn = AneFfnSetting::fromGiven(ane ? 0.5 : 0.0, 512);
  const uint32_t context = [parameters[@"context_tokens"] unsignedIntValue];
  if (config.optionalFeatures.compactDraftHead && config.optionalFeatures.compactRanking.empty())
    config.optionalFeatures.compactRanking = config.modelRoot / "head-ranked.splash";
  auto resources = RuntimeResources::create(config, context);
  model::Runtime runtime(resources->modelContext());
  const auto setupTrace = trace->snapshot();
  double warmupSeconds = 0;
  // Equalize pipeline/FFN cache preparation before future measurements.
  // These direct runtime requests do not publish a prefix in Engine Cache.
  if (performance) {
    const auto warmupStart = Clock::now();
    static_cast<void>(runtime.warmupPrefill(model::ExecutionLimits::prefillTokenBudget));
    static_cast<void>(runtime.warmupDecodeBatch(width));
    warmupSeconds = seconds(warmupStart);
  }
  EngineConfig engineConfig;
  engineConfig.maxContext = context;
  engineConfig.vocabularySize = std::visit([](const auto &target) { return target.vocabularySize; }, config.model.target);
  connectToGovernor(engineConfig, resources->memoryGovernor());
  if (stageProfile) {
    const auto unavailable = resources->backend().setTimestampObserver([trace](const metal::CommandTimestampProfile &profile) { trace->timed(profile); });
    if (!unavailable.empty()) throw std::runtime_error("stage timestamp profiling unavailable: " + unavailable);
  }
  NSMutableDictionary *memoryPoints = [NSMutableDictionary dictionary];
  if (reportMemory) memoryPoints[@"ready"] = memoryReport(*resources, resources->actualMemoryReport(runtime.actualRuntimeMemory()));
  const auto requestTrace = trace->snapshot();
  std::optional<ActualMemoryReport> prefillMemory;
  double telemetrySeconds = 0;
  Events events(width, input);
  if (reportMemory) events.prefillMemory = [&] {
    const auto sampledAt = Clock::now();
    prefillMemory = resources->actualMemoryReport(runtime.actualRuntimeMemory(), false);
    telemetrySeconds += seconds(sampledAt);
  };
  Engine engine(engineConfig, resources->cache(), runtime, events);
  const auto aneBefore = resources->aneFfnSnapshot();
  const auto start = Clock::now();
  for (uint32_t i = 0; i < width; ++i) {
    EngineRequest request;
    request.id = i + 1; request.maxNewTokens = output; request.returnProgress = true;
    request.deadlineMilliseconds = 1800000.0;
    request.flags = RequestIgnoreEndOfSequence;
    if (featureCorrectness) {
      request.sampling.temperature = [parameters[@"temperature"] floatValue];
      request.sampling.topK = [parameters[@"top_k"] unsignedIntValue];
      request.sampling.topP = parameters[@"top_p"] ? [parameters[@"top_p"] floatValue] : 1.0F;
      request.sampling.seed = [parameters[@"seed"] unsignedLongLongValue];
    }
    NSArray *tokens = prompts[i][@"token_ids"];
    if (tokens.count != input) throw std::invalid_argument("prompt length mismatch");
    for (NSNumber *token in tokens) request.prompt.push_back(token.unsignedIntValue);
    events.requests[i].submitted = Clock::now();
    engine.submit(std::move(request));
  }
  while (!engine.idle()) {
    if (seconds(start) > 1800) throw std::runtime_error("matrix batch timeout");
    if (!engine.tick(seconds(start) * 1000)) std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  const double total = seconds(start);
  const auto memory = resources->backend().memoryStats();
  const auto after = resources->aneFfnSnapshot();
  const bool ranAne = after.state == AneFfnSnapshot::State::Split && after.evaluations > aneBefore.evaluations;
  NSMutableArray *responses = [NSMutableArray array];
  for (uint32_t i = 0; i < width; ++i) {
    const auto &request = events.requests[i];
    if (!request.complete || request.output.size() != output || request.cached || request.prefillEnd == Clock::time_point{})
      throw std::runtime_error("forced output or cold cache assertion failed");
    NSMutableArray *tokens = [NSMutableArray array];
    NSMutableArray *eos = [NSMutableArray array];
    for (uint32_t j = 0; j < request.output.size(); ++j) {
      [tokens addObject:@(request.output[j])];
      if (request.output[j] == 248044 || request.output[j] == 248046) [eos addObject:@(j)];
    }
    NSMutableDictionary *response = [@{@"prompt_id":prompts[i][@"id"],
      @"input_sha256":prompts[i][@"input_sha256"], @"input_tokens":@(input),
      @"output_token_ids":tokens, @"eos_positions":eos, @"cached_input_tokens":@(request.cached),
      @"memory_peak_bytes":@(memory.peakAllocatedBytes)} mutableCopy];
    if (performance) response[@"timings"] = @{
      @"prefill_wall_seconds":@(std::chrono::duration<double>(request.prefillEnd - request.submitted).count()),
      @"decode_wall_seconds":@(std::chrono::duration<double>(request.end - request.prefillEnd).count()),
      @"total_wall_seconds":@(std::chrono::duration<double>(request.end - request.submitted).count()),
      @"ttft_seconds":@(std::chrono::duration<double>(request.first - request.submitted).count())};
    [responses addObject:response];
  }
  NSMutableDictionary *result = [@{@"status":@"complete", @"identity":payload[@"identity"],
    @"parameters":parameters, @"requests":responses, @"compiled_block_size":@(model::ExecutionLimits::draftQueryRows),
    @"compiled_source_sha256":@SPLASH_BUILD_SOURCE_SHA256,
    @"actual_kv_format":[NSString stringWithUTF8String:std::string(kv::formatName(resources->cacheIdentity().kvLayout.format)).c_str()],
    @"actual_context_tokens":@(engine.snapshot().maximumContextTokens),
    @"device_name":[NSString stringWithUTF8String:resources->backend().capabilities().deviceName.c_str()],
    @"model_layout_sha256":[NSString stringWithUTF8String:resources->cacheIdentity().modelLayoutSha256.c_str()], @"actual_batch_widths":events.widths,
    @"actual_drafter_block_size":draft ? @(model::ExecutionLimits::draftQueryRows) : [NSNull null],
    @"actual_tail_prefill_splits":@(tailSplits), @"tail_plan_calls":@(tailPlanCalls),
    @"memory_peak_bytes":@(memory.peakAllocatedBytes), @"drafted_tokens":@(events.drafted),
    @"accepted_draft_tokens":@(events.accepted), @"ane_actual":@{
      @"enabled":@(ranAne), @"share":@(after.share), @"minimum_rows":@(after.minimumRows),
      @"auto_calibration":@NO, @"evaluations":@(after.evaluations - aneBefore.evaluations),
      @"fallback_reason":[NSString stringWithUTF8String:after.reason.empty() ? "no qualifying prefill evaluations" : after.reason.c_str()]}} mutableCopy];
  const auto features = runtime.optionalFeatureReport();
  result[@"target_manifest_sha256"] = [NSString stringWithUTF8String:resources->cacheIdentity().targetModelSha256.c_str()];
  result[@"optional_features"] = @{
    @"configured":@{@"prompt_lookup":@(features.configured.promptLookup),
                    @"compact_draft_head":@(features.configured.compactDraftHead),
                    @"fast_topk":@(features.configured.fastTopK)},
    @"lookup_minimum_match":@(features.configured.lookupMinimumMatch), @"lookup_maximum_match":@4,
    @"compact_requested_capacity":@(features.configured.compactCapacity), @"compact_capacity":@(features.compactCapacity),
    @"lookup_hits":@(features.lookupHits), @"lookup_attempts":@(features.lookupAttempts),
    @"lookup_accepted_tokens":@(features.lookupAccepted),
    @"lookup_fallback_reason":[NSString stringWithUTF8String:features.lookupReason.c_str()],
    @"compact_cycles":@(features.compactCycles), @"compact_fallbacks":@(features.compactFallbacks),
    @"compact_rows":@(features.compactRows),
    @"compact_validated":@(features.compactValidated), @"compact_validation_comparisons":@(features.compactValidationComparisons),
    @"compact_validation_max_error":@(features.compactValidationMaximumError),
    @"compact_coverage_fallbacks":@(features.compactCoverageFallbacks), @"compact_request_fallbacks":@(features.compactRequestFallbacks),
    @"compact_fallback_reason":[NSString stringWithUTF8String:features.compactReason.c_str()],
    @"fast_topk_selections":@(features.fastTopKSelections), @"fast_topk_fallbacks":@(features.fastTopKFallbacks),
    @"fast_topk_fallback_reason":[NSString stringWithUTF8String:features.fastTopKReason.c_str()],
    @"extra_metal_bytes":@(features.extraMetalBytes)};
  if (reportMemory) {
    memoryPoints[@"end"] = memoryReport(*resources, resources->actualMemoryReport(runtime.actualRuntimeMemory()));
    if (prefillMemory) memoryPoints[@"prefill_complete"] = memoryReport(*resources, *prefillMemory);
    result[@"memory_checkpoints"] = memoryPoints;
    result[@"memory_report"] = memoryPoints[@"end"];
#ifdef SPLASH_MATRIX_SWIFT27B
    if ([memoryPoints[@"end"][@"ffn_fp16_cache_bytes"] unsignedLongLongValue] != 0)
      throw std::runtime_error("Swift retained unexpected full FP16 FFN cache");
    result[@"actual_ffn_weight_mode"] = @"existing_quantized_path";
#else
    result[@"actual_ffn_weight_mode"] = ffnMode;
#endif
    result[@"actual_ffn_prefill_variant"] = ffnVariant;
    result[@"ffn_paths"] = pathReport(trace->since(requestTrace));
    result[@"startup_ffn_paths"] = pathReport(setupTrace);
    result[@"warmup_ffn_paths"] = pathReport([&] { bench::FfnTrace::Counts delta; for (const auto &[key, count] : requestTrace) { const auto old = setupTrace.find(key); const auto difference = count - (old == setupTrace.end() ? 0 : old->second); if (difference) delta[key] = difference; } return delta; }());
  }
  if (stageProfile) {
    if (!trace->timingError().empty()) throw std::runtime_error(trace->timingError());
    NSMutableArray *stages = [NSMutableArray array];
    for (const auto &[name, stage] : trace->stages())
      [stages addObject:@{@"scope":[NSString stringWithUTF8String:name.c_str()], @"dispatches":@(stage.dispatches),
        @"gpu_ticks":@(stage.ticks), @"estimated_gpu_seconds":@(stage.estimatedSeconds)}];
    if (!stages.count) throw std::runtime_error("stage timestamp profile has no samples");
    result[@"ffn_stage_profile"] = @{@"diagnostic_only":@YES, @"counter_barriers":@YES,
      @"sampling":trace->encoderBoundarySamples() ? @"encoder_boundary" : @"dispatch_boundary",
      @"seconds_definition":@"GPU timestamp deltas scaled to each command GPU span; event gap includes wait and encoder transition",
      @"gpu_command_span_seconds":@(trace->gpuSpanSeconds()), @"stages":stages};
  }
  if (performance) {
    if (!tailPlanCalls) throw std::runtime_error("tail split override was not exercised");
    const auto startup = resources->startupTimings();
    const auto loaded = resources->modelContext().model.loadTimings;
    result[@"startup_timings"] = @{@"resources_seconds":@(startup.totalSeconds), @"source_weights_seconds":@(loaded.sourceSeconds), @"ffn_cache_build_seconds":@(loaded.ffnCacheSeconds), @"ane_prepare_seconds":@(startup.anePrepareSeconds), @"storage_seconds":@(startup.storageSeconds), @"warmup_seconds":@(warmupSeconds), @"hot_telemetry_seconds":@(telemetrySeconds)};
    result[@"warmup"] = @{@"prefill_rows":@(model::ExecutionLimits::prefillTokenBudget), @"decode_width":@(width), @"iterations":@1};
    result[@"timings"] = @{@"prefill_wall_seconds":@(events.prefill),
      @"decode_wall_seconds":@(events.decode), @"total_wall_seconds":@(total)};
  }
  return result;
}
}
namespace splash::ops {
uint32_t matrixTailSplits() { return tailSplits; }
void matrixRecordTailSplit(uint32_t rows, uint32_t splits) {
  if (rows != 32) return;
  if (splits != tailSplits) throw std::logic_error("encoded tail split does not match matrix setting");
  ++tailPlanCalls;
}
}
int main() {
  @autoreleasepool {
    try {
      NSData *input = [[NSFileHandle fileHandleWithStandardInput] readDataToEndOfFile];
      NSError *error = nil;
      NSDictionary *payload = [NSJSONSerialization JSONObjectWithData:input options:0 error:&error];
      if (!payload || ![payload[@"protocol"] isEqual:@"qwen4b-matrix-v1"] ||
          ![payload[@"operation"] isEqual:@"run_batch"])
        throw std::invalid_argument("invalid matrix JSON protocol");
      NSData *json = [NSJSONSerialization dataWithJSONObject:run(payload) options:0 error:&error];
      if (!json) throw std::runtime_error("cannot encode matrix result");
      std::cout.write(static_cast<const char *>(json.bytes), json.length); std::cout << '\n';
      return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
  }
}
