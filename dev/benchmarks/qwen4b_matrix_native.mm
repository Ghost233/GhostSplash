// JSON matrix adapter. Correctness mode emits no timings and never calibrates ANE.
#include "engine/Engine.hpp"
#include "engine/RuntimeResources.hpp"
#include "model/Runtime.hpp"
#import <Foundation/Foundation.h>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace splash;
using namespace splash::engine;
namespace {
using Clock = std::chrono::steady_clock;
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
    if (processed == promptTokens) requests.at(id - 1).prefillEnd = Clock::now();
  }
  void tokens(uint64_t id, std::span<const uint32_t> tokens) override {
    auto &request = requests.at(id - 1);
    if (request.output.empty()) request.first = Clock::now();
    request.output.insert(request.output.end(), tokens.begin(), tokens.end());
  }
  void maskRequested(uint64_t, std::span<const uint32_t>) override { throw std::logic_error("unexpected mask"); }
  void completed(uint64_t id, EngineFinishReason reason, uint32_t, uint32_t count,
                 std::span<const float>) override {
    auto &request = requests.at(id - 1);
    if (reason != EngineFinishReason::Length || count != request.output.size())
      throw std::runtime_error("forced generation did not finish at length");
    request.complete = true; request.end = Clock::now();
  }
  void failed(uint64_t, LaneOutcome outcome, std::string message) override {
    throw std::runtime_error(std::string(laneOutcomeWire(outcome).code) + ": " + message);
  }
};
NSDictionary *run(NSDictionary *payload) {
  NSDictionary *parameters = payload[@"job"][@"parameters"];
  NSArray *prompts = payload[@"prompts"];
  if (![payload[@"mode"] isEqual:@"performance"] && ![payload[@"mode"] isEqual:@"correctness"])
    throw std::invalid_argument("explicit performance/correctness mode is required");
  const bool performance = [payload[@"mode"] isEqual:@"performance"];
  const uint32_t input = [parameters[@"input_tokens"] unsignedIntValue];
  const uint32_t output = [parameters[@"output_tokens"] unsignedIntValue];
  const uint32_t width = [parameters[@"concurrency"] unsignedIntValue];
  tailSplits = [parameters[@"tail_prefill_splits"] unsignedIntValue];
  if (!width || width > 4 || prompts.count != width || !tailSplits || tailSplits > 16 ||
      ![parameters[@"ignore_eos"] boolValue] || ![parameters[@"cold_requests"] boolValue] ||
      [parameters[@"auto_calibration"] boolValue] || [parameters[@"temperature"] doubleValue] != 0)
    throw std::invalid_argument("invalid fixed matrix contract");
  if (!performance && (input > 128 || output > 32))
    throw std::invalid_argument("correctness runs limited to input128/output32; long runs need explicit performance mode");
  if (performance && (input != 32768 || (output != 512 && output != 2048)))
    throw std::invalid_argument("performance requires exact32768 input and512/2048 output");
  const bool draft = [parameters[@"drafter_enabled"] boolValue];
  if (draft && [parameters[@"drafter_block_size"] unsignedIntValue] != model::ExecutionLimits::draftQueryRows)
    throw std::invalid_argument("binary draft block mismatch");
  RuntimeResourcesConfig config;
  config.modelRoot = string(payload[@"model_root"]);
  config.metallibPath = string(payload[@"metallib"]);
  config.model = model::inspectModelRoot(config.modelRoot);
  if ((!std::holds_alternative<model::Qwen3_5_4BLayout>(config.model.target) &&
       !std::holds_alternative<model::Qwen3_5_4BDFlashLayout>(config.model.target)) ||
      (draft && config.model.draft.kind != model::DraftKind::Standard))
    throw std::invalid_argument("matrix requires Qwen3.5-4B and Standard DFlash or NoDraft");
  if (config.model.hasDraft() != draft) throw std::invalid_argument("model drafter mismatch");
  config.kvFormat = format(parameters[@"kv_format"]);
  config.buildId = string(payload[@"identity"][@"code_sha256"]);
  config.maximumMemoryBytes = [parameters[@"memory_budget_bytes"] unsignedLongLongValue];
  const bool ane = [parameters[@"ane_requested"] boolValue];
  if (ane && ([parameters[@"ane_share"] doubleValue] != 0.5 ||
              [parameters[@"ane_minimum_rows"] unsignedIntValue] != 512))
    throw std::invalid_argument("ANE must use manual share0.5/minimum512");
  config.aneFfn = AneFfnSetting::fromGiven(ane ? 0.5 : 0.0, 512);
  const uint32_t context = [parameters[@"context_tokens"] unsignedIntValue];
  auto resources = RuntimeResources::create(config, context);
  model::Runtime runtime(resources->modelContext());
  // Equalize pipeline/FFN cache preparation before future measurements.
  // These direct runtime requests do not publish a prefix in Engine Cache.
  if (performance) {
    static_cast<void>(runtime.warmupPrefill(model::ExecutionLimits::prefillTokenBudget));
    static_cast<void>(runtime.warmupDecodeBatch(width));
  }
  EngineConfig engineConfig;
  engineConfig.maxContext = context;
  engineConfig.vocabularySize = std::visit([](const auto &target) { return target.vocabularySize; }, config.model.target);
  connectToGovernor(engineConfig, resources->memoryGovernor());
  Events events(width, input);
  Engine engine(engineConfig, resources->cache(), runtime, events);
  const auto aneBefore = resources->aneFfnSnapshot();
  const auto start = Clock::now();
  for (uint32_t i = 0; i < width; ++i) {
    EngineRequest request;
    request.id = i + 1; request.maxNewTokens = output; request.returnProgress = true;
    request.deadlineMilliseconds = 1800000.0;
    request.flags = RequestIgnoreEndOfSequence;
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
  if (performance) {
    if (!tailPlanCalls) throw std::runtime_error("tail split override was not exercised");
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
