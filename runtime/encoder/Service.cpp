#include "encoder/Service.hpp"
#include "AwakeClock.hpp"
#include "engine/NativeRuntime.hpp"
#include "engine/Protocol.hpp"
#include "model/ModernBert.hpp"
#include "model/D3.hpp"
#include "model/GgufTarget.hpp"
#include "StderrLine.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <cmath>
#include <sys/select.h>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <unistd.h>

namespace splash::encoder {
using engine::systemUnixMicros;
using engine::EngineFinishReason;
namespace {
std::atomic<bool> stopping{false};
void interruptEncoder(int) { stopping.store(true, std::memory_order_relaxed); }
uint64_t micros() { return static_cast<uint64_t>(awakeMilliseconds() * 1000.0); }
struct Pending {
  protocol::RequestFrame request;
  uint64_t submitted = 0;
};
}

template<class Model>
int serveEncoder(int argc, char **argv, const std::filesystem::path &metallib, std::string_view buildId) {
  constexpr bool d3 = std::is_same_v<Model, model::D3>;
  if (argc < 5 || (argc - 5) % 2)
    throw std::invalid_argument("serve-encoder-native takes MODEL_ROOT CONTEXT|auto MEMORY_BYTES|auto and encoder options");
  const auto amount = [](std::string_view value, uint64_t automatic) {
    if (value == "auto") return automatic;
    if (value.empty() || !std::ranges::all_of(value, [](char c) { return c >= '0' && c <= '9'; }))
      throw std::invalid_argument("encoder context and memory must be unsigned integers or auto");
    return static_cast<uint64_t>(std::stoull(std::string(value)));
  };
  const uint64_t requestedContext = amount(argv[3],8192);
  if (requestedContext < 10 || requestedContext > (d3 ? 262144 : 8192)) throw std::invalid_argument("encoder context is out of range");
  const uint32_t context = static_cast<uint32_t>(requestedContext);
  const uint64_t maximum = amount(argv[4],uint64_t{4} << 30);
  bool selected = true;
  bool q4h = d3;
  uint32_t submitLimit = 4;
  for (int i = 5; i < argc; i += 2) {
    const std::string_view key = argv[i], value = argv[i + 1];
    if (key == "--encoder-option-queries" && (value == "on" || value == "off")) selected = value == "on";
    else if (d3 && key == "--kv-format" && (value == "q4h" || value == "bf16")) q4h = value == "q4h";
    else if (key == "--encoder-submit-batch-size" && (value == "1" || value == "2" || value == "4")) submitLimit = std::stoul(std::string(value));
    else throw std::invalid_argument("invalid encoder option: " + std::string(key));
  }
  if (!context || !maximum) throw std::invalid_argument("invalid encoder context or memory limit");
  protocol::ProtocolLimits limits; limits.maxPromptTokens = context;
  std::mutex mutex, outputMutex; std::condition_variable ready;
  std::deque<Pending> queue;
  std::unordered_set<uint64_t> pending, cancelled;
  std::atomic<bool> failed{false}, loaded{false};
  std::atomic<uint64_t> weightBytes{0}, quantizedBytes{0}, workspaceBytes{0}, allocatedBytes{0}, peakBytes{0}, requests{0}, gpuMicros{0};
  std::atomic<uint64_t> kvBytes{0};
  std::atomic<uint32_t> active{0}, maximumBatch{0};
  std::atomic<uint64_t> batchCount{0};
  const auto send = [&](protocol::EngineEvent event) {
    auto result = protocol::serializeEvent(event, limits);
    if (!result) throw std::runtime_error(result.issue->describe());
    std::lock_guard lock(outputMutex);
    const auto &bytes = *result.value;
    size_t at = 0;
    while (at < bytes.size()) {
      ssize_t written = ::write(STDOUT_FILENO, bytes.data() + at, bytes.size() - at);
      if (written < 0 && errno == EINTR) { if (stopping) throw std::runtime_error("encoder shutdown interrupted output"); continue; }
      if (written <= 0) throw std::runtime_error("encoder protocol output failed");
      at += written;
    }
  };
  const auto error = [&](uint64_t id, std::string message) {
    send(protocol::ErrorEvent{protocol::FailureClass::RequestError, id, false, "invalid_encoder_request", std::move(message)});
  };
  const auto status = [&] {
    std::ostringstream s;
    s << "{\"schema_version\":" << protocol::kStatusSchemaVersion
      << ",\"healthy\":true,\"ready\":" << (loaded ? "true" : "false")
      << ",\"maximum_context_tokens\":" << context
      << ",\"active_requests\":" << active << ",\"model_kind\":\"" << (d3 ? "vllm-d3" : "modern-bert") << "\","
      << "\"weights\":{\"release_supported\":false,\"released\":false,\"restoring\":false},\"memory\":{"
      << "\"loaded_weight_bytes\":" << weightBytes << ",\"quantized_weight_bytes\":" << quantizedBytes << ",\"auxiliary_weight_bytes\":" << weightBytes - quantizedBytes << ",\"fp16_weight_cache_bytes\":0,\"workspace_bytes\":" << workspaceBytes
      << ",\"kv_bytes\":" << kvBytes << ",\"state_bytes\":0,\"allocated_bytes\":" << allocatedBytes << ",\"peak_allocated_bytes\":" << peakBytes
      << ",\"max_memory_bytes\":" << maximum << "},\"telemetry\":{\"completed_requests\":" << requests
      << ",\"submission_batches\":" << batchCount << ",\"maximum_submission_batch\":" << maximumBatch << ",\"total_prefill_gpu_seconds\":" << double(gpuMicros) / 1e6 << "},\"features\":{"
      << "\"drafter\":\"none\",\"ane\":\"off\",\"compact_draft_head\":\"off\",\"fast_topk\":\"off\",\"prompt_lookup\":\"off\","
      << "\"encoder_option_queries\":" << (selected ? "true" : "false") << ",\"encoder_submit_batch_limit\":" << submitLimit << ",\"causal_prefix_cache\":false},"
      << "\"ane_ffn\":{\"state\":\"off\",\"share\":0,\"minimum_rows\":0,\"reason\":\"GPU encoder\"},"
      << "\"identity\":{\"kv\":{\"format\":\"" << (d3 ? (q4h ? "q4h" : "bf16") : "none") << "\"},\"cache\":{\"build_id\":\"" << buildId << "\"}},"
      << "\"memory_governor\":{\"limit_bytes\":" << maximum << "},"
      << "\"warmup\":{\"detail\":\"typed encoder full/selected-head numerical check\"},"
      << "\"startup_features\":{\"drafter\":\"none\",\"draft_layers\":0,\"draft_block_size\":0,\"draft_tokens\":0,"
      << "\"prefill_token_budget\":" << SPLASH_PREFILL_TOKEN_BUDGET << ",\"ffn_cache_bytes\":0,\"optional\":{"
      << "\"prompt_lookup\":false,\"fast_topk\":false,\"compact_draft_head\":false,\"compact_capacity\":0,"
      << "\"compact_validation_maximum_error\":0,\"compact_coverage_fallbacks\":0,\"compact_request_fallbacks\":0}}}";
    return s.str();
  };
  stopping = false;
  std::signal(SIGPIPE, SIG_IGN);
  struct sigaction action{}; action.sa_handler = interruptEncoder; sigemptyset(&action.sa_mask);
  for (int number : {SIGINT, SIGTERM, SIGHUP}) sigaction(number, &action, nullptr);
  std::jthread worker([&](std::stop_token stop) {
    try {
      metal::MetalBackend backend(metallib.string());
      backend.setOperationGuard([] { if (stopping) throw std::runtime_error("encoder shutdown requested"); });
      const auto modelPaths=model::findTargetGgufs(std::filesystem::path(argv[2]) / "target");
      if (modelPaths.size() != 1)
        throw std::invalid_argument("decision models require a single GGUF target");
      const auto &modelPath=modelPaths.front();
      auto owner=[&] {
        if constexpr(d3)return std::make_unique<Model>(backend,modelPath,context,maximum,q4h);
        else return std::make_unique<Model>(backend,modelPath,context,maximum);
      }();
      auto &model=*owner;
      weightBytes = model.weightBytes(); quantizedBytes = model.quantizedWeightBytes(); workspaceBytes = model.workspaceBytes();
      // Startup validates selected queries against the complete last-head path.
      std::vector<uint32_t> prompt, options;
      if constexpr (d3) {
        prompt={248045, 8948, 198, 8, 10, 11, 12, 13, 14, 198};
        options={model.codeTokens()[0],model.codeTokens()[1]};
      } else {
        prompt={model.bosToken(),8,model.sepToken(),model.maskToken(),10,
                model.maskToken(),11,model.sepToken(),12,model.sepToken()};
        options={3,5};
      }
      for (uint32_t kind = 0; kind < 3; ++kind) {
        auto full = model.score(prompt, options, kind, false);
        auto chosen = model.score(prompt, options, kind, true);
        for (size_t i = 0; i < full.size(); ++i)
          if (std::fabs(full[i] - chosen[i]) > 1e-3f * std::max(1.0f, std::fabs(full[i])))
            throw std::runtime_error("encoder selected-query numerical validation failed");
      }
      allocatedBytes = backend.memoryStats().allocatedBytes;
      if constexpr(d3) kvBytes=model.kvBytes();
      peakBytes = backend.memoryStats().peakAllocatedBytes;
      loaded = true;
      logLine(d3 ? "D3: native weights=" : "ModernBERT: quantized weights=", weightBytes.load(), "; workspace=", workspaceBytes.load(),
              "; KV=", d3 ? (q4h ? "q4h" : "bf16") : "none", "; FP16 cache=0; selected option queries=", selected ? "on" : "off", "; numerical check passed");
      send(protocol::ReadyEvent{4, context, false});
      while (!stop.stop_requested() && !stopping) {
        std::vector<Pending> batch;
        {
          std::unique_lock lock(mutex);
          ready.wait_until(lock, AwakeClock::now() + std::chrono::milliseconds(100), [&] { return !queue.empty() || stopping || stop.stop_requested(); });
          if (queue.empty()) continue;
          auto next = std::min_element(queue.begin(), queue.end(), [](const auto &a, const auto &b) {
            return a.request.priority < b.request.priority;
          });
          const auto priority = next->request.priority;
          const auto rows = next->request.promptTokens.size();
          batch.push_back(std::move(*next)); queue.erase(next);
          // Never hold a short request behind a much longer submission batch.
          for (auto it = queue.begin(); it != queue.end() && batch.size() < submitLimit && rows <= 1024;) {
            auto size = it->request.promptTokens.size();
            if (it->request.priority == priority && size <= 1024 && size <= 2 * rows && rows <= 2 * size) {
              batch.push_back(std::move(*it)); it = queue.erase(it);
            } else ++it;
          }
        }
        const auto retire = [&](uint64_t id) {
          std::lock_guard lock(mutex); pending.erase(id); cancelled.erase(id); --active;
        };
        const auto isCancelled = [&](uint64_t id) {
          std::lock_guard lock(mutex); return cancelled.contains(id);
        };
        std::vector<model::ModernBertInput> inputs;
        std::vector<size_t> selectedRequests;
        for (size_t i = 0; i < batch.size(); ++i) {
          auto &item = batch[i]; auto &r = item.request;
          send(protocol::StartEvent{r.requestId, static_cast<uint32_t>(i), 0});
          if (isCancelled(r.requestId)) {
            send(protocol::DoneEvent{r.requestId, EngineFinishReason::Cancelled, 0, 0, 0, 0, micros() - item.submitted, {}});
            retire(r.requestId); continue;
          }
          try {
            if (r.absoluteDeadlineUnixMicros <= systemUnixMicros() || micros() - item.submitted >= r.remainingDeadlineMicros)
              throw std::invalid_argument("encoder request deadline expired");
            uint32_t kind = (r.flags & RequestDecisionScoreType) ? 1 : (r.flags & RequestDecisionNoulType) ? 2 : 0;
            if ((d3 ? bool(r.flags & RequestDecisionScoring) : !(r.flags & RequestDecisionScoring)) ||
                (r.flags & RequestDecisionScoreType && r.flags & RequestDecisionNoulType) ||
                r.logicalMaxOutputTokens || !r.sampling.isNeutral() || r.constraint != ConstraintMode::None ||
                !r.imageSpans.empty() || r.flags & RequestIgnoreEndOfSequence)
              throw std::invalid_argument("encoder serves typed decision scoring only");
            model::ModernBertInput input{r.promptTokens, r.scoreTokens, kind};
            model.validate(input); inputs.push_back(input); selectedRequests.push_back(i);
          } catch (const std::invalid_argument &e) { error(r.requestId, e.what()); retire(r.requestId); }
        }
        if (!inputs.empty()) {
          auto logits = model.scoreBatch(inputs, selected);
          workspaceBytes = model.workspaceBytes();
          if constexpr(d3) kvBytes=model.kvBytes();
          const auto timing = model.lastTiming();
          gpuMicros += static_cast<uint64_t>(timing.gpuSeconds * 1e6);
          requests += inputs.size(); ++batchCount;
          maximumBatch = std::max(maximumBatch.load(), static_cast<uint32_t>(inputs.size()));
          for (size_t slot = 0; slot < selectedRequests.size(); ++slot) {
            auto &item = batch[selectedRequests[slot]]; auto &r = item.request;
            bool cancellation = isCancelled(r.requestId);
            if (cancellation) logits[slot].clear();
            send(protocol::DoneEvent{r.requestId, cancellation ? EngineFinishReason::Cancelled : EngineFinishReason::Stop,
                static_cast<uint32_t>(r.promptTokens.size()), 0, static_cast<uint64_t>(timing.wallSeconds * 1e6), 0,
                micros() - item.submitted, std::move(logits[slot])});
            retire(r.requestId);
          }
        }
        allocatedBytes = backend.memoryStats().allocatedBytes; peakBytes = backend.memoryStats().peakAllocatedBytes;
      }
    } catch (const std::exception &e) {
      if (stopping || stop.stop_requested()) return;
      failed = true; stopping = true;
      writeStderrLine(std::string(d3 ? "error: D3 native failure: " : "error: ModernBERT native failure: ") + e.what());
      try { if (loaded) send(protocol::ErrorEvent{protocol::FailureClass::EngineUnhealthy, 0, false, "encoder_execution_failed", e.what()}); } catch (...) {}
      ready.notify_all();
    }
  });
  protocol::FrameParser parser(limits);
  uint8_t buffer[65536];
  while (!stopping) {
    // Poll bounds shutdown even if the parent keeps its input pipe open.
    fd_set input; FD_ZERO(&input); FD_SET(STDIN_FILENO, &input); timeval timeout{0, 100000};
    int available = select(STDIN_FILENO + 1, &input, nullptr, nullptr, &timeout);
    if (available < 0 && errno == EINTR) continue;
    if (available < 0) throw std::runtime_error("encoder input poll failed");
    if (!available) continue;
    ssize_t count = ::read(STDIN_FILENO, buffer, sizeof buffer);
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) throw std::runtime_error("encoder protocol read failed");
    if (!count) { if (auto issue = parser.finish()) throw std::runtime_error(issue->describe()); break; }
    std::span<const uint8_t> bytes(buffer, static_cast<size_t>(count));
    while (!bytes.empty()) {
      auto step = parser.consume(bytes); bytes = bytes.subspan(step.consumedBytes);
      if (step.issue) throw std::runtime_error(step.issue->describe());
      if (!step.frame) continue;
      auto decoded = protocol::decodeFrame(std::move(*step.frame), limits);
      if (!decoded) {
        if (decoded.issue->failureClass == protocol::FailureClass::RequestError) { error(decoded.issue->requestId, decoded.issue->message); continue; }
        throw std::runtime_error(decoded.issue->describe());
      }
      std::visit([&](auto &&message) {
        using T = std::remove_cvref_t<decltype(message)>;
        if constexpr (std::is_same_v<T, protocol::RequestFrame>) {
          std::lock_guard lock(mutex);
          if (!loaded || pending.size() >= 4 || pending.contains(message.requestId)) { error(message.requestId, "encoder is not ready, full, or request id is repeated"); return; }
          pending.insert(message.requestId); ++active; queue.push_back({std::move(message), micros()}); ready.notify_one();
        } else if constexpr (std::is_same_v<T, protocol::CancelFrame>) {
          std::lock_guard lock(mutex); if (pending.contains(message.requestId)) cancelled.insert(message.requestId);
        } else if constexpr (std::is_same_v<T, protocol::StatusRequestFrame> ||
                             std::is_same_v<T, protocol::ReleaseRequestFrame>) {
          send(protocol::StatusJsonEvent{message.correlationId, status()});
        } else { error(message.requestId, "encoder does not accept token masks"); }
      }, std::move(*decoded.value));
    }
  }
  stopping = true; worker.request_stop(); ready.notify_all(); worker.join();
  return failed ? 70 : 0;
}
int serveModernBert(int argc,char **argv,const std::filesystem::path &lib,std::string_view id) {
  return serveEncoder<model::ModernBert>(argc,argv,lib,id);
}
int serveD3(int argc,char **argv,const std::filesystem::path &lib,std::string_view id) {
  return serveEncoder<model::D3>(argc,argv,lib,id);
}
} // namespace splash::encoder
