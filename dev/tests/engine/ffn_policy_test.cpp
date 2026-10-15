#include "TestChecks.hpp"
#include "benchmarks/FfnTrace.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/Gguf.h"
#include "ops/Linear.hpp"
#include "model/ModelFactory.hpp"
#include <cmath>
#include <iostream>

using namespace splash;
int main(int argc, char **argv) {
  ops::Projection down(2560, 9216, ops::BlockWeights{{ops::QuantizedSegment::planes(GGUF_FMT_Q4K, 2560, 9216, {}, {}, {})}});
  down.ffn = {ops::FfnRole::Down, ops::FfnPrefillVariant::M64N32, true};
  const auto view = down.leadingInputs(4608);
  test::require(view.inputSize == 4608 && view.planeInputs() == 9216 && view.ffn.leadingView &&
      view.ffn.role == ops::FfnRole::Down && view.ffn.variant == ops::FfnPrefillVariant::M64N32 && view.ffn.trace,
      "ANE down view lost logical K, physical stride or FFN policy");
  DeviceCapabilities device;
  device.appleGpuFamily = 10; device.gpuCoreCount = 20;
  ops::Linear linear(device);
  const auto small = linear.prefillPlan(down, 8, ops::LinearEpilogue::Residual);
  const auto decode = linear.decodePlan(down, 1, ops::LinearEpilogue::Residual);
  test::require(small.workload().phase == ops::LinearPhase::Prefill && decode.workload().phase == ops::LinearPhase::Decode,
      "small prefill must not stand for a real decode plan");
  const auto large = linear.prefillPlan(view, 512, ops::LinearEpilogue::Residual);
  test::require(ops::ffnPrefillKernel(view, large) == ops::FfnPrefillKernel::M64N32 &&
      ops::ffnPrefillKernel(view, decode) == ops::FfnPrefillKernel::Baseline,
      "target FFN candidate leaked into decode or did not reach ANE views");
  auto ordinary = down; ordinary.ffn.role = ops::FfnRole::None;
  test::require(ops::ffnPrefillKernel(ordinary, large) == ops::FfnPrefillKernel::Baseline,
      "FFN candidate changed an unrelated projection");
  ops::Projection gate(9216, 2560, ops::BlockWeights{{ops::QuantizedSegment::planes(GGUF_FMT_Q4K, 9216, 2560, {}, {}, {})}});
  auto up = gate;
  gate.ffn = {ops::FfnRole::Gate, ops::FfnPrefillVariant::FusedM64N32};
  up.ffn = {ops::FfnRole::Up, ops::FfnPrefillVariant::FusedM64N32};
  test::require(ops::ffnPrefillGateUpEligible({&gate,&up,&down},64) &&
      !ops::ffnPrefillGateUpEligible({&gate,&up,&down},33), "fused FFN eligibility ignored its row limit");
  auto iq = ops::Projection(9216,2560,ops::BlockWeights{{ops::QuantizedSegment::planes(GGUF_FMT_IQ4XS,9216,2560,{},{},{})}});
  iq.ffn = {ops::FfnRole::Up,ops::FfnPrefillVariant::Q4KGroupScale};
  test::require(ops::ffnPrefillKernel(iq,linear.prefillPlan(iq,512,ops::LinearEpilogue::None)) == ops::FfnPrefillKernel::Baseline,
      "group-scale candidate treated a codebook format as uniform integers");
  metal::CommandGraph graph;
  graph.add("unrelated_attention", {}, metal::DispatchSize{1,1,1});
  const size_t start = graph.dispatches().size();
  graph.add("gguf_prefill_q4k_r_leading_inputs", {}, metal::DispatchSize{1,1,1});
  graph.endCommandBuffer();
  test::require(!graph.dispatches().front().commandBufferBoundaryAfter && graph.dispatches().back().commandBufferBoundaryAfter,
      "explicit command-buffer boundary moved to another dispatch");
  graph.attributeFrom(start, "target_ffn/prefill/ane_view", 512);
  bench::FfnTrace trace;
  const auto before = trace.snapshot();
  trace.submitted(graph.command());
  auto counts = trace.since(before);
  test::require(counts.size() == 1 && counts.begin()->second == 1 &&
      std::get<2>(counts.begin()->first) == 512,
      "submission trace included unrelated work or dropped logical rows");
  trace.submitted(graph.command());
  test::require(trace.since(before).begin()->second == 2, "submission counts are not cumulative");
  metal::CommandTimestampProfile timestamps;
  timestamps.valid = true; timestamps.encoderBoundarySamples = true; timestamps.gpuSpanSeconds = .006;
  timestamps.dispatches = {{"weights","target_ffn/ane_weight_staging",0,100,110,false},
      {"ffn","target_ffn/prefill/ane_view",512,110,130,false}, {"join","target_ffn/ane_join",512,150,160,true}};
  trace.timed(timestamps);
  test::require(std::fabs(trace.stages().at("event_wait_and_encoder_gap").estimatedSeconds-.002) < 1e-12 &&
      trace.encoderBoundarySamples(), "diagnostic profiling lost event gaps or sample method");
  if (argc > 1) {
    const auto descriptor = model::inspectModelRoot(argv[1]);
    const uint64_t original = model::modelWeightBytes(argv[1], descriptor);
    const uint64_t quantized = model::modelWeightBytes(argv[1], descriptor, {.mode=model::FfnWeightMode::Quantized});
    const uint64_t matrixBytes = uint64_t{2560}*9216*2;
    test::require(original-quantized == 96*matrixBytes, "actual GGUF footprint does not identify 96 FFN caches");
    for (const uint64_t limit : {matrixBytes-1,matrixBytes,uint64_t{1}<<30}) {
      const uint64_t limited = model::modelWeightBytes(argv[1], descriptor, {.mode=model::FfnWeightMode::LimitedFp16,.budgetBytes=limit});
      test::require(limited-quantized == (limit/matrixBytes)*matrixBytes && limited<=original,
          "limited cache footprint did not select complete matrices within its budget");
    }
  }
  std::cout << "FFN policy, leading-input stride and submission attribution passed\n";
}
