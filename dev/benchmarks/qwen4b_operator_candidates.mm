#include "model/ModelFactory.hpp"
#include "ops/Linear.hpp"
#include "ops/SingleTokenLinear.hpp"
#include "ops/GgufPrefillCache.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using namespace splash;
struct Operand { const ops::Projection *projection; ops::LinearEpilogue epilogue; };
struct Case { std::string label; metal::CommandGraph graph; std::vector<double> milliseconds; };
int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 5 && argc != 6) throw std::invalid_argument("usage: qwen4b-operator-candidates METALLIB MODEL_ROOT decode|prefill ROUNDS [--m64n32]");
      const bool prefill = std::string(argv[3]) == "prefill";
      if (!prefill && std::string(argv[3]) != "decode") throw std::invalid_argument("invalid phase");
      const bool alternatePrefill = argc == 6;
      if (alternatePrefill && (!prefill || std::string(argv[5]) != "--m64n32"))
        throw std::invalid_argument("--m64n32 is a prefill candidate only");
      const uint32_t rounds = std::stoul(argv[4]);
      if (!rounds) throw std::invalid_argument("rounds must be positive");
      metal::MetalBackend backend(argv[1]);
      auto model = model::loadModel(backend, argv[2], model::inspectModelRoot(argv[2]));
      const std::vector<model::Qwen3_8LayerWeights> *layers = nullptr;
      const ops::Projection *head = nullptr;
      if (const auto *target = std::get_if<model::Qwen3_5_4BWeights>(&model.target)) {
        layers = &target->layers; head = &target->logitsProjection;
      } else if (const auto *target = std::get_if<model::Qwen3_5_4BDFlashWeights>(&model.target)) {
        layers = &target->layers; head = &target->logitsProjection;
      } else throw std::invalid_argument("requires Qwen3.5-4B");
      const ops::Linear linear(backend.capabilities());
      std::vector<Operand> operands;
      for (const auto &layer : *layers) {
        if (!prefill) std::visit([&](const auto &mixer) {
          operands.push_back({&mixer.inputProjection, ops::LinearEpilogue::None});
          operands.push_back({&mixer.outputProjection, ops::LinearEpilogue::Residual});
        }, layer.mixer);
        operands.push_back({&layer.gateProjection, ops::LinearEpilogue::None});
        operands.push_back({&layer.upProjection, ops::LinearEpilogue::UpWithGate});
        operands.push_back({&layer.downProjection, ops::LinearEpilogue::Residual});
      }
      if (!prefill) operands.push_back({head, ops::LinearEpilogue::None});
      uint32_t maxK=0, maxN=0;
      ops::LinearScratchSize scratchSize;
      for (const auto &operand : operands) {
        const auto &p = *operand.projection;
        maxK=std::max(maxK,p.inputSize); maxN=std::max(maxN,p.outputSize);
        const auto plan = p.destination == ops::FloatOutput::Float32 ? linear.decodePlan(p,1) :
                          linear.prefillPlan(p,prefill ? 2048 : 8,operand.epilogue);
        scratchSize.include(plan.scratchSize());
      }
      const uint32_t rows=prefill ? 2048 : 8;
      auto allocate = [&](uint64_t bytes) {
        auto b=backend.allocateBuffer(std::max<uint64_t>(bytes,4),metal::BufferStorage::Shared,"candidate operand");
        std::memset(b.contents(),0,b.sizeBytes()); return b;
      };
      auto input=allocate(uint64_t{rows}*maxK*2), output=allocate(uint64_t{rows}*maxN*4), auxiliary=allocate(uint64_t{rows}*maxN*2);
      auto partials=allocate(std::max(scratchSize.partials,uint64_t{maxN}*4*4));
      ops::LinearScratch scratch{allocate(scratchSize.input),allocate(scratchSize.sums),partials,allocate(scratchSize.counters),allocate(scratchSize.rotated)};
      std::mt19937 random(71); std::uniform_real_distribution<float> unit(-0.5F,0.5F);
      for (auto b : {input,auxiliary}) { auto *x=static_cast<__bf16 *>(b.contents()); for (uint64_t i=0;i<b.sizeBytes()/2;++i) x[i]=__bf16(unit(random)); }
      std::vector<Case> cases;
      auto make = [&](std::string label,uint32_t step,uint32_t columns,uint32_t splits,
                      ops::GgufHalfPrefillTile tile=ops::GgufHalfPrefillTile::M32N64) {
        Case c{std::move(label),{}, {}};
        for (const auto &operand : operands) {
          const auto &p=*operand.projection;
          if (step==0 && columns==0) {
            auto blocks=p.blocks(); for (auto &s:blocks.segments) s.prefillHalf={};
            ops::Projection original(p.outputSize,p.inputSize,std::move(blocks)); original.destination=p.destination;
            const auto plan=p.destination==ops::FloatOutput::Float32 ? linear.decodePlan(original,1) :
                            linear.prefillPlan(original,rows,operand.epilogue);
            linear.add(c.graph,{.input=input,.output=output,.residual=auxiliary,.gateScratch=auxiliary,.scratch=scratch},original,plan);
          } else if (prefill) {
            const char ep=operand.epilogue==ops::LinearEpilogue::None ? 'a' : operand.epilogue==ops::LinearEpilogue::Residual ? 'r' : 'g';
            for (const auto &s:p.blocks().segments)
              ops::addGgufHalfPrefill(c.graph,input,s,output,{p.inputSize,rows,p.outputSize,s.columnOffset},ep,auxiliary,step==1?p.inputSize:step,tile);
          } else {
            auto blocks=p.blocks();
            for (const auto &s:blocks.segments) if (s.isFloat())
              ops::addGgufFloat(c.graph,input,s,output,1,p.outputSize,s.columnOffset,p.destination,ops::FloatTile::Simdgroup);
            std::erase_if(blocks.segments,[](const auto &s){return s.isFloat();});
            if (!blocks.segments.empty()) {
              ops::Projection quantized(p.outputSize,p.inputSize,std::move(blocks)); quantized.destination=p.destination;
              ops::addSingleTokenLinear(c.graph,input,quantized,output,auxiliary,operand.epilogue,partials,splits,columns);
            }
          }
        }
        cases.push_back(std::move(c));
      };
      make("original-staged",0,0,1);
      if (prefill) for (uint32_t step : {64U,128U,256U,512U,1U}) make(step==1?"half-fullK":"half-K"+std::to_string(step),step,0,1);
      if (alternatePrefill) make("half-M64N32-K64",64,0,1,ops::GgufHalfPrefillTile::M64N32);
      if (!prefill) for (uint32_t columns : {32U,64U}) for (uint32_t splits : {1U,2U,4U}) make("single-N"+std::to_string(columns)+"-S"+std::to_string(splits),0,columns,splits);
      for (auto &c:cases) for (double elapsed=0;elapsed<0.1;) elapsed+=backend.submitCommandAsync(c.graph.dispatches()).wait().gpuSeconds;
      for (uint32_t round=0;round<rounds;++round) for (size_t i=0;i<cases.size();++i) {
        auto &c=cases[round%2?cases.size()-i-1:i]; c.milliseconds.push_back(backend.submitCommandAsync(c.graph.dispatches()).wait().gpuSeconds*1000);
      }
      std::printf("%s actual 4B projection stream, %zu projections, %u rounds, %u input rows; excludes cache construction/state/attention\n",argv[3],operands.size(),rounds,prefill?rows:1);
      for (auto &c:cases) { std::sort(c.milliseconds.begin(),c.milliseconds.end()); std::printf("%s median_ms=%.4f\n",c.label.c_str(),c.milliseconds[c.milliseconds.size()/2]); }
      return 0;
    } catch (const std::exception &e) { std::fprintf(stderr,"operator candidates: %s\n",e.what()); return 1; }
  }
}
