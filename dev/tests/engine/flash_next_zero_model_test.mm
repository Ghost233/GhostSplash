// Standalone full-geometry execution over an explicitly synthetic sparse
// zero-weight fixture. This is NOT a real model/accuracy/performance test.
#include "model/FlashNextRuntime.hpp"
#include "TestChecks.hpp"
#include "ops/PageStorage.hpp"
#include <array>
#include <iostream>
using namespace splash;
int main(int argc,char **argv) {
  @autoreleasepool {
    try {
      if(argc!=3)throw std::invalid_argument("usage: flash-next-zero-model SPLASH_METALLIB SYNTHETIC_FIXTURE_ROOT");
      const std::filesystem::path root(argv[2]);
      if(!std::filesystem::is_regular_file(root/"SYNTHETIC-ZERO-FIXTURE.json"))throw std::invalid_argument("synthetic fixture marker is required");
      std::cout<<"SYNTHETIC ZERO WEIGHTS ONLY: full 48-layer wiring smoke; no real-model quality or speed claim\n"<<std::flush;
      metal::MetalBackend backend(argv[1]);
      model::ModelDescriptor descriptor;
      descriptor.name="SYNTHETIC ZERO Flash Next";descriptor.flashNext=model::FlashNextLayout{};
      descriptor.targetSource=model::TargetSource::Gguf;descriptor.visionSource=model::VisionSource::None;
      descriptor.capabilities={248320,262144};descriptor.targetKvLayout=descriptor.flashNext->kvLayout();
      descriptor.sourceIdentity=model::weightDigest("TEST ONLY: synthetic Flash Next zero payload");
      test::require(descriptor.valid(),"synthetic descriptor must exercise the real Flash Next path");
      auto loaded=model::loadModel(backend,root,descriptor);
      test::require(loaded.flashNext && !loaded.draft.actualAllocatedBytes,"synthetic load must not create a fake draft");
      std::cout<<"loaded static_metal_bytes="<<loaded.targetActualAllocatedBytes()<<" expert_cache_limit="<<loaded.flashNext->layout.expertCacheBytes<<'\n'<<std::flush;
      const uint64_t budget=30ull<<30;
      const metal::AllocationAdmission admit=[&](uint64_t bytes,const std::function<void()> &allocate) {
        if(bytes>budget-backend.memoryStats().allocatedBytes)return metal::AllocationResult{metal::AllocationFailure::EngineBudget};
        allocate();return metal::AllocationResult{};
      };
      auto layout=loaded.targetKvLayout(kv::Format::Int8);const auto extent=layout.minimumExtentPages();
      kv::PageStorage pages(backend,admit,layout,extent,extent);
      auto allocated=pages.allocateExtent(0);test::require(bool(allocated),"synthetic KV extent must fit budget");
      ops::ExecutionPlans operators(backend.capabilities());
      auto runtime=model::createRuntime({backend,loaded,pages,nullptr,operators,nullptr,admit});
      const std::array<uint32_t,1> prompt{42},table{0};const std::array<uint32_t,2> options{0,1};
      ModelRequest score;score.id=1;score.prompt=prompt;score.scoreTokens=options;
      auto admitted=runtime->begin(score);test::require(admitted.granted(),"synthetic score state must fit budget");
      auto memory=runtime->actualRuntimeMemory();
      std::cout<<"before_score state_bytes="<<memory.stateActualAllocatedBytes<<" runtime_decode_bytes="<<memory.sharedDecodeActualAllocatedBytes<<" backend_bytes="<<backend.memoryStats().allocatedBytes<<'\n'<<std::flush;
      BatchPlan pre;pre.kind=WorkKind::Prefill;pre.items={{1,1,0}};
      ModelBatchItem item;item.requestId=1;item.tokenCount=1;item.pageTable=table;item.pageTableRevision=1;
      auto ticket=runtime->submit(pre,{&item,1},{});auto result=ticket->wait().at(0);
      test::require(result.failure.empty() && result.finished && result.consumedPromptTokens==1 &&
                    result.scoreLogits==std::vector<float>{0,0},"48-layer synthetic zero score must be exactly finite zeros");
      std::cout<<"score PASS actual_layers_per_token=48 options=[0,0]\n"<<std::flush;
      runtime->end(1);
      ModelRequest generation;generation.id=2;generation.prompt=prompt;generation.maxNewTokens=2;
      test::require(runtime->begin(generation).granted(),"synthetic generation state must fit budget");
      pre.items={{2,1,0}};item.requestId=2;
      auto initial=runtime->submit(pre,{&item,1},{})->wait().at(0);
      test::require(initial.failure.empty() && initial.outputTokens.empty() && initial.consumedPromptTokens==1,
                    "first synthetic token must remain a pending nonterminal anchor");
      BatchPlan decode;decode.kind=WorkKind::Decode;decode.items={{2,1,0}};item.logicalPosition=1;
      auto final=runtime->submit(decode,{&item,1},{})->wait().at(0);
      test::require(final.failure.empty() && final.outputTokens==std::vector<uint32_t>{0,0} && final.outputTokensWithoutKv==1,
                    "plain decode must commit token zero and emit the final zero anchor without KV");
      auto telemetry=runtime->telemetry();memory=runtime->actualRuntimeMemory();
      test::require(telemetry.expertCacheMisses>0 && telemetry.expertCacheHits>0 &&
                    telemetry.expertCacheResidentBytes<=20ull<<30,"synthetic expert paging must have real misses, reuse and bounded residence");
      std::cout<<"generation PASS [0,0] actual_layers_per_token=48 state_bytes="<<memory.stateActualAllocatedBytes
        <<" expert_hits="<<telemetry.expertCacheHits<<" expert_misses="<<telemetry.expertCacheMisses
        <<" expert_resident_bytes="<<telemetry.expertCacheResidentBytes<<" host_resident_bytes="<<telemetry.weightHostCacheResidentBytes
        <<" backend_bytes="<<backend.memoryStats().allocatedBytes<<'\n'<<std::flush;
      runtime->end(2);test::require(runtime->actualRuntimeMemory().stateActualAllocatedBytes==0,"synthetic active state leaked after end");
      test::require(backend.healthy() && backend.memoryStats().peakAllocatedBytes<=budget,"synthetic smoke exceeded budget or failed Metal health");
      std::cout<<"observed_backend_peak_bytes="<<backend.memoryStats().peakAllocatedBytes<<" strict_budget_bytes="<<budget<<'\n';
      std::cout<<"FULL SYNTHETIC ZERO 48-LAYER WIRING PASS; real downloaded model still requires separate validation\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
  }
}
