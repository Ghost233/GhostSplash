#include "model/FlashNextRuntime.hpp"
#include "TestFiles.hpp"
#include "TestGguf.hpp"
#include "TestChecks.hpp"
#include "ops/PageStorage.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
using namespace splash;
using test::require;
namespace {
test::gguf::Key integers(const std::string &name, const std::vector<uint64_t> &values) {
  test::gguf::Key key{name,test::gguf::kArray,{}};
  test::gguf::append(key.value,test::gguf::kUint64);
  test::gguf::append<uint64_t>(key.value,values.size());
  for (auto value:values)test::gguf::append(key.value,value);
  return key;
}
}
int main(int argc,char **argv) {
  @autoreleasepool {
    try {
      if(argc!=2)throw std::invalid_argument("usage: flash-next-runtime SPLASH_METALLIB");
      test::TemporaryDirectory directory("splash-flash-state");
      const auto path=directory.path()/"ple.gguf";
      std::vector<uint64_t> offsets(16),vocabularies(16,1);
      std::iota(offsets.begin(),offsets.end(),0);
      // Compact synthetic PLE table is solely an admission fixture. No
      // generation tensors are supplied or a model answer claimed here.
      test::writeFile(path,test::gguf::file({integers("qwen4exp.ple.layer_multipliers",{1,3,5}),
        integers("qwen4exp.ple.head_offsets",offsets),integers("qwen4exp.ple.head_vocab_sizes",vocabularies)},
        {{"per_layer_token_embd.weight",{160,16},20,test::gguf::Bytes(90*16)}}));
      metal::MetalBackend backend(argv[1]);
      model::LoadedModel loaded;
      loaded.flashNext=std::make_shared<model::FlashNextWeights>();
      loaded.flashNext->layout.maximumContextTokens=16;
      loaded.flashNext->index=std::make_shared<model::GgufShardIndex>(std::vector<std::filesystem::path>{path});
      loaded.descriptor.flashNext=loaded.flashNext->layout;
      ops::ExecutionPlans operators(backend.capabilities());
      bool deny=false;
      const metal::AllocationAdmission admit=[&](uint64_t,const std::function<void()> &allocate) {
        if(deny)return metal::AllocationResult{metal::AllocationFailure::HostPressure};
        allocate();return metal::AllocationResult{};
      };
      const auto kvLayout=loaded.flashNext->layout.kvLayout();
      const uint32_t extent=kvLayout.minimumExtentPages();
      kv::PageStorage pages(backend,admit,kvLayout,extent,extent);
      model::FlashNextRuntime runtime({backend,loaded,pages,nullptr,operators,nullptr,admit});
      require(!runtime.supportsPrefixCache() && runtime.snapshotBytes()==0 && !runtime.snapshot(1),
              "Flash Next must never publish or restore incomplete recurrent state");
      require(runtime.maximumPrefillTokenBudget()==1 && runtime.recommendedWarmupPrefillRows()==1,
              "SSD autoregressive work must stay bounded at one token per dispatch");
      const auto base=backend.memoryStats().allocatedBytes;
      const uint32_t token=42;
      ModelRequest request;request.id=1;request.prompt={&token,1};request.maxNewTokens=8;
      request.restoredTokens=1;
      test::rejects([&]{static_cast<void>(runtime.begin(request));},"cannot restore prefix", "Flash Next accepted a cached prefix");
      request.restoredTokens=0;deny=true;
      auto pressure=runtime.begin(request);
      require(!pressure.granted() && pressure.failure==StateFailure::MemoryPressure &&
              pressure.allocationFailure==metal::AllocationFailure::HostPressure,
              "state admission must return the governor's denial");
      require(backend.memoryStats().allocatedBytes==base,"denied admission allocated partial state");
      deny=false;
      std::array<uint32_t,4> lanes{};
      for(uint64_t id=1;id<=4;++id) {
        request.id=id;auto admitted=runtime.begin(request);require(admitted.granted(),"four small-context lanes should fit fixture");
        lanes[id-1]=*admitted.lane;
      }
      std::sort(lanes.begin(),lanes.end());require(lanes==std::array<uint32_t,4>{0,1,2,3},"resident lane IDs must be unique");
      request.id=5;auto full=runtime.begin(request);
      require(!full.granted() && full.failure==StateFailure::ConcurrencyLimit,"fifth lane must wait");
      const auto occupied=backend.memoryStats().allocatedBytes;
      runtime.suspend(2);
      require(backend.memoryStats().allocatedBytes<occupied,"suspension must release actual state");
      request.id=2;auto resumed=runtime.resume(request);require(resumed.granted(),"suspended lane should resume cold");
      require(backend.memoryStats().allocatedBytes==occupied,"cold resume must restore only one state allocation");
      test::rejects([&]{static_cast<void>(runtime.beginRestore(2,32,{},false,{}));},"not supported","Flash Next faked cached-state restoration");
      for(uint64_t id=1;id<=4;++id)runtime.end(id);
      require(backend.memoryStats().allocatedBytes==base,"ended lanes leaked state buffers");
      std::cout<<"Flash Next cold admission, bounded lanes, suspend/resume and no-prefix-cache contracts passed\n";
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
  }
}
