#include "GgufFormatReference.hpp"
#include "TestBuffers.hpp"
#include "ops/Linear.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
using namespace splash;
using namespace gguf_reference;
namespace {
uint16_t bf(float value) {uint32_t bits;std::memcpy(&bits,&value,4);return uint16_t((bits+0x7fff+((bits>>16)&1))>>16);}
float widen(uint16_t value) {uint32_t bits=uint32_t{value}<<16;float out;std::memcpy(&out,&bits,4);return out;}
}
int main(int argc,char **argv) {
  @autoreleasepool {
    try {
      if(argc!=2)throw std::invalid_argument("usage: flash-next-projection-extent SPLASH_METALLIB");
      metal::MetalBackend backend(argv[1]);ops::Linear linear(backend.capabilities());std::mt19937 random(817);
      const auto buffer=[&](uint64_t bytes) {auto b=test::sharedBuffer(backend,bytes);std::memset(b.contents(),0,bytes);return b;};
      const auto upload=[&](const std::vector<uint8_t> &bytes) {auto b=buffer(bytes.size());std::memcpy(b.contents(),bytes.data(),bytes.size());return b;};
      const auto exercise=[&](Fmt format,uint32_t rows,uint32_t columns,ops::FloatOutput destination,uint64_t oldBufferBytes) {
        const uint32_t physical=(rows+255)/256*256;
        const uint32_t physicalColumns=(columns+255)/256*256;
        auto source=makeNative(format,physical,columns,random);
        std::vector<uint8_t> native(uint64_t{physical}*rowBytes(format,physicalColumns));
        for(uint32_t row=0;row<physical;++row)std::memcpy(native.data()+uint64_t{row}*rowBytes(format,physicalColumns),source.data()+uint64_t{row}*rowBytes(format,columns),rowBytes(format,columns));
        std::vector<float> values,halfValues;const auto packed=repack(format,native,physical,physicalColumns,&values,&halfValues);
        auto p0=upload(packed.w0),meta=upload(packed.meta);
        auto p1=packed.w1.empty()?metal::MetalBuffer{}:upload(packed.w1);
        ops::Projection p(physical,physicalColumns,ops::BlockWeights{{ops::QuantizedSegment::planes(format,rows,physicalColumns,p0,p1,meta)}});
        const bool fp32=destination==ops::FloatOutput::Float32;
        ops::LinearWorkload work{{physical,physicalColumns},fp32?8u:1u,fp32?ops::LinearPhase::Decode:ops::LinearPhase::Prefill,ops::LinearEpilogue::None,ops::WeightLayout::Block32};
        const auto baseline=linear.plan(work,p);
        const auto plan=ops::Linear::plan(work,baseline.configuration(),destination);
        test::require(plan.storageRows()==8,"single-token Flash Next quantized projection must reserve its staged 8 rows");
        auto input=buffer(uint64_t{plan.storageRows()}*physicalColumns*2);
        auto *x=static_cast<uint16_t*>(input.contents());for(uint32_t k=0;k<columns;++k)x[k]=bf(float(int(k%31)-15)*0.002f);
        const auto size=plan.scratchSize();ops::LinearScratch scratch;
        const auto maybe=[&](uint64_t bytes) {return bytes?buffer(bytes):metal::MetalBuffer{};};
        scratch.input=maybe(size.input);scratch.sums=maybe(size.sums);scratch.partials=maybe(size.partials);
        scratch.counters=maybe(size.counters);scratch.rotated=maybe(size.rotated);
        auto shortOutput=buffer(oldBufferBytes);ops::LinearBuffers bad;bad.input=input;bad.output=shortOutput;bad.scratch=scratch;
        bool rejected=false;
        try{metal::CommandGraph graph;static_cast<void>(linear.add(graph,bad,p,plan));}catch(const std::invalid_argument &){rejected=true;}
        test::require(rejected,"old logical/BF16-sized output must be rejected before GPU submission");
        auto output=buffer(uint64_t{plan.storageRows()}*physical*ops::elementBytes(destination));
        ops::LinearBuffers good;good.input=input;good.output=output;good.scratch=scratch;
        metal::CommandGraph graph;static_cast<void>(linear.add(graph,good,p,plan));
        static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
        for(uint32_t n=0;n<rows;++n) {
          double expected=0,magnitude=0;
          for(uint32_t k=0;k<columns;++k){const double term=double(widen(x[k]))*halfValues[uint64_t{n}*physicalColumns+k];expected+=term;magnitude+=std::abs(term);}
          const double actual=destination==ops::FloatOutput::Float32?static_cast<const float*>(output.contents())[n]:widen(static_cast<const uint16_t*>(output.contents())[n]);
          const double bound=destination==ops::FloatOutput::Float32?2e-5*magnitude+1e-7:0.004*std::abs(expected)+2e-5*magnitude+1e-7;
          test::require(std::isfinite(actual) && std::abs(actual-expected)<=bound,"Flash Next real quantized matvec diverged from fp64 oracle");
        }
      };
      exercise(IQ3XXS,640,2560,ops::FloatOutput::BFloat16,8ull*640*2);
      exercise(Q20,2560,640,ops::FloatOutput::Float32,8ull*2560*2);
      std::cout<<"Flash Next padded 640->768 gate and Q2_0 FP32 down projection extents and numerical results passed\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
  }
}
