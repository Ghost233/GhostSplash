#include "GgufFormatReference.hpp"
#include "model/D3.hpp"
#include "TestChecks.hpp"
#include "model/GgufTarget.hpp"
#include "model/Qwen3_5_4B.hpp"
#include "model/WeightStore.hpp"
#include "ops/Linear.hpp"
#import <Foundation/Foundation.h>
#include <iostream>
using namespace splash;
int main(int argc,char **argv) {
  if(argc!=3 && argc!=4)return 2;
  try {
    metal::MetalBackend backend(argv[1]);std::filesystem::path root(argv[2]);
    NSData *data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:(root/"config.json").c_str()]];
    NSDictionary *config=[NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    std::vector<uint32_t> ids;for(NSNumber *id in config[@"decision_config"][@"token_ids"])ids.push_back(id.unsignedIntValue);
    auto path=model::findTargetGguf(root/"target");model::WeightSource source(path);model::GgufFile file(source);
    model::QwenTargetDimensions geometry=file.metadata().unsignedValue("qwen35.embedding_length")==1024
      ?static_cast<model::QwenTargetDimensions>(model::Qwen3_5_0_8BLayout{})
      :static_cast<model::QwenTargetDimensions>(model::Qwen3_5_4BLayout{});
    geometry.decisionHeadSize=256;
    // Exercise arbitrary row order, including a distant code ID, not a contiguous slice.
    std::swap(ids.front(),ids.back());std::swap(ids[1],ids[128]);
    model::WeightImages images(backend,"d3-head-test");model::GgufTargetLoader loader(backend,images,path,geometry,ids);
    auto head=loader.head();static_cast<void>(model::readNorm(head,geometry.hiddenSize,true,"norm"));
    auto projection=model::readBlockProjection(head,256,geometry.hiddenSize,"D3 head");projection.destination=ops::FloatOutput::Float32;
    auto bias=head.section(256*4,"zero bias");head.finish();
    for(uint32_t i=0;i<256;++i)test::require(static_cast<float *>(bias.contents())[i]==0,"D3 synthesized bias is not zero");
    ops::Linear linear(backend.capabilities());auto plan=linear.prefillPlan(projection,1,ops::LinearEpilogue::None);auto need=plan.scratchSize();
    auto alloc=[&](uint64_t bytes){if(!bytes)return metal::MetalBuffer{};auto b=backend.allocateBuffer(bytes,metal::BufferStorage::Shared,"d3-head-test");std::memset(b.contents(),0,bytes);return b;};
    auto input=alloc(uint64_t{plan.storageRows()}*geometry.hiddenSize*2), output=alloc(uint64_t{plan.storageRows()}*256*4);
    ops::LinearScratch scratch{alloc(need.input),alloc(need.sums),alloc(need.partials),alloc(need.counters),alloc(need.rotated)};
    const auto &tensor=file.require("output.weight");uint64_t rowBytes=tensor.bytes/tensor.rows();
    auto format=gguf_reference::Fmt(gguf_format_of(tensor.type));std::vector<uint8_t> raw(rowBytes);std::vector<float> x(geometry.hiddenSize),w(x.size());
    uint64_t checked=0;double maximumError=0;
    for(uint32_t sample=0;sample<8;++sample){
      for(uint32_t k=0;k<x.size();++k){uint16_t bits=test::floatToBf16(float(int((k*7+sample*13)%31)-15)/64);static_cast<uint16_t *>(input.contents())[k]=bits;x[k]=test::bf16ToFloat(bits);}
      metal::CommandGraph graph;linear.add(graph,{.input=input,.output=output,.scratch=scratch},projection,plan);
      static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
      for(uint32_t i=0;i<ids.size();++i){source.readData(tensor.offset+ids[i]*rowBytes,raw);gguf_reference::rowValues(format,raw.data(),geometry.hiddenSize,w.data());auto dot=gguf_reference::dot(x.data(),w.data(),x.size());
        double actual=static_cast<float *>(output.contents())[i];double error=std::fabs(actual-dot.value);
        bool staged=plan.configuration().tile!=ops::LinearTile::GgufRegister;
        test::require(std::isfinite(actual)&&error<=gguf_reference::projectionBound(dot,staged),"D3 selected code row/mapping outside existing FP64 projection bound");maximumError=std::max(maximumError,error);++checked;}
      test::require(static_cast<float *>(output.contents())[255]==0,"D3 padding logit is not zero");
    }
    std::cout<<"D3 real head FP64 projections="<<checked<<" maximum_error="<<maximumError<<" PASS\n";
    if(argc==4 && std::string_view(argv[3])=="--model") {
      model::D3 model(backend,path,262144,40ull<<30,true);
      std::vector<uint32_t> options{model.codeTokens()[0],model.codeTokens()[1],model.codeTokens()[2]};
      for(uint32_t count:{37,129,2049,4096,32768}) {
        std::vector<uint32_t> tokens(count);for(uint32_t i=0;i<count;++i)tokens[i]=10+(i*17)%997;
        auto full=model.score(tokens,options,0,false), selected=model.score(tokens,options,0,true);
        auto repeated=model.score(tokens,options,0,true);
        for(uint32_t i=0;i<full.size();++i){
          test::require(std::fabs(full[i]-selected[i])<=1e-3f*std::max(1.0f,std::fabs(full[i])),"D3 selected final query differs from full at original 1e-3 bound");
          test::require(std::fabs(repeated[i]-selected[i])<=1e-3f*std::max(1.0f,std::fabs(selected[i])),"D3 repeated request differs at original 1e-3 bound");
        }
        std::cout<<"D3 real full/selected/repeated rows="<<count<<" PASS"<<std::endl;
      }
    }

  } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
  return 0;
}
