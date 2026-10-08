#include "TestBuffers.hpp"
#include "model/StateLayout.hpp"
#include "ops/GDN.hpp"
#include <array>
#include <cstring>
#include <iostream>
using namespace splash;
namespace {
uint16_t bf(float value) {
  uint32_t bits;std::memcpy(&bits,&value,4);
  return uint16_t((bits+0x7fff+((bits>>16)&1))>>16);
}
}
int main(int argc,char **argv) {
  @autoreleasepool {
    try {
      if(argc!=2)throw std::invalid_argument("usage: flash-next-gdn-parity SPLASH_METALLIB");
      metal::MetalBackend backend(argv[1]);
      const ops::GdnShape shape{16,48,128,10240,16640};
      const model::GdnStateLayout state{2,3,10240,48,128,128};
      const auto zero=[&](uint64_t bytes) {auto b=test::sharedBuffer(backend,bytes);std::memset(b.contents(),0,bytes);return b;};
      std::array<metal::MetalBuffer,2> copied{zero(state.cellBytes()),zero(state.cellBytes())};
      std::array<metal::MetalBuffer,2> swapped{zero(state.cellBytes()),zero(state.cellBytes())};
      const auto packed=zero(shape.packedWidth*2),conv=zero(shape.convolutionDimension*4*2),
          decayWeight=zero(48*4),dt=zero(48*2),norm=zero(128*4);
      for(uint32_t i=0;i<shape.convolutionDimension*4;++i)static_cast<uint16_t*>(conv.contents())[i]=bf(0.12f+0.03f*(i%4));
      for(uint32_t h=0;h<48;++h){static_cast<float*>(decayWeight.contents())[h]=-0.15f-0.001f*h;static_cast<uint16_t*>(dt.contents())[h]=bf(0.04f);}
      for(uint32_t d=0;d<128;++d)static_cast<float*>(norm.contents())[d]=0.8f+0.003f*d;
      const auto run=[&](std::array<metal::MetalBuffer,2> &cells,uint32_t layer) {
        const uint64_t co=layer*state.convolutionLayerBytes(),re=state.convolutionBytes()+layer*state.recurrentLayerBytes();
        auto q=zero(16*128*2),k=zero(16*128*2),v=zero(48*128*2),decay=zero(48*4),beta=zero(48*2),
             rows=zero(48*128*2),out=zero(48*128*2);
        metal::CommandGraph graph;
        ops::GDN::addPrefill(graph,{packed,conv,backend.view(cells[0],co,state.convolutionLayerBytes()),
          backend.view(cells[1],co,state.convolutionLayerBytes()),q,k,v,decayWeight,dt,decay,beta,
          backend.view(cells[0],re,state.recurrentLayerBytes()),backend.view(cells[1],re,state.recurrentLayerBytes()),
          rows,{norm,true},out},shape,1,ops::GdnHeadOrder::Tiled);
        static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
        return out;
      };
      // Independent old commit strategy: copy each completed layer back.
      // Candidate: all layers read R0 and write R1, then swap exactly once.
      for(uint32_t token=0;token<9;++token) {
        for(uint32_t layer=0;layer<2;++layer) {
          auto *input=static_cast<uint16_t*>(packed.contents());
          for(uint32_t i=0;i<16640;++i)input[i]=bf(float(int((i*17+token*31+layer*97)%101)-50)*0.003f);
          const auto old=run(copied,layer),candidate=run(swapped,layer);
          test::require(!std::memcmp(old.contents(),candidate.contents(),old.sizeBytes()),"parity changed a layer's recurrent output");
          const uint64_t co=layer*state.convolutionLayerBytes(),re=state.convolutionBytes()+layer*state.recurrentLayerBytes();
          std::memcpy(static_cast<uint8_t*>(copied[0].contents())+co,static_cast<const uint8_t*>(copied[1].contents())+co,state.convolutionLayerBytes());
          std::memcpy(static_cast<uint8_t*>(copied[0].contents())+re,static_cast<const uint8_t*>(copied[1].contents())+re,state.recurrentLayerBytes());
        }
        std::swap(swapped[0],swapped[1]);
        test::require(!std::memcmp(copied[0].contents(),swapped[0].contents(),state.cellBytes()),
                      "whole-forward parity swap changed carried convolution or recurrent state");
      }
      std::cout<<"Flash Next GDN whole-forward parity: 2 layers x 9 tokens, all outputs and states bitwise identical\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
  }
}
