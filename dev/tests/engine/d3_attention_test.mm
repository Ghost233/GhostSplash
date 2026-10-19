#include "ops/D3.hpp"
#include "LinearNumerics.hpp"
#include "TestChecks.hpp"
#include <cmath>
#include <bit>
#include <cstring>
#include <iostream>
#include <limits>
using namespace splash;
using namespace splash::test;
void check(metal::MetalBackend &b,uint32_t n,uint32_t heads,bool selected,bool nonuniform=false) {
  const uint32_t stride=ops::D3::storageRows(n), kv=heads/4;
  auto allocate=[&](uint64_t bytes){auto r=b.allocateBuffer(bytes,metal::BufferStorage::Shared,"d3-test");std::memset(r.contents(),0,bytes);return r;};
  auto q=allocate(uint64_t{stride}*heads*256*2), k=allocate(uint64_t{stride}*kv*256*2);
  auto v=allocate(k.sizeBytes()), transposed=allocate(k.sizeBytes()), out=allocate(q.sizeBytes());
  auto *qq=static_cast<uint16_t *>(q.contents()), *kk=static_cast<uint16_t *>(k.contents()), *vv=static_cast<uint16_t *>(v.contents());
  for(uint32_t h=0;h<kv;++h) for(uint32_t t=0;t<n;++t) {
    if(nonuniform) kk[(h*stride+t)*256]=floatToBf16(float(t%3)/2);
    for(uint32_t g=0;g<4;++g) if(nonuniform) qq[((h*stride+t)*4+g)*256]=floatToBf16(float((t+g)%5)/4);
    for(uint32_t c=0;c<256;++c) vv[(h*256+c)*stride+t]=floatToBf16(float((t+c+h)%8)/8);
  }
  metal::CommandGraph graph;
  ops::D3::transposeValues(graph,v,transposed,{n,stride,selected?1:n,selected?n-1:0,heads,kv});
  ops::D3::attention(graph,q,k,transposed,out,{n,stride,selected?1:n,selected?n-1:0,heads,kv});
  static_cast<void>(b.submitCommandAsync(graph.command()).wait());
  const auto *actual=static_cast<const uint16_t *>(out.contents());
  for(uint32_t h=0;h<heads;++h) for(uint32_t t=0;t<(selected?1:n);++t) {
    uint32_t query=selected?n-1:t; double maximum=-INFINITY, sum=0;
    auto score=[&](uint32_t j){return double(bf16ToFloat(qq[(((h/4)*stride+query)*4+h%4)*256]))*bf16ToFloat(kk[((h/4)*stride+j)*256])/16;};
    for(uint32_t j=0;j<n;++j) maximum=std::max(maximum,score(j));
    for(uint32_t j=0;j<n;++j) sum+=std::exp(score(j)-maximum);
    for(uint32_t c=0;c<256;++c) {
      double expected=0;
      for(uint32_t j=0;j<n;++j) {
        double exponent=std::exp(score(j)-maximum);
        expected+=bf16ToFloat(floatToBf16(float(exponent)))*bf16ToFloat(vv[((h/4)*256+c)*stride+j]);
      }
      expected/=sum; double value=bf16ToFloat(actual[(((h/4)*stride+t)*4+h%4)*256+c]);
      double error=4*n*std::numeric_limits<float>::epsilon()*std::max(1.0,std::fabs(expected));
      require(value>=bf16ToFloat(floatToBf16(float(expected-error))) && value<=bf16ToFloat(floatToBf16(float(expected+error))),"D3 noncausal attention outside FP64 mixed-precision bound n="+std::to_string(n)+" heads="+std::to_string(heads)+" selected="+std::to_string(selected)+" h="+std::to_string(h)+" t="+std::to_string(t)+" c="+std::to_string(c)+" got="+std::to_string(value)+" expected="+std::to_string(expected));
    }
  }
  std::cout<<"D3 FP64 attention n="<<n<<" heads="<<heads<<" selected="<<selected<<" nonuniform="<<nonuniform<<" PASS\n";
}
void quantized(metal::MetalBackend &b,uint32_t heads) {
  const uint32_t n=37, stride=ops::D3::storageRows(n), kv=heads/4;
  auto alloc=[&](uint64_t bytes){auto x=b.allocateBuffer(bytes,metal::BufferStorage::Shared,"d3-q4h-test");std::memset(x.contents(),0,bytes);return x;};
  auto q=alloc(uint64_t{stride}*heads*256*2), k=alloc(uint64_t{stride}*kv*256*2),v=alloc(k.sizeBytes());
  auto decoded=alloc(k.sizeBytes()),out=alloc(q.sizeBytes()),codes=alloc(uint64_t{stride}*kv*2*128),scales=alloc(uint64_t{stride}*kv*2*4);
  auto *keys=static_cast<uint16_t *>(k.contents()),*values=static_cast<uint16_t *>(v.contents());
  for(uint32_t h=0;h<kv;++h)for(uint32_t t=0;t<n;++t)for(uint32_t d=0;d<256;++d){
    keys[(h*stride+t)*256+d]=floatToBf16(float(int(d%15)-7)/64);
    values[(h*256+d)*stride+t]=floatToBf16(float(int((d+t+h)%15)-7)/64);
  }
  D3AttentionParams p{n,stride,1,n-1,heads,kv,1};metal::CommandGraph g;
  ops::D3::quantizeKv(g,k,v,decoded,codes,scales,p);ops::D3::attention(g,q,k,decoded,out,p,scales);
  static_cast<void>(b.submitCommandAsync(g.command()).wait());
  auto *bytes=static_cast<uint8_t *>(codes.contents());auto *scale=static_cast<float *>(scales.contents());
  for(uint32_t tensor=0;tensor<2;++tensor)for(uint32_t h=0;h<kv;++h)for(uint32_t t=0;t<n;++t){
    uint32_t row=(tensor*kv+h)*stride+t;require(scale[row]==1.0f/64,"D3 Q4H scale mismatch");
    for(uint32_t d=0;d<256;++d){int code=(bytes[row*128+d/2]>>((d%2)*4))&15;code=code>=8?code-16:code;
      int expected=int((tensor?d+t+h:d)%15)-7;require(code==expected,"D3 Q4H signed nibble mismatch");}
  }
  auto *actual=static_cast<uint16_t *>(out.contents());
  for(uint32_t h=0;h<heads;++h)for(uint32_t c=0;c<256;++c){
    double expected=0;
    for(uint32_t j=0;j<64;++j){double mean=0;for(uint32_t t=0;t<n;++t)mean+=double(int((c/64*64+j+t+h/4)%15)-7)/64;
      expected+=(std::popcount((c%64)&j)%2?-1:1)*mean/n/8;}
    double value=bf16ToFloat(actual[((h/4)*stride*4+h%4)*256+c]);
    double error=4*n*std::numeric_limits<float>::epsilon()*std::max(1.0,std::fabs(expected));
    require(value>=bf16ToFloat(floatToBf16(float(expected-error)))&&value<=bf16ToFloat(floatToBf16(float(expected+error))),"D3 Q4H inverse H64 outside FP64 bound");
  }
  std::cout<<"D3 Q4H codes/scales/FP64 inverse H64 heads="<<heads<<" PASS\n";
}
int main(int argc,char **argv){
  if(argc!=2)return 2;
  try{metal::MetalBackend b(argv[1]);for(uint32_t heads:{8,16}){for(uint32_t n:{1,37,128,129})for(bool selected:{false,true})check(b,n,heads,selected);check(b,128,heads,false,true);check(b,2049,heads,true);quantized(b,heads);}}
  catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}return 0;
}
