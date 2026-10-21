#include "model/D3.hpp"
#include "model/GgufTarget.hpp"
#include "model/Qwen3_5_4B.hpp"
#include "model/WeightImages.hpp"
#include "ops/D3.hpp"
#include "ops/Embedding.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"
#include "ops/PagedAttention.hpp"
#import <Foundation/Foundation.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <variant>

namespace splash::model {
using metal::MetalBuffer;
struct D3::Impl {
  metal::MetalBackend &backend;
  ops::Linear linear;
  WeightImages images;
  std::variant<Qwen3_5_0_8BWeights, Qwen3_5_4BWeights> weights;
  QwenTargetDimensions geometry;
  uint32_t context, rows = 0;
  bool q4h;
  uint64_t maximum, weightSize = 0, quantizedSize = 0;
  std::vector<uint32_t> codes;
  ops::LinearScratch scratch;
  MetalBuffer tokens, hidden[2], normed, packed, queries, keys, values, transposedValues, attention;
  MetalBuffer gate, intermediate, mixed, residual, gpacked, gqueries, gkeys, gvalues, decay, beta, recurrentRows;
  MetalBuffer conv[2], state[2], cosine, sine, logits;
  MetalBuffer kvCodes, kvScales;
  metal::CommandTiming timing;

  MetalBuffer allocate(uint64_t bytes, std::string_view name) {
    if (!bytes) return {};
    if (bytes > maximum - std::min(maximum, backend.memoryStats().allocatedBytes))
      throw std::runtime_error("D3 allocation exceeds --max-memory");
    auto b = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, name);
    if (backend.memoryStats().allocatedBytes > maximum) throw std::runtime_error("D3 aligned allocation exceeds --max-memory");
    std::memset(b.contents(), 0, b.sizeBytes());
    return b;
  }
  MetalBuffer view(const MetalBuffer &b, uint64_t offset) { return backend.view(b, offset, b.sizeBytes()-offset); }
  auto workspace(uint32_t n) {
    const auto &d = geometry;
    const uint64_t r = ops::D3::storageRows(n), chunk = SPLASH_PREFILL_TOKEN_BUDGET;
    return std::vector<std::pair<MetalBuffer *, uint64_t>>{
      {&tokens,4*r*4}, {&hidden[0],r*d.hiddenSize*2}, {&hidden[1],r*d.hiddenSize*2},
      {&normed,chunk*d.hiddenSize*2}, {&packed,r*d.packedFullWidth*2},
      {&queries,r*d.attentionWidth*2}, {&attention,r*d.attentionWidth*2},
      {&keys,r*d.attentionKvHeads*256*2}, {&values,r*d.attentionKvHeads*256*2},
      {&transposedValues,r*d.attentionKvHeads*256*2},
      {&kvCodes,q4h?r*d.attentionKvHeads*2*128:0}, {&kvScales,q4h?r*d.attentionKvHeads*2*4:0},
      {&cosine,r*32*4}, {&sine,r*32*4}, {&logits,4*128*256*4},
      {&gate,chunk*d.intermediateSize*4}, {&intermediate,chunk*d.intermediateSize*2},
      {&mixed,chunk*d.attentionWidth*2}, {&residual,chunk*d.hiddenSize*2},
      {&gpacked,chunk*d.packedGdnWidth*2}, {&gqueries,chunk*d.gdnKeyHeads*128*2},
      {&gkeys,chunk*d.gdnKeyHeads*128*2}, {&gvalues,chunk*d.attentionWidth*2},
      {&decay,chunk*d.gdnValueHeads*4}, {&beta,chunk*d.gdnValueHeads*2},
      {&recurrentRows,chunk*d.attentionWidth*2},
      {&conv[0],3ull*d.convolutionDimension*2}, {&conv[1],3ull*d.convolutionDimension*2},
      {&state[0],uint64_t{d.gdnValueHeads}*128*128*4}, {&state[1],uint64_t{d.gdnValueHeads}*128*128*4}};
  }
  void reserve(uint32_t n) {
    if (ops::D3::storageRows(n) <= rows) return;
    auto sizes = workspace(n);
    for (auto &[b, bytes] : sizes) *b = {};
    for (auto &[b, bytes] : sizes) *b = allocate(bytes, "d3-workspace");
    rows = ops::D3::storageRows(n);
    auto *co = static_cast<float *>(cosine.contents()), *si = static_cast<float *>(sine.contents());
    for (uint32_t r=0; r<rows; ++r) for (uint32_t c=0; c<32; ++c) {
      float inverse = float(std::pow(geometry.rotaryTheta, -double(c)/32));
      co[r*32+c] = std::cos(float(r)*inverse); si[r*32+c] = std::sin(float(r)*inverse);
    }
  }
  Impl(metal::MetalBackend &b, const std::filesystem::path &path, uint32_t capacity, uint64_t limit, bool quantized)
      : backend(b), linear(b.capabilities()), images(b,"d3"), context(capacity), q4h(quantized), maximum(limit) {
    if (!context || context>262144 || !maximum) throw std::invalid_argument("invalid D3 context or memory limit");
    auto configPath = path.parent_path().parent_path()/"config.json";
    if (std::filesystem::file_size(configPath)>8*1024*1024) throw GgufError("D3 config exceeds size limit");
    NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:configPath.c_str()]];
    id config = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
    if (![config isKindOfClass:[NSDictionary class]] || ![config[@"decision_config"] isKindOfClass:[NSDictionary class]] ||
        ![config[@"decision_config"][@"type"] isEqual:@"vllm-d3"])
      throw GgufError("missing D3 decision configuration");
    id ids = config[@"decision_config"][@"token_ids"];
    if (![ids isKindOfClass:[NSArray class]] || [ids count]<2 || [ids count]>255) throw GgufError("invalid D3 code IDs");
    std::set<uint32_t> seen;
    for (id v in ids) {
      if (![v isKindOfClass:[NSNumber class]] || [v doubleValue] != [v unsignedIntValue] ||
          [v unsignedIntValue]>=248320 || !seen.insert([v unsignedIntValue]).second)
        throw GgufError("invalid or repeated D3 code ID");
      codes.push_back([v unsignedIntValue]);
    }
    WeightSource source(path); GgufFile file(source);
    bool lite = file.metadata().unsignedValue("qwen35.embedding_length") == 1024;
    geometry = lite ? static_cast<QwenTargetDimensions>(Qwen3_5_0_8BLayout{})
                    : static_cast<QwenTargetDimensions>(Qwen3_5_4BLayout{});
    geometry.decisionHeadSize = 256;
    const auto plans = gguf::planImages(file, "qwen35", geometry, {.headTokens = codes});
    uint64_t estimated=0; for (const auto &p:plans) estimated += gguf::loadedBytes(p);
    for (auto &[buffer,bytes]:workspace(context)) estimated += (bytes+16383)/16384*16384;
    if (estimated>maximum) throw GgufError("D3 context and weights do not fit --max-memory");
    GgufTargetLoader loader(b,images,{path},"qwen35",geometry,{},{},codes);
    if (lite) { Qwen3_5_0_8BLayout layout; layout.decisionHeadSize=256; weights=loadQwen3_5_0_8BWeights(b,layout,std::ref(loader)); }
    else { Qwen3_5_4BLayout layout; layout.decisionHeadSize=256; weights=loadQwen3_5_4BWeights(b,layout,std::ref(loader)); }
    weightSize=b.memoryStats().allocatedBytes;
    ops::LinearScratchSize need;
    std::visit([&](const auto &w) {
      const auto include=[&](const ops::Projection &p) {
        need.include(linear.prefillScratchSize(p.shape()));
        for (const auto &s:p.blocks().segments) if (!s.isFloat())
          quantizedSize += s.plane0.allocatedBytes();
      };
      include(w.logitsProjection);
      for (const auto &l:w.layers) {
        std::visit([&](const auto &m) { include(m.inputProjection); include(m.outputProjection); },l.mixer);
        include(l.gateProjection); include(l.upProjection); include(l.downProjection);
      }
    },weights);
    // Native image allocations combine norms and projection planes. Keep those
    // whole allocations in weightBytes; quantizedWeightBytes reports source GGUF payloads.
    quantizedSize=0;
    for (const auto &t:file.tensors()) if (t.type!=ggml::kF32 && t.type!=ggml::kBF16 && t.type!=ggml::kF16) {
      quantizedSize += t.name=="output.weight" ? t.bytes/t.rows()*codes.size() : t.bytes;
    }
    scratch={allocate(need.input,"d3-linear-input"),allocate(need.sums,"d3-linear-sums"),
             allocate(need.partials,"d3-linear-partials"),allocate(need.counters,"d3-linear-counters"),
             allocate(need.rotated,"d3-linear-rotation")};
    reserve(128);
  }
  void validate(const ModernBertInput &in) const {
    if (in.tokens.empty() || in.tokens.size()>context || in.kind>2 || in.options.size()<2 || in.options.size()>codes.size() ||
        std::ranges::any_of(in.tokens,[&](uint32_t t){return t>=geometry.vocabularySize;}))
      throw std::invalid_argument("invalid D3 decision input");
    std::set<uint32_t> seen;
    for (uint32_t t:in.options) if (std::ranges::find(codes,t)==codes.end() || !seen.insert(t).second)
      throw std::invalid_argument("unknown or repeated D3 answer code");
  }
  template<class W> void encode(metal::CommandGraph &g, const W &w, const ModernBertInput &in, uint32_t slot, bool selected) {
    const auto &d=geometry; const uint32_t n=in.tokens.size();
    auto input=view(tokens,uint64_t{slot}*rows*4); std::memcpy(input.contents(),in.tokens.data(),in.tokens.size_bytes());
    ops::Embedding::add(g,input,w.tokenEmbedding,hidden[0],n);
    const auto chunk=[&](auto fn) {for (uint32_t first=0;first<n;first+=SPLASH_PREFILL_TOKEN_BUDGET) fn(first,std::min<uint32_t>(SPLASH_PREFILL_TOKEN_BUDGET,n-first));};
    for (uint32_t layer=0;layer<w.layers.size();++layer) {
      const auto &l=w.layers[layer]; auto src=hidden[layer&1], dst=hidden[(layer&1)^1];
      bool last=layer+1==w.layers.size(), sliced=last&&selected;
      if (const auto *m=std::get_if<QwenAttentionWeights>(&l.mixer)) {
        chunk([&](uint32_t first,uint32_t count) {
          ops::Normalization::addRms(g,view(src,uint64_t{first}*d.hiddenSize*2),l.inputNorm,normed,d.hiddenSize,count);
          auto p=view(packed,uint64_t{first}*d.packedFullWidth*2);
          linear.addPrefill(g,normed,m->inputProjection,p,count,scratch);
          ops::PagedAttention::addPrefillProjection(g,p,m->queryNorm,m->keyNorm,view(cosine,uint64_t{first}*32*4),view(sine,uint64_t{first}*32*4),
            view(queries,uint64_t{first}*4*256*2),view(keys,uint64_t{first}*256*2),view(values,uint64_t{first}*2),count,rows,d.attentionQueryHeads,{1,d.attentionKvHeads,256,q4h?kv::Format::Int4Hadamard:kv::Format::BFloat16});
        });
        D3AttentionParams params{n,rows,sliced?1:n,sliced?n-1:0,d.attentionQueryHeads,d.attentionKvHeads,uint32_t(q4h)};
        if(q4h) ops::D3::quantizeKv(g,keys,values,transposedValues,kvCodes,kvScales,params);
        else ops::D3::transposeValues(g,values,transposedValues,params);
        ops::D3::attention(g,queries,keys,transposedValues,attention,params,kvScales);
      } else {
        for (auto &b:conv) ops::D3::zero(g,b);
        for (auto &b:state) ops::D3::zero(g,b);
      }
      uint32_t block=0;
      chunk([&](uint32_t first,uint32_t count) {
        if (sliced && first+count!=n) return;
        uint32_t start=sliced?n-1:first, active=sliced?1:count;
        auto residualInput=view(src,uint64_t{start}*d.hiddenSize*2), result=view(dst,uint64_t{start}*d.hiddenSize*2);
        if (const auto *m=std::get_if<QwenGdnWeights>(&l.mixer)) {
          ops::Normalization::addRms(g,residualInput,l.inputNorm,normed,d.hiddenSize,active);
          linear.addPrefill(g,normed,m->inputProjection,gpacked,active,scratch);
          const uint32_t a=block++&1, b=a^1;
          ops::GDN::addPrefill(g,backend,{gpacked,m->convolutionWeights,conv[a],conv[b],gqueries,gkeys,gvalues,
            m->decay,m->timeBias,decay,beta,state[a],state[b],recurrentRows,m->mixerNorm,mixed},
            {d.gdnKeyHeads,d.gdnValueHeads,128,d.convolutionDimension,d.packedGdnWidth},active,ops::GdnStateFormat::Float32,m->outputHeadOrder);
          linear.addPrefillResidual(g,mixed,m->outputProjection,residualInput,residual,active,scratch);
        } else {
          const auto &attnWeights=std::get<QwenAttentionWeights>(l.mixer);
          auto attn=sliced?attention:view(attention,uint64_t{first}*4*256*2);
          ops::PagedAttention::addPrefillGate(g,view(packed,uint64_t{start}*d.packedFullWidth*2),attn,mixed,active,rows,d.attentionQueryHeads,{1,d.attentionKvHeads,256,kv::Format::BFloat16});
          linear.addPrefillResidual(g,mixed,attnWeights.outputProjection,residualInput,residual,active,scratch);
        }
        ops::Normalization::addRms(g,residual,l.postAttentionNorm,normed,d.hiddenSize,active);
        linear.addPrefillSwiGlu(g,{&l.gateProjection,&l.upProjection,&l.downProjection},{normed,gate,intermediate,scratch},residual,result,active);
      });
    }
    ops::Normalization::addRms(g,view(hidden[d.layers&1],uint64_t{n-1}*d.hiddenSize*2),w.finalNorm,normed,d.hiddenSize,1);
    linear.addPrefill(g,normed,w.logitsProjection,view(logits,uint64_t{slot}*128*256*4),1,scratch);
  }
};
D3::D3(metal::MetalBackend &b,const std::filesystem::path &p,uint32_t c,uint64_t m,bool q):impl_(std::make_unique<Impl>(b,p,c,m,q)){}
D3::~D3()=default;
void D3::validate(const ModernBertInput &r)const{impl_->validate(r);}
std::vector<std::vector<float>> D3::scoreBatch(std::span<const ModernBertInput> in,bool selected){
  if(in.empty()||in.size()>4)throw std::invalid_argument("invalid D3 batch size");
  uint32_t rows=0;for(const auto &r:in){validate(r);rows=std::max<uint32_t>(rows,r.tokens.size());}impl_->reserve(rows);
  metal::CommandGraph g;uint32_t slot=0;
  for(const auto &r:in){std::visit([&](const auto &w){impl_->encode(g,w,r,slot,selected);},impl_->weights);++slot;}
  impl_->timing=impl_->backend.submitCommandAsync(g.command()).wait();
  std::vector<std::vector<float>> result;
  for(uint32_t i=0;i<in.size();++i){auto &row=result.emplace_back();const auto *scores=static_cast<const float *>(impl_->logits.contents())+i*128*256;
    for(uint32_t t:in[i].options){float v=scores[std::ranges::find(impl_->codes,t)-impl_->codes.begin()];if(!std::isfinite(v))throw std::runtime_error("nonfinite D3 logit");row.push_back(v);}}
  return result;
}
std::vector<float> D3::score(std::span<const uint32_t> t,std::span<const uint32_t> o,uint32_t k,bool s){ModernBertInput r{t,o,k};return scoreBatch({&r,1},s).front();}
uint64_t D3::weightBytes()const noexcept{return impl_->weightSize;}
uint64_t D3::quantizedWeightBytes()const noexcept{return impl_->quantizedSize;}
uint64_t D3::workspaceBytes()const noexcept{return impl_->backend.memoryStats().allocatedBytes-impl_->weightSize-kvBytes();}
metal::CommandTiming D3::lastTiming()const noexcept{return impl_->timing;}
std::span<const uint32_t> D3::codeTokens()const noexcept{return impl_->codes;}
uint64_t D3::kvBytes()const noexcept{return impl_->kvCodes.allocatedBytes()+impl_->kvScales.allocatedBytes();}
}
