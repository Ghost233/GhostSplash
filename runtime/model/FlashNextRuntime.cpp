#include "model/FlashNextRuntime.hpp"
#include "AwakeClock.hpp"
#include "model/FlashNextNgram.hpp"
#include "model/FlashNextNumerics.hpp"
#include "model/GgufImageLayout.hpp"
#include "ops/FlashNext.hpp"
#include "ops/Embedding.hpp"
#include "ops/GDN.hpp"
#include "metal/abi/Gguf.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
namespace splash::model {
namespace {
using metal::MetalBuffer;
using metal::CommandGraph;
using Clock = AwakeClock;
float asFloat(uint16_t b) { uint32_t u = uint32_t{b} << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint16_t asBfloat(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
struct CompletedTicket final : ModelBatchTicket {
  std::vector<ModelStepResult> results;
  double milliseconds = 0;
  bool ready() const noexcept override { return true; }
  std::vector<ModelStepResult> wait() override { return results; }
  double wallMilliseconds() const noexcept override { return milliseconds; }
};
uint64_t scratchBytes(const FlashNextLayout &l, const ops::ExecutionPlans &operators) {
  uint64_t bytes = 64ull << 20; // fixed single-row buffers, sampling logits and page table
  for (const auto &[out, in] : std::array<std::pair<uint32_t,uint32_t>,4>{{{248320,2560},{12288,2560},{10240,10240},{2560,6144}}}) {
    const auto s = operators.linear().prefillScratchSize({out,in,ops::WeightLayout::Block32});
    bytes += s.input + s.sums + s.partials + s.counters + s.rotated;
  }
  return bytes + uint64_t{l.maximumContextTokens / 4} * sizeof(float);
}
FlashNextNgramConfig ngramConfig(const FlashNextWeights &w) {
  FlashNextNgramConfig config; config.embeddingRows = w.index->require("per_layer_token_embd.weight").tensor.rows();
  const auto copy = [&](const char *key, auto &to) {
    auto values = w.index->unsignedArray(key);
    if (!values || values->size() != to.size()) throw GgufError(std::string("invalid Flash Next PLE array: ") + key);
    std::copy(values->begin(), values->end(), to.begin());
  };
  copy("qwen4exp.ple.layer_multipliers", config.multipliers);
  copy("qwen4exp.ple.head_offsets", config.offsets);
  copy("qwen4exp.ple.head_vocab_sizes", config.vocabularies);
  return config;
}
} // namespace
ModelMemoryPlan flashNextRuntimeMemory(const FlashNextWeights &w, const ops::ExecutionPlans &operators) {
  return {w.layout.stateBytes(), 0, scratchBytes(w.layout, operators) + w.layout.expertCacheBytes};
}
struct FlashNextRuntime::Impl {
  RuntimeContext context;
  FlashNextWeights &weights;
  const FlashNextLayout &layout;
  FlashNextNgramConfig hashConfig;
  std::map<std::string, MetalBuffer> scratch;
  ops::LinearScratch linearScratch;
  uint64_t scratchAllocated = 0;
  ModelTelemetry counters;
  double gpuSeconds=0;
  struct Request {
    uint64_t id;
    uint32_t lane=0;
    std::vector<uint32_t> prompt;
    SamplingParameters sampling;
    uint32_t flags = 0, maxNew = 0, generated = 0, committed = 0;
    ConstraintMode constraint = ConstraintMode::None;
    std::vector<uint32_t> scoreTokens, mask;
    std::optional<uint32_t> pending;
    std::mt19937_64 random;
    FlashNextNgram hash;
    std::array<MetalBuffer,2> gdn;
    MetalBuffer ple, rawKeys, pooledKeys;
    std::vector<uint32_t> outputHistory;
    bool active = true, logitsReady = false;
    Request(const ModelRequest &r, const FlashNextNgramConfig &h) : id(r.id), prompt(r.prompt.begin(),r.prompt.end()),
        sampling(r.sampling), flags(r.flags), maxNew(r.maxNewTokens), constraint(r.constraint),
        scoreTokens(r.scoreTokens.begin(),r.scoreTokens.end()), random(r.sampling.seed), hash(h) {}
  };
  std::map<uint64_t,std::unique_ptr<Request>> requests;
  explicit Impl(RuntimeContext c) : context(std::move(c)), weights(*context.model.flashNext),
      layout(weights.layout), hashConfig(ngramConfig(weights)) {
    const auto add = [&](std::string name, uint64_t bytes) {
      auto buffer = context.backend.allocateBuffer(bytes, metal::BufferStorage::Shared, "flash-" + name);
      std::memset(buffer.contents(),0,bytes); scratchAllocated += buffer.sizeBytes(); scratch.emplace(std::move(name),buffer);
    };
    constexpr uint64_t rows = 8;
    for (const char *n : {"hyper","normalized","hc_gates","ple_key","ple_query","ple_gated","ple_normal","ple_out"}) add(n,rows*layout.hyperWidth()*2);
    for (const char *n : {"mixed","block","embedding","value"}) add(n,rows*layout.hiddenSize*2);
    add("ffn_out",rows*layout.hiddenSize*4);
    add("lowrank",rows*((layout.lowRank+255)/256*256)*2); add("inject",rows*layout.streams*2);
    add("gate",rows*((layout.intermediateSize+255)/256*256)*2); add("up",rows*((layout.intermediateSize+255)/256*256)*2); add("product",rows*((layout.intermediateSize+255)/256*256)*2);
    add("accum",layout.hiddenSize*4); add("router",layout.experts*4); add("shared_gate",16);
    add("gdn_packed",rows*16640*2); add("gdn_qkv",rows*10240*2); add("gdn_z",rows*6144*2);
    add("gdn_alpha",rows*48*2); add("gdn_beta",rows*48*2); add("conv_weights",10240*4*2);
    add("decay_weights",48*4); add("dt",48*2); add("gdn_queries",16*128*2); add("gdn_keys",16*128*2);
    add("gdn_values",48*128*2); add("decay",48*4); add("beta",48*2);
    add("recurrent_rows",48*128*2); add("gdn_hidden",rows*48*128*2);
    add("attn_qg",rows*layout.queryHeads*layout.headDimension*4);
    for (const char *n : {"attn_q","attn_gate","attn_out"}) add(n,rows*layout.queryHeads*layout.headDimension*2);
    for (const char *n : {"attn_k","attn_v"}) add(n,rows*layout.kvHeads*layout.headDimension*2);
    add("index_q",rows*layout.indexHeads*layout.indexWidth*2); add("index_k",rows*layout.indexWidth*2);
    add("scores",std::max(1u,layout.maximumContextTokens/layout.indexRatio)*4);
    add("selected",(layout.indexBudget+layout.indexRatio-1)*4);
    add("position",4); add("cos",layout.rotaryDimension/2*4); add("sin",layout.rotaryDimension/2*4);
    add("page_table",uint64_t{(layout.maximumContextTokens+31)/32}*sizeof(SplashKvPage));
    add("logits",rows*layout.vocabularySize*4); add("token",4); add("ple_native",16*160*2); add("ple_tokens",16*4);
    ops::LinearScratchSize need;
    for (const auto &[name,p] : weights.projections) {
      static_cast<void>(name); if(p.blocks().segments.front().isFloat()) continue; const auto s = context.operators.linear().prefillScratchSize(p.shape());
      need.input=std::max(need.input,s.input); need.sums=std::max(need.sums,s.sums);
      need.input=std::max(need.input,s.input);need.sums=std::max(need.sums,s.sums);need.rotated=std::max(need.rotated,s.rotated);
      need.partials=std::max(need.partials,s.partials); need.counters=std::max(need.counters,s.counters); need.rotated=std::max(need.rotated,s.rotated);
    }
    for (const auto &[n,k] : std::array<std::pair<uint32_t,uint32_t>,2>{{{layout.intermediateSize,layout.hiddenSize},{layout.hiddenSize,layout.intermediateSize}}}) {
      const ops::ProjectionShape shape{(n+255)/256*256,(k+255)/256*256,ops::WeightLayout::Block32};
      const auto a=context.operators.linear().prefillScratchSize(shape),d=context.operators.linear().decodeScratchSize(shape);
      const ops::LinearScratchSize s{std::max(a.input,d.input),std::max(a.sums,d.sums),std::max(a.partials,d.partials),std::max(a.counters,d.counters),std::max(a.rotated,d.rotated)};
      need.input=std::max(need.input,s.input);need.sums=std::max(need.sums,s.sums);need.rotated=std::max(need.rotated,s.rotated);
      need.partials=std::max(need.partials,s.partials); need.counters=std::max(need.counters,s.counters);
    }
    const auto alloc = [&](uint64_t bytes,const char *name) { if(!bytes)return MetalBuffer{};add(name,bytes);return b(name); };
    linearScratch.input=alloc(need.input,"linear_input"); linearScratch.sums=alloc(need.sums,"linear_sums");
    linearScratch.partials=alloc(need.partials,"linear_partials"); linearScratch.counters=alloc(need.counters,"linear_counters");
    linearScratch.rotated=alloc(need.rotated,"linear_rotated");
  }
  MetalBuffer b(std::string_view name) const { return scratch.at(std::string(name)); }
  void run(CommandGraph &g) { gpuSeconds+=context.backend.submitCommandAsync(g.dispatches()).wait().gpuSeconds; }
  void project(CommandGraph &g, const ops::Projection &p, MetalBuffer in, MetalBuffer out, ops::FloatOutput type=ops::FloatOutput::BFloat16) {
    if (p.blocks().segments.front().isFloat()) {
      ops::addGgufFloat(g,in,p.blocks().segments.front(),out,1,p.outputSize,0,type,context.operators.linear().ggufFloatTile(1,p.outputSize));
    } else {
      const bool fp32=type==ops::FloatOutput::Float32;
      ops::LinearWorkload workload{{p.outputSize,p.inputSize},fp32?8u:1u,fp32?ops::LinearPhase::Decode:ops::LinearPhase::Prefill,ops::LinearEpilogue::None,ops::WeightLayout::Block32};
      auto plan=context.operators.linear().plan(workload,p);
      if(type==ops::FloatOutput::Float32) plan=context.operators.linear().plan(workload,plan.configuration(),type);
      ops::LinearBuffers buffers; buffers.input=in; buffers.output=out; buffers.scratch=linearScratch;
      static_cast<void>(context.operators.linear().add(g,buffers,p,plan));
    }
  }
  void project(CommandGraph &g,const std::string &name,MetalBuffer in,MetalBuffer out,ops::FloatOutput type=ops::FloatOutput::BFloat16) { project(g,weights.projection(name),in,out,type); }
  void gather(std::string_view name,std::span<const uint64_t> rows,MetalBuffer destination) {
    const auto &tensor=weights.index->require(name).tensor;
    const auto id=gguf_format_of(tensor.type);
    if(id==GGUF_FMT_COUNT)throw GgufError("Flash Next embedding format unsupported");
    const auto rowBytes=ggufRowBytes(kQuantFormats[id],tensor.columns());
    MetalBuffer native=b("ple_native");
    if(rows.size()*rowBytes>native.sizeBytes()) throw GgufError("Flash Next embedding staging is too small");
    for(size_t i=0;i<rows.size();++i) {
      auto page=weights.hostPages->acquire({WeightUnitKind::EmbeddingRow,std::string(name),rows[i]},weights.index->rows(name,rows[i],1));
      std::memcpy(static_cast<uint8_t*>(native.contents())+i*rowBytes,page.data(),rowBytes);
      static_cast<uint32_t*>(b("ple_tokens").contents())[i]=i;
    }
    ops::EmbeddingWeights table(uint32_t(rows.size()),uint32_t(tensor.columns()),ops::NativeRows(native,id));
    CommandGraph g;ops::Embedding::add(g,b("ple_tokens"),table,destination,rows.size());run(g);
  }
  void hyperMix(CommandGraph &g,const std::string &prefix,bool inject) {
    ops::FlashNext::addGroupedRms(g,b("hyper"),weights.norm(prefix+"norm.weight"),b("normalized"),layout.hiddenSize,layout.streams,1);
    project(g,prefix+"down.weight",b("normalized"),b("lowrank"));
    ops::FlashNext::addHcSilu(g,b("lowrank"),b("lowrank"),layout.lowRank,layout.streams,1);
    project(g,prefix+"up.weight",b("lowrank"),b("hc_gates"));
    ops::FlashNext::addHcMix(g,b("normalized"),b("hc_gates"),b("mixed"),layout.hiddenSize,layout.streams,1);
    if(inject)project(g,prefix+"inject.weight",b("normalized"),b("inject"));
  }
  void ple(Request &r,const std::string &p,FlashNextNgram::Keys keys) {
    gather("per_layer_token_embd.weight",keys,b("embedding"));
    CommandGraph g;
    project(g,p+"ple_key.weight",b("embedding"),b("ple_key")); project(g,p+"ple_value.weight",b("embedding"),b("value"));
    ops::FlashNext::addGroupedRms(g,b("ple_key"),weights.norm(p+"ple_norm_key.weight"),b("ple_key"),layout.hiddenSize,layout.streams,1);
    ops::FlashNext::addGroupedRms(g,b("hyper"),weights.norm(p+"ple_norm_query.weight"),b("ple_query"),layout.hiddenSize,layout.streams,1);
    ops::FlashNext::addPleGate(g,b("ple_key"),b("ple_query"),b("value"),b("ple_gated"),layout.hiddenSize,layout.streams,1);
    ops::FlashNext::addGroupedRms(g,b("ple_gated"),weights.norm(p+"ple_norm_conv.weight"),b("ple_normal"),layout.hiddenSize,layout.streams,1);
    ops::FlashNext::addPleConv(g,b("ple_gated"),b("ple_normal"),r.ple,weights.tensors.at(p+"ple_conv1d.weight"),b("ple_out"),r.ple,layout.hiddenSize,layout.streams,1,layout.pleTaps,layout.pleDilation);
    ops::FlashNext::addAdd(g,b("hyper"),b("ple_out"),b("hyper"),layout.hyperWidth(),1);run(g);
  }
  void gdn(Request &r,const std::string &p,uint32_t ordinal) {
    CommandGraph projections;
    project(projections,p+"attn_qkv.weight",b("mixed"),b("gdn_qkv")); project(projections,p+"attn_gate.weight",b("mixed"),b("gdn_z"));
    project(projections,p+"ssm_beta.weight",b("mixed"),b("gdn_beta")); project(projections,p+"ssm_alpha.weight",b("mixed"),b("gdn_alpha"));run(projections);
    auto *packed=static_cast<uint16_t*>(b("gdn_packed").contents());
    auto *qkv=static_cast<const uint16_t*>(b("gdn_qkv").contents());
    auto *z=static_cast<const uint16_t*>(b("gdn_z").contents());
    auto *alpha=static_cast<const uint16_t*>(b("gdn_alpha").contents());auto *beta=static_cast<const uint16_t*>(b("gdn_beta").contents());
    std::memcpy(packed,qkv,4096*2);
    auto *conv=static_cast<uint16_t*>(b("conv_weights").contents());
    const auto *sourceConv=static_cast<const float*>(weights.tensors.at(p+"ssm_conv1d.weight").contents());
    for(uint32_t channel=0;channel<4096;++channel)for(uint32_t t=0;t<4;++t)conv[channel*4+t]=asBfloat(sourceConv[channel*4+t]);
    auto *decay=static_cast<float*>(b("decay_weights").contents());auto *dt=static_cast<uint16_t*>(b("dt").contents());
    const auto *sourceDecay=static_cast<const float*>(weights.tensors.at(p+"ssm_a").contents());
    const auto *sourceDt=static_cast<const float*>(weights.tensors.at(p+"ssm_dt.bias").contents());
    for(uint32_t head=0;head<48;++head) {
      const uint32_t tiled=(head%3)*16+head/3;
      std::memcpy(packed+4096+head*128,qkv+4096+tiled*128,256);
      std::memcpy(packed+10240+head*128,z+tiled*128,256);
      packed[16384+head]=beta[tiled];packed[16432+head]=alpha[tiled];
      decay[head]=sourceDecay[tiled];dt[head]=asBfloat(sourceDt[tiled]);
      for(uint32_t dim=0;dim<128;++dim)for(uint32_t t=0;t<4;++t)conv[(4096+head*128+dim)*4+t]=asBfloat(sourceConv[(4096+tiled*128+dim)*4+t]);
    }
    const auto state=layout.gdnStateLayout();
    const uint64_t co=ordinal*state.convolutionLayerBytes(),re=state.convolutionBytes()+ordinal*state.recurrentLayerBytes();
    CommandGraph g;
    ops::GdnPrefillBuffers buffers{b("gdn_packed"),b("conv_weights"),context.backend.view(r.gdn[0],co,state.convolutionLayerBytes()),
      context.backend.view(r.gdn[1],co,state.convolutionLayerBytes()),b("gdn_queries"),b("gdn_keys"),b("gdn_values"),b("decay_weights"),b("dt"),b("decay"),b("beta"),
      context.backend.view(r.gdn[0],re,state.recurrentLayerBytes()),context.backend.view(r.gdn[1],re,state.recurrentLayerBytes()),b("recurrent_rows"),weights.norm(p+"ssm_norm.weight"),b("gdn_hidden")};
    ops::GDN::addPrefill(g,buffers,layout.gdnShape(),1,ops::GdnHeadOrder::Tiled);
    ops::FlashNext::addGdnGate(g,b("recurrent_rows"),context.backend.view(b("gdn_packed"),10240*2,6144*2),weights.norm(p+"ssm_norm.weight"),b("gdn_hidden"),128,16,48,1,true);
    project(g,p+"ssm_out.weight",b("gdn_hidden"),b("block"));run(g);

  }
  void ropeTable(uint32_t position) {
    *static_cast<uint32_t*>(b("position").contents())=0;
    auto *cos=static_cast<float*>(b("cos").contents());auto *sin=static_cast<float*>(b("sin").contents());
    for(uint32_t i=0;i<layout.rotaryDimension/2;++i) { const float angle=position/std::pow(layout.rotaryTheta,float(2*i)/layout.rotaryDimension);cos[i]=std::cos(angle);sin[i]=std::sin(angle); }
  }
  void rope(CommandGraph &g,MetalBuffer tensor,uint32_t width,uint32_t heads) {
    ops::FlashNext::addRope(g,tensor,b("cos"),b("sin"),b("position"),tensor,width,heads,1,layout.rotaryDimension,1);
  }
  void writeKv(const ModelBatchItem &item,uint32_t layer,uint32_t position) {
    if(position/32>=item.pageTable.size())throw std::logic_error("Flash Next KV page table is too short");
    auto spans=context.kvPages.spans(item.pageTable[position/32]);const uint32_t token=position%32;
    for(uint32_t head=0;head<layout.kvHeads;++head)for(uint32_t role=0;role<2;++role) {
      auto *values=static_cast<const uint16_t*>(b(role?"attn_v":"attn_k").contents())+head*layout.headDimension;
      float scale=0;
      if(context.kvPages.layout().format==kv::Format::Int8) {
        for(uint32_t d=0;d<layout.headDimension;++d)scale=std::max(scale,std::abs(asFloat(values[d])));
        scale=scale/127.f;
        std::memcpy(spans[layer*4+role*2+1].data()+splash_kv_scale_element(head,token)*4,&scale,4);
      }
      for(uint32_t d=0;d<layout.headDimension;++d) {
        const uint64_t at=role?splash_kv_value_element(head,token,d):splash_kv_key_element(head,token,d);
        if(context.kvPages.layout().format==kv::Format::BFloat16)std::memcpy(spans[layer*4+role*2].data()+at*2,values+d,2);
        else { const int8_t code=scale>0?int8_t(std::clamp(std::nearbyint(asFloat(values[d])/scale),-127.f,127.f)):0;std::memcpy(spans[layer*4+role*2].data()+at,&code,1); }
      }
    }
  }
  void attention(Request &r,const std::string &p,uint32_t ordinal,const ModelBatchItem &item,uint32_t position) {
    CommandGraph projections;
    project(projections,p+"attn_q.weight",b("mixed"),b("attn_qg"));project(projections,p+"attn_k.weight",b("mixed"),b("attn_k"));project(projections,p+"attn_v.weight",b("mixed"),b("attn_v"));
    project(projections,p+"indexer.q_proj.weight",b("mixed"),b("index_q"));project(projections,p+"indexer.k_proj.weight",b("mixed"),b("index_k"));run(projections);
    auto *qg=static_cast<const uint16_t*>(b("attn_qg").contents());
    for(uint32_t h=0;h<layout.queryHeads;++h){std::memcpy(static_cast<uint16_t*>(b("attn_q").contents())+h*256,qg+h*512,512);std::memcpy(static_cast<uint16_t*>(b("attn_gate").contents())+h*256,qg+h*512+256,512);}
    const uint64_t rawLayerBytes=uint64_t{layout.maximumContextTokens}*layout.indexWidth*2;
    const uint64_t pooledLayerBytes=rawLayerBytes/layout.indexRatio;
    auto raw=context.backend.view(r.rawKeys,ordinal*rawLayerBytes,rawLayerBytes);
    auto pooled=context.backend.view(r.pooledKeys,ordinal*pooledLayerBytes,pooledLayerBytes);
    std::memcpy(static_cast<uint8_t*>(raw.contents())+uint64_t{position}*layout.indexWidth*2,b("index_k").contents(),layout.indexWidth*2);
    ropeTable(position);
    CommandGraph norms;
    ops::Normalization::addRms(norms,b("attn_q"),weights.norm(p+"attn_q_norm.weight"),b("attn_q"),256,layout.queryHeads);
    ops::Normalization::addRms(norms,b("attn_k"),weights.norm(p+"attn_k_norm.weight"),b("attn_k"),256,layout.kvHeads);
    ops::Normalization::addRms(norms,b("index_q"),weights.norm(p+"indexer.q_norm.weight"),b("index_q"),128,layout.indexHeads);
    rope(norms,b("attn_q"),256,layout.queryHeads);rope(norms,b("attn_k"),256,layout.kvHeads);rope(norms,b("index_q"),128,layout.indexHeads);run(norms);
    writeKv(item,ordinal,position);
    const uint32_t blocks=(position+1)/layout.indexRatio;
    if((position+1)%layout.indexRatio==0) {
      CommandGraph pool;ops::FlashNext::addQsaPool(pool,raw,pooled,layout.indexWidth,layout.indexRatio,1,blocks-1);run(pool);
      auto current=context.backend.view(pooled,uint64_t{blocks-1}*layout.indexWidth*2,layout.indexWidth*2);
      ropeTable((blocks-1)*layout.indexRatio);CommandGraph norm;
      ops::Normalization::addRms(norm,current,weights.norm(p+"indexer.k_norm.weight"),current,128,1);rope(norm,current,128,1);run(norm);
    }
    *static_cast<uint32_t*>(b("position").contents())=position;
    context.kvPages.writeEntries(item.pageTable,0,b("page_table"));
    CommandGraph selected;
    if(blocks)ops::FlashNext::addQsaScores(selected,b("index_q"),pooled,b("position"),b("scores"),layout.indexWidth,layout.indexHeads,1,blocks,layout.indexRatio);
    ops::FlashNext::addQsaSelect(selected,b("scores"),b("position"),b("selected"),1,blocks,layout.indexRatio,layout.indexBudget);
    ops::FlashNext::addQsaAttention(selected,b("attn_q"),b("attn_gate"),b("page_table"),b("selected"),b("attn_out"),context.kvPages.layers()[ordinal],layout.queryHeads,1,layout.indexBudget+layout.indexRatio-1,item.pageTable.size(),context.kvPages.layout().format);
    project(selected,p+"attn_output.weight",b("attn_out"),b("block"));run(selected);
  }
  void moe(const std::string &p) {
    CommandGraph route;project(route,p+"ffn_gate_inp.weight",b("mixed"),b("router"),ops::FloatOutput::Float32);run(route);
    const auto *logits=static_cast<const float*>(b("router").contents());std::vector<uint32_t> experts(layout.experts);std::iota(experts.begin(),experts.end(),0);
    for(uint32_t i=0;i<layout.experts;++i)requireFiniteFlashNextLogit(logits[i],"Flash Next router produced non-finite logits");
    std::partial_sort(experts.begin(),experts.begin()+layout.expertsPerToken,experts.end(),[&](uint32_t a,uint32_t c){return logits[a]==logits[c]?a<c:logits[a]>logits[c];});
    std::array<float,10> masses{};float total=0;
    for(uint32_t i=0;i<layout.expertsPerToken;++i){masses[i]=std::exp(logits[experts[i]]-logits[experts[0]]);total+=masses[i];}
    std::memset(b("accum").contents(),0,layout.hiddenSize*4);
    for(uint32_t i=0;i<layout.expertsPerToken;++i) {
      const auto gate=acquireFlashNextExpert(weights,context.backend,p+"ffn_gate_exps.weight",experts[i]);
      const auto up=acquireFlashNextExpert(weights,context.backend,p+"ffn_up_exps.weight",experts[i]);
      const auto down=acquireFlashNextExpert(weights,context.backend,p+"ffn_down_exps.weight",experts[i]);
      CommandGraph g;project(g,gate->projection,b("mixed"),b("gate"));project(g,up->projection,b("mixed"),b("up"));
      ops::FlashNext::addSiluMul(g,b("gate"),b("up"),b("product"),layout.intermediateSize,1);
      project(g,down->projection,b("product"),b("ffn_out"),ops::FloatOutput::Float32);run(g);
      auto *accumulation=static_cast<float*>(b("accum").contents());
      const auto *expert=static_cast<const float*>(b("ffn_out").contents());
      for(uint32_t d=0;d<layout.hiddenSize;++d)accumulation[d]+=expert[d]*(masses[i]/total);
    }
    const auto *shared=static_cast<const float*>(weights.tensors.at(p+"ffn_gate_inp_shexp.weight").contents());
    const auto *mixed=static_cast<const uint16_t*>(b("mixed").contents());float scalar=0;
    for(uint32_t i=0;i<layout.hiddenSize;++i)scalar+=shared[i]*asFloat(mixed[i]);
    CommandGraph g;project(g,p+"ffn_gate_shexp.weight",b("mixed"),b("gate"));project(g,p+"ffn_up_shexp.weight",b("mixed"),b("up"));
    ops::FlashNext::addSiluMul(g,b("gate"),b("up"),b("product"),layout.intermediateSize,1);project(g,p+"ffn_down_shexp.weight",b("product"),b("ffn_out"),ops::FloatOutput::Float32);run(g);
    auto *accumulation=static_cast<float*>(b("accum").contents());const auto *sharedOut=static_cast<const float*>(b("ffn_out").contents());
    const float scale=1.f/(1.f+std::exp(-scalar));
    for(uint32_t d=0;d<layout.hiddenSize;++d)accumulation[d]+=sharedOut[d]*scale;
    CommandGraph rounded;ops::FlashNext::addFloatToBf16(rounded,b("accum"),b("block"),layout.hiddenSize,1);run(rounded);
  }
  void forward(Request &r,uint32_t token,const ModelBatchItem &item,bool headNeeded=true) {
    if(r.committed>=layout.maximumContextTokens)throw std::invalid_argument("Flash Next context exhausted");
    const auto keys=r.hash.push(token);const std::array<uint64_t,1> tokenRow{token};gather("token_embd.weight",tokenRow,b("embedding"));
    CommandGraph initial;ops::FlashNext::addBroadcast(initial,b("embedding"),b("hyper"),layout.hiddenSize,layout.streams,1);run(initial);
    uint32_t ai=0,gi=0;
    for(uint32_t layer=0;layer<layout.layers;++layer) {
      const std::string p="blk."+std::to_string(layer)+".";
      if(layer==layout.pleLayer)ple(r,p,keys);
      CommandGraph mix;hyperMix(mix,p+"hc_attn_",true);run(mix);
      if(layout.isAttention(layer))attention(r,p,ai++,item,r.committed);else gdn(r,p,gi++);
      CommandGraph scatter;ops::FlashNext::addHcScatter(scatter,b("hyper"),b("block"),b("inject"),b("hyper"),layout.hiddenSize,layout.streams,1);run(scatter);
      CommandGraph ffnMix;hyperMix(ffnMix,p+"hc_ffn_",true);run(ffnMix);moe(p);
      CommandGraph ffnScatter;ops::FlashNext::addHcScatter(ffnScatter,b("hyper"),b("block"),b("inject"),b("hyper"),layout.hiddenSize,layout.streams,1);run(ffnScatter);
    }
    if(headNeeded) {CommandGraph head;hyperMix(head,"output_hc_",false);project(head,"output.weight",b("mixed"),b("logits"),ops::FloatOutput::Float32);run(head);}
    // Each GDN layer read its own previous-token R0 state and wrote R1.
    // All layers are complete now; swapping avoids a CPU copy of every recurrent matrix.
    std::swap(r.gdn[0],r.gdn[1]);
    ++r.committed;r.logitsReady=headNeeded;
  }
  uint32_t sample(Request &r) {
    auto *logits=static_cast<float*>(b("logits").contents());std::vector<uint32_t> candidates;
    std::vector<uint32_t> counts(layout.vocabularySize);std::vector<bool> prompt(layout.vocabularySize);
    for(uint32_t t:r.prompt)prompt[t]=true;for(uint32_t t:r.outputHistory)++counts[t];
    for(uint32_t t=0;t<layout.vocabularySize;++t) {
      requireFiniteFlashNextLogit(logits[t],"Flash Next output produced non-finite logits");
      if(r.constraint==ConstraintMode::TokenMask && (r.mask.empty() || !(r.mask[t/32]&(1u<<(t%32)))))continue;
      if((r.flags&RequestIgnoreEndOfSequence) && (t==layout.stopTokens[0]||t==layout.stopTokens[1]))continue;
      if((prompt[t]||counts[t])&&r.sampling.repetitionPenalty!=1.f)logits[t]=logits[t]>=0?logits[t]/r.sampling.repetitionPenalty:logits[t]*r.sampling.repetitionPenalty;
      if(counts[t])logits[t]-=r.sampling.presencePenalty+r.sampling.frequencyPenalty*counts[t];
      candidates.push_back(t);
    }
    if(candidates.empty())throw std::invalid_argument("Flash Next mask admits no tokens");
    auto order=[&](uint32_t a,uint32_t c){return logits[a]==logits[c]?a<c:logits[a]>logits[c];};
    if(r.sampling.temperature==0.f)return *std::min_element(candidates.begin(),candidates.end(),[&](auto a,auto c){return order(a,c);});
    std::sort(candidates.begin(),candidates.end(),order);
    if(r.sampling.topK && candidates.size()>r.sampling.topK)candidates.resize(r.sampling.topK);
    std::vector<double> probabilities;double sum=0;
    for(uint32_t t:candidates){double p=std::exp((double(logits[t])-logits[candidates[0]])/r.sampling.temperature);if(p<r.sampling.minP)break;probabilities.push_back(p);sum+=p;}
    if(probabilities.empty())throw FlashNextNumericalFailure("Flash Next sampling empty distribution");
    double cumulative=0;size_t retain=0;for(;retain<probabilities.size();++retain){cumulative+=probabilities[retain];if(cumulative>=r.sampling.topP*sum){++retain;break;}}
    probabilities.resize(retain);std::discrete_distribution<size_t> distribution(probabilities.begin(),probabilities.end());return candidates[distribution(r.random)];
  }
};
FlashNextRuntime::FlashNextRuntime(RuntimeContext c) : impl_(std::make_unique<Impl>(std::move(c))) {}
FlashNextRuntime::~FlashNextRuntime() = default;
void FlashNextRuntime::checkHealth() {
  if(!impl_->context.backend.healthy())throw std::runtime_error(impl_->context.backend.unhealthyReason());
}
StateAdmission FlashNextRuntime::begin(const ModelRequest &request) {
  if(request.restoredTokens)throw std::invalid_argument("Flash Next cannot restore prefix cache state");
  if(request.constraint!=ConstraintMode::None)throw std::invalid_argument("Flash Next constrained decoding is not supported yet");
  if(!request.images.empty())throw std::invalid_argument("Flash Next is language only");
  if(request.prompt.empty())throw std::invalid_argument("Flash Next prompt is empty");
  if(request.prompt.size()>impl_->layout.maximumContextTokens)throw std::invalid_argument("Flash Next prompt exceeds context");
  uint32_t lane=0;
  std::array<bool,ExecutionLimits::maximumBatchWidth> used{};
  for(const auto &[id,r]:impl_->requests) {static_cast<void>(id);if(r->active)used.at(r->lane)=true;}
  while(lane<used.size() && used[lane])++lane;
  if(lane>=used.size())return {{},StateFailure::ConcurrencyLimit};
  if(impl_->requests.contains(request.id))throw std::logic_error("Flash Next request already exists");
  for(auto t:request.prompt)if(t>=impl_->layout.vocabularySize)throw std::invalid_argument("Flash Next token outside vocabulary");
  auto r=std::make_unique<Impl::Request>(request,impl_->hashConfig);
  r->lane=lane;
  const auto &l=impl_->layout;const auto state=l.gdnStateLayout();
  const auto allocate=[&] {
    for(auto &cell:r->gdn){cell=impl_->context.backend.allocateBuffer(state.cellBytes(),metal::BufferStorage::Shared,"flash-gdn-state");std::memset(cell.contents(),0,cell.sizeBytes());}
    const uint64_t pleBytes=uint64_t{l.pleTaps-1}*l.pleDilation*l.hyperWidth()*2;
    r->ple=impl_->context.backend.allocateBuffer(pleBytes,metal::BufferStorage::Shared,"flash-ple-state");std::memset(r->ple.contents(),0,pleBytes);
    const uint64_t rawBytes=uint64_t{l.kvLayout().attentionLayers}*l.maximumContextTokens*l.indexWidth*2;
    r->rawKeys=impl_->context.backend.allocateBuffer(rawBytes,metal::BufferStorage::Shared,"flash-indexer-raw");
    r->pooledKeys=impl_->context.backend.allocateBuffer(rawBytes/l.indexRatio,metal::BufferStorage::Shared,"flash-indexer-pooled");
    std::memset(r->rawKeys.contents(),0,r->rawKeys.sizeBytes());std::memset(r->pooledKeys.contents(),0,r->pooledKeys.sizeBytes());
  };
  if(impl_->context.admitAllocation) {
    auto admitted=impl_->context.admitAllocation(l.stateBytes(),allocate);
    if(!admitted)return {{},StateFailure::MemoryPressure,admitted.failure};
  } else allocate();
  impl_->requests.emplace(request.id,std::move(r));return {lane,StateFailure::None};
}
void FlashNextRuntime::suspend(uint64_t id) {
  auto &r=*impl_->requests.at(id);r.gdn={};r.ple={};r.rawKeys={};r.pooledKeys={};r.active=false;r.logitsReady=false;
}
StateAdmission FlashNextRuntime::resume(const ModelRequest &request) {
  auto found=impl_->requests.find(request.id);
  if(found==impl_->requests.end())return begin(request);
  if(found->second->active)throw std::logic_error("Flash Next request is already resident");
  auto old=std::move(found->second);impl_->requests.erase(found);
  auto admission=begin(request);
  if(!admission.granted()){impl_->requests.emplace(request.id,std::move(old));return admission;}
  auto &r=*impl_->requests.at(request.id);r.random=old->random;r.generated=old->generated;r.outputHistory=std::move(old->outputHistory);r.pending=old->pending;return admission;
}
std::unique_ptr<StateRestore> FlashNextRuntime::beginRestore(uint64_t,uint32_t,std::shared_ptr<const CompositeState>,bool,std::function<void()>) {
  throw std::invalid_argument("Flash Next prefix state restoration is not supported");
}
void FlashNextRuntime::setDraftContextPlan(uint64_t,DraftContextPlan) {}
std::unique_ptr<ModelBatchTicket> FlashNextRuntime::submit(const BatchPlan &plan,std::span<const ModelBatchItem> items,std::function<void()> completion) {
  const auto started=Clock::now();const auto gpuStart=impl_->gpuSeconds;auto ticket=std::make_unique<CompletedTicket>();
  if(plan.items.size()!=items.size())throw std::invalid_argument("Flash Next batch item count mismatch");
  for(size_t i=0;i<items.size();++i) {
    const auto &item=items[i];auto &r=*impl_->requests.at(item.requestId);
    if(!r.active)throw std::logic_error("Flash Next request has no resident state");
    auto result=flashNextLaneResult(r.id,[&] {
    ModelStepResult result;result.requestId=r.id;
    if(plan.kind==WorkKind::Prefill) {
      const auto &batch=plan.items[i];
      if(batch.promptOffset!=r.committed || batch.tokenCount>r.prompt.size()-batch.promptOffset)
        throw std::logic_error("Flash Next prefill position mismatch");
      for(uint32_t j=0;j<batch.tokenCount;++j)impl_->forward(r,r.prompt[batch.promptOffset+j],item,batch.promptOffset+j+1==r.prompt.size());
      result.consumedPromptTokens=batch.tokenCount;impl_->counters.targetPrefillRows+=batch.tokenCount;
      if(r.committed==r.prompt.size()) {
        if(!r.scoreTokens.empty()) {
          const auto *logits=static_cast<const float*>(impl_->b("logits").contents());
          for(uint32_t t:r.scoreTokens){if(t>=impl_->layout.vocabularySize)throw std::logic_error("Flash Next score token outside vocabulary");requireFiniteFlashNextLogit(logits[t],"Flash Next score logit is non-finite");result.scoreLogits.push_back(logits[t]);}result.finished=true;
        } else if(r.constraint==ConstraintMode::TokenMask)result.nextDecodeStage=DecodeStage::ApplyInitialMask;
        else if(!r.pending)r.pending=impl_->sample(r);
      }
    } else {
      if(plan.decodeStage==DecodeStage::ApplyInitialMask) {
        if(r.mask.empty())throw std::logic_error("Flash Next initial mask is missing");
        if(!r.logitsReady)throw std::logic_error("Flash Next initial logits are missing");
      } else {
        if(!r.pending)throw std::logic_error("Flash Next decode has no pending token");
        const auto committed=*r.pending;impl_->forward(r,committed,item);r.pending.reset();
        result.outputTokens.push_back(committed);++r.generated;r.outputHistory.push_back(committed);
      }
      if(r.generated<r.maxNew)r.pending=impl_->sample(r);
    }
    if(r.pending) {
      const bool stop=*r.pending==impl_->layout.stopTokens[0] || *r.pending==impl_->layout.stopTokens[1];
      if(stop || r.maxNew-r.generated==1){result.outputTokens.push_back(*r.pending);++r.generated;r.outputHistory.push_back(*r.pending);result.outputTokensWithoutKv=1;result.finished=stop;}
    }
    return result;
    });
    ticket->results.push_back(std::move(result));
  }
  ticket->milliseconds=std::chrono::duration<double,std::milli>(Clock::now()-started).count();
  const double seconds=ticket->milliseconds/1000;const auto gpu=impl_->gpuSeconds-gpuStart;
  if(plan.kind==WorkKind::Prefill){impl_->counters.lastPrefillWallSeconds=seconds;impl_->counters.totalPrefillWallSeconds+=seconds;impl_->counters.lastPrefillGpuSeconds=gpu;impl_->counters.totalPrefillGpuSeconds+=gpu;}
  else{impl_->counters.lastDecodeWallSeconds=seconds;impl_->counters.totalDecodeWallSeconds+=seconds;impl_->counters.lastDecodeGpuSeconds=gpu;impl_->counters.totalDecodeGpuSeconds+=gpu;}
  if(completion)completion();return ticket;
}
std::shared_ptr<const CompositeState> FlashNextRuntime::snapshot(uint64_t) { return {}; }
uint64_t FlashNextRuntime::snapshotBytes() const noexcept { return 0; }
uint64_t FlashNextRuntime::reclaimIdleState(bool,IdleMemory scope) noexcept {
  return scope == IdleMemory::BuffersThenCaches ? releaseFlashNextExpertPage(impl_->weights) : 0;
}
std::optional<std::string> FlashNextRuntime::provideMask(uint64_t id,std::span<const uint32_t> words) {
  auto &r=*impl_->requests.at(id);
  if(words.size()!=maskWordsPerToken(impl_->layout.vocabularySize))return "Flash Next mask must contain exactly one vocabulary row";
  r.mask.assign(words.begin(),words.end());return {};
}
void FlashNextRuntime::end(uint64_t id) { impl_->requests.erase(id); }
WarmupStepResult FlashNextRuntime::warmupPrefill(uint32_t rows) {
  if(!rows || rows>ExecutionLimits::prefillTokenBudget)throw std::invalid_argument("Flash Next warmup rows invalid");
  std::vector<uint32_t> prompt(rows,248044);const uint64_t id=std::numeric_limits<uint64_t>::max();
  ModelRequest request;request.id=id;request.prompt=prompt;request.maxNewTokens=2;
  auto admission=begin(request);if(!admission.granted())throw metal::MetalAllocationError("Flash Next warmup state admission failed",admission.allocationFailure);
  const auto needed=(rows+31)/32;std::vector<uint32_t> pages(needed);std::iota(pages.begin(),pages.end(),0);
  BatchPlan plan;plan.kind=WorkKind::Prefill;plan.items={{id,rows,0}};
  ModelBatchItem item;item.requestId=id;item.tokenCount=rows;item.pageTable=pages;item.pageTableRevision=1;
  auto result=submit(plan,{&item,1},{});WarmupStepResult out;out.wallSeconds=result->wallMilliseconds()/1000;out.detail="Flash Next cold SSD autoregressive prefill, "+std::to_string(rows)+" rows";
  out.lanes.push_back({result->wait().front(),impl_->requests.at(id)->pending,rows});end(id);return out;
}
WarmupStepResult FlashNextRuntime::warmupDecodeBatch(uint32_t width) {
  if(!width || width>ExecutionLimits::maximumBatchWidth)throw std::invalid_argument("Flash Next warmup width invalid");
  std::vector<uint64_t> ids;std::vector<ModelBatchItem> items;BatchPlan pre;pre.kind=WorkKind::Prefill;
  const uint32_t token=248044;
  for(uint32_t i=0;i<width;++i){const uint64_t id=std::numeric_limits<uint64_t>::max()-i;ModelRequest r;r.id=id;r.prompt={&token,1};r.maxNewTokens=4;auto a=begin(r);if(!a.granted()){for(auto old:ids)end(old);throw metal::MetalAllocationError("Flash Next warmup state admission failed",a.allocationFailure);}ids.push_back(id);pre.items.push_back({id,1,0});ModelBatchItem item;item.requestId=id;item.pageTableRevision=1;item.tokenCount=1;items.push_back(item);}
  std::vector<std::array<uint32_t,1>> pages(width);for(uint32_t i=0;i<width;++i){pages[i]={i};items[i].pageTable=pages[i];}
  static_cast<void>(submit(pre,items,{})->wait());BatchPlan decode;decode.kind=WorkKind::Decode;
  for(uint32_t i=0;i<width;++i){decode.items.push_back({items[i].requestId,1,0});items[i].logicalPosition=1;}
  auto ticket=submit(decode,items,{});auto steps=ticket->wait();WarmupStepResult out;out.wallSeconds=ticket->wallMilliseconds()/1000;out.detail="Flash Next serial autoregressive decode, "+std::to_string(width)+" lanes";
  for(uint32_t i=0;i<width;++i){const uint64_t id=items[i].requestId;out.lanes.push_back({steps[i],impl_->requests.at(id)->pending,impl_->requests.at(id)->committed});end(id);}return out;
}
WarmupStepResult FlashNextRuntime::warmupCompositeStateRestore() { throw std::logic_error("Flash Next prefix cache is unavailable"); }
ModelMemoryActual FlashNextRuntime::actualRuntimeMemory() const {
  uint64_t state=0;for(const auto &[id,r]:impl_->requests){static_cast<void>(id);for(const auto &cell:r->gdn)state+=cell.sizeBytes();state+=r->ple.sizeBytes()+r->rawKeys.sizeBytes()+r->pooledKeys.sizeBytes();}
  return {state,0,impl_->scratchAllocated+flashNextExpertTelemetry(impl_->weights).residentBytes,0};
}
ModelTelemetry FlashNextRuntime::telemetry() const noexcept {
  auto out=impl_->counters;out.stateAllocatedBytes=actualRuntimeMemory().stateActualAllocatedBytes;
  const auto experts=flashNextExpertTelemetry(impl_->weights),host=impl_->weights.hostPages->telemetry();
  out.expertCacheHits=experts.hits;out.expertCacheMisses=experts.misses;out.expertCacheBytesRead=experts.bytesRead;
  out.expertCacheResidentBytes=experts.residentBytes;out.expertCacheEvictions=experts.evictions;
  out.weightHostCacheHits=host.hits;out.weightHostCacheMisses=host.misses;out.weightHostCacheResidentBytes=host.residentBytes;return out;
}
} // namespace splash::model
