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
#include <cstdlib>
#include <future>
#include <atomic>
#include <mutex>
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
constexpr uint32_t kMtpDrafts=3;
uint64_t mtpStateBytes(const FlashNextLayout &l) {
  const uint64_t index=uint64_t{l.maximumContextTokens}*l.indexWidth*2;
  const kv::Layout cache{1,l.kvHeads,l.headDimension,kv::Format::BFloat16};
  return index+index/l.indexRatio+uint64_t{(l.maximumContextTokens+31)/32}*cache.bytesPerModelPage()+l.hyperWidth()*2;
}
struct CompletedTicket final : ModelBatchTicket {
  std::vector<ModelStepResult> results;
  double milliseconds = 0;
  bool ready() const noexcept override { return true; }
  std::vector<ModelStepResult> wait() override { return results; }
  double wallMilliseconds() const noexcept override { return milliseconds; }
};
struct AsyncTicket final : ModelBatchTicket {
  std::future<std::unique_ptr<ModelBatchTicket>> future;
  std::unique_ptr<ModelBatchTicket> completed;
  std::shared_ptr<std::atomic<bool>> done=std::make_shared<std::atomic<bool>>(false);
  bool ready() const noexcept override { return done->load(); }
  std::vector<ModelStepResult> wait() override { if(!completed)completed=future.get();return completed->wait(); }
  double wallMilliseconds() const noexcept override { return completed?completed->wallMilliseconds():0; }
};
uint64_t scratchBytes(const FlashNextLayout &l, const ops::ExecutionPlans &operators) {
  uint64_t bytes = 768ull << 20; // bounded batch buffers, sampling logits and page table
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
  return {w.layout.stateBytes()+(w.hasMtp?mtpStateBytes(w.layout):0), 0,
      scratchBytes(w.layout, operators)+flashNextExpertBudget(w)+(w.hasMtp?32ull<<20:0)};
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
  bool useMtp=false;
  std::atomic<bool> computing=false;
  std::atomic<uint64_t> stateAtSubmission=0;
  mutable std::mutex publicTelemetryMutex;
  ModelTelemetry publicTelemetry;
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
    MetalBuffer mtpRawKeys,mtpPooledKeys,targetFeature;
    std::unique_ptr<kv::PageStorage> mtpKv;
    std::vector<uint32_t> mtpPages;
    uint32_t mtpCommitted=0,verifyRows=0,mtpDepth=kMtpDrafts;
    bool captureVerify=false;
    std::vector<uint32_t> outputHistory;
    bool active = true, logitsReady = false;
    Request(const ModelRequest &r, const FlashNextNgramConfig &h) : id(r.id), prompt(r.prompt.begin(),r.prompt.end()),
        sampling(r.sampling), flags(r.flags), maxNew(r.maxNewTokens), constraint(r.constraint),
        scoreTokens(r.scoreTokens.begin(),r.scoreTokens.end()), random(r.sampling.seed), hash(h) {}
  };
  std::map<uint64_t,std::unique_ptr<Request>> requests;
  explicit Impl(RuntimeContext c) : context(std::move(c)), weights(*context.model.flashNext),
      layout(weights.layout), hashConfig(ngramConfig(weights)) {
    const char *setting=std::getenv("SPLASH_MTP");
    if(setting && std::string_view(setting)!="0" && std::string_view(setting)!="1")
      throw std::invalid_argument("SPLASH_MTP must be 0 or 1");
    useMtp=weights.hasMtp && (!setting || std::string_view(setting)!="0");
    const auto add = [&](std::string name, uint64_t bytes) {
      auto buffer = context.backend.allocateBuffer(bytes, metal::BufferStorage::Shared, "flash-" + name);
      std::memset(buffer.contents(),0,bytes); scratchAllocated += buffer.sizeBytes(); scratch.emplace(std::move(name),buffer);
    };
    constexpr uint64_t rows = kFlashNextPrefillRows;
    for (const char *n : {"hyper","normalized","hc_gates","ple_key","ple_query","ple_gated","ple_normal","ple_out"}) add(n,rows*layout.hyperWidth()*2);
    for (const char *n : {"mixed","block","embedding","value"}) add(n,rows*layout.hiddenSize*2);
    add("ffn_out",rows*layout.hiddenSize*4);
    add("lowrank",rows*((layout.lowRank+255)/256*256)*2); add("inject",rows*layout.streams*2);
    add("gate",rows*((layout.intermediateSize+255)/256*256)*2); add("up",rows*((layout.intermediateSize+255)/256*256)*2); add("product",rows*((layout.intermediateSize+255)/256*256)*2);
    add("accum",rows*layout.hiddenSize*4); add("router",rows*layout.experts*4);
    add("expert_input",rows*layout.hiddenSize*2); add("expert_results",rows*layout.expertsPerToken*layout.hiddenSize*4);
    add("projection_input",rows*layout.hyperWidth()*2);
    add("projection_output",std::max<uint64_t>(rows*layout.hyperWidth()*2,32ull*layout.vocabularySize*4)); add("shared_gate",16);
    if(weights.hasMtp) {
      for(const char *name:{"mtp_pairs","mtp_normalized","mtp_hidden"})add(name,rows*layout.hyperWidth()*2);
      for(const char *name:{"mtp_embed_norm","mtp_embed"})add(name,rows*layout.hiddenSize*2);
      add("verify_logits",8ull*layout.vocabularySize*4);
      add("verify_gdn",uint64_t{kMtpDrafts+1}*layout.gdnStateLayout().layers*16640*2);
      add("verify_ple_before",uint64_t{layout.pleTaps-1}*layout.pleDilation*layout.hyperWidth()*2);
      add("verify_ple_norm",uint64_t{kMtpDrafts+1}*layout.hyperWidth()*2);
    }
    add("gdn_packed",rows*16640*2); add("gdn_qkv",rows*10240*2); add("gdn_z",rows*6144*2);
    add("gdn_alpha",rows*48*2); add("gdn_beta",rows*48*2); add("conv_weights",10240*4*2);
    add("decay_weights",48*4); add("dt",48*2); add("gdn_queries",rows*16*128*2); add("gdn_keys",rows*16*128*2);
    add("gdn_values",rows*48*128*2); add("decay",rows*48*4); add("beta",rows*48*2);
    add("recurrent_rows",rows*48*128*2); add("gdn_hidden",rows*48*128*2);
    add("attn_qg",rows*layout.queryHeads*layout.headDimension*4);
    for (const char *n : {"attn_q","attn_gate","attn_out"}) add(n,rows*layout.queryHeads*layout.headDimension*2);
    for (const char *n : {"attn_k","attn_v"}) add(n,rows*layout.kvHeads*layout.headDimension*2);
    add("index_q",rows*layout.indexHeads*layout.indexWidth*2); add("index_k",rows*layout.indexWidth*2);
    add("scores",rows*std::max(1u,layout.maximumContextTokens/layout.indexRatio)*4);
    add("selected",rows*(layout.indexBudget+layout.indexRatio-1)*4);
    add("position",rows*4); add("cos",rows*layout.rotaryDimension/2*4); add("sin",rows*layout.rotaryDimension/2*4);
    add("page_table",uint64_t{(layout.maximumContextTokens+31)/32}*sizeof(SplashKvPage));
    add("logits",8ull*layout.vocabularySize*4); add("token",4);
    const auto &embeddingTensor=weights.index->require("token_embd.weight").tensor;
    const auto &pleTensor=weights.index->require("per_layer_token_embd.weight").tensor;
    const auto nativeBytes=std::max(ggufRowBytes(kQuantFormats[gguf_format_of(embeddingTensor.type)],embeddingTensor.columns()),
        layout.pleHeads*ggufRowBytes(kQuantFormats[gguf_format_of(pleTensor.type)],pleTensor.columns()));
    add("ple_native",rows*nativeBytes); add("ple_tokens",rows*layout.pleHeads*4);
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
  void project(CommandGraph &g, const ops::Projection &p, MetalBuffer in, MetalBuffer out,
               ops::FloatOutput type=ops::FloatOutput::BFloat16, uint32_t rows=1,
               uint32_t inputWidth=0, uint32_t outputWidth=0) {
    if (!inputWidth) inputWidth=p.inputSize;
    if (!outputWidth) outputWidth=p.blocks().segments.front().outputSize;
    const bool fp32=type==ops::FloatOutput::Float32;
    const bool floating=p.blocks().segments.front().isFloat();
    if(!floating && fp32 && rows>32) {
      for(uint32_t first=0;first<rows;first+=32) {
        const uint32_t count=std::min(32u,rows-first);
        auto source=context.backend.view(in,uint64_t{first}*inputWidth*2,in.sizeBytes()-uint64_t{first}*inputWidth*2);
        auto destination=context.backend.view(out,uint64_t{first}*outputWidth*4,out.sizeBytes()-uint64_t{first}*outputWidth*4);
        project(g,p,source,destination,type,count,inputWidth,outputWidth);
      }
      return;
    }
    ops::LinearWorkload workload{{p.outputSize,p.inputSize},fp32?(rows+7)/8*8:rows,
        fp32?ops::LinearPhase::Decode:ops::LinearPhase::Prefill,
        ops::LinearEpilogue::None,ops::WeightLayout::Block32};
    std::optional<ops::LinearPlan> plan;
    if (!floating) plan=context.operators.linear().plan(workload,p);
    if (!floating && fp32) plan=context.operators.linear().plan(workload,plan->configuration(),type);
    const uint32_t storage=floating?rows:plan->storageRows();
    if (p.inputSize!=inputWidth || storage>rows) {
      ops::FlashNext::addCopyRows(g,in,b("projection_input"),inputWidth,rows,storage,inputWidth,p.inputSize);
      in=b("projection_input");
    }
    const bool compact=p.outputSize!=outputWidth || storage>rows;
    const auto destination=compact?b("projection_output"):out;
    if (floating) ops::addGgufFloat(g,in,p.blocks().segments.front(),destination,rows,p.outputSize,0,type,
                                   context.operators.linear().ggufFloatTile(rows,p.outputSize));
    else {
      ops::LinearBuffers buffers;buffers.input=in;buffers.output=destination;buffers.scratch=linearScratch;
      static_cast<void>(context.operators.linear().add(g,buffers,p,*plan));
    }
    if (compact) ops::FlashNext::addCopyRows(g,destination,out,outputWidth,rows,rows,p.outputSize,outputWidth,fp32);
  }
  void project(CommandGraph &g,const std::string &name,MetalBuffer in,MetalBuffer out,
               ops::FloatOutput type=ops::FloatOutput::BFloat16,uint32_t rows=1) {
    if (name.starts_with("mtp.")) {
      const auto [outWidth,inWidth]=weights.mtpShapes.at(name);
      project(g,weights.projection(name),in,out,type,rows,inWidth,outWidth);
    } else {
      const auto &tensor=weights.index->require(name).tensor;
      project(g,weights.projection(name),in,out,type,rows,tensor.columns(),tensor.rows());
    }
  }
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
  void hyperMix(CommandGraph &g,const std::string &prefix,bool inject,uint32_t rows=1) {
    ops::FlashNext::addGroupedRms(g,b("hyper"),weights.norm(prefix+"norm.weight"),b("normalized"),layout.hiddenSize,layout.streams,rows);
    project(g,prefix+"down.weight",b("normalized"),b("lowrank"),ops::FloatOutput::BFloat16,rows);
    ops::FlashNext::addHcSilu(g,b("lowrank"),b("lowrank"),layout.lowRank,layout.streams,rows);
    project(g,prefix+"up.weight",b("lowrank"),b("hc_gates"),ops::FloatOutput::BFloat16,rows);
    ops::FlashNext::addHcMix(g,b("normalized"),b("hc_gates"),b("mixed"),layout.hiddenSize,layout.streams,rows);
    if(inject)project(g,prefix+"inject.weight",b("normalized"),b("inject"),ops::FloatOutput::BFloat16,rows);
  }
  void ple(Request &r,const std::string &p,std::span<const uint64_t> keys,uint32_t rows=1) {
    gather("per_layer_token_embd.weight",keys,b("embedding"));
    CommandGraph g;
    project(g,p+"ple_key.weight",b("embedding"),b("ple_key"),ops::FloatOutput::BFloat16,rows); project(g,p+"ple_value.weight",b("embedding"),b("value"),ops::FloatOutput::BFloat16,rows);
    ops::FlashNext::addGroupedRms(g,b("ple_key"),weights.norm(p+"ple_norm_key.weight"),b("ple_key"),layout.hiddenSize,layout.streams,rows);
    ops::FlashNext::addGroupedRms(g,b("hyper"),weights.norm(p+"ple_norm_query.weight"),b("ple_query"),layout.hiddenSize,layout.streams,rows);
    ops::FlashNext::addPleGate(g,b("ple_key"),b("ple_query"),b("value"),b("ple_gated"),layout.hiddenSize,layout.streams,rows);
    ops::FlashNext::addGroupedRms(g,b("ple_gated"),weights.norm(p+"ple_norm_conv.weight"),b("ple_normal"),layout.hiddenSize,layout.streams,rows);
    ops::FlashNext::addPleConv(g,b("ple_gated"),b("ple_normal"),r.ple,weights.tensors.at(p+"ple_conv1d.weight"),b("ple_out"),r.ple,layout.hiddenSize,layout.streams,rows,layout.pleTaps,layout.pleDilation);
    ops::FlashNext::addAdd(g,b("hyper"),b("ple_out"),b("hyper"),layout.hyperWidth(),rows);run(g);
    if(r.captureVerify)std::memcpy(b("verify_ple_norm").contents(),b("ple_normal").contents(),uint64_t{rows}*layout.hyperWidth()*2);
  }
  void gdn(Request &r,const std::string &p,uint32_t ordinal,uint32_t rows=1,bool prepareInputs=true) {
    CommandGraph projections;
    if(prepareInputs)for (const auto &[name,buffer] : std::array<std::pair<const char*,const char*>,4>{{
        {"attn_qkv.weight","gdn_qkv"},{"attn_gate.weight","gdn_z"},
        {"ssm_beta.weight","gdn_beta"},{"ssm_alpha.weight","gdn_alpha"}}})
      project(projections,p+name,b("mixed"),b(buffer),ops::FloatOutput::BFloat16,rows);
    if(prepareInputs)run(projections);
    auto *packed=static_cast<uint16_t*>(b("gdn_packed").contents());
    const auto *qkv=static_cast<const uint16_t*>(b("gdn_qkv").contents());
    const auto *z=static_cast<const uint16_t*>(b("gdn_z").contents());
    const auto *alpha=static_cast<const uint16_t*>(b("gdn_alpha").contents());
    const auto *beta=static_cast<const uint16_t*>(b("gdn_beta").contents());
    auto *conv=static_cast<uint16_t*>(b("conv_weights").contents());
    const auto *sourceConv=static_cast<const float*>(weights.tensors.at(p+"ssm_conv1d.weight").contents());
    auto *decay=static_cast<float*>(b("decay_weights").contents());
    auto *dt=static_cast<uint16_t*>(b("dt").contents());
    const auto *sourceDecay=static_cast<const float*>(weights.tensors.at(p+"ssm_a").contents());
    const auto *sourceDt=static_cast<const float*>(weights.tensors.at(p+"ssm_dt.bias").contents());
    for(uint32_t channel=0;channel<4096;++channel)for(uint32_t t=0;t<4;++t)
      conv[channel*4+t]=asBfloat(sourceConv[channel*4+t]);
    for(uint32_t head=0;head<48;++head) {
      const uint32_t tiled=(head%3)*16+head/3;
      decay[head]=sourceDecay[tiled];dt[head]=asBfloat(sourceDt[tiled]);
      for(uint32_t dim=0;dim<128;++dim)for(uint32_t t=0;t<4;++t)
        conv[(4096+head*128+dim)*4+t]=asBfloat(sourceConv[(4096+tiled*128+dim)*4+t]);
      if(prepareInputs)for(uint32_t row=0;row<rows;++row) {
        auto *out=packed+row*16640;
        std::memcpy(out+4096+head*128,qkv+row*10240+4096+tiled*128,256);
        std::memcpy(out+10240+head*128,z+row*6144+tiled*128,256);
        out[16384+head]=beta[row*48+tiled];out[16432+head]=alpha[row*48+tiled];
      }
    }
    if(prepareInputs)for(uint32_t row=0;row<rows;++row)std::memcpy(packed+row*16640,qkv+row*10240,4096*2);
    if(prepareInputs && r.captureVerify)
      std::memcpy(static_cast<uint8_t*>(b("verify_gdn").contents())+uint64_t{ordinal}*r.verifyRows*16640*2,packed,uint64_t{rows}*16640*2);
    const auto state=layout.gdnStateLayout();
    const uint64_t co=ordinal*state.convolutionLayerBytes(),re=state.convolutionBytes()+ordinal*state.recurrentLayerBytes();
    CommandGraph g;
    ops::GdnPrefillBuffers buffers{b("gdn_packed"),b("conv_weights"),context.backend.view(r.gdn[0],co,state.convolutionLayerBytes()),
      context.backend.view(r.gdn[1],co,state.convolutionLayerBytes()),b("gdn_queries"),b("gdn_keys"),b("gdn_values"),b("decay_weights"),b("dt"),b("decay"),b("beta"),
      context.backend.view(r.gdn[0],re,state.recurrentLayerBytes()),context.backend.view(r.gdn[1],re,state.recurrentLayerBytes()),b("recurrent_rows"),weights.norm(p+"ssm_norm.weight"),b("gdn_hidden")};
    ops::GDN::addPrefill(g,buffers,layout.gdnShape(),rows,ops::GdnHeadOrder::Tiled);
    if(prepareInputs) {
    const auto gate=context.backend.view(b("gdn_packed"),10240*2,uint64_t{rows-1}*16640*2+6144*2);
    ops::FlashNext::addGdnGate(g,b("recurrent_rows"),gate,weights.norm(p+"ssm_norm.weight"),b("gdn_hidden"),128,16,48,rows,true,16640);
    project(g,p+"ssm_out.weight",b("gdn_hidden"),b("block"),ops::FloatOutput::BFloat16,rows);
    }
    run(g);
  }
  void ropeTable(uint32_t position,uint32_t rows=1,uint32_t step=1) {
    auto *indices=static_cast<uint32_t*>(b("position").contents());
    auto *cos=static_cast<float*>(b("cos").contents());auto *sin=static_cast<float*>(b("sin").contents());
    for(uint32_t row=0;row<rows;++row) {
      indices[row]=row;
      for(uint32_t i=0;i<layout.rotaryDimension/2;++i) {
        const float angle=(position+row*step)/std::pow(layout.rotaryTheta,float(2*i)/layout.rotaryDimension);
        cos[row*(layout.rotaryDimension/2)+i]=std::cos(angle);sin[row*(layout.rotaryDimension/2)+i]=std::sin(angle);
      }
    }
  }
  void rope(CommandGraph &g,MetalBuffer tensor,uint32_t width,uint32_t heads,uint32_t rows=1) {
    ops::FlashNext::addRope(g,tensor,b("cos"),b("sin"),b("position"),tensor,width,heads,rows,layout.rotaryDimension,rows);
  }
  void writeKv(Request &r,const ModelBatchItem &item,uint32_t layer,uint32_t position,uint32_t row=0,bool mtp=false) {
    auto &store=mtp?*r.mtpKv:context.kvPages;
    if(position/32>=item.pageTable.size())throw std::logic_error("Flash Next KV page table is too short");
    auto spans=store.spans(item.pageTable[position/32]);const uint32_t token=position%32;
    for(uint32_t head=0;head<layout.kvHeads;++head)for(uint32_t role=0;role<2;++role) {
      auto *values=static_cast<const uint16_t*>(b(role?"attn_v":"attn_k").contents())+(row*layout.kvHeads+head)*layout.headDimension;
      float scale=0;
      if(store.layout().format==kv::Format::Int8) {
        for(uint32_t d=0;d<layout.headDimension;++d)scale=std::max(scale,std::abs(asFloat(values[d])));
        scale=scale/127.f;
        std::memcpy(spans[layer*4+role*2+1].data()+splash_kv_scale_element(head,token)*4,&scale,4);
      }
      for(uint32_t d=0;d<layout.headDimension;++d) {
        const uint64_t at=role?splash_kv_value_element(head,token,d):splash_kv_key_element(head,token,d);
        if(store.layout().format==kv::Format::BFloat16)std::memcpy(spans[layer*2+role].data()+at*2,values+d,2);
        else { const int8_t code=scale>0?int8_t(std::clamp(std::nearbyint(asFloat(values[d])/scale),-127.f,127.f)):0;std::memcpy(spans[layer*4+role*2].data()+at,&code,1); }
      }
    }
  }
  void attention(Request &r,const std::string &p,uint32_t ordinal,const ModelBatchItem &item,uint32_t position,uint32_t rows=1,bool cacheOnly=false) {
    const bool mtp=p.starts_with("mtp.");
    auto &store=mtp?*r.mtpKv:context.kvPages;
    CommandGraph projections;
    for(const auto &[name,out] : std::array<std::pair<const char*,const char*>,5>{{
        {"attn_q.weight","attn_qg"},{"attn_k.weight","attn_k"},{"attn_v.weight","attn_v"},
        {"indexer.q_proj.weight","index_q"},{"indexer.k_proj.weight","index_k"}}}) {
      if(cacheOnly && (std::string_view(out)=="attn_qg" || std::string_view(out)=="index_q"))continue;
      project(projections,p+name,b("mixed"),b(out),ops::FloatOutput::BFloat16,rows);
    }
    run(projections);
    const auto *qg=static_cast<const uint16_t*>(b("attn_qg").contents());
    if(!cacheOnly)for(uint32_t row=0;row<rows;++row)for(uint32_t h=0;h<layout.queryHeads;++h) {
      const auto offset=row*layout.queryHeads+h;
      std::memcpy(static_cast<uint16_t*>(b("attn_q").contents())+offset*256,qg+offset*512,512);
      std::memcpy(static_cast<uint16_t*>(b("attn_gate").contents())+offset*256,qg+offset*512+256,512);
    }
    const uint64_t rawLayerBytes=uint64_t{layout.maximumContextTokens}*layout.indexWidth*2;
    auto raw=context.backend.view(mtp?r.mtpRawKeys:r.rawKeys,ordinal*rawLayerBytes,rawLayerBytes);
    auto pooled=context.backend.view(mtp?r.mtpPooledKeys:r.pooledKeys,ordinal*(rawLayerBytes/layout.indexRatio),rawLayerBytes/layout.indexRatio);
    std::memcpy(static_cast<uint8_t*>(raw.contents())+uint64_t{position}*layout.indexWidth*2,b("index_k").contents(),uint64_t{rows}*layout.indexWidth*2);
    ropeTable(position,rows);
    CommandGraph norms;
    ops::Normalization::addRms(norms,b("attn_k"),weights.norm(p+"attn_k_norm.weight"),b("attn_k"),256,rows*layout.kvHeads);
    rope(norms,b("attn_k"),256,layout.kvHeads,rows);
    if(!cacheOnly) {
      ops::Normalization::addRms(norms,b("attn_q"),weights.norm(p+"attn_q_norm.weight"),b("attn_q"),256,rows*layout.queryHeads);
      ops::Normalization::addRms(norms,b("index_q"),weights.norm(p+"indexer.q_norm.weight"),b("index_q"),128,rows*layout.indexHeads);
      rope(norms,b("attn_q"),256,layout.queryHeads,rows);rope(norms,b("index_q"),128,layout.indexHeads,rows);
    }
    run(norms);
    for(uint32_t row=0;row<rows;++row)writeKv(r,item,ordinal,position+row,row,mtp);
    const uint32_t first=position/layout.indexRatio,blocks=(position+rows)/layout.indexRatio;
    if(blocks>first) {
      CommandGraph pool;ops::FlashNext::addQsaPool(pool,raw,pooled,layout.indexWidth,layout.indexRatio,blocks-first,first);run(pool);
      const uint32_t count=blocks-first;
      auto current=context.backend.view(pooled,uint64_t{first}*layout.indexWidth*2,uint64_t{count}*layout.indexWidth*2);
      ropeTable(first*layout.indexRatio,count,layout.indexRatio);
      CommandGraph norm;ops::Normalization::addRms(norm,current,weights.norm(p+"indexer.k_norm.weight"),current,128,count);rope(norm,current,128,1,count);run(norm);
    }
    if(cacheOnly)return;
    auto *positions=static_cast<uint32_t*>(b("position").contents());
    for(uint32_t row=0;row<rows;++row)positions[row]=position+row;
    store.writeEntries(item.pageTable,0,b("page_table"));
    CommandGraph selected;
    if(blocks)ops::FlashNext::addQsaScores(selected,b("index_q"),pooled,b("position"),b("scores"),layout.indexWidth,layout.indexHeads,rows,blocks,layout.indexRatio);
    ops::FlashNext::addQsaSelect(selected,b("scores"),b("position"),b("selected"),rows,blocks,layout.indexRatio,layout.indexBudget);
    ops::FlashNext::addQsaAttention(selected,b("attn_q"),b("attn_gate"),b("page_table"),b("selected"),b("attn_out"),store.layers()[ordinal],layout.queryHeads,rows,layout.indexBudget+layout.indexRatio-1,item.pageTable.size(),store.layout().format);
    project(selected,p+"attn_output.weight",b("attn_out"),b("block"),ops::FloatOutput::BFloat16,rows);run(selected);
  }
  struct ExpertBinding {
    ops::Projection projection;
    std::shared_ptr<const FlashNextWeights::ExpertPage> lease;
  };
  ExpertBinding expert(const std::string &name,uint32_t id) {
    if(name.starts_with("mtp."))return {weights.projection(name+"."+std::to_string(id)),{}};
    auto page=acquireFlashNextExpert(weights,context.backend,name,id);
    return {page->projection,std::move(page)};
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
      const auto gate=expert(p+"ffn_gate_exps.weight",experts[i]);
      const auto up=expert(p+"ffn_up_exps.weight",experts[i]);
      const auto down=expert(p+"ffn_down_exps.weight",experts[i]);
      CommandGraph g;project(g,gate.projection,b("mixed"),b("gate"));project(g,up.projection,b("mixed"),b("up"));
      ops::FlashNext::addSiluMul(g,b("gate"),b("up"),b("product"),layout.intermediateSize,1);
      project(g,down.projection,b("product"),b("ffn_out"),ops::FloatOutput::Float32);run(g);
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
  void moeBatch(const std::string &p,uint32_t rows) {
    if(rows==1){moe(p);return;}
    CommandGraph route;project(route,p+"ffn_gate_inp.weight",b("mixed"),b("router"),ops::FloatOutput::Float32,rows);run(route);
    struct RoutedRow { uint32_t row,rank; };
    std::map<uint32_t,std::vector<RoutedRow>> groups;
    std::vector<float> masses(rows*layout.expertsPerToken);
    const auto *logits=static_cast<const float*>(b("router").contents());
    for(uint32_t row=0;row<rows;++row) {
      std::vector<uint32_t> experts(layout.experts);std::iota(experts.begin(),experts.end(),0);
      const auto *scores=logits+row*layout.experts;
      for(uint32_t i=0;i<layout.experts;++i)requireFiniteFlashNextLogit(scores[i],"Flash Next router produced non-finite logits");
      std::partial_sort(experts.begin(),experts.begin()+layout.expertsPerToken,experts.end(),[&](uint32_t a,uint32_t c){return scores[a]==scores[c]?a<c:scores[a]>scores[c];});
      float total=0;
      for(uint32_t rank=0;rank<layout.expertsPerToken;++rank){auto &mass=masses[row*layout.expertsPerToken+rank];mass=std::exp(scores[experts[rank]]-scores[experts[0]]);total+=mass;groups[experts[rank]].push_back({row,rank});}
      for(uint32_t rank=0;rank<layout.expertsPerToken;++rank)masses[row*layout.expertsPerToken+rank]/=total;
    }
    const auto *mixed=static_cast<const uint16_t*>(b("mixed").contents());
    auto *result=static_cast<float*>(b("expert_results").contents());
    for(const auto &[expertId,selected] : groups) {
      auto *input=static_cast<uint16_t*>(b("expert_input").contents());
      for(uint32_t i=0;i<selected.size();++i)std::memcpy(input+i*layout.hiddenSize,mixed+selected[i].row*layout.hiddenSize,layout.hiddenSize*2);
      const auto gate=expert(p+"ffn_gate_exps.weight",expertId);
      const auto up=expert(p+"ffn_up_exps.weight",expertId);
      const auto down=expert(p+"ffn_down_exps.weight",expertId);
      const uint32_t count=static_cast<uint32_t>(selected.size());
      CommandGraph g;
      project(g,gate.projection,b("expert_input"),b("gate"),ops::FloatOutput::BFloat16,count,layout.hiddenSize,layout.intermediateSize);
      project(g,up.projection,b("expert_input"),b("up"),ops::FloatOutput::BFloat16,count,layout.hiddenSize,layout.intermediateSize);
      ops::FlashNext::addSiluMul(g,b("gate"),b("up"),b("product"),layout.intermediateSize,count);
      project(g,down.projection,b("product"),b("ffn_out"),ops::FloatOutput::Float32,count,layout.intermediateSize,layout.hiddenSize);run(g);
      const auto *out=static_cast<const float*>(b("ffn_out").contents());
      for(uint32_t i=0;i<count;++i)std::memcpy(result+(selected[i].row*layout.expertsPerToken+selected[i].rank)*layout.hiddenSize,out+i*layout.hiddenSize,layout.hiddenSize*4);
    }
    CommandGraph shared;
    project(shared,p+"ffn_gate_shexp.weight",b("mixed"),b("gate"),ops::FloatOutput::BFloat16,rows);
    project(shared,p+"ffn_up_shexp.weight",b("mixed"),b("up"),ops::FloatOutput::BFloat16,rows);
    ops::FlashNext::addSiluMul(shared,b("gate"),b("up"),b("product"),layout.intermediateSize,rows);
    project(shared,p+"ffn_down_shexp.weight",b("product"),b("ffn_out"),ops::FloatOutput::Float32,rows);run(shared);
    auto *accum=static_cast<float*>(b("accum").contents());
    const auto *sharedWeight=static_cast<const float*>(weights.tensors.at(p+"ffn_gate_inp_shexp.weight").contents());
    const auto *sharedOut=static_cast<const float*>(b("ffn_out").contents());
    for(uint32_t row=0;row<rows;++row) {
      float scalar=0;for(uint32_t i=0;i<layout.hiddenSize;++i)scalar+=sharedWeight[i]*asFloat(mixed[row*layout.hiddenSize+i]);
      const float scale=1.f/(1.f+std::exp(-scalar));
      for(uint32_t col=0;col<layout.hiddenSize;++col) {
        float value=0;
        for(uint32_t rank=0;rank<layout.expertsPerToken;++rank)value+=result[(row*layout.expertsPerToken+rank)*layout.hiddenSize+col]*masses[row*layout.expertsPerToken+rank];
        accum[row*layout.hiddenSize+col]=value+sharedOut[row*layout.hiddenSize+col]*scale;
      }
    }
    CommandGraph rounded;ops::FlashNext::addFloatToBf16(rounded,b("accum"),b("block"),layout.hiddenSize,rows);run(rounded);
  }
  void draftForward(Request &r,std::span<const uint32_t> tokens,std::span<const uint16_t> features,bool headNeeded) {
    const uint32_t rows=static_cast<uint32_t>(tokens.size());
    if(!r.mtpKv || !rows || rows>kFlashNextPrefillRows || features.size()!=uint64_t{rows}*layout.hyperWidth())
      throw std::invalid_argument("invalid MTP token/previous-stream pairing");
    std::vector<uint64_t> tokenRows(tokens.begin(),tokens.end());
    gather("token_embd.weight",tokenRows,b("embedding"));
    std::memcpy(b("mtp_pairs").contents(),features.data(),features.size_bytes());
    CommandGraph fused;
    ops::FlashNext::addGroupedRms(fused,b("embedding"),weights.norm("mtp.embedding_norm.weight"),b("mtp_embed_norm"),layout.hiddenSize,1,rows);
    ops::FlashNext::addGroupedRms(fused,b("mtp_pairs"),weights.norm("mtp.hidden_norm.weight"),b("mtp_normalized"),layout.hyperWidth(),1,rows);
    project(fused,"mtp.fc_embedding.weight",b("mtp_embed_norm"),b("mtp_embed"),ops::FloatOutput::BFloat16,rows);
    project(fused,"mtp.fc_hidden.weight",b("mtp_normalized"),b("mtp_hidden"),ops::FloatOutput::BFloat16,rows*layout.streams);
    ops::FlashNext::addBroadcast(fused,b("mtp_embed"),b("hyper"),layout.hiddenSize,layout.streams,rows);
    ops::FlashNext::addAdd(fused,b("hyper"),b("mtp_hidden"),b("hyper"),layout.hyperWidth(),rows);run(fused);
    ModelBatchItem item;item.requestId=r.id;item.pageTable=r.mtpPages;item.pageTableRevision=1;
    const std::string prefix="mtp.blk.0.";
    CommandGraph mix;hyperMix(mix,prefix+"hc_attn_",true,rows);run(mix);
    attention(r,prefix,0,item,r.mtpCommitted,rows,!headNeeded);
    // Teacher rows only publish attention KV/index state. Their output streams
    // are supplied by the target on the next call, so no MTP FFN is needed.
    if(!headNeeded) { r.mtpCommitted+=rows;return; }
    CommandGraph scatter;ops::FlashNext::addHcScatter(scatter,b("hyper"),b("block"),b("inject"),b("hyper"),layout.hiddenSize,layout.streams,rows);run(scatter);
    CommandGraph ffn;hyperMix(ffn,prefix+"hc_ffn_",true,rows);run(ffn);moeBatch(prefix,rows);
    CommandGraph finish;ops::FlashNext::addHcScatter(finish,b("hyper"),b("block"),b("inject"),b("hyper"),layout.hiddenSize,layout.streams,rows);run(finish);
    if(headNeeded) {
      CommandGraph head;hyperMix(head,"mtp.output_hc_",false,rows);
      auto last=context.backend.view(b("mixed"),uint64_t{rows-1}*layout.hiddenSize*2,layout.hiddenSize*2);
      project(head,"output.weight",last,b("logits"),ops::FloatOutput::Float32);run(head);
    }
    r.mtpCommitted+=rows;
  }
  void forwardBatch(Request &r,std::span<const uint32_t> tokens,const ModelBatchItem &item,bool headNeeded=true,bool allHeads=false) {
    const uint32_t rows=static_cast<uint32_t>(tokens.size());
    if(!rows || rows>kFlashNextPrefillRows || rows>layout.maximumContextTokens-r.committed)
      throw std::invalid_argument("Flash Next batch exceeds its context or row budget");
    std::vector<uint64_t> keys,tokenRows;keys.reserve(rows*layout.pleHeads);
    for(uint32_t token:tokens){auto next=r.hash.push(token);keys.insert(keys.end(),next.begin(),next.end());tokenRows.push_back(token);}
    gather("token_embd.weight",tokenRows,b("embedding"));
    CommandGraph initial;ops::FlashNext::addBroadcast(initial,b("embedding"),b("hyper"),layout.hiddenSize,layout.streams,rows);run(initial);
    uint32_t ai=0,gi=0;
    for(uint32_t layer=0;layer<layout.layers;++layer) {
      const std::string p="blk."+std::to_string(layer)+".";
      if(layer==layout.pleLayer)ple(r,p,keys,rows);
      CommandGraph mix;hyperMix(mix,p+"hc_attn_",true,rows);run(mix);
      if(layout.isAttention(layer))attention(r,p,ai++,item,r.committed,rows);else gdn(r,p,gi++,rows);
      CommandGraph scatter;ops::FlashNext::addHcScatter(scatter,b("hyper"),b("block"),b("inject"),b("hyper"),layout.hiddenSize,layout.streams,rows);run(scatter);
      CommandGraph ffnMix;hyperMix(ffnMix,p+"hc_ffn_",true,rows);run(ffnMix);moeBatch(p,rows);
      CommandGraph ffnScatter;ops::FlashNext::addHcScatter(ffnScatter,b("hyper"),b("block"),b("inject"),b("hyper"),layout.hiddenSize,layout.streams,rows);run(ffnScatter);
    }
    if(headNeeded) {
      CommandGraph head;hyperMix(head,"output_hc_",false,rows);
      const auto last=context.backend.view(b("mixed"),uint64_t{rows-1}*layout.hiddenSize*2,layout.hiddenSize*2);
      if(allHeads)project(head,"output.weight",b("mixed"),b("verify_logits"),ops::FloatOutput::Float32,rows);
      else project(head,"output.weight",last,b("logits"),ops::FloatOutput::Float32);
      run(head);
    }
    std::swap(r.gdn[0],r.gdn[1]);r.committed+=rows;r.logitsReady=headNeeded;
    if(weights.hasMtp && !r.captureVerify) {
      const auto *hidden=static_cast<const uint16_t*>(b("hyper").contents());
      std::vector<uint16_t> saved(hidden,hidden+uint64_t{rows}*layout.hyperWidth());
      if(useMtp && r.maxNew>1 && r.scoreTokens.empty()) {
        std::vector<uint16_t> previous(saved.size());
        std::memcpy(previous.data(),r.targetFeature.contents(),layout.hyperWidth()*2);
        if(rows>1)std::memcpy(previous.data()+layout.hyperWidth(),saved.data(),uint64_t{rows-1}*layout.hyperWidth()*2);
        draftForward(r,tokens,previous,false);
      }
      std::memcpy(r.targetFeature.contents(),saved.data()+uint64_t{rows-1}*layout.hyperWidth(),layout.hyperWidth()*2);
    }
  }
  void forward(Request &r,uint32_t token,const ModelBatchItem &item,bool headNeeded=true) {
    forwardBatch(r,{&token,1},item,headNeeded);
  }
  uint32_t draftToken(const Request &r) {
    const auto *logits=static_cast<const float*>(b("logits").contents());
    uint32_t best=0;bool found=false;
    for(uint32_t token=0;token<layout.vocabularySize;++token) {
      requireFiniteFlashNextLogit(logits[token],"MTP produced non-finite logits");
      if((r.flags&RequestIgnoreEndOfSequence) && (token==layout.stopTokens[0]||token==layout.stopTokens[1]))continue;
      if(!found || logits[token]>logits[best]){best=token;found=true;}
    }
    if(!found)throw std::logic_error("MTP vocabulary has no candidate");return best;
  }
  ModelStepResult speculative(Request &r,const ModelBatchItem &item) {
    if(!r.pending || r.mtpCommitted!=r.committed)throw std::logic_error("MTP teacher cache is not at the committed prefix");
    ModelStepResult result;result.requestId=r.id;
    const uint32_t start=r.committed;
    const auto beforeHash=r.hash;
    std::vector<uint16_t> previous(layout.hyperWidth());std::memcpy(previous.data(),r.targetFeature.contents(),layout.hyperWidth()*2);
    const uint32_t capacity=std::min({r.mtpDepth+1,r.maxNew-r.generated,layout.maximumContextTokens-start});
    std::vector<uint32_t> window{*r.pending};
    for(uint32_t i=1;i<capacity;++i) {
      const auto last=window.back();draftForward(r,{&last,1},previous,true);
      const uint32_t next=draftToken(r);window.push_back(next);
      std::memcpy(previous.data(),b("hyper").contents(),layout.hyperWidth()*2);
      if(!(r.flags&RequestIgnoreEndOfSequence) && (next==layout.stopTokens[0]||next==layout.stopTokens[1]))break;
    }
    result.draftedTokens=static_cast<uint32_t>(window.size()-1);
    std::memcpy(b("verify_ple_before").contents(),r.ple.contents(),r.ple.sizeBytes());
    r.captureVerify=true;r.verifyRows=static_cast<uint32_t>(window.size());
    forwardBatch(r,window,item,true,true);r.captureVerify=false;
    const auto *features=static_cast<const uint16_t*>(b("hyper").contents());
    std::vector<uint16_t> verified(features,features+window.size()*layout.hyperWidth());
    const auto *logits=static_cast<const float*>(b("verify_logits").contents());
    uint32_t retained=0;
    for(uint32_t row=0;row<window.size();++row) {
      const auto token=window[row];result.outputTokens.push_back(token);++r.generated;r.outputHistory.push_back(token);++retained;
      const bool stop=!(r.flags&RequestIgnoreEndOfSequence) && (token==layout.stopTokens[0]||token==layout.stopTokens[1]);
      if(stop || r.generated==r.maxNew){r.pending.reset();result.finished=stop;break;}
      r.pending=sample(r,logits+uint64_t{row}*layout.vocabularySize);
      if(row+1==window.size() || *r.pending!=window[row+1])break;
    }
    result.acceptedDraftTokens=retained-1;
    if(result.draftedTokens) {
      if(result.acceptedDraftTokens*2<result.draftedTokens)r.mtpDepth=1;
      else if(result.acceptedDraftTokens==result.draftedTokens)r.mtpDepth=std::min(kMtpDrafts,r.mtpDepth+1);
    }
    if(retained<window.size()) {
      // GDN's previous cell remained untouched during batch verify. Replay
      // only recurrence operands for the retained prefix, not all projections.
      std::swap(r.gdn[0],r.gdn[1]);uint32_t ordinal=0;
      for(uint32_t layer=0;layer<layout.layers;++layer)if(!layout.isAttention(layer)) {
        const auto *packed=static_cast<const uint8_t*>(b("verify_gdn").contents())+uint64_t{ordinal}*r.verifyRows*16640*2;
        std::memcpy(b("gdn_packed").contents(),packed,uint64_t{retained}*16640*2);
        gdn(r,"blk."+std::to_string(layer)+".",ordinal++,retained,false);
      }
      std::swap(r.gdn[0],r.gdn[1]);
      std::memcpy(r.ple.contents(),b("verify_ple_before").contents(),r.ple.sizeBytes());
      CommandGraph history;ops::FlashNext::addPleCommit(history,b("verify_ple_norm"),r.ple,layout.hiddenSize,layout.streams,retained,layout.pleTaps,layout.pleDilation);run(history);
      r.hash=beforeHash;for(uint32_t i=0;i<retained;++i)static_cast<void>(r.hash.push(window[i]));
      r.committed=start+retained;
    }
    // Speculative K/V past the logical length is invisible. Rewrite every
    // retained MTP row using true target streams, so future drafts use teacher context.
    r.mtpCommitted=start;
    std::vector<uint16_t> teacher(uint64_t{retained}*layout.hyperWidth());
    std::memcpy(teacher.data(),r.targetFeature.contents(),layout.hyperWidth()*2);
    if(retained>1)std::memcpy(teacher.data()+layout.hyperWidth(),verified.data(),uint64_t{retained-1}*layout.hyperWidth()*2);
    draftForward(r,std::span<const uint32_t>(window).first(retained),teacher,false);
    std::memcpy(r.targetFeature.contents(),verified.data()+uint64_t{retained-1}*layout.hyperWidth(),layout.hyperWidth()*2);
    return result;
  }
  uint32_t sample(Request &r,const float *source=nullptr) {
    if(source)std::memcpy(b("logits").contents(),source,layout.vocabularySize*4);
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
    if(impl_->weights.hasMtp) {
      const kv::Layout cache{1,l.kvHeads,l.headDimension,kv::Format::BFloat16};
      const uint32_t pages=(l.maximumContextTokens+31)/32;
      const metal::AllocationAdmission admitted=[](uint64_t,const std::function<void()> &work){work();return metal::AllocationResult{};};
      r->mtpKv=std::make_unique<kv::PageStorage>(impl_->context.backend,admitted,cache,pages,pages);
      if(!r->mtpKv->allocateExtent(0))throw std::logic_error("MTP KV allocation failed inside admitted lane");
      r->mtpPages.resize(pages);std::iota(r->mtpPages.begin(),r->mtpPages.end(),0);
      const uint64_t index=uint64_t{l.maximumContextTokens}*l.indexWidth*2;
      r->mtpRawKeys=impl_->context.backend.allocateBuffer(index,metal::BufferStorage::Shared,"mtp-indexer-raw");
      r->mtpPooledKeys=impl_->context.backend.allocateBuffer(index/l.indexRatio,metal::BufferStorage::Shared,"mtp-indexer-pooled");
      r->targetFeature=impl_->context.backend.allocateBuffer(l.hyperWidth()*2,metal::BufferStorage::Shared,"mtp-target-streams");
      std::memset(r->targetFeature.contents(),0,r->targetFeature.sizeBytes());
    }
  };
  if(impl_->context.admitAllocation) {
    auto admitted=impl_->context.admitAllocation(l.stateBytes()+(impl_->weights.hasMtp?mtpStateBytes(l):0),allocate);
    if(!admitted)return {{},StateFailure::MemoryPressure,admitted.failure};
  } else allocate();
  impl_->requests.emplace(request.id,std::move(r));return {lane,StateFailure::None};
}
void FlashNextRuntime::suspend(uint64_t id) {
  auto &r=*impl_->requests.at(id);r.gdn={};r.ple={};r.rawKeys={};r.pooledKeys={};r.mtpKv.reset();r.mtpRawKeys={};r.mtpPooledKeys={};r.targetFeature={};r.active=false;r.logitsReady=false;
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
  if(impl_->computing.load())throw std::logic_error("Flash Next has a compute batch in flight");
  impl_->stateAtSubmission=actualRuntimeMemory().stateActualAllocatedBytes;
  {std::lock_guard lock(impl_->publicTelemetryMutex);impl_->publicTelemetry=impl_->counters;}
  impl_->computing=true;
  auto ticket=std::make_unique<AsyncTicket>();
  std::vector<ModelBatchItem> owned(items.begin(),items.end());
  ticket->future=std::async(std::launch::async,[this,plan,owned=std::move(owned),completion=std::move(completion),done=ticket->done]() mutable {
    try {
      auto result=submitSynchronous(plan,owned);
      {std::lock_guard lock(impl_->publicTelemetryMutex);impl_->publicTelemetry=impl_->counters;}
      impl_->computing=false;done->store(true);if(completion)completion();return result;
    } catch(...) {impl_->computing=false;done->store(true);if(completion)completion();throw;}
  });
  return ticket;
}
std::unique_ptr<ModelBatchTicket> FlashNextRuntime::submitSynchronous(const BatchPlan &plan,std::span<const ModelBatchItem> items) {
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
      impl_->forwardBatch(r,std::span<const uint32_t>(r.prompt).subspan(batch.promptOffset,batch.tokenCount),item,batch.promptOffset+batch.tokenCount==r.prompt.size());
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
        if(impl_->useMtp && r.constraint==ConstraintMode::None) {
          result=impl_->speculative(r,item);
        } else {
          const auto committed=*r.pending;impl_->forward(r,committed,item);r.pending.reset();
          result.outputTokens.push_back(committed);++r.generated;r.outputHistory.push_back(committed);
        }
      }
      if(r.generated<r.maxNew && !(impl_->useMtp && r.constraint==ConstraintMode::None && plan.decodeStage!=DecodeStage::ApplyInitialMask))r.pending=impl_->sample(r);
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
  return ticket;
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
  auto result=submit(plan,{&item,1},{});const auto steps=result->wait();WarmupStepResult out;out.wallSeconds=result->wallMilliseconds()/1000;out.detail="Flash Next cold SSD autoregressive prefill, "+std::to_string(rows)+" rows";
  out.lanes.push_back({steps.front(),impl_->requests.at(id)->pending,rows});end(id);return out;
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
  if(impl_->computing.load())return {impl_->stateAtSubmission.load(),0,impl_->scratchAllocated+flashNextExpertTelemetry(impl_->weights).residentBytes,0};
  uint64_t state=0;for(const auto &[id,r]:impl_->requests){static_cast<void>(id);for(const auto &cell:r->gdn)state+=cell.sizeBytes();state+=r->ple.sizeBytes()+r->rawKeys.sizeBytes()+r->pooledKeys.sizeBytes()+r->mtpRawKeys.sizeBytes()+r->mtpPooledKeys.sizeBytes()+r->targetFeature.sizeBytes()+(r->mtpKv?r->mtpKv->actualAllocatedBytes():0);}
  return {state,0,impl_->scratchAllocated+flashNextExpertTelemetry(impl_->weights).residentBytes,0};
}
ModelTelemetry FlashNextRuntime::telemetry() const noexcept {
  ModelTelemetry out;
  {std::lock_guard lock(impl_->publicTelemetryMutex);out=impl_->publicTelemetry;}
  out.stateAllocatedBytes=actualRuntimeMemory().stateActualAllocatedBytes;
  const auto experts=flashNextExpertTelemetry(impl_->weights),host=impl_->weights.hostPages->telemetry();
  out.expertCacheHits=experts.hits;out.expertCacheMisses=experts.misses;out.expertCacheBytesRead=experts.bytesRead;
  out.expertCacheResidentBytes=experts.residentBytes;out.expertCacheEvictions=experts.evictions;
  out.weightHostCacheHits=host.hits;out.weightHostCacheMisses=host.misses;out.weightHostCacheResidentBytes=host.residentBytes;return out;
}
} // namespace splash::model
