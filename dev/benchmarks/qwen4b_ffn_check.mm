// Real GGUF FFN operands, production decode and ANE GPU views.
// Independent GGUF row decoding supplies both half-weight and affine references.
#include "tests/engine/GgufFormatReference.hpp"
#include "tests/engine/LinearNumerics.hpp"
#include "model/ModelFactory.hpp"
#include "model/GgufTarget.hpp"
#include "model/GgufFile.hpp"
#include "ops/GgufPrefillCache.hpp"
#include "ops/Linear.hpp"
#import <Foundation/Foundation.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>

using namespace splash;
namespace {
uint64_t comparisons = 0;
bool originalContract = true;
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
metal::MetalBuffer allocate(metal::MetalBackend &backend, uint64_t bytes) {
  auto buffer = backend.allocateBuffer(std::max<uint64_t>(bytes, 4), metal::BufferStorage::Shared, "FFN numerical fixture");
  std::memset(buffer.contents(), 0, buffer.sizeBytes()); return buffer;
}
void emit(NSDictionary *record) {
  NSData *json = [NSJSONSerialization dataWithJSONObject:record options:0 error:nullptr];
  require(json != nil, "cannot encode numerical result");
  std::cout.write(static_cast<const char *>(json.bytes), json.length); std::cout << '\n' << std::flush;
}
std::set<uint32_t> sampleColumns(uint32_t n) { return {0,31,32,63,64,255,256,n/2,n-1}; }
std::set<uint32_t> sampleRows(uint32_t n) { return {0,std::min(31U,n-1),std::min(32U,n-1),n/2,n-1}; }
struct Row {
  uint32_t column;
  std::vector<float> half, codes, scale, offset;
};
Row readRow(model::WeightSource &source, const model::GgufTensor &tensor, uint32_t format, uint32_t column, uint32_t k) {
  const auto fmt = gguf_reference::Fmt(format);
  const uint32_t bytes = gguf_reference::rowBytes(fmt, uint32_t(tensor.columns()));
  std::vector<uint8_t> native(bytes); source.readData(tensor.offset + uint64_t{column} * bytes, native);
  Row row{column, std::vector<float>(k), {}, {}, {}};
  if (format == GGUF_FMT_Q4K) { row.codes.resize(k); row.scale.resize(k/32); row.offset.resize(k/32); }
  for (uint32_t g=0; g<k/32; ++g) {
    uint8_t p0[32], p1[8]; float values[32];
    gguf_reference::groupPack(fmt, native.data(), g, values, p0, p1);
    for (uint32_t e=0; e<32; ++e) row.half[g*32+e] = gguf_reference::h2f(gguf_reference::f2h(values[e]));
    if (format != GGUF_FMT_Q4K) continue;
    // Independent GGML block_q4_K header, rather than the shader's repacked meta decoder.
    const auto *block = native.data() + (g/8)*144;
    uint16_t d, m; std::memcpy(&d,block,2); std::memcpy(&m,block+2,2);
    const auto *sc = block+4; const uint32_t j=g%8;
    const uint32_t scale = j<4 ? sc[j]&63 : (sc[j+4]&15) | ((sc[j-4]>>6)<<4);
    const uint32_t offset = j<4 ? sc[j+4]&63 : (sc[j+4]>>4) | ((sc[j]>>6)<<4);
    row.scale[g] = gguf_reference::h2f(d)*float(scale);
    row.offset[g] = -gguf_reference::h2f(m)*float(offset);
    for (uint32_t e=0;e<32;++e) {
      uint32_t word; std::memcpy(&word,p0+4*((e&15)/4),4);
      const uint32_t pair = (e>=16?2:0)+(e&3)/2;
      row.codes[g*32+e] = float((word>>(pair*4+(e&1)*16))&15);
    }
  }
  return row;
}
struct Dot { double half=0, delayed=0, magnitude=0, weightError=0; };
Dot dot(const Row &weight, const uint16_t *input, bool delayed) {
  Dot result;
  for (uint32_t g=0;g<weight.half.size()/32;++g) {
    double codes=0, ones=0, rounding=0;
    for (uint32_t e=0;e<32;++e) {
      const uint32_t i=g*32+e; const double x=test::bf16ToFloat(input[i]);
      const double product=x*weight.half[i]; result.half+=product; result.magnitude+=std::fabs(product);
      if (delayed) {
        codes+=x*weight.codes[i]; ones+=x;
        const double affine=double(weight.scale[g])*weight.codes[i]+weight.offset[g];
        result.weightError+=std::fabs(x)*std::fabs(affine-weight.half[i]);
        rounding+=x*(double(weight.half[i])-affine);
      }
    }
    if (delayed) result.delayed+=double(weight.scale[g])*codes+double(weight.offset[g])*ones+rounding;
  }
  if (!delayed) result.delayed=result.half;
  return result;
}
float bf(double v) { return test::bf16ToFloat(test::floatToBf16(float(v))); }
float silu(float v) { return v/(1+std::exp(-v)); }
ops::Projection cached(metal::MetalBackend &backend, const ops::Projection &full, const ops::Projection &view) {
  auto whole=full.blocks().segments.front();
  whole.prefillHalf=allocate(backend,ops::ggufPrefillCacheBytes(whole));
  ops::writeGgufPrefillCache(backend,whole,whole.prefillHalf);
  auto blocks=view.blocks(); auto &segment=blocks.segments.front();
  const uint32_t n=view.outputSize,k=view.inputSize;
  if (k==full.inputSize) segment.prefillHalf=backend.view(whole.prefillHalf,0,uint64_t{n}*k*2);
  else {
    segment.prefillHalf=allocate(backend,uint64_t{n}*k*2);
    const auto *source=static_cast<const uint16_t *>(whole.prefillHalf.contents());
    auto *to=static_cast<uint16_t *>(segment.prefillHalf.contents());
    for (uint32_t row=0;row<n;++row) std::copy_n(source+uint64_t{row}*full.inputSize,k,to+uint64_t{row}*k);
  }
  // The compact half image has logical K stride; quant planes are not read by its prefill kernel.
  ops::Projection result(n,k,std::move(blocks)); result.ffn=view.ffn;
  result.ffn.variant=ops::FfnPrefillVariant::Baseline; return result;
}
ops::LinearScratch scratch(metal::MetalBackend &backend, const ops::LinearScratchSize &s) {
  return {allocate(backend,s.input),allocate(backend,s.sums),allocate(backend,s.partials),allocate(backend,s.counters),allocate(backend,s.rotated)};
}
void fill(metal::MetalBuffer buffer,uint32_t seed) {
  auto *values=static_cast<uint16_t *>(buffer.contents());
  for (uint64_t i=0;i<buffer.sizeBytes()/2;++i) values[i]=test::floatToBf16(float(int((i*17+seed*13)%101)-50)/53);
}
bool delayed(const ops::Projection &p, uint32_t rows) {
  return p.ffn.variant==ops::FfnPrefillVariant::Q4KGroupScale && rows>=64 &&
      p.blocks().segments.front().formatId==GGUF_FMT_Q4K;
}
void check(metal::MetalBackend &backend, model::WeightSource &source, const model::GgufTensor &native,
    const ops::Projection &full, const ops::Projection &q, ops::LinearEpilogue ep, uint32_t layer,
    const char *role, const std::vector<uint32_t> &rowCounts, const char *scope) {
  const auto &segment=q.blocks().segments.front(); const uint32_t n=q.outputSize,k=q.inputSize;
  require(!segment.isFloat() && !segment.prefillHalf && !q.rotation,"requires uncached quantized operand");
  auto h=cached(backend,full,q);
  std::vector<Row> weights;
  for (auto column:sampleColumns(n)) weights.push_back(readRow(source,native,segment.formatId,column,k));
  const auto *half=static_cast<const uint16_t *>(h.blocks().segments.front().prefillHalf.contents());
  for (const auto &row:weights) for (uint32_t i=0;i<k;++i)
    require(half[uint64_t{row.column}*k+i]==gguf_reference::f2h(row.half[i]),"GPU cache differs from independent native GGUF decoding");
  ops::Linear linear(backend.capabilities());
  for (uint32_t rows:rowCounts) {
    const auto plan=linear.prefillPlan(q,rows,ep); const uint32_t storage=plan.storageRows();
    const auto input=allocate(backend,uint64_t{storage}*k*2), aux=allocate(backend,uint64_t{storage}*n*2);
    auto outputQ=allocate(backend,uint64_t{storage+128}*n*2),outputH=allocate(backend,uint64_t{storage+128}*n*2);
    fill(input,layer); fill(aux,layer+11); const auto work=scratch(backend,plan.scratchSize());
    for(auto output:{outputQ,outputH}) std::fill_n(static_cast<uint16_t *>(output.contents()),output.sizeBytes()/2,0xffff);
    auto run=[&](const ops::Projection &p,metal::MetalBuffer output) {
      metal::CommandGraph graph; linear.add(graph,{.input=input,.output=output,.residual=aux,.gateScratch=aux,.scratch=work},p,linear.prefillPlan(p,rows,ep));
      static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
      uint32_t written=linear.prefillPlan(p,rows,ep).storageRows();
      if (plan.configuration().tile==ops::LinearTile::GgufPrefill) {
        const auto choice=ops::ffnPrefillKernel(p,plan);
        const uint32_t tile=choice==ops::FfnPrefillKernel::M64N32 || (p.blocks().segments.front().prefillHalf && rows>=64)?64:32;
        written=choice==ops::FfnPrefillKernel::Q4KGroupScale?rows:(rows+tile-1)/tile*tile;
      }
      const auto *out=static_cast<const uint16_t *>(output.contents());
      for(uint64_t at=uint64_t{written}*n;at<uint64_t{storage+128}*n;++at) require(out[at]==0xffff,"FFN candidate wrote past its padded rows");
    };
    run(q,outputQ);run(h,outputH);
    const auto *x=static_cast<const uint16_t *>(input.contents()),*a=static_cast<const uint16_t *>(aux.contents());
    const auto *outQ=static_cast<const uint16_t *>(outputQ.contents()),*outH=static_cast<const uint16_t *>(outputH.contents());
    double maximum=0,square=0,maxCpu=0,maxOriginal=0,maxBudget=0;uint64_t differences=0,checked=0;bool originalPass=true;
    for(uint64_t i=0;i<uint64_t{rows}*n;++i) {
      const double v=test::bf16ToFloat(outQ[i]),other=test::bf16ToFloat(outH[i]);require(std::isfinite(v)&&std::isfinite(other),"nonfinite FFN output");
      const double d=std::fabs(v-other);maximum=std::max(maximum,d);square+=d*d;differences+=outQ[i]!=outH[i];
    }
    for(auto row:sampleRows(rows)) for(const auto &weight:weights) {
      const bool shifted=delayed(q,rows);const Dot value=dot(weight,x+uint64_t{row}*k,shifted);const uint64_t at=uint64_t{row}*n+weight.column;
      for(auto output:{outQ,outH}) {
        const bool candidate=output==outQ&&shifted;float ref=float(candidate?value.delayed:value.half),original=float(value.half);
        double slack=value.magnitude*0x1p-20,originalSlack=slack;
        if(ep==ops::LinearEpilogue::Residual) {ref+=test::bf16ToFloat(a[at]);original+=test::bf16ToFloat(a[at]);}
        if(ep==ops::LinearEpilogue::UpWithGate) {
          const float multiplier=silu(test::bf16ToFloat(a[at]));
          slack=(slack+test::ulpBf16(ref))*std::fabs(multiplier);
          originalSlack=(originalSlack+test::ulpBf16(original))*std::fabs(multiplier);
          ref=bf(ref)*multiplier;original=bf(original)*multiplier;
        }
        const double actual=test::bf16ToFloat(output[at]),error=std::fabs(actual-bf(ref));
        const double oldError=std::fabs(actual-bf(original));
        require(error<=test::ulpBf16(ref)+slack,"FFN candidate exceeds its independent arithmetic reference");
        const bool pass=oldError<=test::ulpBf16(original)+originalSlack;
        if(!candidate) require(pass,"original half-weight contract failed");
        originalPass&=pass;maxCpu=std::max(maxCpu,error);maxOriginal=std::max(maxOriginal,oldError);maxBudget=std::max(maxBudget,value.weightError);++checked;
      }
    }
    originalContract&=originalPass;++comparisons;
    emit(@{@"status":originalPass?@"pass":@"weight_rounding_difference",@"scope":[NSString stringWithUTF8String:scope],
      @"phase":@"prefill",@"layer":@(layer),@"role":[NSString stringWithUTF8String:role],@"format":[NSString stringWithUTF8String:segment.name()],
      @"rows":@(rows),@"input_columns":@(k),@"physical_input_columns":@(q.planeInputs()?q.planeInputs():k),@"output_columns":@(n),
      @"different_bf16_elements":@(differences),@"max_abs_difference":@(maximum),@"rms_difference":@(std::sqrt(square/(uint64_t{rows}*n))),
      @"cpu_sample_count":@(checked),@"cpu_max_abs_error":@(maxCpu),@"original_half_max_abs_error":@(maxOriginal),
      @"max_weight_rounding_error_budget":@(maxBudget),@"original_half_contract_pass":@(originalPass)});
  }
}

// Actual production GateUp decode plan, followed by the actual residual Down plan.
// Prefill uses addPrefillSwiGlu, so the fused candidate and ANE views are exercised too.
void pipeline(metal::MetalBackend &backend,model::WeightSource &source,const model::GgufFile &file,
    const model::Qwen3_8LayerWeights &layer,uint32_t index,bool decode,bool aneView) {
  const std::string prefix="blk."+std::to_string(index)+".";
  auto gate=aneView?layer.gateProjection.leadingRows(backend,4608):layer.gateProjection;
  auto up=aneView?layer.upProjection.leadingRows(backend,4608):layer.upProjection;
  auto down=aneView?layer.downProjection.leadingInputs(4608):layer.downProjection;
  auto gateH=cached(backend,layer.gateProjection,gate),upH=cached(backend,layer.upProjection,up),downH=cached(backend,layer.downProjection,down);
  const uint32_t rows=decode?model::ExecutionLimits::targetVerifyRows:512,k=gate.inputSize,n=up.outputSize,d=down.outputSize;
  ops::Linear linear(backend.capabilities());
  const auto pairPlan=decode?linear.decodePlan(up,1,ops::LinearEpilogue::GateUp,&gate):linear.prefillPlan(up,rows,ops::LinearEpilogue::UpWithGate);
  const auto downPlan=decode?linear.decodePlan(down,1,ops::LinearEpilogue::Residual):linear.prefillPlan(down,rows,ops::LinearEpilogue::Residual);
  const uint32_t storage=pairPlan.storageRows();auto sizes=pairPlan.scratchSize();sizes.include(downPlan.scratchSize());const auto work=scratch(backend,sizes);
  auto input=allocate(backend,uint64_t{storage}*k*2),residual=allocate(backend,uint64_t{storage}*d*2),gateScratch=allocate(backend,uint64_t{storage}*n*2);
  auto intermediateQ=allocate(backend,uint64_t{storage}*n*2),intermediateH=allocate(backend,uint64_t{storage}*n*2);
  auto outputQ=allocate(backend,uint64_t{storage}*d*2),outputH=allocate(backend,uint64_t{storage}*d*2);fill(input,index+21);fill(residual,index+8);
  auto run=[&](const ops::Projection &g,const ops::Projection &u,const ops::Projection &w,metal::MetalBuffer intermediate,metal::MetalBuffer output) {
    metal::CommandGraph graph;
    if(decode) {
      linear.add(graph,{.input=input,.output=intermediate,.gateScratch=gateScratch,.scratch=work},u,
          linear.decodePlan(u,1,ops::LinearEpilogue::GateUp,&g),&g);
      linear.add(graph,{.input=intermediate,.output=output,.residual=residual,.scratch=work},w,linear.decodePlan(w,1,ops::LinearEpilogue::Residual));
    } else linear.addPrefillSwiGlu(graph,{&g,&u,&w},{input,gateScratch,intermediate,work},residual,output,rows);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  };
  run(gate,up,down,intermediateQ,outputQ);run(gateH,upH,downH,intermediateH,outputH);
  const auto *x=static_cast<const uint16_t *>(input.contents()),*r=static_cast<const uint16_t *>(residual.contents());
  const auto *iq=static_cast<const uint16_t *>(intermediateQ.contents()),*ih=static_cast<const uint16_t *>(intermediateH.contents());
  const auto *oq=static_cast<const uint16_t *>(outputQ.contents()),*oh=static_cast<const uint16_t *>(outputH.contents());
  uint64_t intermediateDiff=0,outputDiff=0;double maxDifference=0;bool originalPass=true;uint64_t samples=0;
  for(uint64_t at=0;at<uint64_t{rows}*n;++at) {require(std::isfinite(test::bf16ToFloat(iq[at]))&&std::isfinite(test::bf16ToFloat(ih[at])),"nonfinite gate/up");intermediateDiff+=iq[at]!=ih[at];}
  for(uint64_t at=0;at<uint64_t{rows}*d;++at) {require(std::isfinite(test::bf16ToFloat(oq[at]))&&std::isfinite(test::bf16ToFloat(oh[at])),"nonfinite down");outputDiff+=oq[at]!=oh[at];maxDifference=std::max(maxDifference,std::fabs(double(test::bf16ToFloat(oq[at]))-test::bf16ToFloat(oh[at])));}
  if(decode) require(intermediateDiff==0&&outputDiff==0,"cache strategy changed real decode outputs");
  const bool gs=!decode&&delayed(gate,rows),us=!decode&&delayed(up,rows),ds=!decode&&delayed(down,rows);
  for(auto column:sampleColumns(n)) {
    const Row gw=readRow(source,file.require(prefix+"ffn_gate.weight"),gate.blocks().segments.front().formatId,column,k);
    const Row uw=readRow(source,file.require(prefix+"ffn_up.weight"),up.blocks().segments.front().formatId,column,k);
    for(auto row:sampleRows(rows)) {
      const Dot g=dot(gw,x+uint64_t{row}*k,gs),u=dot(uw,x+uint64_t{row}*k,us);
      for(auto output:{iq,ih}) {
        const bool candidate=output==iq;const float gv=bf(candidate?g.delayed:g.half),uv=bf(candidate?u.delayed:u.half);
        const float originalG=bf(g.half),originalU=bf(u.half);
        const double eg=g.magnitude*0x1p-20+test::ulpBf16(gv),eu=u.magnitude*0x1p-20+test::ulpBf16(uv);
        const double derivative=1+(std::fabs(gv)+eg)/4;
        const float expected=bf(uv*silu(gv));
        const double bound=test::ulpBf16(expected)+eu*(std::fabs(silu(gv))+eg*derivative)+std::fabs(uv)*eg*derivative;
        const double actual=test::bf16ToFloat(output[uint64_t{row}*n+column]);
        require(std::fabs(actual-expected)<=bound,"gate/up pipeline exceeds independent nonlinear reference");
        const float original=bf(originalU*silu(originalG));
        const double originalBound=test::ulpBf16(original)+eu*(std::fabs(silu(originalG))+eg*derivative)+std::fabs(originalU)*eg*derivative;
        const bool pass=std::fabs(actual-original)<=originalBound;
        if(!candidate||(!gs&&!us)) require(pass,"original gate/up contract failed");
        originalPass&=pass;++samples;
      }
    }
  }
  for(auto column:sampleColumns(d)) {
    const Row dw=readRow(source,file.require(prefix+"ffn_down.weight"),down.blocks().segments.front().formatId,column,n);
    for(auto row:sampleRows(rows)) for(auto path:{0,1}) {
      const bool candidate=path==0;const auto *values=candidate?iq:ih;const auto *out=candidate?oq:oh;
      const Dot v=dot(dw,values+uint64_t{row}*n,ds);const uint64_t at=uint64_t{row}*d+column;
      const float ref=float(candidate?v.delayed:v.half)+test::bf16ToFloat(r[at]);
      const float original=float(v.half)+test::bf16ToFloat(r[at]);const double actual=test::bf16ToFloat(out[at]);
      require(std::fabs(actual-bf(ref))<=test::ulpBf16(ref)+v.magnitude*0x1p-20,"down pipeline exceeds independent reference");
      const bool pass=std::fabs(actual-bf(original))<=test::ulpBf16(original)+v.magnitude*0x1p-20;
      if(!candidate||!ds) require(pass,"original down contract failed");originalPass&=pass;++samples;
    }
  }
  originalContract&=originalPass;++comparisons;
  emit(@{@"status":originalPass?@"pass":@"weight_rounding_difference",@"scope":aneView?@"ane_gpu_views":@"full_ffn",
    @"phase":decode?@"decode":@"prefill_pipeline",@"layer":@(index),@"rows":@(rows),@"lanes":decode?@1:@0,
    @"intermediate_bit_differences":@(intermediateDiff),@"output_bit_differences":@(outputDiff),@"max_abs_difference":@(maxDifference),
    @"cpu_sample_count":@(samples),@"original_half_contract_pass":@(originalPass)});
}
ops::FfnPrefillVariant variant(const std::string &name) {
  if(name=="baseline")return ops::FfnPrefillVariant::Baseline;
  if(name=="m64n32")return ops::FfnPrefillVariant::M64N32;
  if(name=="fused_m64n32")return ops::FfnPrefillVariant::FusedM64N32;
  if(name=="q4k_group_scale")return ops::FfnPrefillVariant::Q4KGroupScale;
  throw std::invalid_argument("invalid FFN numerical candidate");
}
}
int main(int argc,char **argv) {
  @autoreleasepool {
    try {
      const bool regression=argc==5&&std::string(argv[4])=="--rounding-regression";
      require(argc==3||argc==4||regression,"usage: qwen4b-ffn-check METALLIB MODEL_ROOT [baseline|m64n32|fused_m64n32|q4k_group_scale] [--rounding-regression]");
      metal::MetalBackend backend(argv[1]);const auto descriptor=model::inspectModelRoot(argv[2]);
      auto loaded=model::loadModel(backend,argv[2],descriptor,{.mode=model::FfnWeightMode::Quantized,.variant=variant(argc>=4?argv[3]:"baseline")});
      require(loaded.targetFfnCacheAllocatedBytes()==0,"quantized loading retained FFN cache");
      model::WeightSource source(model::findTargetGguf(std::filesystem::path(argv[2])/"target"));const model::GgufFile file(source);
      const std::vector<model::Qwen3_8LayerWeights> *layers=nullptr;
      if(const auto *target=std::get_if<model::Qwen3_5_4BWeights>(&loaded.target))layers=&target->layers;
      if(const auto *target=std::get_if<model::Qwen3_5_4BDFlashWeights>(&loaded.target))layers=&target->layers;
      require(layers&&layers->size()==32,"requires Qwen3.5-4B");std::set<std::pair<uint32_t,uint32_t>> covered;
      if(regression) {
        const auto &gate=layers->front().gateProjection;
        require(gate.blocks().segments.front().formatId==GGUF_FMT_Q4K,"rounding regression requires Q4_K gate");
        check(backend,source,file.require("blk.0.ffn_gate.weight"),gate,gate,ops::LinearEpilogue::None,0,"gate",{64,257},"rounding_regression");
        source.checkUnchanged();
        emit(@{@"status":originalContract?@"complete":@"needs_weight_rounding_analysis",@"comparisons":@(comparisons),
            @"original_half_contract_pass":@(originalContract),@"resident_ffn_cache_bytes":@(loaded.targetFfnCacheAllocatedBytes()),@"rounding_regression":@YES});
        return originalContract?0:2;
      }
      for(uint32_t i=0;i<layers->size();++i) {
        const auto &layer=(*layers)[i];const std::string prefix="blk."+std::to_string(i)+".";
        for(const auto &[p,role,ep]:std::array<std::tuple<const ops::Projection *,const char *,ops::LinearEpilogue>,3>{{
            {&layer.gateProjection,"gate",ops::LinearEpilogue::None},{&layer.upProjection,"up",ops::LinearEpilogue::UpWithGate},{&layer.downProjection,"down",ops::LinearEpilogue::Residual}}})
          check(backend,source,file.require(prefix+"ffn_"+role+".weight"),*p,*p,ep,i,role,{8,33,64,136},"full_projection");
        const auto formats=std::pair{layer.gateProjection.blocks().segments.front().formatId,layer.downProjection.blocks().segments.front().formatId};
        if(!covered.insert(formats).second)continue;
        pipeline(backend,source,file,layer,i,true,false);
        pipeline(backend,source,file,layer,i,false,false);
        pipeline(backend,source,file,layer,i,false,true);
        const auto g=layer.gateProjection.leadingRows(backend,4608),u=layer.upProjection.leadingRows(backend,4608),d=layer.downProjection.leadingInputs(4608);
        check(backend,source,file.require(prefix+"ffn_gate.weight"),layer.gateProjection,g,ops::LinearEpilogue::None,i,"gate",{64,65,127,128,129,255,256,257,512,2048},"ane_leading_rows");
        check(backend,source,file.require(prefix+"ffn_up.weight"),layer.upProjection,u,ops::LinearEpilogue::UpWithGate,i,"up",{512},"ane_leading_rows");
        check(backend,source,file.require(prefix+"ffn_down.weight"),layer.downProjection,d,ops::LinearEpilogue::Residual,i,"down",{64,65,127,128,129,255,256,257,512,2048},"ane_leading_inputs");
      }
      source.checkUnchanged();emit(@{@"status":originalContract?@"complete":@"needs_weight_rounding_analysis",@"matrices":@96,@"comparisons":@(comparisons),
        @"real_decode_covered":@YES,@"ane_gpu_views_covered":@YES,@"original_half_contract_pass":@(originalContract),@"resident_ffn_cache_bytes":@(loaded.targetFfnCacheAllocatedBytes())});
      return originalContract?0:2;
    }catch(const std::exception &error){std::cerr<<"FFN numerical check: "<<error.what()<<'\n';return 1;}
  }
}
