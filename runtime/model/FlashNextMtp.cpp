#include "model/FlashNextMtp.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include <algorithm>
#include <cstring>
#include <tuple>
#include <vector>
namespace splash::model {
namespace {
float widen(uint16_t value) {
  uint32_t bits=uint32_t{value}<<16;float result;std::memcpy(&result,&bits,4);return result;
}
metal::MetalBuffer read(metal::MetalBackend &backend,const SourceTensor &tensor,
                       const std::vector<uint64_t> &shape,bool norm=false,
                       uint64_t first=0,uint64_t elements=0,uint64_t padded=0) {
  if(tensor.dtype!="BF16" || tensor.shape!=shape)throw std::invalid_argument("incompatible BF16 Flash Next MTP tensor");
  if(!elements)elements=tensor.bytes/2;
  if(first>tensor.bytes/2 || elements>tensor.bytes/2-first)throw std::invalid_argument("MTP tensor range exceeds checkpoint");
  auto buffer=backend.allocateBuffer(std::max(elements,padded)*4,metal::BufferStorage::Shared,"flash-mtp-weight");
  std::memset(buffer.contents(),0,buffer.sizeBytes());
  std::vector<uint16_t> stage(1u<<20);
  auto *out=static_cast<float*>(buffer.contents());
  for(uint64_t at=0;at<elements;at+=stage.size()) {
    const auto count=std::min<uint64_t>(stage.size(),elements-at);
    tensor.read((first+at)*2,{reinterpret_cast<uint8_t*>(stage.data()),static_cast<size_t>(count*2)});
    for(uint64_t i=0;i<count;++i)out[at+i]=norm?widen(stage[i])+1.f:widen(stage[i]);
  }
  return buffer;
}
}
void loadFlashNextMtpWeights(metal::MetalBackend &backend,FlashNextWeights &weights,
                            const std::filesystem::path &directory) {
  SafetensorsCheckpoint checkpoint(directory);
  const auto projection=[&](const char *source,const std::string &name,uint32_t rows,uint32_t columns,
                            const std::vector<uint64_t> &shape={},uint64_t first=0) {
    const auto &tensor=checkpoint.require(source);const uint32_t physical=(rows+7)/8*8;
    auto buffer=read(backend,tensor,shape.empty()?std::vector<uint64_t>{rows,columns}:shape,false,first,uint64_t{rows}*columns,uint64_t{physical}*columns);
    weights.actualAllocatedBytes+=buffer.sizeBytes();
    weights.mtpShapes.emplace(name,std::pair{rows,columns});
    weights.projections.emplace(name,ops::Projection{physical,columns,ops::BlockWeights{{ops::QuantizedSegment::floats(physical,columns,buffer)}}});
  };
  const auto vector=[&](const char *source,const std::string &name,uint32_t width,bool norm=true,
                       bool matrix=false) {
    auto buffer=read(backend,checkpoint.require(source),matrix?std::vector<uint64_t>{1,width}:std::vector<uint64_t>{width},norm);
    weights.actualAllocatedBytes+=buffer.sizeBytes();weights.tensors.emplace(name,buffer);
  };
  projection("fc_embedding.weight","mtp.fc_embedding.weight",2560,2560);
  projection("fc_hidden.weight","mtp.fc_hidden.weight",2560,2560);
  vector("pre_fc_norm_embedding.weight","mtp.embedding_norm.weight",2560);
  vector("pre_fc_norm_hidden.weight","mtp.hidden_norm.weight",10240);
  const auto hyper=[&](const std::string &source,const std::string &target,bool inject) {
    vector((source+"hc_norm.weight").c_str(),target+"norm.weight",10240);
    projection((source+"input_mix_weight_down.weight").c_str(),target+"down.weight",320,10240);
    projection((source+"input_mix_weight_up.weight").c_str(),target+"up.weight",10240,320);
    if(inject)projection((source+"block_inject_weight.weight").c_str(),target+"inject.weight",4,10240);
  };
  hyper("hyper_connection_mixer.","mtp.output_hc_",false);
  hyper("layers.0.attn_hyper_connection.","mtp.blk.0.hc_attn_",true);
  hyper("layers.0.mlp_hyper_connection.","mtp.blk.0.hc_ffn_",true);
  const std::string source="layers.0.self_attn.",target="mtp.blk.0.";
  projection((source+"q_proj.weight").c_str(),target+"attn_q.weight",12288,2560);
  projection((source+"k_proj.weight").c_str(),target+"attn_k.weight",512,2560);
  projection((source+"v_proj.weight").c_str(),target+"attn_v.weight",512,2560);
  projection((source+"o_proj.weight").c_str(),target+"attn_output.weight",2560,6144);
  vector((source+"q_norm.weight").c_str(),target+"attn_q_norm.weight",256);
  vector((source+"k_norm.weight").c_str(),target+"attn_k_norm.weight",256);
  projection((source+"indexer.index_qk_proj.weight").c_str(),target+"indexer.q_proj.weight",512,2560,{640,2560});
  projection((source+"indexer.index_qk_proj.weight").c_str(),target+"indexer.k_proj.weight",128,2560,{640,2560},512ull*2560);
  vector((source+"indexer.q_layernorm.weight").c_str(),target+"indexer.q_norm.weight",128);
  vector((source+"indexer.k_layernorm.weight").c_str(),target+"indexer.k_norm.weight",128);
  projection("layers.0.mlp.gate.weight",target+"ffn_gate_inp.weight",512,2560);
  projection("layers.0.mlp.shared_expert.gate_proj.weight",target+"ffn_gate_shexp.weight",640,2560);
  projection("layers.0.mlp.shared_expert.up_proj.weight",target+"ffn_up_shexp.weight",640,2560);
  projection("layers.0.mlp.shared_expert.down_proj.weight",target+"ffn_down_shexp.weight",2560,640);
  vector("layers.0.mlp.shared_expert_gate.weight",target+"ffn_gate_inp_shexp.weight",2560,false,true);
  for(const auto &[role,out,in] : std::vector<std::tuple<const char*,uint32_t,uint32_t>>{
      {"gate",640,2560},{"up",640,2560},{"down",2560,640}}) {
    const auto &tensor=checkpoint.require(std::string("layers.0.mlp.switch_mlp.")+role+"_proj.weight");
    auto buffer=read(backend,tensor,{512,out,in});weights.actualAllocatedBytes+=buffer.sizeBytes();
    for(uint32_t expert=0;expert<512;++expert) {
      auto view=backend.view(buffer,uint64_t{expert}*out*in*4,uint64_t{out}*in*4);
      const auto name=target+"ffn_"+role+"_exps.weight."+std::to_string(expert);
      weights.mtpShapes.emplace(name,std::pair{out,in});
      weights.projections.emplace(name,ops::Projection{out,in,ops::BlockWeights{{ops::QuantizedSegment::floats(out,in,view)}}});
    }
  }
  checkpoint.checkUnchanged();weights.hasMtp=true;
}
}
