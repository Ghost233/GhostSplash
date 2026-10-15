#include "ops/ModernBert.hpp"
#include "ops/BufferExtent.hpp"
#include "metal/abi/GgufRepack.h"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace splash::ops {
namespace {
using Buffers = std::vector<metal::MetalBuffer>;
void count(const Buffers &b, size_t n) {
  if (b.size() != n) throw std::invalid_argument("invalid encoder buffer count");
}
void rows(ModernRowsParams p) {
  if (!p.rows || !p.width || p.width > p.stride) throw std::invalid_argument("invalid encoder row geometry");
}
}
void ModernBert::embedding(metal::CommandGraph &g, Buffers b, ModernRowsParams p) {
  count(b,3); rows(p);
  if (p.width % 32 || !p.kind) throw std::invalid_argument("invalid Q8 encoder embedding");
  requireBytes(b[0],uint64_t{p.rows}*4,"encoder input tokens");
  requireBytes(b[1],uint64_t{p.kind}*(p.width/32)*34,"encoder token table");
  requireBytes(b[2],uint64_t{p.rows}*p.stride*4,"encoder hidden rows");
  g.add("modern_embed_q8",std::move(b),p,{p.rows,1,1});
}
void ModernBert::normalization(metal::CommandGraph &g, Buffers b, ModernNormParams p) {
  count(b,6);
  if (!p.rows || !p.width || p.width>p.stride || !(p.epsilon>0) || !std::isfinite(p.epsilon) || p.add_delta>1 || p.normalize_hidden>1)
    throw std::invalid_argument("invalid encoder normalization geometry");
  requireBytes(b[0],uint64_t{p.rows}*p.stride*4,"encoder residual stream");
  if (p.add_delta) {
    if (p.delta_stride<p.width) throw std::invalid_argument("invalid encoder residual stride");
    requireBytes(b[1],uint64_t{p.rows}*p.delta_stride*4,"encoder residual delta");
    requireBytes(b[2],uint64_t{p.width}*4,"encoder residual bias");
  }
  requireBytes(b[3],uint64_t{p.width}*4,"encoder norm weight");
  requireBytes(b[4],uint64_t{p.width}*4,"encoder norm bias");
  requireBytes(b[5],uint64_t{p.rows}*p.stride*2,"encoder normalized rows");
  g.add("modern_norm",std::move(b),p,{p.rows,1,1});
}
void ModernBert::typeEmbedding(metal::CommandGraph &g, Buffers b, ModernRowsParams p) {
  count(b,2); rows(p); if (p.kind>2) throw std::invalid_argument("invalid encoder question type");
  requireBytes(b[0],uint64_t{p.rows}*p.stride*4,"encoder typed rows");
  requireBytes(b[1],uint64_t{3}*p.width*4,"encoder type embeddings");
  g.add("modern_type_add",std::move(b),p,{(uint64_t{p.rows}*p.stride+255)/256,1,1});
}
void ModernBert::prepareQkv(metal::CommandGraph &g, Buffers b, ModernQkvParams p) {
  count(b,6);
  if (!p.tokens || !p.heads || p.width!=p.heads*64 || p.source_stride<3*p.width ||
      p.padded_tokens!=keyRows(p.tokens) || !p.query_rows || p.query_rows>p.tokens ||
      p.padded_queries!=queryRows(p.query_rows) || p.selected>1 || !(p.theta>=0) || !std::isfinite(p.theta))
    throw std::invalid_argument("invalid encoder QKV geometry");
  requireBytes(b[0],uint64_t{p.tokens}*p.source_stride*4,"encoder projected QKV");
  requireBytes(b[1],uint64_t{3}*p.width*4,"encoder QKV bias");
  if (p.selected) requireBytes(b[2],uint64_t{p.query_rows}*4,"encoder query positions");
  requireBytes(b[3],uint64_t{p.heads}*p.padded_queries*64*2,"encoder queries");
  for (size_t i : {4U,5U}) requireBytes(b[i],uint64_t{p.heads}*p.padded_tokens*64*2,"encoder keys/values");
  g.add("modern_qkv_prepare",std::move(b),p,{p.padded_tokens,p.heads,1},{64,1,1});
}
void ModernBert::attention(metal::CommandGraph &g, Buffers b, ModernAttentionParams p) {
  count(b,4);
  if (!p.tokens || !p.heads || p.padded_tokens!=keyRows(p.tokens) || !p.query_rows || p.query_rows>p.tokens ||
      p.padded_queries<queryRows(p.query_rows) || p.padded_queries>p.padded_tokens || p.window%2)
    throw std::invalid_argument("invalid encoder attention geometry");
  for (size_t i : {0U,3U}) requireBytes(b[i],uint64_t{p.heads}*p.padded_queries*64*2,"encoder attention Q/output");
  for (size_t i : {1U,2U}) requireBytes(b[i],uint64_t{p.heads}*p.padded_tokens*64*2,"encoder attention K/V");
  g.add("modern_attention",std::move(b),p,{(p.query_rows+63)/64,p.heads,1});
}
void ModernBert::gatherAttention(metal::CommandGraph &g, Buffers b, ModernRowsParams p) {
  count(b,2); rows(p);
  if (p.width%64 || p.kind<p.rows) throw std::invalid_argument("invalid encoder attention gather");
  requireBytes(b[0],uint64_t{p.kind}*p.width*2,"encoder head-major rows");
  requireBytes(b[1],uint64_t{p.rows}*p.stride*2,"encoder gathered rows");
  g.add("modern_attention_gather",std::move(b),p,{(uint64_t{p.rows}*p.stride+255)/256,1,1});
}
void ModernBert::activation(metal::CommandGraph &g, Buffers b, ModernActivationParams p) {
  count(b,3);
  if (!p.rows || !p.width || p.width>p.destination_stride || p.head>1 || p.source_stride<p.width*(p.head?1:2))
    throw std::invalid_argument("invalid encoder FFN activation geometry");
  requireBytes(b[0],uint64_t{p.rows}*p.source_stride*4,"encoder FFN projection");
  requireBytes(b[1],uint64_t{p.width}*4,"encoder FFN bias");
  requireBytes(b[2],uint64_t{p.rows}*p.destination_stride*2,"encoder FFN activation");
  g.add("modern_ffn_activate",std::move(b),p,{(uint64_t{p.rows}*p.destination_stride+255)/256,1,1});
}
namespace {
void selectRows(metal::CommandGraph &g, Buffers b, ModernRowsParams p, bool floating) {
  count(b,3); rows(p);
  if (!p.kind || p.rows>p.kind) throw std::invalid_argument("invalid encoder selection geometry");
  uint32_t bytes=floating?4:2;
  requireBytes(b[0],uint64_t{p.kind}*p.stride*bytes,"encoder source rows");
  requireBytes(b[1],uint64_t{p.rows}*4,"encoder selection positions");
  requireBytes(b[2],uint64_t{p.rows}*p.stride*bytes,"encoder selected rows");
  g.add(floating?"modern_select_float":"modern_select_bfloat",std::move(b),p,{(uint64_t{p.rows}*p.stride+255)/256,1,1});
}
}
void ModernBert::selectFloat(metal::CommandGraph &g, Buffers b, ModernRowsParams p) { selectRows(g,std::move(b),p,true); }
void ModernBert::selectBfloat(metal::CommandGraph &g, Buffers b, ModernRowsParams p) { selectRows(g,std::move(b),p,false); }
void ModernBert::classifier(metal::CommandGraph &g, Buffers b, ModernRowsParams p) {
  count(b,5); rows(p);
  requireBytes(b[0],uint64_t{p.rows}*p.stride*4,"encoder classifier input");
  requireBytes(b[1],uint64_t{p.width}*4,"encoder classifier bias");
  requireBytes(b[2],uint64_t{p.width}*2,"encoder classifier weight");
  requireBytes(b[3],4,"encoder classifier output bias");
  requireBytes(b[4],uint64_t{p.rows}*4,"encoder classifier scores");
  g.add("modern_classifier",std::move(b),p,{p.rows,1,1});
}
void ModernBert::repackQ8(metal::MetalBackend &backend, const metal::MetalBuffer &native,
                         const metal::MetalBuffer &planes, uint32_t n, uint32_t k) {
  if (!n || n%256 || !k || k%256) throw std::invalid_argument("invalid encoder Q8 repack geometry");
  const uint64_t plane0=uint64_t{n}*k, meta=uint64_t{n}*(k/32)*2;
  requireBytes(native,uint64_t{n}*(k/32)*34,"encoder native Q8 rows");
  requireBytes(planes,plane0+meta,"encoder Q8 planes");
  if (plane0>UINT32_MAX) throw std::invalid_argument("encoder Q8 tensor exceeds repack range");
  GgufRepackParams p{n,k,GGUF_FMT_Q80,k/32*34,0,static_cast<uint32_t>(plane0)};
  static_cast<void>(backend.submit({"gguf_repack",{{0,native},{1,planes}},{{2,&p,sizeof(p)}},
                                   {uint64_t{n}*(k/32)/256,1,1},{256,1,1}}));
}
} // namespace splash::ops
