#include "ops/D3.hpp"
#include "ops/BufferExtent.hpp"
#include <stdexcept>

namespace splash::ops {
void D3::attention(metal::CommandGraph &g, const metal::MetalBuffer &q,
                    const metal::MetalBuffer &k, const metal::MetalBuffer &v,
                    const metal::MetalBuffer &out, D3AttentionParams p,const metal::MetalBuffer &scales) {
  if (!p.tokens || p.stride < storageRows(p.tokens) || p.stride % 128 ||
      !p.query_rows || p.query_rows > p.tokens || p.query_start > p.tokens - p.query_rows ||
      p.kv_heads == 0 || p.query_heads != 4 * p.kv_heads || (p.kv_heads != 2 && p.kv_heads != 4))
    throw std::invalid_argument("invalid D3 noncausal attention geometry");
  for (const auto &b : {q, out}) requireBytes(b, uint64_t{p.stride} * p.query_heads * 256 * 2, "D3 Q/output");
  for (const auto &b : {k, v}) requireBytes(b, uint64_t{p.stride} * p.kv_heads * 256 * 2, "D3 K/V");
  if(p.q4h)requireBytes(scales,uint64_t{p.stride}*p.kv_heads*2*4,"D3 Q4H attention scales");
  g.add("d3_attention", {q,k,v,out,p.q4h?scales:q}, p, {(p.query_rows+63)/64,p.query_heads,1});
}
void D3::zero(metal::CommandGraph &g, const metal::MetalBuffer &b) {
  if (b.sizeBytes() % 4 || b.sizeBytes() / 4 > UINT32_MAX) throw std::invalid_argument("invalid D3 zero buffer");
  uint32_t words = b.sizeBytes() / 4;
  if (words) g.add("d3_zero", {b}, words, {(words+255ull)/256,1,1});
}
void D3::transposeValues(metal::CommandGraph &g,const metal::MetalBuffer &in,
                          const metal::MetalBuffer &out,D3AttentionParams p) {
  if (!p.stride || p.stride%128 || (p.kv_heads!=2 && p.kv_heads!=4))
    throw std::invalid_argument("invalid D3 value transpose geometry");
  uint64_t elements=uint64_t{p.stride}*p.kv_heads*256;
  requireBytes(in,elements*2,"D3 source values"); requireBytes(out,elements*2,"D3 transposed values");
  g.add("d3_values",{in,out},p,{(elements+255)/256,1,1});
}
void D3::quantizeKv(metal::CommandGraph &g,const metal::MetalBuffer &keys,
                     const metal::MetalBuffer &values,const metal::MetalBuffer &decoded,
                     const metal::MetalBuffer &codes,const metal::MetalBuffer &scales,D3AttentionParams p) {
  if (!p.tokens || p.tokens>p.stride || (p.kv_heads!=2 && p.kv_heads!=4) || p.q4h!=1)
    throw std::invalid_argument("invalid D3 Q4H geometry");
  uint64_t rows=uint64_t{p.stride}*p.kv_heads;
  for (const auto &b:{keys,values,decoded}) requireBytes(b,rows*256*2,"D3 Q4H staging");
  requireBytes(codes,rows*2*128,"D3 Q4H codes");requireBytes(scales,rows*2*4,"D3 Q4H scales");
  g.add("d3_quantize_kv",{keys,values,codes,scales},p,{uint64_t{p.tokens}*p.kv_heads*2,1,1});
  g.add("d3_decode_kv",{codes,keys,decoded},p,{uint64_t{p.tokens}*p.kv_heads*2,1,1});
}
}
