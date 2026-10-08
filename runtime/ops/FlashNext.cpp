#include "ops/FlashNext.hpp"
#include "metal/abi/FlashNext.h"
#include "ops/BufferExtent.hpp"
#include <cmath>
#include <stdexcept>
namespace splash::ops {
namespace {
void shape(uint32_t width, uint32_t streams, uint32_t rows) {
  if (!width || !streams || !rows)
    throw std::invalid_argument("empty Flash Next shape");
}
uint64_t bytes(uint32_t width, uint32_t streams, uint32_t rows) {
  return uint64_t{width} * streams * rows * 2;
}
} // namespace
void FlashNext::addGroupedRms(metal::CommandGraph &g, metal::MetalBuffer input,
                              const NormWeights &w, metal::MetalBuffer output,
                              uint32_t width, uint32_t streams, uint32_t rows,
                              float eps) {
  shape(width, streams, rows);
  if (!std::isfinite(eps) || eps <= 0)
    throw std::invalid_argument("invalid Flash Next epsilon");
  requireBytes(input, bytes(width, streams, rows), "Flash Next RMS input");
  requireBytes(output, bytes(width, streams, rows), "Flash Next RMS output");
  g.add(normKernel("flash_next_group_rms", w, width * streams),
        {input, w.buffer, output}, FlashNextParams{width, streams, rows, eps},
        {uint64_t{rows} * streams, 1, 1});
}
void FlashNext::addHcSilu(metal::CommandGraph &g, metal::MetalBuffer input,
                          metal::MetalBuffer output, uint32_t width,
                          uint32_t streams, uint32_t rows) {
  shape(width, streams, rows);
  requireBytes(input, bytes(width, 1, rows), "HC silu input");
  requireBytes(output, bytes(width, 1, rows), "HC silu output");
  g.add("flash_next_hc_silu", {input, output},
        FlashNextParams{width, streams, rows, 0},
        {(uint64_t{width} * rows + 255) / 256, 1, 1});
}
void FlashNext::addHcMix(metal::CommandGraph &g, metal::MetalBuffer norm,
                         metal::MetalBuffer gate, metal::MetalBuffer output,
                         uint32_t width, uint32_t streams, uint32_t rows) {
  shape(width, streams, rows);
  requireBytes(norm, bytes(width, streams, rows), "HC mix normalized");
  requireBytes(gate, bytes(width, streams, rows), "HC mix gate");
  requireBytes(output, bytes(width, 1, rows), "HC mix output");
  g.add("flash_next_hc_mix", {norm, gate, output},
        FlashNextParams{width, streams, rows, 0},
        {(uint64_t{width} * rows + 255) / 256, 1, 1});
}
void FlashNext::addHcScatter(metal::CommandGraph &g,
                             metal::MetalBuffer residual,
                             metal::MetalBuffer block,
                             metal::MetalBuffer inject,
                             metal::MetalBuffer output, uint32_t width,
                             uint32_t streams, uint32_t rows) {
  shape(width, streams, rows);
  requireBytes(residual, bytes(width, streams, rows), "HC scatter residual");
  requireBytes(block, bytes(width, 1, rows), "HC scatter block");
  requireBytes(inject, bytes(streams, 1, rows), "HC scatter inject");
  requireBytes(output, bytes(width, streams, rows), "HC scatter output");
  g.add("flash_next_hc_scatter", {residual, block, inject, output},
        FlashNextParams{width, streams, rows, 0},
        {(uint64_t{width} * streams * rows + 255) / 256, 1, 1});
}
void FlashNext::addPleGate(metal::CommandGraph &g, metal::MetalBuffer key,
                           metal::MetalBuffer query, metal::MetalBuffer value,
                           metal::MetalBuffer output, uint32_t width,
                           uint32_t streams, uint32_t rows) {
  shape(width, streams, rows);
  requireBytes(key, bytes(width, streams, rows), "PLE key");
  requireBytes(query, bytes(width, streams, rows), "PLE query");
  requireBytes(value, bytes(width, 1, rows), "PLE value");
  requireBytes(output, bytes(width, streams, rows), "PLE gated");
  g.add("flash_next_ple_gate", {key, query, value, output},
        FlashNextParams{width, streams, rows, 0},
        {uint64_t{rows} * streams, 1, 1});
}
void FlashNext::addPleConv(metal::CommandGraph &g, metal::MetalBuffer gated,
                           metal::MetalBuffer norm, metal::MetalBuffer state,
                           metal::MetalBuffer weights,
                           metal::MetalBuffer output, metal::MetalBuffer next,
                           uint32_t width, uint32_t streams, uint32_t rows,
                           uint32_t taps, uint32_t dilation) {
  shape(width, streams, rows);
  if (taps < 2 || !dilation)
    throw std::invalid_argument("invalid PLE convolution");
  const uint64_t history = uint64_t{taps - 1} * dilation;
  requireBytes(gated, bytes(width, streams, rows), "PLE direct");
  requireBytes(norm, bytes(width, streams, rows), "PLE normalized");
  requireBytes(state, history * width * streams * 2, "PLE state");
  requireBytes(next, history * width * streams * 2, "PLE next state");
  requireBytes(weights, uint64_t{taps} * width * streams * 4,
               "PLE conv weights");
  requireBytes(output, bytes(width, streams, rows), "PLE output");
  const FlashNextConvParams p{width, streams, rows, taps, dilation};
  g.add("flash_next_ple_conv", {gated, norm, state, weights, output}, p,
        {(uint64_t{width} * streams * rows + 255) / 256, 1, 1});
  g.add("flash_next_ple_commit", {norm, state, next}, p,
        {(uint64_t{width} * streams + 255) / 256, 1, 1});
}
void FlashNext::addQsaPool(metal::CommandGraph &g, metal::MetalBuffer raw,
                           metal::MetalBuffer pooled, uint32_t width,
                           uint32_t ratio, uint32_t blocks, uint32_t first) {
  if (!width || ratio < 2 || !blocks)
    throw std::invalid_argument("invalid QSA pool");
  const uint64_t end = uint64_t{first} + blocks;
  requireBytes(raw, end * ratio * width * 2, "QSA raw keys");
  requireBytes(pooled, end * width * 2, "QSA pooled keys");
  g.add("flash_next_qsa_pool", {raw, pooled},
        FlashNextPoolParams{width, ratio, blocks, first},
        {(uint64_t{width} * blocks + 255) / 256, 1, 1});
}
void FlashNext::addRope(metal::CommandGraph &g, metal::MetalBuffer input,
                        metal::MetalBuffer cos, metal::MetalBuffer sin,
                        metal::MetalBuffer positions, metal::MetalBuffer output,
                        uint32_t width, uint32_t heads, uint32_t rows,
                        uint32_t rotary, uint32_t tableRows) {
  shape(width, heads, rows);
  if (!rotary || rotary % 2 || rotary > width || !tableRows)
    throw std::invalid_argument("invalid Flash Next RoPE");
  requireBytes(input, bytes(width, heads, rows), "Flash Next RoPE input");
  requireBytes(output, bytes(width, heads, rows), "Flash Next RoPE output");
  requireBytes(cos, uint64_t{tableRows} * (rotary / 2) * 4,
               "Flash Next RoPE cosine");
  requireBytes(sin, uint64_t{tableRows} * (rotary / 2) * 4,
               "Flash Next RoPE sine");
  requireBytes(positions, uint64_t{rows} * 4, "Flash Next RoPE positions");
  g.add("flash_next_rope", {input, cos, sin, positions, output},
        FlashNextRopeParams{width, heads, rows, rotary, tableRows},
        {(uint64_t{width - rotary / 2} * heads * rows + 255) / 256, 1, 1});
}
void FlashNext::addQsaScores(metal::CommandGraph &g, metal::MetalBuffer query,
                             metal::MetalBuffer keys,
                             metal::MetalBuffer positions,
                             metal::MetalBuffer scores, uint32_t width,
                             uint32_t heads, uint32_t rows, uint32_t blocks,
                             uint32_t ratio) {
  shape(width, heads, rows);
  if (ratio < 2)
    throw std::invalid_argument("invalid QSA scores");
  if (!blocks)
    return;
  requireBytes(query, bytes(width, heads, rows), "QSA queries");
  requireBytes(keys, bytes(width, 1, blocks), "QSA block keys");
  requireBytes(positions, uint64_t{rows} * 4, "QSA query positions");
  requireBytes(scores, uint64_t{rows} * blocks * 4, "QSA scores");
  g.add("flash_next_qsa_scores", {query, keys, positions, scores},
        FlashNextQsaParams{width, heads, rows, blocks, ratio, 0},
        {(uint64_t{rows} * blocks + 255) / 256, 1, 1});
}
void FlashNext::addQsaSelect(metal::CommandGraph &g, metal::MetalBuffer scores,
                             metal::MetalBuffer positions,
                             metal::MetalBuffer selected, uint32_t rows,
                             uint32_t blocks, uint32_t ratio, uint32_t budget) {
  if (!rows || ratio < 2 || !budget || budget % ratio)
    throw std::invalid_argument("invalid QSA selection");
  requireBytes(scores, uint64_t{rows} * blocks * 4, "QSA scores");
  requireBytes(positions, uint64_t{rows} * 4, "QSA query positions");
  requireBytes(selected, uint64_t{rows} * (uint64_t{budget} + ratio - 1) * 4,
               "QSA selected indices");
  g.add("flash_next_qsa_select", {scores, positions, selected},
        FlashNextQsaParams{0, 0, rows, blocks, ratio, budget}, {rows, 1, 1},
        {1, 1, 1});
}
void FlashNext::addGdnGate(metal::CommandGraph &g, metal::MetalBuffer recurrent,
                           metal::MetalBuffer gate, const NormWeights &norm,
                           metal::MetalBuffer output, uint32_t width,
                           uint32_t keyHeads, uint32_t valueHeads,
                           uint32_t tokens, bool tiled, uint32_t gateStride) {
  if (!width || !keyHeads || !valueHeads || valueHeads % keyHeads || !tokens)
    throw std::invalid_argument("invalid Flash Next GDN gate");
  if (!gateStride)
    gateStride = width * valueHeads;
  if (gateStride < uint64_t{width} * valueHeads)
    throw std::invalid_argument("short Flash Next GDN gate stride");
  requireBytes(recurrent, bytes(width, valueHeads, tokens),
               "Flash Next GDN recurrent");
  requireBytes(gate,
               rowBytes(tokens, gateStride, uint64_t{width} * valueHeads, 2),
               "Flash Next GDN gate");
  requireBytes(output, bytes(width, valueHeads, tokens),
               "Flash Next GDN hidden");
  g.add(normKernel("flash_next_gdn_gate", norm, width),
        {recurrent, gate, norm.buffer, output},
        FlashNextGdnParams{width, keyHeads, valueHeads, tokens, gateStride,
                           tiled ? 1u : 0u},
        {uint64_t{tokens} * valueHeads, 1, 1});
}
void FlashNext::addQsaAttention(
    metal::CommandGraph &g, metal::MetalBuffer query, metal::MetalBuffer gate,
    metal::MetalBuffer table, metal::MetalBuffer selected,
    metal::MetalBuffer output, SplashKvLayer layer, uint32_t heads,
    uint32_t rows, uint32_t stride, uint32_t entries, kv::Format format) {
  if (!heads || heads % 2 || !rows || !stride || !entries ||
      !layer.extent_pages || !kv::validFormat(format))
    throw std::invalid_argument("invalid Flash Next selected attention");
  requireBytes(query, bytes(256, heads, rows), "QSA attention query");
  requireBytes(gate, bytes(256, heads, rows), "QSA attention gate");
  requireBytes(output, bytes(256, heads, rows), "QSA attention output");
  requireBytes(selected, uint64_t{rows} * stride * 4, "QSA attention selected");
  requireBytes(table, uint64_t{entries} * sizeof(SplashKvPage),
               "QSA attention page table");
  g.add(format == kv::Format::Int8 ? "flash_next_qsa_attention_int8"
                                   : "flash_next_qsa_attention_bf16",
        {query, gate, table, selected, output},
        FlashNextAttentionParams{layer, heads, rows, stride, entries},
        {heads, rows, 1});
}
void FlashNext::addBroadcast(metal::CommandGraph &g, metal::MetalBuffer input,
                             metal::MetalBuffer output, uint32_t width,
                             uint32_t streams, uint32_t rows) {
  shape(width, streams, rows);
  requireBytes(input, bytes(width, 1, rows), "Flash Next broadcast input");
  requireBytes(output, bytes(width, streams, rows),
               "Flash Next broadcast output");
  g.add("flash_next_broadcast", {input, output},
        FlashNextParams{width, streams, rows, 0},
        {(uint64_t{width} * streams * rows + 255) / 256, 1, 1});
}
namespace {
void binary(metal::CommandGraph &g, const char *name, metal::MetalBuffer a,
            metal::MetalBuffer b, metal::MetalBuffer output, uint32_t width,
            uint32_t rows) {
  shape(width, 1, rows);
  requireBytes(a, bytes(width, 1, rows), "Flash Next binary input a");
  requireBytes(b, bytes(width, 1, rows), "Flash Next binary input b");
  requireBytes(output, bytes(width, 1, rows), "Flash Next binary output");
  g.add(name, {a, b, output}, FlashNextParams{width, 1, rows, 0},
        {(uint64_t{width} * rows + 255) / 256, 1, 1});
}
} // namespace
void FlashNext::addAdd(metal::CommandGraph &g, metal::MetalBuffer a,
                       metal::MetalBuffer b, metal::MetalBuffer output,
                       uint32_t width, uint32_t rows) {
  binary(g, "flash_next_add", a, b, output, width, rows);
}
void FlashNext::addSiluMul(metal::CommandGraph &g, metal::MetalBuffer a,
                           metal::MetalBuffer b, metal::MetalBuffer output,
                           uint32_t width, uint32_t rows) {
  binary(g, "flash_next_silu_mul", a, b, output, width, rows);
}
void FlashNext::addSigmoidMul(metal::CommandGraph &g, metal::MetalBuffer a,
                              metal::MetalBuffer gate,
                              metal::MetalBuffer output, uint32_t width,
                              uint32_t rows) {
  binary(g, "flash_next_sigmoid_mul", a, gate, output, width, rows);
}
void FlashNext::addScaledAccum(metal::CommandGraph &g, metal::MetalBuffer input,
                               metal::MetalBuffer acc, uint32_t width,
                               uint32_t rows, float scale) {
  shape(width, 1, rows);
  if (!std::isfinite(scale))
    throw std::invalid_argument("non-finite Flash Next expert coefficient");
  requireBytes(input, bytes(width, 1, rows), "Flash Next accumulate input");
  requireBytes(acc, uint64_t{width} * rows * 4, "Flash Next accumulate output");
  g.add("flash_next_scaled_accum", {input, acc},
        FlashNextParams{width, 1, rows, scale},
        {(uint64_t{width} * rows + 255) / 256, 1, 1});
}
void FlashNext::addFloatToBf16(metal::CommandGraph &g, metal::MetalBuffer input,
                               metal::MetalBuffer output, uint32_t width,
                               uint32_t rows) {
  shape(width, 1, rows);
  requireBytes(input, uint64_t{width} * rows * 4, "Flash Next fp32 input");
  requireBytes(output, bytes(width, 1, rows), "Flash Next bf16 output");
  g.add("flash_next_float_to_bf16", {input, output},
        FlashNextParams{width, 1, rows, 0},
        {(uint64_t{width} * rows + 255) / 256, 1, 1});
}
} // namespace splash::ops
