// One-row GGUF GEMV: every quant format, FP64 products over the original
// GGML rows, nonzero residual/gate epilogues, FP32 logits, mixed F32
// segments, MDGG0001 tile boundaries and poisoned reusable K partials.
#include "GgufFormatReference.hpp"
#include "TestBuffers.hpp"
#include "ops/SingleTokenLinear.hpp"
#include "ops/GDN.hpp"
#include "metal/abi/GDN.h"
#include "LinearNumerics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace splash;
using namespace splash::ops;
using namespace gguf_reference;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::test::bf16ToFloat;
using splash::test::floatToBf16;

namespace {

constexpr uint64_t kGuard = 128;
constexpr uint8_t kGuardByte = 0xA5;
constexpr uint32_t kPadding = 64;

struct Guarded final {
  MetalBuffer backing, view;
  uint64_t bytes;
  Guarded(MetalBackend &backend, uint64_t size) : bytes(size) {
    backing = test::sharedBuffer(backend, bytes + 2 * kGuard);
    std::memset(backing.contents(), kGuardByte, bytes + 2 * kGuard);
    view = backend.view(backing, kGuard, bytes);
  }
  void requireIntact() const {
    const auto *data = static_cast<const uint8_t *>(backing.contents());
    test::require(std::all_of(data, data + kGuard, [](uint8_t b) { return b == kGuardByte; }) &&
                      std::all_of(data + kGuard + bytes, data + 2 * kGuard + bytes,
                                  [](uint8_t b) { return b == kGuardByte; }),
                  "single-token buffer guard overwritten");
  }
};

MetalBuffer upload(MetalBackend &backend, const std::vector<uint8_t> &bytes) {
  auto buffer = test::sharedBuffer(backend, bytes.size());
  std::memcpy(buffer.contents(), bytes.data(), bytes.size());
  return buffer;
}

struct Weights final {
  Projection projection;
  std::vector<Dot> dots;
  uint32_t covered;
};

Weights weights(MetalBackend &backend, Fmt format, const std::vector<float> &input, uint32_t columns,
                 bool mixed, std::mt19937 &rng) {
  const uint32_t inputs = static_cast<uint32_t>(input.size());
  const auto native = makeNative(format, columns, inputs, rng);
  const auto packed = repack(format, native, columns, inputs, nullptr);
  BlockWeights blocks;
  blocks.segments.push_back(QuantizedSegment::planes(format, columns, inputs, upload(backend, packed.w0),
                            kQuantFormats[format].plane1_bytes ? upload(backend, packed.w1) : MetalBuffer{},
                            upload(backend, packed.meta)));
  std::vector<Dot> dots;
  std::vector<float> row(inputs);
  for (uint32_t column = 0; column < columns; ++column) {
    rowValues(format, native.data() + uint64_t{column} * rowBytes(format, inputs), inputs, row.data());
    dots.push_back(dot(input.data(), row.data(), inputs));
  }
  uint32_t covered = columns;
  if (mixed) {
    constexpr uint32_t floats = 32;
    auto buffer = test::sharedBuffer(backend, uint64_t{floats} * inputs * sizeof(float));
    auto *data = static_cast<float *>(buffer.contents());
    for (uint32_t column = 0; column < floats; ++column) {
      for (uint32_t k = 0; k < inputs; ++k)
        data[uint64_t{column} * inputs + k] = float(int((column * 11 + k * 7) % 29) - 14) * 0.00317f;
      dots.push_back(dot(input.data(), data + uint64_t{column} * inputs, inputs));
    }
    blocks.segments.push_back(QuantizedSegment::floats(floats, inputs, buffer));
    blocks.segments.back().columnOffset = columns;
    covered += floats;
  }
  return {Projection(covered + kPadding, inputs, std::move(blocks)), std::move(dots), covered};
}

struct Interval final { double lo, hi; };
Interval product(const Dot &d, bool quantized) {
  const double error = projectionBound(d, quantized);
  return {d.value - error, d.value + error};
}
float bf16(double value) { return bf16ToFloat(floatToBf16(static_cast<float>(value))); }
Interval epilogueInterval(Interval interval, float auxiliary, LinearEpilogue epilogue) {
  if (epilogue == LinearEpilogue::Residual)
    return {float(float(interval.lo) + auxiliary), float(float(interval.hi) + auxiliary)};
  if (epilogue == LinearEpilogue::UpWithGate) {
    const double silu = auxiliary / (1 + std::exp(-double(auxiliary)));
    const double t = std::fabs(auxiliary) * 1.4426950408889634;
    const double error = std::fabs(silu) *
        (2 * (3 + std::floor(2 * t)) + 2 * std::log(2.0) * t + 6) * 0x1p-24;
    double lo = INFINITY, hi = -INFINITY;
    for (const double up : {double(bf16(interval.lo)), double(bf16(interval.hi))})
      for (const double gate : {silu - error, silu + error}) {
        lo = std::min(lo, up * gate);
        hi = std::max(hi, up * gate);
      }
    return {lo, hi};
  }
  return interval;
}

void checkCase(MetalBackend &backend, Fmt format, uint32_t inputs, bool sparse, bool mixed,
                uint32_t columns, std::mt19937 &rng, uint32_t tile = 32) {
  std::vector<float> x(inputs);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  for (uint32_t k = 0; k < inputs; ++k)
    x[k] = bf16(sparse ? (k % 256 == 97 ? ((k / 256) & 1 ? -100000.0 : 100000.0) : 0.0) : unit(rng));
  Weights w = weights(backend, format, x, columns, mixed, rng);
  const uint32_t stride = w.projection.outputSize;
  Guarded input(backend, uint64_t{inputs} * 2), aux(backend, uint64_t{stride} * 2);
  auto *inputData = static_cast<uint16_t *>(input.view.contents());
  auto *auxData = static_cast<uint16_t *>(aux.view.contents());
  for (uint32_t k = 0; k < inputs; ++k) inputData[k] = floatToBf16(x[k]);
  for (uint32_t column = 0; column < stride; ++column)
    auxData[column] = floatToBf16(float(int(column % 13) - 6) * 0.375f + 0.1875f);
  Guarded partials(backend, singleTokenLinearScratchBytes(w.projection, 8));
  for (const uint32_t splits : {1u, 8u, 2u, 4u}) {
    std::fill_n(static_cast<uint32_t *>(partials.view.contents()), partials.bytes / 4, 0x7FC00000u);
    std::vector<uint16_t> plain;
    for (const LinearEpilogue epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate}) {
      Guarded output(backend, uint64_t{stride} * 2);
      std::memset(output.view.contents(), 0xFF, output.bytes);
      metal::CommandGraph graph;
      addSingleTokenLinear(graph, input.view, w.projection, output.view, aux.view, epilogue, partials.view, splits, tile);
      static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
      const auto *out = static_cast<const uint16_t *>(output.view.contents());
      for (uint32_t column = 0; column < w.covered; ++column) {
        const Interval expected = epilogueInterval(product(w.dots[column], column < columns),
                                                   bf16ToFloat(auxData[column]), epilogue);
        const float value = bf16ToFloat(out[column]);
        test::require(std::isfinite(value) && value >= bf16(expected.lo) && value <= bf16(expected.hi),
                      std::string(fmtName(format)) + " single-token output outside FP64 bound, column " +
                          std::to_string(column) + ", split " + std::to_string(splits));
      }
      for (uint32_t column = w.covered; column < stride; ++column)
        test::require(out[column] == 0xFFFF, "single-token projection wrote padding columns");
      if (epilogue == LinearEpilogue::None) plain.assign(out, out + stride);
      output.requireIntact();
    }
    Guarded logits(backend, uint64_t{stride} * sizeof(float));
    auto *values = static_cast<float *>(logits.view.contents());
    std::fill_n(values, stride, -12345.0f);
    w.projection.destination = FloatOutput::Float32;
    metal::CommandGraph graph;
    addSingleTokenLinear(graph, input.view, w.projection, logits.view, {}, LinearEpilogue::None,
                          partials.view, splits, tile);
    static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
    for (uint32_t column = 0; column < w.covered; ++column) {
      const Interval expected = product(w.dots[column], column < columns);
      test::require(std::isfinite(values[column]) && values[column] >= expected.lo && values[column] <= expected.hi,
                    "single-token FP32 logits outside unrounded FP64 bound");
      test::require(floatToBf16(values[column]) == plain[column],
                    "single-token FP32 logits do not round to the BF16 plain output");
    }
    for (uint32_t column = w.covered; column < stride; ++column)
      test::require(values[column] == -12345.0f, "single-token FP32 logits wrote padding columns");
    logits.requireIntact();
    w.projection.destination = FloatOutput::BFloat16;
    input.requireIntact(); aux.requireIntact(); partials.requireIntact();
  }
}

// One command reuses the same split scratch through a dependent gate/up/
// down chain. Different strides and partition counts must not leave stale
// partials or let a later projection overwrite an unfinished reduction.
void splitScratchChain(MetalBackend &backend, std::mt19937 &rng, uint32_t tile = 32) {
  constexpr uint32_t inputs = 2560, intermediate = 512;
  std::vector<float> x(inputs);
  for (uint32_t k = 0; k < inputs; ++k) x[k] = bf16(float(int(k % 23) - 11) * 0.071f);
  const auto gate = weights(backend, Q4K, x, intermediate, false, rng);
  const auto up = weights(backend, Q5K, x, intermediate, false, rng);
  const auto down = weights(backend, Q6K, std::vector<float>(intermediate, 1.0f), 256, false, rng);
  Guarded input(backend, inputs * 2), residual(backend, uint64_t{down.projection.outputSize} * 2);
  for (uint32_t k = 0; k < inputs; ++k)
    static_cast<uint16_t *>(input.view.contents())[k] = floatToBf16(x[k]);
  for (uint32_t n = 0; n < down.projection.outputSize; ++n)
    static_cast<uint16_t *>(residual.view.contents())[n] = floatToBf16(float(int(n % 17) - 8) * 0.37f);
  struct Chain final {
    Guarded gate, up, down;
    Chain(MetalBackend &device, uint32_t middleStride, uint32_t outputStride)
        : gate(device, uint64_t{middleStride} * 2), up(device, uint64_t{middleStride} * 2),
          down(device, uint64_t{outputStride} * 2) {
      for (const auto *buffer : {&gate, &up, &down}) std::memset(buffer->view.contents(), 0xFF, buffer->bytes);
    }
  };
  Chain separate(backend, gate.projection.outputSize, down.projection.outputSize);
  Chain combined(backend, gate.projection.outputSize, down.projection.outputSize);
  const uint64_t scratchBytes = std::max({singleTokenLinearScratchBytes(gate.projection, 8),
      singleTokenLinearScratchBytes(up.projection, 2), singleTokenLinearScratchBytes(down.projection, 4)});
  Guarded partials(backend, scratchBytes);
  const auto poison = [&] {
    std::fill_n(static_cast<uint32_t *>(partials.view.contents()), partials.bytes / 4, 0x7FC00000u);
  };
  const auto encode = [&](metal::CommandGraph &graph, Chain &output, uint32_t stage) {
    if (stage == 0)
      addSingleTokenLinear(graph, input.view, gate.projection, output.gate.view, {}, LinearEpilogue::None,
                            partials.view, 8, tile);
    else if (stage == 1)
      addSingleTokenLinear(graph, input.view, up.projection, output.up.view, output.gate.view,
                            LinearEpilogue::UpWithGate, partials.view, 2, tile);
    else
      addSingleTokenLinear(graph, output.up.view, down.projection, output.down.view, residual.view,
                            LinearEpilogue::Residual, partials.view, 4, tile);
  };
  poison();
  for (uint32_t stage = 0; stage < 3; ++stage) {
    metal::CommandGraph graph;
    encode(graph, separate, stage);
    static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
  }
  poison();
  metal::CommandGraph graph;
  for (uint32_t stage = 0; stage < 3; ++stage) encode(graph, combined, stage);
  // One submission and one wait: no CPU synchronization between consumers
  // of the scratch or between the gate/up/down data dependencies.
  static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
  const std::array<const Guarded *, 3> expected{&separate.gate, &separate.up, &separate.down};
  const std::array<const Guarded *, 3> actual{&combined.gate, &combined.up, &combined.down};
  for (uint32_t stage = 0; stage < 3; ++stage) {
    test::require(std::memcmp(expected[stage]->view.contents(), actual[stage]->view.contents(), actual[stage]->bytes) == 0,
                  "shared K partials changed dependent chain output at stage " + std::to_string(stage));
    const auto *out = static_cast<const uint16_t *>(actual[stage]->view.contents());
    const uint32_t covered = stage < 2 ? intermediate : 256;
    for (uint32_t column = 0; column < covered; ++column)
      test::require(std::isfinite(bf16ToFloat(out[column])), "shared K partials produced nonfinite chain output");
    for (uint64_t column = covered; column < actual[stage]->bytes / 2; ++column)
      test::require(out[column] == 0xFFFF, "shared K partials chain wrote padding columns");
    expected[stage]->requireIntact(); actual[stage]->requireIntact();
  }
  partials.requireIntact(); input.requireIntact(); residual.requireIntact();
}

// A fused single token must equal the original three-pass prefill path,
// including every recurrent state element, carried convolution input and
// intermediate rounding boundary. Current state and projection input are
// nonzero, so consuming padded tokens or overwriting before reading fails.
void gdnSingle(MetalBackend &backend, bool floatNorm, GdnHeadOrder order) {
  const GdnShape shape{16, 32, 128, 8192, 12544};
  const auto bfBuffer = [&](uint64_t elements, uint32_t seed) {
    auto buffer = test::sharedBuffer(backend, elements * 2);
    auto *data = static_cast<uint16_t *>(buffer.contents());
    for (uint64_t i = 0; i < elements; ++i)
      data[i] = floatToBf16(float(int((i * 7 + seed) % 31) - 15) * 0.03125f);
    return buffer;
  };
  auto packed = bfBuffer(shape.packedWidth, 13);
  auto convWeights = bfBuffer(shape.convolutionDimension * 4, 19);
  auto convIn = bfBuffer(shape.convolutionDimension * 3, 23);
  auto aScale = test::sharedBuffer(backend, shape.valueHeads * 4);
  auto stateIn = test::sharedBuffer(backend, uint64_t{shape.valueHeads} * 128 * 128 * 4);
  for (uint32_t head = 0; head < shape.valueHeads; ++head)
    static_cast<float *>(aScale.contents())[head] = -0.1f * float(head + 1);
  for (uint64_t i = 0; i < stateIn.sizeBytes() / 4; ++i)
    static_cast<float *>(stateIn.contents())[i] = float(int((i * 17) % 53) - 26) * 0.00127f;
  auto timeBias = bfBuffer(shape.valueHeads, 29);
  auto normBuffer = test::sharedBuffer(backend, 128 * (floatNorm ? 4 : 2));
  for (uint32_t i = 0; i < 128; ++i) {
    const float value = 0.7313f + float(i % 19) * 0.0271f;
    if (floatNorm) static_cast<float *>(normBuffer.contents())[i] = value;
    else static_cast<uint16_t *>(normBuffer.contents())[i] = floatToBf16(value);
  }
  struct Outputs final {
    Guarded carry, queries, keys, values, decay, beta, state, rows, hidden;
    Outputs(MetalBackend &device)
        : carry(device, 3 * 8192 * 2), queries(device, 16 * 128 * 2), keys(device, 16 * 128 * 2),
          values(device, 32 * 128 * 2), decay(device, 32 * 4), beta(device, 32 * 2),
          state(device, 32 * 128 * 128 * 4), rows(device, 32 * 128 * 2), hidden(device, 32 * 128 * 2) {}
  };
  std::array<Outputs, 2> original{Outputs(backend), Outputs(backend)};
  std::array<Outputs, 2> fused{Outputs(backend), Outputs(backend)};
  const auto buffers = [&](Outputs &o, MetalBuffer convolutionIn, MetalBuffer recurrentIn) {
    return GdnPrefillBuffers{packed, convWeights, convolutionIn, o.carry.view, o.queries.view, o.keys.view, o.values.view,
                            aScale, timeBias, o.decay.view, o.beta.view, recurrentIn, o.state.view, o.rows.view,
                            {normBuffer, floatNorm}, o.hidden.view};
  };
  MetalBuffer originalConv = convIn, fusedConv = convIn, originalState = stateIn, fusedState = stateIn;
  // Three steps exercise both buffer slots and then write the first slot
  // again. Each chain carries its own state; no reference state is fed into
  // the fused path, so a mismatch cannot be masked at the following step.
  for (uint32_t step = 0; step < 3; ++step) {
    packed = bfBuffer(shape.packedWidth, 13 + step * 9);
    Outputs &expectedOutput = original[step % 2], &actualOutput = fused[step % 2];
    const auto old = buffers(expectedOutput, originalConv, originalState);
    metal::CommandGraph reference;
    reference.add("prefill_gdn_prepare_vh32", {old.packed, old.convolutionWeights, old.convolutionIn,
        old.convolutionOut, old.queries, old.keys, old.values, old.decayWeights, old.timeBias, old.decay, old.beta},
        GDNPrefillParams{1}, {16, 1, 1}, {128, 1, 1});
    reference.add("prefill_gdn_scan_vh32", {old.queries, old.keys, old.values, old.decay, old.beta,
        old.recurrentIn, old.recurrentOut, old.recurrentRows}, GDNPrefillParams{1}, {256, 1, 1}, {128, 1, 1});
    reference.add(floatNorm ? "prefill_gdn_gate_vh32_f32" : "prefill_gdn_gate_vh32",
        {old.recurrentRows, old.packed, old.mixerNorm.buffer, old.hidden},
        GDNGatePrefillParams{order == GdnHeadOrder::Tiled}, {32, 1, 1}, {128, 1, 1});
    static_cast<void>(backend.submitCommandAsync(reference.command()).wait());
    metal::CommandGraph graph;
    GDN::addPrefill(graph, buffers(actualOutput, fusedConv, fusedState), shape, 1, order);
    test::require(graph.dispatches().size() == 1, "single-token GDN did not fuse its three stages");
    static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
    const std::array<const Guarded *, 9> expected{&expectedOutput.carry, &expectedOutput.queries, &expectedOutput.keys,
        &expectedOutput.values, &expectedOutput.decay, &expectedOutput.beta, &expectedOutput.state,
        &expectedOutput.rows, &expectedOutput.hidden};
    const std::array<const Guarded *, 9> actual{&actualOutput.carry, &actualOutput.queries, &actualOutput.keys,
        &actualOutput.values, &actualOutput.decay, &actualOutput.beta, &actualOutput.state,
        &actualOutput.rows, &actualOutput.hidden};
    for (size_t i = 0; i < expected.size(); ++i) {
      test::require(std::memcmp(expected[i]->view.contents(), actual[i]->view.contents(), actual[i]->bytes) == 0,
                    "fused single-token GDN differs from original three-pass result, step " +
                        std::to_string(step) + " buffer " + std::to_string(i));
      expected[i]->requireIntact(); actual[i]->requireIntact();
    }
    originalConv = expectedOutput.carry.view; originalState = expectedOutput.state.view;
    fusedConv = actualOutput.carry.view; fusedState = actualOutput.state.view;
  }
}

void extents(MetalBackend &backend, std::mt19937 &rng) {
  constexpr uint32_t inputs = 256, columns = 256, splits = 8;
  std::vector<float> x(inputs, 1.0f);
  auto w = weights(backend, Q4K, x, columns, false, rng);
  const uint32_t stride = w.projection.outputSize;
  const std::array<test::BufferExtent, 4> sizes = {{{0, inputs * 2, 2, "single-token input"},
      {1, stride * 2, 2, "single-token output"}, {2, stride * 2, 2, "single-token auxiliary"},
      {3, singleTokenLinearScratchBytes(w.projection, splits), 4, "single-token partials"}}};
  test::requireExtents(backend, sizes, [&](metal::CommandGraph &graph, const std::vector<MetalBuffer> &buffers) {
    addSingleTokenLinear(graph, buffers[0], w.projection, buffers[1], buffers[2],
                          LinearEpilogue::Residual, buffers[3], splits);
  });
  for (const uint32_t tile : {0u, 16u, 128u}) {
    metal::CommandGraph graph;
    test::rejects([&] { addSingleTokenLinear(graph, {}, w.projection, {}, {}, LinearEpilogue::None, {}, 1, tile); },
                  "single-token column tile", "invalid single-token column tile accepted");
    test::require(graph.empty(), "invalid single-token column tile encoded a dispatch");
  }
  for (const uint32_t split : {0u, 3u, 16u}) {
    metal::CommandGraph graph;
    test::rejects([&] { addSingleTokenLinear(graph, {}, w.projection, {}, {}, LinearEpilogue::None, {}, split); },
                  "single-token K splits", "invalid single-token K partition accepted");
    test::require(graph.empty(), "invalid single-token K partition encoded a dispatch");
  }
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2) { std::cerr << "usage: gguf-single-token METALLIB\n"; return 2; }
    try {
      MetalBackend backend(argv[1]);
      std::mt19937 rng(47);
      for (const uint32_t tile : {32u, 64u})
        for (uint32_t format = 0; format < GGUF_FMT_COUNT; ++format) {
          checkCase(backend, Fmt(format), 2560, false, true, 512, rng, tile);
          checkCase(backend, Fmt(format), 2560, true, false, 256, rng, tile);
        }
      // FFN down's long K and Q6_K head operands, including the second
      // storage tile and the split boundaries of the actual 4B geometry.
      for (const uint32_t tile : {32u, 64u}) {
        checkCase(backend, Q6K, 9216, false, false, 512, rng, tile);
        splitScratchChain(backend, rng, tile);
      }
      extents(backend, rng);
      for (const bool floatNorm : {false, true})
        for (const auto order : {GdnHeadOrder::Grouped, GdnHeadOrder::Tiled})
          gdnSingle(backend, floatNorm, order);
    } catch (const std::exception &error) {
      std::cerr << "gguf-single-token: FAIL: " << error.what() << '\n';
      return 1;
    }
    std::cout << "gguf-single-token: PASS\n";
    return 0;
  }
}
