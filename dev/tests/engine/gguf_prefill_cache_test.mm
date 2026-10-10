// Persistent FP16 FFN prefill: staged half bytes, direct TensorOps results,
// BF16 epilogues, chunk tails and reconstruction after idle release.
#include "GgufFormatReference.hpp"
#include "TestBuffers.hpp"
#include "TestAdmission.hpp"
#include "model/WeightImages.hpp"
#include "ops/GgufPrefillCache.hpp"
#include "LinearNumerics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <charconv>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash;
using namespace gguf_reference;
using splash::test::bf16ToFloat;
using splash::test::floatToBf16;
using metal::MetalBuffer;

namespace {

float silu(float value) { return value / (1.0f + std::exp(-value)); }

constexpr uint32_t N = 64, K = 256, Offset = 64;
constexpr uint16_t Unwritten = 0xffff;
constexpr std::array<Fmt, 5> Formats{Q4K, Q5K, Q6K, IQ4XS, Q80};
constexpr std::array<uint32_t, 4> Rows{33, 136, 168, 2048};

struct Tensor {
  ops::QuantizedSegment segment;
  std::vector<uint16_t> half;
};

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

MetalBuffer image(model::WeightImages &images, std::string name, std::vector<uint8_t> bytes) {
  const size_t size = bytes.size();
  return images.loadAuxiliary(std::move(name), size,
      [bytes = std::move(bytes)](std::span<uint8_t> destination, const MetalBuffer &) {
        std::copy(bytes.begin(), bytes.end(), destination.begin());
      });
}

Tensor makeTensor(model::WeightImages &images, Fmt format, uint32_t columns, uint32_t inputs,
                  std::mt19937 &random) {
  Tensor tensor;
  std::vector<float> values;
  // Even a 64-output view uses the physical 256-row plane tile stride.
  const uint32_t planeRows = (columns + QUANT_TILE_ROWS - 1) / QUANT_TILE_ROWS * QUANT_TILE_ROWS;
  const Packed packed = repack(format, makeNative(format, planeRows, inputs, random), planeRows, inputs, &values);
  values.resize(uint64_t{columns} * inputs);
  const std::string name = std::string(fmtName(format)) + "/" + std::to_string(columns) + "x" +
                           std::to_string(inputs);
  tensor.segment = ops::QuantizedSegment::planes(
      format, columns, inputs, image(images, name + "/plane0", packed.w0),
      kQuantFormats[format].plane1_bytes ? image(images, name + "/plane1", packed.w1) : MetalBuffer{},
      image(images, name + "/meta", packed.meta));
  tensor.half.reserve(values.size());
  for (const float value : values) tensor.half.push_back(f2h(value));
  return tensor;
}

void checkHalf(const Tensor &tensor) {
  const auto *half = static_cast<const uint16_t *>(tensor.segment.prefillHalf.contents());
  require(half && std::equal(tensor.half.begin(), tensor.half.end(), half),
          std::string(tensor.segment.name()) + " cache differs from single FP16 rounding");
}

void cache(metal::MetalBackend &backend, model::WeightImages &images, Tensor &tensor) {
  const auto source = tensor.segment;
  tensor.segment.prefillHalf = images.loadAuxiliary(
      std::string(source.name()) + "/prefill-fp16", ops::ggufPrefillCacheBytes(source),
      [&backend, source](std::span<uint8_t>, const MetalBuffer &destination) {
        ops::writeGgufPrefillCache(backend, source, destination);
      });
  checkHalf(tensor);
}

std::vector<uint16_t> run(metal::MetalBackend &backend, const Tensor &tensor, uint32_t rows,
                          char epilogue, bool cached, uint32_t inputStep = GGUF_PREFILL_STEP,
                          ops::GgufHalfPrefillTile tile = ops::GgufHalfPrefillTile::M32N64) {
  const uint32_t columns = tensor.segment.outputSize, inputs = tensor.segment.inputSize;
  const uint32_t stride = Offset + columns + 64;
  const bool fullReference = columns == N && inputs == K;
  const uint32_t storage = (rows + GGUF_PREFILL_ROWS - 1) / GGUF_PREFILL_ROWS * GGUF_PREFILL_ROWS;
  const auto input = test::sharedBuffer(backend, uint64_t{storage} * inputs * 2);
  const auto auxiliary = test::sharedBuffer(backend, uint64_t{storage} * stride * 2);
  // Extra rows and untouched leading columns check the dispatch bounds.
  const auto output = test::sharedBuffer(backend, uint64_t{storage + 128} * stride * 2);
  auto *x = static_cast<uint16_t *>(input.contents());
  auto *aux = static_cast<uint16_t *>(auxiliary.contents());
  auto *out = static_cast<uint16_t *>(output.contents());
  for (uint32_t r = 0; r < storage; ++r)
    for (uint32_t k = 0; k < inputs; ++k)
      x[uint64_t{r} * inputs + k] = floatToBf16(float(int((r * 17 + k * 13) % 101) - 50) / 53);
  for (uint32_t r = 0; r < storage; ++r)
    for (uint32_t n = 0; n < stride; ++n)
      aux[uint64_t{r} * stride + n] = floatToBf16(
          epilogue == 'r' ? 64.f + float(int((r + n) % 31) - 15) : float(int((r * 3 + n) % 41) - 20) / 7);
  std::fill(out, out + uint64_t{storage + 128} * stride, Unwritten);
  metal::CommandGraph graph;
  const GgufPrefillParams params{inputs, rows, stride, Offset};
  if (cached) {
    ops::addGgufHalfPrefill(graph, input, tensor.segment, output, params, epilogue,
                           epilogue == 'a' ? MetalBuffer{} : auxiliary, inputStep, tile);
  } else {
    const auto grid = metal::DispatchSize{storage / GGUF_PREFILL_ROWS, columns / GGUF_TILE_COLUMNS, 1};
    if (epilogue == 'a') {
      // Plain prefill now binds three segment slots, including the unused ones.
      std::vector<MetalBuffer> buffers{input};
      for (uint32_t i = 0; i < 3; ++i)
        buffers.insert(buffers.end(), {tensor.segment.plane0, tensor.segment.plane1Slot(), tensor.segment.meta});
      buffers.push_back(output);
      graph.add(std::string("gguf_prefill_") + tensor.segment.name() + "_a", std::move(buffers),
                GgufPrefillSegmentsParams{inputs, rows, stride, {columns, 0, 0}, {Offset, 0, 0}},
                grid, {GGUF_PREFILL_THREADS, 1, 1});
    } else {
      std::vector<MetalBuffer> buffers{input, tensor.segment.plane0, tensor.segment.plane1Slot(),
                                      tensor.segment.meta, output, auxiliary};
      graph.add(std::string("gguf_prefill_") + tensor.segment.name() + "_" + epilogue,
                std::move(buffers), params, grid, {GGUF_PREFILL_THREADS, 1, 1});
    }
  }
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  // Incomplete simdgroups compute their 32-row tile, as production does.
  const uint32_t simdgroupRows = cached && tile == ops::GgufHalfPrefillTile::M64N32 ? 64 :
                                GGUF_PREFILL_SIMDGROUP_ROWS;
  const uint32_t writtenRows = (rows + simdgroupRows - 1) / simdgroupRows * simdgroupRows;
  for (uint32_t r = 0; r < storage + 128; ++r)
    for (uint32_t n = 0; n < stride; ++n) {
      const uint64_t at = uint64_t{r} * stride + n;
      if (r >= writtenRows || n < Offset || n >= Offset + columns) {
        require(out[at] == Unwritten, "prefill wrote past its row or segment extent");
        continue;
      }
      if (r >= rows) continue;
      require(std::isfinite(bf16ToFloat(out[at])), "prefill produced a nonfinite output");
      const uint32_t column = n - Offset;
      if (!fullReference &&
          ((r != 0 && r != 31 && r != 32 && r != 127 && r != 128 && r != rows / 2 && r != rows - 1) ||
           (column != 0 && column != 31 && column != 32 && column != 63 && column != 64 &&
            column != columns / 2 && column != columns - 1)))
        continue;
      // Full FP32 reference for the small fixture; sampled FP64 dot products
      // at row/column and tile boundaries for the real 4B shapes. Every real
      // KS64 output also compares bitwise with the original staged GEMM;
      // larger K steps keep this same numerical bound.
      float sum = 0;
      double exact = 0;
      double magnitude = 0;
      for (uint32_t k = 0; k < inputs; ++k) {
        const float activation = bf16ToFloat(x[uint64_t{r} * inputs + k]);
        const float weight = h2f(tensor.half[uint64_t{n - Offset} * inputs + k]);
        if (fullReference) sum = std::fma(activation, weight, sum);
        else exact += double(activation) * weight;
        magnitude += std::fabs(double(activation) * weight);
      }
      if (!fullReference) sum = float(exact);
      const float auxValue = bf16ToFloat(aux[at]);
      float reference = sum;
      double slack = magnitude * 0x1p-20;
      if (epilogue == 'r') reference += auxValue;
      if (epilogue == 'g') {
        reference = bf16ToFloat(floatToBf16(sum)) * silu(auxValue);
        slack = (slack + splash::test::ulpBf16(sum)) * std::fabs(silu(auxValue));
      }
      const float expected = bf16ToFloat(floatToBf16(reference));
      const float actual = bf16ToFloat(out[at]);
      require(std::isfinite(actual) && std::fabs(actual - expected) <=
                  splash::test::ulpBf16(reference) + slack,
              std::string(tensor.segment.name()) + " prefill CPU/BF16 reference mismatch");
    }
  return {out, out + uint64_t{storage + 128} * stride};
}

void compare(metal::MetalBackend &backend, const Tensor &tensor, uint32_t rows, char epilogue,
             uint32_t inputStep, ops::GgufHalfPrefillTile tile) {
  const auto cached = run(backend, tensor, rows, epilogue, true, inputStep, tile);
  const auto staged = run(backend, tensor, rows, epilogue, false);
  require(cached.size() == staged.size(), "prefill output extents differ");
  size_t differences = 0;
  const uint32_t columns = tensor.segment.outputSize, stride = Offset + columns + 64;
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t column = 0; column < columns; ++column) {
      const uint64_t at = uint64_t{row} * stride + Offset + column;
      differences += cached[at] != staged[at];
    }
  const std::string description = std::string(tensor.segment.name()) + " " +
      std::to_string(tensor.segment.outputSize) + "x" + std::to_string(tensor.segment.inputSize) +
      " rows=" + std::to_string(rows) + " ep=" + epilogue + " Kstep=" + std::to_string(inputStep) +
      (tile == ops::GgufHalfPrefillTile::M64N32 ? " M64N32" : " M32N64");
  if (inputStep == GGUF_PREFILL_STEP && tile == ops::GgufHalfPrefillTile::M32N64)
    require(cached == staged, description + " direct-half differs from staged prefill");
  else
    std::cout << description << ": " << differences << " BF16 bit differences; CPU bounds/guards ok\n";
}

void realShapes(metal::MetalBackend &backend, std::mt19937 &random, uint32_t inputStep,
                ops::GgufHalfPrefillTile tile) {
  for (const auto [columns, inputs] : {std::pair{9216u, 2560u}, std::pair{2560u, 9216u}})
    for (const Fmt format : Formats) {
      if (inputs % inputStep) continue;
      model::WeightImages images(backend, "prefill-cache-real-4B-shape");
      Tensor tensor = makeTensor(images, format, columns, inputs, random);
      cache(backend, images, tensor);
      require(ops::ggufPrefillCacheBytes(tensor.segment) == uint64_t{columns} * inputs * 2,
              "real 4B cache has the wrong byte count");
      for (const uint32_t rows : Rows)
        for (const char epilogue : {'a', 'r', 'g'}) compare(backend, tensor, rows, epilogue, inputStep, tile);
    }
}

class WriterFailure final : public std::runtime_error {
public:
  WriterFailure() : std::runtime_error("injected prefill cache writer failure") {}
};

void writerFailure(metal::MetalBackend &backend) {
  const uint64_t baseline = backend.memoryStats().allocatedBytes;
  {
    model::WeightImages images(backend, "prefill-cache-writer-failure");
    // Keep one successful image alive so destruction also exercises its
    // cleanup after a later auxiliary image failed before registration.
    static_cast<void>(image(images, "source", std::vector<uint8_t>(4096, 0x5a)));
    const uint64_t sourceBytes = backend.memoryStats().allocatedBytes;
    bool caught = false;
    try {
      static_cast<void>(images.loadAuxiliary("failed-cache", 4096,
          [](std::span<uint8_t>, const MetalBuffer &) { throw WriterFailure(); }));
    } catch (const WriterFailure &error) {
      require(std::string(error.what()) == "injected prefill cache writer failure",
              "auxiliary writer replaced the original exception message");
      caught = true;
    }
    require(caught, "auxiliary writer replaced or swallowed the original exception");
    require(images.contents().size() == 1, "failed auxiliary image was registered");
    require(backend.memoryStats().allocatedBytes == sourceBytes, "failed writer leaked its allocation");
  }
  require(backend.memoryStats().allocatedBytes == baseline, "writer failure cleanup leaked source memory");
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 2 || argc > 4) {
      std::cerr << "usage: gguf-prefill-cache <production.metallib> [64|128|256|512|2560|9216] [--m64n32]\n";
      return 2;
    }
    try {
      uint32_t inputStep = GGUF_PREFILL_STEP;
      const bool alternate = argc >= 3 && std::string(argv[argc - 1]) == "--m64n32";
      const int numericArguments = argc - int(alternate);
      require(numericArguments == 2 || numericArguments == 3, "invalid prefill candidate arguments");
      const auto tile = alternate ? ops::GgufHalfPrefillTile::M64N32 : ops::GgufHalfPrefillTile::M32N64;
      if (numericArguments == 3) {
        const std::string step(argv[2]);
        const auto [end, error] = std::from_chars(step.data(), step.data() + step.size(), inputStep);
        require(error == std::errc{} && end == step.data() + step.size() &&
                    (inputStep == 64 || inputStep == 128 || inputStep == 256 || inputStep == 512 ||
                     inputStep == 2560 || inputStep == 9216), "invalid prefill K step");
      }
      require(!alternate || inputStep == 64, "M64N32 requires the 64-input step");
      metal::MetalBackend backend(argv[1]);
      model::WeightImages images(backend, "prefill-cache-fixture");
      std::mt19937 random(42);
      std::vector<Tensor> tensors;
      // All quantized sources precede every derived cache in restore order.
      for (const Fmt format : Formats) tensors.push_back(makeTensor(images, format, N, K, random));
      for (Tensor &tensor : tensors) cache(backend, images, tensor);
      const uint64_t allocated = backend.memoryStats().allocatedBytes;
      std::vector<std::vector<uint16_t>> beforeRelease;
      for (const Tensor &tensor : tensors) {
        if (K % inputStep == 0)
          for (const uint32_t rows : Rows)
            for (const char epilogue : {'a', 'r', 'g'}) compare(backend, tensor, rows, epilogue, inputStep, tile);
        beforeRelease.push_back(run(backend, tensor, 168, 'g', true));
      }
      const auto contents = images.contents();
      std::vector<std::vector<uint8_t>> snapshots;
      for (const auto &entry : contents) snapshots.emplace_back(entry.bytes.begin(), entry.bytes.end());
      images.release();
      require(backend.memoryStats().allocatedBytes == 0, "weight release left allocated memory");
      for (const auto &entry : images.contents()) require(entry.bytes.empty(), "released image retained bytes");
      while (!images.restore(splash::test::admitAll)) {}
      require(backend.memoryStats().allocatedBytes == allocated, "restore changed allocation accounting");
      const auto restored = images.contents();
      for (size_t i = 0; i < restored.size(); ++i)
        require(std::equal(snapshots[i].begin(), snapshots[i].end(), restored[i].bytes.begin()),
                "restored source or cache content differs");
      for (size_t i = 0; i < tensors.size(); ++i) {
        checkHalf(tensors[i]);
        require(beforeRelease[i] == run(backend, tensors[i], 168, 'g', true), "restore changed prefill output");
      }
      realShapes(backend, random, inputStep, tile);
      writerFailure(backend);
      std::cout << "FP16 prefill cache: 5 formats, 4 chunk sizes, 3 epilogues, real 4B shapes, "
                   "release/restore and writer failure cleanup ok; Kstep=" << inputStep << (alternate ? " M64N32" : " M32N64") << '\n';
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "FP16 prefill cache: FAIL: " << error.what() << '\n';
      return 1;
    }
  }
}
