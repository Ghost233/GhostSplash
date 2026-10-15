// Native-format feasibility and isolated matmul throughput, not an LLM score.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Params { uint32_t pages, pageBytes; };
struct Shape { const char *op; uint32_t m, n, k; };
struct Variant {
  const char *name;
  uint32_t bits;
  id<MTLComputePipelineState> pipeline;
  id<MTLBuffer> rhs;
  id<MTLBuffer> output;
  double maxError = 0;
  std::vector<double> samples;
};
constexpr uint32_t kThreads = 256;
constexpr uint32_t kChecks = 4;
constexpr uint32_t kCheckPages = 2;
constexpr uint32_t kGroups = 128;
constexpr uint32_t kPages = 32;
constexpr uint32_t kRounds = 11;

id<MTLBuffer> allocate(id<MTLDevice> device, uint64_t bytes) {
  auto b = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  if (!b) throw std::runtime_error("buffer allocation failed");
  std::memset(b.contents, 0, bytes);
  return b;
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> lib,
                                     const std::string &name) {
  auto f = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
  if (!f) throw std::runtime_error("missing kernel " + name);
  NSError *error = nil;
  auto p = [device newComputePipelineStateWithFunction:f error:&error];
  if (!p) throw std::runtime_error(error.localizedDescription.UTF8String);
  if (p.maxTotalThreadsPerThreadgroup < kThreads)
    throw std::runtime_error("kernel cannot launch 256 threads: " + name);
  return p;
}

int code(uint64_t i, uint32_t bits, bool fullRange) {
  // Every signed encoding boundary occurs in correctness fixtures. Timings
  // use common [-2,1] values: all formats represent exactly the same operands.
  const uint32_t width = fullRange ? (bits == 16 ? 256 : 1U << bits) : 4;
  const uint32_t mixed = uint32_t(i * 37 + (i / 31) * 17 + 11);
  return int(mixed % width) - int(width / 2);
}

void fill(Variant &v, uint64_t count, bool fullRange) {
  std::memset(v.rhs.contents, 0, v.rhs.length);
  auto *bytes = static_cast<uint8_t *>(v.rhs.contents);
  for (uint64_t i = 0; i < count; ++i) {
    const int value = code(i, v.bits, fullRange);
    if (v.bits == 16) static_cast<__bf16 *>(v.rhs.contents)[i] = __bf16(value);
    else if (v.bits == 8) static_cast<int8_t *>(v.rhs.contents)[i] = int8_t(value);
    else {
      const uint32_t perByte = 8 / v.bits;
      bytes[i / perByte] |= uint8_t((uint32_t(value) & ((1U << v.bits) - 1))
                                   << ((i % perByte) * v.bits));
    }
  }
}

double run(id<MTLCommandQueue> queue, Variant &v, id<MTLBuffer> input,
           Params p, uint32_t groups) {
  auto command = [queue commandBuffer];
  command.label = [NSString stringWithUTF8String:v.name];
  auto encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:v.pipeline];
  [encoder setBuffer:input offset:0 atIndex:0];
  [encoder setBuffer:v.rhs offset:0 atIndex:1];
  [encoder setBuffer:v.output offset:0 atIndex:2];
  [encoder setBytes:&p length:sizeof(p) atIndex:3];
  [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted)
    throw std::runtime_error(command.error ? command.error.localizedDescription.UTF8String
                                           : "GPU command failed");
  const double ms = (command.GPUEndTime - command.GPUStartTime) * 1000;
  if (!(ms > 0) || !std::isfinite(ms)) throw std::runtime_error("invalid GPU timing");
  return ms;
}

void check(Variant &v, Shape s, id<MTLBuffer> input) {
  const auto *a = static_cast<const __bf16 *>(input.contents);
  const auto *out = static_cast<const float *>(v.output.contents);
  for (uint32_t task = 0; task < kChecks; ++task)
    for (uint32_t row = 0; row < s.m; ++row)
      for (uint32_t col = 0; col < s.n; ++col) {
        double expected = 0;
        for (uint32_t page = 0; page < kCheckPages; ++page)
          for (uint32_t k = 0; k < s.k; ++k) {
            const uint64_t at = (uint64_t(task) * kCheckPages + page) * s.k * s.n
                                + uint64_t(col) * s.k + k;
            expected += float(a[row * s.k + k]) * code(at, v.bits, true);
          }
        const double observed = out[(task * ((s.m + 7) / 8 * 8) + row) * s.n + col];
        const double error = std::abs(observed - expected);
        if (!std::isfinite(observed) || error > 1e-5)
          throw std::runtime_error(std::string(s.op) + " " + v.name +
                                   " CPU-reference mismatch: " + std::to_string(error));
        v.maxError = std::max(v.maxError, error);
      }
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

NSDictionary *measure(id<MTLDevice> device, id<MTLLibrary> lib,
                       id<MTLCommandQueue> queue, Shape s, bool correctnessOnly) {
  const uint32_t groups = correctnessOnly ? kChecks : kGroups;
  const uint32_t pages = correctnessOnly ? kCheckPages : kPages;
  const uint64_t elements = uint64_t(groups) * pages * s.k * s.n;
  auto input = allocate(device, uint64_t((s.m + 7) / 8 * 8) * s.k * 2);
  auto *a = static_cast<__bf16 *>(input.contents);
  for (uint32_t i = 0; i < s.m * s.k; ++i)
    a[i] = __bf16(float(int((i * 19 + i / 7) % 31) - 15) / 32);
  std::vector<Variant> variants;
  for (const auto &[name, bits] : std::array<std::pair<const char *, uint32_t>, 4>{
           {{"bf16", 16}, {"int8", 8}, {"int4", 4}, {"int2", 2}}}) {
    const std::string kernel = std::string(s.op) + "_m" + std::to_string(s.m) + "_" + name;
    Variant v{name, bits, pipeline(device, lib, kernel),
              allocate(device, elements * bits / 8),
              allocate(device, uint64_t(groups) * ((s.m + 7) / 8 * 8) * s.n * 4), 0, {}};
    fill(v, uint64_t(kChecks) * kCheckPages * s.k * s.n, true);
    std::memset(v.output.contents, 0xff, v.output.length);
    run(queue, v, input, {kCheckPages, s.k * s.n * bits / 8}, kChecks);
    check(v, s, input);
    std::cerr << s.op << " M=" << s.m << " " << name << " CPU reference PASS\n";
    if (!correctnessOnly) fill(v, elements, false);
    variants.push_back(std::move(v));
  }
  if (!correctnessOnly) {
    for (uint32_t round = 0; round < kRounds + 3; ++round)
      for (uint32_t offset = 0; offset < variants.size(); ++offset) {
        const uint32_t index = round % 2 ? (round + variants.size() - offset) % variants.size()
                                        : (round + offset) % variants.size();
        Variant &v = variants[index];
        const double ms = run(queue, v, input, {pages, s.k * s.n * v.bits / 8}, groups);
        if (round >= 3) v.samples.push_back(ms);
      }
  }
  NSMutableArray *results = [NSMutableArray array];
  for (const Variant &v : variants) {
    NSMutableArray *samples = [NSMutableArray array];
    for (double value : v.samples) [samples addObject:@(value)];
    NSMutableDictionary *result = [@{
      @"format": [NSString stringWithUTF8String:v.name], @"rhs_bytes": @(v.rhs.length),
      @"rhs_allocated_bytes": @(v.rhs.allocatedSize), @"cpu_reference_pass": @YES,
      @"max_abs_error": @(v.maxError), @"gpu_ms_samples": samples
    } mutableCopy];
    if (!v.samples.empty()) result[@"median_gpu_ms"] = @(median(v.samples));
    [results addObject:result];
  }
  return @{@"operation": [NSString stringWithUTF8String:s.op], @"m": @(s.m),
           @"physical_m": @((s.m + 7) / 8 * 8), @"n": @(s.n), @"k": @(s.k), @"groups": @(groups),
           @"pages_per_group": @(pages), @"rhs_pages": @(groups * pages),
           @"formats": results};
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--correctness-only"))
        throw std::invalid_argument("usage: kv-tensor-benchmark METALLIB [--correctness-only]");
      if (@available(macOS 27.0, *)) {} else
        throw std::runtime_error("this experimental INT2 probe requires macOS 27");
      auto device = MTLCreateSystemDefaultDevice();
      if (!device) throw std::runtime_error("no Metal device");
      NSError *error = nil;
      auto lib = [device newLibraryWithURL:[NSURL fileURLWithPath:
                  [NSString stringWithUTF8String:argv[1]]] error:&error];
      if (!lib) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto queue = [device newCommandQueue];
      if (!queue) throw std::runtime_error("no Metal queue");
      const bool correctnessOnly = argc == 3;
      NSMutableArray *cases = [NSMutableArray array];
      for (Shape shape : std::array<Shape, 4>{{{"qk", 4, 32, 256}, {"pv", 4, 256, 32},
                                             {"qk", 32, 32, 256}, {"pv", 32, 256, 32}}}) {
        @autoreleasepool { [cases addObject:measure(device, lib, queue, shape, correctnessOnly)]; }
      }
      NSDictionary *report = @{
        @"device": device.name, @"os": NSProcessInfo.processInfo.operatingSystemVersionString,
        @"scope": @"isolated BF16 x native integer/BF16 matmul; not attention or model quality",
        @"compiler_target_macos": @"27.0", @"metal_language": @"4.1", @"correctness_only": @(correctnessOnly),
        @"query_heads_per_kv_head": @4, @"head_dimension": @256,
        @"timed_rhs": @"common integer [-2,1] values; no scales or rotation",
        @"timing_policy": @"3 warmups, 11 samples, alternating format order, GPU command time",
        @"pass": @YES, @"cases": cases
      };
      auto data = [NSJSONSerialization dataWithJSONObject:report
                    options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:&error];
      if (!data) throw std::runtime_error(error.localizedDescription.UTF8String);
      std::cout.write(static_cast<const char *>(data.bytes), data.length);
      std::cout << '\n';
      return 0;
    } catch (const std::exception &e) {
      std::cerr << "kv-tensor-benchmark: " << e.what() << '\n';
      return 1;
    }
  }
}
