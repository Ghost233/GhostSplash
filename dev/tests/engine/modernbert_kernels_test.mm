#include "metal/CommandGraph.hpp"
#include "metal/abi/ModernBert.h"
#include "ops/ModernBert.hpp"
#include "LinearNumerics.hpp"
#include "TestChecks.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

using namespace splash;
using namespace splash::test;
metal::MetalBuffer shared(metal::MetalBackend &backend, uint64_t bytes) {
  auto b = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, "modern-test");
  std::memset(b.contents(), 0, bytes); return b;
}

void uniformAttention(metal::MetalBackend &b, uint32_t tokens, uint32_t window) {
  uint32_t pad = (tokens + 127) / 128 * 128;
  auto q = shared(b, uint64_t{pad} * 64 * 2), k = shared(b, uint64_t{pad} * 64 * 2);
  auto v = shared(b, uint64_t{pad} * 64 * 2), output = shared(b, uint64_t{pad} * 64 * 2);
  std::memset(output.contents(), 0xFF, output.sizeBytes());
  auto *values = static_cast<uint16_t *>(v.contents());
  for (uint32_t t = 0; t < tokens; ++t)
    for (uint32_t c = 0; c < 64; ++c) values[t * 64 + c] = floatToBf16(float(t + c) / 128.0f);
  metal::CommandGraph g;
  ops::ModernBert::attention(g, {q, k, v, output}, ModernAttentionParams{tokens, pad, 1, tokens, pad, window});
  static_cast<void>(b.submitCommandAsync(g.command()).wait());
  const auto *got = static_cast<const uint16_t *>(output.contents());
  for (uint32_t t = 0; t < tokens; ++t)
    for (uint32_t c = 0; c < 64; ++c) {
      double sum = 0; uint32_t count = 0;
      for (uint32_t j = 0; j < tokens; ++j)
        if (!window || std::abs(int(t) - int(j)) <= int(window / 2)) { sum += bf16ToFloat(values[j * 64 + c]); ++count; }
      uint16_t expected = floatToBf16(static_cast<float>(sum / count));
      if (got[t * 64 + c] != expected)
        throw std::runtime_error("uniform attention differs from FP64 mean at t=" + std::to_string(t) +
            " c=" + std::to_string(c) + ": actual=" + std::to_string(bf16ToFloat(got[t * 64 + c])) +
            " expected=" + std::to_string(bf16ToFloat(expected)));
    }
  std::cout << "uniform FP64 attention tokens=" << tokens << " window=" << window << " PASS\n";
}
void normalization(metal::MetalBackend &b, uint32_t width) {
  uint32_t stride = (width + 255) / 256 * 256;
  auto hidden = shared(b, uint64_t{stride} * 4), delta = shared(b, uint64_t{stride} * 4);
  auto weight = shared(b, uint64_t{stride} * 4), bias = shared(b, uint64_t{stride} * 4);
  auto out = shared(b, uint64_t{stride} * 2);
  auto *h = static_cast<float *>(hidden.contents()), *w = static_cast<float *>(weight.contents());
  std::vector<double> original(width); double mean = 0;
  for (uint32_t c = 0; c < width; ++c) { h[c] = float(int(c % 31) - 15) / 16; w[c] = 1; original[c] = h[c]; mean += h[c]; }
  mean /= width; double variance = 0;
  for (double x : original) variance += (x - mean) * (x - mean); variance /= width;
  metal::CommandGraph g;
  ops::ModernBert::normalization(g, {hidden, delta, bias, weight, bias, out}, ModernNormParams{1, width, stride, stride, 0, 1, 1e-5f});
  static_cast<void>(b.submitCommandAsync(g.command()).wait());
  for (uint32_t c = 0; c < width; ++c) {
    double expected = (original[c] - mean) / std::sqrt(variance + 1e-5);
    require(std::fabs(h[c] - expected) <= 2e-6 * std::max(1.0, std::fabs(expected)), "LayerNorm outside FP64 bound");
  }
  std::cout << "FP64 LayerNorm width=" << width << " PASS\n";
}
void nonuniformAttention(metal::MetalBackend &b) {
  constexpr uint32_t n = 128;
  auto q = shared(b, n * 64 * 2), k = shared(b, n * 64 * 2);
  auto v = shared(b, n * 64 * 2), out = shared(b, n * 64 * 2);
  auto *qq = static_cast<uint16_t *>(q.contents()), *kk = static_cast<uint16_t *>(k.contents()), *vv = static_cast<uint16_t *>(v.contents());
  for (uint32_t t = 0; t < n; ++t) {
    qq[t * 64] = floatToBf16(float(t % 5) / 4); kk[t * 64] = floatToBf16(float(t % 3) / 2);
    for (uint32_t c = 0; c < 64; ++c) vv[t * 64 + c] = floatToBf16(float(t + c) / 128);
  }
  metal::CommandGraph g;
  ops::ModernBert::attention(g, {q,k,v,out}, ModernAttentionParams{n,n,1,n,n,0});
  static_cast<void>(b.submitCommandAsync(g.command()).wait());
  const auto *got = static_cast<const uint16_t *>(out.contents());
  for (uint32_t t = 0; t < n; ++t) {
    double maximum = bf16ToFloat(qq[t * 64]) * 0.125, denominator = 0;
    std::vector<double> probability(n);
    for (uint32_t j = 0; j < n; ++j) {
      double exponent = std::exp(double(bf16ToFloat(qq[t * 64])) * bf16ToFloat(kk[j * 64]) * 0.125 - maximum);
      denominator += exponent; probability[j] = bf16ToFloat(floatToBf16(float(exponent)));
    }
    for (uint32_t c = 0; c < 64; ++c) {
      double value = 0;
      for (uint32_t j = 0; j < n; ++j) value += probability[j] * bf16ToFloat(vv[j * 64 + c]);
      value /= denominator;
      // Four FP32 reductions/divisions, each bounded by gamma_128, before BF16 rounding.
      double error = 4 * n * std::numeric_limits<float>::epsilon() * std::max(1.0, std::fabs(value));
      double actual = bf16ToFloat(got[t * 64 + c]);
      require(actual >= bf16ToFloat(floatToBf16(float(value - error))) &&
              actual <= bf16ToFloat(floatToBf16(float(value + error))), "nonuniform attention outside FP64 mixed-precision bound");
    }
  }
  std::cout << "nonuniform attention FP64 mixed-precision bound PASS\n";
}
void activation(metal::MetalBackend &b, bool head) {
  constexpr uint32_t width = 384, rows = 3, stride = 512;
  uint32_t sourceStride = head ? width : width * 2;
  auto input = shared(b, uint64_t{rows} * sourceStride * 4), bias = shared(b, width * 4), output = shared(b, rows * stride * 2);
  auto *x = static_cast<float *>(input.contents());
  for (uint32_t i = 0; i < rows * sourceStride; ++i) x[i] = float(int(i % 97) - 48) / 32;
  metal::CommandGraph g;
  ops::ModernBert::activation(g, {input,bias,output}, ModernActivationParams{rows,width,sourceStride,stride,uint32_t(head)});
  static_cast<void>(b.submitCommandAsync(g.command()).wait());
  const auto *got = static_cast<const uint16_t *>(output.contents());
  for (uint32_t r = 0; r < rows; ++r) for (uint32_t c = 0; c < stride; ++c) {
    double value = 0;
    if (c < width) {
      double a = x[r * sourceStride + c];
      value = head ? std::max(a,0.0) : .5 * a * (1 + std::erf(a / std::sqrt(2.0))) * x[r * sourceStride + width + c];
    }
    // FP32 polynomial and multiplication error; the original BF16 rounding is retained.
    double error = 16 * std::numeric_limits<float>::epsilon() * std::max(1.0,std::fabs(value));
    double actual = bf16ToFloat(got[r * stride + c]);
    require(actual >= bf16ToFloat(floatToBf16(float(value-error))) && actual <= bf16ToFloat(floatToBf16(float(value+error))),
            "FFN activation outside independent FP64 bound");
    if (c >= width) require(got[r * stride + c] == 0, "FFN padding is not zero");
  }
  std::cout << "FP64 " << (head ? "ReLU" : "GeGLU-erf") << " PASS\n";
}
int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2) return 2;
    try {
      metal::MetalBackend b(argv[1]);
      normalization(b, 384); normalization(b, 1024);
      nonuniformAttention(b); activation(b,false); activation(b,true);
      uniformAttention(b, 128, 0); uniformAttention(b, 128, 128);
      for (uint32_t repeat = 0; repeat < 64; ++repeat) { uniformAttention(b, 129, 0); uniformAttention(b, 129, 128); }
    } catch (const std::exception &e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
  }
  return 0;
}
