// Independent double-precision numerical oracles for the Flash Next operators.
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "ops/FlashNext.hpp"
#include "tuning/LinearNumerics.hpp"
#import <Foundation/Foundation.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
using namespace splash;
using metal::CommandGraph;
using metal::MetalBuffer;
using ops::FlashNext;
using ops::tuning::bf16ToFloat;
using ops::tuning::floatToBf16;
namespace {
MetalBuffer values(metal::MetalBackend &b, const std::vector<float> &v) {
  auto x = b.allocateBuffer(v.size() * 2, metal::BufferStorage::Shared, {});
  auto *p = static_cast<uint16_t *>(x.contents());
  for (size_t i = 0; i < v.size(); ++i)
    p[i] = floatToBf16(v[i]);
  return x;
}
void check(const MetalBuffer &x, const std::vector<double> &expected,
           const char *name) {
  const auto *p = static_cast<const uint16_t *>(x.contents());
  for (size_t i = 0; i < expected.size(); ++i) {
    const double a = bf16ToFloat(p[i]);
    test::require(std::isfinite(a) && std::abs(a - expected[i]) <=
                                          0.012 * (1 + std::abs(expected[i])),
                  std::string(name) + " index " + std::to_string(i));
  }
}
void groupedNorm(metal::MetalBackend &b) {
  auto input = values(b, {1, 2, 3, 4, 0, 0, 0, 0, -2, 1, -1, 2, 1, 1, 1, 1});
  auto gamma = values(b, {1, 2, 1, 0.5, 1, 1, 1, 1});
  auto out = values(b, std::vector<float>(16));
  CommandGraph graph;
  FlashNext::addGroupedRms(graph, input, {gamma, false}, out, 4, 2, 2);
  static_cast<void>(b.submitCommandAsync(graph.dispatches()).wait());
  const double s = std::sqrt(7.5 + 1e-6), t = std::sqrt(2.5 + 1e-6),
               u = std::sqrt(1 + 1e-6);
  check(out,
        {1 / s, 4 / s, 3 / s, 2 / s, 0, 0, 0, 0, -2 / t, 2 / t, -1 / t, 1 / t,
         1 / u, 1 / u, 1 / u, 1 / u},
        "group RMS");
}
void hyperConnection(metal::MetalBackend &b) {
  // Zero logits: sigmoid=1/2, injection=1. Four streams average to a quarter
  // of their already halved sum. No numerical oracle shares a kernel loop.
  auto norm = values(b, {2, 4, 6, 8, 10, 12, 14, 16});
  auto logits = values(b, std::vector<float>(8));
  auto mixed = values(b, {0, 0});
  auto inject = values(b, {0, 0, 0, 0});
  auto result = values(b, std::vector<float>(8));
  auto low = values(b, {-4, 0, 4});
  auto activated = values(b, {0, 0, 0});
  CommandGraph graph;
  FlashNext::addHcMix(graph, norm, logits, mixed, 2, 4, 1);
  FlashNext::addHcScatter(graph, norm, mixed, inject, result, 2, 4, 1);
  FlashNext::addHcSilu(graph, low, activated, 3, 4, 1);
  static_cast<void>(b.submitCommandAsync(graph.dispatches()).wait());
  check(mixed, {4, 5}, "HC mean");
  check(result, {6, 9, 10, 13, 14, 17, 18, 21}, "HC scatter");
  check(activated, {-0.2689414213699951, 0, 0.7310585786300049},
        "HC lowrank silu");
}
void ple(metal::MetalBackend &b) {
  auto key = values(b, {1, 1, 1, 1, 1, 1, 1, 1});
  auto query = values(b, {1, 1, 1, 1, -1, -1, -1, -1});
  auto value = values(b, {2, -2, 4, 0});
  auto gated = values(b, std::vector<float>(8));
  CommandGraph gate;
  FlashNext::addPleGate(gate, key, query, value, gated, 4, 2, 1);
  static_cast<void>(b.submitCommandAsync(gate.dispatches()).wait());
  const double a = 1 / (1 + std::exp(-std::sqrt(2.0))), c = 1 - a;
  check(gated, {2 * a, -2 * a, 4 * a, 0, 2 * c, -2 * c, 4 * c, 0},
        "PLE signed square-root gate");
  const uint32_t rows = 7, width = 2;
  std::vector<float> input;
  for (uint32_t r = 0; r < rows; ++r) {
    input.push_back(float(r + 1));
    input.push_back(float(-int(r) - 1));
  }
  auto normed = values(b, input),
       direct = values(b, std::vector<float>(rows * width, 1));
  auto state =
      values(b, {10, 20, 30, 40, 50, 60}); // three history rows, dilation 3
  auto next = values(b, std::vector<float>(6));
  auto output = values(b, std::vector<float>(14));
  auto weights =
      b.allocateBuffer(4 * sizeof(float), metal::BufferStorage::Shared, {});
  const float w[] = {0.5, 1, -0.25, 0.5};
  std::memcpy(weights.contents(), w, sizeof(w));
  CommandGraph conv;
  FlashNext::addPleConv(conv, direct, normed, state, weights, output, next,
                        width, 1, rows, 2, 3);
  static_cast<void>(b.submitCommandAsync(conv.dispatches()).wait());
  std::vector<double> expected;
  // Explicit previous rows, not the operator's state indexing formula.
  const double prev0[] = {10, 30, 50, 1, 2, 3, 4},
               prev1[] = {20, 40, 60, -1, -2, -3, -4};
  for (uint32_t r = 0; r < rows; ++r) {
    const double z0 = 0.5 * prev0[r] + r + 1,
                 z1 = -0.25 * prev1[r] - 0.5 * (r + 1);
    expected.push_back(1 + z0 / (1 + std::exp(-z0)));
    expected.push_back(1 + z1 / (1 + std::exp(-z1)));
  }
  check(output, expected, "PLE dilated causal convolution");
  check(next, {5, -5, 6, -6, 7, -7}, "PLE history commit");
}
MetalBuffer floats(metal::MetalBackend &b, const std::vector<float> &v) {
  auto x = b.allocateBuffer(v.size() * sizeof(float),
                            metal::BufferStorage::Shared, {});
  std::memcpy(x.contents(), v.data(), v.size() * sizeof(float));
  return x;
}
MetalBuffer positions(metal::MetalBackend &b, const std::vector<uint32_t> &v) {
  auto x = b.allocateBuffer(v.size() * sizeof(uint32_t),
                            metal::BufferStorage::Shared, {});
  std::memcpy(x.contents(), v.data(), v.size() * sizeof(uint32_t));
  return x;
}
void qsa(metal::MetalBackend &b) {
  auto raw = values(b, {1, 2, 3, 4, 5, 6, 7, 8});
  auto pool = values(b, {-5, -5, 0, 0});
  auto x = values(b, {1, 2, 3, 4, 5, 6}),
       rot = values(b, std::vector<float>(6));
  auto cos = floats(b, {0, 0}), sin = floats(b, {1, 1}),
       pos = positions(b, {0});
  CommandGraph graph;
  FlashNext::addQsaPool(graph, raw, pool, 2, 2, 1, 1);
  FlashNext::addRope(graph, x, cos, sin, pos, rot, 6, 1, 1, 4, 1);
  static_cast<void>(b.submitCommandAsync(graph.dispatches()).wait());
  check(pool, {-5, -5, 6, 7}, "QSA partial block pool");
  check(rot, {-3, -4, 1, 2, 5, 6}, "split half partial RoPE");
  auto keys = values(b, {1, 0, 0, 1, -1, 1, 2, 0});
  auto queries = values(b, {1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1});
  auto qpos = positions(b, {1, 6, 8});
  auto scores = floats(b, std::vector<float>(12));
  auto selected =
      b.allocateBuffer(15 * sizeof(int32_t), metal::BufferStorage::Shared, {});
  CommandGraph selection;
  FlashNext::addQsaScores(selection, queries, keys, qpos, scores, 2, 2, 3, 4,
                          2);
  FlashNext::addQsaSelect(selection, scores, qpos, selected, 3, 4, 2, 4);
  static_cast<void>(b.submitCommandAsync(selection.dispatches()).wait());
  auto empty = b.allocateBuffer(1, metal::BufferStorage::Shared, {});
  auto smallPositions = positions(b, {0, 2});
  auto tailOnly =
      b.allocateBuffer(14 * sizeof(int32_t), metal::BufferStorage::Shared, {});
  CommandGraph first;
  FlashNext::addQsaSelect(first, empty, smallPositions, tailOnly, 2, 0, 4, 4);
  static_cast<void>(b.submitCommandAsync(first.dispatches()).wait());
  const auto *tail = static_cast<const int32_t *>(tailOnly.contents());
  const int32_t goldenTail[] = {0, -1, -1, -1, -1, -1, -1,
                                0, 1,  2,  -1, -1, -1, -1};
  test::require(std::equal(tail, tail + 14, goldenTail),
                "QSA incomplete first block");
  auto *sv = static_cast<const float *>(scores.contents());
  test::require(std::isinf(sv[1]) && sv[1] < 0, "QSA future block scored");
  test::require(std::abs(sv[11] - std::sqrt(2.0)) < 1e-5,
                "QSA relu head sum scaling");
  auto *ids = static_cast<const int32_t *>(selected.contents());
  const std::vector<std::vector<int32_t>> golden = {
      {-1, -1, -1, 0, 1}, {0, 1, 2, 3, 6}, {0, 1, 6, 7, 8}};
  for (size_t row = 0; row < 3; ++row) {
    std::vector<int32_t> got(ids + row * 5, ids + (row + 1) * 5);
    std::sort(got.begin(), got.end());
    test::require(got == golden[row], "QSA block selection/tail mismatch");
  }
}
void gdnAndUtilities(metal::MetalBackend &b) {
  auto input = values(b, {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6});
  auto z = values(b, {0, 0, 1, 1, -1, -1, 2, 2, -2, -2, 3, 3});
  auto norm = floats(b, {2, 4});
  auto out = values(b, std::vector<float>(12));
  CommandGraph graph;
  FlashNext::addGdnGate(graph, input, z, {norm, true}, out, 2, 2, 6, 1, true);
  static_cast<void>(b.submitCommandAsync(graph.dispatches()).wait());
  std::vector<double> expected;
  for (double v : {0, 2, 1, -2, -1, 3}) {
    const double factor = 1 / (1 + std::exp(-v));
    expected.push_back(2 * factor);
    expected.push_back(4 * factor);
  }
  check(out, expected, "sigmoid GDN gate and tiled output");
  auto a = values(b, {0, 1, -1}), v = values(b, {2, 2, 2}),
       r = values(b, {0, 0, 0});
  auto acc = floats(b, {1, 1, 1}), bf = values(b, {0, 0, 0});
  CommandGraph utility;
  FlashNext::addSiluMul(utility, a, v, r, 3, 1);
  FlashNext::addScaledAccum(utility, r, acc, 3, 1, 0.5f);
  FlashNext::addFloatToBf16(utility, acc, bf, 3, 1);
  static_cast<void>(b.submitCommandAsync(utility.dispatches()).wait());
  check(bf, {1, 1.7310585786300049, 0.7310585786300049},
        "scaled expert accumulation");
}
void selectedAttention(metal::MetalBackend &b) {
  for (auto format : {kv::Format::BFloat16, kv::Format::Int8}) {
    const uint32_t data = 2 * 32 * 256 * (format == kv::Format::Int8 ? 1 : 2),
                   scale = format == kv::Format::Int8 ? 256 : 0;
    const uint32_t pages = 2;
    auto extent = b.allocateBuffer(uint64_t(pages) * 4 * (data + scale),
                                   metal::BufferStorage::Shared, {});
    test::require((extent.gpuAddress() & SPLASH_KV_PAGE_INDEX_MASK) == 0,
                  "KV extent address alignment");
    std::memset(extent.contents(), 0, extent.sizeBytes());
    auto *base = static_cast<uint8_t *>(extent.contents());
    for (uint32_t page = 0; page < pages; ++page)
      for (uint32_t head = 0; head < 2; ++head)
        for (uint32_t token = 0; token < 32; ++token) {
          const uint32_t absolute = page * 32 + token;
          const float value = (head ? -1 : 1) * (absolute == 0    ? 2
                                                 : absolute == 2  ? 4
                                                 : absolute == 33 ? 6
                                                                  : 100);
          if (format == kv::Format::Int8) {
            auto *ks = reinterpret_cast<float *>(
                base + splash_kv_offset(pages, data, scale, 1,
                                        SPLASH_KV_KEY_SCALES, page));
            auto *vs = reinterpret_cast<float *>(
                base + splash_kv_offset(pages, data, scale, 1,
                                        SPLASH_KV_VALUE_SCALES, page));
            ks[splash_kv_scale_element(head, token)] = 1;
            vs[splash_kv_scale_element(head, token)] = 1;
          }
          const float key = absolute == 2 ? 1 : absolute == 33 ? 2 : 0;
          const auto keyoff =
              splash_kv_offset(pages, data, scale, 1, SPLASH_KV_KEYS, page);
          if (format == kv::Format::Int8)
            reinterpret_cast<int8_t *>(
                base + keyoff)[splash_kv_key_element(head, token, 0)] =
                int8_t(key);
          else
            reinterpret_cast<uint16_t *>(
                base + keyoff)[splash_kv_key_element(head, token, 0)] =
                floatToBf16(key);
          for (uint32_t c = 0; c < 256; ++c) {
            const auto off =
                splash_kv_offset(pages, data, scale, 1, SPLASH_KV_VALUES, page);
            const auto idx = splash_kv_value_element(head, token, c);
            if (format == kv::Format::Int8)
              reinterpret_cast<int8_t *>(base + off)[idx] = int8_t(value);
            else
              reinterpret_cast<uint16_t *>(base + off)[idx] =
                  floatToBf16(value);
          }
        }
    auto table = b.allocateBuffer(2 * sizeof(SplashKvPage),
                                  metal::BufferStorage::Shared, {});
    auto *entries = static_cast<SplashKvPage *>(table.contents());
    entries[0] = splash_kv_page_entry(extent.gpuAddress(), 0);
    entries[1] = splash_kv_page_entry(extent.gpuAddress(), 1);
    std::vector<float> qValues(4 * 256);
    for (uint32_t h = 0; h < 4; ++h)
      qValues[h * 256] = 16;
    auto q = values(b, qValues), gate = values(b, std::vector<float>(4 * 256));
    auto out = values(b, std::vector<float>(4 * 256));
    auto selected =
        b.allocateBuffer(4 * sizeof(int32_t), metal::BufferStorage::Shared, {});
    const int32_t ids[] = {0, 2, 33, -1};
    std::memcpy(selected.contents(), ids, sizeof(ids));
    CommandGraph graph;
    FlashNext::addQsaAttention(graph, q, gate, table, selected, out,
                               splash_kv_layer(pages, data, scale, 1), 4, 1, 4,
                               2, format);
    static_cast<void>(b.submitCommandAsync(graph.dispatches()).wait());
    const double e = std::exp(1.0),
                 weighted = (2 + 4 * e + 6 * e * e) / (1 + e + e * e) / 2;
    std::vector<double> expected(4 * 256);
    for (uint32_t h = 0; h < 4; ++h)
      for (uint32_t c = 0; c < 256; ++c)
        expected[h * 256 + c] = h < 2 ? weighted : -weighted;
    check(out, expected, "selected paged attention ignores excluded values");
  }
}
void wideAndChunked(metal::MetalBackend &b) {
  constexpr uint32_t width = 2560, streams = 4, rows = 3;
  std::vector<float> v(width * streams * rows), g(width * streams, 1);
  for (size_t i = 0; i < v.size(); ++i)
    v[i] = float(int(i % 16) - 8);
  auto input = values(b, v), gamma = floats(b, g),
       out = values(b, std::vector<float>(v.size()));
  CommandGraph rms;
  FlashNext::addGroupedRms(rms, input, {gamma, true}, out, width, streams,
                           rows);
  static_cast<void>(b.submitCommandAsync(rms.dispatches()).wait());
  std::vector<double> expected;
  for (float x : v)
    expected.push_back(x / std::sqrt(21.5 + 1e-6));
  check(out, expected, "production-width grouped RMS");
  const uint32_t cw = 4, n = 13, history = 9;
  std::vector<float> norm(n * cw), direct(n * cw, 1), old(history * cw),
      weights(cw * 4);
  for (size_t i = 0; i < norm.size(); ++i)
    norm[i] = float(int(i % 7) - 3) * 0.25f;
  for (size_t i = 0; i < old.size(); ++i)
    old[i] = float(i % 3) * 0.5f;
  for (size_t i = 0; i < weights.size(); ++i)
    weights[i] = float(int(i % 4) - 1) * 0.25f;
  auto normalized = values(b, norm), gated = values(b, direct),
       w = floats(b, weights), state = values(b, old);
  auto single = values(b, std::vector<float>(n * cw)),
       finalState = values(b, std::vector<float>(history * cw));
  CommandGraph full;
  FlashNext::addPleConv(full, gated, normalized, state, w, single, finalState,
                        cw, 1, n, 4, 3);
  static_cast<void>(b.submitCommandAsync(full.dispatches()).wait());
  auto chunkState = values(b, old),
       chunks = values(b, std::vector<float>(n * cw));
  uint32_t begin = 0;
  for (uint32_t count : {2u, 1u, 4u, 6u}) {
    CommandGraph graph;
    FlashNext::addPleConv(graph, b.view(gated, begin * cw * 2, count * cw * 2),
                          b.view(normalized, begin * cw * 2, count * cw * 2),
                          chunkState, w,
                          b.view(chunks, begin * cw * 2, count * cw * 2),
                          chunkState, cw, 1, count, 4, 3);
    static_cast<void>(b.submitCommandAsync(graph.dispatches()).wait());
    begin += count;
  }
  test::require(std::memcmp(single.contents(), chunks.contents(), n * cw * 2) ==
                    0,
                "PLE convolution depends on chunking");
  test::require(std::memcmp(finalState.contents(), chunkState.contents(),
                            history * cw * 2) == 0,
                "PLE aliased history commit differs");
}
void productionQsa(metal::MetalBackend &b) {
  constexpr uint32_t dim = 128, heads = 4, blocks = 600, rows = 3, ratio = 4,
                     budget = 2048, stride = 2051;
  const uint32_t qpositions[] = {0, 2050, 2398}, primes[] = {29, 31, 37, 43};
  std::vector<float> keys(blocks * dim), query(rows * heads * dim);
  for (uint32_t block = 0; block < blocks; ++block)
    for (uint32_t h = 0; h < heads; ++h)
      keys[block * dim + h] = float(int(block % primes[h]) - 10) / 4;
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t h = 0; h < heads; ++h)
      query[(row * heads + h) * dim + h] = 1;
  auto k = values(b, keys), q = values(b, query),
       p = positions(b, {0, 2050, 2398});
  auto scores = floats(b, std::vector<float>(rows * blocks));
  auto selected =
      b.allocateBuffer(rows * stride * 4, metal::BufferStorage::Shared, {});
  CommandGraph graph;
  FlashNext::addQsaScores(graph, q, k, p, scores, dim, heads, rows, blocks,
                          ratio);
  FlashNext::addQsaSelect(graph, scores, p, selected, rows, blocks, ratio,
                          budget);
  static_cast<void>(b.submitCommandAsync(graph.dispatches()).wait());
  const auto *actual = static_cast<const float *>(scores.contents());
  const auto *ids = static_cast<const int32_t *>(selected.contents());
  for (uint32_t row = 0; row < rows; ++row) {
    const uint32_t complete = (qpositions[row] + 1) / ratio;
    std::vector<std::pair<double, uint32_t>> ranked;
    for (uint32_t block = 0; block < blocks; ++block) {
      if (block >= complete) {
        test::require(std::isinf(actual[row * blocks + block]) &&
                          actual[row * blocks + block] < 0,
                      "future QSA score");
        continue;
      }
      double score = 0;
      for (uint32_t h = 0; h < heads; ++h)
        score += std::max(0.0, double(keys[block * dim + h]));
      score /= std::sqrt(128.0);
      test::require(std::abs(score - actual[row * blocks + block]) < 1e-5,
                    "production QSA score parity");
      ranked.push_back({score, block});
    }
    std::sort(ranked.begin(), ranked.end(), [](auto a, auto c) {
      return a.first != c.first ? a.first > c.first : a.second < c.second;
    });
    std::vector<int32_t> expected;
    for (size_t i = 0; i < std::min<size_t>(budget / ratio, ranked.size()); ++i)
      for (uint32_t member = 0; member < ratio; ++member)
        expected.push_back(int32_t(ranked[i].second * ratio + member));
    for (uint32_t token = complete * ratio; token <= qpositions[row]; ++token)
      expected.push_back(int32_t(token));
    expected.resize(stride, -1);
    std::sort(expected.begin(), expected.end());
    std::vector<int32_t> got(ids + row * stride, ids + (row + 1) * stride);
    std::sort(got.begin(), got.end());
    test::require(got == expected, "production QSA budget/tail/topk parity");
  }
}
void bufferContracts(metal::MetalBackend &b) {
  using test::BufferExtent;
  const BufferExtent pleExtents[] = {
      {0, 32, 2, "PLE direct"}, {1, 32, 2, "PLE normalized"},
      {2, 144, 2, "PLE state"}, {3, 128, 4, "PLE conv weights"},
      {4, 32, 2, "PLE output"}, {5, 144, 2, "PLE next state"}};
  test::requireExtents(b, pleExtents, [](CommandGraph &g, const auto &v) {
    FlashNext::addPleConv(g, v[0], v[1], v[2], v[3], v[4], v[5], 4, 2, 2, 4, 3);
  });
  const BufferExtent attentionExtents[] = {
      {0, 4096, 2, "QSA attention query"},
      {1, 4096, 2, "QSA attention gate"},
      {2, 16, 8, "QSA attention page table"},
      {3, 56, 4, "QSA attention selected"},
      {4, 4096, 2, "QSA attention output"}};
  test::requireExtents(b, attentionExtents, [](CommandGraph &g, const auto &v) {
    FlashNext::addQsaAttention(g, v[0], v[1], v[2], v[3], v[4], {2, 0}, 4, 2, 7,
                               2, kv::Format::Int8);
  });
  const BufferExtent gdnExtents[] = {{0, 48, 2, "Flash Next GDN recurrent"},
                                     {1, 56, 2, "Flash Next GDN gate"},
                                     {2, 8, 4, "norm weight"},
                                     {3, 48, 2, "Flash Next GDN hidden"}};
  test::requireExtents(b, gdnExtents, [](CommandGraph &g, const auto &v) {
    FlashNext::addGdnGate(g, v[0], v[1], {v[2], true}, v[3], 2, 2, 6, 2, true,
                          16);
  });
  const BufferExtent ropeExtents[] = {{0, 48, 2, "Flash Next RoPE input"},
                                      {1, 24, 4, "Flash Next RoPE cosine"},
                                      {2, 24, 4, "Flash Next RoPE sine"},
                                      {3, 8, 4, "Flash Next RoPE positions"},
                                      {4, 48, 2, "Flash Next RoPE output"}};
  test::requireExtents(b, ropeExtents, [](CommandGraph &g, const auto &v) {
    FlashNext::addRope(g, v[0], v[1], v[2], v[3], v[4], 6, 2, 2, 4, 3);
  });
}
} // namespace
int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 2)
        throw std::invalid_argument("flash-next-metal METALLIB");
      metal::MetalBackend backend(argv[1]);
      groupedNorm(backend);
      hyperConnection(backend);
      ple(backend);
      qsa(backend);
      gdnAndUtilities(backend);
      selectedAttention(backend);
      wideAndChunked(backend);
      productionQsa(backend);
      bufferContracts(backend);
      std::cout << "Flash Next operators passed\n";
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
      return 1;
    }
  }
}
