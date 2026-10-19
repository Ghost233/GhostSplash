#include "ops/D3.hpp"
#include "LinearNumerics.hpp"
#include "TestChecks.hpp"
#include <cstring>
#include <iostream>
#include <vector>

using namespace splash;
using namespace splash::test;

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  try {
    metal::MetalBackend backend(argv[1]);
    for (uint32_t heads : {8, 16}) {
      constexpr uint32_t n = 128, prefix = 64;
      const uint32_t stride = ops::D3::storageRows(n), kv = heads / 4;
      auto allocate = [&](uint64_t bytes) {
        auto b = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, "d3-prefix-proof");
        std::memset(b.contents(), 0, b.sizeBytes());
        return b;
      };
      auto q = allocate(uint64_t{stride} * heads * 256 * 2);
      auto k = allocate(uint64_t{stride} * kv * 256 * 2);
      auto v = allocate(k.sizeBytes()), transposed = allocate(k.sizeBytes());
      auto out = allocate(q.sizeBytes());
      auto *values = static_cast<uint16_t *>(v.contents());
      const D3AttentionParams params{n, stride, n, 0, heads, kv, 0};
      for (uint32_t suffix : {0, 1}) {
        for (uint32_t h = 0; h < kv; ++h)
          for (uint32_t c = 0; c < 256; ++c)
            for (uint32_t t = prefix; t < n; ++t)
              values[(h * 256 + c) * stride + t] = floatToBf16(float(suffix));
        metal::CommandGraph graph;
        ops::D3::transposeValues(graph, v, transposed, params);
        ops::D3::attention(graph, q, k, transposed, out, params);
        static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
        const auto *actual = static_cast<const uint16_t *>(out.contents());
        for (uint32_t h = 0; h < heads; ++h)
          for (uint32_t t = 0; t < prefix; ++t)
            for (uint32_t c = 0; c < 256; ++c)
              require(actual[(((h / 4) * stride + t) * 4 + h % 4) * 256 + c] ==
                          floatToBf16(float(suffix) * 0.5f),
                      "D3 prefix output must equal exact suffix-dependent mean");
      }
      std::cout << "{\"query_heads\":" << heads << ",\"prefix_tokens\":64,\"suffix_tokens\":64,"
                   "\"prefix_input_unchanged\":true,\"prefix_output_A\":0,\"prefix_output_B\":0.5,"
                   "\"exact_equality\":true,\"status\":\"passed\"}\n";
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
