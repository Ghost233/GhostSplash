// Correctness only: proposal/retention bounds and lane isolation for each
// compiled block. No timing, warmup measurements or calibration.
#include "TestBuffers.hpp"
#include "ops/Sampling.hpp"
#include "metal/abi/ExecutionGeometry.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

using namespace splash;

namespace {
struct Guarded final {
  metal::MetalBuffer backing, view;
  uint64_t bytes;
  Guarded(metal::MetalBackend &backend, uint64_t size) : bytes(size) {
    backing = test::sharedBuffer(backend, size + 256);
    std::memset(backing.contents(), 0xA5, size + 256);
    view = backend.view(backing, 128, size);
  }
  void intact() const {
    const auto *b = static_cast<const uint8_t *>(backing.contents());
    test::require(std::all_of(b, b + 128, [](uint8_t value) { return value == 0xA5; }) &&
                  std::all_of(b + 128 + bytes, b + 256 + bytes, [](uint8_t value) { return value == 0xA5; }),
                  "acceptance wrote past its compiled block or lane boundary");
  }
};
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: draft-block-acceptance METALLIB");
    metal::MetalBackend backend(argv[1]);
    ops::Sampling sampling(4096);
    constexpr uint32_t block = SPLASH_TARGET_VERIFY_ROWS, proposals = SPLASH_DRAFT_PROPOSAL_TOKENS;
    const auto dummy = test::sharedBuffer(backend, 16);
    for (const uint32_t lanes : {1u, 2u, 4u}) {
      Guarded draft(backend, uint64_t{lanes} * proposals * 4), target(backend, uint64_t{lanes} * block * 4),
              retained(backend, uint64_t{lanes} * 4), accepted(backend, uint64_t{lanes} * 4);
      std::vector<ops::SamplingPolicy> policies(lanes);
      for (const uint32_t rejected : {0u, proposals / 2, proposals}) {
        for (const uint32_t limit : {1u, block}) {
          auto *d = static_cast<uint32_t *>(draft.view.contents());
          auto *t = static_cast<uint32_t *>(target.view.contents());
          for (uint32_t lane = 0; lane < lanes; ++lane) {
            for (uint32_t row = 0; row < block; ++row) t[lane * block + row] = 100 + lane * 100 + row;
            for (uint32_t row = 0; row < proposals; ++row) d[lane * proposals + row] = t[lane * block + row];
            if (rejected < proposals) d[lane * proposals + rejected] += 1000;
          }
          std::vector<uint32_t> maximum(lanes, limit);
          metal::CommandGraph graph;
          sampling.addAcceptance(graph, {draft.view, dummy, dummy, dummy, dummy, target.view,
                                          retained.view, accepted.view}, maximum, policies, 30, 31);
          static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
          for (uint32_t lane = 0; lane < lanes; ++lane) {
            const uint32_t kept = static_cast<const uint32_t *>(retained.view.contents())[lane];
            const uint32_t matches = static_cast<const uint32_t *>(accepted.view.contents())[lane];
            test::require(matches == rejected && matches <= proposals &&
                            kept == std::min(rejected + 1, limit) && kept <= block,
                          "compiled block produced a wrong acceptance or output bound");
          }
          draft.intact(); target.intact(); retained.intact(); accepted.intact();
        }
      }
    }
    std::cout << "draft-block-acceptance: PASS block=" << block << " lanes=1,2,4\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "draft-block-acceptance: FAIL: " << error.what() << '\n';
    return 1;
  }
}
