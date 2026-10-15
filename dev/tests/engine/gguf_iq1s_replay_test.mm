// Reuse the original fixture generators, planner and buffer/counter oracles.
#define SPLASH_GGUF_PROJECTION_TEST_MAIN gguf_projection_suite_main
#include "gguf_projection_test.mm"
#undef SPLASH_GGUF_PROJECTION_TEST_MAIN

#include <charconv>
#include <fstream>

static_assert(SPLASH_TARGET_VERIFY_ROWS == 8,
              "the captured IQ1S fixture requires DRAFT_BLOCK_SIZE=8");

namespace {
void compareReplayBits(const std::vector<uint16_t> &a,
                       const std::vector<uint16_t> &b, uint32_t rows,
                       uint32_t columns, const std::string &label) {
  const uint64_t count = uint64_t{rows} * columns;
  if (a.size() < count || b.size() < count) {
    fail(label + ": missing output rows");
    return;
  }
  for (uint64_t i = 0; i < count; ++i) {
    if (a[i] == b[i]) continue;
    std::cerr << label << ": row=" << i / columns
              << " column=" << i % columns << " a=0x" << std::hex
              << unsigned(a[i]) << " b=0x" << unsigned(b[i]) << std::dec
              << '\n';
    fail(label + ": strict BF16 equality failed");
    return;
  }
}

void replayIq1s(MetalBackend &backend, const Linear &linear,
               const char *fixture, uint32_t repeats) {
  // This restores the captured 40-format suite state immediately before
  // generating the IQ1S spread gate/up pair; it is not a newly chosen seed.
  std::ifstream state(fixture);
  if (!(state >> rng)) throw std::runtime_error("cannot restore IQ1S RNG state");
  constexpr uint32_t N = 768, K = 2304, lanes = 2;
  constexpr uint32_t columns = N + kPadding;
  const Tensor g = tensor(backend, IQ1S, N, K);
  const Tensor u = tensor(backend, IQ1S, N, K);
  const Projection gate = projection({&g}, columns);
  const Projection up = projection({&u}, columns);
  const auto x = activations(Inputs::Dense, kMaximumRows, K);

  for (const uint32_t splits : {1u, 4u}) {
    const auto workload = decode({columns, K}, lanes, LinearEpilogue::GateUp);
    auto configuration = config(LinearTile::GgufStaged, workload, splits, true);
    const auto twoPasses = Linear::plan(workload, configuration, FloatOutput::BFloat16);
    configuration.oneGateUpPass = true;
    const auto onePass = Linear::plan(workload, configuration, FloatOutput::BFloat16);
    const auto rows = storageRows(x, K, workload.rows, onePass.storageRows());
    Outcome firstTwo, firstOne;
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
      const auto label = "IQ1S 768x2304 S=" + std::to_string(splits) +
                         " L=2 repeat=" + std::to_string(repeat);
      // run() retains the original spread bindings, padding, guard bands and
      // zero-counter assertions. Each call gets fresh guarded scratch.
      const auto twoA = run(backend, linear, twoPasses, up, &gate, rows, {},
                            kPoisonNaN, N, label + " two-A");
      const auto twoB = run(backend, linear, twoPasses, up, &gate, rows, {},
                            kPoisonNaN, N, label + " two-B");
      const auto oneA = run(backend, linear, onePass, up, &gate, rows, {},
                            kPoisonFinite, N, label + " one-A");
      const auto oneB = run(backend, linear, onePass, up, &gate, rows, {},
                            kPoisonFinite, N, label + " one-B");
      compareReplayBits(twoA.output, twoB.output, workload.rows, columns, label + " two/two");
      compareReplayBits(twoA.gate, twoB.gate, workload.rows, columns, label + " gate/gate");
      compareReplayBits(oneA.output, oneB.output, workload.rows, columns, label + " one/one");
      compareReplayBits(oneA.output, twoA.output, workload.rows, columns, label + " one/two");
      compareReplayBits(oneB.output, twoB.output, workload.rows, columns, label + " one-B/two-B");
      if (repeat == 0) {
        firstTwo = twoA;
        firstOne = oneA;
      } else {
        compareReplayBits(twoA.output, firstTwo.output, workload.rows, columns, label + " two/first");
        compareReplayBits(twoA.gate, firstTwo.gate, workload.rows, columns, label + " gate/first");
        compareReplayBits(oneA.output, firstOne.output, workload.rows, columns, label + " one/first");
      }
    }
    section("Frozen IQ1S spread S=" + std::to_string(splits) + " L=2, " +
            std::to_string(repeats) + " repeats");
  }
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 3 && argc != 4) {
      std::cerr << "usage: gguf-iq1s-replay METALLIB RNG_FIXTURE [REPEATS]\n";
      return 2;
    }
    uint32_t repeats = 128;
    if (argc == 4) {
      const std::string_view argument(argv[3]);
      const auto parsed = std::from_chars(argument.data(), argument.data() + argument.size(), repeats);
      if (parsed.ec != std::errc{} || parsed.ptr != argument.data() + argument.size() || !repeats) {
        std::cerr << "REPEATS must be a positive integer\n";
        return 2;
      }
    }
    try {
      MetalBackend backend(argv[1]);
      const Linear linear(backend.capabilities());
      replayIq1s(backend, linear, argv[2], repeats);
    } catch (const std::exception &error) {
      std::cerr << "IQ1S replay: FAIL: " << error.what() << '\n';
      return 1;
    }
    std::cout << "IQ1S replay: " << (failures ? "FAIL" : "PASS")
              << " (" << failures << " failures)\n";
    return failures ? 1 : 0;
  }
}
