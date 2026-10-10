// DFlash draft selector against a direct CPU reference: the sharded top-16
// scan and its reduce must return exactly the sixteen largest logits of every
// proposal row in (value desc, id asc) order, and the codebook walk must pick
// the same tokens as a double-precision evaluation of the same scores. The
// vocabularies cover the production size, an odd size that misaligns the
// 16-byte vectors and leaves shards with only a few tokens, and one wider
// than a single register chunk per thread. The logits are fp32, and their
// order is decided below the bf16 spacing.
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "ops/DraftSelector.hpp"
#include "tuning/LinearNumerics.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using namespace splash::ops;

constexpr uint32_t kRows = SPLASH_DRAFT_QUERY_ROWS;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kCandidates = SPLASH_DRAFT_CANDIDATES;
constexpr uint32_t kRank = SPLASH_DRAFT_SELECTOR_RANK;
constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;

using splash::test::rejects;
using splash::test::require;
using splash::test::requireExtent;

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  float unit() {
    return static_cast<float>(next() & 0xFFFFFF) / 8388608.0F - 1.0F;
  }
  uint32_t next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(state_ >> 33);
  }

private:
  uint64_t state_;
};

MetalBuffer allocate(MetalBackend &backend, uint64_t bytes) {
  MetalBuffer buffer =
      backend.allocateBuffer(bytes, BufferStorage::Shared, "draft selector");
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

MetalBuffer randomBfloat(MetalBackend &backend, uint64_t count, Random &random,
                         float scale) {
  MetalBuffer buffer = allocate(backend, count * sizeof(uint16_t));
  auto *values = static_cast<uint16_t *>(buffer.contents());
  for (uint64_t index = 0; index < count; ++index)
    values[index] = tuning::floatToBf16(random.unit() * scale);
  return buffer;
}

// Row patterns: peaked logits with a few spikes (the production shape),
// uniform noise, heavy ties from a seven-value alphabet with -inf runs, and
// a row with ten finite tokens so -inf tokens fill the tail by id.
enum class Pattern : uint8_t { Peaked, Uniform, Ties, Sparse };

void fillRow(float *row, uint32_t vocabulary, Pattern pattern,
             Random &random) {
  for (uint32_t token = 0; token < vocabulary; ++token) {
    float value = 0.0F;
    switch (pattern) {
    case Pattern::Peaked:
      value = -8.0F + 3.0F * random.unit() * random.unit();
      break;
    case Pattern::Uniform:
      value = 4.0F * random.unit();
      break;
    case Pattern::Ties:
      value = random.next() % 11 == 0
                  ? -INFINITY
                  : static_cast<float>(int(random.next() % 7)) - 3.0F;
      break;
    case Pattern::Sparse:
      value = -INFINITY;
      break;
    }
    row[token] = value;
  }
  if (pattern == Pattern::Peaked) {
    for (uint32_t spike = 0; spike < 40; ++spike)
      row[random.next() % vocabulary] = 4.0F + 6.0F * random.unit();
  }
  if (pattern == Pattern::Sparse) {
    for (uint32_t finite = 0; finite < 10; ++finite)
      row[random.next() % vocabulary] = random.unit();
  }
}

// The row's sixteen largest tokens: value descending, id ascending on ties.
std::vector<uint32_t> referenceTop16(const float *row, uint32_t vocabulary) {
  std::vector<uint32_t> order(vocabulary);
  std::iota(order.begin(), order.end(), 0U);
  const auto beats = [&](uint32_t a, uint32_t b) {
    return row[a] > row[b] || (row[a] == row[b] && a < b);
  };
  const size_t keep = std::min<size_t>(kCandidates, vocabulary);
  std::partial_sort(order.begin(), order.begin() + keep, order.end(), beats);
  order.resize(keep);
  return order;
}

struct Case final {
  uint32_t vocabulary;
  uint32_t lanes;
  bool sampling;
  bool unary = false;
  bool mixed = false;
};

void runCase(MetalBackend &backend, const Case &c) {
  Random random(0x5e1ec7 + uint64_t{c.vocabulary} * 8 + c.lanes * 2 + c.sampling);
  const uint32_t rows = c.lanes * kRows;
  const uint32_t positions = c.lanes * kPositions;
  const auto workspace = DraftSelector::workspace(positions);
  const DraftSelector selector(c.vocabulary);

  MetalBuffer logits = allocate(backend, uint64_t{rows} * c.vocabulary * sizeof(float));
  auto *logitRows = static_cast<float *>(logits.contents());
  const std::array patterns{Pattern::Peaked, Pattern::Uniform, Pattern::Ties,
                            Pattern::Sparse};
  for (uint32_t row = 0; row < rows; ++row)
    fillRow(logitRows + uint64_t{row} * c.vocabulary, c.vocabulary,
            patterns[(row / kRows + row % kRows) % patterns.size()], random);
  const MetalBuffer selectorHidden =
      c.unary ? MetalBuffer{} : randomBfloat(backend, uint64_t{rows} * kRank, random, 0.1F);
  const DraftCodebooks codebooks{
      c.unary ? MetalBuffer{} : randomBfloat(backend, uint64_t{c.vocabulary} * kRank, random, 0.1F),
      c.unary ? MetalBuffer{} : randomBfloat(backend, uint64_t{c.vocabulary} * kRank, random, 0.1F)};
  DraftSelectorBuffers buffers{
      logits,
      allocate(backend, workspace.partialIdsBytes),
      allocate(backend, workspace.partialValuesBytes),
      allocate(backend, workspace.candidatesBytes),
      allocate(backend, workspace.unaryBytes),
      selectorHidden,
      allocate(backend,
               uint64_t{c.lanes} * SPLASH_SAMPLING_UNIFORMS * sizeof(float)),
      allocate(backend, uint64_t{positions} * sizeof(uint32_t)),
      allocate(backend, workspace.proposalProbabilitiesBytes)};
  auto *uniforms = static_cast<float *>(buffers.uniforms.contents());
  for (uint32_t index = 0; index < c.lanes * SPLASH_SAMPLING_UNIFORMS; ++index)
    uniforms[index] = (random.unit() + 1.0F) * 0.5F;
  std::vector<uint32_t> anchors(c.lanes);
  std::vector<SamplingPolicy> policies(c.lanes);
  for (uint32_t lane = 0; lane < c.lanes; ++lane) {
    anchors[lane] = random.next() % c.vocabulary;
    policies[lane] = SamplingPolicy{16, c.sampling && (!c.mixed || lane != 0) ? 0.8F : 0.0F, 1.0F, false};
  }

  if (c.unary && !c.sampling) {
    // Greedy unary selection requires neither random uniforms nor Q
    // probabilities, and never binds selector hidden rows/codebooks.
    buffers.uniforms = {};
    buffers.proposalProbabilities = {};
    if (c.lanes == 1) { buffers.candidates = {}; buffers.unary = {}; }
  }
  CommandGraph graph;
  if (c.unary) selector.addUnary(graph, buffers, anchors, policies);
  else selector.add(graph, buffers, codebooks, anchors, policies);
  require(graph.dispatches().size() == (c.unary && !c.sampling && c.lanes == 1 ? 2u : 3u),
          "draft selector dispatch count changed");
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());

  const auto *candidates =
      static_cast<const uint32_t *>(buffers.candidates.contents());
  const auto *unary = static_cast<const float *>(buffers.unary.contents());
  const auto *tokens =
      static_cast<const uint32_t *>(buffers.proposedTokens.contents());
  const auto *probabilities =
      static_cast<const float *>(buffers.proposalProbabilities.contents());
  const auto *hidden =
      static_cast<const uint16_t *>(selectorHidden.contents());
  const auto *predecessors =
      static_cast<const uint16_t *>(codebooks.predecessor.contents());
  const auto *successors =
      static_cast<const uint16_t *>(codebooks.successor.contents());

  for (uint32_t lane = 0; lane < c.lanes; ++lane) {
    uint32_t predecessor = anchors[lane];
    for (uint32_t position = 0; position < kPositions; ++position) {
      const uint32_t global = lane * kPositions + position;
      const float *row =
          logitRows + (uint64_t{lane} * kRows + position + 1) * c.vocabulary;
      const auto expected = referenceTop16(row, c.vocabulary);
      if (c.unary && !c.sampling) {
        require(tokens[global] == expected.front(), "standard greedy token differs from the full-vocabulary argmax");
        continue;
      }
      for (uint32_t rank = 0; rank < kCandidates; ++rank) {
        const uint32_t id = candidates[global * kCandidates + rank];
        const float value = unary[global * kCandidates + rank];
        if (rank < expected.size()) {
          require(id == expected[rank] && value == row[expected[rank]],
                  "draft top-16 candidates differ from the exact sorted order");
        } else {
          require(id == 0xFFFFFFFFU && value == -INFINITY,
                  "draft top-16 padding lost the empty sentinel");
        }
      }

      // Scores as the kernel defines them, in double.
      std::array<double, kCandidates> scores{};
      for (uint32_t rank = 0; rank < kCandidates; ++rank) {
        const uint32_t candidate =
            std::min(candidates[global * kCandidates + rank], c.vocabulary - 1);
        double edge = 0.0;
        for (uint32_t dim = 0; !c.unary && dim < kRank; ++dim) {
          edge += double(tuning::bf16ToFloat(predecessors[uint64_t{predecessor} * kRank + dim])) *
                  tuning::bf16ToFloat(hidden[(uint64_t{lane} * kRows + position + 1) * kRank + dim]) *
                  tuning::bf16ToFloat(successors[uint64_t{candidate} * kRank + dim]);
        }
        scores[rank] = double(unary[global * kCandidates + rank]) + edge;
      }
      const uint32_t token = tokens[global];
      uint32_t selected = kCandidates;
      for (uint32_t rank = 0; rank < kCandidates; ++rank)
        if (candidates[global * kCandidates + rank] == token)
          selected = selected == kCandidates ? rank : selected;
      require(selected < kCandidates,
              "draft selector proposed a token outside its candidates");
      if (policies[lane].samples()) {
        const double maximum = *std::max_element(scores.begin(), scores.end());
        double sum = 0.0;
        std::array<double, kCandidates> reference{};
        for (uint32_t rank = 0; rank < kCandidates; ++rank) {
          reference[rank] = std::exp((scores[rank] - maximum) / 0.8);
          sum += reference[rank];
        }
        const float uniform = uniforms[lane * SPLASH_SAMPLING_UNIFORMS +
                                       SPLASH_UNIFORM_PROPOSALS + position];
        std::array<double, kCandidates> cumulative{};
        uint32_t expectedSelection = kCandidates - 1;
        for (uint32_t rank = 0; rank < kCandidates; ++rank) {
          const float probability = probabilities[global * kCandidates + rank];
          require(std::fabs(probability - reference[rank] / sum) < 1e-4,
                  "draft selector probabilities diverged from the softmax");
          cumulative[rank] = (rank ? cumulative[rank - 1] : 0.0) + reference[rank] / sum;
          if (expectedSelection == kCandidates - 1 && cumulative[rank] > uniform)
            expectedSelection = rank;
        }
        // A draw within fp32 rounding of the crossed boundary may go either way.
        const double boundary =
            std::fabs(cumulative[std::min(selected, expectedSelection)] - uniform);
        require(selected == expectedSelection || boundary < 1e-5,
                "draft selector drew a different candidate than the reference");
      } else {
        uint32_t expectedSelection = 0;
        for (uint32_t rank = 1; rank < kCandidates; ++rank)
          if (scores[rank] > scores[expectedSelection])
            expectedSelection = rank;
        // fp32 accumulation over 256 products of magnitude 1e-3 stays far
        // below this margin; only an exact tie could legitimately differ.
        require(selected == expectedSelection ||
                    (!c.unary && std::fabs(scores[selected] - scores[expectedSelection]) < 1e-5),
                "draft selector picked a different greedy candidate");
      }
      predecessor = token;
    }
  }
}

struct Guarded final {
  MetalBuffer backing, view;
  uint64_t bytes;
  Guarded(MetalBackend &backend, uint64_t size) : bytes(size) {
    backing = allocate(backend, size + 256);
    std::memset(backing.contents(), 0xA5, size + 256);
    view = backend.view(backing, 128, size);
  }
  void intact() const {
    const auto *data = static_cast<const uint8_t *>(backing.contents());
    require(std::all_of(data, data + 128, [](uint8_t b) { return b == 0xA5; }) &&
                std::all_of(data + 128 + bytes, data + 256 + bytes, [](uint8_t b) { return b == 0xA5; }),
            "standard argmax wrote a buffer guard");
  }
};

uint32_t scalarArgmax(const float *row, uint32_t vocabulary) {
  uint32_t selected = 0;
  for (uint32_t token = 1; token < vocabulary; ++token)
    if (row[token] > row[selected]) selected = token;
  return selected;
}

void greedyUnary(MetalBackend &backend, uint32_t vocabulary, uint32_t lanes) {
  const uint32_t positions = lanes * kPositions;
  Guarded logits(backend, uint64_t{lanes} * kRows * vocabulary * 4),
      ids(backend, uint64_t{positions} * SPLASH_DRAFT_SAMPLING_SHARDS * 4 * (lanes == 1 ? 1 : kCandidates)),
      values(backend, ids.bytes), tokens(backend, positions * 4),
      candidates(backend,uint64_t{positions} * kCandidates * 4), unary(backend,candidates.bytes);
  auto *rows = static_cast<float *>(logits.view.contents());
  std::fill_n(rows, logits.bytes / 4, -INFINITY);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    // The anchor row is deliberately larger than every proposal logit.
    // Selecting rows 0..6 instead of 1..7 must fail the scalar reference.
    rows[uint64_t{lane} * kRows * vocabulary + vocabulary - 1] = 100000.0f;
    for (uint32_t position = 0; position < kPositions; ++position) {
      float *row = rows + (uint64_t{lane} * kRows + position + 1) * vocabulary;
      if (position == kPositions - 1) continue; // all -inf ties select id 0
      const uint32_t low = (lane * 23 + position * 17 + 3) % vocabulary;
      row[low] = 3.125f;
      row[vocabulary - 1] = position & 1 ? std::nextafter(3.125f, INFINITY) : 3.125f;
    }
  }
  const DraftSelectorBuffers buffers{logits.view, ids.view, values.view,
      lanes == 1 ? MetalBuffer{} : candidates.view, lanes == 1 ? MetalBuffer{} : unary.view,
      {}, {}, tokens.view, {}};
  std::vector<uint32_t> anchors(lanes, 0);
  std::vector<SamplingPolicy> policies(lanes);
  const DraftSelector selector(vocabulary);
  CommandGraph graph;
  selector.addUnary(graph, buffers, anchors, policies);
  if (lanes == 1)
    require(graph.dispatches().size() == 2 && graph.dispatches()[0].pipelineName == "draft_select_argmax_sharded" &&
                graph.dispatches()[1].pipelineName == "draft_select_argmax_merge",
            "one greedy unary lane did not use the direct argmax path");
  else
    require(graph.dispatches().size() == 3 && graph.dispatches()[0].pipelineName == "draft_select_top16_sharded" &&
                graph.dispatches()[1].pipelineName == "draft_select_unary_merge",
            "wider greedy unary batch did not preserve the measured top-16 path");
  static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
  const auto *selected = static_cast<const uint32_t *>(tokens.view.contents());
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t position = 0; position < kPositions; ++position)
      require(selected[lane * kPositions + position] ==
                  scalarArgmax(rows + (uint64_t{lane} * kRows + position + 1) * vocabulary, vocabulary),
              "standard greedy selection differs from scalar FP32 argmax/tie order");
  logits.intact(); ids.intact(); values.intact(); tokens.intact(); candidates.intact(); unary.intact();
  // A sampled lane must never enter the greedy fast path, even if the
  // others are greedy. Missing candidate/Q buffers must then be rejected.
  policies.back().temperature = 0.8f;
  CommandGraph mixed;
  rejects([&] { selector.addUnary(mixed, buffers, anchors, policies); },
          lanes == 1 ? "draft selector partial id buffer holds" : "proposal uniform buffer holds",
          "sampled unary policy incorrectly used the greedy-only path");
  require(mixed.empty(), "mixed unary policy encoded work before checking its true Q workspace");
}

void greedyExtents(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003, lanes = 1, positions = lanes * kPositions;
  const DraftSelector selector(vocabulary);
  const std::array<uint32_t, lanes> anchors{};
  const std::array<SamplingPolicy, lanes> policies{};
  const std::array<splash::test::BufferExtent, 4> sizes{{
      {0, uint64_t{lanes} * kRows * vocabulary * 4, 4, "draft logits"},
      {1, uint64_t{positions} * SPLASH_DRAFT_SAMPLING_SHARDS * 4, 4, "draft selector partial id"},
      {2, uint64_t{positions} * SPLASH_DRAFT_SAMPLING_SHARDS * 4, 4, "draft selector partial value"},
      {3, positions * 4, 4, "proposed token"}}};
  splash::test::requireExtents(backend, sizes, [&](CommandGraph &graph, const std::vector<MetalBuffer> &buffer) {
    selector.addUnary(graph, {buffer[0], buffer[1], buffer[2], {}, {}, {}, {}, buffer[3], {}}, anchors, policies);
  });
}

// Each buffer the selector reaches, at its extent and one element short, for
// three lanes of which the second samples: every lane's eight query rows of
// logits and selector hidden rows, the workspaces of its seven positions,
// each codebook's row of every token, and the sampled lanes' proposal
// uniforms and probabilities up to the second lane's.
void bufferExtents(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003, lanes = 3, sampledLanes = 2;
  const DraftSelector selector(vocabulary);
  const auto workspace = DraftSelector::workspace(lanes * kPositions);
  const DraftSelectorBuffers buffers{
      allocate(backend, uint64_t{lanes} * kRows * vocabulary * sizeof(float)),
      allocate(backend, workspace.partialIdsBytes),
      allocate(backend, workspace.partialValuesBytes),
      allocate(backend, workspace.candidatesBytes),
      allocate(backend, workspace.unaryBytes),
      allocate(backend, uint64_t{lanes} * kRows * kRank * 2),
      allocate(backend, uint64_t{lanes} * SPLASH_SAMPLING_UNIFORMS * sizeof(float)),
      allocate(backend, uint64_t{lanes} * kPositions * sizeof(uint32_t)),
      allocate(backend, workspace.proposalProbabilitiesBytes)};
  const DraftCodebooks codebooks{allocate(backend, uint64_t{vocabulary} * kRank * 2),
                                 allocate(backend, uint64_t{vocabulary} * kRank * 2)};
  const std::array<uint32_t, lanes> anchors{1, 2, 3};
  const std::array policies{SamplingPolicy{}, SamplingPolicy{16, 0.8F, 1.0F}, SamplingPolicy{}};
  const uint64_t candidates = uint64_t{lanes} * kPositions * kCandidates;
  for (const auto &[member, bytes, element, what] :
       std::initializer_list<std::tuple<MetalBuffer DraftSelectorBuffers::*, uint64_t, uint64_t, const char *>>{
           {&DraftSelectorBuffers::logits, uint64_t{lanes} * kRows * vocabulary * 4, 4, "draft logits"},
           {&DraftSelectorBuffers::partialIds, candidates * SPLASH_DRAFT_SAMPLING_SHARDS * 4, 4,
            "draft selector partial id"},
           {&DraftSelectorBuffers::partialValues, candidates * (SPLASH_DRAFT_SAMPLING_SHARDS + kCandidates) * 4, 4,
            "draft selector partial value"},
           {&DraftSelectorBuffers::candidates, candidates * 4, 4, "draft candidate"},
           {&DraftSelectorBuffers::unary, candidates * 4, 4, "draft candidate score"},
           {&DraftSelectorBuffers::selectorHidden, uint64_t{lanes} * kRows * kRank * 2, 2, "draft selector hidden"},
           {&DraftSelectorBuffers::proposedTokens, uint64_t{lanes} * kPositions * 4, 4, "proposed token"},
           {&DraftSelectorBuffers::uniforms,
            ((sampledLanes - 1) * SPLASH_SAMPLING_UNIFORMS + SPLASH_UNIFORM_PROPOSALS + kPositions) * 4, 4,
            "proposal uniform"},
           {&DraftSelectorBuffers::proposalProbabilities, uint64_t{sampledLanes} * kPositions * kCandidates * 4, 4,
            "proposal probability"}})
    requireExtent(backend, buffers.*member, bytes, element, what, [&](CommandGraph &graph, const MetalBuffer &buffer) {
      DraftSelectorBuffers changed = buffers;
      changed.*member = buffer;
      selector.add(graph, changed, codebooks, anchors, policies);
    });
  DraftSelectorBuffers unary = buffers;
  unary.selectorHidden = {};
  for (const auto &[member, bytes, element, what] :
       std::initializer_list<std::tuple<MetalBuffer DraftSelectorBuffers::*, uint64_t, uint64_t, const char *>>{
           {&DraftSelectorBuffers::logits, uint64_t{lanes} * kRows * vocabulary * 4, 4, "draft logits"},
           {&DraftSelectorBuffers::partialIds, candidates * SPLASH_DRAFT_SAMPLING_SHARDS * 4, 4,
            "draft selector partial id"},
           {&DraftSelectorBuffers::partialValues, candidates * SPLASH_DRAFT_SAMPLING_SHARDS * 4, 4,
            "draft selector partial value"},
           {&DraftSelectorBuffers::candidates, candidates * 4, 4, "draft candidate"},
           {&DraftSelectorBuffers::unary, candidates * 4, 4, "draft candidate score"},
           {&DraftSelectorBuffers::proposedTokens, uint64_t{lanes} * kPositions * 4, 4, "proposed token"},
           {&DraftSelectorBuffers::uniforms,
            ((sampledLanes - 1) * SPLASH_SAMPLING_UNIFORMS + SPLASH_UNIFORM_PROPOSALS + kPositions) * 4, 4,
            "proposal uniform"},
           {&DraftSelectorBuffers::proposalProbabilities, uint64_t{sampledLanes} * kPositions * kCandidates * 4, 4,
            "proposal probability"}})
    requireExtent(backend, unary.*member, bytes, element, what, [&](CommandGraph &graph, const MetalBuffer &buffer) {
      DraftSelectorBuffers changed = unary;
      changed.*member = buffer;
      selector.addUnary(graph, changed, anchors, policies);
    });
  for (const auto &[member, what] :
       {std::pair{&DraftCodebooks::predecessor, "draft predecessor codebook"},
        std::pair{&DraftCodebooks::successor, "draft successor codebook"}})
    requireExtent(backend, codebooks.*member, uint64_t{vocabulary} * kRank * 2, 2, what,
                  [&](CommandGraph &graph, const MetalBuffer &buffer) {
                    DraftCodebooks changed = codebooks;
                    changed.*member = buffer;
                    selector.add(graph, buffers, changed, anchors, policies);
                  });
}

void invalidRequests(MetalBackend &backend) {
  rejects([] { DraftSelector(0); }, "invalid draft selector vocabulary",
          "a selector over no vocabulary was accepted");
  const DraftSelector selector(1024);
  const auto workspace = DraftSelector::workspace(kPositions);
  const DraftSelectorBuffers buffers{
      allocate(backend, uint64_t{kRows} * 1024 * sizeof(float)),
      allocate(backend, workspace.partialIdsBytes),
      allocate(backend, workspace.partialValuesBytes),
      allocate(backend, workspace.candidatesBytes),
      allocate(backend, workspace.unaryBytes),
      allocate(backend, uint64_t{kRows} * kRank * 2),
      allocate(backend, SPLASH_SAMPLING_UNIFORMS * sizeof(float)),
      allocate(backend, kPositions * sizeof(uint32_t)),
      allocate(backend, workspace.proposalProbabilitiesBytes)};
  const DraftCodebooks codebooks{allocate(backend, uint64_t{1024} * kRank * 2),
                                 allocate(backend, uint64_t{1024} * kRank * 2)};
  const std::array<uint32_t, 2> anchors{1, 2};
  const std::array<SamplingPolicy, 1> policies{SamplingPolicy{}};
  CommandGraph graph;
  rejects([&] { selector.add(graph, buffers, codebooks, anchors, policies); },
          "invalid draft selector batch", "anchors without a policy each were accepted");
  require(graph.empty(), "invalid draft selector request encoded a graph");
  rejects([&] { selector.addUnary(graph, buffers, anchors, policies); },
          "invalid draft selector batch", "unary anchors without a policy each were accepted");
  require(graph.empty(), "invalid unary selector request encoded a graph");
}

// Explicit measurement mode compares the old unary top-16 graph with the
// exact greedy argmax graph in A-B-B-A order, over identical resident logits.
// It reports complete GPU command times and checks output identity every run.
void benchmarkGreedy(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 248320;
  for (const uint32_t lanes : {1u}) {
    const uint32_t positions = lanes * kPositions;
    const auto workspace = DraftSelector::workspace(positions);
    DraftSelectorBuffers buffers{allocate(backend,uint64_t{lanes} * kRows * vocabulary * 4),
        allocate(backend,workspace.partialIdsBytes), allocate(backend,workspace.partialValuesBytes),
        allocate(backend,workspace.candidatesBytes), allocate(backend,workspace.unaryBytes), {},
        allocate(backend,uint64_t{lanes} * SPLASH_SAMPLING_UNIFORMS * 4),
        allocate(backend,uint64_t{positions} * 4), allocate(backend,workspace.proposalProbabilitiesBytes)};
    Random random(8137);
    for (uint32_t row = 0; row < lanes * kRows; ++row)
      fillRow(static_cast<float *>(buffers.logits.contents()) + uint64_t{row} * vocabulary,
              vocabulary, Pattern::Peaked, random);
    std::vector<uint32_t> anchors(lanes,0);
    std::vector<SamplingPolicy> policies(lanes);
    SelectorBatchParams params{}; params.lanes=lanes; params.vocabulary=vocabulary;
    CommandGraph baseline, candidate;
    baseline.add("draft_select_top16_sharded", {buffers.logits,buffers.partialIds,buffers.partialValues},
                 vocabulary, {uint64_t{positions} * SPLASH_DRAFT_SAMPLING_SHARDS,1,1});
    baseline.add("draft_select_unary_merge", {buffers.partialIds,buffers.partialValues,buffers.candidates,buffers.unary},
                 positions, {positions,1,1}, {32,1,1});
    baseline.add("draft_select_unary", {buffers.candidates,buffers.unary,buffers.uniforms,
                 buffers.proposedTokens,buffers.proposalProbabilities}, params, {lanes,1,1}, {1,1,1});
    DraftSelector(vocabulary).addUnary(candidate,buffers,anchors,policies);
    static_cast<void>(backend.submitCommandAsync(baseline.command()).wait());
    const auto *tokens=static_cast<const uint32_t *>(buffers.proposedTokens.contents());
    const std::vector<uint32_t> expected(tokens,tokens+positions);
    static_cast<void>(backend.submitCommandAsync(candidate.command()).wait());
    std::array<double,2> gpu{}; std::array<uint32_t,2> count{};
    for (uint32_t repeat=0;repeat<10;++repeat)
      for (const uint32_t which : {0u,1u,1u,0u}) {
        const auto timing=backend.submitCommandAsync(which ? candidate.command() : baseline.command()).wait();
        gpu[which]+=timing.gpuSeconds; ++count[which];
        require(std::equal(expected.begin(),expected.end(),tokens), "greedy benchmark changed proposal tokens");
      }
    std::cout << "greedy_unary_benchmark lanes=" << lanes << " vocabulary=" << vocabulary
              << " baseline_gpu_ms=" << gpu[0]*1000/count[0] << " candidate_gpu_ms=" << gpu[1]*1000/count[1]
              << " reduction_percent=" << (1-gpu[1]/gpu[0])*100 << " samples=" << count[0]
              << " proposals_identical=true\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2 && (argc != 3 || std::string(argv[2]) != "--benchmark-greedy"))
      throw std::invalid_argument("usage: draft-selector METALLIB [--benchmark-greedy]");
    MetalBackend backend(argv[1]);
    if (argc == 3) { benchmarkGreedy(backend); return 0; }
    invalidRequests(backend);
    bufferExtents(backend);
    greedyExtents(backend);
    for (const uint32_t vocabulary : {1u, 3u, 17u, 1003u, 248320u})
      for (const uint32_t lanes : {1u, 2u, 3u, 4u}) greedyUnary(backend, vocabulary, lanes);
    for (const uint32_t vocabulary : {248320U, 1003U, 270005U}) {
      for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
        runCase(backend, {vocabulary, lanes, false});
        runCase(backend, {vocabulary, lanes, true});
        runCase(backend, {vocabulary, lanes, false, true});
        runCase(backend, {vocabulary, lanes, true, true});
        if (lanes > 1) runCase(backend, {vocabulary, lanes, true, true, true});
      }
    }
    std::cout << "draft_selector_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "draft_selector_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
