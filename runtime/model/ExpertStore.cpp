#include "model/ExpertStore.hpp"
#ifdef SPLASH_RUNTIME_INSTRUMENTATION
#include "model/RuntimeInstrumentation.hpp"
#endif
#include "AwakeClock.hpp"
#include <bit>
#include <chrono>

#include "metal/abi/MoE.h"
#include "model/GgufImageLayout.hpp"
#include "model/GgufPreparation.hpp"
#include "model/Model.hpp"
#include "model/SlotRecycler.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightSource.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <exception>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace splash::model {
namespace {

constexpr uint32_t kNoSlot = UINT32_MAX;
// The most bytes one read of a read-ahead reads (Impl::Piece), and the
// readers that read ahead: what the round that stops a read-ahead waits for
// is the pieces its readers took. An M5 Max's SSD reads 460 KiB pieces
// at 7.25 GB/s from four threads and at 7.41 GB/s from eighteen, but each
// then takes 276 us instead of 1223 us.
constexpr uint64_t kAheadPieceBytes = 512 << 10;
constexpr unsigned kAheadReaders = 4;
// Apple silicon's pages: a read the kernel does straight into memory
// (F_NOCACHE without the buffer cache) starts and ends on them in the file
// and in memory. With its ends elsewhere it goes through the buffer cache:
// on an M5 Pro, four threads read 460 KiB pieces at 8.1 GB/s that way and
// at 9.3 GB/s (of the pieces' bytes) widened to whole pages, with 0.04
// instead of 0.09 CPU seconds per GB.
constexpr uint64_t kPageBytes = 16 << 10;
// A decode cycle's most rows: every lane's verify rows.
constexpr uint32_t kCycleRows = ExecutionLimits::maximumBatchWidth * ExecutionLimits::targetVerifyRows;
// A prefill chunk is long when its rows route at least this many times as
// many experts as a block has: routed evenly, that many routes leave about
// e^-3, 5%, of the experts unrouted, so it routes nearly every expert.
constexpr uint64_t kLongChunkRoutes = 3;

// ExpertStoreStats::uses and Impl::stepTimes hold a use per ops::MoePhase,
// at its value.
static_assert(static_cast<size_t>(ops::MoePhase::Prefill) == 0 && static_cast<size_t>(ops::MoePhase::Decode) == 1);

constexpr uint64_t alignUp(uint64_t value, uint64_t alignment) noexcept {
  return (value + alignment - 1) / alignment * alignment;
}

// One projection of an expert of some size: rows x columns in format, its
// native rows rowBytes each, and the bytes of its planes.
struct ProjectionLayout final {
  uint32_t format = 0;
  uint64_t rows = 0;
  uint64_t columns = 0;
  uint64_t rowBytes = 0;
  std::array<uint64_t, 3> planes{}; // plane0, plane1, meta
  [[nodiscard]] uint64_t nativeBytes() const noexcept { return rows * rowBytes; }
  bool operator==(const ProjectionLayout &) const = default;
};
using ExpertLayout = std::array<ProjectionLayout, 3>; // gate, up, down

ExpertLayout expertLayout(const gguf::StreamedExperts &experts, ops::MoeShape shape) {
  const uint64_t intermediate = shape.expertIntermediateSize, hidden = shape.hiddenSize;
  const std::array<uint64_t, 3> rows{intermediate, intermediate, hidden}, columns{hidden, hidden, intermediate};
  ExpertLayout layout;
  for (size_t z = 0; z < layout.size(); ++z) {
    const gguf::Repack &repack = experts.projections[z];
    if (repack.format >= GGUF_FMT_COUNT || repack.rows != shape.experts * rows[z] || repack.columns != columns[z])
      throw std::invalid_argument("streamed experts do not match the MoE shape");
    const QuantFormat &format = kQuantFormats[repack.format];
    const GgufPlaneBytes planes = ggufPlaneBytes(format, rows[z], columns[z]);
    layout[z] = {repack.format, rows[z], columns[z], ggufRowBytes(format, columns[z]),
                 {planes.plane0, planes.plane1, planes.meta}};
  }
  return layout;
}

// The arena bytes of `slots` experts of a layout: each projection's planes
// in a region of their own.
uint64_t arenaBytes(const ExpertLayout &layout, uint64_t slots) noexcept {
  uint64_t bytes = 0;
  for (const ProjectionLayout &projection : layout)
    for (const uint64_t plane : projection.planes) bytes += alignWeightOffset(slots * plane);
  return bytes;
}

uint64_t weightBytes(const ExpertLayout &layout) noexcept {
  uint64_t bytes = 0;
  for (const auto &projection : layout)
    for (const uint64_t plane : projection.planes) bytes += plane;
  return bytes;
}

// Places of experts in the staging: place p's rows of projection z from
// at(p, z) on, in width[z] bytes, projection after projection, place after
// place. A place is a page wider than its expert's rows widened to whole
// pages, so that the rows of a run of consecutive experts in consecutive
// places, read widened to whole pages from the first place on, fit in them
// (Impl::Run), and every place starts on a page.
struct Places final {
  std::array<uint64_t, 3> base{}, width{};
  uint32_t count = 0;

  Places() = default;
  Places(const std::array<uint64_t, 3> &nativeBytes, uint32_t places) : count(places) {
    uint64_t at = 0;
    for (size_t z = 0; z < 3; ++z) {
      base[z] = at;
      width[z] = alignUp(nativeBytes[z], kPageBytes) + kPageBytes;
      at += uint64_t{places} * width[z];
    }
  }
  [[nodiscard]] uint64_t at(uint32_t place, uint32_t z) const noexcept { return base[z] + place * width[z]; }
  [[nodiscard]] uint64_t bytes() const noexcept { return base[2] + uint64_t{count} * width[2]; }
};

// What a store's blocks alone decide (the store and ExpertStore::
// fixedBytes): each block's layout, the layouts and their blocks, and the
// native bytes of each projection of a place, the most of a streamed
// layout's expert (of any layout's when none streams: the staging then only
// reads the sizes of one block, Impl::fill).
struct Geometry final {
  std::vector<uint32_t> layoutOf;
  std::vector<ExpertLayout> layouts;
  std::vector<uint32_t> blocks;
  std::array<uint64_t, 3> nativeBytes{};

  // A layout of several blocks streams; one of one block holds every expert.
  [[nodiscard]] bool streams(uint32_t layout) const noexcept { return blocks[layout] > 1; }
};

Geometry geometryOf(const StreamedBlocks &streamed, ops::MoeShape shape) {
  if (!shape.valid() || streamed.blocks.empty()) throw std::invalid_argument("invalid expert store");
  Geometry g;
  for (const gguf::StreamedExperts &experts : streamed.blocks) {
    const ExpertLayout layout = expertLayout(experts, shape);
    const auto found = std::ranges::find(g.layouts, layout);
    const uint32_t index = static_cast<uint32_t>(found - g.layouts.begin());
    if (found == g.layouts.end()) {
      g.layouts.push_back(layout);
      g.blocks.push_back(0);
    }
    ++g.blocks[index];
    g.layoutOf.push_back(index);
  }
  const bool streaming = std::ranges::any_of(g.blocks, [](uint32_t blocks) { return blocks > 1; });
  for (uint32_t layout = 0; layout < g.layouts.size(); ++layout)
    if (g.streams(layout) || !streaming)
      for (size_t z = 0; z < 3; ++z)
        g.nativeBytes[z] = std::max(g.nativeBytes[z], g.layouts[layout][z].nativeBytes());
  return g;
}

// The staging: the experts of a block a decode cycle's rows may route, and
// at least two places, of which a round of a load that is not a decode
// step's stages one half.
Places stagingOf(const Geometry &g, ops::MoeShape shape) {
  return {g.nativeBytes,
          static_cast<uint32_t>(std::max<uint64_t>(
              2, std::min<uint64_t>(shape.experts, uint64_t{kCycleRows} * shape.expertsPerToken)))};
}

uint64_t fixedBytesOf(const Geometry &g, uint32_t blocks, ops::MoeShape shape, bool resizable = false) {
  const auto charged = [resizable](uint64_t bytes) { return resizable ? (bytes + 16383) / 16384 * 16384 : bytes; };
  uint64_t bytes = 0;
  for (uint64_t part : {stagingOf(g, shape).bytes(), uint64_t{blocks} * shape.experts * sizeof(uint32_t),
                         uint64_t{shape.experts} * sizeof(uint32_t), uint64_t{shape.experts} * 3 * sizeof(uint64_t),
                         uint64_t{4} * sizeof(uint32_t), uint64_t{kCycleRows} * shape.routerWidth() * sizeof(float),
                         uint64_t{kCycleRows} * shape.routesPerToken() * sizeof(float),
                         uint64_t{kCycleRows} * shape.routesPerToken() * sizeof(float)}) bytes += charged(part);
  if (resizable) bytes += charged(uint64_t{blocks} * shape.experts * MOE_SLOT_ADDRESSES * sizeof(uint64_t));
  return bytes;
}

// The smallest store (ExpertStore::minimumBytes): its fixed bytes and an
// arena of each layout of a block's experts, which for a layout of one block
// are all of its experts.
uint32_t chunkSlotsFor(const ExpertLayout &layout, uint32_t experts) {
  uint64_t slots = std::min<uint64_t>(experts,
      std::bit_floor(std::max<uint64_t>(1, (128ULL << 20) / arenaBytes(layout, 1))));
  while (experts % slots) slots /= 2;
  return static_cast<uint32_t>(slots);
}

uint64_t minimumBytesOf(const Geometry &g, uint32_t blocks, ops::MoeShape shape, bool resizable = false) {
  uint64_t bytes = fixedBytesOf(g, blocks, shape, resizable);
  for (const ExpertLayout &layout : g.layouts) {
    const uint64_t chunkSlots = chunkSlotsFor(layout, shape.experts);
    bytes += resizable ? shape.experts / chunkSlots * ((arenaBytes(layout, chunkSlots) + 16383) / 16384 * 16384)
                       : arenaBytes(layout, shape.experts);
  }
  return bytes;
}

} // namespace

struct ExpertStore::Impl {
#ifdef SPLASH_RUNTIME_INSTRUMENTATION
  RuntimeInstrumentation::ExpertObserver observer;
#endif
  StorageGrowthGuard growthGuard;
  uint64_t capacityRevision = 1;
  size_t growthCursor = 0;
  std::unique_ptr<ReadScratchPool> readScratch;
  metal::AllocationAdmission optionalRestore;
  // A run of the rows of projection z of consecutive experts in consecutive
  // places, from staging byte `at` on: read straight from `file` that stores
  // them, `bytes` of it from `offset` on (the rows widened to whole pages),
  // the first expert's rows `inPage` bytes in; or, without a file, built by
  // readRepackRows into `bytes` bytes.
  struct Run final {
    const WeightSource *file = nullptr;
    uint64_t at = 0, offset = 0, bytes = 0, inPage = 0;
  };
  // `count` consecutive experts of block `block` from `expert` on, in
  // consecutive places from `place` on, and each projection's run of them.
  struct Group final {
    uint32_t block = 0, expert = 0, count = 0, place = 0;
    std::array<Run, 3> runs;
  };
  // A read of bytes [from, to) of group `group`'s run of projection z.
  struct Piece final {
    uint32_t group = 0, z = 0;
    uint64_t from = 0, to = 0;
  };
  struct Reads final {
    std::vector<Group> groups;
    std::vector<Piece> pieces;
  };
  // The experts of one layout in one arena (of some of its blocks, when the
  // layout's would exceed the largest buffer): their slots, a streamed
  // size's recycled in its order (SlotRecycler).
  struct Size final {
    ExpertLayout layout;
    uint32_t blocks = 0;
    // Whether its layout is of several blocks (Geometry::streams).
    bool streams = false;
    uint32_t slots = 0;
    // Projection z's plane i from regions[z][i] of the arena on.
    std::array<std::array<uint64_t, 3>, 3> regions{};
    metal::MetalBuffer arena;
    struct Chunk final { metal::MetalBuffer memory; std::array<std::array<uint64_t, 3>, 3> regions; };
    std::vector<Chunk> chunks;
    uint32_t chunkSlots = 0;
    uint64_t addressOffset = 0;
    // The repack of its staged rows into slots.
    MoeExpertLoadParams load{};
    uint32_t loadGroups = 0;
    // Which slot a load's missing expert takes.
    SlotRecycler recycling{0};
    std::vector<uint32_t> ownerBlock, ownerExpert; // kNoSlot when free
  };
  struct Block final {
    uint32_t size = 0;
    gguf::StreamedExperts experts;
    std::vector<uint32_t> slot; // each expert's, kNoSlot when not resident
    // Its router and shared-expert gate (bind), which predict its routes.
    ops::QuantizedSegment router, scalarGate;
  };
  // A step's tiles (ops::MoeLoadPart): the grouping's descriptors and count.
  struct PartTiles final {
    metal::MetalBuffer descriptors, count;
  };

  Impl(metal::MetalBackend &backend, ops::MoeShape shape, Geometry geometry, std::shared_ptr<SsdReaders> readers)
      : backend(backend), shape(shape), geometry(std::move(geometry)), readers(std::move(readers)) {}
  // The readers of a read-ahead read into this store's memory: they end the
  // pieces they took before it goes.
  ~Impl() {
    if (ahead.batch) readers->stop(*ahead.batch);
  }

  // Round `round` of the `rounds` of block `block`'s load (addLoad) of `rows`
  // rows in the tiles of `parts`; `predicted` when the step predicted the
  // next block's routes (predictionRoutes and predictionWeights), from the
  // rows `verifyRows` names when the step routes its verify rows alone
  // (ops::MoeBuffers::verifyRows).
  void load(uint32_t block, ops::MoePhase phase, uint32_t rows, bool predicted, std::span<const PartTiles> parts,
            uint32_t round, uint32_t rounds, const metal::MetalBuffer &verifyRows);
  void route(uint32_t block, std::span<const std::span<const MoeTileDescriptor>> parts, ops::MoePhase phase,
             uint32_t rows, ExpertStoreStats::Use &use);
  // Whether a load of `rows` rows of `phase` is a long prefill chunk's,
  // which routes nearly every expert of its block.
  [[nodiscard]] bool longChunk(ops::MoePhase phase, uint32_t rows) const noexcept {
    return phase == ops::MoePhase::Prefill &&
           uint64_t{rows} * shape.expertsPerToken >= kLongChunkRoutes * shape.experts;
  }
  // The experts a load of `rows` rows may route.
  [[nodiscard]] uint64_t routable(uint32_t rows) const noexcept {
    return std::min<uint64_t>(shape.experts, uint64_t{rows} * shape.expertsPerToken);
  }
  // The places a round of a load of `phase` that may route `routable`
  // experts stages: every place for a decode step's that they hold, else
  // half of them.
  [[nodiscard]] uint32_t roundPlaces(ops::MoePhase phase, uint64_t routable) const noexcept {
    return phase == ops::MoePhase::Decode && routable <= places.count ? places.count : places.count / 2;
  }
  [[nodiscard]] bool streams(uint32_t block) const noexcept { return sizes[blocks[block].size].streams; }
  // Adds to `reads` a group of the experts from `expert` on at the places
  // from `place` on of `staged`, `count` of them, read in pieces of at most
  // pieceBytes (whole pages, or whole rows for a built run), and returns it.
  const Group &addGroup(Reads &reads, uint32_t block, uint32_t expert, uint32_t count, uint32_t place,
                        const Places &staged, uint64_t pieceBytes) const;
  // Where expert k of `group` starts in the staging, its projection z.
  [[nodiscard]] uint64_t expertAt(const Group &group, uint32_t k, uint32_t z) const noexcept {
    const Run &run = group.runs[z];
    return run.at + run.inPage + k * sizes[blocks[group.block].size].layout[z].nativeBytes();
  }
  void readPiece(const Group &group, const Piece &piece) const;
  void stage(uint32_t block, uint32_t capacity, ExpertStoreStats::Use &use);
  std::span<const uint32_t> predictedExperts(uint32_t rows, const metal::MetalBuffer &verifyRows);
  void planAhead(uint32_t block, std::span<const uint32_t> experts, ExpertStoreStats::Use &use);
  void startAhead();
  // The host step after a long chunk's load: block `block`'s experts that
  // are not resident read ahead into the whole staging.
  void readBlockAhead(uint32_t block, ops::MoePhase phase);
  void countAhead(size_t group, size_t from, size_t to);
  void retireAhead();
  void evict(Size &size, uint32_t slot);
  void abandon();
  void fill();
  void publish();
  void resizeArenas(uint64_t memoryBytes, bool allowGrowth = true);
  void refreshAddresses(Size &size);
  void updateArenaStats();
  void prepareRestore(uint64_t targetBytes);
  metal::MetalBuffer loadArena(const Size &size) const {
    return resizable ? backend.view(addresses, size.addressOffset,
        uint64_t{size.blocks} * shape.experts * MOE_SLOT_ADDRESSES * sizeof(uint64_t)) : size.arena;
  }

  metal::MetalBackend &backend;
  const ops::MoeShape shape;
  const Geometry geometry;
  const std::shared_ptr<SsdReaders> readers;
  // What holds the sources open, and their checks (StreamedBlocks).
  std::shared_ptr<const void> files;
  std::vector<std::function<void()>> checks;
  std::vector<Size> sizes;
  std::vector<Block> blocks;
  // The staging's places, and which of them the round being staged holds.
  Places places;
  std::vector<uint8_t> taken;
  // Slot tables (blocks x experts), the staging, a round's jobs' slots and
  // their projections' sources (MoE::addExpertLoad's), and its grid
  // (MTLDispatchThreadgroupsIndirectArguments), all Shared.
  metal::MetalBuffer tables, staging, jobs, sources, grid;
  // A decode step's prediction of the next block's routes (addLoad): its
  // router's fp32 scores of this block's input, then each row's experts and
  // routing weights as the routing selects them (ops::MoE::addRoutes), for
  // up to a cycle's rows; each expert's summed weight over the rows (negative
  // when none) orders the experts it names (predictedExperts).
  metal::MetalBuffer predictionScores, predictionRoutes, predictionWeights;
  std::vector<float> predictionSum;
  std::vector<uint32_t> everyExpert, predictionOrder;
  // A read-ahead (startAhead) of block `block`'s experts, one a group, on the
  // readers (`batch`), until the next round halts it (stage takes what it
  // read) and the next read-ahead retires it (retireAhead); its use counts
  // each group once (`counted`). And the read-ahead a step plans
  // (planAhead), which its `then` starts: block `block`'s experts, most
  // likely first, counted in `use`.
  struct Ahead final {
    uint32_t block = kNoSlot;
    Reads reads;
    std::shared_ptr<SsdReaders::Batch> batch;
    ExpertStoreStats::Use *use = nullptr;
    std::vector<uint8_t> counted;
  } ahead;
  struct Plan final {
    uint32_t block = kNoSlot;
    std::vector<uint32_t> experts;
    ExpertStoreStats::Use *use = nullptr;
  } plan;
  uint64_t arenaTotal = 0;
  bool resizable = false;
  metal::MetalBuffer addresses;

  // A load and release() or restore(), which never overlap (no command is in
  // flight while the store releases or restores its memory), hand the state
  // below between their threads under `mutex`. stats() reads `published`, a
  // copy of the counters each of them leaves under statsMutex, and released()
  // the engine thread's own flag: neither waits for a load's reads.
  mutable std::mutex mutex;
  uint64_t serial = 0;
  // The routed experts of block pendingBlock's load that were not resident
  // and are not staged yet (`waiting`), in expert order, with the slots they
  // claimed, and those the round being staged stages: their slots hold no
  // expert until its repack ran.
  std::vector<std::pair<uint32_t, uint32_t>> pending, staged;
  std::vector<uint8_t> waiting;
  uint32_t pendingBlock = kNoSlot;
  // A round's reads.
  Reads now;
  // The tiles of the load being routed, one span per step it serves.
  std::vector<std::span<const MoeTileDescriptor>> partTiles;
  // The load that last routed each expert of the block being loaded.
  std::vector<uint64_t> routedIn;
  std::string failure;
  ExpertStoreStats counters;
  mutable std::mutex statsMutex;
  ExpertStoreStats published;
  bool released = false;
  uint64_t restoreTargetBytes = 0;
  // Per ops::MoePhase, the round trips of the loads' host steps, which their
  // commands' completions add from Metal's thread.
  struct StepTimes final {
    LatencyHistogram pickup, work, resume;
  };
  std::array<StepTimes, 2> stepTimes;
};

// The expert in a slot, if any, is no longer resident.
void ExpertStore::Impl::evict(Size &size, uint32_t slot) {
  const uint32_t owner = size.ownerBlock[slot];
  if (owner == kNoSlot) return;
  const uint32_t expert = size.ownerExpert[slot];
  blocks[owner].slot[expert] = kNoSlot;
  // A table entry of no resident expert names slot 0, which is inside the
  // arena: the kernels read only routed experts' entries, which a load set,
  // so it is never used.
  static_cast<uint32_t *>(tables.contents())[uint64_t{owner} * shape.experts + expert] = 0;
  size.ownerBlock[slot] = size.ownerExpert[slot] = kNoSlot;
  --counters.resident;
  counters.residentWeightBytes -= weightBytes(size.layout);
}

// Routes block `index`'s load: the experts its tiles name, each once; those
// resident keep their slots, and the others claim one each, in expert order
// (stage reads consecutive experts together).
void ExpertStore::Impl::route(uint32_t index, std::span<const std::span<const MoeTileDescriptor>> parts,
                              ops::MoePhase phase, uint32_t rows, ExpertStoreStats::Use &use) {
  Block &block = blocks[index];
  Size &size = sizes[block.size];
  const bool chunk = longChunk(phase, rows);
  ++serial;
  pendingBlock = index;
  for (std::span<const MoeTileDescriptor> tiles : parts)
    for (const MoeTileDescriptor &tile : tiles) {
      // Past the routed experts' tiles come the shared expert's (id experts).
      const uint32_t expert = tile.expert;
      if (expert >= shape.experts || routedIn[expert] == serial) continue;
      routedIn[expert] = serial;
      ++use.routed;
      if (const uint32_t slot = block.slot[expert]; slot != kNoSlot) {
        size.recycling.touch(slot, serial, chunk);
        continue;
      }
      pending.push_back({expert, kNoSlot});
    }
  // Consecutive experts read together (stage).
  std::ranges::sort(pending);
  uint32_t *table = static_cast<uint32_t *>(tables.contents()) + uint64_t{index} * shape.experts;
  for (auto &[expert, slot] : pending) {
    const uint64_t ghosts = size.recycling.ghostHits();
    slot = size.recycling.claim(uint64_t{index} << 32 | expert, serial, chunk);
    use.ghostHits += size.recycling.ghostHits() - ghosts;
    evict(size, slot);
    size.ownerBlock[slot] = index;
    size.ownerExpert[slot] = expert;
    block.slot[expert] = slot;
    table[expert] = slot;
    waiting[expert] = 1;
    ++counters.resident;
    counters.residentWeightBytes += weightBytes(size.layout);
  }
  use.read += pending.size();
}

const ExpertStore::Impl::Group &ExpertStore::Impl::addGroup(Reads &reads, uint32_t index, uint32_t expert,
                                                            uint32_t count, uint32_t place, const Places &staged,
                                                            uint64_t pieceBytes) const {
  const Block &block = blocks[index];
  Group &group = reads.groups.emplace_back(Group{index, expert, count, place, {}});
  for (uint32_t z = 0; z < 3; ++z) {
    const ProjectionLayout &projection = sizes[block.size].layout[z];
    const uint64_t bytes = count * projection.nativeBytes();
    Run &run = group.runs[z];
    run.at = staged.at(place, z);
    run.bytes = bytes;
    uint64_t piece = pieceBytes / projection.rowBytes * projection.rowBytes;
    if (const auto stored = storedRepackRows(block.experts.projections[z], uint64_t{expert} * projection.rows,
                                             uint64_t{count} * projection.rows)) {
      const uint64_t start = stored->offset / kPageBytes * kPageBytes;
      run.file = stored->file;
      run.offset = start;
      run.inPage = stored->offset - start;
      run.bytes = std::min(alignUp(stored->offset + bytes, kPageBytes), stored->file->bytes()) - start;
      piece = pieceBytes;
    }
    piece = std::max(piece, projection.rowBytes);
    for (uint64_t from = 0; from < run.bytes; from += piece)
      reads.pieces.push_back(
          {static_cast<uint32_t>(reads.groups.size() - 1), z, from, std::min(from + piece, run.bytes)});
  }
  return group;
}

void ExpertStore::Impl::readPiece(const Group &group, const Piece &piece) const {
  const Run &run = group.runs[piece.z];
  const std::span<uint8_t> to{static_cast<uint8_t *>(staging.contents()) + run.at + piece.from, piece.to - piece.from};
  if (run.file) return run.file->readFile(run.offset + piece.from, to);
  const ProjectionLayout &projection = sizes[blocks[group.block].size].layout[piece.z];
  readRepackRows(blocks[group.block].experts.projections[piece.z],
                 uint64_t{group.expert} * projection.rows + piece.from / projection.rowBytes,
                 to.size() / projection.rowBytes, to.data(), readScratch.get());
}

// Stages a round of block `index`'s load: first the experts it routes that
// the read-ahead began, where they are, reading what the readers had not or
// failed to; then the others waiting, in expert order, in the lowest places
// left, up to `capacity` in all, each projection's runs of consecutive
// experts in consecutive places in reads of at most kLoadStepBytes. It
// writes their repack's jobs and grid.
void ExpertStore::Impl::stage(uint32_t index, uint32_t capacity, ExpertStoreStats::Use &use) {
  std::ranges::fill(taken, uint8_t{0});
  now.groups.clear();
  now.pieces.clear();
  auto *jobSlots = static_cast<uint32_t *>(jobs.contents());
  auto *jobSources = static_cast<uint64_t *>(sources.contents());
  const auto addJob = [&](const Group &group, uint32_t k) {
    const uint32_t expert = group.expert + k;
    const size_t job = staged.size();
    jobSlots[job] = blocks[index].slot[expert];
    for (uint32_t z = 0; z < 3; ++z) jobSources[3 * job + z] = expertAt(group, k, z);
    staged.push_back({expert, jobSlots[job]});
    waiting[expert] = 0;
  };
  // A read-ahead's group is one expert, and its pieces follow one another.
  // Of the pieces its readers still read (it is halted), the round waits for
  // those of an expert it stages, a few hundred us at most, and keeps the
  // places of the others.
  const std::vector<Piece> &aheadPieces = ahead.reads.pieces;
  const SsdReaders::Batch *batch = ahead.batch.get();
  const auto reading = [&](size_t piece) { return piece < batch->taken() && !batch->ended(piece); };
  for (size_t piece = 0, end = 0; batch && piece < aheadPieces.size(); piece = end) {
    const uint32_t at = aheadPieces[piece].group;
    const Group &group = ahead.reads.groups[at];
    while (end < aheadPieces.size() && aheadPieces[end].group == at) ++end;
    const bool routed = ahead.block == index && waiting[group.expert];
    bool ended = true;
    for (size_t rest = piece; rest < end; ++rest) {
      if (routed && reading(rest)) {
        const auto started = AwakeClock::now();
        while (reading(rest)) std::this_thread::yield();
        use.readWaitNanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(AwakeClock::now() - started).count();
      }
      ended = ended && !reading(rest);
    }
    if (!ended) {
      taken[group.place] = 1;
      continue;
    }
    countAhead(at, piece, end);
    if (!routed) continue;
    taken[group.place] = 1;
    now.groups.push_back(group);
    bool whole = true;
    for (size_t rest = piece; rest < end; ++rest)
      if (rest >= batch->taken() || batch->failed(rest)) {
        const Piece &unread = aheadPieces[rest];
        now.pieces.push_back({static_cast<uint32_t>(now.groups.size() - 1), unread.z, unread.from, unread.to});
        whole = false;
      }
    use.prefetched += whole;
    addJob(now.groups.back(), 0);
  }
  const uint64_t left = std::ranges::count_if(pending, [&](const auto &entry) { return waiting[entry.first]; });
  const uint64_t due = std::max<uint64_t>(staged.size(), std::min<uint64_t>(capacity, staged.size() + left));
  uint32_t place = 0;
  for (size_t at = 0; at < pending.size() && staged.size() < due;) {
    if (!waiting[pending[at].first]) {
      ++at;
      continue;
    }
    while (place < places.count && taken[place]) ++place;
    if (place == places.count) throw std::logic_error("an expert load's round has fewer places than experts due");
    // Consecutive experts waiting, in consecutive free places, read together.
    uint32_t count = 1;
    while (at + count < pending.size() && staged.size() + count < due &&
           pending[at + count].first == pending[at].first + count && waiting[pending[at + count].first] &&
           place + count < places.count && !taken[place + count])
      ++count;
    const Group &group = addGroup(now, index, pending[at].first, count, place, places, kLoadStepBytes);
    for (uint32_t k = 0; k < count; ++k) {
      taken[place + k] = 1;
      addJob(group, k);
    }
    at += count;
  }
  std::atomic<uint64_t> bytes = 0;
  if (!now.pieces.empty()) {
    const auto started = AwakeClock::now();
    readers->run(now.pieces.size(), [&](size_t task) {
      const Piece &piece = now.pieces[task];
      readPiece(now.groups[piece.group], piece);
      bytes += piece.to - piece.from;
    });
    use.readWaitNanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(AwakeClock::now() - started).count();
  }
  use.readBytes += bytes;
  // What the load read is the sources' as they were opened.
  if (!staged.empty())
    for (const std::function<void()> &check : checks) check();
  std::erase_if(pending, [&](const auto &entry) { return !waiting[entry.first]; });
  auto *threadgroups = static_cast<uint32_t *>(grid.contents());
  threadgroups[0] = sizes[blocks[index].size].loadGroups;
  threadgroups[1] = static_cast<uint32_t>(staged.size());
  threadgroups[2] = 3;
}

// The next block's experts a decode step's prediction names for its `rows`
// rows, or those of them below their lanes' verify rows when the step routes
// them alone, by their summed routing weights, the largest first.
std::span<const uint32_t> ExpertStore::Impl::predictedExperts(uint32_t rows, const metal::MetalBuffer &verifyRows) {
  const auto *selected = static_cast<const uint32_t *>(predictionRoutes.contents());
  const auto *weights = static_cast<const float *>(predictionWeights.contents());
  // The selector wrote them before the step's GPU work ran up to this load.
  const auto *verified = static_cast<const uint32_t *>(verifyRows.contents());
  const uint32_t routes = shape.routesPerToken();
  predictionOrder.clear();
  for (uint64_t row = 0; row < rows; ++row) {
    if (verified && row % SPLASH_TARGET_VERIFY_ROWS >= verified[row / SPLASH_TARGET_VERIFY_ROWS]) continue;
    for (uint32_t route = 0; route < shape.expertsPerToken; ++route) {
      const uint64_t at = row * routes + route;
      const uint32_t expert = selected[at];
      if (expert >= shape.experts) continue;
      float &sum = predictionSum[expert];
      if (sum < 0) {
        sum = 0;
        predictionOrder.push_back(expert);
      }
      sum += weights[at];
    }
  }
  std::ranges::stable_sort(predictionOrder, std::ranges::greater{},
                           [&](uint32_t expert) { return predictionSum[expert]; });
  for (const uint32_t expert : predictionOrder) predictionSum[expert] = -1;
  return predictionOrder;
}

// Plans a read-ahead of block `index`'s experts `experts` (most likely
// first), which the step's `then` starts (startAhead).
void ExpertStore::Impl::planAhead(uint32_t index, std::span<const uint32_t> experts, ExpertStoreStats::Use &use) {
  plan.block = index;
  plan.experts.assign(experts.begin(), experts.end());
  plan.use = &use;
}

// Once the GPU went on (readers that start reading as the step ends hold its
// release back: 35B decode on an M6, 30-60 us a step with four of them),
// retires the last read-ahead and reads the planned experts ahead, in that
// order, into the places the round just staged left free, as many as there
// are. A release forgets the plan. It runs after the step let the GPU go on,
// where a throw would outlive the command, so it does not throw: an error
// becomes the store's failure, which the next load reports.
void ExpertStore::Impl::startAhead() {
  std::lock_guard lock(mutex);
  if (plan.block == kNoSlot) return;
  try {
    retireAhead();
    uint32_t place = 0;
    for (const uint32_t expert : plan.experts) {
      while (place < places.count && taken[place]) ++place;
      if (place == places.count) break;
      // Expert after expert, so that the experts a round finds read whole
      // are the likeliest; a piece starts on a page and so ends on one.
      addGroup(ahead.reads, plan.block, expert, 1, place++, places, kAheadPieceBytes);
    }
    const uint32_t block = std::exchange(plan.block, kNoSlot);
    if (ahead.reads.groups.empty()) return;
    ahead.block = block;
    ahead.use = plan.use;
    ahead.counted.assign(ahead.reads.groups.size(), 0);
    ahead.batch = readers->start(
        ahead.reads.pieces.size(),
        [this](size_t task) {
          const Piece &piece = ahead.reads.pieces[task];
          readPiece(ahead.reads.groups[piece.group], piece);
        },
        kAheadReaders);
  } catch (const std::exception &error) {
    plan.block = kNoSlot;
    if (failure.empty()) failure = std::string("reading experts ahead failed: ") + error.what();
    publish();
  }
}

// Counts group `group` of the read-ahead, its pieces [from, to), which
// ended or were never taken, once: the bytes its readers read, and its
// expert if they read all of it.
void ExpertStore::Impl::countAhead(size_t group, size_t from, size_t to) {
  if (ahead.counted[group]) return;
  ahead.counted[group] = 1;
  const SsdReaders::Batch &batch = *ahead.batch;
  bool whole = true;
  for (size_t p = from; p < to; ++p) {
    const Piece &piece = ahead.reads.pieces[p];
    if (p < batch.taken() && !batch.failed(p)) ahead.use->readBytes += piece.to - piece.from;
    else whole = false;
  }
  ahead.use->readAhead += whole;
}

// Ends the read-ahead once its readers ended the pieces they took; what it
// read that no round counted counts in its use.
void ExpertStore::Impl::retireAhead() {
  if (ahead.batch) {
    readers->stop(*ahead.batch);
    const std::vector<Piece> &pieces = ahead.reads.pieces;
    for (size_t piece = 0, end = 0; piece < pieces.size(); piece = end) {
      while (end < pieces.size() && pieces[end].group == pieces[piece].group) ++end;
      countAhead(pieces[piece].group, piece, end);
    }
    ahead.batch.reset();
  }
  ahead.reads.groups.clear();
  ahead.reads.pieces.clear();
  ahead.block = kNoSlot;
}

// Frees the slots of the experts a failed load had not repacked: those
// waiting and those its last round staged.
void ExpertStore::Impl::abandon() {
  if (pendingBlock == kNoSlot) return;
  Block &block = blocks[pendingBlock];
  Size &size = sizes[block.size];
  uint32_t *table = static_cast<uint32_t *>(tables.contents()) + uint64_t{pendingBlock} * shape.experts;
  for (const std::vector<std::pair<uint32_t, uint32_t>> *list : {&pending, &staged})
    for (const auto &[expert, slot] : *list) {
      waiting[expert] = 0;
      if (slot == kNoSlot || block.slot[expert] != slot) continue;
      block.slot[expert] = kNoSlot;
      table[expert] = 0;
      size.ownerBlock[slot] = size.ownerExpert[slot] = kNoSlot;
      size.recycling.free(slot);
      --counters.resident;
      counters.residentWeightBytes -= weightBytes(size.layout);
    }
  pending.clear();
  staged.clear();
  pendingBlock = kNoSlot;
}

// Reads every expert of each size of one block into its slot, expert e into
// slot e, through the staging, as many a command as it places.
void ExpertStore::Impl::fill() {
  for (uint32_t index = 0; index < blocks.size(); ++index) {
    if (streams(index)) continue;
    Block &block = blocks[index];
    Size &size = sizes[block.size];
    std::array<uint64_t, 3> nativeBytes;
    for (size_t z = 0; z < 3; ++z) nativeBytes[z] = size.layout[z].nativeBytes();
    const uint64_t count = std::min<uint64_t>(shape.experts, staging.sizeBytes() / Places(nativeBytes, 1).bytes());
    if (!count) throw std::invalid_argument("the expert staging holds less than one expert");
    const Places batch(nativeBytes, static_cast<uint32_t>(count));
    uint32_t *table = static_cast<uint32_t *>(tables.contents()) + uint64_t{index} * shape.experts;
    for (uint32_t expert = 0; expert < shape.experts; expert += batch.count) {
      const uint32_t experts = std::min(batch.count, shape.experts - expert);
      now.groups.clear();
      now.pieces.clear();
      const Group &group = addGroup(now, index, expert, experts, 0, batch, kLoadStepBytes);
      readers->run(now.pieces.size(), [&](size_t task) { readPiece(group, now.pieces[task]); });
      for (const std::function<void()> &check : checks) check();
      auto *jobSlots = static_cast<uint32_t *>(jobs.contents());
      auto *jobSources = static_cast<uint64_t *>(sources.contents());
      for (uint32_t k = 0; k < experts; ++k) {
        const uint32_t slot = expert + k;
        jobSlots[k] = slot;
        for (uint32_t z = 0; z < 3; ++z) jobSources[3 * k + z] = expertAt(group, k, z);
        size.ownerBlock[slot] = index;
        size.ownerExpert[slot] = slot;
        block.slot[slot] = slot;
        table[slot] = slot;
      }
      auto *threadgroups = static_cast<uint32_t *>(grid.contents());
      threadgroups[0] = size.loadGroups;
      threadgroups[1] = experts;
      threadgroups[2] = 3;
      metal::CommandGraph graph;
      ops::MoE::addExpertLoad(graph, staging, loadArena(size), jobs, sources, size.load, experts,
                              metal::IndirectGrid{grid, 0});
      static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
    }
    counters.resident += shape.experts;
    counters.residentWeightBytes += uint64_t{shape.experts} * weightBytes(size.layout);
  }
}

// Leaves a copy of the counters and the failure for stats().
void ExpertStore::Impl::publish() {
  counters.capacityWeightBytes = 0;
  counters.allocatedBytes = staging.allocatedBytes() + tables.allocatedBytes() + jobs.allocatedBytes() +
      sources.allocatedBytes() + grid.allocatedBytes() + predictionScores.allocatedBytes() +
      predictionRoutes.allocatedBytes() + predictionWeights.allocatedBytes() + addresses.allocatedBytes();
  for (const auto &size : sizes) {
    counters.capacityWeightBytes += released ? 0 : uint64_t{size.slots} * weightBytes(size.layout);
    counters.allocatedBytes += size.arena.allocatedBytes();
    for (const auto &chunk : size.chunks) counters.allocatedBytes += chunk.memory.allocatedBytes();
  }
  counters.auxiliaryBytes = counters.allocatedBytes - counters.capacityWeightBytes;
  counters.released = released;
  const std::lock_guard lock(statsMutex);
  published = counters;
  published.failure = failure;
}

void ExpertStore::Impl::load(uint32_t index, ops::MoePhase phase, uint32_t rows, bool predicted,
                             std::span<const PartTiles> parts, uint32_t round, uint32_t rounds,
                             const metal::MetalBuffer &verifyRows) {
  std::lock_guard lock(mutex);
  // No job runs until the round's are staged; the last round's ran.
  static_cast<uint32_t *>(grid.contents())[1] = 0;
  staged.clear();
  if (!failure.empty()) throw std::runtime_error("expert store failed earlier: " + failure);
  ExpertStoreStats::Use &use = counters.uses[static_cast<size_t>(phase)];
  try {
    if (round == 0) {
      // A load whose later rounds did not run (its command failed) frees the
      // slots of the experts it had not staged; one that fails before it
      // routes frees none of the last load's (abandon).
      abandon();
      partTiles.clear();
      for (const PartTiles &part : parts) {
        const auto *counted = static_cast<const MoeTileCount *>(part.count.contents());
        const auto *descriptors = static_cast<const MoeTileDescriptor *>(part.descriptors.contents());
        const uint64_t bound = part.descriptors.sizeBytes() / sizeof(MoeTileDescriptor);
        if (!counted || !descriptors || counted->tiles > bound)
          throw std::runtime_error("a MoE block's tiles are out of bounds");
        partTiles.push_back({descriptors, counted->tiles});
      }
      route(index, partTiles, phase, rows, use);
      ++use.loads;
    } else if (pendingBlock != index) {
      throw std::logic_error("an expert load's round follows another block's load");
    } else if (pending.empty()) {
      // An earlier round staged every expert, and the read-ahead of the next
      // block runs on.
      return;
    }
    // The read-ahead halts: its readers take no more pieces.
    if (ahead.batch) readers->halt(*ahead.batch);
    stage(index, roundPlaces(phase, routable(rows)), use);
    // Once the round is staged, what comes next is read ahead while the GPU
    // repacks it: the load's experts still waiting, else the next block's
    // likely experts.
    ahead.block = kNoSlot;
    if (!pending.empty()) {
      if (round + 1 == rounds) throw std::logic_error("an expert load routes more experts than its rounds stage");
      std::vector<uint32_t> waitingExperts;
      for (const auto &entry : pending) waitingExperts.push_back(entry.first);
      planAhead(index, waitingExperts, use);
      return;
    }
    // A long chunk's next block is read ahead by the host step after its
    // last repack (readBlockAhead).
    if (predicted) {
      std::vector<uint32_t> experts;
      for (const uint32_t expert : predictedExperts(rows, verifyRows))
        if (blocks[index + 1].slot[expert] == kNoSlot) experts.push_back(expert);
      planAhead(index + 1, experts, use);
    }
  } catch (const std::exception &error) {
    abandon();
    failure = error.what();
    publish();
    throw;
  }
  publish();
}

void ExpertStore::Impl::readBlockAhead(uint32_t index, ops::MoePhase phase) {
  std::lock_guard lock(mutex);
  if (!failure.empty()) throw std::runtime_error("expert store failed earlier: " + failure);
  // The GPU repacked the load's last round: no place is taken.
  std::ranges::fill(taken, uint8_t{0});
  std::vector<uint32_t> experts;
  for (const uint32_t expert : everyExpert)
    if (blocks[index].slot[expert] == kNoSlot) experts.push_back(expert);
  planAhead(index, experts, counters.uses[static_cast<size_t>(phase)]);
}

ExpertStore::ExpertStore(metal::MetalBackend &backend, StreamedBlocks streamed, ops::MoeShape shape,
                         ExpertStoreOptions options)
    : impl_(std::make_shared<Impl>(backend, shape, geometryOf(streamed, shape), std::move(options.readers))) {
  Impl &impl = *impl_;
  const Geometry &geometry = impl.geometry;
  impl.resizable = options.resizable;
  impl.optionalRestore = std::move(options.optionalRestore);
  if (!options.memoryBytes || !impl.readers) throw std::invalid_argument("invalid expert store");
  impl.files = std::move(streamed.sources);
  impl.checks = std::move(streamed.checks);
  for (uint32_t block = 0; block < streamed.blocks.size(); ++block)
    impl.blocks.push_back(
        {.experts = std::move(streamed.blocks[block]), .slot = std::vector<uint32_t>(shape.experts, kNoSlot)});
  if (options.boundedReadScratch) {
    uint64_t scratch = 0;
    for (const auto &block : impl.blocks)
      for (const auto &projection : block.experts.projections)
        scratch = std::max(scratch, repackReadTemporaryBytes(projection));
    if (scratch) impl.readScratch = std::make_unique<ReadScratchPool>(scratch, impl.readers->readers() + 1,
        [&backend](uint64_t bytes, const std::function<void()> &allocate, const std::function<uint64_t()> &held) {
          return backend.allocateHostMemory(bytes, allocate, false, held);
        });
  }
  const uint32_t blocks = static_cast<uint32_t>(impl.blocks.size());
  if (const uint64_t minimum = minimumBytesOf(geometry, blocks, shape, options.resizable); options.memoryBytes < minimum)
    throw std::invalid_argument("expert memory of " + std::to_string(options.memoryBytes >> 20) +
                                " MiB holds fewer slots than the smallest store (" + std::to_string(minimum >> 20) +
                                " MiB): a block's experts of each size and every expert of a size of one block, "
                                "beside the staging and tables");
  const uint64_t fixed = fixedBytesOf(geometry, blocks, shape, options.resizable);
  const uint64_t arenaBudget = options.memoryBytes > fixed ? options.memoryBytes - fixed : 0;
  const uint64_t largest = options.bufferBytes && options.bufferBytes < backend.capabilities().maxBufferLengthBytes
                               ? options.bufferBytes
                               : backend.capabilities().maxBufferLengthBytes;
  // Every streamed size holds the same share of its experts, but at least a
  // block's (a prefill chunk may route them all) and at most all of them; a
  // size of one block all of its: the largest share the arena holds.
  const auto slotsAt = [&](uint32_t size, double share) {
    const uint64_t all = uint64_t{impl.sizes[size].blocks} * shape.experts;
    if (!impl.sizes[size].streams) return all;
    return std::clamp<uint64_t>(static_cast<uint64_t>(share * double(all)), shape.experts, all);
  };
  const auto bytesAt = [&](double share) {
    uint64_t bytes = 0;
    for (uint32_t size = 0; size < impl.sizes.size(); ++size)
      bytes += arenaBytes(impl.sizes[size].layout, slotsAt(size, share));
    return bytes;
  };
  // Each layout's blocks in parts[layout] sizes of consecutive blocks: one,
  // and one more while a size's arena takes more than the largest buffer.
  std::vector<uint32_t> parts(geometry.layouts.size(), 1), sizeLayout;
  double low = 0.0;
  for (;;) {
    impl.sizes.clear();
    sizeLayout.clear();
    for (uint32_t layout = 0; layout < geometry.layouts.size(); ++layout) {
      std::vector<uint32_t> members;
      for (uint32_t block = 0; block < impl.blocks.size(); ++block)
        if (geometry.layoutOf[block] == layout) members.push_back(block);
      for (uint32_t part = 0; part < parts[layout]; ++part) {
        const size_t from = members.size() * part / parts[layout], to = members.size() * (part + 1) / parts[layout];
        for (size_t member = from; member < to; ++member)
          impl.blocks[members[member]].size = static_cast<uint32_t>(impl.sizes.size());
        impl.sizes.push_back({.layout = geometry.layouts[layout],
                              .blocks = static_cast<uint32_t>(to - from),
                              .streams = geometry.streams(layout)});
        sizeLayout.push_back(layout);
      }
    }
    // Arenas a layout splits into hold a block's experts each.
    if (bytesAt(0.0) > arenaBudget)
      throw std::invalid_argument("expert memory of " + std::to_string(options.memoryBytes >> 20) +
                                  " MiB holds fewer slots (" + std::to_string(bytesAt(0.0) >> 20) +
                                  " MiB) than a block's experts of each size and arena and every expert of a size "
                                  "of one block, beside the staging and tables (" +
                                  std::to_string(fixed >> 20) + " MiB)");
    double high = 1.0;
    low = 0.0;
    if (bytesAt(high) <= arenaBudget) low = high;
    for (int step = 0; step < 64 && low < high; ++step) {
      const double middle = (low + high) / 2;
      (bytesAt(middle) <= arenaBudget ? low : high) = middle;
    }
    uint32_t over = 0;
    while (over < impl.sizes.size() && arenaBytes(impl.sizes[over].layout, slotsAt(over, low)) <= largest) ++over;
    if (options.resizable || over == impl.sizes.size()) break;
    if (impl.sizes[over].blocks == 1)
      throw std::invalid_argument("a block's experts take more than the device's largest buffer");
    ++parts[sizeLayout[over]];
  }
  if (options.resizable) {
    impl.addresses = backend.allocateBuffer((uint64_t{blocks} * shape.experts * MOE_SLOT_ADDRESSES * sizeof(uint64_t) + 16383) / 16384 * 16384,
                                           metal::BufferStorage::Shared, "experts/addresses");
    std::memset(impl.addresses.contents(), 0, impl.addresses.sizeBytes());
  }
  const auto charged = [&](uint64_t bytes) { return options.resizable ? (bytes + 16383) / 16384 * 16384 : bytes; };
  uint64_t addressOffset = 0;
  for (uint32_t index = 0; index < impl.sizes.size(); ++index) {
    Impl::Size &size = impl.sizes[index];
    size.addressOffset = addressOffset;
    addressOffset += uint64_t{size.blocks} * shape.experts * MOE_SLOT_ADDRESSES * sizeof(uint64_t);
    size.slots = options.resizable ? 0 : static_cast<uint32_t>(slotsAt(index, low));
    size.chunkSlots = chunkSlotsFor(size.layout, shape.experts);
    uint64_t region = 0;
    for (size_t z = 0; z < 3; ++z) {
      const ProjectionLayout &projection = size.layout[z];
      for (size_t plane = 0; plane < 3; ++plane) {
        size.regions[z][plane] = region;
        region += alignWeightOffset(uint64_t{size.slots} * projection.planes[plane]);
      }
      size.load.projections[z] = {projection.format | (options.resizable ? MOE_ADDRESSED_SLOTS : 0),
                                  static_cast<uint32_t>(projection.rows),
                                  static_cast<uint32_t>(projection.columns),
                                  static_cast<uint32_t>(projection.rowBytes),
                                  {size.regions[z][0], size.regions[z][1], size.regions[z][2]},
                                  {projection.planes[0], projection.planes[1], projection.planes[2]}};
    }
    size.loadGroups = ops::MoE::expertLoadGroups(size.load);
    if (!options.resizable)
      size.arena = backend.allocateBuffer(region, metal::BufferStorage::Shared,
                                          "experts/arena-" + std::to_string(index));
    impl.arenaTotal += region;
    size.recycling = SlotRecycler(size.slots);
    size.ownerBlock.assign(size.slots, kNoSlot);
    size.ownerExpert.assign(size.slots, kNoSlot);
  }
  impl.places = stagingOf(geometry, shape);
  impl.taken.assign(impl.places.count, 0);
  impl.staging = backend.allocateBuffer(charged(impl.places.bytes()), metal::BufferStorage::Shared, "experts/staging");
  const uint64_t tableBytes = uint64_t{shape.experts} * impl.blocks.size() * sizeof(uint32_t);
  impl.tables = backend.allocateBuffer(charged(tableBytes), metal::BufferStorage::Shared, "experts/slot-tables");
  std::memset(impl.tables.contents(), 0, tableBytes);
  const auto predictionBuffer = [&](uint64_t perRow, const char *label) {
    return backend.allocateBuffer(charged(uint64_t{kCycleRows} * perRow * sizeof(float)),
                                  metal::BufferStorage::Shared, label);
  };
  impl.predictionScores = predictionBuffer(shape.routerWidth(), "experts/prediction-scores");
  impl.predictionRoutes = predictionBuffer(shape.routesPerToken(), "experts/prediction-routes");
  impl.predictionWeights = predictionBuffer(shape.routesPerToken(), "experts/prediction-weights");
  impl.predictionSum.assign(shape.experts, -1);
  impl.everyExpert.resize(shape.experts);
  for (uint32_t expert = 0; expert < shape.experts; ++expert) impl.everyExpert[expert] = expert;
  impl.jobs = backend.allocateBuffer(charged(uint64_t{shape.experts} * sizeof(uint32_t)), metal::BufferStorage::Shared,
                                     "experts/jobs");
  impl.sources = backend.allocateBuffer(charged(uint64_t{shape.experts} * 3 * sizeof(uint64_t)), metal::BufferStorage::Shared,
                                        "experts/sources");
  impl.grid = backend.allocateBuffer(charged(4 * sizeof(uint32_t)), metal::BufferStorage::Shared, "experts/grid");
  std::memset(impl.grid.contents(), 0, 4 * sizeof(uint32_t));
  impl.routedIn.assign(shape.experts, 0);
  impl.waiting.assign(shape.experts, 0);
  if (options.resizable) impl.resizeArenas(options.memoryBytes);
  impl.counters.resizable = options.resizable;
  impl.counters.budgetBytes = options.memoryBytes;
  impl.counters.minimumBytes = fixed;
  for (const auto &size : impl.sizes) {
    impl.counters.weightBytes += uint64_t{size.blocks} * shape.experts * weightBytes(size.layout);
    impl.counters.minimumBytes += options.resizable
        ? shape.experts / size.chunkSlots * alignUp(arenaBytes(size.layout, size.chunkSlots), kPageBytes)
        : arenaBytes(size.layout, shape.experts);
  }
  impl.counters.arenaBytes = impl.arenaTotal;
  impl.counters.stagingBytes = impl.staging.sizeBytes();
  impl.counters.slots = 0;
  for (const Impl::Size &size : impl.sizes) {
    impl.counters.slots += size.slots;
    impl.counters.experts += uint64_t{size.blocks} * shape.experts;
  }
  impl.fill();
  impl.publish();
}

void ExpertStore::Impl::refreshAddresses(Size &size) {
  auto *entries = reinterpret_cast<uint64_t *>(static_cast<uint8_t *>(addresses.contents()) + size.addressOffset);
  std::memset(entries, 0, uint64_t{size.blocks} * shape.experts * MOE_SLOT_ADDRESSES * sizeof(uint64_t));
  for (uint32_t c = 0; c < size.chunks.size(); ++c) {
    const auto &chunk = size.chunks[c];
    const uint64_t base = chunk.memory.gpuAddress();
    for (uint32_t slot = 0; slot < size.chunkSlots; ++slot)
      for (uint32_t z = 0; z < 3; ++z)
        for (uint32_t p = 0; p < 3; ++p)
          entries[(uint64_t{c} * size.chunkSlots + slot) * MOE_SLOT_ADDRESSES + z * 3 + p] =
              base ? base + chunk.regions[z][p] + uint64_t{slot} * size.layout[z].planes[p] : 0;
  }
}

void ExpertStore::Impl::resizeArenas(uint64_t memoryBytes, bool allowGrowth) {
  const uint64_t fixed = fixedBytesOf(geometry, static_cast<uint32_t>(blocks.size()), shape, true);
  const auto chunkBytes = [](const Size &s) { return (arenaBytes(s.layout, s.chunkSlots) + 16383) / 16384 * 16384; };
  const auto slotsFor = [&](const Size &s, double fraction) {
    const uint64_t all = uint64_t{s.blocks} * shape.experts;
    const uint64_t slots = s.streams ? std::clamp<uint64_t>(uint64_t(fraction * double(all)), shape.experts, all) : all;
    const auto rounded = static_cast<uint32_t>(slots / s.chunkSlots * s.chunkSlots);
    return allowGrowth ? rounded : std::min(rounded, s.slots);
  };
  const auto total = [&](double fraction) {
    uint64_t sum = fixed;
    for (const auto &size : sizes) sum += slotsFor(size, fraction) / size.chunkSlots * chunkBytes(size);
    return sum;
  };
  if (total(0) > memoryBytes) throw std::invalid_argument("adaptive expert budget is below its chunked floor");
  double low = 0, high = 1;
  if (total(1) <= memoryBytes) low = 1;
  for (unsigned n = 0; n < 64 && low < high; ++n) {
    const double mid = (low + high) / 2;
    (total(mid) <= memoryBytes ? low : high) = mid;
  }
  // Release all donating chunks before allocating any receiving chunks.
  for (auto &size : sizes) {
    const uint32_t slots = slotsFor(size, low);
    if (slots >= size.slots) continue;
    auto recycling = size.recycling;
    auto ownerBlock = size.ownerBlock, ownerExpert = size.ownerExpert;
    recycling.resize(slots);
    ownerBlock.resize(slots);
    ownerExpert.resize(slots);
    for (uint32_t slot = slots; slot < size.slots; ++slot) evict(size, slot);
    size.recycling = std::move(recycling);
    size.ownerBlock = std::move(ownerBlock);
    size.ownerExpert = std::move(ownerExpert);
    size.chunks.resize(slots / size.chunkSlots);
    size.slots = slots;
    refreshAddresses(size);
  }
  for (auto &size : sizes) {
    const uint32_t slots = slotsFor(size, low);
    if (!allowGrowth || slots <= size.slots) continue;
    // Prepare all allocating metadata off to the side. A failed vector or
    // KeyIndex growth must leave the live recycler and its owner arrays intact.
    auto recycling = size.recycling;
    auto ownerBlock = size.ownerBlock, ownerExpert = size.ownerExpert;
    recycling.resize(slots);
    ownerBlock.resize(slots, kNoSlot);
    ownerExpert.resize(slots, kNoSlot);
    const auto old = size.chunks.size();
    try {
      while (size.chunks.size() < slots / size.chunkSlots) {
        Size::Chunk chunk;
        uint64_t at = 0;
        for (uint32_t z = 0; z < 3; ++z)
          for (uint32_t p = 0; p < 3; ++p) {
            chunk.regions[z][p] = at;
            at += alignWeightOffset(uint64_t{size.chunkSlots} * size.layout[z].planes[p]);
          }
        chunk.memory = backend.allocateBuffer(chunkBytes(size), metal::BufferStorage::Shared,
                                               "experts/chunk", metal::BufferReach::Addressed);
        size.chunks.push_back(std::move(chunk));
      }
      size.recycling = std::move(recycling);
      size.ownerBlock = std::move(ownerBlock);
      size.ownerExpert = std::move(ownerExpert);
      size.slots = slots;
    } catch (...) {
      size.chunks.resize(old);
      refreshAddresses(size);
      throw;
    }
    refreshAddresses(size);
  }
  updateArenaStats();
}

void ExpertStore::Impl::updateArenaStats() {
  arenaTotal = 0;
  counters.slots = 0;
  for (const auto &size : sizes) {
    counters.slots += size.slots;
    for (const auto &chunk : size.chunks) arenaTotal += chunk.memory.allocatedBytes();
  }
  counters.arenaBytes = arenaTotal;
}

uint64_t ExpertStore::hostAllocatedBytes() const noexcept {
  return impl_->readScratch ? impl_->readScratch->allocatedBytes() : 0;
}

uint64_t ExpertStore::allocatedBytes() const {
  const Impl &impl = *impl_;
  uint64_t sum = impl.staging.allocatedBytes() + impl.tables.allocatedBytes() + impl.jobs.allocatedBytes() +
      impl.sources.allocatedBytes() + impl.grid.allocatedBytes() + impl.predictionScores.allocatedBytes() +
      impl.predictionRoutes.allocatedBytes() + impl.predictionWeights.allocatedBytes() + impl.addresses.allocatedBytes();
  for (const auto &size : impl.sizes) {
    sum += size.arena.allocatedBytes();
    for (const auto &chunk : size.chunks) sum += chunk.memory.allocatedBytes();
  }
  return sum;
}

void ExpertStore::resize(uint64_t memoryBytes) {
  if (impl_->growthGuard && memoryBytes > allocatedBytes())
    throw std::invalid_argument("expert receiving requires an authorized single-block plan");
  resize(memoryBytes, !impl_->growthGuard);
}

void ExpertStore::shrink(uint64_t memoryBytes) { resize(memoryBytes, false); }

void ExpertStore::resize(uint64_t memoryBytes, bool allowGrowth) {
  Impl &impl = *impl_;
  const std::lock_guard lock(impl.mutex);
  if (!impl.resizable || impl.released) throw std::logic_error("expert store is not resizable while released or static");
  impl.retireAhead();
  impl.ahead.block = impl.plan.block = kNoSlot;
  impl.abandon();
  ++impl.capacityRevision;
  try { impl.resizeArenas(memoryBytes, allowGrowth); impl.counters.budgetBytes = memoryBytes; }
  catch (...) { impl.updateArenaStats(); impl.publish(); throw; }
  impl.publish();
}

void ExpertStore::setGrowthGuard(StorageGrowthGuard guard) {
  if (!guard) throw std::invalid_argument("expert growth guard is empty");
  impl_->growthGuard = std::move(guard);
}

void ExpertStore::Impl::prepareRestore(uint64_t targetBytes) {
  ++capacityRevision;
  for (auto &size : sizes) {
    const uint32_t slots = size.streams ? shape.experts : size.slots;
    size.chunks.resize(slots / size.chunkSlots);
    size.recycling.resize(slots); size.ownerBlock.resize(slots); size.ownerExpert.resize(slots);
    size.slots = slots; refreshAddresses(size);
  }
  growthCursor = 0;
  updateArenaStats(); counters.budgetBytes = targetBytes;
  restoreTargetBytes = targetBytes;
}

void ExpertStore::prepareRestore(uint64_t targetBytes) {
  const std::lock_guard lock(impl_->mutex);
  if (!impl_->resizable || !impl_->released || targetBytes < impl_->counters.minimumBytes)
    throw std::invalid_argument("expert restore target requires released chunked storage above its floor");
  try { impl_->prepareRestore(targetBytes); }
  catch (const std::bad_alloc &) {
    impl_->publish();
    throw metal::MetalAllocationError("expert restore metadata awaits memory");
  }
  impl_->publish();
}

std::optional<StorageGrowthPlan> ExpertStore::planGrowOne(uint64_t maximumBytes) const {
  auto &impl = *impl_;
  const std::lock_guard lock(impl.mutex);
  if (!impl.resizable || impl.released) return std::nullopt;
  const uint64_t before = allocatedBytes();
  for (size_t offset = 0; offset < impl.sizes.size(); ++offset) {
    const size_t group = (impl.growthCursor + offset) % impl.sizes.size();
    const auto &size = impl.sizes[group];
    if (!size.streams || uint64_t{size.slots} + size.chunkSlots > uint64_t{size.blocks} * impl.shape.experts) continue;
    const uint64_t bytes = (arenaBytes(size.layout, size.chunkSlots) + 16383) / 16384 * 16384;
    if (before > maximumBytes || bytes > maximumBytes - before) continue;
    uint64_t weights = 0;
    for (const auto &projection : size.layout) for (const auto plane : projection.planes) weights += plane * size.chunkSlots;
    return StorageGrowthPlan{impl.capacityRevision, impl.capacityRevision * (impl.sizes.size() + 1) + group + 1,
        before, bytes, bytes, bytes - weights, static_cast<uint32_t>(group)};
  }
  return std::nullopt;
}

StorageGrowthReceipt ExpertStore::growOne(const StorageGrowthPlan &plan) {
  auto &impl = *impl_;
  const std::lock_guard lock(impl.mutex);
  if (!impl.resizable || impl.released || plan.group >= impl.sizes.size() || plan.revision != impl.capacityRevision ||
      plan.candidate != impl.capacityRevision * (impl.sizes.size() + 1) + plan.group + 1 || plan.beforeBytes != allocatedBytes())
    throw std::invalid_argument("expert single-block plan is stale");
  auto &size = impl.sizes[plan.group];
  const uint64_t bytes = (arenaBytes(size.layout, size.chunkSlots) + 16383) / 16384 * 16384;
  uint64_t weights = 0;
  for (const auto &projection : size.layout) for (const auto plane : projection.planes) weights += plane * size.chunkSlots;
  if (!size.streams || uint64_t{size.slots} + size.chunkSlots > uint64_t{size.blocks} * impl.shape.experts ||
      plan.contributionBytes != bytes || plan.peakBytes != bytes || plan.auxiliaryBytes != bytes - weights ||
      (impl.growthGuard && !impl.growthGuard(plan)))
    throw std::invalid_argument("expert single-block plan is not authorized");
  impl.retireAhead(); impl.ahead.block = impl.plan.block = kNoSlot; impl.abandon();
  ++impl.capacityRevision;
  const uint32_t slots = size.slots + size.chunkSlots;
  const auto old = size.chunks.size();
  try {
    // No temporary Metal/host payload: the owner's container metadata stays
    // inside the existing runtime-overhead reserve, outside store contribution.
    auto recycling = size.recycling;
    auto ownerBlock = size.ownerBlock, ownerExpert = size.ownerExpert;
    recycling.resize(slots); ownerBlock.resize(slots, kNoSlot); ownerExpert.resize(slots, kNoSlot);
    Impl::Size::Chunk chunk;
    uint64_t at = 0;
    for (uint32_t z = 0; z < 3; ++z) for (uint32_t p = 0; p < 3; ++p) {
      chunk.regions[z][p] = at;
      at += alignWeightOffset(uint64_t{size.chunkSlots} * size.layout[z].planes[p]);
    }
    chunk.memory = impl.backend.allocateBuffer(bytes, metal::BufferStorage::Shared, "experts/chunk", metal::BufferReach::Addressed);
    size.chunks.push_back(std::move(chunk));
    size.recycling = std::move(recycling); size.ownerBlock = std::move(ownerBlock); size.ownerExpert = std::move(ownerExpert);
    size.slots = slots;
  } catch (...) {
    size.chunks.resize(old); impl.refreshAddresses(size); impl.updateArenaStats(); impl.publish(); throw;
  }
  impl.growthCursor = (plan.group + 1) % impl.sizes.size();
  impl.refreshAddresses(size); impl.updateArenaStats();
  impl.counters.budgetBytes = allocatedBytes(); impl.publish();
  return {allocatedBytes() - plan.beforeBytes, 1};
}

ExpertStore::~ExpertStore() = default;

void ExpertStore::bind(uint32_t block, ops::BlockMoeWeights &weights) {
  Impl &impl = *impl_;
  Impl::Block &streamed = impl.blocks.at(block);
  streamed.router = weights.router;
  streamed.scalarGate = weights.sharedScalarGate;
  const Impl::Size &size = impl.sizes[streamed.size];
  ops::BlockExpertProjection *const projections[] = {&weights.gate, &weights.up, &weights.down};
  for (size_t z = 0; z < 3; ++z) {
    const ProjectionLayout &projection = size.layout[z];
    const auto plane = [&](size_t index) {
      if (impl.resizable) return impl.backend.view(impl.addresses, size.addressOffset + z * 3 * sizeof(uint64_t),
          uint64_t{size.blocks} * impl.shape.experts * MOE_SLOT_ADDRESSES * sizeof(uint64_t) - z * 3 * sizeof(uint64_t));
      const uint64_t bytes = uint64_t{size.slots} * projection.planes[index];
      return bytes ? impl.backend.view(size.arena, size.regions[z][index], bytes) : metal::MetalBuffer{};
    };
    projections[z]->routed =
        ops::QuantizedSegment::planes(projection.format, static_cast<uint32_t>((impl.resizable ? uint64_t{size.blocks} * impl.shape.experts : size.slots) * projection.rows),
                                      static_cast<uint32_t>(projection.columns), plane(0), plane(1), plane(2));
  }
  weights.slotTable = impl.backend.view(impl.tables, uint64_t{block} * impl.shape.experts * sizeof(uint32_t),
                                        uint64_t{impl.shape.experts} * sizeof(uint32_t));
  weights.slots = impl.resizable ? size.blocks * impl.shape.experts : size.slots;
  weights.indirectSlots = impl.resizable;
  // The experts of a size of one block stay in their slots: nothing loads
  // them.
  weights.stream = impl.streams(block) ? this : nullptr;
  weights.streamBlock = block;
}

void ExpertStore::addLoad(metal::CommandGraph &graph, std::span<const ops::MoeLoadPart> parts,
                          uint32_t block) const {
  const Impl &impl = *impl_;
  const Impl::Size &size = impl.sizes[impl.blocks.at(block).size];
  const ops::MoeShape shape = impl.shape;
  if (parts.empty()) throw std::invalid_argument("an expert load serves no step");
  if (!impl.streams(block)) throw std::invalid_argument("a block of a size of one block does not load");
  const ops::MoePhase phase = parts.front().plan->phase();
  uint32_t rows = 0;
  std::vector<Impl::PartTiles> tiles;
  tiles.reserve(parts.size());
  for (const ops::MoeLoadPart &part : parts) {
    if (part.plan->phase() != phase) throw std::invalid_argument("an expert load's steps differ in phase");
    rows += part.plan->rows();
    tiles.push_back({part.buffers->scratch.tileDescriptors, part.buffers->scratch.tileCount});
  }
  const uint64_t routable = impl.routable(rows);
  const uint64_t perRound = impl.roundPlaces(phase, routable);
  const uint32_t rounds = static_cast<uint32_t>((routable + perRound - 1) / perRound);
  // A decode step's load predicts the next block's routes, when its experts
  // stream: that block's routing of this block's input.
  const bool predicted =
      phase == ops::MoePhase::Decode && block + 1 < impl.blocks.size() && impl.streams(block + 1);
  if (predicted) {
    if (parts.size() != 1) throw std::invalid_argument("a decode load serves one step");
    const Impl::Block &next = impl.blocks[block + 1];
    ops::MoE::addRoutes(graph, parts.front().buffers->input, next.router, next.scalarGate, impl.predictionScores,
                        impl.predictionRoutes, impl.predictionWeights, rows, shape,
                        parts.front().plan->configuration().ggufRouterTile);
  }
  // A decode step's loads keep the GPU awake: they come about every
  // millisecond and most end within one, about as long as an idle GPU takes
  // to come back to full speed; a prefill chunk's read for tens of ms.
  const metal::HostStepGpu gpu = phase == ops::MoePhase::Decode ? metal::HostStepGpu::Awake : metal::HostStepGpu::Idle;
  const auto timed = [impl = impl_, phase](const metal::HostStepTiming &timing) {
    Impl::StepTimes &times = impl->stepTimes[static_cast<size_t>(phase)];
    const auto nanoseconds = [](double seconds) { return uint64_t(seconds * 1e9); };
    times.pickup.add(nanoseconds(timing.pickupSeconds));
    times.work.add(nanoseconds(timing.workSeconds));
    times.resume.add(nanoseconds(timing.resumeSeconds));
  };
  // Each step's read-ahead starts once the GPU went on (startAhead).
  const auto startAhead = [impl = impl_] { impl->startAhead(); };
  const metal::MetalBuffer verifyRows = predicted ? parts.front().buffers->verifyRows : metal::MetalBuffer{};
  for (uint32_t round = 0; round < rounds; ++round) {
    graph.host(
        [impl = impl_, block, phase, rows, predicted, tiles, round, rounds, verifyRows] {
          impl->load(block, phase, rows, predicted, tiles, round, rounds, verifyRows);
        },
        timed, gpu, startAhead);
    ops::MoE::addExpertLoad(graph, impl.staging, impl.loadArena(size), impl.jobs, impl.sources, size.load,
                            static_cast<uint32_t>(std::min<uint64_t>(routable, impl.places.count)),
                            metal::IndirectGrid{impl.grid, 0});
  }
#ifdef SPLASH_RUNTIME_INSTRUMENTATION
  if (impl.observer) {
    for (const auto &part : parts) {
      const uint32_t partRows = part.plan->rows();
      const auto buffers = *part.buffers;
      graph.host([owner = impl_, block, phase, partRows, buffers, this] {
        const auto routes = uint64_t{partRows} * owner->shape.routesPerToken();
        owner->observer(*this, {block, partRows, phase,
            {static_cast<const uint16_t *>(buffers.input.contents()), uint64_t{partRows} * owner->shape.hiddenSize},
            {static_cast<const uint32_t *>(buffers.scratch.selectedExperts.contents()), routes},
            {static_cast<const float *>(buffers.scratch.routingWeights.contents()), routes}});
      });
    }
  }
#endif
  // A long chunk routes nearly every expert of the next block too: once the
  // GPU has repacked the last round, the next block's experts are read ahead
  // into the whole staging, while the GPU runs this block's experts and the
  // next block's mixing, as many as the next load would otherwise read
  // with the GPU waiting.
  if (impl.longChunk(phase, rows) && block + 1 < impl.blocks.size() && impl.streams(block + 1))
    graph.host([impl = impl_, block, phase] { impl->readBlockAhead(block + 1, phase); }, timed,
               metal::HostStepGpu::Idle, startAhead);
}

ExpertStoreStats ExpertStore::stats() const {
  ExpertStoreStats result;
  {
    const std::lock_guard lock(impl_->statsMutex);
    result = impl_->published;
  }
  for (size_t use = 0; use < impl_->stepTimes.size(); ++use) {
    const Impl::StepTimes &times = impl_->stepTimes[use];
    result.uses[use].pickup = times.pickup.snapshot();
    result.uses[use].work = times.work.snapshot();
    result.uses[use].resume = times.resume.snapshot();
  }
  return result;
}

bool ExpertStore::released() const noexcept { return impl_->released; }

void ExpertStore::release() {
  std::lock_guard lock(impl_->mutex);
  Impl &impl = *impl_;
  if (impl.released) return;
  ++impl.capacityRevision;
  // A read-ahead in flight writes into the memory released below: its
  // readers end the pieces they took, and it is forgotten.
  impl.retireAhead();
  impl.ahead.block = impl.plan.block = kNoSlot;
  for (Impl::Block &block : impl.blocks) std::ranges::fill(block.slot, kNoSlot);
  std::memset(impl.tables.contents(), 0, impl.tables.sizeBytes());
  for (Impl::Size &size : impl.sizes) {
    if (impl.resizable) {
      for (auto &chunk : size.chunks) impl.backend.releaseMemory(chunk.memory);
      impl.refreshAddresses(size);
    } else impl.backend.releaseMemory(size.arena);
    std::ranges::fill(size.ownerBlock, kNoSlot);
    std::ranges::fill(size.ownerExpert, kNoSlot);
    size.recycling.reset();
  }
  impl.backend.releaseMemory(impl.staging);
  impl.counters.resident = 0;
  impl.counters.residentWeightBytes = 0;
  impl.released = true;
  impl.restoreTargetBytes = 0;
  impl.publish();
}

bool ExpertStore::restore(const metal::AllocationAdmission &admit) {
  std::lock_guard lock(impl_->mutex);
  Impl &impl = *impl_;
  if (!impl.released) throw std::logic_error("expert memory is not released");
  if (impl.growthGuard && impl.resizable && !impl.restoreTargetBytes) {
    try {
      impl.prepareRestore(impl.restoreTargetBytes ? impl.restoreTargetBytes : impl.counters.minimumBytes);
    } catch (const std::bad_alloc &) {
      impl.publish();
      throw metal::MetalAllocationError("minimum expert restore metadata awaits memory");
    }
  }
  if (impl.optionalRestore && impl.resizable) {
    uint64_t required = impl.staging.sizeBytes(), optional = 0;
    for (const auto &size : impl.sizes) {
      const size_t floor = size.streams ? impl.shape.experts / size.chunkSlots : size.chunks.size();
      for (size_t index = 0; index < size.chunks.size(); ++index)
        (index < floor ? required : optional) += size.chunks[index].memory.sizeBytes();
    }
    std::vector<const metal::MetalBuffer *> restored;
    size_t chunks = 1;
    for (const auto &size : impl.sizes) chunks += size.chunks.size();
    try { restored.reserve(chunks); }
    catch (const std::bad_alloc &) {
      throw metal::MetalAllocationError("expert restore rollback metadata awaits memory");
    }
    const auto restore = [&](bool aboveFloor) {
      for (auto &size : impl.sizes) {
        const size_t floor = size.streams ? impl.shape.experts / size.chunkSlots : size.chunks.size();
        for (size_t index = 0; index < size.chunks.size(); ++index) {
          if ((index >= floor) != aboveFloor) continue;
          auto &memory = size.chunks[index].memory;
          impl.backend.restoreMemory(memory);
          restored.push_back(&memory);
        }
      }
      if (!aboveFloor) {
        impl.backend.restoreMemory(impl.staging);
        restored.push_back(&impl.staging);
      }
    };
    try {
      const auto floor = admit(required, [&] { restore(false); });
      if (!floor) throw metal::MetalAllocationError("expert restore floor awaits memory", floor.failure);
      if (optional) {
        const auto extra = impl.optionalRestore(optional, [&] { restore(true); });
        if (!extra) throw metal::MetalAllocationError("optional expert restore awaits memory", extra.failure);
      }
      for (auto &size : impl.sizes) impl.refreshAddresses(size);
      impl.released = false;
      impl.fill();
      impl.updateArenaStats();
      impl.publish();
      return true;
    } catch (...) {
      for (const auto *memory : restored) impl.backend.releaseMemory(*memory);
      for (auto &block : impl.blocks) std::ranges::fill(block.slot, kNoSlot);
      std::memset(impl.tables.contents(), 0, impl.tables.sizeBytes());
      for (auto &size : impl.sizes) {
        std::ranges::fill(size.ownerBlock, kNoSlot); std::ranges::fill(size.ownerExpert, kNoSlot);
        size.recycling.reset(); impl.refreshAddresses(size);
      }
      impl.counters.resident = impl.counters.residentWeightBytes = 0;
      impl.released = true;
      impl.updateArenaStats();
      impl.publish();
      throw;
    }
  }
  uint64_t bytes = impl.staging.sizeBytes();
  for (const Impl::Size &size : impl.sizes) {
    if (impl.resizable) for (const auto &chunk : size.chunks) bytes += chunk.memory.sizeBytes();
    else bytes += size.arena.sizeBytes();
  }
  std::vector<const metal::MetalBuffer *> restored;
  const metal::AllocationResult admitted = admit(bytes, [&] {
    try {
      for (Impl::Size &size : impl.sizes) {
        if (impl.resizable) {
          for (auto &chunk : size.chunks) {
            impl.backend.restoreMemory(chunk.memory);
            restored.push_back(&chunk.memory);
          }
          impl.refreshAddresses(size);
        } else {
          impl.backend.restoreMemory(size.arena);
          restored.push_back(&size.arena);
        }
      }
      impl.backend.restoreMemory(impl.staging);
    } catch (const metal::MetalAllocationError &) {
      // A buffer the driver refused leaves them all released.
      for (const metal::MetalBuffer *buffer : restored) impl.backend.releaseMemory(*buffer);
      throw;
    }
  });
  if (!admitted)
    throw metal::MetalAllocationError(std::string("unable to restore the expert arena: ") +
                                          metal::allocationFailureName(admitted.failure),
                                      admitted.failure);
  impl.released = false;
  impl.fill();
  impl.publish();
  return true;
}

ExpertCacheRange ExpertStore::storageRange(const StreamedBlocks &blocks, ops::MoeShape shape, bool resizable) {
  const Geometry g = geometryOf(blocks, shape);
  const uint32_t count = static_cast<uint32_t>(blocks.blocks.size());
  ExpertCacheRange result;
  result.fixedBytes = fixedBytesOf(g, count, shape, resizable);
  result.minimumBytes = minimumBytesOf(g, count, shape, resizable);
  result.fullBytes = result.fixedBytes;
  for (uint32_t i = 0; i < g.layouts.size(); ++i) {
    const auto &layout = g.layouts[i];
    const uint64_t all = uint64_t{g.blocks[i]} * shape.experts;
    result.weightBytes += all * weightBytes(layout);
    if (resizable) {
      const uint32_t slots = chunkSlotsFor(layout, shape.experts);
      const uint64_t allocation = alignUp(arenaBytes(layout, slots), kPageBytes);
      result.chunks.push_back({slots * weightBytes(layout), allocation, slots,
                               shape.experts / slots, static_cast<uint32_t>(all / slots)});
      result.fullBytes += all / slots * allocation;
    } else result.fullBytes += arenaBytes(layout, all);
  }
  return result;
}

uint64_t ExpertStore::fixedBytes(const StreamedBlocks &blocks, ops::MoeShape shape, bool resizable) {
  return fixedBytesOf(geometryOf(blocks, shape), static_cast<uint32_t>(blocks.blocks.size()), shape, resizable);
}

uint64_t ExpertStore::minimumBytes(const StreamedBlocks &blocks, ops::MoeShape shape, bool resizable) {
  return minimumBytesOf(geometryOf(blocks, shape), static_cast<uint32_t>(blocks.blocks.size()), shape, resizable);
}

metal::MetalBuffer residentExpertSlots(metal::MetalBackend &backend, uint32_t experts) {
  metal::MetalBuffer table =
      backend.allocateBuffer(residentExpertSlotsBytes(experts), metal::BufferStorage::Shared,
                             "experts/resident-slots");
  auto *slot = static_cast<uint32_t *>(table.contents());
  for (uint32_t expert = 0; expert < experts; ++expert) slot[expert] = expert;
  return table;
}

#ifdef SPLASH_RUNTIME_INSTRUMENTATION
void RuntimeInstrumentation::setExpertObserver(ExpertStore &store, ExpertObserver observer) {
  store.impl_->observer = std::move(observer);
}

std::array<RuntimeInstrumentation::ExpertProjection, 3>
RuntimeInstrumentation::expertProjections(const ExpertStore &store, uint32_t index, uint32_t expert) {
  const auto &impl = *store.impl_;
  const auto &block = impl.blocks.at(index);
  const auto &size = impl.sizes[block.size];
  if (expert >= impl.shape.experts) throw std::runtime_error("observed expert ID exceeds its model shape");
  const auto *table = static_cast<const uint32_t *>(impl.tables.contents()) + uint64_t{index} * impl.shape.experts;
  const uint32_t slot = table[expert];
  if (slot >= size.slots || block.slot[expert] != slot ||
      size.ownerBlock[slot] != index || size.ownerExpert[slot] != expert)
    throw std::runtime_error("observed expert has no matching physical slot owner");
  std::array<ExpertProjection, 3> result;
  for (uint32_t z = 0; z < 3; ++z) {
    auto &projection = result[z];
    projection.source = block.experts.projections[z];
    projection.rows = size.layout[z].rows;
    projection.firstRow = uint64_t{expert} * projection.rows;
    for (uint32_t p = 0; p < 3; ++p) {
      const uint64_t bytes = size.layout[z].planes[p];
      if (!bytes) continue;
      const auto &allocation = impl.resizable ? size.chunks.at(slot / size.chunkSlots).memory : size.arena;
      const auto &regions = impl.resizable ? size.chunks.at(slot / size.chunkSlots).regions : size.regions;
      const uint64_t local = impl.resizable ? slot % size.chunkSlots : slot;
      const uint64_t offset = regions[z][p] + local * bytes;
      if (impl.resizable) {
        const auto *addresses = reinterpret_cast<const uint64_t *>(
            static_cast<const uint8_t *>(impl.addresses.contents()) + size.addressOffset);
        if (addresses[uint64_t{slot} * MOE_SLOT_ADDRESSES + z * 3 + p] != allocation.gpuAddress() + offset)
          throw std::runtime_error("observed expert address differs from its owning Metal allocation");
      }
      projection.planes[p] = impl.backend.view(allocation, offset, bytes);
    }
  }
  return result;
}
#endif

} // namespace splash::model
