#pragma once

// What a store of streamed routed experts reports (ExpertStore::stats,
// /status), apart from the store, whose header brings Metal and the MoE
// operators with it.

#include "model/LatencyHistogram.hpp"

#include <cstdint>
#include <string>
#include <optional>
#include <vector>

namespace splash::model {

// Weight bytes include the format's codes, scales and metadata, with no
// arena padding, staging, address/slot tables or other working buffers.
struct ExpertCacheChunk final {
  uint64_t weightBytes = 0, allocationBytes = 0;
  uint32_t slots = 0, minimumChunks = 0, maximumChunks = 0;
};

struct ExpertCacheRange final {
  uint64_t weightBytes = 0, minimumBytes = 0, fullBytes = 0, fixedBytes = 0;
  std::vector<ExpertCacheChunk> chunks;
};

// Metadata is available before loading; runtime-only quantities remain
// unknown until the loader and runtime have established their requirements.
struct ExpertStorageSnapshot final {
  uint64_t weightBytes = 0, residentLayoutBytes = 0, residentSlotTableBytes = 0;
  ExpertCacheRange contiguous, resizable;
  std::string layout = "unloaded";
  bool released = false;
  std::optional<uint64_t> budgetBytes, allocatedBytes, auxiliaryBytes;
  std::optional<uint64_t> capacityWeightBytes, residentWeightBytes, currentFloorBytes;
  std::optional<uint64_t> minimumRequiredBytes;
};

struct ExpertStoreStats final {
  uint64_t weightBytes = 0, capacityWeightBytes = 0, residentWeightBytes = 0;
  uint64_t allocatedBytes = 0, auxiliaryBytes = 0, minimumBytes = 0, budgetBytes = 0;
  bool resizable = false, released = false;
  uint64_t arenaBytes = 0;
  uint64_t stagingBytes = 0;
  uint64_t slots = 0;
  uint64_t experts = 0;  // every block's routed experts
  uint64_t resident = 0; // slots holding an expert
  // Per ops::MoePhase, at its value (uses[0] prefill, uses[1] decode): the
  // blocks loaded, the distinct experts they routed, those not resident and
  // of them those read ahead, the experts read ahead whole (routed or not),
  // the bytes read from the sources (ahead or not, routed or not), and the
  // round trips of the loads' host steps, one per round (metal::
  // HostStepTiming: the GPU's work before a step to the runner taking it up,
  // its work, its end to the GPU's work after it).
  struct Use final {
    uint64_t loads = 0;
    uint64_t routed = 0;
    uint64_t read = 0;
    uint64_t prefetched = 0;
    uint64_t readAhead = 0;
    uint64_t readBytes = 0;
    LatencyStats pickup, work, resume;
    uint64_t ghostHits = 0, readWaitNanoseconds = 0;
  };
  Use uses[2];
  // Why a load failed, which fails every later load too; empty while none
  // has.
  std::string failure;
};

} // namespace splash::model
