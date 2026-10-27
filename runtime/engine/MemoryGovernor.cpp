#include "engine/MemoryGovernor.hpp"

#include "engine/MemoryPlan.hpp"
#include "AwakeClock.hpp"

#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>

namespace splash::engine {

namespace {

// What compressing the anonymous pages frees: the share the compressor does
// not keep at its ratio of held to own pages, at most half of them.
uint64_t compressionSavings(const HostMemoryPages &pages) noexcept {
  if (!pages.compressor || pages.compressed / 2 >= pages.compressor) return pages.anonymous / 2;
  if (pages.compressed <= pages.compressor) return 0;
  return pages.anonymous -
         static_cast<uint64_t>(static_cast<unsigned __int128>(pages.anonymous) * pages.compressor / pages.compressed);
}

} // namespace

HostMemory estimateHostMemory(const HostMemoryPages &statistics, uint64_t pageSize,
                              bool compression) noexcept {
  // free_count includes the speculative pages, which external_page_count
  // also counts; wired file pages are in neither.
  if (!pageSize || statistics.speculative > statistics.free) return {};
  const uint64_t maximum = std::numeric_limits<uint64_t>::max();
  uint64_t pages = statistics.free - statistics.speculative;
  for (uint64_t reclaimable : {statistics.fileBacked, statistics.purgeable}) {
    if (reclaimable > maximum - pages) return {};
    pages += reclaimable;
  }
  const uint64_t credit = compression ? compressionSavings(statistics) : 0;
  if (credit > maximum - pages || pages + credit > maximum / pageSize) return {};
  return {(pages + credit) * pageSize, (pages + credit / 2) * pageSize};
}

std::optional<HostMemory> queryHostMemory() noexcept {
  mach_port_t host = mach_host_self();
  vm_size_t pageSize = 0;
  vm_statistics64_data_t statistics{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  kern_return_t pageResult = host_page_size(host, &pageSize);
  kern_return_t statisticsResult = pageResult == KERN_SUCCESS
      ? host_statistics64(host, HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&statistics),
                          &count)
      : pageResult;
  mach_port_deallocate(mach_task_self(), host);
  if (statisticsResult != KERN_SUCCESS || !pageSize) return std::nullopt;

  return estimateHostMemory(
      {.free = statistics.free_count,
       .speculative = statistics.speculative_count,
       .fileBacked = statistics.external_page_count,
       .purgeable = statistics.purgeable_count,
       .anonymous = statistics.internal_page_count,
       .compressor = statistics.compressor_page_count,
       .compressed = statistics.total_uncompressed_pages_in_compressor},
      pageSize, querySystemMemoryPressure().value_or(MemoryPressure::Critical) != MemoryPressure::Critical);
}

std::optional<MemoryPressure> querySystemMemoryPressure() noexcept {
  uint32_t level = 0;
  size_t size = sizeof(level);
  if (sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &size,
                   nullptr, 0) != 0 || size != sizeof(level))
    return std::nullopt;
  switch (level) {
  case DISPATCH_MEMORYPRESSURE_NORMAL: return MemoryPressure::Normal;
  case DISPATCH_MEMORYPRESSURE_WARN: return MemoryPressure::Warning;
  case DISPATCH_MEMORYPRESSURE_CRITICAL: return MemoryPressure::Critical;
  default: return std::nullopt;
  }
}

MemoryGovernor::Reservation::Reservation(MemoryGovernor *owner, uint64_t bytes)
    : owner_(owner), bytes_(bytes) {}

MemoryGovernor::Reservation::~Reservation() { release(); }

MemoryGovernor::Reservation::Reservation(Reservation &&other) noexcept
    : owner_(other.owner_), bytes_(other.bytes_) {
  other.owner_ = nullptr;
  other.bytes_ = 0;
}

void MemoryGovernor::Reservation::release() noexcept {
  if (owner_ && bytes_)
    owner_->release(bytes_);
  owner_ = nullptr;
  bytes_ = 0;
}

MemoryGovernor::MemoryGovernor(metal::MetalBackend &backend, uint64_t limitBytes,
                               uint64_t hostReserveBytes, uint64_t hostSpareReserveBytes,
                               HostMemoryProvider hostMemory, uint64_t untrackedReserveBytes, uint64_t hostHeldBytes)
    : backend_(backend), limitBytes_(limitBytes), hostReserveBytes_(hostReserveBytes),
      hostSpareReserveBytes_(hostSpareReserveBytes), hostMemory_(std::move(hostMemory)),
      untrackedReserveBytes_(untrackedReserveBytes), hostHeldBytes_(hostHeldBytes) {
  if (!limitBytes_) {
    throw std::invalid_argument("memory governor limit must be positive");
  }
  if (!hostReserveBytes_) {
    throw std::invalid_argument("host memory reserve must be positive");
  }
  if (!hostMemory_) {
    throw std::invalid_argument("host memory provider must be present");
  }
  if (chargedBytes(true) > limitBytes_) {
    throw metal::MetalAllocationError(
        "existing Metal allocations exceed memory governor limit",
        metal::AllocationFailure::EngineBudget);
  }
  const std::optional<HostMemory> host = hostMemory_();
  if (!host || host->available <= hostReserveBytes_) {
    throw metal::MetalAllocationError(
        "host available memory does not satisfy the system reserve",
        metal::AllocationFailure::HostPressure);
  }
}

MemoryGovernor::MemoryGovernor(metal::MetalBackend &backend, const EngineMemoryBreakdown &budget,
                               HostMemoryProvider hostMemory)
    : MemoryGovernor(backend, budget.hardBudgetBytes,
                     EngineMemoryPolicy::hostAvailableReserveBytes(budget.physicalMemoryBytes),
                     EngineMemoryPolicy::hostSpareReserveBytes(budget.physicalMemoryBytes), std::move(hostMemory),
                     budget.pipelineReserveBytes + budget.runtimeOverheadReserveBytes, budget.streamedTableBytes) {}

uint64_t
MemoryGovernor::chargedBytes(bool refreshDevice) const noexcept {
  metal::MetalMemoryStats memory = refreshDevice
      ? backend_.refreshMemoryStats()
      : backend_.memoryStats();
  // The backend's buffers are charged in full, the rest of the device's
  // footprint only where it exceeds the untracked reserve.
  const uint64_t limit = std::numeric_limits<uint64_t>::max();
  const uint64_t accounted = untrackedReserveBytes_ <= limit - memory.allocatedBytes
      ? memory.allocatedBytes + untrackedReserveBytes_
      : limit;
  const uint64_t device = std::max(accounted, memory.deviceCurrentAllocatedBytes);
  const auto host = backend_.hostMemoryStats();
  const unsigned __int128 total = static_cast<unsigned __int128>(device) + hostHeldBytes_ + host.fixedBytes + host.temporaryBytes;
  return total > limit ? limit : static_cast<uint64_t>(total);
}

std::optional<HostMemory> MemoryGovernor::sampleHost() const noexcept {
  try {
    return hostMemory_();
  } catch (...) {
    return std::nullopt;
  }
}

uint64_t MemoryGovernor::hostHeadroomBytes(const std::optional<HostMemory> &host,
                                           uint64_t reservedBytes) const noexcept {
  if (!host || host->spare <= hostSpareReserveBytes_)
    return 0;
  const uint64_t spareAfterReserve = host->spare - hostSpareReserveBytes_;
  const uint64_t withheld = reservedBytes + std::min(pendingAutomaticBytes_, UINT64_MAX - reservedBytes);
  return withheld < spareAfterReserve ? spareAfterReserve - withheld : 0;
}

std::optional<MemoryGovernor::Reservation>
MemoryGovernor::tryReserve(uint64_t bytes, metal::AllocationFailure &failure, bool elastic, bool automaticBlock) {
  if (!bytes) {
    throw std::invalid_argument("memory reservation must be positive");
  }
  uint64_t observed = chargedBytes(true);
  bool overflows =
      reservedBytes_ > std::numeric_limits<uint64_t>::max() - bytes;
  uint64_t requested =
      overflows ? std::numeric_limits<uint64_t>::max() : reservedBytes_ + bytes;
  const std::optional<HostMemory> host = sampleHost();
  host_ = hostState(host);
  const MemoryPressure pressure = effectivePressure(host_);
  const uint64_t allocationLimit = limitBytes_ - (elastic ? automaticStorageWithheldBytes_ : 0);
  bool engineFits = !overflows && observed <= allocationLimit &&
                    requested <= allocationLimit - observed;
  // Growth leaves the warning margin free above the host's spare reserve,
  // stops while macOS warns and, while a request waits for memory, waits for
  // the recovery margin once the host has run short, unless a request in
  // service needs it (setServing).
  const uint64_t hostRoom = hostHeadroomBytes(host, 0);
  const bool hostRoomFits =
      requested <= hostRoom && hostRoom - requested >= kHostWarningMarginBytes;
  // A request that only the host headroom refuses waits for host memory
  // while the idle headroom may still clear the margin. Hold host pressure:
  // once a control pass finds the request waiting, the paced reclaim frees
  // toward the recovery margin for it.
  if (engineFits && !serving_ && !hostRoomFits)
    host_.constrained = true;
  const bool hostRefuses =
      pressure == MemoryPressure::Critical ||
      (!serving_ && ((systemPressure_ == MemoryPressure::Warning && !(pressureManaged_ && automaticBlock)) || !hostRoomFits ||
                     growthHeld(host_)));
  if (hostRefuses || !engineFits) {
    // The host's refusal lifts with its pressure, the limit's only once
    // memory is freed: a refusal they share is the host's.
    failure = hostRefuses ? metal::AllocationFailure::HostPressure
                          : metal::AllocationFailure::EngineBudget;
    if (deniedReservations_ != std::numeric_limits<uint64_t>::max()) {
      ++deniedReservations_;
    }
    return std::nullopt;
  }
  reservedBytes_ = requested;
  return Reservation(this, bytes);
}

void MemoryGovernor::setHostHeldBytes(uint64_t bytes) {
  if (bytes > limitBytes_) throw std::invalid_argument("host-held memory exceeds engine limit");
  hostHeldBytes_ = bytes;
}

uint64_t MemoryGovernor::actualBytes() const noexcept {
  const auto host = backend_.hostMemoryStats();
  const unsigned __int128 total = static_cast<unsigned __int128>(backend_.memoryStats().allocatedBytes) +
                                  hostHeldBytes_ + host.fixedBytes + host.temporaryBytes;
  return total > UINT64_MAX ? UINT64_MAX : static_cast<uint64_t>(total);
}

void MemoryGovernor::refreshHostHeld() {
  if (hostHeldProvider_) hostHeldBytes_ = hostHeldProvider_();
}

uint64_t MemoryGovernor::localReservedBytes() const noexcept {
  const uint64_t actual = actualBytes();
  return allocationScope_ ? (actual < allocationPeak_ ? allocationPeak_ - actual : 0) : reservedBytes_;
}

void MemoryGovernor::publishAccounting(std::string) { refreshHostHeld(); }

void MemoryGovernor::setNonBackendHostHeld(std::function<uint64_t()> provider) {
  hostHeldProvider_ = std::move(provider);
  refreshHostHeld();
}

void MemoryGovernor::enablePressureControl(MemoryPolicySources sources) {
  pressureManaged_ = true;
  memoryPolicy_ = std::move(sources);
  observeAutomaticGrowth(sampleHost());
}

uint64_t MemoryGovernor::policyNow() const {
  return memoryPolicy_.milliseconds ? memoryPolicy_.milliseconds() :
      uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(AwakeClock::now().time_since_epoch()).count());
}

void MemoryGovernor::observeAutomaticGrowth(const std::optional<HostMemory> &host) {
  if (!pressureManaged_) return;
  const uint64_t now = policyNow();
  MemoryPolicyObservation observation{
      MemoryPolicyObservation::Pressure(uint8_t(systemPressure_) + 1), pressureSequence_};
  try { if (memoryPolicy_.pressure) observation = memoryPolicy_.pressure(); }
  catch (...) { observation = {}; }
  if (now < policyNow_ || observation.unsafeSequence != pressureSequence_) normalSince_.reset();
  policyNow_ = now; pressureSequence_ = observation.unsafeSequence;
  if (!host) growthReason_ = "telemetry_unavailable";
  else if (observation.pressure == MemoryPolicyObservation::Pressure::Unknown) growthReason_ = "pressure_unknown";
  else if (observation.pressure == MemoryPolicyObservation::Pressure::Critical || systemPressure_ == MemoryPressure::Critical)
    growthReason_ = "system_pressure";
  else if (host->spare < hostSpareReserveBytes_ || host->spare - hostSpareReserveBytes_ < kHostRecoveryMarginBytes)
    growthReason_ = "recovery_reserve";
  else {
    // A new valid post-allocation sample already reflects actual ownership.
    pendingAutomaticBytes_ = 0;
    if (!normalSince_) normalSince_ = now;
    growthReason_ = now - *normalSince_ < 30'000 ? "headroom_observation" :
        now < nextAutomaticAt_ ? "block_cooldown" : "ready";
    return;
  }
  normalSince_.reset();
  if (host) pendingAutomaticBytes_ = 0;
}

metal::AllocationAdmission MemoryGovernor::allocationAdmission(std::string purpose, bool mandatory) {
  return makeAdmission(std::move(purpose), mandatory, std::nullopt);
}

metal::AllocationAdmission MemoryGovernor::makeAdmission(std::string purpose, bool mandatory,
                                                        std::optional<uint64_t> finalStorageGrowth,
                                                        std::optional<AutomaticStorageBlock> block) {
  return [this, purpose = std::move(purpose), mandatory, finalStorageGrowth, block](uint64_t bytes, const std::function<void()> &allocate)
             -> metal::AllocationResult {
    const bool elastic = purpose == "kv_required" || purpose == "state_required";
    const bool automatic = purpose == "expert_optional" || purpose == "ple_optional" ||
                           purpose == "restore_expert_optional";
    const bool contributionPlanned = finalStorageGrowth.has_value();
    const uint64_t contribution = finalStorageGrowth.value_or(bytes);
    const bool retained = elastic || (automatic && !contributionPlanned);
    if (pressureManaged_ && block && allocationScope_)
      return {metal::AllocationFailure::PlanInsufficient, "automatic block requires its own paced reservation"};
    if (pressureManaged_ && automatic && !block) {
      if (!automaticStorageFits(contribution)) return {metal::AllocationFailure::EngineBudget, "automatic store quota exceeded"};
      if (allocationScope_) return {metal::AllocationFailure::PlanInsufficient, "automatic receiving requires one block plan"};
      metal::AllocationFailure failure = metal::AllocationFailure::None;
      auto reservation = tryReserve(bytes, failure, retained);
      if (!reservation) return failure;
      return {metal::AllocationFailure::PlanInsufficient, "automatic receiving requires one block plan"};
    }
    const auto storageBefore = contributionPlanned ? sampleAutomaticStorage() : std::nullopt;
    bool storageSettled = false;
    bool storageStarted = false;
    const auto beginStorage = [&] {
      if (contributionPlanned) {
        automaticStorageContributionActive_ = true;
        storageStarted = true;
        if (block) { automaticBlock_ = block; automaticBlockConsumed_ = false; }
      }
    };
    const auto settleStorage = [&](bool complete = false) {
      if (!storageStarted || storageSettled) return true;
      storageSettled = true;
      automaticStorageContributionActive_ = false;
      const bool consumed = automaticBlockConsumed_;
      automaticBlock_.reset(); automaticBlockConsumed_ = false;
      const auto after = sampleAutomaticStorage();
      if (!storageBefore || !after) {
        automaticStorageDecision(true, "automatic_storage_accounting_unavailable");
        return false;
      }
      if (*after < *storageBefore) withholdReleasedAutomaticStorage(*storageBefore - *after);
      const uint64_t received = *after > *storageBefore ? *after - *storageBefore : 0;
      automaticStorageWithheldBytes_ -= std::min(received, automaticStorageWithheldBytes_);
      if (received > contribution || *after > automaticStorageQuotaBytes_) {
        automaticStorageDecision(true, "automatic_storage_growth_mismatch");
        return false;
      }
      if (block && complete && (!consumed || received != contribution)) {
        automaticStorageDecision(true, "automatic_block_receipt_mismatch");
        return false;
      }
      return true;
    };
    refreshHostHeld();
    if (pressureProvider_) systemPressure_ = pressureProvider_();
    if (pressureManaged_ && allocationScope_) {
      // Nested allocations consume an already begun peak transaction. No IO
      // or model traversal happens here, including under a storage lock.
      if (systemPressure_ == MemoryPressure::Critical ||
          (!allocationMandatory_ && systemPressure_ == MemoryPressure::Warning && !automaticBlock_))
        return metal::AllocationFailure::HostPressure;
      const uint64_t actual = actualBytes();
      if (actual > allocationPeak_ || bytes > allocationPeak_ - actual)
        return {metal::AllocationFailure::PlanInsufficient,
                "local allocation plan insufficient: requested=" + std::to_string(bytes) +
                " remaining=" + std::to_string(actual < allocationPeak_ ? allocationPeak_ - actual : 0)};
      const uint64_t charged = chargedBytes(true);
      const uint64_t allocationLimit = limitBytes_ - ((retained || allocationElastic_) ? automaticStorageWithheldBytes_ : 0);
      if (charged > allocationLimit || allocationPeak_ - actual > allocationLimit - charged)
        return metal::AllocationFailure::EngineBudget;
      if (automatic && (!automaticStorageFits(contribution) || (contributionPlanned && !storageBefore)))
        return {metal::AllocationFailure::EngineBudget, "automatic store quota exceeded"};
      if (contributionPlanned && automaticStorageContributionActive_)
        return {metal::AllocationFailure::PlanInsufficient, "automatic storage contribution already in progress"};
      beginStorage();
      try { allocate(); }
      catch (...) { static_cast<void>(settleStorage()); publishAccounting(); throw; }
      const bool storageValid = settleStorage(true);
      publishAccounting();
      if (!storageValid) return {metal::AllocationFailure::PlanInsufficient, "actual store growth exceeded its contribution plan"};
      return {};
    }
    if (pressureManaged_ && block) {
      const auto host = sampleHost();
      observeAutomaticGrowth(host);
      if (growthReason_ != "ready" || pendingAutomaticBytes_)
        return {metal::AllocationFailure::GrowthDeferred, std::string(growthReason_)};
    }
    const auto finish = [&](std::string_view outcome) {
      if (!pressureManaged_) return;
      allocationScope_ = false;
      refreshHostHeld();
      if (block) {
        const uint64_t now = policyNow();
        policyNow_ = now;
        nextAutomaticAt_ = now > UINT64_MAX - 5'000 ? UINT64_MAX : now + 5'000;
        pendingAutomaticBytes_ = bytes;
        growthReason_ = "awaiting_sample";
        if (outcome != "success") normalSince_.reset();
      }
    };
    metal::AllocationFailure failure = metal::AllocationFailure::None;
    const bool previousServing = serving_;
    if (pressureManaged_ && !mandatory) serving_ = false;
    auto reservation = tryReserve(bytes, failure, retained, block.has_value());
    serving_ = previousServing;
    if (!reservation) { finish("local_refusal"); return failure; }
    if (block) {
      const auto actual = sampleAutomaticStorage();
      if (!actual || *actual > automaticStorageMaximumBytes_ || contribution > automaticStorageMaximumBytes_ - *actual) {
        finish("local_refusal"); return {metal::AllocationFailure::EngineBudget, "automatic block exceeds immutable store maximum"};
      }
      automaticStorageQuotaBytes_ = std::max(automaticStorageQuotaBytes_, *actual + contribution);
    }
    if (automatic && (!automaticStorageFits(contribution) || (contributionPlanned && !storageBefore))) {
      finish("local_refusal");
      if (deniedReservations_ != UINT64_MAX) ++deniedReservations_;
      return {metal::AllocationFailure::EngineBudget, "automatic store quota exceeded"};
    }
    if (contributionPlanned && automaticStorageContributionActive_) {
      finish("local_refusal");
      return {metal::AllocationFailure::PlanInsufficient, "automatic storage contribution already in progress"};
    }
    const uint64_t before = actualBytes();
    if (pressureManaged_) {
      allocationScope_ = true;
      allocationMandatory_ = mandatory;
      allocationElastic_ = retained;
      allocationPeak_ = bytes <= UINT64_MAX - before ? before + bytes : UINT64_MAX;
    }
    beginStorage();
    try {
      if (allocationDriver_) allocationDriver_(purpose, bytes, allocate);
      else allocate();
    } catch (const metal::MetalAllocationError &error) {
      const bool storageValid = settleStorage();
      finish(actualBytes() > before ? "partial" : "failed");
      if (!storageValid) return {metal::AllocationFailure::PlanInsufficient, "actual store growth exceeded its contribution plan: " + std::string(error.what())};
      return {error.failure(), error.what()};
    } catch (...) {
      static_cast<void>(settleStorage());
      finish(actualBytes() > before ? "partial" : "failed");
      throw;
    }
    refreshHostHeld();
    const bool storageValid = settleStorage(true);
    if (pressureManaged_ && actualBytes() > allocationPeak_) {
      finish("partial");
      return {metal::AllocationFailure::PlanInsufficient, "actual allocation exceeded its local growth plan"};
    }
    if (!storageValid) {
      finish("partial");
      return {metal::AllocationFailure::PlanInsufficient, "actual store growth exceeded its contribution plan"};
    }
    finish("success");
    return {};
  };
}

metal::AllocationAdmission MemoryGovernor::automaticStorageAdmission(std::string purpose,
                                                                    uint64_t finalStorageGrowth) {
  if (purpose != "expert_optional" && purpose != "ple_optional" && purpose != "restore_expert_optional")
    throw std::invalid_argument("automatic storage admission requires a storage purpose");
  return makeAdmission(std::move(purpose), false, finalStorageGrowth);
}

metal::AllocationAdmission MemoryGovernor::automaticStorageAdmission(std::string purpose, AutomaticStorageBlock block) {
  if ((purpose == "ple_optional" && block.kind != AutomaticStorageKind::PleBlock) ||
      ((purpose == "expert_optional" || purpose == "restore_expert_optional") && block.kind != AutomaticStorageKind::ExpertBlock) ||
      (purpose != "expert_optional" && purpose != "ple_optional" && purpose != "restore_expert_optional") ||
      block.grossBlocks != 1 || !block.candidate)
    throw std::invalid_argument("automatic storage admission requires one matching block plan");
  return makeAdmission(std::move(purpose), false, block.contributionBytes, block);
}

bool MemoryGovernor::consumeAutomaticStorageBlock(const AutomaticStorageBlock &block) {
  if (!automaticBlock_ || automaticBlockConsumed_ || *automaticBlock_ != block) return false;
  if (pressureManaged_) {
    try {
      if (pressureProvider_) systemPressure_ = pressureProvider_();
    } catch (...) {
      normalSince_.reset();
      growthReason_ = "pressure_unknown";
      throw metal::MetalAllocationError("automatic block pressure unavailable before receipt",
                                       metal::AllocationFailure::HostPressure);
    }
    observeAutomaticGrowth(sampleHost());
    if (growthReason_ != "ready") {
      const bool pressured = growthReason_ == "system_pressure" || growthReason_ == "pressure_unknown" ||
          growthReason_ == "telemetry_unavailable" || growthReason_ == "recovery_reserve";
      throw metal::MetalAllocationError("automatic block refused before receipt: " + std::string(growthReason_),
          pressured ? metal::AllocationFailure::HostPressure : metal::AllocationFailure::GrowthDeferred);
    }
  }
  automaticBlockConsumed_ = true;
  return true;
}

bool MemoryGovernor::setServing(bool serving) noexcept {
  return std::exchange(serving_, serving);
}

void MemoryGovernor::setRequestWaiting(bool waiting) noexcept {
  requestWaiting_ = waiting;
}

void MemoryGovernor::setPressure(MemoryPressure pressure) noexcept {
  systemPressure_ = pressure;
}

void MemoryGovernor::withholdReleasedAutomaticStorage(uint64_t bytes) noexcept {
  automaticStorageReleasedBytes_ += std::min(bytes, UINT64_MAX - automaticStorageReleasedBytes_);
  automaticStorageWithheldBytes_ += std::min(bytes, limitBytes_ - automaticStorageWithheldBytes_);
}

void MemoryGovernor::setAutomaticStorageQuota(uint64_t quota, uint64_t maximum,
                                             std::function<uint64_t()> actual) {
  if (!quota || quota > maximum || maximum > limitBytes_ || !actual)
    throw std::invalid_argument("invalid automatic storage quota");
  if (automaticStorageActual_ && maximum != automaticStorageMaximumBytes_)
    throw std::invalid_argument("automatic storage maximum is immutable");
  automaticStorageQuotaBytes_ = quota;
  automaticStorageMaximumBytes_ = maximum;
  automaticStorageActual_ = std::move(actual);
}

bool MemoryGovernor::automaticStorageFits(uint64_t contribution) const noexcept {
  if (!automaticStorageActual_) return true;
  const auto actual = sampleAutomaticStorage();
  return actual && *actual <= automaticStorageQuotaBytes_ && contribution <= automaticStorageQuotaBytes_ - *actual;
}

model::StorageRestorePlan MemoryGovernor::planAutomaticStorageRestore(
    const model::ExpertCacheRange &range, uint64_t retainedBytes,
    uint64_t workspaceBytes, uint64_t otherAutomaticBytes) {
  refreshHostHeld();
  if (pressureProvider_) systemPressure_ = pressureProvider_();
  const auto memory = evaluate();
  uint64_t ceiling = std::min(automaticStorageQuotaBytes_, automaticStorageMaximumBytes_);
  ceiling -= std::min(ceiling, otherAutomaticBytes);
  uint64_t headroom = memory.hostMeasurementValid && memory.systemPressure != MemoryPressure::Critical ?
      std::min(memory.headroomBytes, memory.hostAvailableBytes -
          std::min(memory.hostAvailableBytes, memory.hostReserveBytes)) : 0;
  headroom -= std::min(headroom, workspaceBytes);
  ceiling = std::min({ceiling, range.fullBytes, retainedBytes + std::min(headroom, UINT64_MAX - retainedBytes)});
  model::StorageRestorePlan plan{range.minimumBytes,
      range.minimumBytes - std::min(range.minimumBytes, retainedBytes), 0};
  plan.peakBytes = plan.requiredBytes + workspaceBytes;
  if (pressureManaged_)
    automaticStorageQuotaBytes_ = plan.targetBytes + otherAutomaticBytes;
  auto chunks = range.chunks;
  for (; !pressureManaged_;) {
    bool progress = false;
    for (auto &chunk : chunks) {
      if (chunk.minimumChunks >= chunk.maximumChunks || plan.targetBytes > ceiling ||
          chunk.allocationBytes > ceiling - plan.targetBytes) continue;
      plan.targetBytes += chunk.allocationBytes;
      ++chunk.minimumChunks;
      progress = true;
    }
    if (!progress) break;
  }
  return plan;
}

std::optional<uint64_t> MemoryGovernor::sampleAutomaticStorage() const noexcept {
  if (!automaticStorageActual_) return std::nullopt;
  try {
    return automaticStorageActual_();
  } catch (...) {
    return std::nullopt;
  }
}

void MemoryGovernor::controlAutomaticStorage() {
  if (automaticStorageControl_) automaticStorageControl_(evaluate());
}

void MemoryGovernor::reclaimed(ReclaimOutcome outcome) noexcept {
  if (outcome == ReclaimOutcome::Untargeted)
    return;
  host_.reclaimExhausted = host_.constrained && outcome == ReclaimOutcome::Exhausted;
}

MemoryGovernorSnapshot MemoryGovernor::snapshot() const noexcept {
  const std::optional<HostMemory> host = sampleHost();
  return snapshot(host, hostState(host));
}

MemoryGovernorSnapshot MemoryGovernor::evaluate() noexcept {
  const std::optional<HostMemory> host = sampleHost();
  observeAutomaticGrowth(host);
  host_ = hostState(host);
  return snapshot(host, host_);
}

MemoryGovernorSnapshot
MemoryGovernor::snapshot(const std::optional<HostMemory> &hostMemory,
                         const HostState &host) const noexcept {
  uint64_t observed = chargedBytes();
  uint64_t used = observed;
  const uint64_t reserved = localReservedBytes();
  if (reserved <= std::numeric_limits<uint64_t>::max() - used) {
    used += reserved;
  } else {
    used = std::numeric_limits<uint64_t>::max();
  }
  uint64_t hostHeadroom = hostHeadroomBytes(hostMemory, localReservedBytes());
  const uint64_t allocationLimit = limitBytes_ - automaticStorageWithheldBytes_;
  const MemoryPressure pressure = effectivePressure(host);
  bool hostGrowthAllowed = systemPressure_ == MemoryPressure::Normal &&
      !growthHeld(host) && hostHeadroom >= kHostWarningMarginBytes;
  return {
      limitBytes_,
      observed,
      used < limitBytes_ ? limitBytes_ - used : 0,
      pressure,
      deniedReservations_,
      hostMemory.has_value(),
      hostMemory ? hostMemory->available : 0,
      hostReserveBytes_,
      hostMemory ? hostMemory->spare : 0,
      hostSpareReserveBytes_,
      hostHeadroom,
      systemPressure_,
      hostGrowthAllowed,
      reserved,
      untrackedReserveBytes_,
      automaticStorageReleasedBytes_, automaticStorageWithheldBytes_,
      requestStartsPaused(), automaticStorageReason_,
      allocationLimit, used < allocationLimit ? allocationLimit - used : 0,
      pressureManaged_, automaticStorageMinimumBytes_, automaticStorageMaximumBytes_, automaticStorageQuotaBytes_,
      sampleAutomaticStorage(), pendingAutomaticBytes_,
      normalSince_ && policyNow_ >= *normalSince_ ? std::min<uint64_t>(30'000, policyNow_ - *normalSince_) : 0,
      policyNow_ < nextAutomaticAt_ ? nextAutomaticAt_ - policyNow_ : 0, growthReason_,
  };
}

MemoryGovernor::HostState
MemoryGovernor::hostState(const std::optional<HostMemory> &hostMemory) const noexcept {
  const uint64_t hostHeadroom = hostHeadroomBytes(hostMemory, localReservedBytes());
  HostState next;
  next.constrained = !hostMemory ||
      hostHeadroom < kHostWarningMarginBytes ||
      (host_.constrained && hostHeadroom < kHostRecoveryMarginBytes);
  next.reclaimExhausted = host_.reclaimExhausted && next.constrained;
  return next;
}

MemoryPressure
MemoryGovernor::effectivePressure(const HostState &host) const noexcept {
  // Availability and the system's warning control growth and paced reclaim.
  // Only the system's critical signal warrants dropping every evictable cache
  // entry.
  if (systemPressure_ == MemoryPressure::Critical)
    return MemoryPressure::Critical;
  if (host.constrained || systemPressure_ == MemoryPressure::Warning)
    return MemoryPressure::Warning;
  return MemoryPressure::Normal;
}

void MemoryGovernor::release(uint64_t bytes) noexcept {
  if (bytes > reservedBytes_) {
    reservedBytes_ = 0;
    return;
  }
  reservedBytes_ -= bytes;
}

std::optional<MemoryReclaimDirective> MemoryPressurePolicy::update(
    const MemoryGovernorSnapshot &snapshot, double nowMilliseconds,
    bool requestWaiting) noexcept {
  if (snapshot.pressure == MemoryPressure::Normal) {
    nextReclaimMilliseconds_ = 0.0;
    return std::nullopt;
  }
  if (snapshot.pressure == MemoryPressure::Critical)
    return MemoryReclaimDirective{.critical = true};
  if (nowMilliseconds < nextReclaimMilliseconds_)
    return continued_.value_or(MemoryReclaimDirective{});
  continued_.reset();

  // Missing telemetry pauses allocation, but is not evidence that live
  // cache must be discarded. Empty extents can still be returned.
  if (!snapshot.hostMeasurementValid &&
      snapshot.systemPressure == MemoryPressure::Normal)
    return MemoryReclaimDirective{};

  // A pass a request waits for frees toward the recovery margin, where growth
  // reopens (hostState). One that nothing waits for only restores the warning
  // margin that growth leaves free: freeing on to the recovery margin would
  // discard cache nothing needs, on every pass while the host's idle headroom
  // stays below it. Such a pass also keeps the resume point, whose loss costs
  // the next request a replay of its whole prompt; a waiting request outranks
  // it. Nor does such a pass take what is in use: the states of the requests
  // in service and the KV they restore through. The replay point it would
  // take is the next resume point, and what it would free goes back to those
  // requests' growth, which neither the margin nor macOS's warning holds
  // back. macOS's warning asks caches to give memory back, whatever the
  // headroom: while it lasts, every pass returns as much as a pass may,
  // still short of what is in use unless a request waits, and keeps the
  // resume point even for a waiting request, whose growth the warning
  // refuses as well, so that what the pass frees would not reach it.
  const bool warning = snapshot.systemPressure == MemoryPressure::Warning;
  const uint64_t margin = requestWaiting ? kHostRecoveryMarginBytes : kHostWarningMarginBytes;
  const uint64_t deficit =
      snapshot.hostHeadroomBytes < margin ? margin - snapshot.hostHeadroomBytes : 0;
  const uint64_t desired = warning ? kHostWarningMarginBytes : deficit;
  // The host samples every 500 ms. Allow counters to settle after a pass
  // with a target, but keep responding if another application continues
  // consuming memory. A pass with none leaves nothing to settle: a request
  // that starts waiting gets its pass at once.
  if (desired)
    nextReclaimMilliseconds_ = nowMilliseconds + 1000.0;
  return MemoryReclaimDirective{
      .targetBytes = std::min(desired, kHostWarningMarginBytes),
      .keepResumePoint = !requestWaiting || warning,
      .upTo = requestWaiting ? ReclaimClass::InUse : ReclaimClass::Ordinary};
}

void MemoryPressurePolicy::reclaimed(const MemoryReclaimDirective &directive,
                                     const MemoryReclaimResult &result) noexcept {
  continued_.reset();
  // Every critical pass evicts everything again by itself.
  if (result.outcome != ReclaimOutcome::Pending || directive.critical)
    return;
  continued_ = directive;
  continued_->targetBytes -= std::min(result.releasedBytes, directive.targetBytes);
}

} // namespace splash::engine
