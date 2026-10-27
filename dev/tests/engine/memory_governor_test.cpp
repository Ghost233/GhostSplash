#include "TestChecks.hpp"
#include "TestMetalMemory.hpp"
#include "TestModel.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/ExpertPlePolicy.hpp"
#include "model/SparseRowStore.hpp"
#include "TestFiles.hpp"
#include "engine/MemoryPlan.hpp"

#include <cstdlib>
#include <array>
#include <atomic>
#include <chrono>
#include <thread>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::rejects;
using splash::test::require;

// Admits bytes through the governor's one admission path, running allocate
// while they are reserved.
metal::AllocationResult admit(MemoryGovernor &governor, uint64_t bytes,
                              const std::function<void()> &allocate = [] {}) {
  return governor.allocationAdmission()(bytes, allocate);
}

// A host whose spare memory is all its available memory: no anonymous
// pages to compress.
std::optional<HostMemory> hostMemory(std::optional<uint64_t> bytes) {
  if (!bytes)
    return std::nullopt;
  return HostMemory{*bytes, *bytes};
}

// Charges bytes to the backend, as an allocation does.
void allocate(uint64_t bytes) {
  test::metalStatistics().allocatedBytes += bytes;
  test::metalStatistics().deviceCurrentAllocatedBytes += bytes;
}

// Returned automatic storage does not increase the elastic KV/state ceiling.
void testReleasedAutomaticStorageDoesNotExpandElasticHeadroom() {
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  MemoryGovernor governor(backend, 8 * kGiB, 2 * kGiB, 2 * kGiB,
                          [] { return hostMemory(64 * kGiB); }, 0);
  require(static_cast<bool>(admit(governor, 6 * kGiB, [] { allocate(6 * kGiB); })),
          "automatic-storage fixture did not allocate");
  const auto before = governor.snapshot();
  // A 1 GiB real chunk has been returned at a command-free point.
  test::metalStatistics().allocatedBytes -= kGiB;
  test::metalStatistics().deviceCurrentAllocatedBytes -= kGiB;
  governor.withholdReleasedAutomaticStorage(kGiB);
  const auto after = governor.snapshot();
  const auto denied = governor.allocationAdmission("kv_required")(3 * kGiB, [] { allocate(3 * kGiB); });
  require(!denied && denied.failure == metal::AllocationFailure::EngineBudget &&
              after.limitBytes == before.limitBytes && after.chargedBytes == 5 * kGiB &&
              after.headroomBytes == before.headroomBytes + kGiB && after.elasticHeadroomBytes == 2 * kGiB &&
              after.elasticLimitBytes == 7 * kGiB && after.localReservedBytes == 0 &&
              after.automaticStorageReleasedBytes == kGiB && after.automaticStorageWithheldBytes == kGiB,
          "released automatic storage enlarged elastic KV/state headroom or changed actual charge/hard limit");
  require(static_cast<bool>(governor.allocationAdmission("state_required")(2 * kGiB, [] { allocate(2 * kGiB); })),
          "withholding took the preexisting KV/state headroom");
  require(static_cast<bool>(governor.allocationAdmission("restore_weights")(kGiB, [] { allocate(kGiB); })),
          "KV/state withholding reduced required nonautomatic weight/floor restore capacity");
}

void testNestedElasticAdmissionRetainsWithheldQuota() {
  test::metalStatistics() = {};
  allocate(5 * kGiB);
  metal::MetalBackend backend("unused");
  MemoryGovernor governor(backend, 8 * kGiB, 2 * kGiB, 2 * kGiB,
                          [] { return hostMemory(64 * kGiB); }, 0);
  governor.enablePressureControl();
  governor.withholdReleasedAutomaticStorage(kGiB);
  const auto denied = governor.allocationAdmission("kv_required")(3 * kGiB, [] {});
  require(denied.failure == metal::AllocationFailure::EngineBudget,
          "local KV admission consumed withheld expert quota");
  require(bool(governor.allocationAdmission("restore_weights")(3 * kGiB, [&] {
    const auto nested = governor.allocationAdmission("state_required")(3 * kGiB, [] {});
    require(nested.failure == metal::AllocationFailure::EngineBudget,
            "required parent bypassed the nested elastic ceiling");
  })), "required weights lost original hard-budget headroom");
  require(bool(governor.allocationAdmission("kv_required")(2 * kGiB, [&] {
    require(bool(governor.allocationAdmission("managed_buffer")(2 * kGiB, [] { allocate(2 * kGiB); })),
            "nested local buffer was charged twice");
  })) && governor.snapshot().chargedBytes == 7 * kGiB && governor.snapshot().elasticHeadroomBytes == 0,
          "local nested allocation lost or exceeded its peak reservation");
}

void testAutomaticDonationCannotBeRefilledByTheOtherStore() {
  test::metalStatistics() = {};
  allocate(7 * kGiB);
  metal::MetalBackend backend("unused");
  MemoryGovernor governor(backend, 8 * kGiB, 2 * kGiB, 2 * kGiB,
                          [] { return hostMemory(64 * kGiB); }, 0);
  governor.enablePressureControl();
  governor.withholdReleasedAutomaticStorage(kGiB);
  bool received = false;
  const auto result = governor.allocationAdmission("ple_optional", false)(kGiB, [&] { received = true; });
  require(!received && result.failure == metal::AllocationFailure::EngineBudget,
          "unplanned PLE growth refilled released expert storage");
}

void testAutomaticPoolQuotaIsIndependentOfGlobalHeadroom() {
  test::metalStatistics() = {};
  allocate(7 * kGiB);
  uint64_t automaticActual = 5 * kGiB;
  metal::MetalBackend backend("unused");
  MemoryGovernor governor(backend, 32 * kGiB, 2 * kGiB, 2 * kGiB,
                          [] { return hostMemory(64 * kGiB); }, 0);
  governor.setAutomaticStorageQuota(5 * kGiB, 6 * kGiB, [&] { return automaticActual; });
  bool received = false;
  const auto result = governor.allocationAdmission("ple_optional", false)(kGiB, [&] {
    received = true;
    automaticActual += kGiB;
    allocate(kGiB);
  });
  require(!received && result.failure == metal::AllocationFailure::EngineBudget &&
              automaticActual == 5 * kGiB && governor.snapshot().chargedBytes == 7 * kGiB,
          "automatic PLE exceeded its current shared store quota using unrelated global headroom");
}

void testCriticalBeforeRealPleReceiptRetainsItsHostPressureCause() {
  struct ZeroRows final : model::RowReader {
    void read(uint64_t, std::span<uint8_t> bytes) override { std::fill(bytes.begin(), bytes.end(), 0); }
    void checkUnchanged() override {}
    uint64_t size() const override { return 12'288 * 16; }
  };
  test::metalStatistics() = {}; test::hostStatistics() = {};
  constexpr uint64_t rows = 12'288;
  constexpr uint32_t rowBytes = 16, stageRows = 8;
  const auto maximum = model::SparseRowStore::maximumResizableBytes(rows, rowBytes, 0, stageRows);
  const auto floor = model::SparseRowStore::minimumResizableBytes(maximum, rowBytes, 0, stageRows, rows);
  model::SparseRowStore store({"local-pressure-cause", rows, rowBytes, 0}, std::make_unique<ZeroRows>(),
      {.cacheBytes = floor, .readers = std::make_shared<model::SsdReaders>(1), .stageRows = stageRows,
       .maximumCacheBytes = maximum, .boundedWorkingMemory = true});
  metal::MetalBackend backend("unused");
  MemoryGovernor governor(backend, 8 * kMiB, 2 * kGiB, 2 * kGiB,
                          [] { return hostMemory(64 * kGiB); }, 0);
  uint64_t now = 0, guardCalls = 0;
  MemoryPressure pressure = MemoryPressure::Normal;
  governor.setPressureProvider([&] { return pressure; });
  governor.setNonBackendHostHeld([&] { return store.nonBackendHostBytes(); });
  governor.setAutomaticStorageQuota(floor, maximum, [&] { return store.committedBytes(); });
  governor.enablePressureControl({[&] { return now; }, [&] {
    return MemoryPolicyObservation{MemoryPolicyObservation::Pressure(uint8_t(pressure) + 1), 0};
  }});
  store.setGrowthGuard([&](const model::StorageGrowthPlan &plan) {
    ++guardCalls;
    return governor.consumeAutomaticStorageBlock({AutomaticStorageKind::PleBlock, 1, plan.candidate,
        plan.contributionBytes, plan.auxiliaryBytes});
  });
  governor.setAllocationDriver([&](std::string_view purpose, uint64_t, const std::function<void()> &allocate) {
    if (purpose == "ple_optional") pressure = MemoryPressure::Critical;
    allocate();
  });
  const auto plan = store.planGrowOne(maximum);
  require(plan.has_value(), "bounded real PLE store has no next block");
  const auto beforeHost = store.hostAllocatedBytes();
  now = 30'000;
  metal::AllocationResult result;
  try {
    result = governor.automaticStorageAdmission("ple_optional", {AutomaticStorageKind::PleBlock, 1, plan->candidate,
        plan->contributionBytes, plan->auxiliaryBytes})(plan->peakBytes, [&] { static_cast<void>(store.growOne(*plan)); });
  } catch (const std::invalid_argument &error) {
    throw std::runtime_error("Critical real PLE veto became an invalid plan after " + std::to_string(guardCalls) +
        " guard calls: " + error.what());
  }
  require(result.failure == metal::AllocationFailure::HostPressure && guardCalls == 1 &&
              store.committedBytes() == floor && store.hostAllocatedBytes() == beforeHost &&
              governor.snapshot().localReservedBytes == 0,
          "Critical real PLE veto lost HostPressure or changed actual owned storage: " + result.detail);
  // An actually stale plan must keep its own error, even if the same
  // allocation boundary also observes Critical before calling the owner.
  pressure = MemoryPressure::Normal;
  governor.setPressure(pressure);
  now = 60'000; static_cast<void>(governor.evaluate()); now = 90'000;
  auto stale = *plan;
  ++stale.revision;
  rejects([&] {
    static_cast<void>(governor.automaticStorageAdmission("ple_optional", {AutomaticStorageKind::PleBlock, 1, plan->candidate,
        plan->contributionBytes, plan->auxiliaryBytes})(plan->peakBytes, [&] { static_cast<void>(store.growOne(stale)); }));
  }, "PLE single-block plan is stale", "a genuine stale plan was relabeled as host pressure");
  require(guardCalls == 1 && store.committedBytes() == floor && store.hostAllocatedBytes() == beforeHost,
          "stale PLE metadata reached consumption or allocated payload");
}

void testLocalAutomaticBlockPacingAndFreshReceiverObservation() {
  test::metalStatistics() = {}; test::hostStatistics() = {};
  allocate(2 * kGiB);
  uint64_t actual = kGiB, now = 0, sequence = 0;
  bool measured = true;
  MemoryPolicyObservation::Pressure observed = MemoryPolicyObservation::Pressure::Normal;
  metal::MetalBackend backend("unused");
  MemoryGovernor governor(backend, 16 * kGiB, 2 * kGiB, 2 * kGiB,
      [&] { return measured ? hostMemory(64 * kGiB) : std::nullopt; }, 0);
  governor.setAutomaticStorageQuota(actual, 10 * kGiB, [&] { return actual; });
  governor.setAutomaticStorageMinimum(kGiB);
  governor.enablePressureControl({[&] { return now; }, [&] { return MemoryPolicyObservation{observed, sequence}; }});
  uint64_t candidate = 1;
  const auto grow = [&](uint64_t net = kGiB) {
    const AutomaticStorageBlock block{AutomaticStorageKind::ExpertBlock, 1, candidate++, net, 0};
    return governor.automaticStorageAdmission("expert_optional", block)(kGiB, [&] {
      if (!governor.consumeAutomaticStorageBlock(block))
        throw metal::MetalAllocationError("receiver rejected stale block", metal::AllocationFailure::GrowthDeferred);
      require(!governor.consumeAutomaticStorageBlock(block), "one receipt authorized multiple actual blocks");
      actual += net; allocate(net);
    });
  };
  now = 29'999;
  require(!grow(), "local growth escaped its 30-second observation");
  now = 30'000;
  governor.setAllocationDriver([&](std::string_view, uint64_t, const std::function<void()> &allocate) {
    ++sequence; // A brief Critical/Unknown event between safe-point samples.
    allocate();
  });
  require(grow().failure == metal::AllocationFailure::GrowthDeferred && actual == kGiB,
          "a cached permit survived an unsafe event before the actual receiving block");
  governor.setAllocationDriver({});
  static_cast<void>(governor.evaluate());
  now = 59'999;
  require(!grow(), "failed receipt reused the earlier stable observation");
  now = 60'000;
  require(bool(grow()) && actual == 2 * kGiB && governor.snapshot().pendingAutomaticBytes == kGiB,
          "one local block did not retain its unresolved allocation reservation");
  measured = false;
  now = 65'000;
  require(!grow() && governor.snapshot().pendingAutomaticBytes == kGiB,
          "unknown telemetry retired growth before an effective post-allocation sample");
  measured = true;
  static_cast<void>(governor.evaluate());
  now = 94'999;
  require(!grow(), "missing telemetry did not restart stable observation");
  now = 95'000;
  require(bool(grow(0)) && actual == 2 * kGiB,
          "a real gross block with zero net contribution was rejected");
  static_cast<void>(governor.evaluate());
  now = 99'999;
  require(!grow(), "zero-net real block skipped the shared local cooldown");
  now = 100'000;
  require(bool(grow()), "one local block was not allowed at the exact 5-second completion boundary");
  observed = MemoryPolicyObservation::Pressure::Unknown;
  now = 105'000;
  require(!grow(), "unknown pressure authorized optional growth");
  observed = MemoryPolicyObservation::Pressure::Normal;
  static_cast<void>(governor.evaluate());
  now = 134'999;
  require(!grow(), "unknown pressure preserved stable growth credit");
  now = 135'000;
  require(bool(grow()), "valid Normal did not recover after a new stable interval");
}

void testWarningRecoversByMeasuredHeadroomAndRechecksTheRealReceiver() {
  using Observation = MemoryPolicyObservation;
  const auto warning = Observation::record(0, Observation::Pressure::Warning);
  require(warning == 2 && Observation::record(warning, Observation::Pressure::Warning) == 2,
          "repeated Warning notifications restarted the recovery sequence");
  const auto critical = Observation::record(warning, Observation::Pressure::Critical);
  const auto unknown = Observation::record(critical, Observation::Pressure::Unknown);
  require(critical == 7 && unknown == 11 &&
              Observation::record(unknown, Observation::Pressure::Warning) == 10,
          "Unknown telemetry erased an effective Critical signal or failed to invalidate recovery");
  test::metalStatistics() = {}; test::hostStatistics() = {};
  allocate(kGiB);
  uint64_t actual = kGiB, now = 0, sequence = 0, spare = 10 * kGiB, allocations = 0;
  MemoryPressure pressure = MemoryPressure::Warning;
  auto observed = MemoryPolicyObservation::Pressure::Warning;
  metal::MetalBackend backend("unused");
  MemoryGovernor governor(backend, 16 * kGiB, 2 * kGiB, 2 * kGiB,
                          [&] { return hostMemory(spare); }, 0);
  governor.setPressure(pressure);
  governor.setPressureProvider([&] { return pressure; });
  governor.setAutomaticStorageQuota(actual, 8 * kGiB, [&] { return actual; });
  governor.enablePressureControl({[&] { return now; }, [&] { return MemoryPolicyObservation{observed, sequence}; }});
  uint64_t candidate = 1;
  const auto grow = [&] {
    const AutomaticStorageBlock block{AutomaticStorageKind::ExpertBlock, 1, candidate++, kMiB, 0};
    return governor.automaticStorageAdmission("expert_optional", block)(kMiB, [&] {
      require(governor.consumeAutomaticStorageBlock(block), "Warning receiver lost its exact block permit");
      const auto nested = governor.allocationAdmission("managed_buffer")(kMiB, [&] {
        ++allocations; actual += kMiB; allocate(kMiB);
      });
      if (!nested) throw metal::MetalAllocationError(nested.detail, nested.failure);
    });
  };
  for (now = 1000; now < 30'000; now += 1000) static_cast<void>(governor.evaluate());
  now = 29'999;
  require(!grow() && allocations == 0, "Warning growth skipped the original stable interval");
  now = 30'000;
  require(bool(grow()) && allocations == 1 && governor.snapshot().systemPressure == MemoryPressure::Warning &&
              !governor.snapshot().hostGrowthAllowed,
          "sustained Warning with sufficient measured headroom never recovered one real block");
  require(governor.allocationAdmission("kv_optional", false)(kMiB, [] {}).failure == metal::AllocationFailure::HostPressure,
          "automatic Warning recovery changed the original nonautomatic KV admission policy");
  static_cast<void>(governor.evaluate());
  now = 34'999;
  require(!grow() && allocations == 1, "Warning recovery skipped the original block cooldown");
  now = 35'000;
  require(bool(grow()) && allocations == 2, "repeated Warning observations restarted recovery indefinitely");

  // A truthful Warning in the hysteresis band holds, rather than growing or
  // retaining the earlier high-headroom stability credit.
  spare = 2 * kGiB + kHostRecoveryMarginBytes - 1;
  now = 40'000;
  require(!grow() && allocations == 2, "Warning below the recovery margin grew optional payload");
  spare = 10 * kGiB;
  static_cast<void>(governor.evaluate());
  now = 69'999;
  require(!grow() && allocations == 2, "low headroom retained earlier stable credit");
  now = 70'000;
  require(bool(grow()) && allocations == 3, "Warning did not restart its measured stable interval");

  // Each late veto happens after admission and immediately before the
  // actual storage owner consumes the plan; none may allocate payload.
  for (unsigned veto = 0; veto < 4; ++veto) {
    const auto before = allocations;
    now += 30'000;
    static_cast<void>(governor.evaluate());
    governor.setAllocationDriver([&](std::string_view, uint64_t, const std::function<void()> &allocate) {
      if (veto == 0) pressure = MemoryPressure::Critical;
      else if (veto == 1) observed = MemoryPolicyObservation::Pressure::Unknown;
      else if (veto == 2) spare = 2 * kGiB + kHostRecoveryMarginBytes - 1;
      else ++sequence; // A brief Critical/Unknown event already ended.
      allocate();
    });
    const auto denied = grow();
    const auto cause = veto == 3 ? metal::AllocationFailure::GrowthDeferred : metal::AllocationFailure::HostPressure;
    require(denied.failure == cause && allocations == before && governor.snapshot().localReservedBytes == 0,
            "fresh unsafe Warning receiver observation allocated payload or retained its reservation");
    pressure = MemoryPressure::Warning; observed = MemoryPolicyObservation::Pressure::Warning; spare = 10 * kGiB;
    governor.setPressure(pressure); governor.setAllocationDriver({});
    static_cast<void>(governor.evaluate());
    now += 29'999;
    require(!grow() && allocations == before, "late unsafe event retained stale recovery credit");
    ++now;
    require(bool(grow()) && allocations == before + 1,
            "Warning did not recover after the replacement stable interval");
  }
}

void testAutomaticPayloadRechecksAfterItsBlockPermitWasConsumed() {
  for (unsigned interruption = 0; interruption < 3; ++interruption) {
    test::metalStatistics() = {}; test::hostStatistics() = {};
    allocate(kGiB);
    uint64_t actual = kGiB, now = 0, sequence = 0, spare = 10 * kGiB, payloads = 0;
    auto observed = MemoryPolicyObservation::Pressure::Warning;
    metal::MetalBackend backend("unused");
    MemoryGovernor governor(backend, 8 * kGiB, 2 * kGiB, 2 * kGiB,
                            [&] { return hostMemory(spare); }, 0);
    governor.setPressure(MemoryPressure::Warning);
    governor.setAutomaticStorageQuota(actual, 4 * kGiB, [&] { return actual; });
    governor.enablePressureControl({[&] { return now; }, [&] { return MemoryPolicyObservation{observed, sequence}; }});
    bool interrupt = true;
    const auto grow = [&] {
      const AutomaticStorageBlock block{AutomaticStorageKind::ExpertBlock, 1, now + 1, kMiB, 0};
      return governor.automaticStorageAdmission("expert_optional", block)(kMiB, [&] {
        require(governor.consumeAutomaticStorageBlock(block), "receiver did not consume its admitted block plan");
        // Read-ahead may drain between the owner's permit and its nested
        // payload admission. A current Warning alone must remain eligible.
        if (interrupt) {
          if (interruption == 0) spare = 2 * kGiB + kHostRecoveryMarginBytes - 1;
          else if (interruption == 1) observed = MemoryPolicyObservation::Pressure::Unknown;
          else ++sequence; // Critical already returned to Warning.
        }
        const auto result = governor.allocationAdmission("managed_buffer")(kMiB, [&] {
          ++payloads; actual += kMiB; allocate(kMiB);
        });
        if (!result) throw metal::MetalAllocationError(result.detail, result.failure);
      });
    };
    now = 30'000;
    const auto denied = grow();
    const auto cause = interruption == 2 ? metal::AllocationFailure::GrowthDeferred : metal::AllocationFailure::HostPressure;
    require(denied.failure == cause && payloads == 0 && actual == kGiB &&
                governor.snapshot().chargedBytes == kGiB && governor.snapshot().localReservedBytes == 0,
            "consumed expert permit survived a drain-time unsafe observation before its actual Metal payload");
    interrupt = false; observed = MemoryPolicyObservation::Pressure::Warning; spare = 10 * kGiB;
    static_cast<void>(governor.evaluate());
    now = 59'999;
    require(!grow() && payloads == 0, "drain-time unsafe observation retained old stable credit");
    now = 60'000;
    require(bool(grow()) && payloads == 1 && actual == kGiB + kMiB,
            "Warning did not resume one expert payload after a replacement stable interval");
  }
}

void testAutomaticPayloadKeepsItsPaidPeakSeparateFromExternalPressure() {
  for (const bool externalDrop : {false, true}) {
    test::metalStatistics() = {}; test::hostStatistics() = {};
    allocate(kGiB);
    constexpr uint64_t first = kGiB, second = 512 * kMiB, total = first + second;
    uint64_t actual = kGiB, now = 0, spare = 4608 * kMiB, payloads = 0;
    metal::MetalBackend backend("unused");
    MemoryGovernor governor(backend, 8 * kGiB, 2 * kGiB, 2 * kGiB,
                            [&] { return hostMemory(spare); }, 0);
    governor.setPressure(MemoryPressure::Warning);
    governor.setAutomaticStorageQuota(actual, 4 * kGiB, [&] { return actual; });
    governor.enablePressureControl({[&] { return now; }, [] {
      return MemoryPolicyObservation{MemoryPolicyObservation::Pressure::Warning, 0};
    }});
    now = 30'000;
    const AutomaticStorageBlock block{AutomaticStorageKind::ExpertBlock, 1, 1, total, 0};
    const auto result = governor.automaticStorageAdmission("expert_optional", block)(total, [&] {
      require(governor.consumeAutomaticStorageBlock(block), "split payload lost its original permit");
      const auto part = [&](uint64_t bytes) {
        const auto admitted = governor.allocationAdmission("managed_buffer")(bytes, [&] {
          ++payloads; actual += bytes; allocate(bytes); spare -= bytes;
        });
        if (!admitted) throw metal::MetalAllocationError(admitted.detail, admitted.failure);
      };
      part(first);
      if (externalDrop) spare -= 512 * kMiB;
      part(second);
    });
    if (externalDrop)
      require(result.failure == metal::AllocationFailure::HostPressure && payloads == 1 &&
                  actual == 2 * kGiB && governor.snapshot().chargedBytes == actual,
              "a paid partial block masked an external decline across its remaining peak margin");
    else
      require(bool(result) && payloads == 2 && actual == kGiB + total && spare == 3 * kGiB,
              "scope-owned first payload was counted twice against an already admitted split peak");
    require(governor.snapshot().localReservedBytes == 0, "split automatic payload retained a finished peak reservation");
  }
}

void testRealPlePrefetchDrainCannotKeepAConsumedUnsafePermit() {
  for (unsigned interruption = 0; interruption < 3; ++interruption) {
    test::metalStatistics() = {}; test::hostStatistics() = {};
    std::atomic<uint64_t> spare{64 * kGiB}, sequence{0};
    std::atomic<MemoryPolicyObservation::Pressure> observed{MemoryPolicyObservation::Pressure::Warning};
    std::atomic<bool> consumed{false}, readStarted{false}, drained{false};
    struct DrainingRows final : model::RowReader {
      std::function<void()> drain;
      explicit DrainingRows(std::function<void()> action) : drain(std::move(action)) {}
      void read(uint64_t, std::span<uint8_t> bytes) override { drain(); std::fill(bytes.begin(), bytes.end(), 0); }
      void checkUnchanged() override {}
      uint64_t size() const override { return 12'288 * 16; }
    };
    auto accounting = std::make_shared<HostMemoryAccounting>();
    MemoryGovernor *admission = nullptr;
    uint64_t payloads = 0;
    const HostMemoryAdmission hostAllocation = [&](uint64_t bytes, const std::function<void()> &allocate,
                                                  const std::function<uint64_t()> &held) {
      HostMemoryAllocation lease;
      const auto body = [&] {
        if (admission) ++payloads;
        allocate();
        lease = HostMemoryAllocation(accounting, held ? held() : bytes, false);
      };
      if (admission) {
        const auto result = admission->allocationAdmission("managed_buffer")(bytes, body);
        if (!result) throw metal::MetalAllocationError(result.detail, result.failure);
      } else body();
      return lease;
    };
    constexpr uint64_t rows = 12'288;
    constexpr uint32_t rowBytes = 16, prefetchRows = 1, stageRows = 8;
    const auto maximum = model::SparseRowStore::maximumResizableBytes(rows, rowBytes, prefetchRows, stageRows);
    const auto floor = model::SparseRowStore::minimumResizableBytes(maximum, rowBytes, prefetchRows, stageRows, rows);
    auto reader = std::make_unique<DrainingRows>([&] {
      readStarted = true;
      const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (!consumed && std::chrono::steady_clock::now() < until)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      require(consumed, "bounded real PLE prefetch never reached its consumed permit");
      if (interruption == 0) spare = 2 * kGiB + kHostRecoveryMarginBytes - 1;
      else if (interruption == 1) observed = MemoryPolicyObservation::Pressure::Unknown;
      else ++sequence; // A brief Critical already ended in Warning.
      drained = true;
    });
    model::SparseRowStore store({"drain-time-pressure", rows, rowBytes, 0}, std::move(reader),
        {.cacheBytes = floor, .readers = std::make_shared<model::SsdReaders>(1), .prefetchRows = prefetchRows,
         .stageRows = stageRows, .maximumCacheBytes = maximum, .boundedWorkingMemory = true,
         .hostAllocation = hostAllocation});
    metal::MetalBackend backend("unused");
    MemoryGovernor governor(backend, 8 * kMiB, 2 * kGiB, 2 * kGiB,
                            [&] { return hostMemory(spare.load()); }, 0, accounting->fixedBytes.load());
    admission = &governor;
    governor.setNonBackendHostHeld([&] { return accounting->fixedBytes.load(); });
    governor.setPressure(MemoryPressure::Warning);
    governor.setAutomaticStorageQuota(floor, maximum, [&] { return store.committedBytes(); });
    uint64_t now = 0;
    governor.enablePressureControl({[&] { return now; }, [&] {
      return MemoryPolicyObservation{observed.load(), sequence.load()};
    }});
    store.setGrowthGuard([&](const model::StorageGrowthPlan &plan) {
      const bool permitted = governor.consumeAutomaticStorageBlock({AutomaticStorageKind::PleBlock, 1,
          plan.candidate, plan.contributionBytes, plan.auxiliaryBytes});
      if (permitted) consumed = true;
      return permitted;
    });
    store.prefetch(std::array<uint32_t, 1>{1});
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!readStarted && std::chrono::steady_clock::now() < until)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    require(readStarted, "real PLE prefetch did not begin its bounded blocking read");
    const auto grow = [&] {
      const auto plan = store.planGrowOne(maximum);
      require(plan.has_value(), "real PLE lost its next physical block");
      return governor.automaticStorageAdmission("ple_optional", {AutomaticStorageKind::PleBlock, 1,
          plan->candidate, plan->contributionBytes, plan->auxiliaryBytes})(plan->peakBytes,
          [&] { static_cast<void>(store.growOne(*plan)); });
    };
    const auto before = accounting->fixedBytes.load();
    now = 30'000;
    const auto denied = grow();
    const auto cause = interruption == 2 ? metal::AllocationFailure::GrowthDeferred : metal::AllocationFailure::HostPressure;
    require(consumed && drained && denied.failure == cause && payloads == 0 &&
                store.committedBytes() == floor && accounting->fixedBytes == before &&
                governor.snapshot().chargedBytes == before && governor.snapshot().localReservedBytes == 0,
            "real PLE drain retained an unsafe consumed permit or allocated a host payload");
    spare = 64 * kGiB; observed = MemoryPolicyObservation::Pressure::Warning;
    static_cast<void>(governor.evaluate());
    now = 59'999;
    require(!grow() && payloads == 0, "real PLE drain failure preserved the previous stable interval");
    now = 60'000;
    require(bool(grow()) && payloads > 0 && store.committedBytes() > floor &&
                accounting->fixedBytes > before && governor.snapshot().localReservedBytes == 0,
            "real PLE did not recover under Warning after a replacement stable interval");
  }
}

void testAutomaticQuotaCountsFinalStoreContributionNotTransientWorkspace() {
  test::metalStatistics() = {}; test::hostStatistics() = {};
  allocate(7 * kGiB);
  uint64_t actual = 5 * kGiB, now = 0;
  metal::MetalBackend backend("unused");
  MemoryGovernor governor(backend, 8 * kGiB, 2 * kGiB, 2 * kGiB,
                          [] { return hostMemory(64 * kGiB); }, 0);
  governor.setAutomaticStorageQuota(actual, 6 * kGiB, [&] { return actual; });
  governor.withholdReleasedAutomaticStorage(kGiB);
  governor.enablePressureControl({[&] { return now; }, [] {
    return MemoryPolicyObservation{MemoryPolicyObservation::Pressure::Normal, 0};
  }});
  require(bool(governor.allocationAdmission("storage_workspace", false)(kGiB, [&] {
    require(bool(governor.allocationAdmission("managed_buffer")(kGiB, [] { allocate(kGiB); })),
            "workspace lost its outer local peak reservation");
    test::metalStatistics().allocatedBytes -= kGiB; test::metalStatistics().deviceCurrentAllocatedBytes -= kGiB;
    require(bool(governor.allocationAdmission("managed_buffer")(kGiB, [] { test::hostStatistics().temporaryBytes += kGiB; })),
            "host workspace lost its local peak reservation");
    require(governor.snapshot().chargedBytes == 8 * kGiB, "host workspace was not charged once");
    test::hostStatistics().temporaryBytes -= kGiB;
  })) && governor.snapshot().automaticStorageWithheldBytes == kGiB,
          "transient workspace counted as final automatic growth");
  now = 30'000;
  const AutomaticStorageBlock block{AutomaticStorageKind::PleBlock, 1, 71, kGiB, 0};
  governor.setAllocationDriver([](std::string_view, uint64_t, const std::function<void()> &allocate) {
    allocate(); throw metal::MetalAllocationError("partial receipt", metal::AllocationFailure::DriverRejected);
  });
  const auto partial = governor.automaticStorageAdmission("ple_optional", block)(kGiB, [&] {
    require(governor.consumeAutomaticStorageBlock(block), "exact partial block plan was refused");
    actual += kGiB; allocate(kGiB);
  });
  require(partial.failure == metal::AllocationFailure::DriverRejected && actual == 6 * kGiB &&
              governor.snapshot().chargedBytes == 8 * kGiB && governor.snapshot().automaticStorageWithheldBytes == 0,
          "partial receipt fabricated rollback or miscounted its final contribution");
}

void testLocalRestoreSelectsRequiredFloorBeforePayloadAndQuota() {
  test::metalStatistics() = {}; test::hostStatistics() = {};
  allocate(1920 * kMiB);
  uint64_t actual = 320 * kMiB;
  metal::MetalBackend backend("unused");
  MemoryGovernor governor(backend, 3000 * kMiB, 1, 1,
                          [] { return hostMemory(64 * kGiB); }, 0);
  governor.enablePressureControl();
  governor.setAutomaticStorageQuota(1000 * kMiB, 1200 * kMiB, [&] { return actual; });
  model::ExpertCacheRange range;
  range.minimumBytes = 512 * kMiB; range.fullBytes = 4096 * kMiB;
  range.chunks = {{.allocationBytes = 128 * kMiB, .minimumChunks = 1, .maximumChunks = 2},
                  {.allocationBytes = 256 * kMiB, .minimumChunks = 1, .maximumChunks = 2}};
  const auto plan = governor.planAutomaticStorageRestore(range, 64 * kMiB, 32 * kMiB, 256 * kMiB);
  require(plan.targetBytes == 512 * kMiB && plan.requiredBytes == 448 * kMiB && plan.peakBytes == 480 * kMiB &&
              governor.snapshot().automaticQuotaBytes == 768 * kMiB &&
              governor.snapshot().automaticActualBytes == actual && governor.snapshot().localReservedBytes == 0,
          "local restore retained the old optional quota or counted retained PLE/helpers as new payload");
}

void testLocalAdaptiveTransferRequiresActualDonationAndOnePacedReceipt() {
  test::metalStatistics() = {}; test::hostStatistics() = {};
  std::vector<uint64_t> expertBlocks(28, 128 * kMiB);
  uint64_t expert = 3584 * kMiB, ple = 512 * kMiB, now = 0;
  allocate(expert);
  test::hostStatistics().fixedBytes = ple;
  metal::MetalBackend backend("unused");
  MemoryGovernor governor(backend, 8 * kGiB, 2 * kGiB, 2 * kGiB,
                          [] { return hostMemory(64 * kGiB); }, 0);
  governor.setPressure(MemoryPressure::Warning);
  governor.setAutomaticStorageQuota(4 * kGiB, 4 * kGiB, [&] { return expert + ple; });
  governor.enablePressureControl({[&] { return now; }, [] {
    return MemoryPolicyObservation{MemoryPolicyObservation::Pressure::Warning, 0};
  }});
  const auto propose = [&](ExpertPlePolicy &policy) {
    ExpertPleSample sample;
    sample.pleCapacityRows = sample.pleCachedRows = 1000;
    require(!policy.observe(0, false, sample, ple, 4 * kGiB), "empty local sample triggered a transfer");
    for (unsigned window = 1; window <= 2; ++window) {
      sample.units = window * 64; sample.pleReads = window * 100;
      sample.pleReused = window * 32; sample.pleWaitNanoseconds = window * 100000000;
      const auto target = policy.observe(window * 10, false, sample, ple, 4 * kGiB);
      require(window == 1 ? !target : target == ple + 128 * kMiB,
              "local quota observations did not retain the original adaptive thresholds");
    }
  };
  ExpertPlePolicy policy(20 * kGiB, 20 * kGiB, kGiB, 32 * kMiB, 20);
  propose(policy);
  const AutomaticStorageBlock block{AutomaticStorageKind::PleBlock, 1, 711, 128 * kMiB, 0};
  now = 29'999;
  require(!governor.automaticStorageAdmission("ple_optional", block)(128 * kMiB, [] {}) &&
              policy.adjustments() == 0 && !policy.trialPending(),
          "an unreceived adaptive proposal escaped local stability or counted as applied");
  now = 30'000;
  require(governor.evaluate().automaticGrowthReason == "ready", "local adaptive gate did not stabilize");
  const uint64_t before = expert + ple;
  const auto donate = [&] {
    const auto bytes = expertBlocks.back(); expertBlocks.pop_back(); expert -= bytes;
    test::metalStatistics().allocatedBytes -= bytes; test::metalStatistics().deviceCurrentAllocatedBytes -= bytes;
    governor.withholdReleasedAutomaticStorage(bytes);
  };
  donate();
  const auto receive = [&](const AutomaticStorageBlock &candidate) {
    return governor.automaticStorageAdmission("ple_optional", candidate)(128 * kMiB, [&] {
      require(expert + ple == before - 128 * kMiB &&
                  governor.snapshot().automaticStorageWithheldBytes == 128 * kMiB,
              "receiving began before the donor actually returned its vector block");
      require(governor.consumeAutomaticStorageBlock(candidate) && !governor.consumeAutomaticStorageBlock(candidate),
              "an adaptive permit authorized more than one real receiving block");
      ple += 128 * kMiB; test::hostStatistics().fixedBytes = ple;
    });
  };
  require(bool(receive(block)) && expert + ple == before &&
              governor.snapshot().automaticStorageWithheldBytes == 0 && governor.automaticStorageQuotaBytes() == before,
          "a gross-one/net-zero transfer borrowed another category's budget or lost actual accounting");
  policy.applied(30, ple, 1);
  require(policy.adjustments() == 1 && policy.trialPending() && policy.cooldownUntil() == 60,
          "real receiving did not start the existing measured-cost trial/cooldown");
  static_cast<void>(governor.evaluate());
  now = 34'999;
  require(!governor.automaticStorageAdmission("ple_optional", {AutomaticStorageKind::PleBlock, 1, 712, 0, 0})(1, [] {}),
          "a net-zero transfer bypassed the local 5-second block cooldown");
  ExpertPlePolicy failed(20 * kGiB, 20 * kGiB, kGiB, 32 * kMiB, 20);
  propose(failed);
  now = 35'000;
  donate();
  governor.setAllocationDriver([](std::string_view, uint64_t, const std::function<void()> &allocate) {
    allocate(); throw metal::MetalAllocationError("partial adaptive receiver", metal::AllocationFailure::DriverRejected);
  });
  const auto partial = receive({AutomaticStorageKind::PleBlock, 1, 713, 128 * kMiB, 0});
  failed.abandonTransfer(ple);
  failed.stop();
  require(partial.failure == metal::AllocationFailure::DriverRejected && expert + ple == before &&
              governor.snapshot().automaticStorageWithheldBytes == 0 && failed.adjustments() == 0 &&
              !failed.trialPending() && !failed.pendingTarget(),
          "partial receiving fabricated rollback, an applied trial, or reusable old target credit");
  test::hostStatistics() = {};
}

void testRequiredScopeCannotLendItsPermissionToAnAutomaticBlock() {
  test::metalStatistics() = {}; test::hostStatistics() = {};
  allocate(3 * kGiB);
  uint64_t actual = kGiB;
  metal::MetalBackend backend("unused");
  MemoryGovernor governor(backend, 8 * kGiB, 2 * kGiB, 2 * kGiB,
                          [] { return hostMemory(64 * kGiB); }, 0);
  governor.enablePressureControl();
  governor.setAutomaticStorageQuota(2 * kGiB, 2 * kGiB, [&] { return actual; });
  const AutomaticStorageBlock block{AutomaticStorageKind::ExpertBlock, 1, 92, kGiB, 0};
  require(bool(governor.allocationAdmission("restore_weights")(2 * kGiB, [&] {
    const auto optional = governor.automaticStorageAdmission("expert_optional", block)(kGiB, [] {});
    require(optional.failure == metal::AllocationFailure::PlanInsufficient,
            "optional receiver borrowed its required parent's unpaced reservation");
    require(bool(governor.allocationAdmission("managed_buffer")(kGiB, [] { allocate(kGiB); })),
            "required payload waited for optional growth policy");
  })), "required restore did not proceed before the optional stable interval");
}

// Available host memory is what macOS can hand out without swapping: free
// pages and pageable file-backed and purgeable pages, and with compression
// what compressing the anonymous pages frees.
void testHostAvailabilityCountsReclaimablePages() {
  constexpr uint64_t pageSize = 16384;
  auto availablePages = [](const HostMemoryPages &pages) {
    return estimateHostMemory(pages, pageSize, false).available / pageSize;
  };
  // free_count includes the speculative pages; they are file-backed too.
  HostMemoryPages pages{
      .free = 15, .speculative = 5, .fileBacked = 35, .purgeable = 5};
  require(availablePages(pages) == 50,
          "host availability does not match macOS reclaimable accounting");
  pages.free += 5;
  pages.speculative += 5;
  require(availablePages(pages) == 50,
          "speculative file pages were counted twice");
  pages.free -= 10;
  require(availablePages(pages) == 40,
          "anonymous or compressed pages did not consume capacity");
  pages.purgeable = 0;
  require(availablePages(pages) == 35,
          "non-purgeable backing received reclaimable credit");
  // Wired file pages leave external_page_count: pinned pages must reduce
  // available memory rather than count as reclaimable.
  pages.fileBacked -= 10;
  require(availablePages(pages) == 25,
          "wired file pages remained available for new allocations");

  // Clean file cache is reclaimable: reading a file takes no capacity from
  // a later allocation.
  require(availablePages({.free = 75}) == 75 &&
              availablePages({.free = 35, .fileBacked = 40}) == 75,
          "clean file cache reduced allocation capacity");
  // A 64 GB M5 Pro, whose hw.memsize less its VM queues left 1.2 GiB more
  // (the firmware carve-out, tag storage) that no allocation can have.
  require(estimateHostMemory(
              {.free = 2'349'632, .speculative = 90'428,
               .fileBacked = 417'802, .purgeable = 23'495},
              pageSize, false).available == 44'245'008'384ULL,
          "unexpected available memory for a 64 GB snapshot");
  const uint64_t maximum = std::numeric_limits<uint64_t>::max();
  const auto none = [](const HostMemory &host) { return !host.available && !host.spare; };
  require(none(estimateHostMemory({.free = maximum, .fileBacked = 1}, 1, false)) &&
              none(estimateHostMemory({.fileBacked = maximum, .purgeable = 1}, 1, false)) &&
              none(estimateHostMemory({.free = maximum}, pageSize, false)) &&
              none(estimateHostMemory({.free = 1, .speculative = 2}, 1, false)) &&
              estimateHostMemory({.free = maximum}, 1, false).available == maximum &&
              none(estimateHostMemory(pages, 0, false)),
          "invalid host counters or arithmetic overflow did not fail closed");
  // With compression (below critical system pressure), compressing the
  // anonymous pages frees what the compressor would not keep: at its
  // present ratio, at most 2:1, and 2:1 while it holds nothing. The spare
  // memory counts half of that credit.
  const auto compressedPages = [](uint64_t compressor, uint64_t compressed) {
    return estimateHostMemory({.free = 10, .fileBacked = 20, .anonymous = 40,
                               .compressor = compressor, .compressed = compressed},
                              1, true);
  };
  const auto both = [](const HostMemory &host, uint64_t available, uint64_t spare) {
    return host.available == available && host.spare == spare;
  };
  require(both(estimateHostMemory({.free = 10, .fileBacked = 20, .anonymous = 40}, 1, false), 30, 30) &&
              both(compressedPages(0, 0), 50, 40) && both(compressedPages(10, 15), 44, 37) &&
              both(compressedPages(10, 40), 50, 40) && both(compressedPages(10, 10), 30, 30) &&
              both(compressedPages(10, 5), 30, 30),
          "compression credit is not the anonymous pages' savings at the compressor's ratio, at most 2:1, "
          "and half of it in the spare memory");
  require(none(estimateHostMemory({.free = maximum, .anonymous = 2}, 1, true)),
          "compression credit overflowed");
  require(EngineMemoryPolicy::hostAvailableReserveBytes(16 * kGiB) ==
                  16 * kGiB / 10 &&
              EngineMemoryPolicy::hostAvailableReserveBytes(48 * kGiB) ==
                  2 * kGiB &&
              EngineMemoryPolicy::hostAvailableReserveBytes(128 * kGiB) ==
                  2 * kGiB,
          "the macOS reserve is a tenth of a small machine, 2 GiB above 20 GiB");
  require(EngineMemoryPolicy::hostSpareReserveBytes(16 * kGiB) == 2 * kGiB &&
              EngineMemoryPolicy::hostSpareReserveBytes(24 * kGiB) == 2 * kGiB &&
              EngineMemoryPolicy::hostSpareReserveBytes(64 * kGiB) == 64 * kGiB / 100 * 7 &&
              EngineMemoryPolicy::hostSpareReserveBytes(128 * kGiB) == 7 * kGiB,
          "the spare reserve is 7% of the Mac's memory, from 2 to 7 GiB");
}

// A Mac whose recommended working set is numerator/denominator of its memory,
// as Metal reports it (2/3 at 32 GB, 3/4 at 36 GB).
DeviceCapabilities mac(uint64_t physicalGiB, uint64_t numerator,
                       uint64_t denominator) {
  DeviceCapabilities device;
  device.deviceName = "governor-test";
  device.appleGpuFamily = 9;
  device.macosMajor = 26;
  device.macosMinor = 4;
  device.physicalMemoryBytes = physicalGiB * kGiB;
  device.recommendedMaxWorkingSetBytes =
      device.physicalMemoryBytes / denominator * numerator;
  device.maxBufferLengthBytes = device.recommendedMaxWorkingSetBytes;
  device.maxThreadgroupMemoryBytes = 32 * 1024;
  device.maxThreadgroupWidth = 1024;
  device.hasUnifiedMemory = true;
  return device;
}

// Grows one request's KV extent by extent, as PageStorage allocates it, and
// returns the pages the governor granted.
uint32_t grantKvPages(MemoryGovernor &governor,
                      const EngineMemoryBreakdown &budget) {
  uint32_t granted = 0;
  while (granted < budget.kvCapacityPages) {
    const uint32_t pages = budget.kvExtentPages;
    const uint64_t bytes = uint64_t{pages} * budget.kvPageBytes;
    if (!admit(governor, bytes, [bytes] { allocate(bytes); }))
      break;
    granted += pages;
  }
  return granted;
}

// The plan budgets the memory Metal holds outside the backend's buffers
// (pipelines, driver allocations) inside its reserves and advertises the
// context the rest of the budget holds. One request must reach that context
// through the plan's governor whatever part of the reserves this memory
// takes; memory beyond them is still charged, and the device never exceeds
// the hard budget.
void testAdvertisedContextIsGrantable() {
  struct Machine {
    const char *name;
    uint64_t physicalGiB, numerator, denominator;
    kv::Format format;
    uint32_t advertisedTokens;
  };
  for (const Machine &machine :
       {Machine{"32 GB INT8", 32, 2, 3, kv::Format::Int8, 69'625},
        Machine{"36 GB INT8", 36, 3, 4, kv::Format::Int8, 253'945},
        Machine{"36 GB BF16", 36, 3, 4, kv::Format::BFloat16, 129'049}}) {
    // The 27B with its draft and vision tower: 16.2 GiB of weights.
    ModelMemoryProfile model =
        test::modelMemoryProfile(15 * kGiB, kGiB / 2, 7 * kGiB / 10);
    model.targetKvLayout.format = machine.format;
    const EngineMemoryPlan plan = test::requireMemoryPlan(
        mac(machine.physicalGiB, machine.numerator, machine.denominator),
        model);
    const EngineMemoryBreakdown &budget = plan.breakdown();
    require(plan.maximumContextTokens() == machine.advertisedTokens,
            std::string(machine.name) + ": unexpected advertised context");
    const uint64_t reserves =
        budget.pipelineReserveBytes + budget.runtimeOverheadReserveBytes;
    for (const uint64_t untracked :
         {uint64_t{0}, 64 * kMiB, 150 * kMiB, 300 * kMiB, reserves,
          reserves + 256 * kMiB}) {
      // After warmup the weights, the arenas and the request's lane state
      // are the backend's buffers; Metal holds the untracked bytes besides.
      test::metalStatistics() = {};
      test::metalStatistics().allocatedBytes =
          budget.targetWeightsBytes + budget.draftWeightsBytes +
          budget.visionWeightsBytes + budget.sharedPrefillBytes +
          budget.sharedDecodeBytes + budget.laneStateBytes;
      test::metalStatistics().deviceCurrentAllocatedBytes =
          test::metalStatistics().allocatedBytes + untracked;
      metal::MetalBackend backend("unused");
      // The plan's governor, as a start and the runtime oracle build it.
      MemoryGovernor governor(backend, budget, [] { return hostMemory(200 * kGiB); });
      require(governor.snapshot().untrackedReserveBytes == reserves &&
              governor.snapshot().localReservedBytes == 0,
              "configured untracked reserve was conflated with pending local admission");
      const uint32_t granted = grantKvPages(governor, budget);
      const std::string context = std::string(machine.name) + ", " +
                                  std::to_string(untracked / kMiB) +
                                  " MiB untracked: granted " +
                                  std::to_string(granted) + " of " +
                                  std::to_string(budget.kvCapacityPages) +
                                  " KV pages";
      require(test::metalStatistics().deviceCurrentAllocatedBytes <=
                  budget.hardBudgetBytes,
              context + ", beyond the hard budget");
      if (untracked <= reserves)
        require(granted == budget.kvCapacityPages,
                context + ", short of the advertised context");
      else
        require(granted < budget.kvCapacityPages,
                context + ", memory beyond the reserves was not charged");
    }
  }
}

// A reservation holds exactly its bytes under the limit while its
// allocation runs. An admission whose allocation the driver denies reports
// that and releases its reservation; a backend defect propagates and
// releases it too.
void testReservationsAndAdmissionClasses() {
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 64 * 1024;
  const uint64_t limit = 64 * 1024;
  MemoryGovernor governor(backend, limit, hostReserve, hostReserve,
                          [] { return hostMemory(hostReserve + 3 * kGiB); }, 0);
  const metal::AllocationResult held = admit(governor, limit - 1, [&] {
    require(governor.snapshot().localReservedBytes == limit - 1 &&
            governor.snapshot().untrackedReserveBytes == 0 && governor.snapshot().chargedBytes == 0,
            "pending local admission was conflated with actual charge or untracked reserve");
    require(static_cast<bool>(admit(governor, 1)),
            "memory governor held more than the reserved bytes");
    const metal::AllocationResult over = admit(governor, 2);
    require(!over && over.failure == metal::AllocationFailure::EngineBudget &&
                governor.snapshot().deniedReservations == 1,
            "memory governor oversold its limit");
  });
  require(static_cast<bool>(held), "memory governor refused bytes under its limit");
  require(governor.snapshot().localReservedBytes == 0,
          "completed local admission remained reserved");
  const metal::AllocationResult driverDenied = admit(governor, 1024, [] {
    throw metal::MetalAllocationError("injected driver allocation denial");
  });
  require(!driverDenied &&
              driverDenied.failure == metal::AllocationFailure::DriverRejected &&
              admit(governor, limit),
          "driver allocation denial did not release its reservation");
  rejects(
      [&] {
        static_cast<void>(admit(governor, 1024, [] {
          throw metal::MetalBackendError("injected backend defect");
        }));
      },
      "injected backend defect", "admission swallowed a backend defect");
  require(static_cast<bool>(admit(governor, limit)),
          "a backend defect leaked its reservation");
}

// Host memory the engine holds outside Metal (a streamed table's row cache)
// is charged in full beside the backend's buffers.
void testHostHeldBytesAreCharged() {
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 64 * 1024;
  const uint64_t limit = 64 * 1024, held = 16 * 1024;
  MemoryGovernor governor(backend, limit, hostReserve, hostReserve, [] {
    return hostMemory(hostReserve + 3 * kGiB);
  }, 0, held);
  require(static_cast<bool>(admit(governor, limit - held)) && !admit(governor, limit - held + 1),
          "memory governor did not charge the host memory the engine holds");
  require(governor.snapshot().chargedBytes == held, "the host memory the engine holds is not in its charge");
}

// Critical pressure stops all growth. Host headroom below the warning margin
// stops it too, as warning pressure, which sheds cache in paced passes and
// lasts until the headroom crosses the recovery margin; with no request
// waiting for memory, growth that leaves the warning margin free resumes
// before then. Exhausting the engine's limit is not host pressure, and
// missing telemetry fails closed without discarding valid cache.
void testHostHeadroomHysteresis() {
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 64 * 1024;
  const uint64_t limit = 64 * 1024;
  std::optional<uint64_t> available = hostReserve + 3 * kGiB;
  MemoryGovernor governor(backend, limit, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
  // Whether one byte more is admitted.
  const auto admitsMore = [&governor] { return static_cast<bool>(admit(governor, 1)); };
  governor.setPressure(MemoryPressure::Critical);
  require(!admitsMore(), "critical pressure did not stop new growth");
  governor.setPressure(MemoryPressure::Normal);
  require(admitsMore(), "normal pressure did not reopen admission");
  available = hostReserve;
  require(!admitsMore(), "host reserve did not stop unified-memory growth");
  // Reaching the reserve is a warning that sheds cache in paced passes;
  // only the OS critical verdict drops every evictable entry.
  require(governor.snapshot().pressure == MemoryPressure::Warning,
          "reaching the host reserve was treated as critical");
  available = hostReserve / 2;
  require(governor.snapshot().pressure == MemoryPressure::Warning && !admitsMore(),
          "low availability escalated to destructive system pressure");
  available = hostReserve + kGiB / 2;
  require(governor.snapshot().pressure == MemoryPressure::Warning &&
              !governor.snapshot().hostGrowthAllowed && !admitsMore(),
          "low host headroom did not request proactive cache reclaim");
  available = hostReserve + 3 * kGiB / 2;
  require(governor.snapshot().pressure == MemoryPressure::Warning,
          "warning pressure recovered without crossing the hysteresis");
  available = hostReserve + 3 * kGiB;
  require(governor.snapshot().pressure == MemoryPressure::Normal &&
              governor.snapshot().hostGrowthAllowed,
          "host recovery did not reopen admission");
  const metal::AllocationResult full = admit(governor, limit, [&] {
    const MemoryGovernorSnapshot snapshot = governor.snapshot();
    require(snapshot.headroomBytes == 0 && snapshot.hostGrowthAllowed,
            "engine budget exhaustion was confused with host pressure");
  });
  require(static_cast<bool>(full), "engine capacity reservation failed");
  available.reset();
  require(!admitsMore() && !governor.snapshot().hostMeasurementValid,
          "missing host memory telemetry did not fail closed");
  MemoryPressurePolicy missingPolicy;
  const auto missing = governor.snapshot();
  const auto missingDirective = missingPolicy.update(missing, 0.0, false);
  require(missing.pressure == MemoryPressure::Warning && missingDirective &&
              !missingDirective->critical && missingDirective->targetBytes == 0,
          "missing telemetry discarded valid cache");
  governor.setPressure(MemoryPressure::Critical);
  require(missingPolicy.update(governor.snapshot(), 1.0, false).value().critical,
          "missing telemetry hid critical system pressure");
  governor.setPressure(MemoryPressure::Normal);
  available = hostReserve + 3 * kGiB;
  require(governor.snapshot().hostHeadroomBytes == 3 * kGiB,
          "host memory headroom accounting is wrong");
}

// Low reclaimable memory must not reopen growth or suppress the pressure
// reclaimer; reclaimable file cache can reopen it.
void testReclaimablePagesReopenGrowth() {
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 64 * 1024;
  const uint64_t limit = 64 * 1024;
  std::optional<uint64_t> available = hostReserve + 3 * kGiB;
  MemoryGovernor governor(backend, limit, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
  const auto admitsMore = [&governor] { return static_cast<bool>(admit(governor, 1)); };
  HostMemoryPages pressurePages{.free = hostReserve, .fileBacked = kGiB / 2};
  available = estimateHostMemory(pressurePages, 1, false).available;
  MemoryPressurePolicy hostPolicy;
  const auto hostDirective = hostPolicy.update(governor.snapshot(), 0.0, false);
  require(!admitsMore() && governor.snapshot().pressure == MemoryPressure::Warning &&
              hostDirective && !hostDirective->critical &&
              hostDirective->targetBytes == kGiB / 2,
          "low reclaimable memory bypassed bounded pressure recovery");
  pressurePages.fileBacked = 3 * kGiB;
  available = estimateHostMemory(pressurePages, 1, false).available;
  require(governor.snapshot().hostGrowthAllowed && admitsMore() &&
              !hostPolicy.update(governor.snapshot(), 1000.0, false),
          "reclaimable host recovery did not reopen normal admission");
}

// Reading the governor moves nothing: /status and the engine's growth
// queries see what it would decide, and only a reservation or the control
// pass moves the host's hysteresis. A recovery that only a read saw leaves
// growth held for a waiting request once the host falls back inside the
// recovery margin.
void testSnapshotLeavesTheHysteresis() {
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 64 * 1024;
  std::optional<uint64_t> available = hostReserve + kGiB / 2;
  MemoryGovernor governor(backend, 64 * 1024, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
  governor.setRequestWaiting(true);
  require(!admit(governor, 1), "growth inside the warning margin was admitted");
  available = hostReserve + 3 * kGiB;
  require(governor.snapshot().hostGrowthAllowed, "a read did not see the host recover");
  available = hostReserve + 3 * kGiB / 2;
  require(!governor.snapshot().hostGrowthAllowed && !admit(governor, 1),
          "a read moved the host's hysteresis");
  available = hostReserve + 3 * kGiB;
  require(governor.evaluate().hostGrowthAllowed,
          "the control pass did not see the host recover");
  available = hostReserve + 3 * kGiB / 2;
  require(governor.snapshot().hostGrowthAllowed && admit(governor, 1),
          "the control pass did not move the host's hysteresis");
}

// The host's pressure asks, once per settled measurement and for at most the
// warning margin, for what measured headroom lacks of the warning margin, or
// of the recovery margin while a request waits for memory; critical pressure
// evicts everything.
void testPressurePolicyDirectives() {
  MemoryPressurePolicy policy;
  MemoryGovernorSnapshot policySnapshot;
  policySnapshot.pressure = MemoryPressure::Warning;
  policySnapshot.hostMeasurementValid = true;
  policySnapshot.hostHeadroomBytes = kGiB / 2;
  const auto firstDirective = policy.update(policySnapshot, 0.0, false);
  require(firstDirective && !firstDirective->critical &&
              firstDirective->targetBytes == kGiB / 2,
          "warning pressure ignored measured headroom");
  require(policy.update(policySnapshot, 500.0, false).value().targetBytes == 0 &&
              policy.update(policySnapshot, 999.0, false).value().targetBytes == 0,
          "warning pressure reclaimed again before telemetry settled");
  // Another application consumed more memory in the SAME warning episode.
  // Earlier reclaimed bytes must not offset this new deficit.
  policySnapshot.hostHeadroomBytes = kGiB / 4;
  require(policy.update(policySnapshot, 1000.0, false).value().targetBytes == 3 * kGiB / 4,
          "persistent warning did not request a new bounded shrink pass");
  policySnapshot.hostHeadroomBytes = 3 * kGiB / 4;
  require(policy.update(policySnapshot, 2000.0, false).value().targetBytes == kGiB / 4,
          "pressure recovery ignored the current smaller deficit");
  policySnapshot.pressure = MemoryPressure::Normal;
  require(!policy.update(policySnapshot, 2100.0, false),
          "normal pressure requested cache reclaim");
  policySnapshot.pressure = MemoryPressure::Warning;
  require(policy.update(policySnapshot, 2101.0, false).value().targetBytes == kGiB / 4,
          "a new pressure episode inherited an old cooldown");
  // Past the warning margin the hysteresis still holds the pressure. A pass
  // nothing waits for then evicts nothing; one a request waits for frees
  // what the recovery margin lacks.
  policySnapshot.hostHeadroomBytes = 3 * kGiB / 2;
  const auto idle = policy.update(policySnapshot, 3101.0, false);
  require(idle && !idle->critical && idle->targetBytes == 0,
          "a shrink nothing waited for went past the warning margin");
  const auto waited = policy.update(policySnapshot, 4101.0, true);
  require(waited && waited->targetBytes == kGiB / 2,
          "a waiting request's shrink stopped short of the recovery margin");
  // The newest publication is what a follow-up resumes from; rebuilding it
  // costs a whole prefill, so a shrink nothing is waiting for leaves it and
  // takes the rest short of what is in use, which includes the replay point
  // a running request publishes for its follow-up. A waiting request
  // outranks both, and Critical takes all.
  policySnapshot.hostHeadroomBytes = kGiB / 4;
  const auto speculative = policy.update(policySnapshot, 6101.0, false);
  require(speculative && speculative->targetBytes == 3 * kGiB / 4 &&
              speculative->keepResumePoint && speculative->upTo == ReclaimClass::Ordinary,
          "a speculative shrink discarded the resume point or took what is in use");
  const auto demanded = policy.update(policySnapshot, 7101.0, true);
  require(demanded && demanded->targetBytes == kGiB && !demanded->keepResumePoint &&
              demanded->upTo == ReclaimClass::InUse,
          "a waiting request could not reach the resume point or what is in use");
  policySnapshot.pressure = MemoryPressure::Critical;
  const auto criticalDirective = policy.update(policySnapshot, 7102.0, false);
  require(criticalDirective && criticalDirective->critical &&
              !criticalDirective->keepResumePoint,
          "critical pressure did not request aggressive reclaim");
}

// A request can need more than the host headroom above the warning margin
// while the idle headroom still clears it. Its refusal must start the paced
// reclaim it waits for, after which it fits.
void testHostRefusalStartsReclaim() {
  test::metalStatistics() = {};
  test::metalStatistics().allocatedBytes = 20 * kGiB;
  test::metalStatistics().deviceCurrentAllocatedBytes = 20 * kGiB;
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 2 * kGiB;
  const uint64_t laneState = 350'224'384;
  std::optional<uint64_t> available = hostReserve + 64 * kGiB;
  MemoryGovernor governor(backend, 40 * kGiB, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
  const metal::AllocationResult beyondLimit = admit(governor, 40 * kGiB);
  require(!beyondLimit && beyondLimit.failure == metal::AllocationFailure::EngineBudget,
          "an engine budget refusal was taken for host pressure");
  // The host refuses the same probe once it has 1.2 GiB of room, and that
  // is the cause reported; a probe beyond the limit holds no host pressure.
  available = hostReserve + kGiB + 200 * kMiB;
  const metal::AllocationResult shared = admit(governor, 40 * kGiB);
  require(!shared && shared.failure == metal::AllocationFailure::HostPressure &&
              governor.snapshot().pressure == MemoryPressure::Normal,
          "a refusal the host shares was reported as the engine's");
  const metal::AllocationResult beyondHost = admit(governor, laneState);
  require(!beyondHost && beyondHost.failure == metal::AllocationFailure::HostPressure,
          "a request beyond the host headroom was admitted");
  // The control pass finds the refused request waiting.
  governor.setRequestWaiting(true);
  const MemoryGovernorSnapshot refused = governor.snapshot();
  MemoryPressurePolicy policy;
  const std::optional<MemoryReclaimDirective> directive = policy.update(refused, 0.0, true);
  require(refused.pressure == MemoryPressure::Warning &&
              !refused.hostGrowthAllowed && directive && !directive->critical &&
              !directive->keepResumePoint &&
              directive->targetBytes == kGiB - 200 * kMiB,
          "a request-sized host refusal did not start the paced reclaim");
  // The reclaim reaches the recovery margin, and the request fits.
  *available += directive->targetBytes;
  require(static_cast<bool>(admit(governor, laneState)) &&
              governor.snapshot().pressure == MemoryPressure::Normal,
          "the waiting request did not fit after the reclaim");
}

// A refusal the host shares with the engine's limit is the host's: it lifts
// with the host's pressure, the limit's only once memory is freed. For a
// request in service the host refuses only under critical pressure.
void testHostRefusalComesBeforeTheEngineLimit() {
  test::metalStatistics() = {};
  test::metalStatistics().allocatedBytes = 14 * kGiB;
  test::metalStatistics().deviceCurrentAllocatedBytes = 14 * kGiB;
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 2 * kGiB;
  std::optional<uint64_t> available = hostReserve + kGiB / 2;
  MemoryGovernor governor(backend, 15 * kGiB, hostReserve, hostReserve, [&available] { return hostMemory(available); }, 0);
  const metal::AllocationResult shared = admit(governor, 2 * kGiB);
  require(!shared && shared.failure == metal::AllocationFailure::HostPressure,
          "a refusal the host shares was reported as the engine's");
  governor.setServing(true);
  const metal::AllocationResult inService = admit(governor, 2 * kGiB);
  require(!inService && inService.failure == metal::AllocationFailure::EngineBudget,
          "the host's margins refused a request in service");
  governor.setPressure(MemoryPressure::Critical);
  const metal::AllocationResult critical = admit(governor, 2 * kGiB);
  require(!critical && critical.failure == metal::AllocationFailure::HostPressure,
          "critical pressure was reported as the engine's limit");
}

// The paced passes up to the next measurement continue what transfers held
// back of a pass's target, less what each releases, until it is met. A new
// measurement replaces it, and every critical pass evicts everything afresh.
void testPolicyContinuesHeldBackTarget() {
  MemoryPressurePolicy policy;
  MemoryGovernorSnapshot pressure{.pressure = MemoryPressure::Warning,
                                  .hostMeasurementValid = true,
                                  .hostHeadroomBytes = kHostRecoveryMarginBytes - 300};
  const auto pass = [&](double now, MemoryReclaimResult result) {
    const MemoryReclaimDirective directive = policy.update(pressure, now, true).value();
    policy.reclaimed(directive, result);
    return directive.targetBytes;
  };
  constexpr MemoryReclaimResult none{};
  require(pass(0.0, {100, ReclaimOutcome::Pending}) == 300 &&
              pass(100.0, {50, ReclaimOutcome::Pending}) == 200 &&
              pass(200.0, {0, ReclaimOutcome::Met}) == 150 && pass(300.0, none) == 0,
          "the paced passes did not continue a held-back target until it was met");
  require(pass(1000.0, {0, ReclaimOutcome::Pending}) == 300, "the pass was not measured");
  pressure.hostHeadroomBytes = kHostRecoveryMarginBytes - 100;
  require(pass(2000.0, none) == 100 && pass(2100.0, none) == 0,
          "a measurement did not replace the held-back target");
  pressure.pressure = MemoryPressure::Critical;
  require(policy.update(pressure, 2150.0, true).value().critical,
          "critical pressure kept the serving footprint");
  static_cast<void>(pass(2200.0, {0, ReclaimOutcome::Pending}));
  pressure.pressure = MemoryPressure::Warning;
  require(pass(2300.0, none) == 0, "evicting everything was continued after critical pressure");
}

// With nothing left to reclaim, the hold for the recovery margin could only be
// lifted by other applications: every request, however small, would wait.
// While reclaim reports that, growth that clears the warning margin proceeds
// and a request beyond it holds no other. A pass that finds memory again, or
// a new episode of host pressure, brings the hold back; a pass that macOS's
// warning alone asked for, while the host had room, waives nothing.
void testExhaustedReclaimWaivesTheHold() {
  test::metalStatistics() = {};
  test::metalStatistics().allocatedBytes = 12 * kGiB;
  test::metalStatistics().deviceCurrentAllocatedBytes = 12 * kGiB;
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 2 * kGiB;
  std::optional<uint64_t> available = hostReserve + kGiB + kGiB / 2;
  MemoryGovernor governor(backend, 40 * kGiB, hostReserve, hostReserve, [&available] { return hostMemory(available); }, 0);
  governor.setRequestWaiting(true);
  const metal::AllocationResult pastMargin = admit(governor, kGiB);
  require(!pastMargin && pastMargin.failure == metal::AllocationFailure::HostPressure,
          "growth past the warning margin was admitted");
  governor.reclaimed(ReclaimOutcome::Untargeted);
  require(!admit(governor, 100 * kMiB) && !governor.snapshot().hostGrowthAllowed,
          "the host refusal did not hold growth for the recovery margin");
  governor.reclaimed(ReclaimOutcome::Exhausted);
  require(governor.snapshot().hostGrowthAllowed && admit(governor, 100 * kMiB),
          "growth within the warning margin still waited after reclaim was exhausted");
  const metal::AllocationResult waived = admit(governor, kGiB);
  require(!waived && waived.failure == metal::AllocationFailure::HostPressure &&
              governor.snapshot().pressure == MemoryPressure::Warning &&
              admit(governor, 100 * kMiB),
          "a request past the warning margin was admitted or held the others");
  governor.reclaimed(ReclaimOutcome::Pending);
  require(!admit(governor, 100 * kMiB), "the hold did not return with memory to reclaim");
  governor.reclaimed(ReclaimOutcome::Exhausted);
  available = hostReserve + 3 * kGiB;
  require(governor.evaluate().pressure == MemoryPressure::Normal, "the host did not recover");
  available = hostReserve + kGiB + kGiB / 2;
  require(!admit(governor, kGiB) && !admit(governor, 100 * kMiB),
          "an earlier episode's exhausted reclaim waived the hold");
  available = hostReserve + 3 * kGiB;
  governor.setPressure(MemoryPressure::Warning);
  require(governor.evaluate().pressure == MemoryPressure::Warning, "macOS's warning was ignored");
  governor.reclaimed(ReclaimOutcome::Exhausted);
  governor.setPressure(MemoryPressure::Normal);
  available = hostReserve + kGiB / 2;
  require(governor.evaluate().pressure == MemoryPressure::Warning, "the host did not run short");
  available = hostReserve + kGiB + kGiB / 2;
  require(!admit(governor, 100 * kMiB),
          "a pass that only macOS's warning asked for waived the next episode's hold");
}

// Growth waits for the recovery margin only while a request waits for
// memory, so that what reclaim frees goes to it. With none waiting, paced
// reclaim only restores the warning margin, and growth that clears that
// margin proceeds while the host's headroom sits between the margins.
void testOnlyAWaitingRequestHoldsGrowth() {
  test::metalStatistics() = {};
  test::metalStatistics().allocatedBytes = 12 * kGiB;
  test::metalStatistics().deviceCurrentAllocatedBytes = 12 * kGiB;
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 2 * kGiB;
  std::optional<uint64_t> available = hostReserve + kGiB / 2;
  MemoryGovernor governor(backend, 40 * kGiB, hostReserve, hostReserve, [&available] { return hostMemory(available); }, 0);
  require(!admit(governor, 100 * kMiB), "growth inside the warning margin was admitted");
  available = hostReserve + kGiB + kGiB / 2;
  require(governor.evaluate().pressure == MemoryPressure::Warning &&
              governor.snapshot().hostGrowthAllowed && admit(governor, 100 * kMiB),
          "growth that cleared the warning margin waited with no request waiting");
  governor.setRequestWaiting(true);
  require(!governor.snapshot().hostGrowthAllowed && !admit(governor, 100 * kMiB),
          "growth did not wait for the recovery margin while a request waited");
  governor.setRequestWaiting(false);
  require(governor.snapshot().hostGrowthAllowed && admit(governor, 100 * kMiB),
          "the hold outlived the waiting request");
}

// A pass with no target leaves nothing to settle: a request that starts
// waiting right after it gets its own pass at once, and that one, which has
// a target, waits for the host's counters before the next.
void testOnlyATargetedPassWaitsToSettle() {
  MemoryPressurePolicy policy;
  const MemoryGovernorSnapshot between{.pressure = MemoryPressure::Warning,
                                       .hostMeasurementValid = true,
                                       .hostHeadroomBytes = kGiB + kGiB / 2};
  const auto idle = policy.update(between, 0.0, false);
  const auto waited = policy.update(between, 10.0, true);
  require(idle && idle->targetBytes == 0 && waited && waited->targetBytes == kGiB / 2,
          "an empty pass made a waiting request wait for its own");
  require(policy.update(between, 20.0, true).value().targetBytes == 0 &&
              policy.update(between, 1010.0, true).value().targetBytes == kGiB / 2,
          "a pass that freed memory did not wait for the host to settle");
}

// Host pressure holds back growth that nothing in flight depends on. What a
// request in service needs is granted while the host is short, into its
// reserve too: holding it back would strand the request and the memory it
// already has. Only the engine's limit and critical pressure refuse it, and
// without the mark the margins apply as before.
void testRequestInServiceGrowsThroughHostPressure() {
  test::metalStatistics() = {};
  test::metalStatistics().allocatedBytes = 14 * kGiB;
  test::metalStatistics().deviceCurrentAllocatedBytes = 14 * kGiB;
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 2 * kGiB;
  std::optional<uint64_t> available = hostReserve + kGiB / 2;
  MemoryGovernor governor(backend, 15 * kGiB, hostReserve, hostReserve, [&available] { return hostMemory(available); }, 0);
  const auto grow = [&](uint64_t bytes) {
    return admit(governor, bytes, [bytes] { allocate(bytes); });
  };
  const metal::AllocationResult idle = grow(400 * kMiB);
  require(!idle && idle.failure == metal::AllocationFailure::HostPressure &&
              !governor.snapshot().hostGrowthAllowed,
          "growth that nothing in service needs cleared no warning margin");

  governor.setServing(true);
  require(grow(400 * kMiB) && !governor.snapshot().hostGrowthAllowed,
          "a request in service waited for the warning margin");
  available = hostReserve / 2;
  require(static_cast<bool>(grow(100 * kMiB)),
          "a request in service waited for the host's reserve");
  const metal::AllocationResult pastLimit = grow(kGiB);
  require(!pastLimit && pastLimit.failure == metal::AllocationFailure::EngineBudget,
          "a request in service grew past the engine's limit");
  governor.setPressure(MemoryPressure::Critical);
  const metal::AllocationResult critical = grow(100 * kMiB);
  require(!critical && critical.failure == metal::AllocationFailure::HostPressure,
          "critical pressure admitted a request in service");
  governor.setPressure(MemoryPressure::Normal);
  require(static_cast<bool>(grow(100 * kMiB)),
          "a request in service stayed refused after critical pressure");

  governor.setServing(false);
  available = hostReserve + kGiB / 2;
  const metal::AllocationResult afterService = grow(100 * kMiB);
  require(!afterService && afterService.failure == metal::AllocationFailure::HostPressure,
          "the mark of a request in service outlived it");
}

// A Mac shared with other applications: their anonymous memory makes the
// available memory, which counts compressing it 2:1, much larger than the
// spare memory, which counts half of that credit. Growth that no request in
// service needs keeps the warning margin above the spare reserve of the spare
// memory, and the paced reclaim returns cache toward it, while memory a
// request in service needs still grows and the governor starts against the
// smaller reserve macOS keeps of the available memory.
void testOptionalGrowthKeepsTheSpareReserve() {
  test::metalStatistics() = {};
  test::metalStatistics().allocatedBytes = 20 * kGiB;
  test::metalStatistics().deviceCurrentAllocatedBytes = 20 * kGiB;
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 2 * kGiB, spareReserve = 7 * kGiB;
  // 4 GiB free and file-backed, and 40 GiB of other applications' anonymous
  // memory: 20 GiB more if compressed 2:1, of which spare counts 10.
  HostMemory host{24 * kGiB, 14 * kGiB};
  MemoryGovernor governor(backend, 100 * kGiB, hostReserve, spareReserve, [&host] {
    return std::optional<HostMemory>(host);
  }, 0);
  const auto grow = [&](uint64_t bytes) {
    return admit(governor, bytes, [bytes] { allocate(bytes); });
  };
  MemoryGovernorSnapshot snapshot = governor.snapshot();
  require(snapshot.hostAvailableBytes == 24 * kGiB && snapshot.hostSpareBytes == 14 * kGiB &&
              snapshot.hostSpareReserveBytes == spareReserve &&
              snapshot.hostHeadroomBytes == 7 * kGiB && snapshot.hostGrowthAllowed,
          "the headroom is not the spare memory above the spare reserve");
  require(static_cast<bool>(grow(5 * kGiB)), "growth within the spare headroom was refused");
  // The growth came out of the host's free memory.
  host = {19 * kGiB, 9 * kGiB};
  require(!grow(kGiB + kGiB / 2),
          "growth no request needs did not keep the warning margin above the spare reserve");
  host = {17 * kGiB + kGiB / 2, 7 * kGiB + kGiB / 2};
  snapshot = governor.evaluate();
  MemoryPressurePolicy policy;
  const auto directive = policy.update(snapshot, 0.0, false);
  require(snapshot.pressure == MemoryPressure::Warning && !snapshot.hostGrowthAllowed &&
              directive && directive->targetBytes == kGiB - snapshot.hostHeadroomBytes,
          "the paced reclaim did not restore the warning margin above the spare reserve");
  governor.setServing(true);
  require(static_cast<bool>(grow(4 * kGiB)),
          "a request in service waited for the spare reserve");
  governor.setServing(false);
  // A start needs only the reserve macOS keeps of the available memory.
  host = {3 * kGiB, 0};
  MemoryGovernor starting(backend, 100 * kGiB, hostReserve, spareReserve, [&host] {
    return std::optional<HostMemory>(host);
  }, 0);
  host = {hostReserve, 0};
  rejects([&] { MemoryGovernor refused(backend, 100 * kGiB, hostReserve, spareReserve, [&host] {
            return std::optional<HostMemory>(host);
          }, 0); },
          "host available memory does not satisfy the system reserve",
          "a start below the reserve macOS keeps was admitted");
}

// macOS's warning asks caches to give memory back. While it lasts, growth
// that no request in service needs stops whatever the host's headroom, and
// the snapshot says so, while what a request in service needs still grows.
// Every paced pass returns the warning margin's worth of cache and keeps the
// resume point, even for a waiting request, which the warning holds back
// too; only a pass a request waits for takes what is in use. Once the level
// is normal again, growth resumes and no pass runs.
void testSystemWarningGivesCacheBack() {
  test::metalStatistics() = {};
  test::metalStatistics().allocatedBytes = 12 * kGiB;
  test::metalStatistics().deviceCurrentAllocatedBytes = 12 * kGiB;
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 2 * kGiB;
  const std::optional<uint64_t> available = hostReserve + 16 * kGiB;
  MemoryGovernor governor(backend, 40 * kGiB, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
  const auto grow = [&](uint64_t bytes) {
    return admit(governor, bytes, [bytes] { allocate(bytes); });
  };
  governor.setPressure(MemoryPressure::Warning);
  const MemoryGovernorSnapshot warned = governor.evaluate();
  const metal::AllocationResult cache = grow(100 * kMiB);
  require(!cache && cache.failure == metal::AllocationFailure::HostPressure &&
              warned.pressure == MemoryPressure::Warning && !warned.hostGrowthAllowed,
          "growth that no request in service needs went on while macOS warned");
  governor.setServing(true);
  require(static_cast<bool>(grow(100 * kMiB)), "macOS's warning refused a request in service");
  governor.setServing(false);

  MemoryPressurePolicy policy;
  const auto first = policy.update(warned, 0.0, false);
  require(first && !first->critical && first->targetBytes == kHostWarningMarginBytes &&
              first->keepResumePoint && first->upTo == ReclaimClass::Ordinary,
          "a pass under macOS's warning did not return the warning margin's worth of cache, "
          "keeping the resume point and what is in use");
  require(policy.update(warned, 500.0, false).value().targetBytes == 0 &&
              policy.update(warned, 1000.0, false).value().targetBytes == kHostWarningMarginBytes,
          "the passes under macOS's warning did not keep the pace");
  const auto waited = policy.update(warned, 2000.0, true);
  require(waited && waited->targetBytes == kHostWarningMarginBytes && waited->keepResumePoint &&
              waited->upTo == ReclaimClass::InUse,
          "a pass under macOS's warning dropped the resume point for a request it holds back, "
          "or kept what is in use from it");

  governor.setPressure(MemoryPressure::Normal);
  const MemoryGovernorSnapshot normal = governor.evaluate();
  require(normal.pressure == MemoryPressure::Normal && normal.hostGrowthAllowed &&
              grow(100 * kMiB) && !policy.update(normal, 3000.0, false),
          "growth or the passes did not follow macOS back to normal pressure");
}

void testNonBackendHostSurvivesUnpublishedTrackedGrowth() {
  test::metalStatistics() = {}; test::hostStatistics() = {.fixedBytes = 80 * kMiB};
  metal::MetalBackend backend("unused");
  MemoryGovernor governor(backend, 400 * kMiB, 10 * kMiB, 10 * kMiB,
                          [] { return hostMemory(100 * kGiB); }, 0);
  governor.enablePressureControl();
  governor.setNonBackendHostHeld([] { return 20 * kMiB; });
  require(governor.snapshot().chargedBytes == 100 * kMiB && governor.actualBytes() == 100 * kMiB,
          "non-backend 20 plus tracked 80 was not charged once");
  require(bool(governor.allocationAdmission("required_host")(40 * kMiB, [&] {
    test::hostStatistics().fixedBytes = 120 * kMiB;
    governor.publishAccounting();
    require(governor.actualBytes() == 140 * kMiB && governor.snapshot().localReservedBytes == 0,
            "actual host allocation retained a duplicate peak reservation");
  })), "local host growth was refused");
  test::hostStatistics().temporaryBytes = 30 * kMiB;
  require(governor.snapshot().chargedBytes == 170 * kMiB, "temporary host storage replaced fixed ownership");
  test::hostStatistics().temporaryBytes = 0; test::hostStatistics().fixedBytes = 90 * kMiB;
  governor.publishAccounting();
  require(governor.actualBytes() == 110 * kMiB, "partial release fabricated full host rollback");
  test::hostStatistics() = {};
}

} // namespace

int main() {
  try {
    testRealPlePrefetchDrainCannotKeepAConsumedUnsafePermit();
    testAutomaticPayloadKeepsItsPaidPeakSeparateFromExternalPressure();
    testAutomaticPayloadRechecksAfterItsBlockPermitWasConsumed();
    testWarningRecoversByMeasuredHeadroomAndRechecksTheRealReceiver();
    testCriticalBeforeRealPleReceiptRetainsItsHostPressureCause();
    testLocalAdaptiveTransferRequiresActualDonationAndOnePacedReceipt();
    testLocalRestoreSelectsRequiredFloorBeforePayloadAndQuota();
    testLocalAutomaticBlockPacingAndFreshReceiverObservation();
    testReleasedAutomaticStorageDoesNotExpandElasticHeadroom();
    testNestedElasticAdmissionRetainsWithheldQuota();
    testAutomaticDonationCannotBeRefilledByTheOtherStore();
    testAutomaticPoolQuotaIsIndependentOfGlobalHeadroom();
    testAutomaticQuotaCountsFinalStoreContributionNotTransientWorkspace();
    testRequiredScopeCannotLendItsPermissionToAnAutomaticBlock();
    {
      test::metalStatistics() = {};
      allocate(kGiB);
      metal::MetalBackend backend("unused");
      MemoryGovernor governor(backend, 4 * kGiB, kGiB, kGiB, [] { return hostMemory(100 * kGiB); }, 0, 512 * kMiB);
      const auto before = governor.snapshot();
      governor.setHostHeldBytes(kGiB);
      require(governor.snapshot().chargedBytes == before.chargedBytes + 512 * kMiB,
              "PLE host growth disappeared from governor accounting");
      require(!admit(governor, 3 * kGiB), "host PLE bytes were ignored by engine admission");
      governor.setHostHeldBytes(256 * kMiB);
      require(governor.snapshot().chargedBytes == kGiB + 256 * kMiB,
              "PLE donation was not returned to the governor");
    }
    testHostAvailabilityCountsReclaimablePages();
    testAdvertisedContextIsGrantable();
    testReservationsAndAdmissionClasses();
    testHostHeldBytesAreCharged();
    testHostHeadroomHysteresis();
    testReclaimablePagesReopenGrowth();
    testSnapshotLeavesTheHysteresis();
    testPressurePolicyDirectives();
    testHostRefusalStartsReclaim();
    testHostRefusalComesBeforeTheEngineLimit();
    testPolicyContinuesHeldBackTarget();
    testExhaustedReclaimWaivesTheHold();
    testOnlyAWaitingRequestHoldsGrowth();
    testOnlyATargetedPassWaitsToSettle();
    testRequestInServiceGrowsThroughHostPressure();
    testOptionalGrowthKeepsTheSpareReserve();
    testSystemWarningGivesCacheBack();
    testNonBackendHostSurvivesUnpublishedTrackedGrowth();
    std::cout << "memory governor tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "memory governor tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
