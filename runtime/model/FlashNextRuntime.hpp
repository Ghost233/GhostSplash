#pragma once
#include "model/ModelFactory.hpp"
namespace splash::model {
inline constexpr uint32_t kFlashNextPrefillRows = 512;
[[nodiscard]] ModelMemoryPlan flashNextRuntimeMemory(const FlashNextWeights &, const ops::ExecutionPlans &);
class FlashNextRuntime final : public RuntimeModel {
public:
  explicit FlashNextRuntime(RuntimeContext);
  ~FlashNextRuntime() override;
  void checkHealth() override;
  bool supportsPrefixCache() const noexcept override { return false; }
  bool supportsIdleWeightRelease() const noexcept override { return false; }
  bool supportsConstrainedDecoding() const noexcept override { return false; }
  uint32_t maximumPrefillTokenBudget() const noexcept override { return kFlashNextPrefillRows; }
  uint32_t recommendedWarmupPrefillRows() const noexcept override { return 1; }
  StateAdmission begin(const ModelRequest &) override;
  void suspend(uint64_t) override;
  StateAdmission resume(const ModelRequest &) override;
  std::unique_ptr<StateRestore> beginRestore(uint64_t, uint32_t, std::shared_ptr<const CompositeState>, bool,
                                            std::function<void()>) override;
  void setDraftContextPlan(uint64_t, DraftContextPlan) override;
  std::unique_ptr<ModelBatchTicket> submit(const BatchPlan &, std::span<const ModelBatchItem>, std::function<void()>) override;
  std::shared_ptr<const CompositeState> snapshot(uint64_t) override;
  uint64_t snapshotBytes() const noexcept override;
  uint64_t reclaimIdleState(bool, IdleMemory) noexcept override;
  std::optional<std::string> provideMask(uint64_t, std::span<const uint32_t>) override;
  void end(uint64_t) override;
  WarmupStepResult warmupPrefill(uint32_t) override;
  WarmupStepResult warmupDecodeBatch(uint32_t) override;
  WarmupStepResult warmupCompositeStateRestore() override;
  ModelMemoryActual actualRuntimeMemory() const override;
  ModelTelemetry telemetry() const noexcept override;
private:
  std::unique_ptr<ModelBatchTicket> submitSynchronous(const BatchPlan &,std::span<const ModelBatchItem>);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace splash::model
