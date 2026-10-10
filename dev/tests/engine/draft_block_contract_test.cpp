#include "model/DFlashDraft.hpp"
#include "metal/abi/Sampling.h"
#include "ops/Linear.hpp"
#include "ops/PagedAttention.hpp"

#include <iostream>
#include <stdexcept>

#ifndef TEST_EXPECTED_DRAFT_BLOCK
#error "the test must state its expected block independently of the ABI"
#endif
static_assert(SPLASH_DRAFT_QUERY_ROWS == TEST_EXPECTED_DRAFT_BLOCK,
              "host/shader draft rows do not match the requested build variant");

int main() {
  try {
    using splash::model::DFlashDraftLayout;
    using splash::model::DraftKind;
    using splash::model::ExecutionLimits;
    const auto require = [](bool condition, const char *message) {
      if (!condition) throw std::runtime_error(message);
    };
    constexpr uint32_t block = TEST_EXPECTED_DRAFT_BLOCK;
    require(ExecutionLimits::draftQueryRows == block && ExecutionLimits::targetVerifyRows == block,
            "model and Metal row contracts disagree");
    require(ExecutionLimits::draftProposalTokens == block - 1 &&
            SPLASH_DRAFT_PROPOSAL_TOKENS == block - 1,
            "the number of proposals is not block minus the anchor");
    require(ExecutionLimits::maximumStepTokens == block + 1,
            "terminal token allowance is missing from the output bound");
    DFlashDraftLayout none, standard, dflash2;
    standard.layers = dflash2.layers = 6;
    standard.kind = DraftKind::Standard;
    require(none.queryRows() == 0 && none.proposalTokens() == 0 && none.supportsExecutionBlock(),
            "target-only execution reports a speculative proposal");
    require(standard.queryRows() == block && standard.proposalTokens() == block - 1 &&
            standard.supportsExecutionBlock(), "standard DFlash refuses its compiled block");
    require(dflash2.supportsExecutionBlock() == (block == 8),
            "DFlash2 was silently matched to a non-eight-row executable");
    for (const uint32_t lanes : {1u, 2u, 4u}) {
      require(lanes * block <= 64 && lanes * (block - 1) < lanes * block,
              "batch proposal and verify counts overlap the next lane");
      require(SPLASH_UNIFORM_CORRECTION == 1 + 2 * (block - 1) &&
              SPLASH_SAMPLING_UNIFORMS == 2 * block,
              "proposal/acceptance/correction uniforms alias");
    }
    splash::DeviceCapabilities device;
    device.appleGpuFamily = 10;
    device.gpuCoreCount = 20;
    splash::ops::Linear linear(device);
    for (const uint32_t lanes : {1u, 2u, 4u}) {
      const uint32_t rows = lanes * block;
      const uint32_t targetStorageRows = linear.decodeStorageRows(
          rows, {2560, 2560});
      const uint32_t storageLanes = targetStorageRows / block;
      require(storageLanes >= lanes && storageLanes <= ExecutionLimits::maximumBatchWidth,
              "4B context storage cannot be represented by arena lane slices");
      // The context FC reads eight captured 2560-wide target layers. Its
      // input view, hidden norm and every proposal projection need padded
      // capacity even though the logical row count remains lanes * block.
      for (const splash::ops::ProjectionShape shape : {
          splash::ops::ProjectionShape{2560, 20480}, {2048, 2560}, {6144, 2560},
          {2560, 4096}, {9216, 2560}, {2560, 9216}}) {
        const uint32_t requiredRows = linear.decodeStorageRows(rows, shape);
        require(storageLanes * block >= requiredRows,
                "4B target storage view does not hold a drafter projection tile");
        require(uint64_t{storageLanes} * block * shape.inputSize * 2 >=
                    uint64_t{requiredRows} * shape.inputSize * 2,
                "drafter FC/proposal input capacity is below its physical tile");
      }
      const auto plan = linear.plan({{2560, 2560}, rows, splash::ops::LinearPhase::Decode,
                                     splash::ops::LinearEpilogue::None});
      require(plan.storageRows() >= rows && plan.storageRows() <= 4 * block,
              "projection padding exceeds the variant's complete decode arena");
      require(block == 8 || plan.input() == splash::ops::LinearInput::Plain,
              "a non-eight variant selected an eight-row register activation table");
      require(plan.storageRows() == (rows <= 8 ? 8u : rows <= 16 ? 16u : rows <= 32 ? 32u : 64u),
              "block projection did not reserve its legal staged row tile");
      if (block == 16 && lanes == 4)
        require(plan.storageRows() == 64, "block16 concurrency4 did not allocate the M64 tile");
    }
    const std::array<uint32_t, 1> histories{0};
    for (const auto format : {splash::kv::Format::Int8, splash::kv::Format::BFloat16,
                             splash::kv::Format::Int4Hadamard, splash::kv::Format::Q4_0Hadamard,
                             splash::kv::Format::Q4_0Unrotated}) {
      const splash::kv::Layout four{1, 4, 256, format}, two{1, 2, 256, format};
      const auto required = splash::ops::PagedAttention::verifyPlan(1, 16, four, histories);
      require(required.queryHeads == 16, "4B GQA4 verifier was incorrectly excluded");
      for (const auto shape : {std::pair{24u, four}, std::pair{16u, two}}) {
        const bool expected = shape.first == 24 ? block != 2 && block != 16 : block != 16;
        bool supported = true;
        try { static_cast<void>(splash::ops::PagedAttention::verifyPlan(1, shape.first, shape.second, histories)); }
        catch (const std::invalid_argument &error) {
          supported = false;
          require(std::string_view(error.what()).find("verify does not support GQA") != std::string_view::npos,
                  "unsupported verifier failed for an unrelated reason");
        }
        require(supported == expected, "legacy verifier capability disagrees with compiled shader entries");
        require(splash::ops::PagedAttention::prefillPlan(32, shape.first, shape.second).queryHeads == shape.first,
                "a valid prefill path was excluded with the unsupported verifier");
      }
    }
    std::cout << "draft-block-contract: PASS block=" << block << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "draft-block-contract: FAIL: " << error.what() << '\n';
    return 1;
  }
}
