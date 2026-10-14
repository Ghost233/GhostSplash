#import <Foundation/Foundation.h>
#include "model/ModelDescriptor.hpp"
#include "model/GgufTarget.hpp"
#include "model/CompactDraftHead.hpp"
#include "model/WeightStore.hpp"
#include "model/WeightImages.hpp"
#include "TestChecks.hpp"
#include <iostream>

using namespace splash;
int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 4) throw std::invalid_argument("usage: real-head METALLIB MODEL_ROOT PUBLIC_RANKING");
      const std::filesystem::path root(argv[2]);
      const auto descriptor = model::inspectModelRoot(root);
      metal::MetalBackend backend(argv[1]);
      model::WeightImages images(backend, descriptor.sourceIdentity);
      std::visit([&](const auto &layout) {
        model::GgufTargetLoader loader(backend, images, model::findTargetGguf(root / "target"), layout);
        auto file = loader.head();
        auto norm = file.section(uint64_t{layout.hiddenSize} * 4, "head-norm");
        auto source = model::readBlockProjection(file, layout.vocabularySize, layout.hiddenSize, "head");
        source.destination = ops::FloatOutput::Float32;
        const auto before = backend.memoryStats().allocatedBytes;
        auto disabled = model::CompactDraftHead::createIfEnabled(false, backend, source, "/missing-unused-ranking");
        test::require(!disabled && backend.memoryStats().allocatedBytes == before, "disabled compact allocated resources");
        auto compact = model::CompactDraftHead::createIfEnabled(true, backend, source, argv[3]);
        const auto checks = compact->validation().comparisons;
        const auto ranking = model::readCompactRanking(argv[3], layout.vocabularySize);
        const auto fixed = compact->capacity() - 256;
        std::vector<uint32_t> history(10000, ranking.front());
        history.push_back(ranking.back());
        test::require(compact->prepare(history) && compact->validRows() == fixed + 1 &&
                      static_cast<uint32_t *>(compact->mapping().contents())[fixed] == ranking.back(), "real dynamic mapping");
        test::require(!compact->prepare(std::vector<uint32_t>{ranking.back(), ranking.front()}), "real coverage ratio fallback");
        history.assign(30000, ranking.front());
        for (uint32_t i = 0; i < 257; ++i) history.push_back(ranking[fixed + i]);
        test::require(!compact->prepare(history) && compact->reason() == "more than 256 distinct rare tokens", "real coverage ID fallback");
        test::require(compact->validation().comparisons == checks, "repeated full-head validation");
        std::cout << "{\"model\":\"" << descriptor.name << "\",\"ranking\":\"" << argv[3]
                  << "\",\"configured\":true,\"validated\":true,\"vocabulary\":" << layout.vocabularySize
                  << ",\"capacity\":" << compact->capacity() << ",\"comparisons\":" << checks
                  << ",\"maximum_absolute_error\":" << compact->validation().maximumAbsoluteError
                  << ",\"disabled_extra_metal_bytes\":0,\"compact_metal_bytes\":" << compact->bytes()
                  << ",\"coverage_ratio_fallback\":true,\"coverage_id_fallback\":true,\"dynamic_mapping_passed\":true}\n";
      }, descriptor.target);
      return 0;
    } catch (const std::exception &error) { std::cerr << "real compact head check FAILED: " << error.what() << '\n'; return 1; }
  }
}
