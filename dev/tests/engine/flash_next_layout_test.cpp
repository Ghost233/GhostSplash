#include "model/FlashNextTarget.hpp"
#include "TestChecks.hpp"
#include <algorithm>
#include <iostream>
using namespace splash::model;
using splash::test::require;
int main() {
  try {
    // Independent geometry from the published GGUF header, revision
    // 99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d. No payload is generated here.
    const FlashNextLayout layout;
    require(layout.valid(), "published Flash Next layout must be supported");
    require(layout.expertCacheBytes == 20ull * 1024 * 1024 * 1024 &&
            layout.hostCacheBytes == 64ull * 1024 * 1024, "SSD weights must use separately bounded caches");
    require(layout.stateBytes() == 1238028288, "Flash Next must count GDN parity, PLE convolution and QSA raw/pooled keys");
    auto wrong = layout; wrong.experts = 256;
    require(!wrong.valid(), "35B expert geometry must not be accepted as Flash Next");
    wrong = layout; wrong.indexRatio = 1;
    require(!wrong.valid(), "dropping QSA compression is unsupported");
    wrong = layout; wrong.pleLayer = 0;
    require(!wrong.valid(), "GGUF PLE layer is zero-based index 1");
    const auto specs = flashNextWeightSpecs(layout);
    require(specs.size() == 1224, "every published Flash Next tensor must be planned");
    const auto named = [&](const char *name) -> const FlashNextWeightSpec & {
      auto found = std::find_if(specs.begin(), specs.end(), [&](const auto &s) { return s.name == name; });
      require(found != specs.end(), std::string("missing Flash Next weight: ") + name); return *found;
    };
    require(named("output_hc_down.weight").dimensions == std::vector<uint64_t>{10240,320}, "head hyper-connection geometry");
    require(named("blk.1.ple_key.weight").dimensions == std::vector<uint64_t>{2560,10240}, "PLE must project keys into all four streams");
    require(named("blk.3.indexer.q_proj.weight").dimensions == std::vector<uint64_t>{2560,512}, "QSA must use all four query heads");
    require(named("blk.0.ssm_alpha.weight").dimensions == std::vector<uint64_t>{2560,48}, "GDN must use all 48 value heads");
    require(named("blk.47.ffn_down_exps.weight").dimensions == std::vector<uint64_t>{640,2560,512}, "last layer expert geometry");
    require(named("token_embd.weight").diskOnly && named("per_layer_token_embd.weight").diskOnly &&
            named("blk.0.ffn_gate_exps.weight").diskOnly, "large tables and experts must never be loaded eagerly");
    require(std::count_if(specs.begin(),specs.end(),[](const auto &s){return s.diskOnly;}) == 146,
            "three expert tables per layer and both embedding tables must be paged");
    std::cout << "Flash Next layout, complete tensor plan and memory bounds passed\n";
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
