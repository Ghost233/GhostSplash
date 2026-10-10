// CPU-only plan/header checks for the audited standard 4B DFlash release.
#include "TestChecks.hpp"
#include "model/AffinePlan.hpp"
#include "model/DFlashDraft.hpp"
#include "model/DraftCheckpoint.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace splash::model;
using splash::test::require;

int main(int argc, char **argv) {
  try {
    require(argc >= 1 && argc <= 3, "usage: standard-draft-checkpoint [SOURCE_DIRECTORY [--tiny]]");
    const bool tiny = argc == 3 && std::string(argv[2]) == "--tiny";
    require(argc != 3 || tiny, "unknown standard draft test mode");
    DFlashDraftLayout layout;
    layout.kind = DraftKind::Standard;
    layout.layers = 6;
    layout.hiddenSize = 2560;
    layout.targetHiddenSize = 20480;
    layout.attentionSize = 4096;
    layout.qkvSize = 6144;
    layout.kvHeads = 8;
    layout.attentionHeadDimension = 128;
    layout.intermediateSize = 9216;
    layout.vocabularySize = 248320;
    layout.rotaryTheta = 10'000'000;
    if (tiny) {
      layout.hiddenSize = 256;
      layout.targetHiddenSize = 2048;
      layout.attentionSize = 512;
      layout.qkvSize = 768;
      layout.kvHeads = 2;
      layout.attentionHeadDimension = 64;
      layout.intermediateSize = 512;
      layout.vocabularySize = 256;
    }
    const uint32_t hidden = layout.hiddenSize, attention = layout.attentionSize,
                   kv = layout.kvHeads * layout.attentionHeadDimension, ffn = layout.intermediateSize;
    auto images = draftCheckpointImages(layout);
    require(images.size() == 7, "standard draft does not plan six layers and model.bin");
    std::map<std::string, std::vector<uint64_t>> expected;
    for (uint32_t layer = 0; layer < 6; ++layer) {
      const std::string p = "layers." + std::to_string(layer) + ".";
      expected[p + "input_layernorm.weight"] = {hidden};
      expected[p + "post_attention_layernorm.weight"] = {hidden};
      for (const auto &name : {"q_norm", "k_norm"}) expected[p + "self_attn." + name + ".weight"] = {layout.attentionHeadDimension};
      expected[p + "self_attn.q_proj.weight"] = {attention, hidden};
      expected[p + "self_attn.k_proj.weight"] = {kv, hidden};
      expected[p + "self_attn.v_proj.weight"] = {kv, hidden};
      expected[p + "self_attn.o_proj.weight"] = {hidden, attention};
      expected[p + "mlp.gate_proj.weight"] = {ffn, hidden};
      expected[p + "mlp.up_proj.weight"] = {ffn, hidden};
      expected[p + "mlp.down_proj.weight"] = {hidden, ffn};
    }
    expected["fc.weight"] = {hidden, layout.targetHiddenSize};
    expected["hidden_norm.weight"] = {hidden};
    expected["norm.weight"] = {hidden};
    uint64_t bytes = 0;
    size_t consumed = 0;
    const auto input = [&](const affine::Input &field) {
      const auto found = expected.find(field.name);
      require(found != expected.end() && field.shape == found->second &&
                  field.dtypes == std::vector<std::string>{"BF16"},
              "standard draft source key, dtype or shape differs: " + field.name);
      expected.erase(found);
      ++consumed;
    };
    for (size_t index = 0; index < images.size(); ++index) {
      const auto &image = images[index];
      require(image.magic == kDFlashStandardLayerMagic && image.layer == index &&
                  image.type == (index == 6 ? 1u : 0u), "standard image header differs");
      require(image.sections.size() == (index == 6 ? 3u : 9u), "standard image contains fabricated sections");
      for (const auto &section : image.sections) {
        require(section.offset % kWeightFileAlignment == 0, "standard section alignment differs");
        if (section.parts.empty()) input(section.input);
        for (const auto &part : section.parts)
          for (const auto &field : part.fields) input(field);
      }
      bytes += image.bytes;
    }
    require(consumed == 69 && expected.empty(), "standard draft does not consume exactly 69 tensors");
    require(tiny || bytes == 357384192, "standard draft prepared image byte count differs");
    if (argc >= 2) {
      SafetensorsCheckpoint source(argv[1]);
      require(source.tensorCount() == 69, "standard checkpoint has unexpected tensors");
      for (auto &image : images) {
        affine::bind(image, source);
        if (tiny) {
          std::vector<uint8_t> prepared(image.bytes, 0xff);
          affine::writeAffineImage(prepared, image);
          std::ifstream file(std::filesystem::path(argv[1]) / "expected" / image.name, std::ios::binary);
          const std::vector<uint8_t> expectedBytes{std::istreambuf_iterator<char>(file), {}};
          require(prepared == expectedBytes, "standard affine quantization differs from independent oracle: " + image.name);
        }
      }
      source.checkUnchanged();
    }
    std::cout << "standard DFlash: exact 69 BF16 tensors, seven images, no fabricated sections, "
              << bytes << " prepared bytes PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
