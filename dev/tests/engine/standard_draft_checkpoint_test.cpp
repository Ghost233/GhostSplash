// CPU-only plan/header checks for the audited standard 4B DFlash release.
#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/DFlashDraft.hpp"
#include "model/DraftCheckpoint.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <set>

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
    layout.targetLayerCount = 8;
    for (uint32_t index = 0; index < layout.targetLayerCount; ++index) layout.targetLayers[index] = 1 + 4 * index;
    layout.attentionSize = 4096;
    layout.qkvSize = 6144;
    layout.kvHeads = 8;
    layout.attentionHeadDimension = 128;
    layout.intermediateSize = 9216;
    layout.vocabularySize = 248320;
    layout.rotaryTheta = 10'000'000;
    if (tiny) {
      layout.hiddenSize = 256;
      layout.attentionSize = 512;
      layout.qkvSize = 768;
      layout.kvHeads = 2;
      layout.attentionHeadDimension = 64;
      layout.intermediateSize = 512;
      layout.vocabularySize = 256;
    }
    const uint32_t hidden = layout.hiddenSize, attention = layout.attentionSize,
                   kv = layout.kvHeads * layout.attentionHeadDimension, ffn = layout.intermediateSize;
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
    expected["fc.weight"] = {hidden, layout.targetHiddenSize()};
    expected["hidden_norm.weight"] = {hidden};
    expected["norm.weight"] = {hidden};
    // Header-only planning checks the real release geometry without reading
    // or allocating its weight payload. A sparse file supplies valid extents.
    splash::test::TemporaryDirectory temporary("splash-standard-plan");
    const auto root = argc >= 2 ? std::filesystem::path(argv[1]) : temporary.path();
    if (argc == 1) {
      std::ostringstream header;
      header << "{";
      uint64_t offset = 0;
      bool first = true;
      for (const auto &[name, shape] : expected) {
        if (!first) header << ",";
        first = false;
        uint64_t count = 1;
        header << "\"" << name << "\":{\"dtype\":\"BF16\",\"shape\":[";
        for (size_t i = 0; i < shape.size(); ++i) {
          if (i) header << ",";
          header << shape[i];
          count *= shape[i];
        }
        const auto next = offset + count * 2;
        header << "],\"data_offsets\":[" << offset << "," << next << "]}";
        offset = next;
      }
      header << "}";
      auto json = header.str();
      while (json.size() % 8) json += ' ';
      const uint64_t length = json.size();
      std::ofstream file(root / "model.safetensors", std::ios::binary);
      file.write(reinterpret_cast<const char *>(&length), sizeof(length));
      file.write(json.data(), json.size());
      file.close();
      std::filesystem::resize_file(root / "model.safetensors", 8 + length + offset);
    }
    const SafetensorsCheckpoint source(root);
    require(source.tensorCount() == 69, "standard checkpoint has unexpected tensors");
    const auto images = planDraftImages(source, layout);
    require(images.size() == 7, "standard draft does not plan six layers and model.bin");
    std::set<std::string> consumed;
    const auto input = [&](const std::string &name) {
      const auto found = expected.find(name);
      const auto &field = source.require(name);
      require(found != expected.end() && field.shape == found->second && field.dtype == "BF16",
              "standard draft source key, dtype or shape differs: " + name);
      consumed.insert(name);
    };
    uint64_t bytes = 0;
    for (size_t index = 0; index < images.size(); ++index) {
      const auto &image = images[index];
      require(image.magic == kGgufImageMagic && image.layer == index &&
                  image.type == (index == 6 ? 1u : 0u), "standard block image header differs");
      require(image.copies.size() == (index == 6 ? 2u : 4u) &&
                  image.repacks.size() == (index == 6 ? 1u : 5u),
              "standard block image contains fabricated sections");
      for (const auto &copy : image.copies) {
        require(copy.destination % kWeightFileAlignment == 0,
                "standard copied section alignment differs");
        input(copy.source.name);
      }
      for (const auto &step : image.repacks) {
        require(step.format == GGUF_FMT_AF4G64 && step.plane0 % kWeightFileAlignment == 0 &&
                    step.meta % kWeightFileAlignment == 0,
                "standard projection block format or alignment differs");
        for (const auto &part : step.sources) input(part.name + ".weight");
      }
      bytes += image.bytes;
    }
    require(consumed.size() == expected.size() && consumed.size() == 69,
            "standard draft does not consume exactly 69 tensors");
    source.checkUnchanged();
    std::cout << "standard DFlash: exact 69 BF16 tensors, seven AF4G64 block images, no fabricated sections, "
              << bytes << " prepared bytes PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
