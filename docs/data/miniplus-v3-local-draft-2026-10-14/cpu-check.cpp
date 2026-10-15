#include "GgufFormatReference.hpp"
#include "model/ModelFactory.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/GgufTarget.hpp"
#include "model/GgufImage.hpp"
#include <iostream>
#include <numeric>

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--reference") {
    uint32_t kind;
    while (std::cin.read(reinterpret_cast<char *>(&kind), sizeof(kind))) {
      uint32_t bytes;
      std::cin.read(reinterpret_cast<char *>(&bytes), sizeof(bytes));
      std::vector<uint8_t> raw(bytes);
      std::cin.read(reinterpret_cast<char *>(raw.data()), bytes);
      if (kind == 0) {
        std::cout.write(reinterpret_cast<const char *>(raw.data()), bytes);
      } else {
        const uint32_t format = gguf_format_of(kind);
        const uint32_t count = kQuantFormats[format].block_elements;
        for (uint32_t group = 0; group < count / 32; ++group) {
          float values[32]; uint8_t plane0[32], plane1[8];
          gguf_reference::groupPack(gguf_reference::Fmt(format), raw.data(), group, values, plane0, plane1);
          std::cout.write(reinterpret_cast<const char *>(values), sizeof(values));
        }
      }
    }
    return 0;
  }
  if (argc != 2) return 2;
  using namespace splash::model;
  const std::filesystem::path root(argv[1]);
  const auto descriptor = inspectModelRoot(root);
  const SafetensorsCheckpoint checkpoint(root / "draft");
  const auto draft = planDraftImages(checkpoint, descriptor.draft);
  WeightSource source(findTargetGguf(root / "target"));
  const GgufFile file(source);
  std::visit([&](const auto &layout) {
    const auto target = gguf::planImages(file, layout);
    uint64_t targetBytes=0, draftBytes=0, converted=0, widenedBytes=0;
    for (const auto &image : target) {
      targetBytes += image.bytes;
      for (const auto &copy : image.copies) if (copy.conversion == gguf::Conversion::DequantizeToFloat32) {
        ++converted;
        const auto &format = kQuantFormats[gguf_format_of(copy.source.type)];
        const auto bytes = copy.source.rows * copy.source.rowBytes / format.block_bytes * format.block_elements * 4;
        if (gguf::convertedBytes(copy.source, copy.conversion, copy.source.rows * copy.source.rowBytes) != bytes)
          throw std::runtime_error("wrong widened GDN copy size");
        widenedBytes += bytes;
      }
    }
    for (const auto &image : draft) draftBytes += image.bytes;
    std::cout << "{\"target_images\":" << target.size() << ",\"draft_images\":" << draft.size()
      << ",\"draft_tensors\":" << checkpoint.tensorCount() << ",\"target_image_bytes\":" << targetBytes
      << ",\"draft_image_bytes\":" << draftBytes << ",\"widened_gdn_tensors\":" << converted
      << ",\"widened_gdn_bytes\":" << widenedBytes << "}\n";
  }, descriptor.target);
}
