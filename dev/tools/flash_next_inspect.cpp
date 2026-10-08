// Read-only GGUF shard metadata inspection. This never reads tensor payloads.
// flash-next-inspect [--header-only] SHARD1 [SHARD2 ...]
// Header-only mode permits an incomplete download but reports loading_allowed=false.
#include "model/GgufShardIndex.hpp"
#include "Checked.hpp"

#include <iostream>
#include <set>
#include <string_view>

namespace {
void jsonString(std::string_view text) {
  constexpr char hex[] = "0123456789abcdef";
  std::cout << '"';
  for (unsigned char c : text) {
    if (c == '"' || c == '\\') std::cout << '\\' << char(c);
    else if (c < 32) std::cout << "\\u00" << hex[c >> 4] << hex[c & 15];
    else std::cout << char(c);
  }
  std::cout << '"';
}
}
int main(int argc, char **argv) {
  using namespace splash::model;
  try {
    auto mode = GgufShardIndex::Mode::Complete;
    int first = 1;
    if (argc > 1 && std::string_view(argv[1]) == "--header-only") {
      mode = GgufShardIndex::Mode::HeaderOnly; first = 2;
    }
    if (first == argc) {
      std::cerr << "usage: flash-next-inspect [--header-only] SHARD1 [SHARD2 ...]\n";
      return 2;
    }
    std::vector<std::filesystem::path> paths;
    for (int i = first; i < argc; ++i) paths.emplace_back(argv[i]);
    GgufShardIndex index(paths, mode);
    bool complete = true;
    uint64_t tensorBytes = 0, expertBytes = 0, pleBytes = 0;
    std::set<const WeightShardIdentity *> shards;
    for (const auto &entry : index.tensors()) {
      complete &= entry.source->complete;
      tensorBytes = splash::checkedAdd(tensorBytes, entry.tensor.bytes, "GGUF inspection total");
      if (entry.tensor.dims.size() == 3 && entry.tensor.name.ends_with("_exps.weight"))
        expertBytes = splash::checkedAdd(expertBytes, entry.tensor.bytes, "GGUF expert total");
      if (entry.tensor.name == "per_layer_token_embd.weight") pleBytes = entry.tensor.bytes;
    }
    std::cout << "{\"mode\":";
    jsonString(mode == GgufShardIndex::Mode::Complete ? "complete" : "header-only");
    std::cout << ",\"loading_allowed\":" << (complete ? "true" : "false")
              << ",\"architecture\":";
    jsonString(index.metadata().architecture());
    std::cout << ",\"tensor_count\":" << index.tensors().size()
              << ",\"tensor_bytes\":" << tensorBytes
              << ",\"expert_tensor_bytes\":" << expertBytes
              << ",\"ple_embedding_bytes\":" << pleBytes
              << ",\"other_tensor_bytes\":" << tensorBytes - expertBytes - pleBytes << ",\"shards\":[";
    bool comma = false;
    for (const auto &entry : index.tensors()) {
      if (!shards.insert(entry.source.get()).second) continue;
      if (comma) std::cout << ',';
      comma = true;
      const auto &source = *entry.source;
      std::cout << "{\"path\":"; jsonString(entry.path.string());
      std::cout << ",\"complete\":" << (source.complete ? "true" : "false")
                << ",\"file_bytes_at_inspection\":" << source.fileBytes
                << ",\"required_file_bytes\":" << source.requiredFileBytes
                << ",\"header_bytes\":" << source.dataOffset << ",\"header_sha256\":";
      jsonString(source.headerSha256); std::cout << '}';
    }
    std::cout << "],\"arrays\":{";
    comma = false;
    for (std::string_view key : {"qwen4exp.ple.layers", "qwen4exp.ple.layer_multipliers",
         "qwen4exp.ple.head_offsets", "qwen4exp.ple.head_vocab_sizes",
         "qwen4exp.attention.compress_ratios", "qwen4exp.recurrent_layers", "qwen4exp.attention.recurrent_layers"}) {
      auto array = index.unsignedArray(key);
      if (!array) continue;
      if (comma) std::cout << ',';
      comma = true;
      jsonString(key); std::cout << ":[";
      for (size_t i = 0; i < array->size(); ++i) { if (i) std::cout << ','; std::cout << (*array)[i]; }
      std::cout << ']';
    }
    std::cout << "},\"tensors\":[";
    comma = false;
    for (const auto &entry : index.tensors()) {
      const auto &tensor = entry.tensor;
      if (comma) std::cout << ',';
      comma = true;
      const bool experts = tensor.dims.size() == 3 && tensor.name.ends_with("_exps.weight");
      std::cout << "{\"name\":"; jsonString(tensor.name);
      std::cout << ",\"path\":"; jsonString(entry.path.string());
      std::cout << ",\"type\":" << tensor.type << ",\"type_name\":"; jsonString(ggmlTypeName(tensor.type));
      std::cout << ",\"dimensions\":[";
      for (size_t i = 0; i < tensor.dims.size(); ++i) { if (i) std::cout << ','; std::cout << tensor.dims[i]; }
      std::cout << "],\"offset\":" << entry.offset << ",\"bytes\":" << tensor.bytes
                << ",\"row_bytes\":" << tensor.bytes / tensor.rows()
                << ",\"expert_count\":" << (experts ? tensor.dims[2] : 0)
                << ",\"expert_bytes\":" << (experts ? tensor.bytes / tensor.dims[2] : 0)
                << ",\"ple_embedding\":" << (tensor.name == "per_layer_token_embd.weight" ? "true" : "false") << '}';
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
