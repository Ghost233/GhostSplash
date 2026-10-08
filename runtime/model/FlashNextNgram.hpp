#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace splash::model {

// The GGUF's PLE hash contract. These arrays are exported by the model;
// deriving new multipliers from an assumed seed changes its embeddings.
struct FlashNextNgramConfig final {
  std::array<uint64_t, 3> multipliers;
  std::array<uint64_t, 16> offsets;
  std::array<uint64_t, 16> vocabularies;
  uint32_t vocabularySize = 248320;
  uint32_t eosToken = 248044;
  uint64_t embeddingRows = 320001536;
};

class FlashNextNgram final {
public:
  using Keys = std::array<uint64_t, 16>;
  using State = std::array<uint32_t, 2>; // newest, then older token

  explicit FlashNextNgram(FlashNextNgramConfig config);
  [[nodiscard]] Keys push(uint32_t token);
  [[nodiscard]] std::vector<Keys> push(std::span<const uint32_t> tokens);
  [[nodiscard]] State state() const noexcept { return previous_; }
  void restore(State state);
  void reset() noexcept;

private:
  FlashNextNgramConfig config_;
  State previous_;
};

} // namespace splash::model
