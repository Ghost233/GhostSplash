#pragma once

#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace splash::model {

// GGUF packs Kev's two affine pointer projections into [query | key].
// The final row supplies query, option rows supply key; bias stays FP32.
inline std::vector<float> kevPointerScores(std::span<const float> projected,
                                          std::span<const float> bias) {
  if (bias.size() != 512 || projected.size() % bias.size() || projected.size() < 3 * bias.size())
    throw std::invalid_argument("invalid Kev pointer head shape");
  const size_t options = projected.size() / bias.size() - 1;
  const auto query = projected.subspan(options * bias.size(), 256);
  std::vector<float> scores;
  scores.reserve(options);
  for (size_t option = 0; option < options; ++option) {
    const auto key = projected.subspan(option * bias.size() + 256, 256);
    float dot = 0;
    for (size_t channel = 0; channel < 256; ++channel)
      dot += (query[channel] + bias[channel]) * (key[channel] + bias[256 + channel]);
    const float score = dot / 16.0F;
    if (!std::isfinite(score)) throw std::runtime_error("Kev pointer score is not finite");
    scores.push_back(score);
  }
  return scores;
}

} // namespace splash::model
