#include "model/FlashNextNgram.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace splash::model {

FlashNextNgram::FlashNextNgram(FlashNextNgramConfig config) : config_(std::move(config)) {
  if (config_.vocabularySize < 2 || config_.eosToken >= config_.vocabularySize || !config_.embeddingRows)
    throw std::invalid_argument("invalid Flash Next ngram vocabulary");
  const uint64_t maximum = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  for (uint64_t multiplier : config_.multipliers)
    if (!multiplier || multiplier > maximum / (config_.vocabularySize - 1))
      throw std::invalid_argument("Flash Next ngram multiplication exceeds signed int64");
  uint64_t offset = 0;
  for (size_t head = 0; head < config_.offsets.size(); ++head) {
    const uint64_t count = config_.vocabularies[head];
    if (!count || config_.offsets[head] != offset || count > config_.embeddingRows - offset)
      throw std::invalid_argument("invalid Flash Next ngram head ranges");
    offset += count;
  }
  reset();
}

FlashNextNgram::Keys FlashNextNgram::push(uint32_t token) {
  if (token >= config_.vocabularySize) throw std::invalid_argument("Flash Next ngram token outside vocabulary");
  const uint64_t bigram = uint64_t(token) * config_.multipliers[0] ^
                         uint64_t(previous_[0]) * config_.multipliers[1];
  const uint64_t trigram = bigram ^ uint64_t(previous_[1]) * config_.multipliers[2];
  Keys result;
  for (size_t head = 0; head < result.size(); ++head) {
    const uint64_t mixed = head < 8 ? bigram : trigram;
    result[head] = mixed % config_.vocabularies[head] + config_.offsets[head];
  }
  // EOS itself still sees the preceding segment. Only subsequent tokens
  // are padded with EOS, matching the model's shift-right-ignore-EOS rule.
  if (token == config_.eosToken) reset();
  else previous_ = {token, previous_[0]};
  return result;
}

std::vector<FlashNextNgram::Keys> FlashNextNgram::push(std::span<const uint32_t> tokens) {
  std::vector<Keys> result;
  result.reserve(tokens.size());
  for (uint32_t token : tokens) result.push_back(push(token));
  return result;
}

void FlashNextNgram::restore(State state) {
  for (uint32_t token : state)
    if (token >= config_.vocabularySize) throw std::invalid_argument("invalid Flash Next ngram history");
  previous_ = state;
}

void FlashNextNgram::reset() noexcept { previous_ = {config_.eosToken, config_.eosToken}; }

} // namespace splash::model
