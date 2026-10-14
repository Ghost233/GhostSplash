#include "model/OptionalFeatures.hpp"
#include "model/CompactDraftHead.hpp"
#include <numeric>
#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include <cstring>

#include <cstdlib>
#include <iostream>

using namespace splash::model;
using splash::test::require;

int main() {
  OptionalFeatures options;
  require(!options.promptLookup && !options.compactDraftHead && !options.fastTopK &&
          options.compactRanking.empty(), "features must default off");
  require(options.lookupMinimumMatch == 2 && options.compactCapacity == 98304, "original tuning defaults");
  auto fileOnly = options;
  fileOnly.compactRanking = "/missing-public-ranking";
  fileOnly.validate();
  require(!fileOnly.compactDraftHead, "ranking path alone must not enable compact");
  auto missingFile = options;
  missingFile.compactDraftHead = true;
  bool missingRejected = false;
  try { missingFile.validate(); } catch (const std::invalid_argument &) { missingRejected = true; }
  require(missingRejected, "enabling compact requires an explicit ranking file");
  options.promptLookup = true;
  require(!options.compactDraftHead && !options.fastTopK, "lookup switch is independent");
  for (uint32_t value : {1U, 5U}) {
    auto invalid = options;
    invalid.lookupMinimumMatch = value;
    bool rejected = false;
    try { invalid.validate(); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "invalid lookup minimum accepted");
  }
  for (uint32_t value : {256U, 33000U, 98305U}) {
    auto invalid = options;
    invalid.compactCapacity = value;
    bool rejected = false;
    try { invalid.validate(); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "invalid compact capacity accepted");
  }
  options.lookupMinimumMatch = 3;
  options.compactCapacity = 65536;
  options.validate();
  const std::vector<uint32_t> two{9,1,2,10,11,12,13,14,15,16,99,1,2};
  const std::vector<uint32_t> three{9,1,2,3,10,11,12,13,14,15,16,99,1,2,3};
  require(lookupProposal(two,7,2) == std::vector<uint32_t>({10,11,12,13,14,15,16}), "two-token suffix lookup");
  require(lookupProposal(two,7,3).empty(), "three-token minimum accepted two-token match");
  require(lookupProposal(three,7,3) == std::vector<uint32_t>({10,11,12,13,14,15,16}), "three-token suffix lookup");
  require(lookupProposal(three,7,4).empty(), "four-token minimum accepted three-token match");

  const std::vector<uint32_t> history{9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4};
  require(lookupProposal(history, 7) == std::vector<uint32_t>({5, 6, 7, 8, 9, 10, 1}),
          "lookup copies an exact earlier continuation");
  require(lookupProposal(std::vector<uint32_t>{1, 2, 3, 1, 2}, 7).empty(),
          "short continuation must fall back");
  require(lookupProposal({}, 7).empty(), "empty lookup");
  require(lookupProposal(history, 0).empty(), "empty proposal");
  require(lookupProposal(std::vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8}, 7).empty(),
          "unmatched suffix must fall back");

  std::vector<uint32_t> ranking(1024);
  std::iota(ranking.begin(), ranking.end(), 0);
  auto rows = compactTokenRows(ranking, std::vector<uint32_t>(200, 7), 768);
  require(rows.reason.empty() && rows.valid == 512 && rows.ids.size() == 768, "static compact rows");
  std::vector<uint32_t> prompt(200, 7);
  prompt.push_back(1000);
  rows = compactTokenRows(ranking, prompt, 768);
  require(rows.reason.empty() && rows.valid == 513 && rows.ids[512] == 1000, "rare prompt supplement");
  require(!compactTokenRows(ranking, std::vector<uint32_t>{1000, 7}, 768).reason.empty(),
          "rare token ratio must fall back");
  prompt.assign(30000, 7);
  for (uint32_t i = 0; i < 257; ++i) prompt.push_back(512 + i);
  require(compactTokenRows(ranking, prompt, 768).reason == "more than 256 distinct rare tokens",
          "dynamic vocabulary budget");
  ranking.resize(248320);
  std::iota(ranking.begin(), ranking.end(), 0);
  for (uint32_t capacity : {98304U,65536U,32768U}) {
    std::vector<uint32_t> input(10000,7); input.push_back(248319);
    rows = compactTokenRows(ranking,input,capacity);
    require(rows.reason.empty() && rows.ids.size()==capacity && rows.valid==capacity-255 &&
            rows.ids[capacity-256]==248319, "requested compact row budget and dynamic supplement");
  }
  const splash::test::TemporaryDirectory temporary("public-compact-ranking");
  const auto file = temporary.path() / "ranking.splash";
  const std::vector<uint32_t> order{2, 0, 3, 1};
  std::vector<uint8_t> bytes(12 + 4 * order.size());
  std::memcpy(bytes.data(), "SPLCHD02", 8);
  const uint32_t vocabulary = 4;
  std::memcpy(bytes.data() + 8, &vocabulary, 4);
  std::memcpy(bytes.data() + 12, order.data(), 4 * order.size());
  splash::test::writeFile(file, bytes);
  require(readCompactRanking(file, vocabulary) == order, "public ranking has no model binding");
  const auto rejected = [&](const std::filesystem::path &path, uint32_t size) {
    try { static_cast<void>(readCompactRanking(path, size)); } catch (const std::invalid_argument &) { return true; }
    return false;
  };
  require(rejected(temporary.path() / "missing", vocabulary), "missing ranking must reject enabling");
  require(rejected(file, 5), "vocabulary mismatch must reject enabling");
  for (uint32_t bad : {4U, 2U}) {
    auto broken = bytes;
    std::memcpy(broken.data() + 16, &bad, 4);
    splash::test::writeFile(file, broken);
    require(rejected(file, vocabulary), "out-of-range or duplicate ranking ID accepted");
  }
  auto truncated = bytes; truncated.pop_back();
  splash::test::writeFile(file, truncated);
  require(rejected(file, vocabulary), "truncated ranking accepted");
  auto unknown = bytes; unknown[7] = '9';
  splash::test::writeFile(file, unknown);
  require(rejected(file, vocabulary), "unknown format version accepted");
  auto trailing = bytes; trailing.push_back(0);
  splash::test::writeFile(file, trailing);
  require(rejected(file, vocabulary), "trailing ranking data accepted");
  bytes[7] = '1';
  splash::test::writeFile(file, bytes);
  require(rejected(file, vocabulary), "legacy ranking must require offline conversion");
  std::cout << "optional features CPU checks passed\n";
}
