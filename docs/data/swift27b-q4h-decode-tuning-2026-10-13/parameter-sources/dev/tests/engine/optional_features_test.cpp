#include "model/OptionalFeatures.hpp"
#include "model/CompactDraftHead.hpp"
#include <numeric>
#include "TestChecks.hpp"

#include <cstdlib>
#include <iostream>

using namespace splash::model;
using splash::test::require;

int main() {
  for (const char *name : {"SPLASH_PROMPT_LOOKUP", "SPLASH_COMPACT_DRAFT_HEAD",
                           "SPLASH_FAST_TOPK", "SPLASH_COMPACT_DRAFT_RANKING",
                           "SPLASH_PROMPT_LOOKUP_MIN_MATCH", "SPLASH_COMPACT_DRAFT_CAPACITY"})
    unsetenv(name);
  const auto off = OptionalFeatures::fromEnvironment();
  require(!off.promptLookup && !off.compactDraftHead && !off.fastTopK &&
          off.compactRanking.empty(), "features must default off");
  setenv("SPLASH_PROMPT_LOOKUP", "1", 1);
  auto options = OptionalFeatures::fromEnvironment();
  require(options.promptLookup && !options.compactDraftHead && !options.fastTopK,
          "lookup switch is independent");
  setenv("SPLASH_COMPACT_DRAFT_HEAD", "1", 1);
  setenv("SPLASH_COMPACT_DRAFT_RANKING", "/tmp/ranking", 1);
  setenv("SPLASH_FAST_TOPK", "1", 1);
  options = OptionalFeatures::fromEnvironment();
  require(options.compactDraftHead && options.fastTopK &&
          options.compactRanking == "/tmp/ranking", "startup switches");
  setenv("SPLASH_FAST_TOPK", "yes", 1);
  bool rejected = false;
  try { static_cast<void>(OptionalFeatures::fromEnvironment()); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "malformed switches must fail configuration");

  unsetenv("SPLASH_FAST_TOPK");
  options = OptionalFeatures::fromEnvironment();
  require(options.lookupMinimumMatch == 2 && options.compactCapacity == 98304, "original tuning defaults");
  setenv("SPLASH_PROMPT_LOOKUP_MIN_MATCH", "3", 1);
  setenv("SPLASH_COMPACT_DRAFT_CAPACITY", "65536", 1);
  options = OptionalFeatures::fromEnvironment();
  require(options.lookupMinimumMatch == 3 && options.compactCapacity == 65536, "startup tuning values");
  for (const char *value : {"1", "5", "3x", "-1"}) {
    setenv("SPLASH_PROMPT_LOOKUP_MIN_MATCH", value, 1);
    bool bad = false;
    try { static_cast<void>(OptionalFeatures::fromEnvironment()); }
    catch (const std::invalid_argument &) { bad = true; }
    require(bad, "invalid lookup minimum accepted");
  }
  unsetenv("SPLASH_PROMPT_LOOKUP_MIN_MATCH");
  for (const char *value : {"256", "33000", "98305", "65536x"}) {
    setenv("SPLASH_COMPACT_DRAFT_CAPACITY", value, 1);
    bool bad = false;
    try { static_cast<void>(OptionalFeatures::fromEnvironment()); }
    catch (const std::invalid_argument &) { bad = true; }
    require(bad, "invalid compact capacity accepted");
  }
  unsetenv("SPLASH_COMPACT_DRAFT_CAPACITY");
  const std::vector<uint32_t> two{9,1,2,10,11,12,13,14,15,16,99,1,2};
  const std::vector<uint32_t> three{9,1,2,3,10,11,12,13,14,15,16,99,1,2,3};
  require(lookupProposal(two,7,2) == std::vector<uint32_t>({10,11,12,13,14,15,16}), "two-token suffix lookup");
  require(lookupProposal(two,7,3).empty(), "three-token minimum accepted two-token match");
  require(lookupProposal(three,7,3) == std::vector<uint32_t>({10,11,12,13,14,15,16}), "three-token suffix lookup");
  require(lookupProposal(three,7,4).empty(), "four-token minimum accepted three-token match");

  setenv("SPLASH_PROMPT_LOOKUP", "0", 1);
  setenv("SPLASH_COMPACT_DRAFT_HEAD", "0", 1);
  setenv("SPLASH_PROMPT_LOOKUP_MIN_MATCH", "invalid", 1);
  setenv("SPLASH_COMPACT_DRAFT_CAPACITY", "invalid", 1);
  options = OptionalFeatures::fromEnvironment();
  require(options.lookupMinimumMatch == 2 && options.compactCapacity == 98304,
          "disabled features must not read tuning overrides");
  unsetenv("SPLASH_PROMPT_LOOKUP_MIN_MATCH");
  unsetenv("SPLASH_COMPACT_DRAFT_CAPACITY");

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
  std::cout << "optional features CPU checks passed\n";
}
