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
                           "SPLASH_FAST_TOPK", "SPLASH_COMPACT_DRAFT_RANKING"})
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
  std::cout << "optional features CPU checks passed\n";
}
