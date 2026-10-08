#include "model/FlashNextNgram.hpp"

#include <array>
#include <cassert>
#include <iostream>
#include <stdexcept>

using splash::model::FlashNextNgram;
using splash::model::FlashNextNgramConfig;

int main() {
  const FlashNextNgramConfig config{
      .multipliers = {23703573157769, 20109073645365, 8052911324071},
      .offsets = {0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                  160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275},
      .vocabularies = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                      20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171}};
  FlashNextNgram batch(config), stream(config);
  const std::array<uint32_t, 5> tokens{7, 11, 248044, 13, 17};
  const auto result = batch.push(tokens);
  // Golden row IDs from the official shift/XOR/modulo formulation, using
  // the real GGUF's arrays. In particular EOS resets AFTER the current row.
  assert(result[2][0] == 4679147 && result[2][7] == 141408284);
  assert(result[2][8] == 161918623 && result[2][15] == 307840018);
  assert(result[3][0] == 5743412 && result[3][7] == 149916042);
  assert(result[3][8] == 160372577 && result[3][15] == 304892711);
  assert(result[4][0] == 8107993 && result[4][7] == 150680275);
  assert(result[4][8] == 164762532 && result[4][15] == 317249635);
  for (size_t row = 0; row < tokens.size(); ++row) assert(stream.push(tokens[row]) == result[row]);
  assert(stream.state() == batch.state());
  FlashNextNgram restored(config);
  restored.restore(batch.state());
  assert(restored.push(123) == batch.push(123));
  bool refused = false;
  try { (void)batch.push(248320); } catch (const std::invalid_argument &) { refused = true; }
  assert(refused);
  auto invalid = config;
  invalid.offsets[1] += 1;
  refused = false;
  try { FlashNextNgram bad(invalid); } catch (const std::invalid_argument &) { refused = true; }
  assert(refused);
  std::cout << "Flash Next PLE hash: PASS (real row IDs, EOS, streaming, restore, bounds)\n";
}
