#pragma once
#include "model/FlashNextTarget.hpp"
namespace splash::model {
// Load the released single-layer BF16 head. Norm weights are folded to the
// GGUF multiplier convention; expert tensors stay separate from target SSD pages.
void loadFlashNextMtpWeights(metal::MetalBackend &, FlashNextWeights &,
                            const std::filesystem::path &directory);
}
