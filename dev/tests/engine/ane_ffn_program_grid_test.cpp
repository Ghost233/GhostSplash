// CPU-only regression for the expanded ANE function grid. No engine is linked:
// the 4096 ABI override exercises the real policy without changing a live model.
#include "metal/abi/ExecutionGeometry.h"
#undef SPLASH_PREFILL_TOKEN_BUDGET
#define SPLASH_PREFILL_TOKEN_BUDGET 4096u
#include "ops/AneFfn.hpp"

#include <iostream>

int main() {
  using splash::ops::AneFfn;
  const uint32_t count = (AneFfn::kMaximumRows - AneFfn::kMinimumRows) / AneFfn::kProgramStep + 1;
  if (AneFfn::kMinimumRows != 512 || AneFfn::kMaximumRows != 4096 || count > 16) {
    std::cerr << "FAIL expanded ANE grid has " << count << " functions\n";
    return 1;
  }
  for (uint32_t rows = 512; rows <= 4096; ++rows) {
    const uint32_t function = 512 + ((rows - 512 + AneFfn::kProgramStep - 1) / AneFfn::kProgramStep) * AneFfn::kProgramStep;
    if (function < rows || function > 4096) {
      std::cerr << "FAIL ANE grid does not cover " << rows << " rows\n";
      return 1;
    }
  }
  std::cout << "ANE4096 function grid: all qualifying chunks covered by " << count << " functions\n";
  return 0;
}
