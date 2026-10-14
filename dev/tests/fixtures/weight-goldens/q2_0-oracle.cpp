// Independent Q2_0 golden generator: dequantize_row_q2_0 verbatim from
// ggml-org/llama.cpp 9c2e0e491a822adae1f0b1c831adb4160057d24f,
// ggml/src/ggml-quants.c (MIT; THIRD_PARTY_NOTICES).
// Only this standalone reference function is built; no llama.cpp engine runs.
// Fixture dimensions and RNG contract are pinned independently of Splash.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <random>
#include <vector>
#define GGML_RESTRICT __restrict
#define GGML_FP16_TO_FP32(x) halfToFloat(x)
constexpr int QK2_0=64;
struct block_q2_0 { uint16_t d; uint8_t qs[16]; };
float halfToFloat(uint16_t bits) { __fp16 h; memcpy(&h,&bits,2); return float(h); }
void dequantize_row_q2_0(const block_q2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK2_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            const int byte_index = j / 4;
            const int bit_offset = (j % 4) * 2;
            const uint8_t q = (x[i].qs[byte_index] >> bit_offset) & 0x03;
            // 00=-1, 01=0, 10=+1, 11=+2
            y[i*qk + j] = ((int)q - 1) * d;
        }
    }
}

int main() {
  // Published fixture contract: 256 x1024, MT19937 seed7+formatid19,
  // all bytes drawn first; each block's scale replaced by finite half.
  std::mt19937 rng(26);
  std::vector<uint8_t> bytes(256*1024/64*18);
  for (auto &b:bytes) b=uint8_t(rng());
  for(size_t b=0;b<bytes.size();b+=18) {
    uint16_t h; do h=uint16_t(rng()); while((h&0x7c00)==0x7c00);
    memcpy(bytes.data()+b,&h,2);
  }
  std::vector<float> output(256*1024);
  dequantize_row_q2_0(reinterpret_cast<block_q2_0*>(bytes.data()), output.data(),output.size());
  fwrite(output.data(),sizeof(float),output.size(),stdout);
}
