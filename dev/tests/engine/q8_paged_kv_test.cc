#include "Q8PageFormatReference.hpp"
#include "TestChecks.hpp"
#include "Q4KvReference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace splash::kv;
using splash::test::rejects;
using splash::test::require;

namespace {

void testByteAccounting() {
  constexpr Layout q4{8, 4, 256, Format::Q4_0Hadamard};
  static_assert(sizeof(SplashQ4Block) == 18 && alignof(SplashQ4Block) == 2);
  static_assert(q4.valid() && q4.elementsPerScale() == 32);
  static_assert(q4.dataBytesPerLayerPage() == 18'432 && q4.scaleBytesPerLayerPage() == 0);
  static_assert(q4.bytesPerModelPage() == 294'912 && q4.extentAlignmentPages() == 32);
  static_assert(1024 * q4.bytesPerModelPage() == 288ULL * 1024 * 1024);
  constexpr Layout plain{8, 4, 256, Format::Q4_0Unrotated};
  static_assert(plain.bytesPerModelPage() == q4.bytesPerModelPage());
  static_assert(usesHadamard(q4.format) && !usesHadamard(plain.format));
  static_assert(storageFormatName(q4.format) != storageFormatName(plain.format));
  constexpr Layout q4h{8, 4, 256, Format::Int4Hadamard};
  static_assert(q4h.valid() && q4h.elementsPerScale() == 256);
  static_assert(q4h.dataBytesPerLayerPage() == 16'384);
  static_assert(q4h.scaleBytesPerLayerPage() == 512);
  static_assert(q4h.bytesPerModelPage() == 270'336);
  static_assert(q4h.extentAlignmentPages() == 128);
  static_assert(1024 * q4h.bytesPerModelPage() == 264ULL * 1024 * 1024);
  static_assert(!Layout{8, 4, 128, Format::Int4Hadamard}.valid());
  static_assert(formatName(Format::Int4Hadamard) == "q4h");
  constexpr Layout bf16{16, 4, 256, Format::BFloat16};
  static_assert(bf16.valid());
  static_assert(bf16.bytesPerModelPage() == 2'097'152);
  static_assert(bf16.extentAlignmentPages() == 1);
  static_assert(bf16.minimumExtentPages() == 32 && bf16.maximumExtentPages() == 96);
  static_assert(4096 * bf16.bytesPerModelPage() == 8ULL * 1024 * 1024 * 1024);
  static_assert(bf16.scaleBytesPerLayerPage() == 0);
  constexpr Layout compact{10, 2, 256, Format::BFloat16};
  static_assert(compact.bytesPerModelPage() == 655'360);
  static_assert(compact.extentAlignmentPages() == 2);
  static_assert(compact.minimumExtentPages() == 104 && compact.maximumExtentPages() == 306);
  // A pool's extents hold whole alignment units, so that every tensor region
  // of an extent starts 64 KiB-aligned: the 512-byte-per-page scale regions
  // need 128 pages for four KV heads and 256 for two, BF16 one or two. The
  // size is chosen per pool between half and one and a half times the
  // 128 MiB target, leaving the fewest of its pages over, the one nearest the
  // target on a tie.
  static_assert(kExtentRegionAlignmentBytes == 64 * 1024);
  static_assert(kOracleLayout.extentAlignmentPages() == 128);
  static_assert(kOracleLayout.minimumExtentPages() == 128 &&
                kOracleLayout.maximumExtentPages() == 128);
  static_assert(kOracleLayout.extentPagesFor(127) == 0 && kOracleLayout.extentPagesFor(128) == 128 &&
                kOracleLayout.extentPagesFor(1000) == 128);
  constexpr Layout compactInt8{10, 2, 256};
  static_assert(compactInt8.bytesPerModelPage() == 332'800);
  static_assert(compactInt8.extentAlignmentPages() == 256);
  static_assert(compactInt8.minimumExtentPages() == 256 && compactInt8.maximumExtentPages() == 512);
  static_assert(compactInt8.extentPagesFor(255) == 0 && compactInt8.extentPagesFor(511) == 256);
  // Both sizes leave nothing over: 512 pages (162.5 MiB) is nearer the target
  // than 256 (81.25 MiB). With 256 pages over, only 256 leaves nothing.
  static_assert(compactInt8.extentPagesFor(10'240) == 512);
  static_assert(compactInt8.extentPagesFor(10'496) == 256);
  // 448 pages divide by 32, 56 and 64 (128 MiB, the target); 97 leaves one
  // page over 32, 48 and 96 extents, of which 48 (96 MiB) is nearest.
  static_assert(bf16.extentPagesFor(31) == 0 && bf16.extentPagesFor(448) == 64 &&
                bf16.extentPagesFor(97) == 48);
  static_assert(!Layout{16, 4, 256, static_cast<Format>(0)}.valid());
  static_assert(kBytesPerModelPage == 1'064'960);
  // One page's keys (or values) and their scales across the layers.
  static_assert(kOracleLayout.attentionLayers * kKeyDataBytesPerLayerPage == 512 * 1024);
  static_assert(kOracleLayout.attentionLayers * kKeyScaleBytesPerLayerPage == 8 * 1024);
  static_assert(kBytesPerModelPage == 2 * (512 + 8) * 1024);
  static_assert(4608 * kBytesPerModelPage == 4'907'335'680ULL); // 147456 tokens.
}

void testStandardQ4Blocks() {
  std::array<float, 32> values{};
  auto zero = splash::test::q4KvReference(values);
  require(zero.d == 0x8000 && std::all_of(std::begin(zero.qs), std::end(zero.qs),
                                      [](uint8_t b) { return b == 0x88; }),
          "zero Q4_0 block changed its scale or offset-nibble representation");
  values[0] = -8; values[1] = 7; values[2] = 0.5f; values[3] = -0.5f;
  values[16] = 1; values[17] = -1;
  auto block = splash::test::q4KvReference(values);
  require(block.d == 0x3c00 && block.qs[0] == 0x90 && block.qs[1] == 0x7f &&
              (block.qs[2] & 15) == 9 && (block.qs[3] & 15) == 8,
          "Q4_0 signed scale, split-half packing or rounding convention changed");
  values[0] = 8;
  require(splash::test::q4KvReference(values).d == 0xbc00,
          "Q4_0 first absolute maximum did not determine the signed scale");
}

void testLayouts() {
  require(splash_kv_key_element(0, 0, 1) == splash_kv_key_element(0, 0, 0) + 1,
          "a key's next dimension is not its next element");
  require(splash_kv_key_element(0, 1, 0) ==
              splash_kv_key_element(0, 0, 0) + kHeadDimension,
          "a key's next token does not follow its head dimension");
  require(splash_kv_value_element(0, 1, 0) ==
              splash_kv_value_element(0, 0, 0) + 1,
          "a value's next token is not its next element");
  require(splash_kv_value_element(0, 0, 1) ==
              splash_kv_value_element(0, 0, 0) + kPageTokens,
          "a value's next dimension does not follow the page's tokens");
  require(splash_kv_key_element(kKvHeads - 1, kPageTokens - 1,
                                kHeadDimension - 1) == kElementsPerLayerPage - 1,
          "the last key element is not the layer page's last");
  require(splash_kv_value_element(kKvHeads - 1, kPageTokens - 1,
                                  kHeadDimension - 1) == kElementsPerLayerPage - 1,
          "the last value element is not the layer page's last");
}

void testBFloat16() {
  for (float value : {0.0f, -0.0f, 1.0f, -3.5f, 1.0e-20f, 65504.0f}) {
    float roundTrip = bfloat16ToFloat(floatToBFloat16(value));
    if (value == 0.0f) {
      require(roundTrip == value, "a zero did not survive BF16");
    } else {
      require(std::abs(roundTrip - value) <= std::abs(value) / 128.0f,
              "a BF16 round trip moved a value by more than 1/128 of it");
    }
  }
  require(std::isinf(bfloat16ToFloat(
              floatToBFloat16(std::numeric_limits<float>::infinity()))),
          "an infinity did not survive BF16");
  require(std::isnan(bfloat16ToFloat(
              floatToBFloat16(std::numeric_limits<float>::quiet_NaN()))),
          "a NaN did not survive BF16");
}

void testQuantization(uint32_t validTokens) {
  uint64_t inputElements = uint64_t{validTokens} * kKvHeads * kHeadDimension;
  std::vector<float> keys(inputElements);
  std::vector<float> values(inputElements);
  for (uint32_t token = 0; token < validTokens; ++token) {
    for (uint32_t head = 0; head < kKvHeads; ++head) {
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
        uint64_t index = logicalIndex(token, head, dimension);
        keys[index] =
            2.5f * std::sin(float(token * 17 + head * 31 + dimension) * 0.013f);
        values[index] =
            1.75f *
            std::cos(float(token * 29 + head * 11 + dimension) * 0.017f);
      }
    }
  }
  auto page = std::make_unique<Q8LayerPage>();
  quantizeLayerPage(keys, values, validTokens, *page);
  std::vector<float> decodedKeys(kElementsPerLayerPage);
  std::vector<float> decodedValues(kElementsPerLayerPage);
  dequantizeLayerPage(*page, validTokens, decodedKeys, decodedValues);

  for (uint32_t token = 0; token < validTokens; ++token) {
    for (uint32_t head = 0; head < kKvHeads; ++head) {
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
        uint64_t logical = logicalIndex(token, head, dimension);
        float keyScale = page->keyScales[splash_kv_scale_element(head, token)];
        float valueScale = page->valueScales[splash_kv_scale_element(head, token)];
        require(std::abs(decodedKeys[logical] - keys[logical]) <=
                    keyScale * 0.51f + 1.0e-7f,
                "a key is off by more than half its scale");
        require(std::abs(decodedValues[logical] - values[logical]) <=
                    valueScale * 0.51f + 1.0e-7f,
                "a value is off by more than half its scale");
        require(decodedKeys[logical] ==
                    dequantizeKey(*page, head, token, dimension),
                "the page's keys and one key's dequantization differ");
        require(decodedValues[logical] ==
                    dequantizeValue(*page, head, token, dimension),
                "the page's values and one value's dequantization differ");
      }
    }
  }
  for (uint32_t token = validTokens; token < kPageTokens; ++token) {
    uint64_t logical = logicalIndex(token, 0, 0);
    require(decodedKeys[logical] == 0.0f && decodedValues[logical] == 0.0f,
            "a row past the valid tokens is not zero");
  }
}

void testZeroAndInvalidInputs() {
  constexpr uint32_t tokens = 1;
  std::vector<float> values(uint64_t{tokens} * kKvHeads * kHeadDimension);
  auto page = std::make_unique<Q8LayerPage>();
  quantizeLayerPage(values, values, tokens, *page);
  require(std::all_of(page->keys.begin(), page->keys.end(),
                      [](int8_t value) { return value == 0; }) &&
              std::all_of(page->values.begin(), page->values.end(),
                          [](int8_t value) { return value == 0; }),
          "a zero page did not quantize to zeros");
  values[0] = std::numeric_limits<float>::quiet_NaN();
  rejects([&] { quantizeLayerPage(values, values, tokens, *page); },
          "logical keys contains a non-finite value",
          "a page with a non-finite value was quantized");
}

} // namespace

int main() {
  try {
    testByteAccounting();
    testStandardQ4Blocks();
    testLayouts();
    testBFloat16();
    testQuantization(kPageTokens);
    testQuantization(17);
    testZeroAndInvalidInputs();
    std::cout << "q8_paged_kv_test: ok\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "q8_paged_kv_test failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
