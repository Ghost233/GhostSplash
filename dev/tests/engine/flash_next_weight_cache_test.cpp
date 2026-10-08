#include "model/WeightPageCache.hpp"
#include "model/GgufShardIndex.hpp"

#include <array>
#include <barrier>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unistd.h>

using namespace splash::model;
namespace {
void require(bool value) { if (!value) throw std::runtime_error("assertion failed"); }
template<class F> void fails(F call, const std::string &message) {
  try { call(); } catch (const std::exception &e) {
    if (std::string(e.what()).find(message) == std::string::npos)
      throw std::runtime_error("expected '" + message + "', got '" + e.what() + "'");
    return;
  }
  throw std::runtime_error("expected failure: " + message);
}
template<class T> void put(std::ofstream &out, T value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof value);
}
void text(std::ofstream &out, const std::string &value) {
  put<uint64_t>(out, value.size()); out.write(value.data(), value.size());
}
void shard(const std::filesystem::path &path, uint16_t number, const std::string &name,
           bool payload = true, uint16_t count = 2, int32_t total = 2, uint64_t offset = 0, uint64_t multiplier = 0, bool scalarMultiplier = false) {
  std::ofstream out(path, std::ios::binary);
  out.write("GGUF", 4); put<uint32_t>(out, 3); put<uint64_t>(out, 1); put<uint64_t>(out, multiplier ? 4 : 3);
  text(out, "split.no"); put<uint32_t>(out, 2); put<uint16_t>(out, number);
  text(out, "split.count"); put<uint32_t>(out, 2); put<uint16_t>(out, count);
  text(out, "split.tensors.count"); put<uint32_t>(out, 5); put<int32_t>(out, total);
  if (multiplier) {
    text(out, "qwen4exp.ple.layer_multipliers");
    if (scalarMultiplier) { put<uint32_t>(out, 10); put<uint64_t>(out, multiplier); }
    else {
      put<uint32_t>(out, 9); put<uint32_t>(out, 10);
      put<uint64_t>(out, 2); put<uint64_t>(out, multiplier); put<uint64_t>(out, 20109073645365);
    }
  }
  text(out, name); put<uint32_t>(out, 3); put<uint64_t>(out, 256); put<uint64_t>(out, 2); put<uint64_t>(out, 2);
  put<uint32_t>(out, 21); put<uint64_t>(out, offset);
  while (uint64_t(out.tellp()) % 32) out.put(0);
  if (payload) {
    out.seekp(offset, std::ios::cur);
    for (int i = 0; i < 440; ++i) out.put(char(i % 251));
  }
}
struct TempDirectory {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("splash-weight-cache-" + std::to_string(getpid()));
  TempDirectory() { std::filesystem::create_directory(path); }
  ~TempDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};
WeightRange raw(const std::filesystem::path &path, uint64_t offset, uint64_t bytes) {
  return {path, offset, bytes, 0, {}};
}
WeightPageKey key(uint64_t unit, WeightUnitKind kind = WeightUnitKind::Expert) {
  return {kind, "gate", unit};
}
void cacheReadsEvictsPinsAndOwnsLifetime(const std::filesystem::path &path) {
  WeightPageCache cache(8);
  {
    auto a = cache.acquire(key(0), raw(path, 0, 4));
    require(a.bytes() == 4 && a.data()[0] == 'a' && a.data()[3] == 'd');
    auto b = cache.acquire(key(1), raw(path, 4, 4));
    auto aCopy = a;
    auto hit = cache.acquire(key(0), raw(path, 0, 4));
    fails([&] { (void)cache.acquire(key(2), raw(path, 0, 4)); }, "pinned");
    auto stats = cache.telemetry();
    require(stats.bytesRead == 8 && stats.residentBytes == 8 && stats.hits == 1 && stats.evictions == 0);
  }
  // The prior hit made expert 0 newest; expert 1 is the LRU victim.
  auto c = cache.acquire(key(2), raw(path, 2, 4));
  require(cache.telemetry().evictions == 1 && cache.telemetry().residentBytes == 8);
  auto a = cache.acquire(key(0), raw(path, 0, 4));
  require(cache.telemetry().hits == 2);
  fails([&] { (void)cache.acquire(key(0), raw(path, 1, 4)); }, "different range");
  fails([&] { (void)cache.acquire(key(9), raw(path, 0, 9)); }, "budget");
  fails([&] { (void)cache.acquire(key(9), raw(path, UINT64_MAX, 1)); }, "offset bounds");
  const uint64_t maximumOffset = uint64_t(std::numeric_limits<off_t>::max());
  fails([&] { (void)cache.acquire(key(9), raw(path, maximumOffset, 1)); }, "offset bounds");
  fails([&] { (void)cache.acquire(key(9), raw(path, maximumOffset - 2, 4)); }, "offset bounds");
  fails([&] { (void)cache.acquire(key(9), raw(path, uint64_t(off_t{-1}), 1)); }, "offset bounds");
  fails([&] { (void)cache.acquire(key(9), raw(path, 7, 2)); }, "truncated");
  // Same tensor/unit in the embedding namespace must not collide with experts.
  WeightPageCache separateKinds(8);
  auto expert = separateKinds.acquire(key(0), raw(path, 0, 4));
  auto row = separateKinds.acquire(key(0, WeightUnitKind::EmbeddingRow), raw(path, 4, 4));
  require(expert.data()[0] == 'a' && row.data()[0] == 'e' && separateKinds.telemetry().misses == 2);
  auto outlivesCache = [&] {
    WeightPageCache temporary(4);
    return temporary.acquire(key(0), raw(path, 0, 4));
  }();
  require(outlivesCache.data()[3] == 'd');
}
void concurrentAcquiresShareOneRead(const std::filesystem::path &path) {
  WeightPageCache cache(4);
  std::barrier ready(8);
  std::array<std::thread, 8> threads;
  std::array<const uint8_t *, 8> addresses{};
  std::array<std::exception_ptr, 8> errors{};
  for (size_t i = 0; i < threads.size(); ++i) threads[i] = std::thread([&, i] {
    try {
      ready.arrive_and_wait();
      auto lease = cache.acquire(key(0), raw(path, 0, 4));
      addresses[i] = lease.data(); require(lease.data()[3] == 'd');
    } catch (...) { errors[i] = std::current_exception(); }
  });
  for (auto &t : threads) t.join();
  for (auto &error : errors) if (error) std::rethrow_exception(error);
  for (auto address : addresses) require(address == addresses[0]);
  auto stats = cache.telemetry();
  require(stats.hits == 7 && stats.misses == 1 && stats.bytesRead == 4 && stats.residentBytes == 4);
}
void multishardRangesAndDownloads(const std::filesystem::path &p1, const std::filesystem::path &p2) {
  shard(p1, 0, "blk.0.ffn_gate_exps.weight"); shard(p2, 1, "blk.1.ffn_gate_exps.weight");
  GgufShardIndex index({p2, p1});
  auto range = index.expert("blk.0.ffn_gate_exps.weight", 1);
  require(range.bytes == 220 && range.offset == index.require("blk.0.ffn_gate_exps.weight").offset + 220);
  require(index.tensors().size() == 2 && range.tensorBytes == 440);
  require(range.source->headerSha256 == "e5bea5de5dcd766a782cfa91a8c19ac2c86daad2cf1b549c59859ea90d72c9af");
  require(range.source->dataOffset == 192 && range.source->requiredFileBytes == range.source->fileBytes);
  WeightPageCache cache(220);
  auto invalid = range; invalid.offset = 0;
  fails([&] { (void)cache.acquire(key(20), invalid); }, "outside the indexed");
  invalid = range; invalid.tensorBytes = 1;
  fails([&] { (void)cache.acquire(key(20), invalid); }, "outside the indexed");
  auto expert = cache.acquire({WeightUnitKind::Expert, "blk.0.ffn_gate_exps.weight", 1}, range);
  require(expert.bytes() == 220 && expert.data()[0] == 220 && expert.data()[219] == 188);
  auto row = index.rows("blk.0.ffn_gate_exps.weight", 3, 1);
  require(row.bytes == 110 && row.offset == range.offset + 110);
  fails([&] { (void)index.expert("blk.0.ffn_gate_exps.weight", 2); }, "bounds");
  fails([&] { (void)index.rows("blk.0.ffn_gate_exps.weight", 4, 1); }, "bounds");
  fails([&] { (void)index.require("missing"); }, "missing");
  shard(p2, 1, "blk.1.ffn_gate_exps.weight", false);
  fails([&] { GgufShardIndex incomplete({p1, p2}); }, "truncated");
  GgufShardIndex headerOnly({p1, p2}, GgufShardIndex::Mode::HeaderOnly);
  auto partial = headerOnly.expert("blk.1.ffn_gate_exps.weight", 0);
  fails([&] { (void)cache.acquire(key(3), partial); }, "incomplete");
  // Altering a shard, including a cached page's shard, must fail before serving a hit.
  std::filesystem::resize_file(p1, range.source->fileBytes - 1);
  fails([&] { (void)cache.acquire({WeightUnitKind::Expert, "blk.0.ffn_gate_exps.weight", 1}, range); }, "truncated");
  shard(p1, 0, "blk.0.ffn_gate_exps.weight");
  GgufShardIndex beforeReplace({p1, p2}, GgufShardIndex::Mode::HeaderOnly);
  auto stable = beforeReplace.rows("blk.0.ffn_gate_exps.weight", 0, 1);
  auto replacement = p1.string() + "-new";
  shard(replacement, 0, "blk.0.ffn_gate_exps.weight"); std::filesystem::rename(replacement, p1);
  fails([&] { (void)cache.acquire(key(4), stable); }, "changed");
}
void splitValidationAndLargeOffsets(const std::filesystem::path &p1, const std::filesystem::path &p2) {
  shard(p1, 0, "a"); shard(p2, 1, "b");
  fails([&] { GgufShardIndex missing({p1}); }, "split metadata");
  fails([&] { GgufShardIndex duplicate({p1, p1}); }, "duplicate GGUF shard file");
  shard(p2, 0, "b");
  fails([&] { GgufShardIndex duplicate({p1, p2}); }, "duplicate GGUF split.no");
  shard(p2, 1, "a");
  fails([&] { GgufShardIndex duplicate({p1, p2}); }, "duplicate tensor");
  shard(p2, 1, "b", true, 2, 3);
  fails([&] { GgufShardIndex wrongCount({p1, p2}); }, "global tensor count");
  shard(p2, 1, "b", true, 3);
  fails([&] { GgufShardIndex wrongSplit({p1, p2}); }, "split metadata");
  std::filesystem::remove(p2);
  fails([&] { GgufShardIndex absent({p1, p2}); }, "cannot open");
  shard(p1, 0, "a", true, 2, 2, 0, 23703573157769);
  shard(p2, 1, "b", true, 2, 2, 0, 23703573157769);
  GgufShardIndex arrays({p2, p1});
  auto multipliers = arrays.unsignedArray("qwen4exp.ple.layer_multipliers");
  require(multipliers && multipliers->size() == 2 && (*multipliers)[0] == 23703573157769 && (*multipliers)[1] == 20109073645365);
  shard(p2, 1, "b", true, 2, 2, 0, 23703573157770);
  fails([&] { GgufShardIndex different({p1, p2}); }, "inconsistent GGUF numeric array");
  shard(p2, 1, "b", true, 2, 2, 0, 23703573157769, true);
  fails([&] { GgufShardIndex mixedKinds({p1, p2}); }, "metadata type");
  // A sparse 4GiB+ fixture validates offset width with just440 physical data bytes.
  shard(p1, 0, "a", true, 1, 1, uint64_t(1) << 32);
  GgufShardIndex large({p1});
  auto bytes = large.rows("a", 0, 1);
  require(bytes.offset > (uint64_t(1) << 32) && bytes.bytes == 110);
  WeightPageCache cache(110);
  auto lease = cache.acquire(key(0), bytes); require(lease.data()[109] == 109);
}
}
int main() {
  try {
    TempDirectory temp;
    auto bytes = temp.path / "bytes";
    { std::ofstream out(bytes); out << "abcdefgh"; }
    cacheReadsEvictsPinsAndOwnsLifetime(bytes);
    concurrentAcquiresShareOneRead(bytes);
    auto p1 = temp.path / "shard1", p2 = temp.path / "shard2";
    multishardRangesAndDownloads(p1, p2);
    splitValidationAndLargeOffsets(p1, p2);
    std::cout << "flash-next weight cache: PASS (LRU, pinning, lifetime, concurrency, exact ranges, split validation, incomplete shards, source identity, >4GiB offsets)\n";
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
