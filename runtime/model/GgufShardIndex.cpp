#include "model/GgufShardIndex.hpp"
#include "Checked.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <set>
#include <sys/stat.h>
#include <unistd.h>

namespace splash::model {
namespace {
// CommonCrypto remains the project's streaming SHA256 implementation. Keep
// SDK deprecation suppression at these calls, leaving parser warnings enabled.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
void beginHeaderDigest(CC_SHA256_CTX &context) { CC_SHA256_Init(&context); }
void updateHeaderDigest(CC_SHA256_CTX &context, const void *bytes, CC_LONG count) {
  CC_SHA256_Update(&context, bytes, count);
}
void endHeaderDigest(CC_SHA256_CTX &context, unsigned char *digest) { CC_SHA256_Final(digest, &context); }
#pragma clang diagnostic pop

class HeaderReader {
public:
  explicit HeaderReader(const std::filesystem::path &path) : path_(path) {
    fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) throw GgufError("cannot open GGUF shard: " + path.string());
    if (fstat(fd_, &state) || !S_ISREG(state.st_mode) || state.st_size < 0) {
      close(fd_); fd_ = -1;
      throw GgufError("cannot stat regular GGUF shard: " + path.string());
    }
    beginHeaderDigest(hash_);
  }
  ~HeaderReader() { if (fd_ >= 0) close(fd_); }
  HeaderReader(const HeaderReader &) = delete;
  HeaderReader &operator=(const HeaderReader &) = delete;
  struct stat state{};
  uint64_t position = 0;
  void bytes(void *destination, uint64_t count) {
    if (position > uint64_t(state.st_size) || count > uint64_t(state.st_size) - position)
      throw GgufError("GGUF header is truncated: " + path_.string());
    auto *to = static_cast<uint8_t *>(destination);
    while (count) {
      const size_t part = std::min<uint64_t>(count, 1u << 20);
      const ssize_t n = pread(fd_, to, part, off_t(position));
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) throw GgufError("GGUF header read failed: " + path_.string());
      updateHeaderDigest(hash_, to, CC_LONG(n));
      position += uint64_t(n); to += n; count -= uint64_t(n);
    }
  }
  void skip(uint64_t count) {
    std::array<uint8_t, 65536> scratch;
    while (count) {
      const uint64_t part = std::min<uint64_t>(count, scratch.size());
      bytes(scratch.data(), part); count -= part;
    }
  }
  template<class T> T scalar() { T value{}; bytes(&value, sizeof value); return value; }
  std::string string() {
    const uint64_t count = scalar<uint64_t>();
    if (count > (8u << 20) || retainedStrings_ > (8u << 20) - count)
      throw GgufError("GGUF header strings exceed bounds");
    retainedStrings_ += count;
    std::string result(count, '\0'); bytes(result.data(), count); return result;
  }
  void skipString() { skip(scalar<uint64_t>()); }
  std::string digest() {
    unsigned char result[CC_SHA256_DIGEST_LENGTH]; endHeaderDigest(hash_, result);
    constexpr char hex[] = "0123456789abcdef";
    std::string out; out.reserve(64);
    for (unsigned char b : result) { out += hex[b >> 4]; out += hex[b & 15]; }
    return out;
  }
private:
  std::filesystem::path path_;
  int fd_ = -1;
  CC_SHA256_CTX hash_{};
  uint64_t retainedStrings_ = 0;
};
uint64_t scalarBytes(uint32_t type) {
  switch (type) {
  case 0: case 1: case 7: return 1;
  case 2: case 3: return 2;
  case 4: case 5: case 6: return 4;
  case 10: case 11: case 12: return 8;
  default: throw GgufError("unknown GGUF metadata type " + std::to_string(type));
  }
}
void skipValue(HeaderReader &r, uint32_t type, unsigned depth = 0) {
  if (depth > 16) throw GgufError("GGUF metadata nesting exceeds bounds");
  if (type == 8) r.skipString();
  else if (type == 9) {
    const uint32_t element = r.scalar<uint32_t>();
    const uint64_t count = r.scalar<uint64_t>();
    if (element == 8 || element == 9) {
      // Every nested value needs at least 8 bytes; reject malicious huge counts
      // before walking them, while keeping tokenizer arrays unrestricted in size.
      if (count > (uint64_t(r.state.st_size) - r.position) / 8) throw GgufError("GGUF metadata array is truncated");
      for (uint64_t i = 0; i < count; ++i) skipValue(r, element, depth + 1);
    } else r.skip(checkedMultiply<GgufError>(count, scalarBytes(element), "GGUF metadata size"));
  } else r.skip(scalarBytes(type));
}
void scalarMetadata(HeaderReader &r, GgufMetadata &m, const std::string &key, uint32_t type) {
  switch (type) {
  case 0: m.unsigneds[key] = r.scalar<uint8_t>(); break;
  case 1: m.unsigneds[key] = uint64_t(r.scalar<int8_t>()); break;
  case 2: m.unsigneds[key] = r.scalar<uint16_t>(); break;
  case 3: m.unsigneds[key] = uint64_t(r.scalar<int16_t>()); break;
  case 4: m.unsigneds[key] = r.scalar<uint32_t>(); break;
  case 5: m.unsigneds[key] = uint64_t(r.scalar<int32_t>()); break;
  case 6: m.floats[key] = r.scalar<float>(); break;
  case 7: m.unsigneds[key] = r.scalar<uint8_t>() != 0; break;
  case 8: m.strings[key] = r.string(); break;
  case 10: m.unsigneds[key] = r.scalar<uint64_t>(); break;
  case 11: m.unsigneds[key] = uint64_t(r.scalar<int64_t>()); break;
  case 12: m.floats[key] = r.scalar<double>(); break;
  default: skipValue(r, type); break;
  }
}
struct Shard {
  GgufMetadata metadata;
  std::map<std::string, std::vector<uint64_t>, std::less<>> arrays;
  std::vector<GgufTensor> tensors;
  std::shared_ptr<WeightShardIdentity> identity;
};
Shard parse(const std::filesystem::path &path, GgufShardIndex::Mode mode) {
  HeaderReader r(path);
  char magic[4]; r.bytes(magic, 4);
  if (std::memcmp(magic, "GGUF", 4) || r.scalar<uint32_t>() != 3) throw GgufError("unsupported GGUF shard: " + path.string());
  const uint64_t tensorCount = r.scalar<uint64_t>(), keyCount = r.scalar<uint64_t>();
  if (tensorCount > 16384 || keyCount > 16384) throw GgufError("implausible GGUF header counts");
  Shard shard;
  std::set<std::string> keys, names;
  uint64_t retainedIntegers = 0;
  for (uint64_t i = 0; i < keyCount; ++i) {
    auto key = r.string();
    if (!keys.insert(key).second) throw GgufError("duplicate GGUF metadata key: " + key);
    const uint32_t type = r.scalar<uint32_t>();
    if (type == 9) {
      const uint32_t element = r.scalar<uint32_t>();
      const uint64_t count = r.scalar<uint64_t>();
      if (key.starts_with("qwen4exp.") && count > 1024) throw GgufError("GGUF architecture array exceeds bounds: " + key);
      const bool integer = element == 0 || element == 1 || element == 2 || element == 3 ||
          element == 4 || element == 5 || element == 7 || element == 10 || element == 11;
      if (integer && count <= 1024) {
        if (count > (1u << 20) - retainedIntegers) throw GgufError("GGUF numeric arrays exceed bounds");
        retainedIntegers += count;
        auto &array = shard.arrays[key];
        for (uint64_t j = 0; j < count; ++j) {
          GgufMetadata value; scalarMetadata(r, value, "value", element);
          array.push_back(value.unsigneds.at("value"));
        }
      } else if (element == 8 || element == 9) {
        if (count > (uint64_t(r.state.st_size) - r.position) / 8) throw GgufError("GGUF metadata array is truncated");
        for (uint64_t j = 0; j < count; ++j) skipValue(r, element, 1);
      } else r.skip(checkedMultiply<GgufError>(count, scalarBytes(element), "GGUF metadata size"));
    } else scalarMetadata(r, shard.metadata, key, type);
  }
  const uint64_t alignment = shard.metadata.unsignedValue("general.alignment").value_or(32);
  if (!alignment || alignment > 65536 || (alignment & (alignment - 1))) throw GgufError("invalid GGUF alignment");
  uint64_t largestEnd = 0;
  std::vector<std::pair<uint64_t, uint64_t>> extents;
  for (uint64_t i = 0; i < tensorCount; ++i) {
    GgufTensor t; t.name = r.string();
    if (!names.insert(t.name).second) throw GgufError("duplicate GGUF tensor: " + t.name);
    const uint32_t rank = r.scalar<uint32_t>();
    if (!rank || rank > 4) throw GgufError("invalid GGUF tensor rank: " + t.name);
    for (uint32_t d = 0; d < rank; ++d) {
      const uint64_t dim = r.scalar<uint64_t>();
      if (!dim) throw GgufError("zero GGUF tensor dimension: " + t.name);
      t.dims.push_back(dim);
    }
    (void)t.elements();
    t.type = r.scalar<uint32_t>(); t.offset = r.scalar<uint64_t>();
    const auto *traits = ggmlTypeTraits(t.type);
    if (!traits || t.columns() % traits->blockElements) throw GgufError("unsupported or unaligned GGUF tensor type: " + t.name);
    t.bytes = checkedMultiply<GgufError>(checkedMultiply<GgufError>(t.rows(), t.columns() / traits->blockElements,
        "GGUF tensor size"), traits->blockBytes, "GGUF tensor size");
    if (t.offset % alignment) throw GgufError("misaligned GGUF tensor offset: " + t.name);
    const uint64_t end = checkedAdd<GgufError>(t.offset, t.bytes, "GGUF tensor extent");
    largestEnd = std::max(largestEnd, end); extents.emplace_back(t.offset, end);
    shard.tensors.push_back(std::move(t));
  }
  std::sort(extents.begin(), extents.end());
  for (size_t i = 1; i < extents.size(); ++i)
    if (extents[i].first < extents[i - 1].second) throw GgufError("overlapping GGUF tensors: " + path.string());
  r.skip((alignment - r.position % alignment) % alignment);
  shard.identity = std::make_shared<WeightShardIdentity>();
  auto &id = *shard.identity;
  id.headerSha256 = r.digest(); id.dataOffset = r.position; id.fileBytes = uint64_t(r.state.st_size);
  id.requiredFileBytes = checkedAdd<GgufError>(id.dataOffset, largestEnd, "GGUF file size");
  id.device = uint64_t(r.state.st_dev); id.inode = uint64_t(r.state.st_ino);
  id.modifiedSeconds = r.state.st_mtimespec.tv_sec; id.modifiedNanoseconds = r.state.st_mtimespec.tv_nsec;
  id.complete = id.fileBytes >= id.requiredFileBytes;
  if (!id.complete && mode == GgufShardIndex::Mode::Complete) throw GgufError("GGUF shard payload is truncated: " + path.string());
  return shard;
}
}
GgufShardIndex::GgufShardIndex(const std::vector<std::filesystem::path> &paths, Mode mode) {
  if (paths.empty() || paths.size() > 65535) throw GgufError("GGUF shard list is empty or too large");
  std::set<std::pair<uint64_t, uint64_t>> files;
  std::set<uint64_t> numbers;
  std::optional<uint64_t> globalCount;
  bool haveFirst = false;
  GgufMetadata architectureMetadata;
  std::map<std::string, unsigned, std::less<>> metadataKinds;
  const auto checkKind = [&](const std::string &name, unsigned kind) {
    if (!name.starts_with("qwen4exp.") && name != "general.architecture") return;
    auto [at, inserted] = metadataKinds.emplace(name, kind);
    if (!inserted && at->second != kind) throw GgufError("inconsistent GGUF architecture metadata type: " + name);
  };
  for (const auto &path : paths) {
    Shard shard = parse(path, mode);
    if (!files.emplace(shard.identity->device, shard.identity->inode).second) throw GgufError("duplicate GGUF shard file: " + path.string());
    for (const auto &[name, array] : shard.arrays) {
      checkKind(name, 3);
      auto [at, inserted] = arrays_.emplace(name, array);
      if (!inserted && at->second != array) throw GgufError("inconsistent GGUF numeric array: " + name);
    }
    const auto mergeScalars = [&](auto &known, const auto &incoming, unsigned kind) {
      for (const auto &[name, value] : incoming) {
        if (!name.starts_with("qwen4exp.") && name != "general.architecture") continue;
        checkKind(name, kind);
        auto [at, inserted] = known.emplace(name, value);
        if (!inserted && at->second != value) throw GgufError("inconsistent GGUF architecture metadata: " + name);
      }
    };
    mergeScalars(architectureMetadata.unsigneds, shard.metadata.unsigneds, 0);
    mergeScalars(architectureMetadata.floats, shard.metadata.floats, 1);
    mergeScalars(architectureMetadata.strings, shard.metadata.strings, 2);
    for (std::string_view key : {"split.count", "split.no", "split.tensors.count"})
      if (shard.metadata.strings.contains(key) || shard.metadata.floats.contains(key) || shard.arrays.contains(key))
        throw GgufError("invalid GGUF split metadata type: " + std::string(key));
    const auto count = shard.metadata.unsignedValue("split.count");
    const auto number = shard.metadata.unsignedValue("split.no");
    const auto tensors = shard.metadata.unsignedValue("split.tensors.count");
    if (count || number || tensors || paths.size() > 1) {
      if (!count || !number || !tensors || *count != paths.size() || *number >= *count ||
          !*tensors || *tensors > 16384 * paths.size()) throw GgufError("invalid or missing GGUF split metadata: " + path.string());
      if (!numbers.insert(*number).second) throw GgufError("duplicate GGUF split.no: " + path.string());
      if (globalCount && *globalCount != *tensors) throw GgufError("inconsistent GGUF global tensor count");
      globalCount = *tensors;
    }
    const bool firstShard = number.value_or(0) == 0;
    if (firstShard) { metadata_ = shard.metadata; haveFirst = true; }
    for (auto &tensor : shard.tensors) {
      if (!lookup_.emplace(tensor.name, tensors_.size()).second) throw GgufError("duplicate tensor across GGUF shards: " + tensor.name);
      const uint64_t offset = checkedAdd<GgufError>(shard.identity->dataOffset, tensor.offset, "GGUF absolute tensor offset");
      tensors_.push_back({std::move(tensor), path, offset, shard.identity});
    }
  }
  if (!haveFirst || (globalCount && *globalCount != tensors_.size())) throw GgufError("missing GGUF shard or global tensor count mismatch");
}
std::optional<std::span<const uint64_t>> GgufShardIndex::unsignedArray(std::string_view name) const {
  auto found = arrays_.find(name);
  if (found == arrays_.end()) return std::nullopt;
  return std::span<const uint64_t>(found->second);
}
const GgufShardTensor &GgufShardIndex::require(std::string_view name) const {
  const auto found = lookup_.find(name);
  if (found == lookup_.end()) throw GgufError("missing GGUF tensor: " + std::string(name));
  return tensors_[found->second];
}
WeightRange GgufShardIndex::rows(std::string_view name, uint64_t first, uint64_t count) const {
  const auto &t = require(name);
  if (!count || first > t.tensor.rows() || count > t.tensor.rows() - first) throw GgufError("GGUF row selection is out of bounds: " + std::string(name));
  const uint64_t stride = t.tensor.bytes / t.tensor.rows();
  return {t.path, t.offset + first * stride, count * stride, t.tensor.bytes, t.source};
}
WeightRange GgufShardIndex::expert(std::string_view name, uint64_t index) const {
  const auto &t = require(name);
  if (t.tensor.dims.size() != 3 || index >= t.tensor.dims[2]) throw GgufError("GGUF expert selection is out of bounds: " + std::string(name));
  return rows(name, index * t.tensor.dims[1], t.tensor.dims[1]);
}
} // namespace splash::model
