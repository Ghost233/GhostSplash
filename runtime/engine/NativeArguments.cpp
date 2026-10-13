#include "engine/NativeArguments.hpp"

#include <charconv>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>

namespace splash::engine {
namespace {

template <typename T>
bool parsePositive(std::string_view value, T &result) {
  const char *end = value.data() + value.size();
  auto parsed = std::from_chars(value.data(), end, result);
  return parsed.ec == std::errc{} && parsed.ptr == end && result != 0;
}

uint64_t parseMaxMemory(std::string_view value) {
  if (value == "auto")
    return 0;
  uint64_t result = 0;
  if (!parsePositive(value, result))
    throw UsageError("MAX_MEMORY_BYTES must be auto or a positive integer");
  return result;
}

uint32_t parseMaxContext(std::string_view value,
                         const model::ModelCapabilities &capabilities) {
  if (value == "auto")
    return 0;
  uint32_t result = 0;
  if (!parsePositive(value, result) ||
      result > capabilities.maximumContextTokens) {
    throw UsageError("MAX_CONTEXT must be auto or an integer in [1, " +
                     std::to_string(capabilities.maximumContextTokens) + "]");
  }
  return result;
}

uint32_t parseMaxImagePatches(std::string_view value) {
  uint32_t result = 0;
  if (!parsePositive(value, result) || result % 4 ||
      result > ops::kMaximumImagePatches)
    throw UsageError("--max-image-patches requires a positive multiple of 4 "
                     "up to " + std::to_string(ops::kMaximumImagePatches));
  return result;
}

bool parseFinite(std::string_view value, double &result) {
  const char *end = value.data() + value.size();
  auto parsed = std::from_chars(value.data(), end, result);
  return parsed.ec == std::errc{} && parsed.ptr == end && std::isfinite(result);
}

bool parseSwitch(std::string_view option, std::string_view value) {
  if (value != "on" && value != "off")
    throw UsageError(std::string(option) + " requires on or off");
  return value == "on";
}

double parseRange(std::string_view option, std::string_view value, double maximum) {
  double result = 0;
  if (!parseFinite(value, result) || result < 0 || result > maximum)
    throw UsageError(std::string(option) + (maximum == std::numeric_limits<double>::max()
        ? " requires a finite nonnegative number"
        : " requires a finite number in [0, " + std::to_string(maximum) + "]"));
  return result;
}

double parseIdleRelease(std::string_view value) {
  if (value == "off")
    return std::numeric_limits<double>::infinity();
  double result = 0.0;
  if (!parseFinite(value, result) || result <= 0.0)
    throw UsageError("--idle-release requires off or a positive number of seconds");
  return result;
}

double parseDecodeShare(std::string_view value) {
  double result = 0.0;
  if (!parseFinite(value, result) || result < 0.0)
    throw UsageError("--decode-share requires a nonnegative number");
  return result;
}

std::filesystem::path requireModelRoot(std::string_view argument) {
  std::error_code error;
  const std::filesystem::path root =
      std::filesystem::canonical(std::filesystem::path(argument), error);
  if (error || !std::filesystem::is_directory(root, error))
    throw UsageError("MODEL_DIRECTORY must name an existing directory");
  if (!std::filesystem::is_directory(root / "target", error))
    throw UsageError("MODEL_DIRECTORY must hold the model's target/ directory");
  return root;
}

} // namespace

NativeArguments parseNativeArguments(int argc, const char *const *argv) {
  if (argc < 5 || std::string_view(argv[1]) != "serve-native") {
    throw UsageError("expected the serve-native command");
  }
  NativeArguments result;
  std::optional<double> aneShare;
  std::optional<uint32_t> aneRows;
  int next = 5;
  if (next < argc && !std::string_view(argv[next]).starts_with("--")) {
    const std::string_view quota(argv[next++]);
    if (quota != "0" && !parsePositive(quota, result.maxCacheDiskBytes))
      throw UsageError("MAX_CACHE_DISK_BYTES must be a nonnegative integer");
  }
  // Options follow as --name value pairs; a missing value fails its check.
  for (; next < argc; next += 2) {
    const std::string_view option(argv[next]);
    const std::string_view value(next + 1 < argc ? argv[next + 1] : "");
    if (option == "--kv-format") {
      if (value != "int8" && value != "bf16" && value != "q4h" &&
          value != "q4_0" && value != "q4_0_unrotated")
        throw UsageError("--kv-format requires int8, bf16, q4h, q4_0 or q4_0_unrotated");
      result.kvFormat = value == "int8" ? kv::Format::Int8
          : value == "q4h" ? kv::Format::Int4Hadamard
          : value == "q4_0" ? kv::Format::Q4_0Hadamard
          : value == "q4_0_unrotated" ? kv::Format::Q4_0Unrotated : kv::Format::BFloat16;
    } else if (option == "--decode-share") {
      result.decodeShare = parseDecodeShare(value);
    } else if (option == "--max-image-patches") {
      result.maxImagePatches = parseMaxImagePatches(value);
    } else if (option == "--cache-dir") {
      if (value.empty())
        throw UsageError("--cache-dir requires a directory");
      result.persistentCacheRoot = std::filesystem::absolute(std::filesystem::path(value));
    } else if (option == "--idle-release") {
      result.idleReleaseSeconds = parseIdleRelease(value);
    } else if (option == "--idle-sleep") {
      if (value != "prevent" && value != "allow")
        throw UsageError("--idle-sleep requires prevent or allow");
      result.preventIdleSleep = value == "prevent";
    } else if (option == "--ane") {
      if (value != "on" && value != "off")
        throw UsageError("--ane requires on or off");
      result.neuralEngine = value == "on";
    } else if (option == "--ane-ffn-share") {
      if (value == "auto") aneShare.reset();
      else aneShare = parseRange(option, value, 1);
    } else if (option == "--ane-ffn-minimum-rows") {
      uint32_t rows = 0;
      if (!parsePositive(value, rows) || rows < ops::AneFfn::kMinimumRows || rows > ops::AneFfn::kMaximumRows)
        throw UsageError("--ane-ffn-minimum-rows requires rows in [512, " + std::to_string(ops::AneFfn::kMaximumRows) + "]");
      aneRows = rows;
    } else if (option == "--ane-power-recovery") {
      result.aneFfn.recovery.enabled = parseSwitch(option, value);
    } else if (option == "--ane-power-reference") {
      try { result.aneFfn.powerReference = ane::PowerReferenceSetting::parse(value); }
      catch (const std::invalid_argument &error) { throw UsageError(error.what()); }
    } else if (option == "--ane-recovery-idle-watts") {
      result.aneFfn.recovery.maximumIdleWatts = parseRange(option, value, std::numeric_limits<double>::max());
    } else if (option == "--ane-recovery-max-power-percent") {
      result.aneFfn.recovery.maximumPowerPercent = parseRange(option, value, 100);
    } else if (option == "--prompt-lookup") {
      result.optionalFeatures.promptLookup = parseSwitch(option, value);
    } else if (option == "--compact-draft-head") {
      result.optionalFeatures.compactDraftHead = parseSwitch(option, value);
    } else if (option == "--fast-topk") {
      result.optionalFeatures.fastTopK = parseSwitch(option, value);
    } else if (option == "--compact-draft-ranking") {
      if (value.empty()) throw UsageError("--compact-draft-ranking requires a file");
      result.optionalFeatures.compactRanking = std::filesystem::absolute(std::filesystem::path(value));
    } else if (option == "--prompt-lookup-min-match" || option == "--compact-draft-capacity") {
      uint32_t number = 0;
      if (!parsePositive(value, number)) throw UsageError(std::string(option) + " requires a positive integer");
      if (option == "--prompt-lookup-min-match") result.optionalFeatures.lookupMinimumMatch = number;
      else result.optionalFeatures.compactCapacity = number;
    } else {
      throw UsageError("unexpected argument " + std::string(option));
    }
  }
  try {
    result.optionalFeatures.validate();
    const auto given = AneFfnSetting::fromGiven(aneShare, aneRows);
    result.aneFfn.given = given.given;
    result.neuralEngine = result.neuralEngine && given.enabled;
  } catch (const std::invalid_argument &error) { throw UsageError(error.what()); }
  result.aneFfn.enabled = result.neuralEngine;
  if (result.aneFfn.recovery.enabled && result.aneFfn.recovery.maximumPowerPercent &&
      result.aneFfn.powerReference.mode == ane::PowerReferenceSetting::Mode::Off)
    throw UsageError("--ane-recovery-max-power-percent requires --ane-power-reference auto or watts");
  if (!result.persistentCacheRoot.empty() && !result.maxCacheDiskBytes)
    throw UsageError("--cache-dir requires a MAX_CACHE_DISK_BYTES quota");
  result.modelRoot = requireModelRoot(argv[2]);
  result.model = model::inspectModelRoot(result.modelRoot);
  if (result.model.hasDraft() && !std::filesystem::is_directory(result.modelRoot / "draft"))
    throw UsageError("MODEL_DIRECTORY must hold the model's draft/ directory for this model");
  result.maxContext = parseMaxContext(argv[3], result.model.capabilities);
  result.maxMemoryBytes = parseMaxMemory(argv[4]);
  return result;
}

} // namespace splash::engine
