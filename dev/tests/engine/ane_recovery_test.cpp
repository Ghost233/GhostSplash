#include "TestChecks.hpp"
#include "ops/AneFfnRecovery.hpp"
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <limits>
#include <vector>
using splash::test::require;
using namespace splash::ops::ane_ffn;
using splash::ane::PowerSample;
using splash::ane::powerWatts;
int main() {
  RecoveryGate gate({.enabled = true});
  gate.stopped(0);
  for (unsigned i = 1; i <= 10; ++i)
    require(!gate.ready(i, {0.0, double(i), i, 1.0}), "cooldown was ignored");
  require(!gate.ready(60, {0.0, 10, 10, 1.0}), "stale power allowed a retry");
  for (unsigned i = 61; i < 70; ++i)
    require(!gate.ready(i, {0.0, double(i), i, 1.0}), "quiet window was too short");
  require(gate.ready(70, {0.0, 70, 70, 1.0}), "valid quiet window did not permit retry");
  gate.failed(70);
  require(gate.nextAttempt() == 190, "failed retry did not back off to 120 seconds");
  gate.failed(190);
  require(gate.nextAttempt() == 430, "second failure did not back off to 240 seconds");
  gate.failed(430);
  require(gate.nextAttempt() == 730, "retry delay was not capped at 300 seconds");
  gate.stable();
  gate.stopped(1000);
  for (unsigned i = 1001; i <= 1061; ++i)
    require(!gate.ready(i, {0.5, double(i), i, 1.0}), "busy ANE allowed a retry");
  for (unsigned i = 1062; i < 1072; ++i)
    require(!gate.ready(i, {std::nullopt, double(i), i, 1.0}), "missing power was treated as zero");
  require(!gate.ready(1073, {std::numeric_limits<double>::quiet_NaN(), 1073, 1073, 1}), "NaN power allowed a retry");
  require(!gate.ready(1074, {-0.1, 1074, 1074, 1}), "negative power allowed a retry");
  for (unsigned i = 1075; i < 1084; ++i)
    require(!gate.ready(i, {0.05, double(i), i, 1.0}), "quiet duration was not reset");
  require(gate.ready(1084, {0.05, 1084, 1084, 1}), "low nonzero power was rejected");
  RecoveryGate duplicates({.enabled = true, .cooldownSeconds = 0});
  duplicates.stopped(0);
  for (unsigned i = 0; i < 10; ++i)
    require(!duplicates.ready(1, {0.0, 1, 1, 1}), "duplicate samples filled the quiet window");
  require(duplicates.quietSeconds() == 1, "duplicate samples changed quiet duration");
  RecoveryGate asynchronous({.enabled = true});
  asynchronous.stopped(0);
  require(asynchronous.ready(61, {0.0, 61, 61, 1, 10}), "background quiet window was ignored");
  require(!asynchronous.ready(62, {0.0, 62, 62, 1, 1}), "busy interval between prefills was ignored");
  require(!asynchronous.ready(70, {0.0, 62, 62, 1, 10}), "stale accumulated window allowed recovery");
  RecoveryGate disabled({});
  disabled.stopped(0);
  for (unsigned i = 1; i <= 100; ++i)
    require(!disabled.ready(i, {0.0, double(i), i, 1.0}), "disabled recovery permitted retry");
  require(!powerWatts(100, 90, "mJ", 1), "reset energy counter was accepted");
  require(!powerWatts(100, 101, "unknown", 1), "unknown units were accepted");
  require(!powerWatts(100, 101, "mJ", 0), "zero interval was accepted");
  require(powerWatts(0, 0, "mJ", 1) == 0, "valid zero delta was rejected");
  require(powerWatts(100, 150, "mJ", 1) == 0.05, "energy-to-power units are wrong");
  require(!RecoverySetting{}.enabled, "recovery default was enabled");
  using splash::ane::PowerReferenceSetting;
  require(PowerReferenceSetting::parse("off").mode == PowerReferenceSetting::Mode::Off,
          "power reference default was enabled");
  require(PowerReferenceSetting::parse("auto").mode == PowerReferenceSetting::Mode::Automatic,
          "automatic power reference was ignored");
  require(PowerReferenceSetting::parse("6.0").manualWatts == 6.0, "manual watts were ignored");
  for (const char *invalid : {"", "0", "-1", "nan", "inf", "6junk"}) {
    bool rejected = false;
    try { static_cast<void>(PowerReferenceSetting::parse(invalid)); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "invalid power reference was accepted");
  }
  require(splash::ane::relativePowerPercent(3, 6) == 50, "relative percentage is wrong");
  require(!splash::ane::relativePowerPercent(3, 0) && !splash::ane::relativePowerPercent(3, 1e-320),
          "invalid relative percentage was serialized as a number");
  const std::vector<PowerSample> loaded{{2.0, 11, 1, 1}, {4.0, 12, 2, 1}, {3.0, 13, 3, 1},
                                     {std::nullopt, 14, 4, 1}, {50.0, 15.5, 5, 1}};
  const auto reference = splash::ane::summarizePowerReference(loaded, 10, 15, 0.0);
  require(reference.valid && reference.referenceWatts == 4 && reference.steadyWatts == 3 && reference.samples == 3,
          "reference peak, steady value or window filtering is wrong");
  require(!splash::ane::summarizePowerReference(loaded, 10, 15, 1.0).valid,
          "busy background produced a trusted reference");
  require(!splash::ane::summarizePowerReference({}, 10, 15, 0.0).valid,
          "missing samples produced a trusted reference");
  RecoverySetting relative{.enabled = true, .cooldownSeconds = 0};
  relative.maximumPowerPercent = 50;
  relative.referenceWatts = 6;
  require(relative.powerThreshold() == 3, "50 percent was not converted to 3 watts");
  RecoveryGate percentage(relative);
  percentage.stopped(0);
  require(!percentage.ready(11, {3.1, 11, 1, 1, 10}), "power above 50 percent allowed recovery");
  require(percentage.ready(12, {3.0, 12, 2, 1, 10}), "power at 50 percent was rejected");
  relative.referenceWatts.reset();
  RecoveryGate unknownReference(relative);
  unknownReference.stopped(0);
  require(!unknownReference.ready(11, {0.0, 11, 1, 1, 10}), "unknown reference was treated as zero load");
  std::cout << "ANE recovery cooldown, power validity, quiet window and backoff: PASS\n";
}
