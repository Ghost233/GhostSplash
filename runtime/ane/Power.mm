#include "ane/Power.hpp"
#include "AwakeClock.hpp"
#include <CoreFoundation/CoreFoundation.h>
#include <condition_variable>
#include <dlfcn.h>
#include <map>
#include <mutex>
#include <thread>
namespace splash::ane {
namespace {
double nowSeconds() { return std::chrono::duration<double>(AwakeClock::now().time_since_epoch()).count(); }
struct Reader {
  void *library = nullptr;
  CFMutableDictionaryRef channels = nullptr, selected = nullptr;
  CFTypeRef subscription = nullptr;
  using Copy = CFDictionaryRef (*)(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
  using Subscribe = CFTypeRef (*)(CFTypeRef, CFMutableDictionaryRef, CFMutableDictionaryRef *, uint64_t, CFTypeRef);
  using Sample = CFDictionaryRef (*)(CFTypeRef, CFMutableDictionaryRef, CFTypeRef);
  using Name = CFStringRef (*)(CFDictionaryRef);
  using Value = int64_t (*)(CFDictionaryRef, int);
  Sample sample = nullptr;
  Name name = nullptr, unit = nullptr;
  Value value = nullptr;
  std::map<std::string, int64_t> previous;
  double previousTime = 0;
  uint64_t sequence = 0;
  std::string error;
  template<class T> T symbol(const char *symbolName) { return reinterpret_cast<T>(dlsym(library, symbolName)); }
  static std::string text(CFStringRef string) {
    char buffer[256]{};
    return string && CFStringGetCString(string, buffer, sizeof buffer, kCFStringEncodingUTF8) ? buffer : "";
  }
  Reader() {
    library = dlopen("/usr/lib/libIOReport.dylib", RTLD_LAZY | RTLD_LOCAL);
    if (!library) { error = "IOReport library unavailable"; return; }
    const auto copy = symbol<Copy>("IOReportCopyChannelsInGroup");
    const auto subscribe = symbol<Subscribe>("IOReportCreateSubscription");
    sample = symbol<Sample>("IOReportCreateSamples");
    name = symbol<Name>("IOReportChannelGetChannelName");
    unit = symbol<Name>("IOReportChannelGetUnitLabel");
    value = symbol<Value>("IOReportSimpleGetIntegerValue");
    if (!copy || !subscribe || !sample || !name || !unit || !value) { error = "IOReport symbols unavailable"; return; }
    CFDictionaryRef raw = copy(CFSTR("Energy Model"), nullptr, 0, 0, 0);
    if (!raw) { error = "Energy Model channels unavailable"; return; }
    channels = CFDictionaryCreateMutableCopy(nullptr, 0, raw);
    CFRelease(raw);
    if (!channels) { error = "Energy Model channel copy unavailable"; return; }
    CFArrayRef entries = static_cast<CFArrayRef>(CFDictionaryGetValue(channels, CFSTR("IOReportChannels")));
    if (!entries || CFGetTypeID(entries) != CFArrayGetTypeID()) { error = "Energy Model channel array unavailable"; return; }
    CFMutableArrayRef ane = CFArrayCreateMutable(nullptr, 0, &kCFTypeArrayCallBacks);
    for (CFIndex i = 0; i < CFArrayGetCount(entries); ++i) {
      CFDictionaryRef entry = static_cast<CFDictionaryRef>(CFArrayGetValueAtIndex(entries, i));
      if (text(name(entry)).starts_with("ANE")) CFArrayAppendValue(ane, entry);
    }
    const bool found = CFArrayGetCount(ane) > 0;
    CFDictionarySetValue(channels, CFSTR("IOReportChannels"), ane);
    CFRelease(ane);
    if (!found) { error = "ANE energy channel unavailable"; return; }
    subscription = subscribe(nullptr, channels, &selected, 0, nullptr);
    if (!subscription) error = "IOReport subscription unavailable";
  }
  ~Reader() {
    if (subscription) CFRelease(subscription);
    if (selected) CFRelease(selected);
    if (channels) CFRelease(channels);
    if (library) dlclose(library);
  }
  PowerSample read() {
    const double now = nowSeconds();
    PowerSample result{std::nullopt, now, ++sequence, previousTime ? now - previousTime : 0};
    if (!subscription) return result;
    CFDictionaryRef report = sample(subscription, channels, nullptr);
    if (!report) { error = "IOReport sample unavailable"; previous.clear(); previousTime = now; return result; }
    CFArrayRef entries = static_cast<CFArrayRef>(CFDictionaryGetValue(report, CFSTR("IOReportChannels")));
    bool valid = entries && CFGetTypeID(entries) == CFArrayGetTypeID() && CFArrayGetCount(entries) > 0;
    std::map<std::string, int64_t> current;
    double watts = 0;
    if (valid) for (CFIndex i = 0; i < CFArrayGetCount(entries); ++i) {
      CFDictionaryRef entry = static_cast<CFDictionaryRef>(CFArrayGetValueAtIndex(entries, i));
      const std::string label = text(name(entry));
      if (!label.starts_with("ANE")) continue;
      const int64_t energy = value(entry, 0);
      current[label] = energy;
      const auto old = previous.find(label);
      const auto power = old == previous.end() ? std::nullopt :
          powerWatts(old->second, energy, text(unit(entry)), result.intervalSeconds);
      if (!power) valid = false;
      else watts += *power;
    }
    CFRelease(report);
    valid = valid && !current.empty() && current.size() == previous.size();
    previous = std::move(current); previousTime = now;
    if (valid) { result.watts = watts; error.clear(); }
    else error = "ANE energy sample warming up or invalid";
    return result;
  }
};
} // namespace
struct PowerMonitor::Impl {
  mutable std::mutex mutex;
  std::condition_variable changed;
  bool enabled = false, stopping = false;
  PowerSample reading;
  std::vector<PowerSample> history;
  std::string error = "not sampling";
  double maximumIdleWatts;
  double quiet = 0;
  std::thread worker;
  explicit Impl(double watts) : maximumIdleWatts(watts), worker([this] { run(); }) {}
  ~Impl() {
    { std::lock_guard lock(mutex); stopping = true; }
    changed.notify_all(); worker.join();
  }
  void run() noexcept {
    std::unique_ptr<Reader> reader;
    std::unique_lock lock(mutex);
    while (!stopping) {
      changed.wait(lock, [&] { return stopping || enabled; });
      if (stopping) break;
      lock.unlock();
      PowerSample next;
      std::string why;
      try {
        if (!reader) reader = std::make_unique<Reader>();
        next = reader->read(); why = reader->error;
      } catch (...) { why = "ANE energy sampling failed"; reader.reset(); }
      lock.lock();
      if (enabled) {
        if (next.watts && *next.watts >= 0 && *next.watts <= maximumIdleWatts &&
            next.intervalSeconds > 0 && next.intervalSeconds <= 2)
          quiet += next.intervalSeconds;
        else quiet = 0;
        next.quietSeconds = quiet;
        reading = next; error = std::move(why);
        history.push_back(next);
        if (history.size() > 32) history.erase(history.begin());
      }
      changed.wait_until(lock, AwakeClock::now() + std::chrono::seconds(1), [&] { return stopping || !enabled; });
      if (!enabled) { reader.reset(); quiet = 0; }
    }
  }
};
PowerMonitor::PowerMonitor(double watts) : impl_(std::make_unique<Impl>(watts)) {}
PowerMonitor::~PowerMonitor() = default;
void PowerMonitor::active(bool value) {
  { std::lock_guard lock(impl_->mutex); if (impl_->enabled != value) {
      impl_->reading = {}; impl_->quiet = 0; impl_->history.clear();
    }
    impl_->enabled = value; }
  impl_->changed.notify_all();
}
PowerSample PowerMonitor::latest() const { std::lock_guard lock(impl_->mutex); return impl_->reading; }
std::vector<PowerSample> PowerMonitor::samples() const { std::lock_guard lock(impl_->mutex); return impl_->history; }
std::string PowerMonitor::reason() const { std::lock_guard lock(impl_->mutex); return impl_->error; }
} // namespace splash::ane
