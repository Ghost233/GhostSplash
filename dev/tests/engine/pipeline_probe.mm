#import <Foundation/Foundation.h>
#include "metal/BackendInstrumentation.hpp"
#include "AwakeClock.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>

using namespace splash::metal;

int main(int argc, const char **argv) {
  @autoreleasepool {
    try {
      if (argc < 3) throw std::invalid_argument("usage: pipeline-probe METALLIB KERNEL...");
      NSMutableArray *rows = [NSMutableArray array];
      const auto measure = [&](MetalBackend &backend, const char *kernel, NSString *state, bool hit) {
        double previousCreationSeconds = 0;
        for (const auto &entry : BackendInstrumentation::pipelineStatistics(backend))
          if (entry.kernel == kernel) previousCreationSeconds = entry.creationSeconds;
        const auto started = splash::AwakeClock::now();
        BackendInstrumentation::preparePipeline(backend, kernel);
        const double seconds = std::chrono::duration<double>(splash::AwakeClock::now() - started).count();
        PipelineStatistics selected;
        for (const auto &entry : BackendInstrumentation::pipelineStatistics(backend))
          if (entry.kernel == kernel) selected = entry;
        if (selected.creations != 1 || selected.failures || selected.cacheHits != uint64_t(hit) ||
            BackendInstrumentation::submittedCommands(backend) != 0)
          throw std::runtime_error("pipeline observation changed creation caching or submitted a test command");
        [rows addObject:@{@"kernel": [NSString stringWithUTF8String:kernel], @"state": state,
                         @"backend_cache_hit": @(hit), @"lookup_total_ms": @(seconds * 1000),
                         @"creation_ms": @((selected.creationSeconds - previousCreationSeconds) * 1000),
                         @"total_creations": @(selected.creations), @"total_cache_hits": @(selected.cacheHits)}];
      };
      for (int i = 2; i < argc; ++i) {
        {
          MetalBackend backend(argv[1]);
          measure(backend, argv[i], @"new_backend_first_lookup", false);
          measure(backend, argv[i], @"same_backend_repeat", true);
        }
        MetalBackend fresh(argv[1]);
        measure(fresh, argv[i], @"second_backend_first_lookup", false);
      }
      MetalBackend device(argv[1]);
      NSDictionary *report = @{@"rows": rows, @"gpu": [NSString stringWithUTF8String:device.capabilities().deviceName.c_str()],
          @"gpu_family": @(device.capabilities().appleGpuFamily), @"gpu_cores": @(device.capabilities().gpuCoreCount),
          @"system_cache_state": @"uncontrolled; no system cache was purged; timings cannot prove system cache hits",
          @"scope": @"pipeline creation and lookup only; no workload dispatches"};
      NSError *error = nil;
      NSData *data = [NSJSONSerialization dataWithJSONObject:report options:0 error:&error];
      if (!data) throw std::runtime_error(error.localizedDescription.UTF8String);
      std::cout << "PIPELINE_PROBE " << [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding].UTF8String << '\n';
      std::cout << "pipeline probe: cache counters and no-dispatch assertion PASS\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "pipeline probe FAIL: " << error.what() << '\n';
      return 1;
    }
  }
}
