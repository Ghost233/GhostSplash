# Metal 测试观测与结果复用

本次只修改测试观测和执行工具。生产算子、Shader Validation 设置、数值容差和
业务断言保持不变。服务构建不链接观测后端；没有新增项目级持久化 pipeline 缓存。

## 日志和命令

```sh
# 最终验收：默认重新运行每个测试，不复用成功结果。
make -j4 check test-sanitizers NATIVE_TEST_ARGS=

# 调试：仅在身份和日志完整匹配时复用成功结果。
make check-native-metal NATIVE_TEST_ARGS=--reuse

# 单项检查，同样输出开始、结束、耗时、退出码和心跳。
MTL_SHADER_VALIDATION=1 .venv/bin/python -u dev/tools/run_native_test.py -- \
  build/engine-tests/gguf-projection build/engine-tests/production-and-test.metallib

# 只测 pipeline 创建与查找，不提交算子工作负载。
make build/engine-tests/pipeline-probe
MTL_SHADER_VALIDATION=1 .venv/bin/python -u dev/tools/run_native_test.py -- \
  build/engine-tests/pipeline-probe build/splash.metallib \
  prefill_attention_q8_split decode_sample_topk32_sharded
```

执行器默认把逐项原始日志和 JSON 记录放到 `build/test-runs/`。
可用 `--directory 路径` 指定位置，`--heartbeat 秒数` 调整心跳，`--input 文件或目录`
声明不在命令参数中的额外测试数据。所有配置通过命令参数传入。

子进程使用 PTY，让 C/C++ 输出按行刷新；执行器持续转发并保存原始输出。
`START`、`HEARTBEAT`、`END` 标明命令、PID、耗时和真实子进程退出码。
取消时只清理该测试的进程组；取消、工具错误或运行期间输入变化都不会记为通过。

测试专用 `MetalBackendObserved.o` 对同步
`newComputePipelineStateWithFunction:error:` 调用记录 `PIPELINE_BEGIN` 和
`PIPELINE_END`，包括内核名、Unix 时间、耗时和成功／失败状态。
`PIPELINE_CACHE` 汇总同一 backend 内的命中、创建和失败次数。
3 个直接调用 Metal API 的旧 Q8 测试也记录同一创建区间。
两组 drafter fixture 共用的包装脚本实时透传原生 stderr，并输出
`FIXTURE_CHILD_END` 与原生退出码；原有 stdout 汇总和数值断言保留。
这不是 shader 实际执行时间，也不包括创建调用前的 function 查找时间。
持续时间使用仓库已有的 `AwakeClock`，系统休眠不计入持续时间。

## 复用规则

复用键包含源码和测试文件内容、实际可执行文件、构建配置、metallib、命令参数、
测试数据、当前目录、GPU、系统 build、Metal 编译器和校验模式。
校验模式包括 Metal debug／Shader Validation 和 sanitizer 运行选项。
只有真实退出 0、输入在运行前后未变、原始日志 SHA-256 完整匹配的记录才可复用。
缺少硬件身份、记录损坏、日志被改动或条件变化时重新执行。

调试阶段可以复用不受修改影响的结果；最终验收显式保持
`NATIVE_TEST_ARGS=`，避免把调试缓存当成最终全量运行。
源码提交、工作区是否有改动及输入内容摘要均记录在逐项 JSON 中。

## 首次诊断数据

Apple M5 Pro、20 GPU 核，Shader Validation 开启。两个独立进程使用同一
metallib，未清理系统缓存。因此“新 backend”只表示项目内存缓存为空，
不能称为已控制系统缓存的冷启动。

| 内核 | 进程 1 首次创建 API（ms） | 同 backend 再查找（ms） | 第二 backend 创建 API（ms） | 进程 2 首次创建 API（ms） |
| --- | ---: | ---: | ---: | ---: |
| `prefill_attention_q8_split` | 0.625584 | 0.000125 | 0.045333 | 0.166667 |
| `decode_sample_topk32_sharded` | 0.740458 | 0.000291 | 0.048209 | 0.255708 |

同 backend 再查找的新增创建次数为 0，命中次数增加 1，说明项目内存缓存生效。
新 backend 和新进程的时间差尚不能证明系统缓存是否命中。
这组结果也不能用于拆分此前 `gguf-projection` 约 8 分 28 秒的编译、执行和
数值检查占比；该次运行没有逐 pipeline 计时。

首次观测全量运行的 Python 为 1,474 项（4 项条件跳过），完整命令退出 0。
该轮发现 fixture 包装会截获 stderr，因此保留这轮结果并补齐透传。
透传修补后，两个真实 drafter fixture 和成功／失败日志用例均通过。
服务二进制和全部 metallib 的 SHA-256 与修补前完全一致。

## 最终完整验收

最终执行 `make -j4 check test-sanitizers NATIVE_TEST_ARGS=`，真实退出码 0。
1,476 项 Python 测试（4 项条件跳过）、47 条 Metal 检查命令，以及源码、
CPU、全部原生构建和 sanitizer 门禁通过。没有使用结果复用。

| 检查 | 总耗时（秒） | 创建 API 累计（秒） | 创建调用数 | backend 内存命中数 |
| --- | ---: | ---: | ---: | ---: |
| GGUF projection | 115.296 | 49.437 | 1,576 | 82,892 |
| MoE | 1,467.164 | 894.217 | 44 | 58,605 |
| MoE，512 专家 | 515.280 | 0.305 | 44 | 58,605 |
| Q8 flash attention | 471.073 | 0.896 | 90 | 无项目级缓存 |

创建 API 以外的时间没有进一步拆分，不能都算作 GPU 执行或 CPU 数值检查。
在相同二进制与 metallib 下重复运行，部分创建调用仍需要数十秒或数分钟；
另一些新进程中的同名调用则降到毫秒级。现有数据不能确认系统缓存的内部命中
或稳定保留策略，因此本轮只保留观测能力，不增加持久化 pipeline 缓存。
这些数据来自 Shader Validation 下的正确性检查，不代表模型服务速度。

原始诊断日志在 `build/upstream-140-merge-20261021/`。
最终命令、退出码、输入版本和诊断数据归档到
`docs/data/metal-test-observation-2026-10-21/`。此次不运行模型性能测试。
