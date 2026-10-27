# Flash Next Q4H KV

新增 Flash Next 稀疏 attention 的 Q4H KV 路径，复用现有 Q4H v1 协议。
启动器默认仍为 INT8；显式开启使用：

```sh
sh swift15flashnextsplash.sh --kv-format q4h
```

Q4_0 和未旋转 Q4_0 仍被拒绝。PLE、专家驻留、ANE、lookup、fast-topk、
compact 和 FFN 权重策略不随 KV 格式改变。

## 实现范围

- Q/K 在 RoPE 后作 H256 旋转，V 作四组 H64 旋转，按原规则舍入至 BF16。
- KV 页沿用相邻有符号 nibble、token-major K/V 和每 token/head 的 FP32 scale。
- 稀疏块 gather 在可复用的 threadgroup tile 中将 nibble 展开为有符号码，
  复用 INT8 的乘加、scale 和 softmax 路径；不创建完整 BF16 KV 副本。
- 合并各 split 后逆变换 H64，恢复值的原始坐标，再交给现有 gate。
- KV 文件记录、历史暂存和热块定位传输原始量化字节，保留现有提交与拒绝行覆盖规则。
- QSA 索引仍由独立的 indexer 投影生成，不对它应用 attention KV 的旋转。
- Q4H 的持久缓存命名空间区分 INT8、BF16 及各自的槽布局。

## 布局大小

以下按 Flash Next 的 12 个 attention 层、2 个 KV head、256 维 head、
每页 32 tokens 和 128 维索引计算，不是内存或性能测量结果。

| 项目 | INT8 | Q4H |
|---|---:|---:|
| 内存中的完整模型页，含索引 | 414 KiB | 222 KiB |
| 一层四个 tokens 的记录 | 4160 B | 2112 B |
| SSD 记录 pitch | 4160 B | 4096 B |
| 整页 SSD 槽，对齐后 | 416 KiB | 408 KiB |
| 单个缺失块的对齐回读 | 8 KiB | 4 KiB |

SSD 槽需要填充，不能假定磁盘占用或顺序暂存读取量减半。
主模型权重、PLE 和单份 GDN 状态的大小不受该格式影响。

## 正确性验证

数值容差和业务断言未放宽，保留 Shader Validation：

```sh
MTL_SHADER_VALIDATION=1 .venv/bin/python dev/tools/run_native_test.py -- \
  build/engine-tests/paged-attention-plan \
  build/engine-tests/production-and-test.metallib --format=q4h

make -j4 check test-sanitizers NATIVE_TEST_ARGS=
```

定点检查已通过：Q4H 稀疏 prefill／verify、H256/H64 投影、逐 token/head
变化的 scale、部分 tile、不同历史长度及多 lane、块定位、SSD 文件与持久缓存往返、
一／双缓冲历史暂存。暂存和块定位输出必须与原页路径逐位一致。

真实模型的 resident／spill 校验要求同一执行顺序下的 token 和 logits 摘要完全一致。
首次尝试在加载前被主机内存门禁拒绝：权重需要 36,084 MiB，主机当时可用 36,847 MiB，
工具另保留 2048 MiB。该次退出码为 1，不计为数值通过。

完整回归 `make -j4 check test-sanitizers NATIVE_TEST_ARGS=` 已通过，退出码为 0：
47 项原生 Metal 检查通过，各自记录的构建输入在运行前后均未变化；
Python 1479 项测试通过，其中 4 项跳过，源码、构建和 sanitizer 检查通过。
默认沙箱中的首次运行因 Metal 设备不可见退出 2，保留该环境失败记录，
随后在原生环境完整重跑并通过。

首次完整回归的版本为 `5b4096b57c97c4039bb7a63b06d395c20e6f2807` 加当时的未提交修改，
该次原生构建标识为
`src-da60824348f100f0a8f0c9c90b75a25d7629839d4feb52306802157c54b547dd`。

主机可用内存随后达到门禁，真实模型全驻留校验已启动；
按用户要求在完成前停止，原生退出码为 -15（SIGTERM），脚本退出码为 143。
该项保留为待验证，不计为数值通过；测试进程和自动等待脚本均已退出。

恢复正确性验证后，三次真实模型运行均因原有 120 秒 Metal 命令看门狗退出，
真实退出码均为 1。输入为 32,768 tokens、上下文容量为 262,144，
专家全驻留、PLE 缓存 128 MiB、双缓冲历史暂存，Shader Validation 均开启。

| KV | Metal 库 | 超时命令的 dispatch 数 | 退出码 |
|---|---|---:|---:|
| Q4H | production-and-test.metallib | 6060 | 1 |
| INT8 | production-and-test.metallib | 2754 | 1 |
| Q4H | splash.metallib，正式生产库 | 6060 | 1 |

错误为 `Metal command completion timed out after 120 seconds`，命令状态为
`scheduled`。这些运行没有完成 resident／spill 的 token 和 logits 摘要对照，
不计为真实模型正确性通过。INT8 也有同类超时，正式生产库也复现 Q4H 超时；
当时尚未确定是公共执行路径、运行负载还是其他原因，不能据此认定 Q4H 数值有误。
原超时、数值容差和业务断言均未修改。

同轮补跑的 `kv-block-cache` 控制测试通过，含 BF16、INT8 和 Q4H 的块定位对照，
退出码为 0。复用完整回归前核对了 656 个源码输入，均未变化。
三次真实模型验证进程均已退出，模型资源已释放。
本轮明细保存在
`build/flash-next-q4h-20261021/correctness-resume-20261021141814Z/summary.json`。

这些初始结果见 `build/flash-next-q4h-20261021/verification.json`。
原始日志、取消记录和失败退出码均保留。

## 长 GPU 提交超时修复

真实模型的大图包含 2754 或 6060 次 dispatch。原提交缺少中间 GPU 完成边界，
看门狗只根据提交和 host step 推进期限。一个只含 9 次 GPU 加法的分段用例也能复现：
GPU buffer 持续完成，旧实现仍在 100 毫秒测试门限处报错。

修复将纯 GPU 图拆成每段最多 256 次 dispatch 的 Metal command buffer。
含事件的图保留原边界，外部事件等待和 512 个 buffer 的队列上限沿用原规则。
逻辑提交仍等待全部分段完成，之后才提交模型状态。

看门狗按成功完成分段的实际 `GPUEndTime` 识别进度。
回调的交付延迟不计为进度，乱序的旧时间戳也不能缩短新期限。
失败 buffer 和保持 GPU 唤醒的空转 buffer 都不计入进度。
生产门限仍为 120 秒；单段 GPU 或 CPU 工作停滞仍会报错。

定点回归覆盖 GPU 完成进度、自动分段、外部等待、乱序回调和 awake／idle host step。
修复前的失败记录与修复后的通过记录均保留。
数值 attention 测试和所有 Shader 源码均未修改。

修复后的原始 32K 校验完整通过，真实退出码为 0：

```sh
MTL_SHADER_VALIDATION=1 .venv/bin/python -u dev/tools/run_native_test.py \
  --directory build/flash-next-timeout-fix-20261021/native -- \
  build/engine-tests/model-runtime-oracle build/splash.metallib \
  install/models-local/.resolved/5e4f5bd7d999acef3a7954198f0b9946a326334fe1197b0e00f441d441304f08 \
  --kv-format q4h --ple-memory 128M --max-context 262144 \
  --spill-tokens 32768 --spill-buffers 2 --spill-only
```

每个请求输入 32,768 tokens，上下文容量为 262,144。
专家权重全驻留，PLE 使用真实 SSD 源文件和 128 MiB 缓存。
同一执行顺序下，resident／spill 的 token 和完整词表 logits 摘要完全一致。
B1／B4／B2 混合 decode、不可读数据重试、状态恢复和资源回收断言均通过。
历史暂存覆盖 39,840 页。主对照中，单条逻辑命令最多使用 27 个 command buffer。
权重及其他构建输入的运行前后哈希均一致，Shader Validation 始终开启。

真实模型使用现有 `TestKvCopy` 内存副本，校验生产的 spill 暂存和块定位读取。
实际 SSD 文件往返另由 `kv-page-tier` 覆盖。
候选版本的内存门禁拒绝和取消记录单独保留，不计为通过。

单独验证超时修复时的原生构建标识为
`src-f6fb5b6b5a0539590701c0b5141763264849a79057453f8c857b793c33e5e4d5`。

## SSD reader 释放时序

完整门禁随后在 `kv-page-tier-host-tsan` 发现释放时序失败：
`the tier left its pages with disk IO still running`。
原 `stop()` 等到了任务结束，但 reader 线程仍可能持有 batch。
batch 捕获的磁盘 slot 引用因此晚于 tier 释放，预算未及时归零。

修复让 `stop()` 等到相关 reader 释放 batch 引用后再返回。
reader 在线程锁内先释放引用，再通知等待方。
原预算归零和回调结束断言保持不变，测试没有增加销毁后的等待。

新增弱引用用例能在不加载模型、不读磁盘时复现。
修复后，普通、ASAN／UBSAN 和 TSAN 检查通过。
原 KV tier TSAN 用例连续 20 轮通过。
此前完整门禁的失败日志和剩余任务的取消记录保留，不计为通过。

包含两项修复的原生构建标识为
`src-8eca891f996a44c92947cf2a70f4643c1e8199ee641fa010d6ad24ddeeaf0cbd`。
最终按 CPU／sanitizer、真实 32K、完整 Metal 三阶段串行验收，退出码均为 0：

- CPU、源码、构建和 sanitizer 检查通过。Python 1479 项测试完成，4 项跳过。
- 最终源码再次完整通过同一 32K 命令。token／logits 对照、故障重试和回收断言均通过。
- 47 项原生 Metal 检查按仓库原有验证模式全部通过，构建输入前后哈希一致。

真实 32K、attention、KV 和 Metal 后端关键检查均显式开启 Shader Validation。
完整 Metal 门禁中，39 项显式开启，2 项原有 GGUF 探针使用 API 验证，6 项沿用默认模式。
这些设置均未修改，各项实际验证模式保存在验证记录中。

验收结束后再次核对 654 个源码输入，均与各原生测试的记录一致。
构建版本核对和 `git diff --check` 通过。测试与模型进程已退出，内存压力为 normal。
最新状态、完整命令、源码版本和退出码见
`build/flash-next-timeout-fix-20261021/verification.json`。
本轮未运行模型性能测试，启动器的 INT8 默认值保持原样。
