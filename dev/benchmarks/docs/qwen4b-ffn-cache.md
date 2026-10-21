# Qwen3.5-4B FFN 实验

本实验保留原 GGUF 权重，比较常驻 FP16 缓存、量化计算和独立算子候选。候选默认关闭。Metal 激活保持 BF16，累加保持 FP32；既有 ANE W8A8 路径保持原实现。

## 执行路径和对照目标

| 权重模式 | 常驻 FFN FP16 缓存 | 没有使用 ANE 的 prefill | ANE split 的 GPU 分支 |
|---|---|---|---|
| `full_fp16` | 全部 96 个矩阵 | 可使用缓存 | 使用量化 plane |
| `quantized` | 0 | 按块读取量化权重并计算 | 使用量化 plane |
| `limited_fp16` | 在指定预算内选择完整矩阵 | 缓存命中的矩阵使用 FP16，其余使用量化路径 | 使用量化 plane |

ANE 的 `leadingRows`、`leadingInputs` 视图不继承 FP16 缓存。因此，ANE 实际参与的大块 prefill 中，缓存对照首先衡量常驻缓存、启动成本和内存压力。低于 ANE 阈值的块可能使用不同路径。结果必须结合实际提交记录解释，不能把整个 32K prefill 都标为缓存计算。

`ffn_paths` 记录成功提交的 target FFN dispatch，包含 scope、kernel、逻辑行数和次数。startup 与 warmup 记录单列。编码后未提交的 graph 不计数。候选策略只标在 target FFN 投影上，并传递到 ANE 的 GPU views；attention 和 drafter 保持原策略。

当前模型的 FFN 包含 Q4_K 56 个、Q5_K 15 个、Q6_K 15 个、IQ4_XS 10 个矩阵。量化计算在 threadgroup 内生成有限的 FP16 tile，不生成完整矩阵的常驻副本。

## 正式服务的实验开关

正式服务可使用 `--experimental-quantized-ffn on`，直接复用 `quantized / baseline`。
默认 `off` 保留原 FFN 缓存策略。开关同时传入内存规划和权重加载，不读取环境变量。

```sh
./splash serve --model OWNER/REPO --experimental-quantized-ffn on
```

`OWNER/REPO` 应替换为实际模型选择。GMD 的服务启动参数也使用同名开关。
只有普通生成的 Qwen3.5-4B 支持开启，可配 Standard DFlash 或不带 drafter。
27B、MoE、Kev、D3 和 ModernBERT 拒绝开启，不静默忽略。
此开关不改变 ANE、KV、上下文、drafter、采样或其他实验算子。

启动清单先列出请求值，Ready 后显示实际状态。
`/status.startup_features.experimental_quantized_ffn` 为 `true` 时，`ffn_cache_bytes` 必须为 `0`。
旧引擎未报告状态字段时显示 `unknown`，不能按请求参数认定已经启用。
关闭只需使用 `--experimental-quantized-ffn off` 并重启。

此前固定 ANE 对照中，全量缓存额外占用 4.21875 GiB，计算未命中这份缓存。
这个结论只适用于当时的模型和路径，不保证所有配置下的速度相同。

2026-10-21 的接入验证使用真实 4B 权重，比较 64／1024 输入、固定 32 输出的 token。
另检查权重释放恢复和 10 GiB 预算、262,144 上下文容量的启动行为。
本轮不重跑原 32K 长轨迹和性能测试。
命令、退出码和状态证据见[验证记录](../../../docs/data/experimental-quantized-ffn-2026-10-21/verification.json)。

## 独立实验 profile

每个 profile 只改变表中的一项策略。先完成前一项的数值与性能分析，再选择下一项，不把全部 profile 展开成笛卡尔积。

| `--profile` | A 参考 | B 候选 |
|---|---|---|
| `cache` | full_fp16，原算子 | quantized，原算子 |
| `m64n32` | quantized，原 M32N64 | quantized，M64N32 |
| `fused_gate_up` | quantized，M64N32 两次投影 | quantized，M64N32 单 dispatch gate/up |
| `q4k_group_scale` | quantized，原算子 | 只对 Q4_K 后移 group scale 和 offset，其余格式回退 |
| `limited_cache` | quantized，无缓存 | limited_fp16，默认预算 1 GiB |

M64N32 每 simdgroup 计算 64 行，整个 threadgroup 覆盖 256 行。stage 为 8 KiB。融合候选共享一次查表初始化和一个双缓冲 stage，保留两次 K walk；它减少 dispatch 和 gate scratch 往返，暂不承诺共享全部输入加载。最终仍按 `bf16(bf16(up) × silu(bf16(gate)))` 舍入。

限额缓存使用 `--cache-budget-bytes`，例如 536870912、1073741824、2147483648。缓存以完整矩阵为单位选取，实际分配不得超过预算。小于一个矩阵的预算允许选择 0 个矩阵。预算预估和实际缓存均单列，不能假设预算会全部用满。

Q4_K 后移候选在组内乘加后应用 scale 和 offset，并用每组 16 个编码的 FP16 舍入残差恢复原权重语义。残差表只放在 SIMD 寄存器中，不生成完整 FFN 的 FP16 副本。独立 CPU 参考从原 GGUF 解码半精度权重，再核对补偿后的点积。原 half 误差门禁保持不变，失败仍返回 `needs_weight_rounding_analysis` 和非零退出。IQ4_XS 保留非均匀查表，绝不当作均匀整数乘 scale。

## 构建与准备

安装器 CPU 测试固定调用 `build/splash`，先构建默认二进制，再构建实验 variant。下面只编译和运行 CPU/Python 门禁，不执行 Metal 测试。

```sh
make -j4 all
make -j4 BUILD=build/ffn-cache-variants/block-8 \
  check-source check-native-cpu check-native-build check-python
make -j4 BUILD=build/ffn-cache-variants/block-8 test-sanitizers

python=.venv/bin/python
profile=cache
experiment_output=build/ffn-experiment-$(date +%Y%m%d-%H%M%S)
model_root=/Users/ghost233/.codex/worktrees/qwen35-4b-optimization/GhostSplash/install/models-local/.resolved/dae35a717bda8642d409dbb0ce8655296cd37aaa2288c2bd2893ca02244c4f43
off_model_root=/Users/ghost233/.codex/worktrees/qwen35-4b-optimization/GhostSplash/install/models-local/.resolved/0db89c15cde311ca967567bc505737ddd7383d4c31e1123b537d564f46a756f8

$python -m dev.benchmarks.qwen4b_ffn_cache --prepare \
  --profile "$profile" --output "$experiment_output" \
  --model-root "$model_root" --off-model-root "$off_model_root"
```

准备仅校验构建、哈希权重并生成固定输入。默认命令仅列计划，不启动推理。旧 v2 产物保持完整；源码改变后须在新目录准备，不复用旧身份记录。

## 正确性门禁

机器空闲后显式运行：

```sh
$python -m dev.benchmarks.qwen4b_ffn_cache --run-correctness \
  --profile "$profile" --output "$experiment_output" \
  --model-root "$model_root" --off-model-root "$off_model_root"
```

数值工具保留真实 96 个矩阵 × 8／33／64／136 行的 384 组基础 prefill 对照。8 行用例明确标为小 prefill。另按真实 gate/down 格式组合选代表层，补充：

- 实际 C1 decode `GateUp` 和 `Residual Down` plan，检查完整 FFN pipeline。
- 512 行的完整 FFN，以及 share=0.5 的 ANE GPU 分片 pipeline。
- gate/up 的 leading rows、down 的 leading inputs，保留原 physical K stride。
- M64 边界行数 64、65、127、128、129、255、256、257，以及 512／2048 行。
- 输出列 255／256 两侧、有效值和填充区保护，避免漏掉 plane tile 地址错误。

独立 CPU GGUF 解码器核对抽样列的完整权重行；抽样 FP64 点积和原 BF16 epilogue 检查数值误差。工具报告实际比较数量、decode 与 ANE views 覆盖及原 half contract。线性用例仅保留一个投影的临时缓存；完整 FFN pipeline 可同时保留至多 gate/up/down 三个临时缓存，结束后释放。

随后比较 64／2048 输入、固定 32 输出的两种配置。保留 token ID 和首次分歧。数值与 token 门禁通过后，才允许性能或分段诊断。事实检查作为补充，不替代数值和轨迹检查。

数值门禁每次使用独立 attempt 目录。失败后可以在同一冻结目录重试，旧 stdout、stderr 和 evidence 保留。

## 32K 长轨迹正确性

先完成同一冻结版本的数值和短轨迹门禁，再显式运行：

```sh
$python -m dev.benchmarks.qwen4b_ffn_cache --run-long-correctness \
  --profile "$profile" --output "$experiment_output" \
  --model-root "$model_root" --off-model-root "$off_model_root"
```

按本轮用户授权，正确性验证允许较高的桌面 GPU 利用率，不用性能入口的 50% 利用率阈值阻塞；未授权活动推理／benchmark 仍拒绝，设备互斥锁和运行中干扰检查仍保留。性能入口保持原 50% 阈值。数值、token、事实和内存断言不变。

用户明确授权指定推理服务并行时，可在正确性命令上追加 `--allow-inference-pid PID`。该参数只允许对应的 `llama-server`，不会忽略相同 PID 的 benchmark，也不能用于性能动作。本轮用户另已授权与推理服务并行做正确性，可使用 `--allow-concurrent-inference`，服务重启后的新 PID 同样记录。benchmark 仍拦截，两项许可都不能用于性能动作。原始 evidence 记录授权范围和实际观察到的并行活动；不停止服务。

此入口使用准备阶段的第 1 份固定 32,768-token 输入，按 A／B 比较实际输出 512 和 2048 token。忽略 EOS、冷前缀、温度 0、Q4H、ANE share=0.5／阈值 512、block=8、splits=16、并发 1 均保持固定。输入与权重身份、实际输出长度、EOS、事实、内存预算及实际算子路径继续校验。

长轨迹仍使用 `mode=correctness`。只有显式 `ffn_long_correctness=true` 且长度和并发匹配时才接受；不 warmup、不启动 stage profile，也不输出性能计时字段。该入口不做速度比较。A／B 首次 token 分歧写入 `long-correctness.json`，并停止该 profile 的后续输出长度。所有原始尝试保留。

最小 Q4_K 舍入回归可直接调用数值工具，并追加 `q4k_group_scale --rounding-regression`，它检查真实第 0 层 gate 的 64 和 257 行；完整数值门禁仍覆盖全部 480 项。最小回归追加 257 行，检查第二片的单行尾部与写入边界。

Q4_K group-scale 标量候选每个 dispatch 最多覆盖 256 行，在相邻行片之间划分 Metal command buffer；逻辑输入、物理权重步长、累加顺序和 epilogue 保持一致。该候选在 ANE 等待前完成独立 GPU 提交，并把汇合与下一层计算分开。显式分片图中，含事件等待的子提交在事件达到要求值后才进入 GPU 队列，避免在队列中消耗驱动的等待期限；异步接口、ANE 的 2 秒保护和整条命令的 120 秒保护不变。后端检查分片与事件信号的总量不超过队列容量，已经提交的片段各自保留资源至 GPU 完成，取消不推进外部事件。

## 性能测量与恢复

```sh
$python -m dev.benchmarks.qwen4b_ffn_cache --run-performance \
  --profile "$profile" --output "$experiment_output" \
  --model-root "$model_root" --off-model-root "$off_model_root" \
  > "$experiment_output/progress.log" 2>&1
```

固定条件：Q4H、ANE share=0.5、阈值 512、关闭自动校准、block=8、tail splits=16、并发 1、context=36864、temperature=0、冷前缀、无磁盘缓存、忽略 EOS。3 份固定输入，每份 32,768 token；实际输出为 512 和 2048 token。

每份输入、每种输出长度按 A／B／B／A 运行，每配置 2 次真重复，共 24 批。每完成一个样本就检查已有同配置重复和跨配置输出；第 33 个以后或长上下文才出现的分歧同样立即阻断后续批次。摘要标为 `needs_divergence_analysis`，已测时间只作原始证据保留。

摘要先给每个 prompt 的配对变化，再汇总配对中位数。参考配置重复的离散程度和 5% 筛选结果单列；小幅变化超出噪声前不宣称更快。峰值内存取有效样本的最大值。

每次尝试使用唯一目录。超时、非零退出、外部干扰和 SIGTERM 取消均记录失败，不记为通过。取消只终止本轮子进程组。`process.json` 保存其 PID，顶层 `performance-run.json` 保存当前组和样本。

恢复时使用同一命令并追加 `--resume`。已完成组会重新核对原始证据、身份和校验结果；不完整组保存历史后从 A 开始重跑完整四批。存在未分析的 token 分歧时拒绝恢复。检查点通过同目录临时文件、fsync 和原子替换写入。

本实验的 profile 和 worktree 共用设备互斥锁；已有本实验进程占用时拒绝启动。运行前检查外部推理及 GPU，原生 benchmark 只要进程存在就视为繁忙，不依赖 CPU 占用。常驻服务按请求状态和 CPU 活动判断；活动推理或 GPU 占用高于 50% 时拒绝启动。运行中继续观察外部推理，干扰结果保留为不可用，不停止其他服务。其他测试入口未接入此锁，仍通过进程检查识别。

## 分段诊断

```sh
$python -m dev.benchmarks.qwen4b_ffn_cache --run-stage-profile \
  --profile "$profile" --output "$experiment_output" \
  --model-root "$model_root" --off-model-root "$off_model_root"
```

此入口另跑 A、B 各一个 32K→512 请求。GPU 时间戳在 dispatch 边界采样；只支持 encoder 边界的设备每个 dispatch 单独使用 encoder。采样屏障会扰动执行，数据标为 `diagnostic_only`，不能混入正式速度对照。

记录 ANE 权重 staging、输入 pack、GPU FFN、join、其他 GPU 工作，以及事件等待和 encoder 转换间隙。原始 GPU ticks 保存；估计秒数按每条命令的 GPU 时间跨度缩放。事件间隙包含等待和 encoder 转换，不等于 ANE 私有执行时间。设备不支持或采样失败时报告原因，不伪造分段时间。

启动耗时单列 source weights、FFN cache build、ANE prepare、storage 和 warmup。正式 prefill／decode 时间仍排除这些启动阶段。热路径内存采样不刷新 Metal 设备计数，不在回调中构建报告；采样 CPU 耗时也单列。

## 内存和产物口径

`memory_checkpoints` 保存 ready、prefill complete 和 end 的当前分配。分类之和必须等于 `backend_current_bytes`。

| 字段 | 口径 |
|---|---|
| `target_source_weights_bytes` | 主模型源 image 实际分配，含量化权重、浮点小张量和对齐 |
| `ffn_quantized_payload_bytes` | FFN plane/meta 有效字节，源权重子集，不重复相加 |
| `ffn_fp16_cache_bytes` | 逐缓存 buffer 的实际分配；quantized 必须为 0 |
| `ffn_cached_matrices` | 持有缓存的投影数；full 为 96，limited 按预算选择 |
| `prefill_workspace_bytes`、`decode_workspace_bytes` | 共享工作区 |
| `kv_bytes` | 主模型 KV |
| `draft_weights_bytes` | Drafter 权重；源 BF16，运行时线性投影 AF4G64 |
| `draft_state_bytes` | Drafter rings/context window，`state_bytes` 子集 |
| `state_bytes` | GDN 和 Drafter 状态，含池内及缓存状态 |
| `ane_bytes` | ANE INT8 权重暂存区和 Metal 缓冲 |
| `unclassified_bytes` | Backend 已追踪、上述分类未覆盖的当前分配 |
| `backend_peak_bytes` | 从启动以来的 backend 缓冲分配峰值，含 load/warmup |
| `device_peak_bytes` | Metal 设备报告的本进程峰值 |

Threadgroup tile、ANE 私有运行时和未通过 MetalBackend 创建的时间戳资源不纳入 backend 分类。它不是进程 RSS 或整机统一内存。分类是当前值，不能把各时间点的独立峰值相加。

产物包括 prepared/plan/inputs、operators-attempts ledger、correctness、results ledger、groups 检查点、summary 和独立 stage-profile。原生 stdout/stderr/evidence 都保存在 attempts 下。原权重、旧测试源码和既有原始结果保持完整。
