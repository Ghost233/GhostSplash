# Qwen 推理参数复用记录

## 适用范围和版本

更新日期：2026-10-13。

这份文档按 Dense（稠密）和 MoE（混合专家）记录基准。
Dense 分别保存 Qwen3.5-4B 和 Swift 27B 的已验证配置，MoE 以 Tiel MiniPlus 35B-A3B 为依据。
2026-10-12 用户选定 Swift 27B 继续以 Q4H 为基准，完整配置见本页的 Swift 27B 一节。
更换同类模型时，仍需核对布局、drafter、算子支持和实际执行路径。
2026-10-11 用户选定 Tiel MiniPlus 继续以 Q4H 为标准。
Qwen3.5-4B 的短输出和长输出使用同一套计算参数，只改变输出预算。
4B Dense 保留用户此前选定的 lookup 和 compact 开启配置，fast top-k 关闭。
4B Dense 的后续 ABAB 结果未证明 compact 有稳定速度收益，选定值与性能建议分别保留。

## Qwen3.5-4B Dense 与 Tiel MoE 基准总表

| 参数 | Dense 当前选定基准 | MoE 当前正式基准 |
| --- | --- | --- |
| 已验证模型 | Qwen3.5-4B | Tiel MiniPlus 35B-A3B |
| KV 格式 | `q4h` | `q4h` |
| 测试入口的历史 splits | `16` | `8`；本轮候选为 `3` |
| Drafter | Standard DFlash，开启 | DFlash2，开启 |
| Drafter block size | `8` | `8` |
| ANE | 开启 | 关闭 |
| ANE share | `0.5` | 不适用 |
| ANE 最小行数 | `512` | 不适用 |
| ANE 自动校准 | 关闭 | 关闭 |
| FFN 权重路径 | `quantized`，变体 `baseline` | 当前 MoE 量化权重路径 |
| FFN 常驻 FP16 缓存 | 关闭，预算 `0` | 无 Dense FFN 缓存；不套用 4B 专用开关 |
| Prompt lookup | 开启 | 关闭 |
| Compact draft head | 开启，保留此前选择 | 关闭 |
| Fast top-k | 关闭 | 关闭 |
| 默认并发 | `1` | `1`；并发 `2` 为吞吐场景候选 |
| 已验证测试的上下文容量 | `36864` | `131072` |
| 固定测试输入 | `32768` tokens | `32768` tokens |
| 固定测试实际输出 | `512`／`2048` tokens | `16384` tokens |

与上一次正式基准相比，Dense 的选定参数保持一致。
MoE 本轮只提出 `tail_prefill_splits: 8 → 3` 的候选调整。
Lookup、compact 和 ANE 在上一套 MoE 基准中已经关闭，本次保持关闭。
主模型权重、drafter、block、KV 格式、并发和上下文容量保持一致。

Tiel 本轮的 `tail_prefill_splits` 只控制 prefill 最后 32 行的历史分片。
其他 prefill 块和 decode 验证使用各自的规划策略。
这个值不会把全部 32K 输入或全部 decode 都改为同一分片数。
配置入口见 [TielPagedAttention.cpp](build/tiel-miniplus-matrix-v1/TielPagedAttention.cpp)。

MoE 的 63 批扫描已完成，`splits=3` 是总耗时中位数最低的候选。
正式基准继续保留 `8`，等待 `8` 与 `3` 的 12 次 ABBA 配对验证。
候选筛选通过后再决定是否更新正式基准，配对运行状态以日志和真实退出码为准。
扫描结果见 [Q4H 参数扫描报告](docs/tiel-miniplus-q4h-output16k-2026-10-11.md)，
配对进度见 [progress.json](build/tiel-miniplus-q4h-output16k-paired-v4/progress.json)。

| 项目 | 本次验证版本 |
| --- | --- |
| 分支 | `codex/qwen4b-quantized-ffn` |
| 提交 | `5ea170cb3304bc1898270e56be3ad19e50c7da81` |
| 验证设备 | Apple M5 Pro |
| 主模型 | `Qwen3.5-4B-UD-Q4_K_XL.gguf`，文本模式 |
| Drafter | Standard DFlash，6 层 |
| 主模型 SHA-256 | `b252c5610a42ca82d20fe2a12813e9d069eed89292907e26c783eeb0bc961bc7` |
| Drafter SHA-256 | `1eb221d36abb13a5f1b972f8d031a9723fad8cbb7d275abe548b60e77577eb42` |

上表的版本和权重身份适用于 4B。Swift 27B 已完成单请求真实模型对照，参数和证据见下文的独立一节；Tiel MiniPlus 35B-A3B 使用另一套参数。
FFN 缓存实验的 `quantized` 模式目前只允许 Qwen3.5-4B。更换模型后，需要重新核对 FFN、ANE 和 drafter 配置。

## Qwen3.5-4B 选定参数

| 参数 | 选定值 | 含义和依据 |
| --- | --- | --- |
| FFN 权重模式 | `quantized` | 保留量化权重，常驻 FFN FP16 缓存为 0 |
| FFN 算子变体 | `baseline` | 使用通过数值和长轨迹验证的路径 |
| FFN 缓存预算 | `0` | 不创建全量或限额 FP16 副本 |
| KV 格式 | `q4h` | Q4H KV 缓存 |
| ANE | 开启 | 复用时检查实际执行状态 |
| ANE share | `0.5` | ANE 承担的 FFN 中间通道比例 |
| ANE 最小行数 | `512` | 按 prefill 块的行数判断，不是总上下文长度 |
| ANE 自动校准 | 关闭 | 固定份额和阈值，避免比较时自动换策略 |
| Drafter | 开启 | 使用已配对的 Standard DFlash |
| Drafter block size | `8` | 1 个 anchor 和 7 个草稿 token |
| Prefill 历史 splits | `16` | 历史分片数，不是 prefill 块的 token 数 |
| 并发 | `1` | 当前验证和性能对照均为单请求 |
| Prompt lookup | 开启 | 命中历史后提出复制草稿，继续由 target 验证 |
| Compact draft head | 开启 | 用户选定保留；需要匹配模型的排名文件 |
| Fast top-k | 关闭 | 当前统一基准采用 greedy，未获得采样快路径的性能对照 |
| Q4_K group-scale 后移 | 关闭 | 该候选尚未通过长输出轨迹验收 |
| 宽 lookup、draft ahead、streamed submit | 未实施 | 当前没有对应可用开关 |

compact 的容量由代码计算，当前词表下为 98,304 行：98,048 个固定候选和最多 256 个动态候选。
当前没有设置候选容量的环境变量。

## 测试控制项

下面这些值用于复现实验。业务请求仍应按任务设置采样、停止条件和输出上限。

| 控制项 | 固定值 |
| --- | --- |
| 实际输入 | 32,768 tokens，保存实际 token ID 和输入摘要 |
| 实际输出 | 分别生成 512 和 2,048 tokens |
| 上下文容量 | 36,864 tokens |
| 内存预算 | 32 GiB，测试上限，不代表实际分配量 |
| 采样 | `temperature=0`，greedy |
| EOS | `ignore_eos=true`，只为保证测试实际输出长度 |
| 前缀缓存 | `cold_requests=true`，测试请求使用冷前缀 |
| 阶段 profiling | `ffn_stage_profile=false`，与正式速度对照分开 |
| 性能预热 | 2,048 行 prefill、并发 1 的 decode，各 1 次；不计入请求耗时 |

`prefillTokenBudget` 的当前编译值为 2,048。调整 `tail_prefill_splits` 不会把这个预算改成 16。

## 配置入口

配置统一通过启动参数传入，修改后需要重启引擎。旧性能配置环境变量不再读取。

```sh
./splash serve --model OWNER/REPO \
  --prompt-lookup on --compact-draft-head on --fast-topk off
```

compact 还需要 `--compact-draft-ranking FILE`。未设置时，引擎读取模型安装目录下的 `head-ranked.splash`。
排名必须与当前 target manifest 指纹和词表匹配。缺失、格式不符或不支持的请求会回退完整 head。
查看 `compact_cycles`、`compact_rows` 和 `compact_fallback_reason`，确认开关真正执行。

本机复现这次固定输入实验，可以使用原排名：

```sh
./splash serve --model OWNER/REPO --compact-draft-head on \
  --compact-draft-ranking "$PWD/build/performance-5ea170c-lookup-baseline-v1/input-frequency-ranking.splash"
```

这个文件只按输入 token 频率排序，其余 ID 按升序排列，没有使用预期输出。
它用于本次实验，不是生产语料排名，也不能跨模型复用。
生产排名的生成方法见 [可选功能说明](dev/benchmarks/docs/optional-features.md#排名文件)。

矩阵测试入口读取下面的 `job.parameters`。改变输出长度时，只替换 `output_tokens`。

```json
{
  "kv_format": "q4h",
  "ane_requested": true,
  "ane_share": 0.5,
  "ane_minimum_rows": 512,
  "auto_calibration": false,
  "drafter_enabled": true,
  "drafter_block_size": 8,
  "tail_prefill_splits": 16,
  "concurrency": 1,
  "input_tokens": 32768,
  "output_tokens": 512,
  "context_tokens": 36864,
  "memory_budget_bytes": 34359738368,
  "temperature": 0,
  "ignore_eos": true,
  "cold_requests": true,
  "ffn_weight_mode": "quantized",
  "ffn_prefill_variant": "baseline",
  "ffn_cache_budget_bytes": 0,
  "ffn_stage_profile": false
}
```

这是完整 payload 中的参数对象。驱动还要提供 `model_root`、`metallib`、源码与权重身份、实际输入 token ID 和事实断言。
原生入口为 [qwen4b_matrix_native.mm](dev/benchmarks/qwen4b_matrix_native.mm)，正确性驱动为 [qwen_optional_features.py](dev/benchmarks/qwen_optional_features.py)。

普通 `serve-native` 支持功能环境变量、`--kv-format q4h` 和 `--ane on|off`。
固定 FFN 权重策略、ANE share／最小行数和历史 splits 当前由测试入口配置，普通服务 CLI 尚不能完整表达这套配置。
`--decode-share` 设置引擎调度份额，不能用它设置 ANE share。
服务端需显式设置 `RuntimeResourcesConfig.ffnLoading`。
固定 ANE 参数需设置 `RuntimeResourcesConfig.aneFfn`，历史 splits 还要接入对应规划。
配置完成后核对实际路径。
本次只记录选定参数。代码中的三个可选功能默认仍关闭。

## 内存和性能依据

4B 的量化 FFN + lookup 实测 Metal 峰值约 5.477 GiB。开启 compact 后约 5.672 GiB，增加约 199 MiB。
全量 FFN FP16 缓存增加 4.21875 GiB。但本次 ANE 配置下，两种缓存模式的请求 FFN kernel 和提交次数完全相同，半精度缓存计算调用为 0。
因此，当前配置保留 `quantized`。完整缓存的速度差不能解释为缓存算子加速。

compact 缩小草稿输出头的候选词表，target 保留完整词表。选中行保留原量化编码和 scale。
lookup 命中时跳过本轮 drafter，包括 compact head。compact 收益取决于剩余草稿轮数和候选排名。
本轮短输出 compact 的 decode 中位数基本持平，长输出约高 3%；额外内存是否可接受，由使用场景决定。

按相同候选容量、Q6_K 输出头估算，额外 Metal 缓冲如下。Tiel MiniPlus 实测为 159.375 MiB。
Swift 27B 的实际输出头是 Q4_K，实测额外缓冲为 270.375 MiB，不能套用 Q6_K 的估算。

| 模型布局 | Head 隐藏维度 | Compact 额外缓冲 |
| --- | --- | --- |
| Qwen3.5-4B | 2560 | 约 199 MiB，已有实测 |
| 27B、Q6_K 输出头的估算布局 | 5120 | 约 398 MiB，仅为估算 |
| Swift 27B、实际 Q4_K 输出头 | 5120 | 实测 270.375 MiB |
| 项目中的 35B-A3B | 2048 | 约 159 MiB；Tiel MiniPlus 实测 159.375 MiB |

模型文件名中的 4-bit 不保证输出头也为 4-bit。额外缓冲需要按实际 head 格式重算。
这些数值描述额外 Metal 分配，不是进程 RSS 或整机内存。

## 验证证据和再次复用

当前提交的独立准确性复测已通过：96 个真实 FFN 矩阵、480 项数值检查、39 种 compact 格式、采样开关和 19 组真实模型运行。
32K 输入、512／2,048 输出下，各开关与全关逐 token 一致。全量 FP16 缓存与量化模式也逐 token 一致。
实际 ANE、事实、输出长度和内存断言均通过。Q4_K group-scale 候选没有启用。

| 本机证据 | 路径 |
| --- | --- |
| 独立准确性验收 | [report.json](build/accuracy-5ea170c-v1/report.json) |
| 准确性说明 | [accuracy-report.md](build/accuracy-5ea170c-v1/accuracy-report.md) |
| 性能配对报告 | [paired-performance-report.md](build/performance-5ea170c-lookup-baseline-v1/paired-performance-report.md) |
| 缓存使用核对 | [ffn-cache-usage.json](build/performance-5ea170c-lookup-baseline-v1/ffn-cache-usage.json) |
| 原始参数与输入 | [payload.json](build/performance-5ea170c-lookup-baseline-v1/01-512-lookup/payload.json) |

性能测试按 A/B/B/A 运行，每配置 2 次，共 24 组。原基准关闭 compact，开启 compact 的对照对应这里选定的复用配置。
2048 输出的 lookup 关闭组有 3／4 个样本检测到其他活动推理，按用户授权保留为诊断数据。
这些结果不能当作隔离环境下的稳定性能上限。

`build/` 产物不随 Git 提交保存。干净 checkout 上缺少这些文件时，按 [FFN 实验说明](dev/benchmarks/docs/qwen4b-ffn-cache.md)重新准备输入和构建，按 [可选功能说明](dev/benchmarks/docs/optional-features.md)重验。
换源码、权重、设备或排名文件后，先核对身份，再跑算子数值、短输入和 32K 长轨迹检查。
记录首次 token 分歧，保持原校验标准。性能对照使用新的结果目录，避免覆盖历史证据。

## 选定配置的再次性能测量

2026-10-11 按上述配置复测，保持同一提交、权重、输入和 compact 排名。
512 和 2,048 输出各运行 4 次，按长短输出交错，合计 8 次。
全部退出码为 0，token、事实、实际 ANE 和 compact 路径校验通过。

| 实际输出 | Prefill 中位数 | Decode 中位数 | 首 token 中位数 | 总耗时中位数 | Metal 峰值 |
| --- | --- | --- | --- | --- | --- |
| 512 tokens | 2000.5 tokens/s | 126.3 tokens/s | 16.38 s | 20.43 s | 5.672 GiB |
| 2,048 tokens | 2130.5 tokens/s | 249.9 tokens/s | 15.51 s | 23.82 s | 5.672 GiB |

每次启动前最后一次 GPU 采样为 20%～45%，运行中没有检测到其他活动推理。
有 5 次运行满足连续两次 GPU 采样均不高于 50%，其余运行的第一次采样较高。
首轮短输出达到 2487.7／148.4 tokens/s，首轮长输出达到 2489.1／280.1 tokens/s；后续样本变慢。
以上中位数保留全部样本，不用最高值代替。当前记录尚不能确定波动原因。

完整范围、逐次负载和上一轮同配置比较见 [复测报告](build/performance-selected-5ea170c-low-load-v1/selected-performance-report.md)，
机器可读数据见 [selected-performance-report.json](build/performance-selected-5ea170c-low-load-v1/selected-performance-report.json)。

## 延长 ABAB 复测结果

2026-10-11 延长为每个输出长度 8 对 AB，共 32 个样本，执行约 14.7 分钟。
A 开启 compact，B 关闭 compact；lookup 均开启，其余参数保持本页配置。短、长输出交错执行。
所有样本退出码为 0，token、事实、实际 ANE 和内存校验通过。源码、构建产物、target 与 drafter 权重校验一致。
全部保留，没有失败、跳过或因速度波动重跑。测试引擎已经退出。

| 实际输出 | Compact | Prefill 中位数 | Decode 中位数 | 首 token 中位数 | 总耗时中位数 | Metal 峰值 |
| --- | --- | --- | --- | --- | --- | --- |
| 512 tokens | 开启 | 2239.8 tokens/s | 139.3 tokens/s | 14.66 s | 18.32 s | 5.672 GiB |
| 512 tokens | 关闭 | 2289.1 tokens/s | 139.7 tokens/s | 14.32 s | 17.93 s | 5.477 GiB |
| 2,048 tokens | 开启 | 2258.0 tokens/s | 265.1 tokens/s | 14.51 s | 22.24 s | 5.672 GiB |
| 2,048 tokens | 关闭 | 2314.5 tokens/s | 261.7 tokens/s | 14.16 s | 21.99 s | 5.477 GiB |

先计算每个相邻配对的收益，再取中位数：compact 的 decode 收益为短输出 −2.11%、长输出 +0.20%。
短输出仅 3／8 对 decode 更快，长输出 4／8 对更快；额外峰值固定为 199.125 MiB。
两个配置的后半段速度都回升，当前波动幅度大于长输出的配对收益。
本轮没有证明 compact 有稳定的速度收益。若按速度和内存选择短、长输出共用参数，下一轮建议以 `--compact-draft-head off` 为基准，保留 compact 开关。
本页前面的开启配置仍记录此前已选定方案；这里给出新增复测证据和后续调整建议。

32 个样本中没有检测到其他活动推理。启动前 GPU 采样为 0%～57%，按授权接受波动。
固定 AB 顺序、桌面负载和实验排名限制仍适用，不能把这轮结果当作隔离环境下的稳定上限。

完整逐对数据见 [延长 ABAB 报告](docs/qwen4b-abab-2026-10-11.md)，
原始记录见 [summary.json](docs/data/qwen4b-abab-2026-10-11/summary.json)，
运行结束核对见 [final-verification.json](docs/data/qwen4b-abab-2026-10-11/final-verification.json)。

本轮完整原始记录已保存到 [raw-results.tar.gz](docs/data/qwen4b-abab-2026-10-11/raw-results.tar.gz)，随 Git 保存。
[归档清单](docs/data/qwen4b-abab-2026-10-11/archive-manifest.json)记录逐文件 SHA-256，便于核对和恢复。
前几轮的 `build/` 链接仍指向本机历史产物。

## Swift 27B Dense 当前基准

2026-10-12 用户选定这份 Swift 27B 以 Q4H 为基准。后续微调保持其他参数固定，从以下配置开始。
2026-10-13 用户将下一轮精简为 `8192` 实际输入、`4096` 实际输出，只比较全局 prefill 分块 `512`／`2048`／`4096`，drafter 全部开启。
A 保留 `2048` 分块，其余计算参数固定；使用 `ignore_eos=true` 保证实际输出数量。
每组性能重复 3 次，共 9 次。数值、事实、实际参数与完整 token 重复检查并入性能运行，不另设完整输出正确性阶段；数值检查和预热排除在正式请求计时之外。
旧的 drafter 关闭组和 `1024`／`8192` 队列已按用户要求停止，已有 12 次完整正确性和 1 次有效性能记录保留；被取消的尝试不计为通过。
历史 32K／16K 报告和配置保留，本轮 8K／4K 结果单独统计。

主模型为 `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`，实际使用混合量化权重。
文件名中的 IQ3_S 不表示全部张量使用 IQ3_S；主模型为 64 层、隐藏维度 5120，MTP 未参与计算。

| 参数 | 当前基准 | 说明 |
| --- | --- | --- |
| KV 格式 | `q4h` | 用户选定；本轮长输出总耗时最低 |
| Drafter | Qwen3.8-27B DFlash2，5 层，开启 | 固定配套权重 |
| Drafter block size | `8` | 当前实现固定为 1 个 anchor 和 7 个草稿 token |
| 全局 prefill 分块 | `2048` | 用户在 8192 追加测试后再次确认；四种分块均已完成对照 |
| 尾段历史 splits | `16` | 测试入口控制 Swift 最后 32 行，不控制全部 prefill 或 decode |
| ANE | 开启，share=`0.5` | 手动份额，实际执行已核对 |
| ANE 最小行数 | `512` | 按块行数判断 |
| ANE 自动校准 | 关闭 | 避免比较时自动换策略 |
| FFN 实际路径 | 现有量化权重路径 | 常驻 FFN FP16 缓存实测为 0；不套用 4B 专用参数 |
| Prompt lookup | 开启 | 后续需要单独测收益 |
| Compact draft head | 开启 | 实验排名必须匹配当前 target manifest；额外 Metal 缓冲 270.375 MiB |
| Fast top-k | 关闭 | 基准使用 greedy |
| 并发 | `1` | 并发 4 的旧正确性失败仍未修复 |
| 上下文容量 | `131072` | 容量保持不变；本轮 8K 输入＋4K 输出的实际轨迹为 12,288 tokens |
| 本轮测试输入／实际输出 | `8192`／`4096` | 固定实际生成数量，`ignore_eos=true`；历史 16K 记录保留 |
| 测试内存预算 | 40 GiB | 统一上限；历史 16K 输出实测 Metal 峰值为 15.797 GiB |
| 前缀／采样 | 冷请求，`temperature=0` | 固定输入和 token ID |

以下历史三轮数据使用 32K 输入、16K 实际输出：prefill 476.3 tokens/s、decode 78.6 tokens/s、首 token 68.80 秒、总耗时 277.27 秒。
耗时排除模型加载和预热。15 组常规测量及 2 组补测的有效请求退出码均为 0。
后台重叠的 BF16 原始尝试已排除并重跑；来源未证实的慢样本保留在三轮统计，补测单列。
性能和数值结论只覆盖 C1 与本轮固定输入，不推广到并发 4 或其他任务。

完整机器可读基准见 [selected-baseline.json](docs/data/swift27b-kv-2026-10-12/selected-baseline.json)。
原始权重 SHA-256、源码与产物身份、排名身份均在记录中。
排名按固定实验输入频率生成，不是通用生产排名；原文件及相关证据已归档。
五格式对照、干扰审计和补测见 [Swift 27B 性能报告](docs/swift27b-kv-2026-10-12.md)。

普通启动器 [swift27bdense.command](swift27bdense.command) 已默认选择 Q4H。
上表的计算参数来自已验证基准；8K 输入、4K 实际输出的四种分块对照已完成。普通服务的自动 ANE 策略和容量由各自入口设置。
当前普通启动器的容量为 262,144，测试容量为 131,072；固定 share／阈值和尾段 splits 仍需对应测试入口。
开启 compact 还需要匹配的排名文件，并核对实际执行与回退计数。

### 后续参数测试顺序

本轮固定 Q4H、8K 输入、4K 实际输出、131,072 容量和并发 1，只改变全局 prefill 分块。
ANE／lookup／compact 与 share／splits 对照已完成，用户保留 A。本轮不重扫这些参数，也不测 drafter 关闭、1024 或 8192 分块。

| 配置 | 全局 prefill 分块 | Drafter | 性能次数 |
| --- | --- | --- | --- |
| A，当前计算基准 | `2048` | DFlash2 开启，block=`8` | 3 |
| P512 | `512` | 同 A | 3 |
| P4096 | `4096` | 同 A | 3 |

运行顺序为 A→P512→P4096、P4096→P512→A、P512→A→P4096。每次运行携带检查，同配置后两次与首个有效样本比较全部 4096 个输出 token；跨分块首次分歧保留。
本轮有效完成9／9次，三个分块各3轮。4096的原ANE编译失败已修复：内部函数网格从128行间隔的29函数缩减为256行间隔的15函数，实际分块、share和阈值不变；2048及以下仍使用128行间隔。旧失败样本保留并排除，完整输出重复和原数值标准均通过。
结果见 [8K／4K分块报告](docs/swift27b-q4h-prefill-8k4k-2026-10-13.md)，归档位于 `docs/data/swift27b-q4h-prefill-8k4k-2026-10-13/`。A2048仍保留为选定基准，没有自动切换参数。
2026-10-13 追加全局分块 `8192` 的 3 轮测试已完成，其他参数和 8K／4K 工作负载不变，真实分块为 `[8160, 32]`。四种分块合计 12 次有效运行；新增对照见 [8192 分块报告](docs/swift27b-q4h-prefill8192-2026-10-13.md)。A2048 仍保留，未自动切换基准。
2026-10-13 用户在追加 8192 对照后再次确认全局分块仍为 `2048`，其他基准参数保持原值。

### Decode 参数微调，已完成

优先比较 decode 相关参数，每次只改变一项。两个启动覆盖参数已实现并完成对照，默认行为和功能开关保持原样。

| 优先级 | 参数 | 候选值 | 当前状态与验收重点 |
| --- | --- | --- | --- |
| 高 | Lookup 最小匹配长度 | `2`／`3`／`4`，上限固定 `4` | 参数入口已实现，五组实验已完成；当前默认下限 2、上限 4，L3 为配对复核候选 |
| 高 | Compact 候选容量 | `98304`／`65536`／`32768` | 容量入口已实现，三种容量对照完成；默认 98304，排名与原回退规则不变 |
| 中 | ANE share 细调 | 名义 `0.4`／`0.5`／`0.6` | 记录实际 14／34、17／34、20／34 通道份额；0.25 和 0.75 已测，不重复粗扫 |
| 低 | 尾段 splits 细调 | `12`／`14`／`16` | 1／2／4／8／16 已测；只影响最后 32 行，不预期大幅改变整体 prefill |
| 后续 | 并发 | `1`／`2` | 区分单请求延迟与总吞吐；lookup／compact 的多 lane 回退必须记录；并发 4 等原 B4 问题修复后再测 |
| 单独覆盖 | 输入内容与长度 | 非重复总结／代码；输入 8K／32K，输出仍固定 4K | 现有样本 lookup 接受约 79% 的输出，需要验证参数能否跨任务复用 |

已启动的五组实验为：A（lookup 下限 2、compact 98304）、L3（下限 3）、L4（下限 4）、C64（compact 65536）、C32（compact 32768）。每组 3 轮，共 15 次性能运行，校验并入运行，保持原标准；不展开完整笛卡尔组合。用户已批准这 5 组并开始执行。默认值与独立功能开关保持原样；当前启动参数为 `--prompt-lookup-min-match`（默认 2，支持 2／3／4）和 `--compact-draft-capacity`（默认 98304，支持 512～98304 的 256 倍数）。无效参数会在启动时拒绝；功能关闭时不执行对应路径。实际生效值与 compact 回退次数会写入每轮记录。

ANE 阈值暂不优先扫描：在当前 8K 输入与 2048 分块下，实际大块为 2048／2048／2048／2016，阈值 512／1024／1536 走相同路径；2048 才会改变 2016 行块的执行。阈值调优需另配较短或不整齐的输入。
DFlash2 block 当前固定 8；本轮不重新加入用户已排除的 drafter 关闭和 1024 分块。

### Swift 27B 启动参数

`swift1527biq3ssplash.sh` 与 `swift27bdense.command` 使用同一入口。默认保留上下文 262144、Q4H、队列 1、内存预算 40G、磁盘缓存 50G、lookup 开启、compact 默认关闭、fast-topk 关闭。启动器默认选择 `build/compact-public-20261014/splash` 参数版独立构建，保留旧的 `build/splash`；可用 `--binary` 选择其他匹配当前参数接口的构建。`--help` 列出完整参数，命令行设置覆盖启动器默认值。

| 配置 | 启动参数 |
|---|---|
| 上下文、内存、缓存 | `--max-context 262144 --max-memory 40G --max-cache-disk 50G` |
| 请求队列与 KV | `--queue-size 1 --kv-format q4h` |
| 原生构建 | `--binary /absolute/path/splash` |
| ANE 比例和阈值 | `--ane-ffn-share auto`，或 `--ane-ffn-share 0.5 --ane-ffn-minimum-rows 512` |
| ANE 开关 | `--disable-ane`，或 `--ane-ffn-share 0` |
| Lookup | `--prompt-lookup on\|off --prompt-lookup-min-match 2\|3\|4` |
| Compact | `--compact-draft-head on\|off --compact-draft-capacity 98304 --compact-draft-ranking FILE` |
| Fast top-k | `--fast-topk on\|off` |

这些性能配置不再读取 `SPLASH_*` 环境变量，原生引擎也不提供环境变量回退。测试工具通过输入 JSON 显式传递功能设置，历史结果文件保持原样。普通 `splash serve` 的 lookup、compact、fast-topk 默认关闭；Swift 基准启动器显式开启前两项。旧入口 `swift1527bsplash.sh` 保留原先 INT8 和队列 4 的默认值，也接受同一组启动参数。

### Compact 公共排名加载（2026-10-14）

服务及相关启动器默认 `--compact-draft-head off`。开启必须同时指定 `--compact-draft-ranking 文件`。
当前公共文件为 `compact-rankings/qwen35-248320-dev-seed.splash`，不绑定主模型权重指纹。
旧 Saluki 排名与生成报告保留；旧格式须一次性离线转换。
文件、布局和投影数值错误拒绝启用，覆盖率不足仍正常回退。此变更不修改其他性能参数。
历史 benchmark 的 compact 开启结果是当时的实验配置，不代表当前服务默认值。
具体规则与正确性证据见 [公共排名加载说明](docs/compact-public-ranking-2026-10-14.md)。

### ANE 低功耗恢复（可选，默认关闭）

2026-10-13 新增 `--ane-power-recovery on`。这个功能保持 ANE 比例、最低行数和全局分块参数不变。
自动模式沿用原有性能熔断；固定比例模式启用本开关后，会测量纯 GPU 基准并加入相同的 8 命令性能熔断，不搜索新比例。

| 参数或条件 | 当前值 | 说明 |
|---|---|---|
| 开关 | 默认 `off` | `--ane-power-recovery on` 才启用 |
| 采样周期 | 1 秒 | 性能退避期间直接读取 IOReport ANE 能量计数，不依赖 `sudo` |
| 首次冷却 | 60 秒 | 冷却和低功耗窗口都满足后才允许探测 |
| 连续低功耗窗口 | 10 秒 | 读数缺失、过期、计数重置或异常值会打断窗口 |
| 低功耗门限 | 0.05 W | 初始候选值；可用 `--ane-recovery-idle-watts WATTS` 修改，不能当作占用百分比 |
| 探测时机 | 下一次满足 ANE 行数阈值的 prefill | 不在 decode 中定时插入探测 |
| 恢复条件 | 原数值校验通过，当前 split 比纯 GPU 至少快 5% | 使用原比例验证，不重新搜索 share |
| 无收益后的冷却 | 120、240、300 秒 | 连续失败延长间隔；恢复后连续 8 命令未再熔断，回到 60 秒 |
| 执行、超时或数值故障 | 保持停用 | 低功耗无法证明故障恢复，这类故障不自动重试 |

采样使用整机功耗，只决定何时允许探测。有效的 0 W 与未知读数分别处理；未知不会触发恢复。
2026-10-13 被动计数验证发现 `Fast-Die CE / ANE0` 的 21 个百分比档位，但已知 ANE 活动时档位计数仍未更新，因此当前不提供可信的计算占用百分比。短正确性用例观测到 2.59 W，尚未进行饱和功耗测量；不能用这个峰值或工具的固定 8 W 作为本机最大功耗。
旧 ANE 任务全部报告、上一命令完成后，恢复才创建新的共享事件，旧回调不能影响新任务。
`/status` 的 `ane_ffn.recovery` 会报告阶段、功耗、冷却余量、连续低功耗时长、探测次数和成功恢复次数。

早期验证构建位于 `build/ane-power-recovery-20261013/`。新的参数版构建位于 `build/ane-startup-cli-20261013/`；使用 `--binary` 指定其中 `splash` 的绝对路径。服务已按用户要求停止，不会自动启动。
开关关闭时，手动模式仍不安装性能熔断器，原有自动模式和异常保护保持原行为。

### 启动时 ANE 参考功耗（手动或自动，默认关闭）

`--ane-power-reference` 控制参考值，与 ANE share 的选择分开。

| 设置 | 行为 |
|---|---|
| 未设置或 `off` | 不测量、不设置功耗参考 |
| `auto` | 启动时等待约 2.2 秒采样预热和后台检查，再连续运行约 5 秒当前配置的 FFN |
| 正的瓦数，例如 `6.0` | 手动使用 6 W，跳过高负载测量 |

自动模式保留当前 share 和 2048 行分块，不编译另一份“全芯片饱和”程序。
记录的是整机 ANE 的 1 秒平均功耗，包含本次窗口的峰值、稳定值（中位数）、后台功耗、有效样本数和实际负载时长。
这个参考与模型和配置有关，不能当作硬件绝对最大功耗。每次自动启动都会重新测量；手动设置不进行测量。
后台不安静、有效完整样本少于 3 个或高负载功耗没有明显上升时，自动参考标记为无效。

开启恢复后，可另设 `--ane-recovery-max-power-percent 50`，将低负载门限设为有效参考的 50%。
例如手动参考 6 W 时，门限为 3 W。这个比例代表相对参考功耗，不代表计算占用率。
不设置百分比时，仍用绝对瓦数门限；同时设置百分比和绝对瓦数时，百分比优先。
请求了百分比但参考检测失败时，不允许自动恢复，不把未知当成 0%。
恢复仍要求原数值校验和至少 5% 的当前速度收益，保持 60 秒冷却和连续 10 秒低负载窗口。

功耗参考的早期验证构建为 `build/ane-power-reference-20261013/`。新的参数版使用以下命令；它展示如何启动，不表示服务已经启动：

```sh
sh swift1527biq3ssplash.sh \
  --binary "$PWD/build/ane-startup-cli-20261013/splash" \
  --ane-ffn-share 0.5 --ane-ffn-minimum-rows 512 \
  --ane-power-reference auto --ane-power-recovery on \
  --ane-recovery-max-power-percent 50
```

手动方式将 `--ane-power-reference auto` 改为 `--ane-power-reference 6.0`。不设置 `--ane-ffn-share` 时仍自动校准比例；设置 `0` 时仅使用 GPU。最低行数必须与固定比例一起指定，且不能超过所选构建的全局 prefill 预算。
`/status` 的 `ane_ffn.power_reference` 保存模式、参考值及测量数据；`ane_ffn.recovery.relative_power_percent` 报告相对功耗，可能超过 100%。
服务已停止，参数转换完成后仍保持停止。新增恢复和功耗参考开关默认关闭。

### 历史完整参数计划

以下表保留为历史记录，当前候选以本节上面的清单为准。
原始慢样本、被拒绝或取消的尝试及首次 token 分歧都保留，数值标准不变。

| 优先级 | 参数 | 建议候选 | 当前条件或限制 |
| --- | --- | --- | --- |
| 1 | ANE 开关 | 关闭、开启且 share=`0.5` | 已支持；先确认其 prefill 收益 |
| 1 | Lookup／compact | 开／开、开／关、关／开、关／关 | 已支持；查清两个功能的独立收益和相互影响 |
| 2 | ANE share | 名义 `0.25`、`0.5`、`0.75` | 底层支持，测试入口需扩展；记录并核对实际通道份额 |
| 2 | 尾段 splits | `1`、`2`、`4`、`8`、`16` | 已支持；只改变最后 32 行的归约，整体 prefill 收益需实测 |
| 3 | 全局 prefill 块预算 | `512`、`1024`、`2048` | 现值 2048；需同步 host、shader、工作区、ANE 预算和预热验收并重新构建 |
| 3 | Drafter 开关 | 关闭、开启 DFlash2 block=`8` | Swift 原生测试入口目前要求 DFlash2，NoDraft 对照需先适配 |
| 后续 | ANE 最小行数 | `512`、`1024`、`1536`、`2048` | 当前 32K 大块均为 2048 行，改阈值可能不改变路径；需搭配尾块测试 |
| 后续 | 工作负载覆盖 | 输入 8K／32K／64K，实际输出 512／2048／16384 | 用于确认参数能否跨任务复用；与单项调优分开 |

第一轮建议只测 5 种独立配置：当前基准、ANE 关闭、仅 lookup、仅 compact、两个功能均关闭。
它们保持 DFlash2、splits、KV 和输入输出一致，共 5 组；每组重复 3 次是 15 次测量。
再按结果进入 ANE share 与 splits 扫描，避免一次展开全部组合。

DFlash2 当前只支持 block=8，不能直接套用 4B Standard DFlash 的 2／4／16 扫描。
ANE 当前最小支持 512 行；share 与最小行数在现有测试入口钉为 0.5／512，其他值需要先扩展入口校验。
Compact 容量当前由代码计算，没有容量开关；候选容量和排名可在功能对照之后单独研究。
并发 2／4 应先补正确性，尤其先处理已知 B4 失败。
Fast top-k 应放到采样任务中比较，greedy 基准不会执行该采样快路径。
上述完整候选表保留为历史计划。本轮按前面的 3 组精简表执行，测试产物使用独立构建，生产引擎默认值保持原有配置。

## Tiel MiniPlus 35B-A3B 当前标准参数

2026-10-11 用户决定继续以 Q4H 作为标准。后续复用和参数微调从下面这套配置开始。
INT8、BF16 和另外两种 Q4 格式保留为对照，历史测量与当时的建议保留。
本次更新记录选定配置，服务启动参数和引擎默认值按各自入口设置。

| 参数 | 当前标准值 | 说明 |
| --- | --- | --- |
| 主模型 | Tiel-Coder-35B-A3B.APEX-I-MiniPlus-V2.1.gguf | 保留现有 GGUF 权重量化 |
| KV 格式 | `q4h` | 用户选定的 Q4H 标准 |
| 上下文容量 | `131072` | 最大容量为 131,072 tokens |
| 尾段历史 splits | `8` | 当前正式基准；仅用于 Tiel 最后 32 行，候选 `3` 待配对验证 |
| Drafter | DFlash2 开启 | 使用现有配套模型 |
| Drafter block size | `8` | 1 个 anchor 和 7 个草稿 token |
| 并发 | `1` | 单请求基准 |
| ANE | 关闭 | 当前 MoE 路径使用 GPU |
| Prompt lookup | 关闭 | 复用最后一轮统一配置 |
| Compact draft head | 关闭 | 保留功能开关 |
| Fast top-k | 关闭 | 本轮使用 greedy |
| FFN 权重路径 | 当前 MoE 路径 | 4B 专用 FFN 缓存参数不适用于这里 |

下面的 `job.parameters` 用于复现最新的固定长度实验。业务请求按需要设置输出预算和停止条件。
`ignore_eos=true` 只用于固定实际输出长度的测量。

```json
{
  "kv_format": "q4h",
  "ane_requested": false,
  "ane_share": 0,
  "ane_minimum_rows": 0,
  "auto_calibration": false,
  "drafter_enabled": true,
  "drafter_block_size": 8,
  "tail_prefill_splits": 8,
  "concurrency": 1,
  "input_tokens": 32768,
  "output_tokens": 16384,
  "context_tokens": 131072,
  "memory_budget_bytes": 34359738368,
  "temperature": 0,
  "ignore_eos": true,
  "cold_requests": true
}
```

功能开关在引擎启动时读取，统一标准使用下面的设置：

```sh
./splash serve --model OWNER/REPO \
  --prompt-lookup off --compact-draft-head off --fast-topk off
```

2026-10-11 同参数复测的三轮中位数为 prefill 2074.7 tokens/s、decode 185.5 tokens/s、首 token 15.79 秒、总耗时 104.13 秒，device 峰值 15.612 GiB。
这组历史数据对应 32,768 输入、16,384 实际输出和 131,072 容量。
更早的 512／2,048 输出性能记录使用 36,864 容量，复用数据时核对对应控制项。
历史证据见 [16K 同参数复测报告](docs/tiel-miniplus-output16k-repeat-2026-10-11.md)。

2026-10-12 完成 63 批、75 个请求的 Q4H 参数扫描。
下表保留基准和候选的三轮中位数，尚未用配对结果确认候选收益。

| 配置 | Prefill tokens/s | Decode tokens/s | 首 token s | 总耗时 s | Device 峰值 GiB |
| --- | --- | --- | --- | --- | --- |
| 正式基准 `splits=8` | 2313.6 | 219.5 | 14.16 | 88.07 | 15.612 |
| 本轮候选 `splits=3` | 2384.7 | 240.3 | 13.74 | 81.92 | 15.733 |
| 备选 `splits=11` | 2283.1 | 236.0 | 14.35 | 83.20 | 15.612 |

扫描中 `splits=3` 的总耗时中位数约低 7%，device 峰值约高 123.75 MiB。
分片会改变浮点归约顺序和后续草稿接受率，速度差异不能全部解释为 attention 算子收益。
本轮输入及后台负载限制仍适用，完整证据见 [Q4H 参数扫描报告](docs/tiel-miniplus-q4h-output16k-2026-10-11.md)。

下面各节记录历史测量和当时的建议。当前复用标准以本节的 Q4H 配置为准。

## Tiel MiniPlus 35B-A3B 首轮独立对照

2026-10-11 完成重新测试，共 236 次测量和 10 次初始正确性运行。包括 6 次 ANE 请求开启后的回退诊断。所有原生运行退出码为 0，完整范围和原始证据见 [Tiel 性能报告](docs/tiel-miniplus-performance-2026-10-11.md)。

当时建议 Tiel 共用基准为 INT8 KV、历史 splits=8、DFlash2 开启且 block=8、并发 1、ANE 关闭、lookup／compact／fast top-k 关闭。保持原 MoE 权重路径，不使用 4B 专用 FFN 缓存参数。此建议没有自动修改生产服务。

最终每种输出 8 对 AB 复核，仅改变 lookup。长输出可单独开启 lookup；短输出保持关闭。下表为各组中位数。

| 实际输出 | Lookup | Prefill tokens/s | Decode tokens/s | 首 token s | 总耗时 s | Metal 峰值 GiB |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 关闭 | 1768.8 | 120.4 | 18.53 | 22.77 | 15.761 |
| 512 | 开启 | 1772.9 | 113.3 | 18.48 | 23.06 | 15.761 |
| 2,048 | 关闭 | 1733.4 | 164.1 | 18.91 | 30.98 | 15.761 |
| 2,048 | 开启 | 1793.1 | 174.8 | 18.27 | 30.26 | 15.761 |

Lookup 的配对 decode 收益中位数为短输出 −5.26%、长输出 +6.37%；分别 0／8 和 8／8 对更快。两组输出逐 token 一致。Metal 峰值列来自 native backend，独立 device 内存账本均为 15.767 GiB。

Tiel 当前实现只支持 DFlash2 block=8，MoE 未启用实际 ANE。主模型权重量化文件固定，扫描的五种格式是 KV 格式。逐项 splits 1～16、并发 1／2／4 和八个交叉组合的完整表格已保存在报告。后台波动按授权保留，不能把这里的中位数当作隔离环境的稳定上限。

## Tiel 的 INT8 与 Q4H 独立对照

2026-10-11 单独复测两种 KV。固定上一节的 INT8 基准参数，只改变 KV，lookup 和 compact 均关闭。每种输出 8 对，共 32 次，交替 AB／BA，实际运行约 19.7 分钟。全部退出码、事实、输出长度和同 KV 轨迹检查通过。

| 实际输出 | KV | Prefill tokens/s | Decode tokens/s | 首 token s | 总耗时 s | Backend Metal 峰值 GiB |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | INT8 | 1807.2 | 122.8 | 18.13 | 22.27 | 15.761 |
| 512 | Q4H | 1848.6 | 126.4 | 17.73 | 21.77 | 15.606 |
| 2,048 | INT8 | 1843.9 | 169.2 | 17.78 | 29.88 | 15.761 |
| 2,048 | Q4H | 1863.8 | 115.9 | 17.58 | 35.33 | 15.606 |

按相邻配对计算，Q4H 的短输出 decode 提升 1.81%，总耗时降低 1.86%；长输出 decode 下降 32.10%，总耗时增加 17.67%。长输出 8／8 对都是 INT8 更快。Q4H 节省 158.75 MiB，device 账本峰值为 INT8 15.767 GiB、Q4H 15.612 GiB。

当时建议共用基准继续使用 INT8，并保留短输出的 Q4H 候选。两种格式的 prefill 基本接近，不能沿用上一轮交叉验证中的较大 prefill 差距。长输出草稿接受率分别为 73.84% 和 44.58%，不同输出轨迹会影响 decode。

完整逐对统计、速度范围、顺序分组和原始证据见 [独立对照报告](docs/tiel-miniplus-int8-q4h-2026-10-11.md)。所有波动样本保留，测试进程已退出，结束时源码、构建产物和原始权重 SHA-256 一致。

## Tiel 五种 KV 格式的三轮复测

2026-10-11 对 INT8、BF16、Q4_0、Q4_0 unrotated 和 Q4H 各复测 3 轮，每种实际输出 512／2,048 tokens，共 30 次，约 21.5 分钟。只改变 KV，其他参数保持 INT8 共同基准。全部真实退出码、事实、输出长度、同 KV 冻结轨迹和内存账本校验通过。

| 输出 | KV | Prefill tokens/s | Decode tokens/s | 首 token s | 总耗时 s | Device 峰值 GiB |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | INT8 | 2047.4 | 134.1 | 16.00 | 19.94 | 15.767 |
| 512 | BF16 | 2149.7 | 130.9 | 15.24 | 19.16 | 16.041 |
| 512 | Q4_0 | 1541.3 | 108.6 | 21.26 | 25.97 | 15.656 |
| 512 | Q4_0 unrotated | 1483.6 | 119.8 | 22.09 | 26.36 | 15.656 |
| 512 | Q4H | 1976.5 | 139.7 | 16.58 | 20.44 | 15.612 |
| 2,048 | INT8 | 1988.4 | 185.1 | 16.48 | 27.75 | 15.767 |
| 2,048 | BF16 | 2121.0 | 165.0 | 15.45 | 28.37 | 16.041 |
| 2,048 | Q4_0 | 1347.0 | 138.0 | 24.33 | 39.17 | 15.656 |
| 2,048 | Q4_0 unrotated | 1561.7 | 135.6 | 20.98 | 36.08 | 15.656 |
| 2,048 | Q4H | 2024.1 | 125.5 | 16.19 | 32.50 | 15.612 |

表中为三轮中位数。BF16 的短输出总耗时最低，INT8 的长输出总耗时和 decode 最好；三种 Q4 中，Q4H 总耗时最低。BF16 与 INT8 的逐轮小幅差异方向不稳定，当时建议共同基准保留 INT8，BF16 与 Q4H 作为备选。

30 个样本中有 3 个检测到后台服务忙，按授权保留。第一轮 Q4H 短输出预热约 165 秒，且做过 1 秒栈采样，潜在时序影响单独标记。没有重跑、丢弃样本或放宽校验。

完整三轮逐次数据、范围、同轮对照、接受率和诊断记录见 [五格式三轮报告](docs/tiel-miniplus-kv5-2026-10-11.md)。源码、构建产物和原始权重哈希一致，测试引擎已经退出。

## Tiel 的 128K 容量与 16K 输出对照

2026-10-11 追加长输出测试。上下文容量固定为 131,072 tokens，实际输入 32,768、实际输出 16,384，实际轨迹为 49,152 tokens。每种 KV 重复 3 轮，共 9 次，测量约 22.1 分钟。这个上限不代表本轮输入达到 128K。

固定 splits=8、DFlash2 block=8、并发 1，ANE／lookup／compact／fast top-k 关闭。greedy、ignore_eos=true、冷请求，内存预算保持 32 GiB。只扩展独立测试入口并加入每 2048 tokens 的进度日志，生产引擎和算子不变。

| KV | Prefill tokens/s | Decode tokens/s | 首 token s | 总耗时 s | Device 峰值 GiB | 草稿接受率 |
| --- | --- | --- | --- | --- | --- | --- |
| INT8 | 1857.2 | 197.5 | 17.64 | 100.62 | 15.926 | 95.91% |
| BF16 | 1930.6 | 187.5 | 16.97 | 104.70 | 16.339 | 92.15% |
| Q4H | 1679.2 | 187.5 | 19.51 | 105.13 | 15.612 | 87.74% |

表中为三轮中位数。9 个样本中有 5 个检测到后台服务忙，全部保留，没有重跑或剔除。INT8 三轮均比 Q4H decode 更快，与 BF16 对比则有 2／3 轮更快。当时建议共用基准暂时保留 INT8；这些结果属于带负载的诊断性测量，不能确定隔离条件下的稳定排名。

Q4H 相比 INT8 少 321.25 MiB，BF16 多 422.5 MiB。实际峰值与容量上限、实际使用长度是不同指标。全部长度、事实、内存账本、同 KV 16K 完整重复轨迹和旧 2048 token 前缀校验通过。9 次 EOS 数均为 0；生成内容可能重复，不能把这些速度推广到其他任务。

完整逐轮数据、启动／预热时间、源代码与构建身份及原始记录见 [16K 输出报告](docs/tiel-miniplus-output16k-2026-10-11.md)。结束时源码、构建产物和原始权重哈希一致，测试引擎已退出。

## Tiel 的 16K 输出同参数复测

2026-10-11 按上一节相同参数再测 3 轮，共 9 次，约 16.8 分钟。131,072 容量、32,768 输入、16,384 实际输出、splits=8、DFlash2 block=8、并发 1 和各功能开关均不变。原生入口字节一致，全部 16K 输出与上轮同 KV 逐 token 一致。

| KV | Prefill tokens/s | Decode tokens/s | 首 token s | 总耗时 s | Device 峰值 GiB |
| --- | --- | --- | --- | --- | --- |
| INT8 | 2087.9 | 200.1 | 15.69 | 98.25 | 15.926 |
| BF16 | 2077.9 | 185.6 | 15.77 | 104.26 | 16.339 |
| Q4H | 2074.7 | 185.5 | 15.79 | 104.13 | 15.612 |

Prefill 中位数相差约 0.64%，decode 中位数仍为 INT8 最快，BF16 与 Q4H 接近。INT8 三轮均比 BF16 decode 更快，比 Q4H 有 2／3 轮更快。当时建议继续以 INT8 为共同基准，保留其他格式开关。

3／9 个样本记录到后台服务忙，全部保留，没有重跑或剔除。三种格式的内存峰值与上轮完全相同；Q4H 比 INT8 少 321.25 MiB，BF16 多 422.5 MiB。两轮速度范围与中位数对照见 [16K 同参数复测报告](docs/tiel-miniplus-output16k-repeat-2026-10-11.md)。

原来的 16K 报告和原始结果保留。新结果单独保存于 build/tiel-miniplus-output16k-v2，旧结果不覆盖。真实退出码、事实、长度、完整重复轨迹、2048 前缀与内存账本全部通过，源码、构建与权重身份不变，测试进程已退出。

当前五组实验的实时日志位于 `build/swift27b-q4h-decode-tuning-v3/controller.log`，数值检查并入性能运行；生产 AIR 保持原样，测试库仅附加原字节拷贝检查需要的测试函数。

五组 decode 参数微调已有效完成15／15次，原标准与默认A完整输出保持一致。结果见 [微调报告](docs/swift27b-q4h-decode-tuning-2026-10-13.md)；两个启动覆盖参数保留，基准尚未自动变更。

## 启动参数与实际状态快速检查

2026-10-14 新增启动清单。服务会先列出全部配置参数，在 `Ready` 后列出实际状态；配置开启、校验通过、正常回退、条件不可用和错误分别报告。API key 隐去内容。状态无法读取或旧构建缺少字段时显示 `unknown`。

MiniPlus V3 的两个启动器和 Swift 1.5 启动器已指向独立构建 `build/startup-checklist-20261014/splash`，其他参数保持不变。启动器仍接受命令行参数覆盖，没有新增环境变量。控制台清单是启动时快照，运行中的 ANE 回退或恢复继续通过原有日志和 `/status` 查看。

说明和调整示例见 [启动检查文档](docs/startup-checklist-2026-10-14.md)。


## VAL APEX I MiniPlus V3 启动器

2026-10-14 新增 `qwen38valapexminiplusv3splash.sh`，加载 `/Users/ghost233/.lmstudio/models/IsValorum/Qwen3.8-27B-VAL-APEX-I-MiniPlus-V3-GGUF/Qwen3.8-27B-VAL-APEX-I-MiniPlus-V3.gguf`。默认端口 8009、上下文 262144、Q4H、队列 1、内存 40G、磁盘缓存 50G、ANE 自动探测、lookup 开启、compact 和 fast top-k 关闭。

新目录没有 drafter，复用已转换的 5 层 Qwen3.8-27B DFlash2，block 固定为 8。结构及完整 CPU 布局规划通过，未测量新模型的接受率或速度。`qwen38miniplusv3splash.sh` 已改为当前 VAL APEX I MiniPlus V3，默认端口 8008；原 EfficientThink 脚本仅作为历史内容保存。Saluki 启动器已删除，历史测试和公共排名保留。

命令及验证记录见 [新 MiniPlus 启动器说明](docs/miniplus-val-apex-launcher-2026-10-14.md)。

2026-10-14 入口修正：继续使用 `sh qwen38miniplusv3splash.sh` 即可加载上述新模型。启动器额外打印主 GGUF 的完整路径，避免名称混淆。8008 旧服务与引擎已退出，本轮没有自动重启。上下文、KV、drafter、ANE 与其他开关不变。
