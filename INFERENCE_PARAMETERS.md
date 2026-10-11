# Qwen 推理参数复用记录

## 适用范围和版本

更新日期：2026-10-11。

这份文档保存我们选定的 Qwen3.5-4B 配置。短输出和长输出使用同一套计算参数，只改变输出预算。
用户决定保留 compact，因此选定配置为 lookup 和 compact 开启，fast top-k 关闭。
这个选择保留可选优化，不表示 compact 已在所有请求中更快。

| 项目 | 本次验证版本 |
| --- | --- |
| 分支 | `codex/qwen4b-quantized-ffn` |
| 提交 | `5ea170cb3304bc1898270e56be3ad19e50c7da81` |
| 验证设备 | Apple M5 Pro |
| 主模型 | `Qwen3.5-4B-UD-Q4_K_XL.gguf`，文本模式 |
| Drafter | Standard DFlash，6 层 |
| 主模型 SHA-256 | `b252c5610a42ca82d20fe2a12813e9d069eed89292907e26c783eeb0bc961bc7` |
| Drafter SHA-256 | `1eb221d36abb13a5f1b972f8d031a9723fad8cbb7d275abe548b60e77577eb42` |

完整正确性和性能证据来自 4B。27B、35B-A3B 目前只核对了 compact 的实现和内存估算，尚未完成这套参数的真实模型对照。
FFN 缓存实验的 `quantized` 模式目前只允许 Qwen3.5-4B。更换模型后，需要重新核对 FFN、ANE 和 drafter 配置。

## 选定参数

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

三个功能开关在引擎启动时读取。修改后需要重启引擎。

```sh
export SPLASH_PROMPT_LOOKUP=1
export SPLASH_COMPACT_DRAFT_HEAD=1
export SPLASH_FAST_TOPK=0
```

compact 还需要 `SPLASH_COMPACT_DRAFT_RANKING`。未设置时，引擎读取模型安装目录下的 `head-ranked.splash`。
排名必须与当前 target manifest 指纹和词表匹配。缺失、格式不符或不支持的请求会回退完整 head。
查看 `compact_cycles`、`compact_rows` 和 `compact_fallback_reason`，确认开关真正执行。

本机复现这次固定输入实验，可以使用原排名：

```sh
export SPLASH_COMPACT_DRAFT_RANKING="$PWD/build/performance-5ea170c-lookup-baseline-v1/input-frequency-ranking.splash"
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

按相同候选容量、Q6_K 输出头估算，额外 Metal 缓冲如下。27B、35B-A3B 的值尚未实测。

| 模型布局 | Head 隐藏维度 | Compact 额外缓冲 |
| --- | --- | --- |
| Qwen3.5-4B | 2560 | 约 199 MiB，已有实测 |
| 项目中的 27B | 5120 | 约 398 MiB |
| 项目中的 35B-A3B | 2048 | 约 159 MiB |

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
本轮没有证明 compact 有稳定的速度收益。若按速度和内存选择短、长输出共用参数，下一轮建议以 `SPLASH_COMPACT_DRAFT_HEAD=0` 为基准，保留 compact 开关。
本页前面的开启配置仍记录此前已选定方案；这里给出新增复测证据和后续调整建议。

32 个样本中没有检测到其他活动推理。启动前 GPU 采样为 0%～57%，按授权接受波动。
固定 AB 顺序、桌面负载和实验排名限制仍适用，不能把这轮结果当作隔离环境下的稳定上限。

完整逐对数据见 [延长 ABAB 报告](docs/qwen4b-abab-2026-10-11.md)，
原始记录见 [summary.json](docs/data/qwen4b-abab-2026-10-11/summary.json)，
运行结束核对见 [final-verification.json](docs/data/qwen4b-abab-2026-10-11/final-verification.json)。

本轮完整原始记录已保存到 [raw-results.tar.gz](docs/data/qwen4b-abab-2026-10-11/raw-results.tar.gz)，随 Git 保存。
[归档清单](docs/data/qwen4b-abab-2026-10-11/archive-manifest.json)记录逐文件 SHA-256，便于核对和恢复。
前几轮的 `build/` 链接仍指向本机历史产物。
