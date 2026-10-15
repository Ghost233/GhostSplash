# Kev-0.8B、Julia-1 和 Laya 的原生接入

## 启动与接口

这三份模型通过 `/v1/systemone` 返回候选项概率。实际输出 token 数为 0。聊天、文本续写和 drafter 不参与这条计算路径。

在本实验 worktree 中运行：

```sh
cd /Users/ghost233/.codex/worktrees/qwen4b-quantized-ffn/GhostSplash
sh kev08bsplash.sh
sh julia1splash.sh
sh layasplash.sh
```

按需要单独启动。每个启动器持有独立的服务和模型进程。

| 模型 | 文件 | 默认地址 |
|---|---|---|
| Kev-0.8B | `ggml-org/Kev-0.8B-GGUF/Kev-0.8B-Q8_0.gguf` | `http://127.0.0.1:8013/v1/systemone` |
| Julia-1 | `ggml-org/Julia-1-GGUF/Julia-1-Q8_0.gguf` | `http://127.0.0.1:8014/v1/systemone` |
| Laya | `ggml-org/Laya-GGUF/Laya-Q8_0.gguf` | `http://127.0.0.1:8015/v1/systemone` |

路径前缀为 `/Users/ghost233/.lmstudio/models/`。原始 GGUF 保留，启动时从文件读取元数据和权重。

| 参数 | Kev-0.8B | Julia-1／Laya |
|---|---|---|
| 默认上下文 | 8192 tokens | 8192 tokens |
| 默认 Metal 内存上限 | 4 GiB | 4 GiB |
| 默认请求队列上限 | 1 | 1 |
| KV | BF16 | 无因果 KV |
| ANE | 默认关闭，可显式请求原有 FFN 分担路径 | 关闭，当前没有 GeGLU ANE 程序 |
| Drafter／lookup／compact／fast-topk | 关闭 | 关闭 |
| FFN 常驻 FP16 缓存 | 原有缓存开启，528,482,304 bytes，约 504 MiB | 0 bytes |
| 最后一层只计算选项 query | 不适用 | 默认开启 |
| 每次提交可合并的短请求数 | 原有 Qwen 调度 | 默认最多 4 个 |

公共开关使用启动参数。Julia 和 Laya 的两个计算开关可以单独调整：

```sh
sh julia1splash.sh --encoder-option-queries off
sh layasplash.sh --queue-size 4 --encoder-submit-batch-size 1
```

`--encoder-submit-batch-size` 接受 1、2、4。1 表示每个请求单独提交。批处理保持各个请求的输入、选项位置和输出独立。多个短请求依次放入一个 Metal command，共用临时工作区。长请求仍单独提交。实际合并数量在 `/status` 的 `telemetry.maximum_submission_batch` 中报告。

## 计算与内存

Kev-0.8B 复用 Qwen 混合骨干。新增 16 个 GDN value head 和 8Q／2KV 的专用实例。模型包含 24 层，hidden 为 1024，FFN 宽度为 3584。512 维决策投影仍分为 256 维 query 和 256 维 key。

Julia 和 Laya 共用 ModernBERT 权重加载与算子。Julia 为 22 层 encoder 加 2 层决策头，Laya 为 28 层加 2 层决策头。局部层使用对称窗口，全局层保留完整双向注意力。每层的大矩阵可分行计算，整段输入的双向语义保留。

权重保持 Q8_0 存储。乘法内部按块展开，不创建全量 FP16 权重副本。激活使用 BF16，残差、LayerNorm 统计和投影输出使用 FP32。新增 Q8_0 FP32 prefill 输出算子，保留原有 BF16 路径。

决策头只执行当前问题类型。最后一层可只计算选项位置的 query 和输出，所有 key／value 仍来自整段输入。覆盖整个 head 的模式保留，用于数值对照。

同一模型进程内，大工作区按实际输入增长并复用。完整上下文所需的内存先通过预算检查，再公告 Ready。`max-memory` 的 Metal 缓冲预算与 Python tokenizer 的物理内存分别统计。模型权重在 encoder 进程结束前常驻。

`/status` 报告量化权重、辅助权重、工作区、KV、FP16 缓存和 Metal 总峰值。启动列表显示实际启用的功能及源码构建 ID。

## 正确性验收

数值验证与端到端验证分别记录。

| 检查 | 验收 |
|---|---|
| Kev 真实 Q8_0 决策 head | 4096 次 FP64 对照，沿用原有未舍入误差界限 |
| GDN 16-head | 单 token、37、1000、2048 tokens，分块连续状态和 head 排列对照 |
| 8Q／2KV attention | 原有 Q8 页面误差界限、1～4 个请求和长历史用例 |
| FP32 Q8_0 prefill | FP64 界限及 BF16 舍入结果逐位对照 |
| ModernBERT LayerNorm | 384／1024 维 FP64 对照 |
| 全局／局部 attention | 均匀、非均匀 softmax；128／129 tokens；NaN 哨兵；重复提交 |
| GeGLU／ReLU | 独立 FP64 计算及原有 BF16 舍入规则 |
| 元数据 | 缺少 head、错误 shape／类型、因果模式明确失败 |
| 真实请求 | 每模型 13 组，包括精确 128／512／1049／2048／4096／8192 tokens 和混合问题类型 |
| 重复与并发 | 重复长输入、4 个不同请求，沿用 `1e-3 × max(1, abs(value))` 一致性要求 |
| 完整 head／选项 query | 相同输入和参数，沿用 `1e-3` 一致性要求 |

同一 GGUF 的 llama.cpp 作为端到端参考。选择结果与实际输入 token 数逐项核对。FP64 算子验证独立执行。概率差异保留在原始结果中，回答一致不替代算子校验。

本轮修复过两个实际失败：V 的物理布局与 MPP PV 读取不一致，以及 FP32／BF16 合作矩阵转换后部分输出错误。失败日志保留。attention 最终使用逻辑坐标逐元素写回，重复运行仍须通过同一 FP64 断言。

源码门禁还覆盖架构、完整 Python、CPU、Metal 以及 ASan／UBSan／TSan。历史终端日志的原始尾空格通过单文件 Git 属性保留，原始数据不改写。

## 速度与内存对照

性能阶段使用固定输入 128、512、2048、8192 tokens。每组交错运行 3 轮，每轮 5 个请求。每次请求前缀不同，实际 token 数逐项核对，避免把前缀缓存命中算作完整预填充加速。

端到端耗时包含 tokenizer、HTTP 和原生执行。输入 tokens/s 使用实际输入数量除以耗时。进程内存统计包含 Python 服务和原生子进程，采用 macOS 的 physical footprint；RSS 另行保存。各进程生命周期峰值的和，与同时采样的总峰值分别记录。

llama.cpp 的上下文容量设置为 8193，实际输入最多 8192。额外 1 个位置用于它的服务端保留槽。Splash 容量为 8192。主权重保持各自当前 Q8_0 GGUF。Kev 的 KV 固定 BF16，ANE 关闭。

| 模型 | 实际输入 tokens | llama.cpp 平均 ms | Splash 平均 ms | 倍速 | llama.cpp 输入 tokens/s | Splash 输入 tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| Kev-0.8B | 128 | 35.19 | 24.12 | 1.46× | 3,637 | 5,308 |
| Kev-0.8B | 512 | 85.00 | 45.08 | 1.89× | 6,024 | 11,357 |
| Kev-0.8B | 2048 | 301.91 | 126.15 | 2.39× | 6,784 | 16,234 |
| Kev-0.8B | 8192 | 1413.21 | 528.96 | 2.67× | 5,797 | 15,487 |
| Julia-1 | 128 | 11.17 | 9.53 | 1.17× | 11,460 | 13,431 |
| Julia-1 | 512 | 15.92 | 12.83 | 1.24× | 32,155 | 39,891 |
| Julia-1 | 2048 | 58.28 | 28.70 | 2.03× | 35,142 | 71,370 |
| Julia-1 | 8192 | 461.99 | 145.11 | 3.18× | 17,732 | 56,455 |
| Laya | 128 | 20.47 | 14.48 | 1.41× | 6,254 | 8,843 |
| Laya | 512 | 43.69 | 28.31 | 1.54× | 11,719 | 18,088 |
| Laya | 2048 | 177.26 | 96.43 | 1.84× | 11,554 | 21,238 |
| Laya | 8192 | 1173.45 | 519.48 | 2.26× | 6,981 | 15,770 |

8192-token 输入的内存对照如下。物理内存列为同时采样的进程 footprint 峰值；RSS 列为请求完成后的快照中位数。

| 模型 | llama.cpp footprint MiB | Splash footprint MiB | llama.cpp RSS MiB | Splash RSS MiB |
|---|---:|---:|---:|---:|
| Kev-0.8B | 2,411.5 | 3,589.1 | 3,184.9 | 1,060.3 |
| Julia-1 | 8,766.7 | 637.9 | 8,950.4 | 851.7 |
| Laya | 2,417.5 | 1,021.0 | 2,819.4 | 1,027.7 |

Kev 保留约 504 MiB 的 FFN FP16 缓存，物理内存高于参考版本。Julia 和 Laya 的长输入 footprint 较低。macOS 的 footprint 与 RSS 计法不同，mmap 只读权重、压缩页及 IOKit 映射的归属也不同，两个指标分别保留。表中 footprint 差额不能直接解释为权重大小或独立显存差额。

主表共 360 个有效计时请求。Kev 首轮使用 8192 的参考 ubatch，另存为预览数据；主表采用双方 2048 分块的复测。Julia 的 8192-token 输入还追加了 Flash Attention auto／on 路径核对，两者内存量级相同，单次诊断耗时接近；主表保留 auto 设置。


## 可复现入口

```sh
make -j4 BUILD=build/decision-modernbert-20261015 \
  BUILD_ID_PYTHON=/Users/ghost233/.codex/worktrees/qwen4b-quantized-ffn/GhostSplash/.venv/bin/python \
  check-native-cpu check-native-build

make BUILD=build/decision-modernbert-20261015 \
  BUILD_ID_PYTHON=/Users/ghost233/.codex/worktrees/qwen4b-quantized-ffn/GhostSplash/.venv/bin/python \
  check-source check-native-metal check-python test-sanitizers
```

`BUILD_ID_PYTHON` 使用绝对路径。构建测试会切换到临时目录，父 make 传入的相对 `.venv` 路径在那里不存在。

原始记录、逐轮结果和命令保存在 [数据目录](data/decision-models-2026-10-15/)。主表可从 [JSON](data/decision-models-2026-10-15/performance-summary.json) 或 [CSV](data/decision-models-2026-10-15/performance-summary.csv) 核对。
