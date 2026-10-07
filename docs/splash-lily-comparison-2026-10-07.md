# Splash 与 Lily：35B-A3B 速度、兼容性与优化机会

查阅日期：2026-10-07。本文比较 Perplexity 的 Lily 与当前工作区的 Splash。相关公开测试的模型为 **Qwen3.6-35B-A3B**，不能直接外推到其他 35B-A3B 模型、微调权重或量化版本。本次完成资料与源码分析，没有重新运行两引擎的性能测试。

源码范围：Splash 本地 HEAD `3c4166b813fce21403f00cdf02f5da24ca5f3160`，另有用户原有的未提交修改；Lily 查阅时主分支为 `1ed972ed3f0bd5616c997c9507c25616c63394fc`。本次只新增此笔记。

## 结论

- 公开的同机端到端比较支持 Splash 在这个模型上的优势：生成吞吐约为 Lily 的 **1.49–1.78 倍**，32K 有效输入处理吞吐约为 **1.28 倍**。这是发布方的历史结果，不是当前两版本的本机重测。[Splash 发布测试](https://inco.ai/blog/splash/)
- Lily 可以扩大兼容范围。当前限制来自架构、量化布局、GPU 能力及实现和验证范围；源码没有把运行时权重锁死到某个 Hugging Face revision。[Lily 配置校验](https://github.com/perplexityai/pplx-garden/blob/1ed972ed3f0bd5616c997c9507c25616c63394fc/lily/src/config.rs)
- Splash 可以借鉴 Lily，但许多优化思路已经存在。较具体的差异是跨层残差加归一化融合、小行数宽矩阵的调度，以及独立 MoE 分支的并发执行。它们都是待验证的候选方案，尚无证据证明移植后能提高 Splash 的整体吞吐。

## 速度证据及其边界

### 同一台机器上的历史比较

Splash 2026-09-17 发布文章与本地 `docs/performance.md` 给出 M5 Pro、16 核 GPU、48 GB 的测量。文章使用选定的 SPEED-Bench 编码提示，HTTP 请求、最多 1024 输出 token；单请求报告中位数，decode 包括思考 token。引擎采用各自推荐设置，因此比较的是完整运行路径，无法归因到某一个 kernel。[发布文章](https://inco.ai/blog/splash/)、[本地说明](performance.md)

| 指标 | Splash | Lily | Splash / Lily |
| --- | ---: | ---: | ---: |
| 短提示生成 | 210 tok/s | 118 tok/s | 1.78× |
| 8K 提示后生成 | 156 tok/s | 105 tok/s | 1.49× |
| 16K 提示后生成 | 149 tok/s | 94 tok/s | 1.59× |
| 32K 提示后生成 | 143 tok/s | 85 tok/s | 1.68× |
| 32K 有效输入处理 | 2011 tok/s | 1572 tok/s | 1.28× |

有效输入处理吞吐由未缓存输入 token 数与 HTTP 首 token 时间计算，包含服务开销，不能直接视为纯 GPU prefill kernel 吞吐。[指标定义与原表](https://inco.ai/blog/splash/)

有两个复现限制：发布文章未在所查页面给出 Lily 的完整运行 commit；而 Lily 当前 README 规定服务端 greedy、关闭 thinking、非流式且拒绝 speculative decoding，与发布文章的统一方法描述不能完全对齐。因此这些数据足以作为历史同机端到端证据，不能证明当前版本在严格一致协议下仍保持同样倍率。[Lily 当前接口契约](https://github.com/perplexityai/pplx-garden/blob/1ed972ed3f0bd5616c997c9507c25616c63394fc/lily/README.md)

### Lily 自己的测试不是另一组同机对照

Lily 2026-09-02 的报告在 **M5 Max、40 核 GPU、128 GB、macOS 27.0** 上运行，使用固定 MLX Q4 checkpoint，64 个生成步骤。部分结果如下：

| 上下文 | Lily 生成 | Lily 输入处理 |
| --- | ---: | ---: |
| 256 | 194.7 tok/s | 2866.6 tok/s |
| 32K | 151.1 tok/s | 3927.5 tok/s |
| 128K | 92.7 tok/s | 1792.7 tok/s |

这与前述 M5 Pro 不是同一硬件或测量协议，不能拿 194.7 对 210 判断胜负。Lily 报告还说明，比较的 MLX 路径产生完整 vocabulary log-probability 向量，Lily 只选 argmax；两边自由生成的 token 轨迹也可能不同。这组数据展示 Lily 的能力，并未隔离纯 kernel 优势。[Lily 原始报告和复现条件](https://github.com/perplexityai/pplx-garden/blob/1ed972ed3f0bd5616c997c9507c25616c63394fc/lily/docs/2026-09-02-performance-mlx-0.32.2.md)

## “只针对特定版本”具体限制什么

### 权重 revision 与支持的架构是两件事

README 为基准和测试固定了模型 revision `38740b847e4cb78f352aba30aa41c76e08e6eb46`。`TextConfig::from_model_dir` 读取用户提供的本地目录，并校验配置；所查 loader 没有对 Hugging Face 仓库名、revision 或整套权重 hash 做白名单校验。因此不能把复现实验所用的 revision 解释为运行时只能使用那一份权重。[README](https://github.com/perplexityai/pplx-garden/blob/1ed972ed3f0bd5616c997c9507c25616c63394fc/lily/README.md)、[配置入口](https://github.com/perplexityai/pplx-garden/blob/1ed972ed3f0bd5616c997c9507c25616c63394fc/lily/src/config.rs)

当前实际校验包括：hidden size 2048、40 层、词表 248320、256 专家/top-8、专家宽度 512、固定的全注意力与 GDN 头尺寸，以及每四层一层全注意力。量化要求 affine 4-bit/group64，router 等另有 Q8 布局要求。Lily 还要求 Apple GPU family 10 或更新，即 M5 及以后。[配置校验](https://github.com/perplexityai/pplx-garden/blob/1ed972ed3f0bd5616c997c9507c25616c63394fc/lily/src/config.rs)、[权重 loader](https://github.com/perplexityai/pplx-garden/blob/1ed972ed3f0bd5616c997c9507c25616c63394fc/lily/src/weights.rs)、[平台要求](https://github.com/perplexityai/pplx-garden/blob/1ed972ed3f0bd5616c997c9507c25616c63394fc/lily/README.md)

### 为什么不能只删掉校验

限制不全在入口。GDN shader 固定 `DIM=128`；小矩阵及量化 kernel 对 group64、tile 和尺寸整除有实际要求。改变配置可能使线程布局、缓存大小、状态更新和权重解释不再匹配。删除校验不会自动产生新的计算实现。[GDN shader](https://github.com/perplexityai/pplx-garden/blob/1ed972ed3f0bd5616c997c9507c25616c63394fc/lily/src/kernels/metal/gdn.metal)、[小矩阵调度](https://github.com/perplexityai/pplx-garden/blob/1ed972ed3f0bd5616c997c9507c25616c63394fc/lily/src/kernels/skinny.rs)

工程上可以分三步扩大兼容范围，以下是基于源码约束的建议，并非当前已支持功能：

1. **同架构、同布局的其他 revision 或微调模型**：先验证张量命名、router 精度、tokenizer、chat template、停止 token 和输出正确性。这类扩展最接近现有执行路径，不必先实现任意模型 runtime。
2. **同模型的其他量化版本**：实现新的格式 loader、反量化与矩阵 kernel，或者把原始高精度权重量化为既有 MLX affine64 布局。GGUF 与 MLX affine64 不能靠改后缀互换；从已有量化权重再次量化也不能保证精度不变。
3. **不同模型尺寸或架构**：增加明确的模型执行计划与 kernel 变体，重新处理注意力、位置编码、MoE/稠密 FFN、缓存及状态恢复，并建立正确性与速度基准。

“统一加载、API 和调度”与“按模型和设备选择特化 kernel”可以同时存在。扩展支持范围不意味着每次推理都必须执行一个低效的统一 kernel；成本主要在实现、调优和验证更多路径。

Splash 也采用有限模型家族的特化设计。目前本地实现的两个主要家族是 Qwen3.8-27B 与 Qwen3.6-35B-A3B，并共享 `QwenHybridLayout`、目标执行器与缓存机制。加载和算子层已区分 MLX affine64 与多种 GGUF block 布局，所以格式兼容面较宽，但还不能据此推断支持任意模型。[本地 README](../README.md)、[共享模型布局](../runtime/model/QwenHybridLayout.hpp)、[35B 布局](../runtime/model/Qwen3_6Moe.hpp)、[加载分流](../runtime/model/QwenTargetLoader.hpp)

## 哪些 Lily 优点值得用于 Splash

### 已有的共同思路

Splash 已有融合的专家 gate/up、按专家分组复用权重、预填充与验证的不同 tiles、设备相关的矩阵路径、归一化时准备 Q4 operand、INT8 paged KV，以及把 drafting/verification/acceptance 放进完整 command graph 的路径。重新添加相同概念不能构成新的性能收益。[MoE 计划](../runtime/ops/MoE.cpp)、[设备策略](../dev/benchmarks/device-policy.md)、[已有融合](../dev/benchmarks/remaining-decode-optimizations.md)

Lily 的“argmax-only”也不能理解成融合了词表投影和 argmax：其代码仍产生完整 F32 logits，再执行 argmax。Splash 对 greedy 请求已有独立 argmax 路径，对采样请求执行概率相关计算，因此“让 Splash greedy 跳过完整 log-probs”不是新发现。[Lily 生成与模型代码](https://github.com/perplexityai/pplx-garden/blob/1ed972ed3f0bd5616c997c9507c25616c63394fc/lily/src/model.rs)、[Splash sampling](../runtime/ops/Sampling.cpp)、[Splash 输出 head](../runtime/model/QwenTarget.cpp)

### 有具体代码差异的候选方案

| Lily 的做法 | Splash 当前代码 | 移植判断 |
| --- | --- | --- |
| FFN 残差相加与下一层 RMSNorm/最终 norm 融合 | `moe_combine` 写回 bf16 hidden；下一层另做 RMSNorm | 优先试验。可能减少 dispatch 和中间读写；必须保留 bf16 舍入、draft 所需 hidden capture、Q4 operand 准备及不同 norm 精度 |
| 按小行数、矩阵宽度选择 staged-A/register-A kernel | `Linear.cpp` 已有模型和设备策略，但不是相同的 kernel 家族 | 对真实 35B head/投影做离线对照，作为候选；不能直接替换所有设备默认策略 |
| concurrent encoder 配合显式依赖 barrier，交错 shared/routed expert 分支 | `MetalBackend.mm` 使用普通 ordered compute encoder；shared expert 已在同一 grouped plan 内 | 改动较大。拆分分支会增加调度和依赖管理，也可能争用内存带宽；先证明独立分支并行有实际收益 |

Lily 来源：[模型融合与分支安排](https://github.com/perplexityai/pplx-garden/blob/1ed972ed3f0bd5616c997c9507c25616c63394fc/lily/src/model.rs)、[Metal 提交](https://github.com/perplexityai/pplx-garden/blob/1ed972ed3f0bd5616c997c9507c25616c63394fc/lily/src/metal.rs)、[skinny kernel 调度](https://github.com/perplexityai/pplx-garden/blob/1ed972ed3f0bd5616c997c9507c25616c63394fc/lily/src/kernels/skinny.rs)。Splash 对照：[MoE combine](../runtime/metal/kernels/shared/moe.metal)、[下一层归一化](../runtime/model/QwenTarget.cpp)、[矩阵计划](../runtime/ops/Linear.cpp)、[Metal 提交](../runtime/metal/MetalBackend.mm)。表中的收益判断均为推断，尚未实测。

### 投机解码使移植的评价标准不同

Lily 的普通生成路径每轮处理一个新 token。Splash 每个请求每轮验证 8 行：一个 anchor 与七个 proposal；四请求时合计 32 行。专家选择、权重复用及小矩阵尺寸都会变化。针对单行 GEMV 的优势，不一定会在 8/16/24/32 行验证中保留。[Lily 模型图](https://github.com/perplexityai/pplx-garden/blob/1ed972ed3f0bd5616c997c9507c25616c63394fc/lily/src/model.rs)、[Splash 执行几何](../runtime/metal/abi/ExecutionGeometry.h)、[验证图](../runtime/model/QwenTarget.cpp)

Splash 的历史记录提供了一个具体反例：attention 的并行 denominator reduction 虽通过数值测试，却改变了 draft 接受情况，使部分提示的整体吞吐下降，因此被撤回。kernel 变快与最终 tok/s 变快不能等同；微调模型也应重新测匹配 draft 的接受率。[历史优化记录](../dev/benchmarks/remaining-decode-optimizations.md)

另一个更贴合 Splash 的研究方向是减少低置信度 proposal 后缀引发的无效专家工作。上游 issue #280 已提出保留八行物理布局、按有效前缀过滤专家路由的方案；其中单请求 5%、四请求 10% 是验收目标，**不是已取得的加速**。这不是 Lily 的现成能力，也不能列为已完成移植。[提案与验收条件](https://github.com/incoai/splash/issues/280)

## 建议的实施与验证顺序

1. **先固定基线并 profile**：同一机器、固定模型 revision/实际张量、温度与 thinking 设置；分开记录冷 prefill、生成和缓存复用。使用现有 `benchmark-decode-profile` 分解 B1–B4 的 MoE、norm、head、attention、draft 与 host 开销。
2. **先试验跨层 MoE combine + RMSNorm 融合**：范围较清楚，也较少依赖单 token 算法。确保需要的未归一化 hidden 仍可捕获，不改变精度或内存计划。
3. **再比较小行数宽矩阵 kernel**：特别是验证块和词表 head；分设备、MLX/GGUF 判断，不从热权重微基准推断全模型收益。
4. **有证据后再做分支并发或有效验证行数**：这些涉及 command 依赖、接受与 commit，单独验证后再叠加。

通过标准应同时包括 greedy 输出和状态恢复正确、采样分布规则正确、draft 接受情况不抵消收益、显存可接受，以及短/长提示、并发 1–4 下的重复 HTTP ABBA 性能收益。使用 `metrics.decode_cycle_ms` 或完整服务时间纳入 host 开销，而非只看某个 GPU kernel 的时间。[现有测量方法](../DEVELOPMENT.md#local-benchmarks)、[性能资格规则](../dev/benchmarks/device-policy.md)、[decode profile](../dev/benchmarks/decode_profile.mm)

当前证据支持继续以 Splash 为主要运行时，选择性借鉴 Lily 的局部实现。扩大同家族权重和量化兼容性、移植某个 kernel、改变投机验证策略是三个独立工作项，应分别建立证据和验收标准。
