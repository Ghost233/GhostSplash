# Flash Next：上游 PR、MTP 接入与 prefill 优化

查阅日期：2026-10-09。目标硬件是本机 M5 Pro / 20 核 GPU / 64 GiB；目标权重是本地 Swift 1.5 Flash Next IQ3_S 两分片，专家权重采用 SSD 按需读取。本文只做源码调研，未运行模型推理，也未验证外部 PR 的性能数字。

## 固定来源与适用边界

| 来源 | 查阅的完整 head | 状态与范围 |
| --- | --- | --- |
| [Splash PR #253](https://github.com/incoai/splash/pull/253) | `9c3f60b9d27c240fc89825086ad1c63c8c22e9b3` | Open；基于 `0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7`，接入主运行时的 GGUF / 原生 MTP |
| [Splash PR #115](https://github.com/incoai/splash/pull/115) | `be8360e64ce944c2fc30e13c80df432a0d37af07` | Open；基于 `edb4b8fa4eee5fef624809cd7f30f0651c58a167`，独立 MLX affine worker 与 M5 Ultra 可选路径 |

#253 作者报告在 M3 Ultra 上完成 CPU / Metal 测试、工具调用、并发和缓存测试，并报告 greedy MTP 开关输出一致；这些是作者提供的证据，不能等同于我们的 Swift IQ3_S + 外部 BF16 drafter + SSD 专家执行的验证。#115 明确表示 rebased 源码尚未完成成对的整模型 HTTP 运行，旧本地 checkpoint/profile 不在 PR 中，未宣称 rebased build 性能。[#253 验证说明](https://github.com/incoai/splash/pull/253)、[#115 验证限制](https://github.com/incoai/splash/pull/115)

两份 PR 均不能直接给出这台 M5 Pro 的吞吐或提速倍率。#253 使用常驻 GGUF 权重；#115 的主要专家算子使用 MLX affine 原始权重，另有 hot-expert BF16 缓存。这与我们超过物理内存的 IQ3_S GGUF 权重不同。[#253 权重加载](https://github.com/janhilgard/splash-incoai/blob/9c3f60b9d27c240fc89825086ad1c63c8c22e9b3/runtime/model/Qwen4Exp.cpp)、[#115 专家缓存契约](https://github.com/mweinbach/splash-flash-next/blob/be8360e64ce944c2fc30e13c80df432a0d37af07/runtime/flash/FlashExpertDenseCache.hpp)

## 1. prefill 应先改变执行顺序

本地旧路径在每个 token 上运行完整 48 层；每层将 router 结果读回 CPU，依次执行 10 个专家，每个专家独立提交并等待 GPU，最后在 CPU 累加输出。GDN packed 行和 attention Q/gate、KV 写入也有 CPU 往返。因此当前 prefill 不具备矩阵批处理的权重摊销优势。[本地执行路径](../runtime/model/FlashNextRuntime.cpp)

#115 的 `FlashBatchPrefill` 接口接受 1–4 个独立请求，每个请求 1–2048 个真实 token，最多 8192 个物理行；全 48 层沿层顺序形成一个图。它明确不生成 padding token，不跨请求复用状态，也不自动提供 prefix snapshots。大行数 MoE 使用 blocked/bucket 路径。[接口](https://github.com/mweinbach/splash-flash-next/blob/be8360e64ce944c2fc30e13c80df432a0d37af07/runtime/flash/FlashBatchPrefill.hpp)、[实现](https://github.com/mweinbach/splash-flash-next/blob/be8360e64ce944c2fc30e13c80df432a0d37af07/runtime/flash/FlashBatchPrefill.cpp#L366)

其分桶先统计每个专家的路由数量、计算偏移、构造稳定 route map，再打包输入和创建矩阵 tile jobs。可以借鉴稳定分组与 canonical scatter，以同一个专家的一次权重读取服务多个 token。[MoE bucket 实现](https://github.com/mweinbach/splash-flash-next/blob/be8360e64ce944c2fc30e13c80df432a0d37af07/runtime/flash/FlashMoEBuckets.cpp#L132)

**本地实施建议（推断）：** 先按层执行小批真实行；dense / HC 使用 batched projection；同层 router 一次读回并分组，然后在 20 GiB 专家缓存约束内逐组载入 IQ3_S gate/up/down，计算这些专家的全部路由行。不要为了实现单个全模型 GPU graph 而将全部专家常驻。分组降低 token 级重复 SSD 读取，但速度收益必须测量。

GDN 的 recurrence、PLE convolution、QSA raw/pool 位置仍须按真实 token 顺序更新。prefill 中间行通常不需 vocabulary logits，只在最后需要采样的位置计算头部；接入 MTP 后须保留用于 teacher cache 的 trunk 特征。[#253 prefill / MTP 配对](https://github.com/janhilgard/splash-incoai/blob/9c3f60b9d27c240fc89825086ad1c63c8c22e9b3/runtime/model/QwenTarget.cpp)、[#115 prefill 状态接口](https://github.com/mweinbach/splash-flash-next/blob/be8360e64ce944c2fc30e13c80df432a0d37af07/runtime/flash/FlashBatchPrefill.hpp)

## 2. MTP 输入与 teacher cache 必须正确对齐

#253 的 MTP 输入是当前 token embedding 与前一个位置的完整 HC streams；首行读取已 carry 的 streams。它在 prefill 和 target verify 中补齐 MTP attention KV，并在提交时 carry 最后一个 retained trunk 行。MTP 头还复用目标词表输出投影。[MTP 实现](https://github.com/janhilgard/splash-incoai/blob/9c3f60b9d27c240fc89825086ad1c63c8c22e9b3/runtime/model/QwenTarget.cpp)

#115 具体执行 global hidden RMS 后，把每条 normalized 2560 维 stream 都应用同一个 `fc_hidden`，然后加入 `fc_embedding` 的投影；MTP teacher priming 只需建立 QSA cache，不需计算完整 attention / MLP 输出。`truncate` 回退逻辑位置，后续写入覆盖旧 provisional 行及重新完成的压缩块。[输入准备](https://github.com/mweinbach/splash-flash-next/blob/be8360e64ce944c2fc30e13c80df432a0d37af07/runtime/flash/FlashMTP.cpp#L579)、[MTP 接口契约](https://github.com/mweinbach/splash-flash-next/blob/be8360e64ce944c2fc30e13c80df432a0d37af07/runtime/flash/FlashMTP.hpp)

本地 `/Users/ghost233/.lmstudio/models/sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16` 的配置和 safetensors header 已只读检查：`model_type=qwen4_exp_mtp`、单 full-attention 层、512 专家 / top-10；32 个 BF16 张量；`pre_fc_norm_hidden.weight=[10240]`、`pre_fc_norm_embedding.weight=[2560]`；`fc_hidden.weight` 和 `fc_embedding.weight` 均为 `[2560,2560]`。没有独立 embedding / vocabulary head 张量。这些是本地文件证据，不代表与 Swift 微调的 acceptance 已获验证。

**本地实施建议（推断）：** 复用目标 token embedding 和 vocabulary head；保留原始 4-stream trunk 特征；按 drafter 张量约定做 global 10240 维 RMS，再逐 stream 投影。不能用最终混合后的 2560 维 hidden 替代完整 10240 维 HC 特征。BF16 drafter 与 IQ3_S trunk 可组成 speculative 路径，但其 acceptance 与性能需实测。

## 3. lossless verify 要回滚所有有状态分支

#115 将 target verify 定义为 provisional trial；每 lane 独立保留 anchor + 已接受 drafts。其 greedy 前缀接受算法比较 `prediction[i]` 和 `input[i+1]`，输出连续匹配 drafts 加首个 correction / bonus，并让 EOS 和输出额度限制影响实际 retained rows。[verify 接口](https://github.com/mweinbach/splash-flash-next/blob/be8360e64ce944c2fc30e13c80df432a0d37af07/runtime/flash/FlashBatchVerify.hpp)、[greedy 前缀算法](https://github.com/mweinbach/splash-flash-next/blob/be8360e64ce944c2fc30e13c80df432a0d37af07/runtime/flash/FlashMTPWindow.hpp)

拒绝后只回退 KV 长度不够。GDN recurrent / convolution、PLE 历史 / convolution、QSA pooled block 与 raw keys、MTP cache / carried features、token history 都必须对应 retained 前缀。#115 的 lazy GDN rollback 保存 trial 初始 FP32 state 和准备后的 operands，部分接受时只重放 recurrence，不重跑 model projections；全接受走免重放路径。[lazy GDN rollback](https://github.com/mweinbach/splash-flash-next/blob/be8360e64ce944c2fc30e13c80df432a0d37af07/runtime/flash/FlashGDNLazyRollback.hpp)

**本地实施建议（推断）：** 首先建立可复现的短 greedy MTP on/off 一致性测试，并覆盖 0/部分/全 draft 接受、EOS、quota、压缩块边界、后续继续生成。先用清晰的 snapshot / restore 确保正确，再按 profiling 决定是否移植 lazy recurrence replay。仅添加 draft forward 而 target 仍逐 token 验证，不会自动带来批量验证提速。

## 4. SSD 与内存约束需保留

#115 的 PLE SSD 路径不会为整张 n-gram 表申请 GPU 内存，每个 executor 有自己的 bounded staging，磁盘 store/cache 共享；预取失败在 graph submission 前发生，不污染请求历史与 convolution。[PLE SSD 契约](https://github.com/mweinbach/splash-flash-next/blob/be8360e64ce944c2fc30e13c80df432a0d37af07/runtime/flash/FlashPLESSD.hpp)

#115 hot-expert BF16 cache 按固定选择建立，保留原始权重，完整 cached jobs 走 BF16，miss / tails 留在 Q4 路径；明确要求提前预留 plannedBytes，并重新验证 whole-K reduction 的数值。它不是可直接用于任意 IQ3 GGUF 的动态 SSD cache。[专家缓存](https://github.com/mweinbach/splash-flash-next/blob/be8360e64ce944c2fc30e13c80df432a0d37af07/runtime/flash/FlashExpertDenseCache.hpp)

**本地实施建议（推断）：** IQ3_S 权重格式保留；先将专家按批次加载和重用，保持持有中的页不可淘汰，host staging、resident expert cache、BF16 draft、verify snapshots 和 request state 都纳入内存计划。避免用大 mmap 的系统 page cache 替代明确预算。SSD bytes/read、miss、eviction 和每层 GPU/host 时间必须与 tok/s 一起记录。

## 5. 验证证据不能混淆

#115 `flash_batch_prefill_oracle` 默认并非 bit-exact，默认 relative L2 上限为 0.015；`_EXACT=1` 才要求 identical bits。它还检查 state metadata、continuation 和 canaries。其 batch MTP oracle 默认 exact。数值阈值通过不能写成 greedy 输出一致或工具语义通过。[prefill oracle](https://github.com/mweinbach/splash-flash-next/blob/be8360e64ce944c2fc30e13c80df432a0d37af07/dev/benchmarks/flash_batch_prefill_oracle.mm)、[MTP oracle](https://github.com/mweinbach/splash-flash-next/blob/be8360e64ce944c2fc30e13c80df432a0d37af07/dev/benchmarks/flash_batch_mtp_oracle.mm)

本地验收顺序建议：

1. 旧短测结果、build ID、模型 hash、实际 GPU / host / SSD 指标保留为基线。
2. 分批 prefill 对比 scalar 路径的中间特征与最终 logits；同时验证 GDN、PLE、QSA 状态，检查 continuation。
3. greedy MTP on/off 逐 token 一致；单独测 draft acceptance、verify GPU 时间和 rollback 时间。
4. 先短输入再 2K / 4K / 8K / 16K，规定超时、内存上限、结束后关闭测试服务；任何超时均记录未完成。
5. 温度非零及工具约束模式必须沿现有 sampling / constraint 契约运行，不得把 greedy 的保证自动扩展为任意采样模式正确。

## 执行优先级

| 优先级 | 修改 | 成功证据 |
| --- | --- | --- |
| P0 | 正确读取外部 BF16 MTP；teacher cache 与 streams 对齐 | shape / norm 单测，短 greedy 输出一致 |
| P0 | 层优先真实多行 prefill + 专家分桶 | 数值与 state continuation 正确，SSD bytes/token 和 wall time 下降 |
| P1 | 批量 target verify + 完整状态回滚 | 拒绝 / 全接受 / EOS / quota 正确，MTP 实测比 AR 快 |
| P1 | GDN packing、Q/gate split、KV write、combine 移到 GPU | CPU 往返与提交次数下降，输出不回退 |
| P2 | bounded 异步 expert I/O、lazy GDN rollback、按测量调 chunk/depth | 同硬件同请求 A/B 改善，不超预算 |

#115 的 M5 Ultra bulk QSA route 只在特定 fresh 2048-row 条件开启，并要求多个配套 flag；本机是 M5 Pro，没有证据可以直接启用这些硬件 profile。先采用通用正确路径再测量调度。[硬件路线说明](https://github.com/mweinbach/splash-flash-next/blob/be8360e64ce944c2fc30e13c80df432a0d37af07/dev/FLASH_NEXT.md)
