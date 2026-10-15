# Qwen3.5-4B 的 INT8 推理优化与 DFlash 验证

## 验收范围

这份记录用于复现 4B 的算子优化和 DFlash 性能测试。测试机器为 M5 Pro，配备 20 个 GPU 核心和 64 GiB 内存。

目标模型是 `Qwen3.5-4B-UD-Q4_K_XL.gguf`。我们只验收 INT8 KV 路径。接口使用单请求、33,792 tokens 上限和 12 GiB 内存预算，关闭 ANE。

Drafter 使用 [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash/tree/9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf)。固定版本为 `9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf`。运行块大小为 8，每轮提出 7 个 token。

目标模型有 32 层，隐藏宽度为 2560。Drafter 有 6 层，读取目标的 8 个捕获层。前 5 层使用 4096 滑动窗口，最后一层保留完整历史。

## 保留的改动

| 项目 | 最终行为 | 验证证据 |
|---|---|---|
| 原生单 token decode | 专用 GGUF GEMV、单步 GDN 融合、单行输出头 | 单 token 数值与 GDN 状态测试 |
| Prefill FFN | 复用 FP16 权重缓存，大块使用 M64N32 | 原解量化舍入、BF16 位比较、CPU 参考和边界保护 |
| 4B 注意力 | 自动使用 r2/r16 query tile | 原 INT8 scalar、cosine 和质量门禁 |
| DFlash | 标准 Transformer、真实词表头、主模型验证与完整状态回放 | 整层参考、采样、接受统计、状态恢复和真实请求 |
| 验证归约 | 共享同一行的 denominator | 旧新输出逐字节一致，独立耗时降低约 9% |
| 32 行尾部预填充 | 4B INT8 的 r16 计划使用单个 history split | 原数值门禁、完整总结、缓存复用和接口 ABBA |

FFN 缓存增加约 4.219 GiB。Decode 继续读取原量化权重。

## 32K 请求的切块与数值变化

真实接口的 32,768-token 冷请求使用 17 个 prefill 批次。前 15 个批次各有 2048 tokens，最后两个批次分别为 2016 和 32 tokens。

缓存必须在 32,736 处发布回放状态。我们保留了这个边界。当前 chunk 和历史 chunk 都从 INT8 paged KV 读取。

原计划在最后 32 行使用 16 个 history split。完整 2K 批次使用 1 个 split。分片会改变分区 softmax、BF16 概率舍入和归约组织，进而改变隐藏状态与生成结果。

最终策略只改变 4B INT8 的 32 行 r16 计划，使用 1 个 split。其他形状、KV 格式及显式 r8 计划继续使用原策略。

**这是数值路径变化。** 固定总结请求的输出从 383 tokens 变为 311 tokens。原 scalar、FP64、cosine 和质量要求保持原值。完整输出保留了全部 12 项事实。

离线工具原先使用 16 个完整 2K 批次。它得到 311 tokens，不能直接作为接口验收。逐 token 比较确认输入相同；工具改用接口的回放边界后，才复现原接口的 383 tokens 和 31.96% 接受率。

## 实测结果

Prefill 速度按原生批次 wall time 计算。Decode 速度按 HTTP 首次输出到结束的时间计算，并扣除首次输出批次。所有冷请求的缓存复用 token 数为 0。

| 请求 | 候选 ABBA 的两次 decode | 最终冷请求 | Drafter 接受率 |
|---|---:|---:|---:|
| 512 tokens 输入、256 tokens 输出 | 175.1、172.5 tok/s | 参见 ABBA 原始记录 | 约 35% |
| 32,768 tokens 输入、311 tokens 输出 | 153.4、152.4 tok/s | 157.3 tok/s | 39.1% |

原版本的同一 32K 请求为 130.7、135.4 tok/s，接受率约为 32.0%。最后单次冷请求的 prefill 为 2513 tok/s，原早期 NoDraft 总结记录为 2212 tok/s。

长时间连续测试出现了吞吐波动。候选 ABBA 的 prefill 为 2130、2511 tok/s；原版本为 2236、2515 tok/s。尾部单分片的主要收益是接受率。原有 FFN 缓存与 query tile 优化承担了 prefill 提升。

最终冷请求的峰值 Metal 分配为约 9.83 GiB。这个数值低于 12 GiB 预算。

随后两次相同请求都复用了 32,736 tokens，只重算末尾 32 行。答案与冷请求完全相同，12 项事实均通过。Decode 分别为 156.6、156.1 tok/s。

## 门禁与收敛

最终源码通过 `check-source`、完整 CPU 回归及全部原生编译入口。单 token、FFN 缓存、GDN、INT8 注意力、DFlash 整层、词表选择、draft 注意力和状态恢复 GPU 检查通过。

原真实采样测试覆盖了惩罚参数、top-k、min-p、混合请求和约束输出。完整 Python 测试共 1256 项通过，4 项按环境跳过。格式问题修复后，65 项相关测试和全库 lint、格式检查通过。

以下实验均已归档撤回：M8 K64、词表头 N32/N128 分组、GDN 大状态行分块、Q-only staging、drafter 8/16 attention split，以及精确 FC。精确 FC 没有提高 32K 接受率，还增加了计算和内存成本。

## 复现与记录

在优化 worktree 中启动：

```sh
./qwen3.54bdflashsplash.sh --kv-format int8
```

完整版本、SHA-256、指标和门禁日志路径见 [最终 JSON 报告](../build/qwen4b-goal150-final-report.json)。接口 ABBA 记录位于 `build/goal150-suffix-abba/`。缓存验收记录位于 `build/goal150-final-cache/`。

测试服务已经停止。原主工作区的 `ghost` 分支保持干净。代码保留在 `codex/qwen35-4b-optimization` worktree，尚未提交或推送。
