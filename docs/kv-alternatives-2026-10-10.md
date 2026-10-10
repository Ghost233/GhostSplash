# 其他 KV 量化候选（2026-10-10）

## 优先考虑什么

针对 Qwen3.5-4B 的 32K 总结，先关注 SAW-INT4 的融合写入和非对称量化设计。它与我们已有的 Hadamard + Q4 路线接近。其次可借鉴 SKVQ 的近期高精度窗口。完整 KVQuant 涉及校准、码本和稀疏数据，接入成本最高。

这是路线比较。下面区分论文结果、作者实现和推断。本轮没有编译、测试、推理或启动服务，也没有扩大开发范围。

## 主要候选的区别

| 候选 | 4-bit 方案 | 准备工作 | 高精度保留 | 当前实现 |
|---|---|---|---|---|
| SAW-INT4 | 每 token、每 head 的非对称 INT4，含 scale 和 zero-point；块 Hadamard 旋转 | 主方案无需数据校准 | 主方案没有近期窗口或稀疏离群值 | SGLang、CUDA、FA3 和 Triton |
| KIVI-4 | K 按通道跨 token 分组，V 按 token 分组 | 无需模型校准或微调 | 近期高精度残留缓存 | CUDA 融合矩阵计算和 Triton |
| SKVQ | 通道重排、分组裁剪、动态非对称量化 | 重排和裁剪需离线校准 | 近期窗口、开头 attention sink | PyTorch 和 CUDA；论文用模拟量化 |
| KVQuant | K 按通道、RoPE 前量化；非均匀码本和稀疏离群值 | 新模型需梯度、Fisher 信息和码本校准 | 例如 1% 离群值；新版支持开头高精度 token | CUDA 压缩、查表和稀疏计算核 |

依据：[SAW 官方说明](https://github.com/togethercomputer/saw-int4)、[SKVQ 官方说明](https://github.com/cat538/SKVQ)、[KVQuant 官方说明](https://github.com/SqueezeAILab/KVQuant)。官方说明所列工具链为 CUDA 或 Triton。本文未找到可直接接入 Splash 的 M5 Metal 实现。

## SAW-INT4：优先借鉴融合和量化粒度

**作者实现。** 官方主方案是块旋转和 INT4 写缓存融合。默认只旋转 K，decode 时给 Q 同步变换；V 旋转可选。示例采用 H128。与现有标准 Q4_0 相比，它改成每 token/head 的非对称量化，不是每 32 元的对称 block。[环境变量](https://github.com/togethercomputer/saw-int4/blob/main/docs/bdr_env_vars.md)、[论文 §3](https://arxiv.org/html/2604.19157v1#S3)

**作者自测。** Qwen3-4B-Thinking-2507 在 GPQA 上，BF16 为 66.6667，K-only 旋转 INT4 为 65.8249。设置为 temperature 0.6、top_p 0.95、最大序列 32K。这里的 32K 不是 32K 文档总结输入。[准确率记录](https://github.com/togethercomputer/saw-int4#accuracy-results-primary-temp06-seq32k-and-top095)

吞吐测试用另一模型 Qwen3-8B，单张 H100 80 GB。16,384 输入、1024 输出、并发 8 时，INT4 为 458 output tok/s，旋转 INT4 为 457。它支持融合后开销小的判断，不能推导 M5 速度。[吞吐记录](https://github.com/togethercomputer/saw-int4#speed-results-primary)

**接入推断。** 这是最值得先看的系统设计。保留页式缓存，融合旋转和写入，再在读取 tile 时解码。我们已有固定 H256/H64 和局部 BF16 stage。可以分别比较旋转块大小、K-only 与 K+V，以及非对称 scale/zero-point。没有必要直接搬整个 SGLang fork，也不能保证这些选择优于当前 Q4_0。

## KIVI-4：区别处理 K 和 V

KIVI 有 4-bit 模式。K 按通道跨 token 分组，V 按 token 分组，同时保留近期高精度残留缓存。方案无需模型校准或微调。论文主要设置为分组大小 32、残留长度 128。[论文 §3.3](https://arxiv.org/html/2402.02750v2#S3.SS3)

**论文结果。** Mistral-7B-v0.2 在 32K LongBench 上，平均分从 43.54 变成 43.53。GovReport 总结分为 32.59 和 32.97，QMSum 为 23.99 和 24.06。所测任务上接近高精度基线；这些小变化不应解释成质量提升。模型、权重和数据集与当前任务不同。[论文 Table 9](https://arxiv.org/html/2402.02750v2)

**工程边界。** 官方实现在 CUDA/Triton 上。prefill 使用完整精度 K/V 算 attention，然后保留压缩缓存。长期缓存更小，不保证 prefill 峰值也更小。移植还需处理分组未满的 K、窗口滑出压缩，以及前缀复用时的近期缓存恢复。它适合提供一个无需校准的质量对照，并未证明比我们的旋转 Q4_0 更好。

## SKVQ：窗口可以单独借鉴

**论文设置。** 通道重排和裁剪用 256 段、每段 4096 token 的 WikiText 训练数据校准。主要实验保留最近 128 个 token，并可保留开头 5 个 sink token。prefill 先用高精度 KV 算 attention，再压缩旧缓存。[论文设置](https://arxiv.org/html/2405.06219v2#S4.SS1)、[窗口策略](https://arxiv.org/html/2405.06219v2#S3.SS2)

Llama-2-13B 的 WikiText 对照中，4-bit SKVQ PPL 为 4.60，普通对称舍入为 4.66。该项比较不使用滑动窗口，包含重排和裁剪。[论文 Table 2](https://arxiv.org/html/2405.06219v2#S4.SS3)

**实现限制。** 官方仓库明确说明，论文结果来自模拟量化。当前解码和 GEMV 核简单且效率低。论文的“7 倍 decode”是 A100、batch 128、200K 长度下的理论分析，不是本机或该仓库的端到端实测。[官方限制](https://github.com/cat538/SKVQ#usage)、[论文性能分析](https://arxiv.org/html/2405.06219v2#S1)

**接入推断。** 单独保留近期窗口，比完整通道重排简单。但窗口需混合格式读取、滑出时压缩，以及快照恢复。按当前 4B raw KV 计算，128 个 token 从 Q4_0 改存 BF16，理论差额约 2.875 MiB，不含页对齐。若同时保留 Q4 备份和 BF16 窗口，BF16 额外数据为 4 MiB，还需计入窗口状态和快照。窗口保护最近内容，不能保证文档前部关键事实得到保护。

完整高精度 prefill 还可能增加峰值。不能将“近期窗口便宜”推导成整个 SKVQ 的 prefill 都便宜。

## KVQuant：质量依据较强，格式改动较多

**作者实现。** 新模型需先计算 Fisher 信息，再生成量化器。论文使用 16 段、每段 2K token 的 WikiText 校准样本及梯度。部署保存码本索引和稀疏离群值，K 在 RoPE 前量化，读取时应用 RoPE。[官方流程](https://github.com/SqueezeAILab/KVQuant#how-the-code-is-structured)、[论文附录 K](https://arxiv.org/html/2401.18079v4#A11)

**论文结果。** SqueezeLLM 4-bit 权重下，LLaMA-7B 的 WikiText PPL 从 5.77 变成 5.79；13B 从 5.17 变成 5.18。KV 用非均匀 4-bit 加 1% 高精度离群值，实际平均为 4.32～4.33 bit。[论文 Table 4](https://arxiv.org/html/2401.18079v4#S4.SS3)

**接入推断。** 当前 Splash 在 RoPE 后存 K。完整 KVQuant 会改变投影、缓存读取和 attention 顺序。M5 还需要码本查询、稀疏索引、离群值追加和矩阵计算融合。可以借鉴“少量难量化值保留高精度”，但应先定位真实误差；否则 metadata、稀疏访问和校准成本可能抵消收益。

## UltraQuant：硬件格式思路可借鉴

UltraQuant 以旋转后的码本量化作为质量参考，部署时使用硬件原生 FP4，配合 FP8 query 和 UE8M0 分组尺度。它还讨论非对称 K/V、去掉 QJL 和分组尺度设计。[论文 §2](https://arxiv.org/html/2606.20474#S2)

这些选择有助于减少查表和不规则读取。公开优化依赖 AMD CDNA4 的 scaled MFMA，M5 需要单独的数据格式与 TensorOps 实现。论文明确指出，短上下文、低并发且 FP8 缓存仍能驻留时，所测收益不一定出现。不能把多轮缓存压力场景的加速倍数用于当前单请求 32K 总结。[论文限制](https://arxiv.org/html/2606.20474)

## 对当前任务的边界

上述研究都没有公开验证我们的 Qwen3.5-4B UD-Q4_K_XL、32K 中文总结。论文中相近 PPL、GPQA 或长文检索成绩，都不能替代这一条件。

更合适的顺序是：先定位现有 Q4 的质量或耗时问题，再选择对应技术。融合写入主要解决开销；非对称 K/V 或近期高精度窗口主要提供质量保护；稀疏离群值需要额外格式。当前 INT8 默认仍有价值。本轮不实施完整算法，也不加入 Q2 或 OSCAR 开发。
