# KV 量化路线比较（2026-10-10）

## 当前选择

建议保留当前固定 Hadamard + 标准 Q4_0 路线，默认仍用 INT8。现有证据支持 Q4 作为省内存选项，还不足以支持立即换成 TurboQuant 或 RotorQuant。本文只比较路线，不扩大开发范围。

当前 Splash 的 Q4_0 每 32 元存 18 bytes，包含 F16 scale。Q/K 用 H256，V 用 H64。attention 按 tile 解码到局部 16 KiB BF16 stage，再交给 TensorOps。缓存保留 Q4_0 压缩表示，计算时局部转换，再使用 BF16 TensorOps。[本地格式](../runtime/ops/PagedKv.hpp)、[tile 实现](../runtime/metal/kernels/common/paged_attention_tile.h)

本文读取了侧对话保存的测试产物。数据对应固定构建，不能证明每项最新源码都已验证。本轮只读取资料和已有数据，没有编译、测试、推理或启动服务。

## TurboQuant：论文与第三方实现

**论文算法。** MSE（Mean Squared Error，均方误差）版本先归一化、随机正交旋转，再用非均匀 Lloyd-Max 码本。product 版本把部分位宽分给残差的 QJL（Quantized Johnson-Lindenstrauss，量化随机投影），用于无偏内积估计。QJL 的无偏性不保证每次回答相同，也不要求 V 使用同一套残差算法。[论文 §3](https://arxiv.org/html/2504.19874v1#S3)

论文的 2.5-bit 和 3.5-bit 配置采用混合位宽。离群通道获得更多位数。它们不能与统一 3-bit 或标准 Q4_0 直接等同。[论文 §4.3](https://arxiv.org/html/2504.19874v1#S4.SS3)

**论文结果。** Llama-3.1-8B 的 LongBench 平均分相同，不代表总结分项无损。

| Llama-3.1-8B 配置 | 平均分 | 总结分 | 总结变化 |
|---|---:|---:|---:|
| 全精度 | 50.06 | 26.55 | — |
| TurboQuant 3.5-bit | 50.06 | 26.00 | −0.55 |
| TurboQuant 2.5-bit | 49.44 | 24.80 | −1.75 |

Ministral-7B 的 2.5-bit 平均分为 49.62，基线为 49.89；总结分为 24.91，基线为 26.09。所测模型和任务与我们的 4B、32K 中文总结不同。[论文 Table 1](https://arxiv.org/html/2504.19874v1#S4.SS3)

**第三方代码。** RotorQuant 作者指定的 llama.cpp fork 中，`turbo3` 存 3-bit 码本索引，没有 QJL。默认 `turbo4` 存 4-bit 索引，也没有 QJL。备用分支才用 3-bit 索引加 1-bit QJL。多个分支名称相近，算法并不完全相同。[格式与宏开关](https://github.com/johndpope/llama-cpp-turboquant/blob/feature/planarquant-kv-cache/ggml/src/ggml-common.h#L269)

## 第三方 TurboQuant 的同模型对照

**实现者自测。** QVAC 的公开记录包含 Qwen3.5-4B、Q8 权重。其缓存格式有独立名称。K 使用带残差的 `tbq`，V 使用 `pq`。位宽包含格式开销，不能只看名称里的 3 或 4。

| Qwen3.5-4B 缓存 K/V | 平均位宽 | LongBench 平均分 | RULER |
|---|---:|---:|---:|
| F16/F16 | 16.00 | 37.52 | 96.2 |
| Q4_0/Q4_0 | 4.50 | 37.22 | 97.5 |
| tbq3_0/pq3_0 | 3.75 | 34.97 | 93.7 |
| tbq4_0/pq4_0 | 4.75 | 37.04 | 94.8 |

这个实现的四位组合反而比 Q4_0 占用更多位数。三位组合存储更小，任务分数也更低。NIAH 表中四种配置都为 100%，仍未覆盖其他任务的差异。以上是作者记录，表中没有重复次数和误差条，权重也与我们的 UD-Q4_K_XL 不同。[原始数据](https://github.com/tetherto/qvac-fabric-llm.cpp/blob/master/docs/turboquant-benchmarks.md)

同一记录的 8K 性能测试里，RTX 5090 上 Q4_0/Q4_0 的 prefill 为 F16 的 0.98 倍，tbq4_0/pq4_0 为 0.42 倍。这说明具体残差与解码实现会影响性能，不能用论文单算子的加速倍数代替 M5 Pro 的完整总结耗时。[性能表](https://github.com/tetherto/qvac-fabric-llm.cpp/blob/master/docs/turboquant-benchmarks.md#qwen35-4b-q8_0---8k-context)

## RotorQuant：格式和 Mac 证据

原始 RotorQuant 使用 3D Clifford rotors。作者仓库还包含 PlanarQuant 的 2D 旋转和 IsoQuant 的 4D 旋转。下文的 `planar3/iso3` 数据对应这些派生实现。[作者仓库](https://github.com/scrya-com/rotorquant)

**代码计算。** 重新读取指定 fork 后，3-bit block 仍是 128 元、50 bytes：48-byte 索引和 2-byte norm。4-bit block 为 68 bytes：64-byte 索引、2-byte norm、2-byte 保留字段。每元分别为 3.125 bit 和 4.25 bit。按结构体计算，不采用过期注释中的字节数。[block 定义](https://github.com/johndpope/llama-cpp-turboquant/blob/feature/planarquant-kv-cache/ggml/src/ggml-common.h#L269)

这些格式含 Metal 写入和解码核，也有 256 维 attention 模板。存在 shader 只说明有实现参考，不能证明整条路径正确或适合当前 M5 TensorOps。[Metal 源码](https://github.com/johndpope/llama-cpp-turboquant/blob/feature/planarquant-kv-cache/ggml/src/ggml-metal/ggml-metal.metal#L7083)

**作者自测。** README 的 M4 Mac Mini、Qwen2.5-3B、K-only Planar3 数据为 48.3 tok/s，F16 为 47.4 tok/s。这是约 1.9% 的差异，不能支持端到端提升数倍。同一 README 的 Python 测试中，4-bit IsoQuant PPL 为 9.03，F16 为 7.59；RotorQuant 为 10.03。[原始表](https://github.com/scrya-com/rotorquant/blob/main/README.md#additional-benchmarks)

**实验者自测。** windagency 在 M4 Pro 24 GB 上测 Codestral-22B，context 为 512，20 chunks。F16/F16 的处理速度为 127.74 tok/s，Iso3/Iso3 为 71.42 tok/s，后者有 114 个 graph splits。这是困惑度测试的处理速度，不是生成速度。实验者把退化归因于未融合的 V 解码，不能直接推断 Splash 也有相同问题。[历史实测](https://github.com/windagency/experiments-kv-cache-compression#historical-benchmarks-codestral-22b-johndpopellama-cpp-turboquant-fork)

该实测的 Iso3/F16 占 492 MiB，反而高于 F16/F16 的 448 MiB。deferred 路线在 prefill 保留 F16，所以相同 PPL 不能证明低位宽的长输入质量，长期缓存大小也不能代表 prefill 峰值。作者 README 的“3-bit 双缓存压缩 10.3 倍”与实际字节数矛盾，本文不采用。[实验记录](https://github.com/windagency/experiments-kv-cache-compression)、[作者说明](https://github.com/scrya-com/rotorquant/blob/main/README.md#how-it-works)

## Q4_0 的现有质量依据

**上游作者自测。** llama.cpp #21038 使用 Qwen3.5-4B-Base、F16 权重和 wiki.test.raw，128 chunks。

| 缓存 | PPL |
|---|---:|
| F16 基线 | 8.3266 ± 0.11623 |
| 未旋转 Q4_0 | 8.3475 |
| 旋转 Q4_0 | 8.3448 |

旋转只额外改善 0.0027 PPL。此模型在该测试中，未旋转 Q4 已接近基线。正文未给量化行误差条、4B KLD 或 32K 总结成绩。不能把“接近基线”写成业务无损。[原始 PR](https://github.com/ggml-org/llama.cpp/pull/21038)

## 已有本地结果

**读取既有数据。** 同一份 32K 请求，5 种格式各跑 2 轮。10 次退出码均为 0，前缀命中均为 0，12 项字面事实全部通过。构建固定为 `splash 0dfd1482…` 和 `metallib 836c990c…`，完整 SHA 见[原始记录](../build/five-kv-final/results.json)。

| 格式 | prefill 两轮（秒） | Metal 峰值（GiB） |
|---|---:|---:|
| INT8 | 14.886 / 18.084 | 4.475 |
| BF16 | 15.684 / 18.772 | 4.985 |
| 旋转 Q4_0 | 20.557 / 25.068 | 4.194 |
| 未旋转 Q4_0 | 20.560 / 25.540 | 4.194 |
| q4h | 15.202 / 16.160 | 4.194 |

已直接读取逐次 JSON，核对请求哈希、实际格式、32,768 输入和 prefill token、零前缀命中、12 项字面事实以及资源释放。保存的二进制和 metallib 的 SHA 与报告一致。当前 `build/splash` 与该保存构建不同，表中数据仅代表这个固定版本。硬件记录为 M5 Pro，20 核 GPU、64 GiB 内存。[保存结果](../build/five-kv-final/results.json)

数据来自[汇总记录](../build/five-kv-final/aggregate.json)。标准 Q4 省内存，但这份实现的 prefill 慢于 INT8。旋转和未旋转 Q4 耗时接近，不能把全部开销归因于 Hadamard。q4h 值得继续比较；它与标准 Q4 的峰值相同，说明 raw KV 差异还受分配对齐影响。

两轮的环境耗时有变化，生成 token 数也不同。decode 不能当作同输出速度比较。一个固定样例通过字面事实检查，只证明这个样例达标。一般总结质量是否等同或优于 BF16，仍需更多文档验证。

## 内存和换路线的边界

**代码计算。** 单序列 32,768 个 token，8 个 full-attention 层、4 个 KV head、head dimension 为 256。包含 K 和 V 两份数据、scale 或 norm。表中不含对齐、工作区、GDN 状态和模型权重。

| 格式 | 存储规则 | 原始 KV |
|---|---|---:|
| BF16 | 2 bytes/元 | 1024 MiB |
| 当前 INT8 | 260 bytes/256 元 | 520 MiB |
| 标准 Q4_0 | 18 bytes/32 元 | 288 MiB |
| 自定义 q4h | 132 bytes/256 元 | 264 MiB |
| Planar3/Iso3 | 50 bytes/128 元 | 200 MiB |
| Planar4/Iso4 | 68 bytes/128 元 | 272 MiB |

本地格式见 [PagedKv](../runtime/ops/PagedKv.hpp)，派生格式见 [fork 结构体](https://github.com/johndpope/llama-cpp-turboquant/blob/feature/planarquant-kv-cache/ggml/src/ggml-common.h)。3-bit 比标准 Q4 节省 88 MiB；4-bit 只节省 16 MiB。这些数值是静态计算，不能替代峰值测量。

**选型推断。** 当前 4B、32K 保留 INT8 默认，标准 Q4_0 用于内存紧张的场景。如果事实提取或总结质量下降，先回退 INT8 并定位误差，再比较非均匀码本。只有同一模型、输入和输出要求下，质量通过且峰值更低，才有理由换路线。

上下文升到 128K 时，3-bit 相对 Q4 可多省 352 MiB。到那时再考虑低位宽更有实际收益。若内存瓶颈来自模型权重、GDN 快照或工作区，换 KV 算法未必解决问题。本轮不加入 Q2 或其他算法开发。
