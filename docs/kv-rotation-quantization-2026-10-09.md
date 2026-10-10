# Current scope: rotated standard Q4_0 (2026-10-09)

This section supersedes the earlier q4h-only and INT2 research notes below.
The target remains the local Qwen3.5-4B UD-Q4_K_XL GGUF, with exactly 32768
input tokens followed by a summary. Development stays in the managed
qwen35-4b-optimization worktree; the original ghost checkout is not modified.
No Q2, OSCAR calibration, activation collection, or calibration file handling
is implemented in this delivery. INT8 is still the default.

## Implemented choices

| Option | Cache representation | Rotation |
|---|---|---|
| `int8` | Existing signed INT8, FP32 scale per token/head | none |
| `bf16` | Existing BF16 | none |
| `q4_0` | Standard 18-byte GGML Q4_0 blocks, F16 scale per 32 elements | Q/K H256 after norm/RoPE; V block H64 |
| `q4_0_unrotated` | Identical standard Q4_0 blocks | none; acceptance control |
| `q4h` | Existing experimental signed INT4, FP32 scale per 256 elements | Q/K H256; V block H64 |

Q4_0's `qs[j]` packs element j in the low nibble and element j+16 in the high
nibble. The signed scale comes from the first absolute maximum / -8, with
GGML's offset-eight rounding convention. K and V are token-major; each head
row has eight standard blocks. Scales are inline, so no separate FP32 scale
region is allocated. Struct size, row packing, and whole-page byte counts
have compile-time assertions and scalar reference checks.

Grouped scales cannot be applied after one full QK product like INT8's
per-head scale. This first path therefore dequantizes each page's one-head
32x256 K tile into **16 KiB threadgroup BF16**, runs TensorOps QK, then
reuses exactly that stage for V and TensorOps PV. The optimized conversion
loads each payload byte and scale once for its two dimensions. V stays
token-major in the local stage; TensorOps consumes this layout directly,
without a software transpose.
Unwritten/masked page slots are zeroed in the local conversion stage, so
NaN scale bytes from an unused slot cannot enter PV as zero times NaN.
The entire cache stays compressed. No native INT4 tensor interpretation of
an interleaved GGML block is attempted. This uses the floating TensorOps
path and does not imply any promised INT4 arithmetic throughput.

The maximum scratch of these compiled shapes (64 fused rows) is below 32 KiB:
16 KiB conversion stage + 12 KiB scores/probabilities + row statistics. Stage
size is fixed regardless of sequence length. The existing split count,
causal masking, FP32 reduction order, page ownership and rejected-row
visibility remain in force.

The existing fixed H256/H64 preparation and H64 inverse reduction are shared
with q4h. The inverse is completed before BF16 output rounding and the
sigmoid gate. Unrotated Q4_0 bypasses both transforms. Status reports both
the exact quantizer and rotation identity; persistent cache names include
the standard layout/rotation version. Neither standard variant aliases q4h
or INT8 pages/state caches.

## Memory and quality boundaries

For this 4B at 32K stored tokens, raw K+V and their scales are BF16 **1024 MiB**,
INT8 **520 MiB**, Q4_0 **288 MiB**, q4h **264 MiB**. Q4_0 uses 24 MiB more than q4h
for its finer groups. These counts exclude weights, GDN state, temporary
workspaces, generated tokens and allocation slack; peak memory must be
measured. Standard Q4_0 is 4.5 bits/element, not 4.

The published Qwen3.5-4B-Base/F16-weight perplexity data does not prove that
our UD-Q4_K_XL model is lossless on 32K summaries. It does not validate q4h.
Rotation is an orthogonal transform before rounding/quantization; it is not
a guarantee that every vector's quantization error improves. Neither INT8
nor Q4_0 is guaranteed to be faster. All five formats passed two runs of the
same synthetic 32K summary on the current implementation. Every run retained
12 specified names, dates, amounts and counts across the document. This is a
fixture result, not a general lossless-quality claim.

## Acceptance commands and current evidence

1. `build/engine-tests/paged-attention-plan build/splash.metallib` includes
   the rotated/plain standard variants, dense H256/H64 references, full
   extent-byte store comparisons, causal scalar attention, mixed extents,
   B1-B4 verify, page boundaries and canaries. Existing numerical bounds
   are unchanged. `--long` adds 131K/260K histories.
   Page-storage and page-tier checks also include opaque Q4_0 page copies,
   extent admission/accounting, disk round trips and persistent adoption for
   the 4B layout, in both rotation modes. Q4H also has volatile and persistent
   page-tier round trips. The full `make check-native-metal` gate passed;
   the optional `--long` extension was not required for the 32K objective.
2. `build/engine-tests/attention-sweep build/splash.metallib --shapes 4b
   --histories 32768 --lanes 1 --kv-format q4_0` is an isolated store/read/
   attention/reduce timing entry. Run the unrotated control with its own
   format option. This sweep excludes QKV norm/RoPE/forward rotation and
   cannot stand in for a full-prefill comparison.
3. All model launchers use `--kv-format VALUE` (also `--kv-format=VALUE`)
   through `dev/tools/launcher_kv.sh`, defaulting to INT8. For the target4B:
   `./qwen3.54bsplash.sh --kv-format q4_0`; the acceptance control is
   `./qwen3.54bsplash.sh --kv-format q4_0_unrotated`. No KV environment
   variable is used. The earlier baseline supports only INT8/BF16 and rejects
   unsupported low-bit formats before model preparation. These commands are
   available for local use. The recorded matrix starts the server module
   directly with these same runtime options. Launcher syntax is covered by
   the Python checks.
4. `python -m dev.tests.summary_32k_real --kv-format q4_0 --output REPORT`
   checks the actual served identity, exact 32768 prompt tokens, cold-cache
   processing, release of active resources, names/dates/amounts/counts at
   start/middle/end, full prefill metrics and native before/after peak
   memory status. Reuse the exact request with `--request-from REPORT` for
   all formats. Preserve failures and full responses. The fixture is
   synthetic, so passing it alone does not establish general summarization
   quality; representative real documents should be checked later too.
5. Native backend-benchmark accepts both standard format names for complete
   runtime profiling. The recorded summary matrix supplies the full-runtime timings.

## Runtime validation and tuning on 2026-10-09

The user authorized tests for all five formats. The final source, native CPU,
complete Metal, Python and sanitizer checks passed. No commit or push was made.
The frozen final build and its source hashes are in
`build/five-kv-final-artifacts/source.json`; the ten final summary runs and
real process exit codes are in `build/five-kv-final/results.json`.

Q4H initially failed the strict byte oracle: a serialized FP32 scale differed
by one bit because of Metal fast division. Q4H now uses precise division for
the scale and quantization code. The scalar byte check and complete Metal
gate passed with their original acceptance bounds. The failure is preserved
in `build/five-kv-q4h-red.log`.

Standard Q4_0 now unpacks both nibbles together and retains token-major V in
the local conversion stage. A paired 11-round attention sweep at 32768 history
tokens plus 2048 prefill rows measured 198.3 ms versus 281.5 ms on the original
path, with identical output bytes. This is a one-layer kernel measurement,
not the full 32K request. The full-model A-B-B-A comparison then ran the
original and optimized frozen builds in the same time window. Each row below
averages two original and two optimized requests, with identical inputs and
identical generated output within the format.

| Format | Original prefill s | Optimized prefill s | Prefill time reduction | Original decode tokens/s | Optimized decode tokens/s |
|---|---:|---:|---:|---:|---:|
| Q4_0, rotated | 28.34 | 23.73 | 16.27% | 29.58 | 31.80 |
| Q4_0, unrotated | 29.48 | 24.71 | 16.19% | 28.20 | 31.74 |

Decode throughput increased 7.51% and 12.53%, respectively. All eight paired
requests retained the twelve facts and released their active resources.
The raw timings and exit codes are in `build/five-kv-q4-abba/results.json`.
Including the initial and final five-format matrices, 28 complete 32K summary
requests passed.

The 256-row GGUF tile, eight-simdgroup tile and INT8-to-BF16 local staging
candidates were slower and removed. Two-step GGUF payload prefetch did not
improve the main projection shapes consistently and was removed too.
Default INT8 retains its original attention and GGUF GEMM paths.

The final five-format matrix uses the same request hash, 32768 input tokens,
zero prefix-cache matches, temperature zero and ordinary autoregressive
decoding. Each format has two fresh-server runs in opposite ordering. The
measured ranges below include system-load variation; their ordering alone
does not prove an optimization gain.

| Format | 32K prefill seconds | Decode tokens/s | Metal allocation peak GiB |
|---|---:|---:|---:|
| INT8 | 14.89–18.08 | 37.84–43.93 | 4.475 |
| BF16 | 15.68–18.77 | 33.55–41.25 | 4.985 |
| Q4_0, rotated | 20.56–25.07 | 32.39–37.18 | 4.194 |
| Q4_0, unrotated | 20.56–25.54 | 32.29–39.49 | 4.194 |
| Q4H | 15.20–16.16 | 44.08–47.26 | 4.194 |

These are Metal allocation peaks, not whole-process resident-memory readings.
All runs retained the twelve fixture facts and released active lanes/pages.
The raw requests, outputs, format identities and memory statuses remain in
the individual reports. The initial baseline INT8 run's slow decode result
was preserved rather than silently discarded.

The earlier static-only audit `build/q4_0-development-status.json` remains a
historical record. Current validation is tracked in
`build/five-kv-validation-report.json`.


## Earlier research and development record (superseded where scope differs)

# KV 旋转量化接入判断（2026-10-09）

本轮仅调查论文、上游状态和本地接口；未运行测试、推理或性能测试，未修改生产代码。目标是 Qwen3.5-4B 的 32768-token 输入总结。上游源码观察点为 llama.cpp `86a283532072722c5f3363d37d59a874d09fa99b`；本地为 `codex/qwen35-4b-optimization` worktree 的未提交适配代码。

## 判断

可以结合，但优先做 **llama.cpp 主线的 Hadamard 旋转 + uniform 4-bit 缓存**的独立候选。TurboQuant 和 RotorQuant 是其他量化路线；不能将三个算法叠加名称后假定同时获得各自宣传收益。现有 Splash 默认已经是 INT8 KV，并非所有缓存都是 BF16；4-bit 对当前默认的收益应按 INT8 计算。[本地 PagedKv](../runtime/ops/PagedKv.hpp)、[服务选项](../server/serve_options.py)

### 方法与证据

| 路线 | 核心差别 | 可用证据与适用边界 |
|---|---|---|
| llama.cpp 主线 | 正交 Hadamard 抑制离群值，继续使用现有 q4_0 等量化格式；Q/K 同基底，V 旋转后在 attention 输出逆旋转 | 已合入，具备 Metal 实现；不是完整 TurboQuant。质量随模型和任务变化。[PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038) |
| TurboQuant | MSE 版本：随机正交旋转、归一化向量、非均匀 Lloyd-Max codebook；product 版本再用残差的 1-bit QJL 消除内积估计偏差 | QJL 是内积估计路线，不应默认给 V 也加相同残差。论文 LongBench 的 3.5-bit 混合位宽和离群通道策略，不等于简单 uniform Q4。论文 A100、Llama/Ministral 结果不能直接代表本机 Qwen4B。[论文 §3–4](https://arxiv.org/html/2504.19874v1) |
| RotorQuant | 名称核实为 John D. Pope / scrya-com 项目；原论文以 3D Clifford rotor 小块替代全维旋转。仓库目前重点还包含 2D PlanarQuant / 4D IsoQuant | 作者有 Metal 研究 shader，llama.cpp 集成指向独立 fork。论文的量化核加速相对 dense BLAS，不能与主线 FWHT 等价比较；未证明当前 Qwen3.5-4B 32K 总结质量。[作者论文](https://www.scrya.com/rotorquant.pdf)、[作者仓库](https://github.com/scrya-com/rotorquant) |

TurboQuant 论文所称 quality-neutral，是所测任务的结果接近；其向量量化仍有误差。理论的无偏内积也不等于每次 logits、softmax 或最终回答相同。[TurboQuant 论文](https://arxiv.org/abs/2504.19874)

RotorQuant README 的 symmetric 3-bit、10.3× 对 FP16 压缩说法不宜采用：仅 3-bit payload 对 16-bit 基线的上限是 5.33×，还未计 norm。其指定 fork 的 `block_planar3_0` / `block_iso3_0` 实际为 128 元的 48-byte payload + 2-byte norm，即 3.125 bit/元、约 5.12×；4-bit 分支是 68 bytes/128，即 4.25 bit/元、约 3.76×。这来自结构体与 static_assert，不采用旁边已经过期的注释字节数。该 README 还使用 prefill F16、随后压缩的 deferred 路线，长期占用减少不保证 prefill 峰值同样降低。[fork 格式源码](https://github.com/johndpope/llama-cpp-turboquant/blob/feature/planarquant-kv-cache/ggml/src/ggml-common.h)、[README](https://github.com/scrya-com/rotorquant/blob/main/README.md)

## “最近合并、Q4 几乎无损”究竟是什么

| 上游事项 | 核实状态 | commit |
|---|---|---|
| [#21038 rotate activations](https://github.com/ggml-org/llama.cpp/pull/21038) | 2026-04-01 merged；质量路线的主改动 | `744c0c7310aad90e99a29c5739e4ee317fb6a748` |
| [#22631 CPU FWHT](https://github.com/ggml-org/llama.cpp/pull/22631) | 2026-05-05 merged；把 dense Hadamard 乘法换成快速蝶形算法 | `a817a22bc66465ae0c08a69fc1571b0ec69570ad` |
| [#25924 Metal FWHT](https://github.com/ggml-org/llama.cpp/pull/25924) | 2026-07-28 merged；实际 Metal GPU 蝶形核 | `f95de9776b5b90dd993f36d2bd66a3eee21c887f` |
| [#29094 Metal F16 输入](https://github.com/ggml-org/llama.cpp/pull/29094) | 2026-09-20 merged；避免额外输入转换 | `9a9f939b8060426c68342fd802ccb515d0f94c1a` |
| [#29602 Metal FWHT 优化](https://github.com/ggml-org/llama.cpp/pull/29602) | 2026-09-29 merged；调度优化，不是新的量化质量算法 | `18bbc46b4697e8623e6f0a8d22e789d387b91617` |
| [#28498 旋转缓存恢复身份](https://github.com/ggml-org/llama.cpp/pull/28498) | 2026-10-05 merged；恢复时拒绝旋转尺寸不匹配 | `210791069bf5460bef3c09b982ee6f0aa9967404` |
| [#21089 TBQ3_0 / TBQ4_0](https://github.com/ggml-org/llama.cpp/pull/21089) | closed、**unmerged**；不能称完整 TurboQuant 已入主线 | 无 merge commit |

FWHT 使用既有 `MUL_MAT` 上的 `GGML_HINT_SRC0_IS_HADAMARD` 分流，没有新增 ggml 运算枚举。当前标准矩阵确定性、归一化且 H²=I；随机 DHD 是不同实验。只有各自对应 K/V 类型量化且维度满足条件才旋转，K/V 分别决定。对于 256 维 Qwen4B：**K/Q 是 H256，V 是四个 H64 小块**。所有 full-attention 层沿用相同方式；GDN 不应用此 KV 变换。[缓存选择源码](https://github.com/ggml-org/llama.cpp/blob/86a283532072722c5f3363d37d59a874d09fa99b/src/llama-kv-cache.cpp#L315)、[计算图](https://github.com/ggml-org/llama.cpp/blob/86a283532072722c5f3363d37d59a874d09fa99b/src/llama-graph.cpp#L2991)

Metal 已包含 `dk256/dv256` 的 q4_0 flash-attention 模板，说明当前头维度具备上游代码参考；不代表 Splash 的 MPP 核可以直接替换。[Metal q4_0 模板](https://github.com/ggml-org/llama.cpp/blob/86a283532072722c5f3363d37d59a874d09fa99b/ggml/src/ggml-metal/kernels/fa_vec_q4_0.metal)

PR #21038 正文给出的 **Qwen3.5-4B-Base、F16 模型权重、wiki.test.raw / 128 chunks** 数据：

| 缓存条件 | PPL | 对 F16 基线变化（计算值） |
|---|---:|---:|
| F16 基线 | 8.3266 ± 0.11623 | — |
| q4_0、此前未旋转 | 8.3475 | +0.251% |
| q4_0、PR 旋转 | 8.3448 | +0.219% |

同一公开表中旋转额外只改善 0.0027 PPL，不能把这个模型 Q4 接近基线全部归功于旋转。正文 `type` 单列按同型 K/V 解释，但只明确展示基线 `-ctk f16 -ctv f16` 命令，没有逐行重复量化参数；量化行没有误差条或 4B KLD。这里不能推断统计显著或 32K 指令总结无损。[原始 PR 正文](https://github.com/ggml-org/llama.cpp/pull/21038)

作为更明确的同型 K/V 补充：另一作者对 Qwen3.5-9B 的重跑表列出 q4_0/q4_0 的 PPL `8.208411 ± 0.055610`、mean KLD `0.004782 ± 0.000194`；F16/F16 PPL `8.195983 ± 0.055532`。这也说明 PPL 近似并不代表概率分布完全一致，且它是 9B 的证据。[原始表评论](https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570)

## 内存：按完整 K 和 V 两份计算

Qwen4B 有 8 个 full-attention 层、每层 4 KV heads、每头 256 元。其余 24 层为 GDN。下表只算一个序列、32768 个已存 token 的 raw KV payload + scale，不含 extent 对齐、预分配、工作区、GDN、模型权重、生成空间或其他服务。

公式：`2 × layers × KV_heads × head_dim × tokens × 每元素实际字节数`。

| 格式 | 实际存储 | 4B / 32K | 4B / 128K |
|---|---|---:|---:|
| BF16 KV | 2 bytes/元 | 1024 MiB | 4096 MiB |
| Splash 现有 INT8 | 256-byte payload + 4-byte F32 scale / token / head / K或V | 520 MiB | 2080 MiB |
| GGML q4_0 | 每 32 元 16-byte nibbles + 2-byte F16 scale；4.5 bit/元 | 288 MiB | 1152 MiB |
| 自定义 uniform4 候选 | 每 256 元 128-byte nibbles + 4-byte F32 scale；4.125 bit/元 | 264 MiB | 1056 MiB |

因此 q4_0 相对 BF16 节省 736 MiB，相对**当前默认 INT8**节省 232 MiB；不能说模型总内存节省 3.56×。自定义每头单 scale 会牺牲局部动态范围，质量待验证，也不是 GGML q4_0 或完整 TurboQuant。[本地布局](../runtime/ops/PagedKv.hpp)、[GGML q4_0 结构](https://github.com/ggml-org/llama.cpp/blob/86a283532072722c5f3363d37d59a874d09fa99b/ggml/src/ggml-common.h#L194)

项目 27B 布局有 16 个 full-attention 层、4 KV heads、256 维头；128K 时同样的 raw KV 计算为 BF16 **8192 MiB**、现有 INT8 **4160 MiB**、GGML q4_0 **2304 MiB**。大模型长上下文的绝对节省更大。[本地 27B 布局](../runtime/model/Qwen3_8.hpp)

GDN recurrent state 是 FP32、卷积历史为 BF16；当前 GGUF 权重混合量化。即使 KV 降到 Q4，也不能将全部 buffer 一起改成 Q4。GDN 固定 live state 不随输入 token 线性增长，前缀快照、并发池和其他 scratch 仍会增加使用量。[本地 StateLayout](../runtime/model/StateLayout.hpp)

## Splash 接入的具体约束

1. **Q/K 顺序**：先完成 QK RMSNorm 与 RoPE，再施加一致的 Hadamard，再量化存 K、用旋转 Q 计算。Hadamard 不是 RoPE；放在 RoPE 前通常不等价。上游 Qwen35 就是此顺序。[Qwen35 源码](https://github.com/ggml-org/llama.cpp/blob/86a283532072722c5f3363d37d59a874d09fa99b/src/models/qwen35.cpp#L277)、[本地投影](../runtime/ops/PagedAttention.cpp)
2. **V 与门控**：PV 输出必须逆旋转到原坐标，再做 Qwen 的逐元素 sigmoid gate。此 gate 与一般 rotation 不交换，不能简单将逆旋转折进 gate 后的 output projection。[上游 PV 后逆旋转](https://github.com/ggml-org/llama.cpp/blob/86a283532072722c5f3363d37d59a874d09fa99b/src/llama-graph.cpp#L3048)、[上游 sigmoid gate](https://github.com/ggml-org/llama.cpp/blob/86a283532072722c5f3363d37d59a874d09fa99b/src/models/qwen35.cpp#L329)
3. **attention 消费格式**：现 KV MPP `matmul2d` 路径直接读 INT8 操作数，scale 后处理。标准整数 packed4 有原生 TensorOps 输入路径可考虑，但不能把 GGML q4_0 的完整 block 字节直接当成 Metal tensor；仍须匹配打包、尺度、stride、alignment 和 operand 组合。非均匀 codebook 则需要额外 lookup/dequantize。不能原生消费时，在读取 tile 时转换到硬件支持的局部格式，保持页式访问；全 cache 恢复 BF16 会抵消内存目的。[MPP tile](../runtime/metal/kernels/common/paged_attention_tile.h)、[本地原生 uint4b 矩阵路径](../runtime/metal/kernels/common/q4_mpp_tiles.h)
4. **尺度寻址**：本地 K token-major、V dimension-major、per-token/head scale。GGML q4_0 的 32 元 group scale 需要新的 ABI 和寻址。每头 F32-scale uniform4 更靠近现有布局，但收益/误差须独立评价。[PagedKv storage identity](../runtime/ops/PagedKv.hpp)
5. **状态一致**：前缀复用、snapshot、persist cache 需记录 quantizer/layout、rotation 大小、版本；若随机化则加 seed/符号身份。旧 namespace 只识别现有格式和几何，不能读取不同基底的同尺寸数据。上游已专门修复这个错误。[PR #28498](https://github.com/ggml-org/llama.cpp/pull/28498)
6. **drafter 后续**：target 的 speculative verify 写缓存必须使用同样的旋转与量化格式。本地 DFlash 有独立的 BF16 ring，可先保持原格式；不应将它描述为已经向 target KV 注入数据。若未来共享或注入 target KV，才须遵循 target 的旋转基底。上游针对其注入路径有修复可参考。[本地 draft ring](../runtime/model/DFlashDraft.hpp)、[PR #25823](https://github.com/ggml-org/llama.cpp/pull/25823)

## 候选与后续验收

第一阶段保留现有 INT8 和 BF16 对照，增加 uniform4/Hadamard 的独立候选；先沿主线 H256(Q/K)、H64(V)作为可解释基线，再决定每头或 group32 scale 布局。TurboQuant codebook/QJL、Rotor/Planar/Iso 是后续对照候选，按实际误差和峰值决定，不从宣传倍数推导本机性能。

用户通知测试后才验证：rotation/inverse 数值 oracle、量化边界与页跨越、prefill/ordinary decode/snapshot 恢复；BF16、INT8、Q4 三路同一 GGUF 和相同输入比 logits/KLD，再测 32K 多位置事实提取与真实总结质量、TTFT/峰值内存/吞吐。仅一个 needle 用例或 PPL 接近不足以认定业务无损。若采用 deferred-F16 prefill，应单独列峰值并确认能够完成32K入参。

## 许可与复用边界

llama.cpp 明确 MIT，可参考或复用时保留许可和署名。[LICENSE](https://github.com/ggml-org/llama.cpp/blob/86a283532072722c5f3363d37d59a874d09fa99b/LICENSE) RotorQuant README 声明 MIT，当前根目录 `LICENSE` 读取为不存在；复制具体实现前需确认对应文件/fork实际许可。TurboQuant arXiv 论文是 CC BY 4.0，论文许可不自动证明任意第三方实现的代码许可。[RotorQuant README](https://github.com/scrya-com/rotorquant/blob/main/README.md)、[论文许可页面](https://arxiv.org/html/2504.19874v1)

## 补充：M5 GPU 加速与存储位宽

M5 GPU Neural Accelerator 并非 INT8 专属。Apple TensorOps 支持浮点以及整数低位宽输入；macOS 26.4 添加 4/8-bit 整数 tensor，macOS 27 添加 2-bit 整数等格式。API 接受某个类型，不等于已证明特定混合精度、形状的底层指令或吞吐；实际利用率需要 GPU profile。[Apple M5 技术讲座](https://developer.apple.com/videos/play/tech-talks/111432/)、[WWDC26 TensorOps](https://developer.apple.com/videos/play/wwdc2026/330/)

本轮只读查询显示本机 macOS 27.0.1；SDK MTLTensor.h 标记 Int4 为 macOS26.4、Int2 为 macOS27.0。MPPTensorOpsMatMul2d.h 列出 bfloat × int8_t / int4b_format / int2b_format → float 等 operand 组合。本项目 q4_mpp_tiles.h 已使用 bfloat 激活与 uint4b_format 权重直接进行 TensorOps；KV 路径则是 BF16 Q/概率 × INT8 K/V，并不是已量化 Q/概率的纯 INT8 GEMM。

后续 KV 候选应先检查标准 integer4 能否匹配原生 TensorOps 布局与尺度，再选择必要的 tile 转换。TurboQuant 非均匀码本的 4-bit index 不能直接按整数4值进行 matmul。3-bit 等自定义存储格式可局部转换为支持的类型，仍使用 TensorOps，但转换开销可能抵消部分带宽收益。未运行编译、测试、推理或性能 profile。

## First port: experimental q4h (compile only)

After the user withdrew test permission, work continued only on source, builds
and static checks. The new option is `--kv-format q4h`; INT8 remains the default.
This ports deterministic H256 for Q/K and block H64 for V, with native integer
INT4 TensorOps. It is neither GGML q4_0 nor the full TurboQuant/RotorQuant method.

The quantizer stores 256 signed four-bit values in [-7,7] and one FP32 scale per
(token, head, tensor). Adjacent dimensions own the low/high nibbles of one byte.
K and V are both token-major: each store owns complete bytes, avoiding concurrent
nibble writes across tokens. A V operand has a 256-element row/stride. QK uses a
transposed right operand; PV uses a non-transposed right operand. MPP consumes
`int4b_format` directly; no full-page expansion is required. Per-head scales use
the existing score/probability epilogues.

Q/K rotation follows QK normalization and RoPE, then rounds to BF16 staging before
quantizing K. V is transformed in blocks of 64. The inverse H64 is fused into the
final FP32 split reduction before BF16 rounding and the sigmoid gate. Weights,
GDN and the independent draft ring retain their existing paths. The persistent
cache namespace includes the q4h_v1 quantizer/layout/rotation identity.

Per-head scales are coarser than q4_0's group32 scales. The public q4_0 quality
results do not establish q4h quality. At 4B/32K, raw payload and scales total
264 MiB: 760 MiB below BF16 and 256 MiB below existing INT8. Physical allocations,
generation tokens and scratch are extra. Quantization step is max(abs)/7, with
coordinate rounding error bounded approximately by max(abs)/14. Hadamard can
improve or worsen that maximum; there is no universal lossless guarantee.

Future test entries, **not invoked after the pause**:

- `make benchmark-kv-tensors`: isolated native BF16/INT8/INT4/INT2 operand probe,
  compiled with Metal4.1 / macOS27. Logical M=4 uses physical M=8 padding; both
  are reported. Timing operands are identical representable integer data. This
  does not measure quantization quality, rotation, scales or complete attention.
- `make benchmark-kv-tensors KV_TENSOR_ARGS=--correctness-only`: scalar FP64
  references and signed encoding boundaries, without timing rounds.
- `build/engine-tests/paged-attention-plan build/splash.metallib`: extended q4h
  plans, per-byte store preservation, scalar causal attention/inverse H64,
  multiple extents/lanes, and a dense matrix H256/H64 projection reference.
  Existing attention error thresholds are unchanged. `--long` adds long history.
- `./qwen3.54bsplash.sh --kv-format q4h`: the dedicated 4B candidate.
  The older baseline does not implement q4h and the launcher explicitly rejects
  that combination. Native backend-benchmark also accepts the new KV option.

Production remains Metal4.0 / macOS26.4. Correctness, actual peak memory, GPU
accelerator utilization, latency and 32K summarization quality are all pending.
The BF16/INT8 check completed before this port is not validation of this source.

## INT2 quality evidence added on 2026-10-09

The OSCAR authors publish a directly relevant **Qwen3.5-4B** preview result on
GPQA (198 questions): BF16 baseline **75.25%**, OSCAR **74.75%**, a **0.50
percentage-point** drop. This is approximately one question difference and
99.34% relative score retention (arithmetic from rounded published scores),
not a general model-accuracy guarantee. The preview table does not report
repeat count or confidence intervals. Qwen3.5 support is on the experimental
SGLang `zhongzhu/hybrid-model` branch with `SGLANG_LLOYD_MAX=1`, not the same
implementation as the current Splash q4h prototype or a validated Metal4B path.
Source: [author README](https://github.com/FutureMLS-Lab/OSCAR/blob/main/README.md#main-results).

OSCAR's separate full-attention Qwen3-4B-Thinking-2507 experiment reports a
five-task reasoning/coding mean of **75.64** for BF16 versus **71.86** for
OSCAR (minus **3.78 points**, 95.00% relative score retention). In that same
setting, naive INT2 averages **0.00** and Hadamard-only QuaRot-INT2 **1.40**.
Qwen configurations use five seeds and a **32768-token maximum generation
length**; this is not a 32768-token input-summary test. Effective OSCAR BPE is
**2.28 at 128K context**, including its small BF16 sink/recent windows. These
are author-reported measurements, not an independent local reproduction.
Source: [OSCAR paper, experiment setup and Table 2](https://arxiv.org/html/2605.17757).

The accuracy strategy is attention-aware offline calibration, separate K/V
rotations, clipping, and mixed-precision protection of sink/recent tokens.
It cannot be reproduced merely by changing q4h's per-head max/7 quantizer to
a two-bit integer range. The Qwen3.5 preview also uses Lloyd-Max quantization.
Native INT2 matmul feasibility does not establish end-to-end INT2 KV quality.

Older independent algorithm evidence: RotateKV reports Llama-3-8B GSM8K
51.33% FP16 versus 50.49% RotateKV (minus 0.84 points); its average bit width
is 2.25 rather than every value being exactly two bits. Its calibration,
reordering, and sink protection differ from our current implementation.
Source: [RotateKV Table 5](https://arxiv.org/html/2501.16383v2#S5.T5).

For the user's summarization objective, averages can conceal task-specific
losses: TurboQuant's table labeled 2.5-bit for Llama-3.1-8B reports LongBench
average **50.06 to 49.44**, while the summarization column is **26.55 to
24.80**. These are benchmark scores, not a percentage of all facts retained.
Source: [TurboQuant Table 1](https://arxiv.org/html/2504.19874v1#S4.T1).

No inference, GPU test, calibration, or benchmark was run for this evidence
update. The local UD-Q4_K_XL model's INT2 summarization accuracy remains unknown.

## Archived q4h-only delivery scope (superseded)

The user narrowed this work to **q4h only**. No q4_0 cache path, calibrated Q2
cache, precision-window implementation, or Q2 calibration loader/tool was added
in the subsequently cancelled planning step. INT8 remains the default. The
existing optional INT2 TensorOps probe is research tooling, not an INT2 cache.
The latest q4h production and native-check sources were rebuilt without executing
any tests. Runtime validation remains deferred until the user explicitly permits it.

## Shared-worktree launcher handoff incorporated

Read `build/launcher-kv-main-thread-handoff.md` and checked the reported
launcher/parser sources. Their KV selection is consistently command-line
`--kv-format`; no legacy KV environment-variable usage remains in these notes.
The shared parser defaults to INT8 and validates all five current formats.
The4B baseline restriction is checked before preparation. The side conversation's
static-only results and source hashes are retained in
`build/launcher-kv-options-status.json`; the main final audit covers these
shared source changes too. No launcher, service, test or benchmark was run.
