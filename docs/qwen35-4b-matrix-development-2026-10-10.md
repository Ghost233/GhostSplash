# Qwen3.5-4B 六维测试矩阵

本轮开发测试入口、编译变体和正确性检查。用户要求隔壁性能测试结束后再测，因此当前不运行 32K 性能矩阵，也不运行 ANE 自动校准。

## 组合与固定条件

| 维度 | 值 | 含义 |
| --- | --- | --- |
| KV 量化 | INT8、BF16、Q4_0 Hadamard、Q4_0 原始通道、Q4H | 目标模型的 KV 存储格式。目标权重固定为 UD-Q4_K_XL，drafter 权重固定为同一份 affine Q4。 |
| 实际输出长度 | 512、2048 | 强制生成到指定长度。现有 IgnoreEndOfSequence 选项会屏蔽 stop token。 |
| ANE | off、on | on 固定 FFN share=0.5，minimum rows=512。禁用自动性能校准和 breaker。 |
| drafter | off、block=2、4、8、16 | block 含一个 anchor，实际每轮 proposal 数分别为 1、3、7、15。使用 Standard DFlash。 |
| prefill 历史 splits | 1～16 的每个整数 | 只改变 4B 的 32 行尾块历史 attention 分片数，2048 行主体与 query tile 大小保持一致。 |
| 请求并发 | 1、2、4 | 同一 Engine 同时提交独立请求，记录实际 decode 批宽。 |

组合数为 `5×2×2×5×16×3=4800`。每组重复 3 次，共 14400 批、33600 个请求。drafter off 只有一个状态，不按 block 重复。

输入固定为 32768 tokens，上下文上限 36864，Metal 内存预算 32 GiB，greedy，冷缓存，关闭磁盘缓存。四条输入复用同一任务与 12 条事实，但第一个 32-token page 不同；共同 BOS 合法。每次创建全新进程、Runtime 和 Cache，实测 cached input 必须为零。未来性能模式在开始计时前统一执行一次 2048 行 prefill 和一次所请求批宽的 decode warmup，准备 pipeline 与 FFN 缓存；这些直接 Runtime 请求不向 Engine 发布共享前缀。正确性模式跳过 warmup。

默认只生成计划。完整性能入口必须显式传入 `--run-performance`，本轮不调用它。短正确性用例采用 64-token 输入与 32-token 输出，以实际触发 32 行尾块分片路径；结果不输出时长、吞吐或速度排名。

## 实际生效检查

block=2、4、8、16 使用分别编译的 host 与 metallib，产物放在 `build/qwen4b-matrix-variants/block-B`。原有默认 block=8 保持不变。Standard drafter 接受这些变体，DFlash2 只接受原 block=8；不兼容的模型必须明确拒绝。

矩阵入口使用生产 Engine，而不是在单 lane 上循环模拟并发。批宽允许随着请求结束下降，但必须实际出现请求的完整宽度；始终没有达到指定宽度的运行不能作为该并发的成功数据。

ANE on 必须报告实际 share、阈值和 evaluations。初始化退回 GPU、执行期间停止或没有实际 ANE 调用，都记录原因；不能将其计作 ANE on 的成功对照。最低 512 行意味着 32 行尾块仍由 GPU 执行，ANE 主要参与长 prefill 主体。

每条记录保留源码、权重、输入和输出身份，真实命令数组、输出 token IDs、冷缓存计数、批宽、Metal 分配峰值、draft proposal 与 acceptance、事实检查，以及首个 EOS 位置和数量。固定输出选项屏蔽 EOS，因此这些 EOS 观测只验证选项实现，不能替代自然终止长度测试。

测试专用 attention 对象覆盖 32 行尾块的 splits，生产服务仍使用原策略。生成器对生产源中的预期策略做严格匹配；策略变化后必须重新审核，不能静默生成陈旧覆盖。

## 测量口径

性能模式的单请求 prefill 时长采用提交到 prompt 完成事件的延迟，含排队和调度；TTFT 另记为提交到首个输出的延迟。decode 时长从 prompt 完成到请求完成，包含首批草稿输出，避免不同 block 的首次批量输出数量影响分母。聚合 prefill/decode 时长分别累加 Engine 按 work kind 报告的 cycle，整批总时长为从提交到全部请求完成的 elapsed。报告这些口径，避免将请求延迟与 GPU kernel 时间混用。

表格同时给出秒与 tokens/s。单请求用该请求的 token 数除以阶段时长；聚合用本批的总 token 数除以对应 Engine 阶段累积时长。整批总吞吐另以总 elapsed 为分母。内存峰值是同一进程的共享 Metal 分配峰值，不是每个 lane 独占内存，也不是进程 RSS。

每次编译后由 CPU stamp 记录真实源码与二进制、metallib 身份。缺失产物、源码变化或 artifact 校验失败必须标记 unsupported。超时、错误输出、事实缺失和未达实际输出长度不能计作成功测量。

## 已完成的 ANE 正确性证据

旧 metallib 的 5120 通道用例通过，2560 通道用例失败，复现最后 512 通道未覆盖。修复后两种宽度均通过 CPU oracle、INT8 packing、零行和两端 guards。

4B 形状的两层手工 split（hidden=2560、intermediate=9216、share=0.5、minimum=512）通过现有数值 verifier，检查全部已编译 chunk 函数。该检查没有运行 startup calibration 或速度对照。结果保存在 `build/ane-4b-kernels-red.json`、`build/ane-4b-kernels-green.json` 和 `build/ane-4b-manual-numerical.json`。

这些证据验证数值与边界，不说明 ANE 的速度收益。生产 launcher 仍禁用 ANE；后续按矩阵实测后再决定默认设置。

## Drafter 几何与 validation 修复

小 block 的 MPP 行数需要补齐到合法 tile。真实 proposal 数保持 block−1；输入 padding 置零，softmax 排除 padding，输出只写真实行。GPU oracle 曾定位到 current V 混用 padded stride 的错误，修复后按实际 Rows 寻址。

B4 的历史 tile 使用 N64，并将 softmax 列循环从固定四次改成 N/32；B16 也使用 N64。默认 B8 保留 M32/N128/K64 及运算顺序。B8 的旧 baseline 在 shader validation 下报告 33536 B，超过 32768 B；这条证据描述的是启用 validation 的运行，不能据此断言未启用 instrumentation 时也超过硬件上限。B8/B16 将少量行统计移到已有 partial，配合 device barrier，满足 validation 资源限制。

这些设备访问与 fence 的性能影响尚未测量。旧版本的速度不能作为当前源码的速度结论。非默认变体仅用于本轮 4B Standard DFlash 测试；不支持的 legacy verify 形状和 DFlash2 block 必须明确拒绝。

短正确性入口支持 `--block off/2/4/8/16`、`--concurrency 1/2/4` 和 `--tail-splits 1..16`。例如 `--run-correctness build/matrix-short-b16-c4.json --block 16 --concurrency 4 --tail-splits 16`。它只处理 64-token 输入、实际 32-token 输出，要求实际 tail 编码被调用、零缓存、事实 42 和完整批宽，不输出时长或吞吐；失败时保留原始输出。

真实 pipeline 元数据查询已确认 validation instrumentation 会把旧 B8 的共享内存从 16768 B 增至 33536 B；未启用 instrumentation 的旧内核没有超过 32768 B。修复后的数值为 16384 B／32768 B。当前统计迁移用于兼容仓库的 validation 门禁，不宣称带来速度收益；元数据证据为 `build/matrix-pipeline-memory-metadata.json`。

## 本轮最终验收

读取已完成的运行结果并取得原始 exec completion：30 组真实 Engine 正确性用例和仓库检查退出码均为 0。收尾阶段没有重跑测试，也没有启动性能矩阵。

| drafter 状态 | 并发 | splits | 通过批数 | 实际请求数 |
| --- | --- | --- | --- | --- |
| off | 1、2、4 | 1、16 | 6/6 | 14 |
| block=2 | 1、2、4 | 1、16 | 6/6 | 14 |
| block=4 | 1、2、4 | 1、16 | 6/6 | 14 |
| block=8 | 1、2、4 | 1、16 | 6/6 | 14 |
| block=16 | 1、2、4 | 1、16 | 6/6 | 14 |

合计 30 批、70 请求、2240 个实际输出 token。每个请求为 64-token 输入和 32-token 输出，使用 INT8 KV、ANE off。记录核对了事实 42、零缓存、输出数量、实际 batch width、实际 splits、tail 编码调用、drafter proposal 计数和对应产物身份。各 block 的 native 与 metallib 哈希仍与 stamp 一致，结果源码哈希与当前代码一致。

`make -j4 check-source check-native-cpu check-native-build check-python` 退出码为 0；Python suite 运行 1288 项测试，跳过 4 项，lint 和 format 检查通过。日志中的 `status: incorrect` 来自覆盖失败输出行为的测试；完整测试断言和进程退出均通过。

小 block 的真实 Engine 检查还定位并修复了 draft context FC 输入 view 容量不足：物理 view 按合法 storage 行数绑定，实际 token 行数及 proposal 数保持 block−1。

完整记录为 `build/qwen4b-matrix-development-checks.json`，逐批原始命令与退出结果为 `build/matrix-engine-final-checks.json`，仓库检查日志为 `build/matrix-repository-accepted-checks.log`。这一验收不覆盖五种 KV、全部 splits、ANE on 或 32K 长输入的完整组合。

剩余工作是用户通知后再执行 4800 个配置的性能矩阵，比较 prefill/decode 秒数与 tokens/s、并发吞吐、内存和草稿接受率。尚无当前版本的 32K 性能排名。

该验收阶段的开发保存在 `codex/qwen35-4b-optimization` worktree，当时尚未提交或推送；后续提交与 ghost 合并见下节。主工作区 `ghost` 干净，当前本地 HEAD 为 `0cd3418f990951bfeab0a8c51798d64058a667bc`，远端为 `d166344b24077ef244dfcdc944ae13c76afbafd7`，领先 86、落后 0；保留已有提交和 worktree 中的并行改动。

## 合并 ghost 后的验证

开发源码先保存为本地提交 `e12ad9f9a681568b3545337ae512fdeb7727620f`，然后合入 `ghost` 的 `0cd3418f990951bfeab0a8c51798d64058a667bc`。两侧共同祖先为 `d166344b24077ef244dfcdc944ae13c76afbafd7`；开发侧一个提交，ghost 侧 86 个提交，实际出现 46 个冲突路径。

上游 `55efea4`、`c067dbd` 把 drafter 和 MLX 统一到 block 权重并删除旧 affine 后端；合并据此迁移 Standard 的全部 69 个 BF16 张量到 AF4G64 block images，保留 4B、NoDraft、四个 block、单 token 路径和 96 个 FP16 FFN cache。上游 `c1fc0aa` 使用压缩 context window 保存 DFlash2 状态；Standard 则保留各层完整 rings 与最后一层完整历史，避免把 2048 窗口策略套到 full-history 层。上游 `d95c865` 搬迁测试参考头，上游 `0cd3418` 删除失效旧启动脚本，这两项意图均保留。

Standard 保留前五层 causal/sliding4096、最后一层 full/noncausal；投影先舍入 BF16 再单独加残差，相关 dispatch 位于 ops 层。新 AF4G64 staged 反量化在 half 精度下执行，与旧 uint4 group FP32 scale/bias 算术不保证逐 bit 一致。原来的速度与接受率不能作为这个合并版本的结论。

合并版本已通过四个 block 的接受逻辑、GDN block、draft attention、Standard layer 与缓存恢复数值检查。Standard 的缓存检查覆盖 6144-token 历史、RAM/disk、跳过 draft restore 时的 promotion；NoDraft 仅存 GDN，DFlash2 保持 window/rebuild/admission。Engine 实际采用 Runtime 的动态 historyCapacity，4096→32768 的合法 Standard 缓存恢复不会受静态 2048 截断。Standard 与 DFlash2 的 independent native-row、每字节 writer、release/restore 和 loader oracle 均通过；Standard 七个 image hash 因有意格式迁移更新，其余 hash 保持。

64 输入、32 实际输出的 30 组真实 Engine 检查再次通过，覆盖 off/2/4/8/16、并发1/2/4、splits1/16。手工 4B ANE verifier 在 share0.5、minimum512 下通过全部 chunk 数值检查，没有校准或速度对照。生产 launcher 仍默认关闭 ANE。

两个新增算子测试的 inherited baseline 在 shader instrumentation 下超出设备共享内存限制：IQ4XS prefill 从 17408 B 被包装为 34816 B；GDN prefill scan 包装后为 35072 B。FP16 cache 与 single-token 用例因此保留 API validation、FP64/标量参考、padding/guard、恢复断言，关闭这两个用例的 shader wrapper；其余上述 block 与 ANE kernel 检查继续启用 shader validation。不能把这一结果写为所有 shader wrapper 或完整 KV GPU suite 都通过。

合并后的 Python suite 已运行 1266 项测试、跳过4项。最终 source、CPU、native build、Python 门禁与最终源码身份的记录保存在 `build/ghost-merge-final-report.json`。本轮仍未执行 32K 性能矩阵，未推送远端。
