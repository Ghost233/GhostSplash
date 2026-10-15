# Tiel MiniPlus：Q4H、16K 输出参数测试计划

## 固定条件

本轮延续用户选定的 Q4H 标准。主模型为 Tiel-Coder-35B-A3B.APEX-I-MiniPlus-V2.1.gguf，使用现有 DFlash2 drafter。
每个请求固定 32,768 实际输入、16,384 实际输出，上下文容量 131,072 tokens。实际使用 49,152 tokens，容量上限不是输入长度。
DFlash2 block=8、ANE 关闭、fast top-k 关闭、32 GiB 内存预算。greedy、ignore_eos=true、冷请求仅作为统一测量控制项。

基准为历史 splits=8、并发 1、lookup 和 compact 关闭。生产引擎、算子、metallib、权重和 token 输入保持冻结；复用与上一轮字节一致的原生入口。

## 逐项扫描

| 项目 | 测试值 | 对照方法 |
| --- | --- | --- |
| 历史 splits | 1～16，8 为基准 | 其他参数保持基准 |
| Prompt lookup | 关闭／开启 | splits=8、并发 1 |
| Compact draft head | 关闭／开启 | splits=8、并发 1 |
| Lookup + compact | 两项同时开启 | 观察组合与各自开启的差异 |
| 并发 | 1／2／4 | 每个请求都输出完整 16K，输入来自冻结的四份冷请求 |

共 21 个配置，每组 3 轮，63 批、75 个请求。三轮按正序、逆序和旋转顺序执行。基准关闭两项功能；开启一项时只改变该项。
并发组同时记录每请求速度／延迟、整批吞吐／耗时、实际 decode batch width 和内存。

prefill 块预算的编译值仍为 2,048。历史 splits 不改变这一预算。
当前模型路径只支持 DFlash2 block=8；MoE 没有实际 ANE FFN；greedy 不执行 fast top-k。上述无有效路径的开关不加入扫描。

## 校验与候选复核

每批检查真实退出码、实际输入／输出长度、事实、冷缓存、实际 KV／上下文／block／splits、功能开关与内存账本。
splits=8／并发 1 的基准和功能开关组必须与上轮冻结的 Q4H 完整 16K 输出逐 token 一致。
其他 splits 或并发保存相对基准的首次分歧，并检查同一配置三轮完整轨迹一致。事实检查不替代算子数值检查；本轮未改算子，复用已完成的源码门禁。

扫描中，同轮总耗时降低中位数至少 2%、至少两轮更快的单请求配置列为候选。最佳合格候选随后做 ABBA 三次，共 12 批、6 个相邻配对。
配对复核要求总耗时降低中位数至少 2%、至少 5／6 对更快，才列为参数建议。这是性能筛选规则，不是正确性容差。
没有合格候选时保留原基准。测试不会自动修改生产服务或根目录选定参数，KV 始终为 Q4H。

按用户此前授权接受后台负载，全部波动和慢样本保留；相邻配对仍属于诊断性测量。失败、超时或取消不记为通过，任一阶段失败后停止后续阶段。

## 运行与保存位置

原始目录：`/Users/ghost233/.codex/worktrees/qwen4b-quantized-ffn/GhostSplash/build/tiel-miniplus-q4h-output16k-sweep-v1`。

- [实时日志](../build/tiel-miniplus-q4h-output16k-sweep-v1/controller.log)
- [当前进度](../build/tiel-miniplus-q4h-output16k-sweep-v1/progress.json)
- [计划与冻结身份](../build/tiel-miniplus-q4h-output16k-sweep-v1/plan.json)
- [已完成样本](../build/tiel-miniplus-q4h-output16k-sweep-v1/summary.json)
- [自动收尾状态](../build/tiel-miniplus-q4h-output16k-sweep-v1/finish-status.json)

完成后生成 `docs/tiel-miniplus-q4h-output16k-2026-10-11.md` 和 `docs/data/tiel-miniplus-q4h-output16k-2026-10-11/`。
若进入候选复核，另生成 `docs/tiel-miniplus-q4h-output16k-paired-2026-10-11.md` 与配套数据目录。
原始载荷、输出、日志、环境采样、内存账本及哈希验证归档保存；不复制权重。

## 受干扰样本复测

用户报告隔壁同时进行了性能测试后，原驱动和自动收尾已停止。停止时有 22 批完整结果，其中 19 批检测到外部推理活动；另一个已进入调度的尝试被取消，不记为通过。
仅保留首轮 splits=3／4／5 的 3 批：启动前两次 GPU 采样均≤50%，且全部运行采样没有检测到其他活动推理或 benchmark。
重新测量 19 个受影响样本，并完成剩余 41 批，共新增 60 批；最终矩阵仍为 63 个逻辑样本。

复测保持原二进制、生产代码、权重、输入和全部参数。启动前要求两次空闲检查通过：GPU≤50%，且无活动外部推理／benchmark。
运行中每次负载采样若检测到外部活动，就中止本次、保存真实退出状态并等待空闲后重试。启动检查之间负载变化也会重新等待。
外部干扰、中止、超时和产品校验失败分别记录；无干扰的慢样本不因速度低而重跑。所有数值、token、长度、事实和内存标准保持原要求。
后续候选配对使用同样门禁。当前 Q4H 标准参数没有修改。

- [新实时日志](../build/tiel-miniplus-q4h-output16k-sweep-v2/controller.log)
- [新进度](../build/tiel-miniplus-q4h-output16k-sweep-v2/progress.json)
- [复测审计与清单](../build/tiel-miniplus-q4h-output16k-sweep-v2/retest-audit.json)
- [原始 22 批数据](data/tiel-miniplus-q4h-output16k-interfered-2026-10-11/summary.json)
- [原始结果归档](data/tiel-miniplus-q4h-output16k-interfered-2026-10-11/raw-results.tar.gz)
- [归档哈希与清单](data/tiel-miniplus-q4h-output16k-interfered-2026-10-11/archive-manifest.json)

旧目录 `build/tiel-miniplus-q4h-output16k-sweep-v1` 保留。复测目录为 `build/tiel-miniplus-q4h-output16k-sweep-v2`，旧的 19 个受影响性能数据不参与新汇总和参数筛选。

## 第二轮外部性能测试后的复测

用户确认隔离复测期间又启动了一轮性能测试。第二次复测已暂停：保留原最早的 3 个样本，本轮新测的 13 个样本保留完整原数据，但因确切重叠区间不明，暂不用于性能筛选并重新测量。正在测量的并发 2 尝试被主动取消，不记为通过。

用户随后明确回复“已经结束，继续复测”。完成状态已记录，第三次复测从原 63 个逻辑样本继续：保留 3 个早期无干扰样本，新测 60 批。模型、权重、原生二进制及数值验收保持一致。
每批开始前用 3 次间隔 15 秒的空闲采样覆盖 30 秒窗口，均要求 GPU≤50% 且无外部活动推理／benchmark；窗口内任一检查不满足就重新等待。运行中继续检查外部负载，受干扰尝试保存并中止。

对高 CPU Python 进程的检查发现它在运行其他项目的 pytest，未发现 GPU 或模型资源；没有将它认定为漏检的 GPU benchmark，没有停止该进程。
当前无法从采样单独证明新一轮外部测试的精确重叠区间，重测 13 个样本采用保守处理，不表示每一条都已经证实受到干扰。

- [当前实时日志](../build/tiel-miniplus-q4h-output16k-sweep-v3/controller.log)
- [当前进度](../build/tiel-miniplus-q4h-output16k-sweep-v3/progress.json)
- [隔壁测试结束的记录](../build/tiel-miniplus-q4h-output16k-sweep-v3/peer-benchmark-finished.json)
- [第二轮受影响范围审计](../build/tiel-miniplus-q4h-output16k-sweep-v3/second-interference-audit.json)
- [第二次复测原始数据归档](data/tiel-miniplus-q4h-output16k-second-interference-2026-10-11/raw-results.tar.gz)
- [归档哈希与清单](data/tiel-miniplus-q4h-output16k-second-interference-2026-10-11/archive-manifest.json)

原目录 v1、v2 及其归档全部保留。当前运行目录为 `build/tiel-miniplus-q4h-output16k-sweep-v3`；标准参数仍为 Q4H／splits=8／并发 1／DFlash2 block=8，lookup 和 compact 关闭，ANE 关闭。

## 并发图比较修正后恢复

新增的跨运行并发 token 断言错误地比较了不同实际执行图。原冷图数值检查及 10 项回归用例通过后恢复，生产二进制和容差不变。17 个有效样本保留，继续 46 批；完整说明和独立的采样恢复失败见[诊断报告](tiel-concurrency-validation-2026-10-12.md)。当前日志为 `build/tiel-miniplus-q4h-output16k-sweep-v4/controller.log`。
