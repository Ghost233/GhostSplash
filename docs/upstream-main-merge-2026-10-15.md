# 实验分支同步 Splash main（2026-10-15）

本次在 `codex/qwen4b-quantized-ffn` 的既有 worktree 合并上游，遵循 ghost 分支的原则：上游已经覆盖的功能采用上游实现，同时删除被替代的本地特殊处理、重复测试和过时说明。保留仍有必要的分支独有功能。

## 合并基线

| 角色 | 完整提交号 |
| --- | --- |
| 合并前实验分支 | `1889aa593a759b2859e67ee2c94f4a76073ce754` |
| 三模型接入保存提交 | `f1567968e10ce3581f580855e086af574239010b` |
| 共同祖先／原上游 main | `cbbe322106fb960ce5e92cac75c52b729962ede9` |
| 新上游 main | `be3b083c5f413673d44e44881042e875e674e333` |
| ghost 参考合并 | `6ea64b6f55ab27a80fc4aa719bc5470cbd48c0ad` |
| ghost 参考清理 | `9d792904949aece261420f76e01253253518d101` |

没有使用 rebase、cherry-pick、stash、reset 或新建替代分支。本次保存了之前未提交的三模型源码、启动器与原始验证数据，再执行普通双亲 merge。

## 冲突和自动合并的处理

初次 merge 有 41 个冲突文件。按共同祖先之后的两侧历史逐项处理，并检查没有冲突标记的相关代码。

| 范围 | 最终处理及历史依据 |
| --- | --- |
| GGUF embedding | 使用上游 `4bb96e8b`、`b9ceabbc`、`30779252` 的 BF16／F32 gather 与统一 decode。删除旧 IQ 定义和旧接口，避免自动合并留下重复实现。 |
| GDN alpha／beta | 使用上游 `b9ceabbc` 的任意可读类型组合与 GPU F32 解码。删除本地 F32／Q3_K、Q4_0、Q8_0 白名单和 CPU 反量化。 |
| BF16／F16、Q5_0／Q5_1、融合 MoE | 使用上游 `30779252`、`6163da18`、`c35a306d` 的加载、筛选和测试。删除过时的格式拒绝断言。 |
| Q2_0 | 保留本地 `1fe29260` 的独有格式。在上游 43 个内部格式之后追加 ID 43，总数 44，避免占用上游 Q5_0 的 ID 39。 |
| GGUF 测试 | preparation 完全采用上游；metadata 仅保留 4B tied head 的必要测试和 Q2_0 常量；projection 保留 FP32 Q8_0 prefill 数值检查及独有 IQ1S replay 测试的入口，移除已被上游覆盖的 IQ 特例。 |
| 小模型 | 保留 `1889aa59`、`f1567968` 的 Kev-4B／0.8B、Julia／Laya、tokenizer、决策 head、共享 encoder 工作区和相应测试。 |
| FFN／可选加速 | 保留 `85700e5a`、`eb0627fe` 的量化 FFN、有限缓存、lookup、公共 compact 排名、fast-topk 和实际内存预留。单行 kernel 补入上游 BF16／F16 解码规则。 |
| ANE | 保留 `3f43aa24` 的启动参数、退避和功耗恢复；吸收上游 `3a7a81be` 的重启上下文保证。 |
| 缓存和内存 | 使用上游 `13f6e117`、`a212b810`、`ffc2d4be` 的自动 SSD 配额、写入预算和主机内存余量逻辑。保留本地 Q4 KV、Standard 历史容量和状态格式的命名隔离。 |
| GDN 状态 | 使用上游 `99aa99db` 的 BF16 recurrent state 选项及 `48a94aaa` 的 pending tape；删除旧 decode commit shader。单 token 融合仅用于 Float32 state、没有内部捕获的调用。 |
| 预填充状态快照 | 使用上游 `985463cf`、`c3dcf53f` 的 reserve／copy／seal。载荷分别处理 NoDraft 的 GDN、Standard 的完整 rings、DFlash2 的 window。Standard 保持在命令边界截断并快照，避免后续 token 覆盖历史 rings。 |
| 请求隔离／thinking | 使用上游 `29a59c3e` 的 cache salt，保留 Kev 首选项之前的缓存边界；lookup 与 drafter 汇合后统一应用上游 thinking cutoff。 |
| 评分文本 | 复用上游 `f5ac03af` 的 `tokenizable()`，让本地 Kev／ModernBERT 同样把孤立 surrogate 作为 U+FFFD 编码。真实 tokenizer 的 TypeError 复现、先失败后通过的回归和三个真实模型的替换对照一并保留。 |
| encoder 释放权重 | 适配上游新增的 ReleaseRequestFrame，明确报告不支持释放并返回 HTTP 409 `unsupported_operation`；不释放仍被使用的缓冲。增加协议与 HTTP 回归检查。 |
| 文档及模型资产 | 更新 README 的能力归属和同步原则；保留本地启动器、模板、draft 转换工具、原始模型、转换产物、历史结果和旧构建。 |

Q2_0 的 golden hash 没有重录：独立 `q2_0-oracle.cpp` 的 seed 26 产生原 hash `b52b2c8c0c2be25f895be42dd814419400de3c0bf8b4f84582f537b9d959badb`。上游把格式遍历扩展到所有格式后，这个原先不在通用 reference 循环内的格式首次被该循环覆盖。现在为它显式固定独立夹具种子，避免内部格式编号调整改变输入。

## 验证方式

使用独立输出目录 `build/merge-upstream-20261015`，保留所有旧构建。Make 使用绝对路径的 `BUILD_ID_PYTHON`，避免子进程切换目录后找不到相对解释器。

```sh
make -j4 BUILD=build/merge-upstream-20261015 \
  BUILD_ID_PYTHON="$PWD/.venv/bin/python" check-native-cpu check-native-build
make -j4 BUILD=build/merge-upstream-20261015 \
  BUILD_ID_PYTHON="$PWD/.venv/bin/python" check-native-metal
make -j4 BUILD=build/merge-upstream-20261015 \
  BUILD_ID_PYTHON="$PWD/.venv/bin/python" test-sanitizers
make BUILD=build/merge-upstream-20261015 \
  BUILD_ID_PYTHON="$PWD/.venv/bin/python" check-source verify-build-identity
```

Python 完整发现仍使用仓库的 `unittest discover` 范围。因安装器默认读取 `build/splash`，本次在测试进程中明确将 `install.paths.BINARY` 指向集成构建，避免误测旧二进制，也不覆盖旧文件。对应完整脚本、源码哈希、真实退出码和日志存于 [验证记录](data/upstream-main-merge-2026-10-15/verification.json)。同时完成 compileall、Ruff lint 和格式门禁。

Metal 使用仓库原有的 `MTL_SHADER_VALIDATION=1`。一次额外同时启用 `MTL_DEBUG_LAYER=1` 的手工诊断触发了仓库已注明的 threadgroup 内存双倍估算问题（GDN 实际 17,536 字节，被报为 35,072 字节）；该次退出 134，记录为测试环境失败。按原门禁设置重跑，不修改算子、数值容差或门禁规则。

真实模型回归使用之前保留的相同输入、参考选项与原始 GGUF，并在相同当前条件下分别启动保存的旧版本与集成版本：每个模型 13 个用例，包含 8K、重复请求和并发 4；Julia／Laya 继续比较完整 head 与裁剪 head。8 次服务启动共完成 104 个主用例，另有重复与并发检查。旧／新版本仍按原 `1e-3 * max(1, abs(x))` 标准比较：Kev 最大概率差为 `0.0005699383`，Julia／Laya 为 `0`。另用 encoder 原生协议检查取消、非法位置隔离、状态查询和不支持的权重释放。

最初直接对照历史 JSON 时，Kev 的 urgency 评分概率从历史的 `0.4323729178` 变为 `0.4294143046`，超出额外加入的跨历史记录比较条件，该次运行退出 1。随后用保存的旧源码和旧二进制冷启动对照，旧／新都得到 `0.4294143046`，重复调用也相同；完整同条件旧／新对照再按原标准通过。因此这不是已证实的合并回归。历史记录无法原样复现的原因尚未进一步归因，原数据、失败断言和两次对照日志均保留；没有修改 golden、容差或原始记录。

MoE 首次运行期间，调用栈位于 `MetalBackend::newPipeline` → `MTLCompilerScheduler` 的同步等待，`MTLCompilerService` 持续约 100% CPU；保留了单次采样记录。该阶段属于 pipeline 编译，日志静默本身不代表测试完成或死锁。

本次不重新运行模型性能矩阵。原始测速记录仍对应原来的源码和构建，不把它们作为合并后版本的速度结论。

## 最终结果

| 检查 | 结果 |
| --- | --- |
| 生产引擎、全部 native 测试及工具构建 | 退出 0 |
| 完整 CPU 门禁 | 退出 0 |
| 完整 Metal 门禁（含 44 格式、BF16 state 与本地独有算子） | 退出 0 |
| ASan／UBSan／TSan | 退出 0 |
| 最终源码完整 Python 发现 | 1,408 项，4 项既有跳过，退出 0 |
| compileall、Ruff lint／format、架构／空白／构建身份 | 退出 0 |
| 三模型旧／新受控对照、两种 encoder 原生协议检查 | 退出 0 |
| 三模型孤立 surrogate／U+FFFD 输入对照 | 6 个真实请求，全部通过，服务均退出 0 |

验证的 native build ID 为 `src-d22fd38e821706a8290ca95f7d89c1be5a97b764ef5f3728cd0aa9c52fc29704`。最终增加的两项 surrogate 回归仅修改 Python 文本准备及其测试；native 源码、构建 ID 与原生测试输入未变化，因此复用已完成的原生门禁，并对最终 Python 源码重新完成全套发现和静态检查。
