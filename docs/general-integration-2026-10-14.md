# 通用增量整合与提交记录

以 `codex/qwen4b-quantized-ffn` 的当前工作目录为基础，整合 Flash Next 分支中可共享的部分。来源提交为 `073416759088602e6bd0e81bc246ee61f755867a`，共同基线为 `3f43aa2460f59330e658fdc89945d72ce92d538c`。没有整体合并 Flash Next 分支。

开始前已备份双方全部未提交和未跟踪文件，并逐文件核对 SHA-256。FFN 分支今天的公共 compact 排名、一次性数值校验、启动清单、混合 GDN gate、转换 drafter 与 MiniPlus 启动器均保留。原有脏文件中只有 `install/gguf.py` 和它的元数据测试有额外修改，都是 Q2_0 的局部增量。

## 整合范围

共享 KV 层统一 INT8、BF16、Q4H、旋转 Q4_0 和未旋转 Q4_0 的格式规则、旋转、量化、打包、解码与存储。普通注意力消费者复用该层。测试保留独立 dense Hadamard／GGML 参考、跨页、未写区域、BF16 位值与 CPU/GPU 字节比较。目标分支没有引入 QSA 或其他 Flash Next 运行时依赖。

Q2_0 为权重格式，原生类型 42，每 64 个元素用 18 字节保存。加入解析、repack、embedding、投影、Python 格式识别和独立 golden。它与 Q4H 等 KV 格式分别处理。独立参考固定于 llama.cpp `9c2e0e491a822adae1f0b1c831adb4160057d24f`，原始输出 SHA-256 为 `b52b2c8c0c2be25f895be42dd814419400de3c0bf8b4f84582f537b9d959badb`。

完整 Python 门禁还发现启动报告没有列入发布包清单，现已补充 `startup_report.py`；实际 staged 包导入与启动检查通过。

## IQ1S 处理

局部 accumulator 构造候选未解决固定复现，出现 101 次严格比较失败；目标原路径也出现 98 次失败。没有将该候选移入目标分支，也没有放宽原断言。

失败集中于单次 gate/up 合并路径，两个独立 pass 的对照稳定。通用算子现对 IQ1S 默认采用两个 pass；显式请求合并时也使用两个已验证的 dispatch，以输出作为临时 gate。其他格式的合并路径不变，没有添加临时权重缓存。固定 RNG、splits 1／4、各 128 次重复保持不变，修正后的比较为 0 失败。原合并 shader 的底层根因仍未修复，该路径由算子保护绕开。

## 来源冻结状态

Flash Next 专用 QSA／GDN／PLE／原生 MTP、专家分页和 PLE 不驻留策略没有带入本分支。来源的 35 GiB 权重／5 GiB 其他内存及启动准入修改单独保存，等待上游 #253／#410。

原 4096 输入／1024 输出试验在预热阶段遇到 native status 30 秒超时，用户随后取消。正式轮数为 0，不能当作性能或实模准确性通过。本轮没有恢复这项测量。

## 验证和版本

最终命令、实际退出码、编译源身份、文件哈希、原始失败和修正结果保存在 [验证记录](data/general-integration-2026-10-14/verification.json)。已有模型权重、转换产物、构建和历史结果保留。新构建在 `build/general-integration-20261014`；当前运行的引擎没有被重启或替换。

Git 提交身份在完成全部必需门禁后记录。没有 push、创建 PR 或自动归档会话／worktree。
