# Compact 公共排名加载

Compact 已改为默认关闭、显式开启并指定公共文件。排名不再绑定主模型权重 SHA。
本次保留原权重、转换产物、旧构建和历史结果，没有运行性能测试。

## 文件与开启方法

公共文件：[qwen35-248320-dev-seed.splash](../compact-rankings/qwen35-248320-dev-seed.splash)。
人工参考信息：[qwen35-248320-dev-seed.json](../compact-rankings/qwen35-248320-dev-seed.json)。
来源是仍然存在的 Saluki 旧排名。取消的是启动器默认路径，旧文件及原始报告没有删除。
导出后 token 顺序完全一致。Saluki 与 MiniPlus 的派生 tokenizer SHA-256 一致，此信息只用于人工选择。

新格式只有 8 字节 `SPLCHD02` 版本标记、little-endian uint32 词表大小和完整 uint32 token ID 排列。
文件大小为 993,292 字节，词表为 248,320。可读 profile 和语料来源放在 JSON 中，运行时不将它们作为绑定条件。
实际 compact 权重从当前模型自己的输出 head 收集，各模型分别构建副本。

原生服务、Saluki、Swift 和 MiniPlus 启动器默认 `--compact-draft-head off`。
相关启动器使用本次独立构建 `build/compact-public-20261014/splash`，其他性能参数保持原值。
关闭时，即使给出排名路径，也不读取文件或分配 compact 资源。不会自动扫描模型目录或测试目录。

显式开启：

```sh
sh qwen38miniplusv3splash.sh \
  --compact-draft-head on \
  --compact-draft-ranking compact-rankings/qwen35-248320-dev-seed.splash
```

仅给出 `--compact-draft-ranking` 不会开启功能。不引入环境变量。
历史二进制保留原实现；使用本次新构建才能读取公共格式并执行严格校验。

## 拒绝启用与正常回退

| 条件 | 行为 |
| --- | --- |
| 开启但未指定文件 | 参数检查拒绝启动 |
| 文件缺失、不可读、版本不符或长度损坏 | 明确报错，拒绝启用 |
| 词表大小不符、ID 越界、重复、遗漏或尾随数据 | 明确报错，拒绝启用 |
| 未安装 drafter，或 head 不是支持的单个未旋转量化 FP32 投影 | 明确报错，拒绝启用 |
| 对应行投影非有限值或与完整 head 不精确相等 | 数值错误，拒绝启用 |
| 不同稀有 token 超过 256 个 | 覆盖率不足，正常使用完整 head |
| 历史稀有 token 比例超过 1% | 覆盖率不足，正常使用完整 head |
| 多 lane 或图像请求 | 保留请求条件回退，使用完整 head |
| Lookup 已提供提议 | 跳过 drafter/head，并单独记录原因 |

结构检查在 compact 分配前完成。开启时，初次构建会检查初始收集和动态行更新路径。
对同样的 8 行 BF16 输入，使用现有 staged decode 内核比较 compact 每一行与完整 head 的对应 token 行。
复用现有精确相等标准，不增加容差。校验工作区随后释放，每轮生成只保留原动态行更新，不重复计算完整 head。

日志和正确性报告记录 `compact_validated`、比较数量、最大误差，以及覆盖率／请求条件回退次数。
文件和数值错误不会进入正常覆盖率回退流程。
主模型保持完整词表验证，接受概率、残差修正与状态提交逻辑没有修改。
Compact 改变候选集合和提议分布，因此草稿、接受率或最终回答不同需要另行评测，不能作为本数值检查的失败条件。

## 一次性转换旧文件

旧格式只作为离线导入来源。运行时不保留两套加载模式。

```sh
.venv/bin/python -m dev.tools.compact_draft_ranking \
  --import-legacy docs/data/saluki27b-compact-seed-2026-10-14/head-ranked.splash \
  --tokenizer-profile qwen35-vocab248320-nfc-bytelevel-bpe \
  --output NEW_PUBLIC_FILE.splash > NEW_PUBLIC_FILE.json
```

工具不会覆盖已有输出。新语料和完整排序 JSON 都可以直接生成公共格式，不再需要主模型指纹参数。

## 最终正确性结果

| 检查 | 结果 |
| --- | --- |
| 公共格式和旧格式导出 | 9 项 Python 测试通过，原文件与 token 顺序保留 |
| 服务参数、启动和 API 回归 | 853 项通过，退出码 0 |
| 安装／启动器回归 | 232 项中 228 项通过，4 项原有外部 CLI 集成跳过，退出码 0 |
| 原生 CPU 检查 | 全部通过，退出码 0 |
| 39 个投影格式、原始行字节、映射与 FP64 原界限 | Metal 校验通过，退出码 0 |
| 同一文件用于不同权重 head | IQ3_S 与 Q6_K 两种合成 head 各自独立通过 |
| 错误映射、损坏文件与不支持布局 | 明确失败，拒绝启用 |
| 关闭生命周期 | malformed 文件不读取，compact 新增分配为 0 |
| Standard DFlash、DFlash2、接受／残差／状态处理 | 相关 Metal 回归通过，退出码 0 |
| MiniPlus V3 真实 head | 786,432 次对应行比较，最大差异 0，关闭新增分配 0 字节 |
| MiniPlus V3 覆盖率 | 1% 比例和 256 个不同 ID 两类回退均通过 |
| Saluki 真实 head | 用户已删除原 GGUF，并明确取消此次测试，未记为通过 |

MiniPlus 启用后的 compact 权重与映射合计 417,202,176 字节，约 397.875 MiB。
这是该模型 head 的实际格式与当前容量对应的副本，不是排名文件体积，也不是整机峰值。
没有做端到端回答质量或接受率评测，也没有测 prefill／decode 速度。

最终命令、源码版本、真实退出码、实际功能状态和原始日志见 [verification.json](data/compact-public-ranking-2026-10-14/verification.json)。
