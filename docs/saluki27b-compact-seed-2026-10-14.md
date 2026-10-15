# Saluki 27B compact 种子排名

已按方案 1，从本仓库的技术文档和源码生成排名。Saluki 启动器最初默认使用这个文件；用户随后取消了默认路径，本次保留当前设置。
本次只运行 CPU 分词、文件校验和启动器测试，没有启动推理或性能测试。

## 排名和模型绑定

排名文件：[head-ranked.splash](data/saluki27b-compact-seed-2026-10-14/head-ranked.splash)。
生成报告：[report.json](data/saluki27b-compact-seed-2026-10-14/report.json)。
语料快照：[corpus.tar.gz](data/saluki27b-compact-seed-2026-10-14/corpus.tar.gz)。
归档校验：[artifact-check.json](data/saluki27b-compact-seed-2026-10-14/artifact-check.json)。

| 项目 | 值 |
| --- | --- |
| 模型 | ConwayResearch/Underdog-Saluki-27B-1.0:IQ2-mix |
| Drafter | incoai/Qwen3.8-27B-DFlash2 |
| 完整词表 | 248,320 个 token |
| 固定候选 | 98,048 个 token |
| 动态候选 | 最多 256 个上下文 token |
| 特殊控制 token | 27 个，全部保留在固定候选中 |
| 排名文件大小 | 993,356 字节，约 0.947 MiB |

排名绑定正式安装的 target manifest：

```text
d44d118bb0b90fded636f4abd4bf0d6d368fcd5fff1aa3e48bbb507da4475ef6
```

模型指纹取自此前该正式安装的 native `/status`。生成前已核对安装记录和各文件身份，包括 GGUF、drafter 和 tokenizer。
生成报告保存安装记录摘要、tokenizer SHA-256 和 native 身份。本次没有重新加载模型来计算指纹。

文件包含正确的 `SPLCHD01` 标记、模型指纹、词表大小和完整 token ID 排列。
已检查排列无重复、无遗漏，特殊控制 token 均在固定候选中。

## 语料和覆盖率

语料限于 Git 已跟踪的文档与源码：根目录的 3 份说明文档、`docs` 下的直接 Markdown 文件、benchmark 说明，以及 `runtime`、`server`、`install` 下的文本源码。
排除模型文件、构建产物、`docs/data`、聊天记录和仓库外文件。
按相对路径 SHA-256 的前 4 字节划分文件：余数为 0 的文件留出验证，其他文件统计词频。
验证文件不参与排名。原始输入共 321 个文件，已归档其当时内容和逐文件 SHA-256。

| 语料 | 文件数 | token 数 | 固定候选覆盖率 | 固定候选外的 token |
| --- | ---: | ---: | ---: | ---: |
| 训练 | 273 | 844,075 | 100% | 0 |
| 留出验证 | 48 | 219,308 | 99.843% | 344 |

训练语料只出现 16,952 种 token，固定候选空间足以容纳它们。
因此训练覆盖率为 100%，不能据此判断模型实际输出的覆盖率。

另将留出文件按完整文本分词，再切成最多 8,192 token 的片段，复核现有历史筛选规则。
61 个片段中，59 个满足条件。以下 2 个片段应回退完整词表，回退标准保持不变。

| 留出片段 | token 数 | 固定候选外 token 数 | 比例 | 结果 |
| --- | ---: | ---: | ---: | --- |
| `docs/kv-rotation-quantization-2026-10-09.md` 的首个片段 | 8,192 | 126 | 1.538% | 超过 1%，回退 |
| `docs/splash-lily-comparison-2026-10-07.md` | 4,009 | 155 | 3.866% | 超过 1%，回退 |

这份排名面向开发文档和代码场景。语料整体覆盖率与实际请求是否启用 compact 是不同的指标。
实际输入、已提交输出和 pending anchor 都会影响历史筛选；不同稀有 token 超过 256 个也会回退。
本次没有验证生成输出、草稿接受率或 GPU 投影，不能把 CPU 覆盖率当作实测加速。

## 启动和关闭

启动器 [saluki27biq2mixsplash.sh](../saluki27biq2mixsplash.sh) 当前不默认传递种子排名路径。
如需使用这份种子排名，请显式传递 `--compact-draft-ranking docs/data/saluki27b-compact-seed-2026-10-14/head-ranked.splash`。
保留当前上下文 262,144、内存上限 30G、队列 2、Q4H、ANE share 14/34 和 20% 功耗恢复阈值。
启动命令保持原样：

```sh
sh saluki27biq2mixsplash.sh
```

关闭 compact 或使用另一份排名：

```sh
sh saluki27biq2mixsplash.sh --compact-draft-head off
sh saluki27biq2mixsplash.sh --compact-draft-ranking /path/to/head-ranked.splash
```

已有进程不会热加载排名。下一次启动才读取文件。
实际解码宽度为 1 时才可能使用 compact；队列容量为 2 不代表每轮都启用。
Lookup 命中也会跳过该轮 drafter，因此 compact 计数不必等于解码轮数。
首次使用 compact 会分配对应行的 Metal 权重副本，实际内存和速度收益尚待运行验证。

## 复现与校验

`report.json` 保存完整生成命令、输入文件清单、tokenizer 指纹、语料划分、覆盖率和排名 SHA-256。
可以从语料快照提取当时输入，按报告中的清单重新运行 `dev.tools.compact_draft_ranking`。
工具拒绝覆盖已存在的排名文件，复现时使用新的输出路径。

更换 tokenizer 后需要重新统计。更换权重、drafter 或安装身份后，应重新核对 native target manifest。
相同 token 排序可以重用，但必须绑定新指纹并复核覆盖率。

CPU 回归测试覆盖排名格式、完整排列、特殊 token、词频排序、验证集隔离，以及启动器的默认路径、关闭和路径覆盖。
保留现有的 1% 与 256 个稀有 token 回退规则。

本轮 25 项 CPU 回归测试全部通过，启动器语法、Ruff 和 `make check-source` 均通过，退出码为 0。
命令、最终源码摘要和日志索引见 [verification.json](data/saluki27b-compact-seed-2026-10-14/verification.json)。

2026-10-14 公共排名变更：compact 当前默认关闭，原绑定文件仅保留作历史来源。
新的显式开启规则见 [公共排名加载说明](compact-public-ranking-2026-10-14.md)。
