# MiniPlus V3 启动器与配套 DFlash2

本文保留原 EfficientThink V3 的实施与验证记录。`qwen38miniplusv3splash.sh` 现已改为用户指定的 VAL APEX I MiniPlus V3，默认端口 8008；当前用法见 [新模型启动说明](miniplus-val-apex-launcher-2026-10-14.md)。

已准备启动器 `qwen38miniplusv3splash.sh`，明确选择用户提供的 V3 主模型和配套 DFlash2。
原始 GGUF 文件保持完整。此次没有运行大模型生成或性能测试。

## 启动参数

```sh
sh qwen38miniplusv3splash.sh
```

| 参数 | 默认值 |
| --- | --- |
| 主模型 | `Qwen3.8-27B-EfficientThink-VAL-APEX-I-MiniPlus-V3.gguf` |
| Drafter 原文件 | `dflash2-qwen38-27b-Q4_K_M.gguf` |
| Drafter 使用格式 | 原文件离线反量化为 BF16 safetensors，运行时由现有 loader 生成 AF4G64 投影 |
| 原生构建 | `build/miniplus-v3-dflash2-20261014/splash` |
| 服务端口 | `8008` |
| 上下文容量 | `262144` |
| KV | `q4h` |
| 队列容量 | `1` |
| 内存预算 | `40G` |
| 磁盘缓存预算 | `50G` |
| Drafter block | `8`，编译时固定 |
| ANE | 原生自动校准 |
| ANE 功耗参考与恢复 | 默认关闭，可用启动参数开启 |
| Lookup | 开启 |
| Compact | 默认关闭，尚无绑定该主模型的排名 |
| Fast top-k | 关闭 |

`--help` 列出完整参数，命令行可覆盖默认值。例如：

```sh
sh qwen38miniplusv3splash.sh --port 8009 --max-memory 30G --queue-size 2
sh qwen38miniplusv3splash.sh --ane-ffn-share 0.5 --ane-ffn-minimum-rows 512
```

启动器检查主模型来源、转换 drafter 来源和已安装模型绑定。
本地安装已准备完成，不需要下载默认 drafter。当前为纯文本模式，不加载 mmproj。
V3 文件包含 64 层主干和 1 层 MTP；此次使用配套的 5 层外部 DFlash2。
这些参数用于启动服务，尚未作为该模型的性能推荐值。

## Drafter 转换

转换产物位于 `install/models-local/drafts/miniplus-v3-dflash2-q4km/bf16`。
包含 `config.json`、`model.safetensors` 和 `conversion.json`。
BF16 safetensors 约 3.58 GiB，只是磁盘格式；原生 CPU 规划的 drafter 权重图为 1,266,647,040 字节，约 1.180 GiB。
权重图字节数不包含 KV、请求状态、ANE 和临时工作区，也不是实测进程峰值。

复现命令如下。工具拒绝覆盖已有目录，复现时选择新的输出目录。

```sh
.venv/bin/python -m dev.tools.dequantize_dflash2_gguf   /path/to/dflash2-qwen38-27b-Q4_K_M.gguf   /path/to/new-bf16-directory   --target /path/to/Qwen3.8-27B-EfficientThink-VAL-APEX-I-MiniPlus-V3.gguf
```

工具复用 [llama.cpp 官方 gguf-py 反量化](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/quants.py)，按块处理 F32、Q4_K 和 Q6_K，再按 round-to-nearest-even 写入 BF16。
它验证主模型与 drafter 的词表、token 类型和 BPE 合并规则一致，并检查完整张量清单与形状。
张量轴顺序转换为 safetensors 格式，Q/K 行保持原序。
GGUF 的 `target_layers` 比 HF decoder 层号大 1，逆转换减去 1；依据 [llama.cpp 的 DFlash 转换实现](https://github.com/ggml-org/llama.cpp/blob/master/conversion/qwen.py)。

本文件的捕获层从 `[6, 20, 34, 48, 62]` 转回 `[5, 19, 33, 47, 61]`。
RMS epsilon 只有在与运行时 `1e-6` 的 FP32 位模式相同后，才写成规范配置值。
转换产物已经过现有原生配置检查，不放宽检查标准。

## 主模型兼容处理

V3 的 48 层 GDN 小投影使用混合 F32/Q4_0 或 F32/Q8_0 格式。
现有 loader 原先只支持混合 F32/Q3_K，本次扩展其相同的 FP32 展开路径。
新增量化小投影的 FP32 字节数合计 47,185,920，即 45 MiB。
主模型其余量化矩阵继续保持原格式，没有创建全量 FFN FP16 缓存。

为保留已有运行版本，启动器选择独立构建目录。
原来的 `build/ane-startup-cli-20261013` 与 `build/splash` 保持不变。

## 校验记录

- 真实模型原生 CPU 规划通过：66 个 target 权重图、6 个 draft 权重图、81 个 drafter 张量。
- 对 81 个 drafter 张量各抽取首、中、尾块，共 243 块、40,704 个权重值。BF16 位模式与独立 C++ GGUF 参考按同样舍入后的结果完全一致。
- 样本 BF16 舍入的最大绝对误差为 `0.007110595703125`，相对于 GGUF 已量化权重的 FP32 解码值；不是相对于原始未量化 checkpoint 的误差。
- Metal 权重准备测试开启 API 与 shader validation。混合投影的 F32 值及头顺序逐位匹配参考，最终 0 个失败。首次复制格式门禁漏项已修复，原失败日志保留。
- 原生 CPU 检查、源码检查、15 项工具测试均通过。
- 安装模块运行 231 项测试，227 项通过，4 项外部 Codex／OpenCode／Pi CLI 集成按原门禁跳过。跳过项与本模型加载无关，完整原因已保存。
- 真实 MiniPlus tokenizer 合约、分词规则和聊天模板校验通过，Mistral 正则误报没有出现。

来源、SHA-256、绑定和原始日志见 [校验目录](data/miniplus-v3-local-draft-2026-10-14/verification.json)。
本次验证范围是转换、加载规划和小型权重准备；尚未验证完整模型生成输出、接受率或性能。

2026-10-14 公共排名变更：compact 当前默认关闭，原绑定文件仅保留作历史来源。
新的显式开启规则见 [公共排名加载说明](compact-public-ranking-2026-10-14.md)。
