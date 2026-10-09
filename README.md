# GhostSplash

GhostSplash 是基于 [incoai/splash](https://github.com/incoai/splash) 的个人魔改版，
主要用于在 Apple silicon Mac 上运行本地 GGUF 模型，适配第三方模型的量化权重、
聊天模板和 draft，并保存本机性能测试记录。

## 分支说明

| 分支 | 用途 | 状态 |
| --- | --- | --- |
| `ghost` | 魔改主分支，继承原 `codex/splash-local-compat` 的完整提交历史 | 日常使用与兼容性维护 |
| `main` | 保留上游基线 | 本次迁移未修改 |
| [`codex/qwen38-flash-next-support`](https://github.com/Ghost233/GhostSplash/tree/codex/qwen38-flash-next-support) | Flash Next：SSD 专家权重缓存、原生 MTP、批量 prefill | **未完成，实验分支** |
| [`codex/whittle27-support`](https://github.com/Ghost233/GhostSplash/tree/codex/whittle27-support) | Whittle 27B、Q5_0 适配及 64K 测速记录 | **未完成，实验分支** |

两个实验分支按原提交复制，尚未合入 `ghost`，也不包含 `ghost` 后续新增的全部改动。
它们保留已有实现与测试记录，分支存在不代表功能已完成验收。

## 我们魔改了什么

### GGUF 兼容性

- 扩展 token embedding 的原生 GGUF 解码，覆盖线性层和专家层已支持的量化格式，
  包括原先未覆盖的部分 IQ 格式与 MXFP4，减少第三方 GGUF 被加载器拒绝的情况。
- 支持 GDN `ssm_alpha` / `ssm_beta` 使用混合 F32/Q3_K 权重，加载时统一解码为 F32。
- 补充 GGUF 元数据、权重准备和投影测试；具体支持范围见
  [开发文档](DEVELOPMENT.md#gguf-targets)。

### 本地模型启动器

增加直接复用本地 LM Studio 模型目录的启动脚本，并为不同模型保留独立配置：

| 模型 | 启动脚本 |
| --- | --- |
| Qwen3.8 27B / Qwen3.6 35B-A3B | [qwen3.827bsplash.sh](qwen3.827bsplash.sh) / [qwen3.635ba3bsplash.sh](qwen3.635ba3bsplash.sh) |
| Ternary Bonsai 2 / Swift Bonsai 2 | [bonsai227bsplash.sh](bonsai227bsplash.sh) / [swiftbonsai2splash.sh](swiftbonsai2splash.sh) |
| Swift 1.5 Qwen3.8 27B | [swift1527bsplash.sh](swift1527bsplash.sh) / [swift1527biq3ssplash.sh](swift1527biq3ssplash.sh) |
| Tiel Coder / Tiel MiniPlus | [tiel35ba3bsplash.sh](tiel35ba3bsplash.sh) / [tielminiplus35ba3bsplash.sh](tielminiplus35ba3bsplash.sh) |
| Qwen3.8 35B-A3B Distill | [qwen38distill35ba3bsplash.sh](qwen38distill35ba3bsplash.sh) |
| Coder390 27B | [coder39027bsplash.sh](coder39027bsplash.sh) |
| Ornith 1.5 35B-A3B 3bit（纯文本） | [ornith1535ba3bsplash.sh](ornith1535ba3bsplash.sh) |

脚本中的模型路径、监听地址、端口和内存配置来自个人环境，使用前需要按本机修改。
启动器监听 `0.0.0.0`（所有 IPv4 网卡），本机使用 `http://127.0.0.1:端口`，
局域网或虚拟机使用这台 Mac 当前可达的 IP；不再依赖固定的虚拟网桥地址。
所有自建启动器默认传入 `--no-webui`，关闭内置聊天页面，只提供 API，
本机 API 地址为 `http://127.0.0.1:端口/v1`。
模型权重、Python 虚拟环境、生成的模型安装目录和编译产物不随 Git 分支迁移。
Coder390 还需要在 `install/models-local/drafts/coder390-dflash2-fp8/bf16/`
准备转换后的 draft；新目录可复用已验证的 BF16 副本，或用下面的命令从作者的
FP8 源目录（含 `manifest.json`、`config.json` 和 `model.safetensors`）转换：

```bash
.venv/bin/python dev/tools/dequantize_coder390_dflash2.py /path/to/fp8-source \
  install/models-local/drafts/coder390-dflash2-fp8/bf16
```

启动器回归测试使用 `.venv/bin/python -m unittest dev.tests.install.test_local_launchers`，
覆盖首次安装、已有安装复用、错误模型记录拒绝、模板适配和实际服务参数。

### 模板与 draft 适配

- **Tiel MiniPlus**：启动时生成独立 tokenizer 副本，将原模板的 JSON 工具调用
  适配为 Splash 使用的 `<function=…>` / `<parameter=…>` 格式，并支持后续 system 消息。
  原始模板及适配边界见 [模板存档](docs/templates/README.md)；已有离线验证，
  真实模型的生成质量仍需单独验证。
- **Coder390**：增加作者发布的逐张量 FP8 E4M3 DFlash2 draft 到 BF16 的
  [转换工具](dev/tools/dequantize_coder390_dflash2.py)及对应测试。
  启动器使用转换后的 draft；历史使用官方 DFlash2 的测速结果与作者 draft 的结果
  在记录中分别注明。

### 性能记录

保存 [本地模型测速与诊断记录](docs/local-model-speed-notes-2026-10-08.md)、
[模型对比](docs/model-benchmark-comparison-2026-10-03.md)和
[Splash / Lily 对比](docs/splash-lily-comparison-2026-10-07.md)。
这些记录对应当时的硬件、权重、量化格式、配置与构建版本；不作为所有模型的速度承诺。

## 未完成的实验

- **Flash Next**：实现分片 GGUF 加载、SSD 专家权重缓存、原生 BF16 MTP 和批量 prefill。
  最后保存的 512 行 prefill 与 teacher-only KV/index 优化仅通过语法检查，
  尚未完成编译及运行验收。历史 8K 长测已完成，16K 被取消；现有实验路径
  尚不支持前缀状态缓存与工具约束。详见该分支的
  [优化及验证记录](https://github.com/Ghost233/GhostSplash/blob/codex/qwen38-flash-next-support/docs/flash-next-optimization-2026-10-09.md)。
- **Whittle 27B**：保存文本模型支持、Q5_0 权重适配、启动器与 64K 测速归档。
  已有测试和测速记录，整体仍按未完成分支保留，未合入魔改主分支；实现说明见该分支的
  [开发文档](https://github.com/Ghost233/GhostSplash/blob/codex/whittle27-support/DEVELOPMENT.md)。

## 从源码使用魔改版

在支持的 Apple silicon Mac 上准备 Xcode / Metal 工具链与 Python 3.12–3.14：

```bash
git switch ghost
make install-environment
make -j4
# 按本机修改模型路径和监听配置，并准备对应模型与 draft 后启动：
sh tielminiplus35ba3bsplash.sh
```

运行检查使用 `make check test-sanitizers`，详细构建和模型准备方式见
[DEVELOPMENT.md](DEVELOPMENT.md)。下方保留上游使用说明；其中 Homebrew 安装命令
安装的是上游 Splash，使用本仓库的魔改需要从源码构建。

## 上游 Splash 说明

[![CI](https://github.com/incoai/splash/actions/workflows/ci.yml/badge.svg)](https://github.com/incoai/splash/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Apple%20silicon-black.svg)](#quick-start)

**A local inference engine for Apple silicon, built around the model.**

Splash runs coding agents and OpenAI or Anthropic compatible applications on
one Mac. It combines [DFlash 2](https://inco.ai/blog/dflash2/) speculative
decoding, specialized Metal kernels, and automatic memory planning, with
vision, tool calling, and a built-in chat page.
It reuses cached prefixes and batches concurrent requests automatically.

## Quick start

Apple M3 or newer, macOS 26.4 or later, and [Homebrew](https://brew.sh).
The 4-bit examples need at least 36 GB of unified memory (48 GB recommended);
24 GB Macs can use [smaller GGUF variants](#models).

```bash
brew install incoai/tap/splash
splash serve --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
```

The first run downloads the model and its matching draft and starts serving
on `127.0.0.1:8000`. Later starts reuse the downloads; leave room on disk for
them ([storage requirements](DEVELOPMENT.md#model-storage)).

Once it prints `Ready`, leave this terminal open. Open <http://127.0.0.1:8000>
in your browser, or run an installed coding agent from another terminal:

```bash
splash opencode    # or: splash claude / splash codex / splash hermes / splash pi
```

For a server started with another `--port`, pass the same `--port` to the agent.

Press Ctrl+C in the server terminal to stop Splash.
For LM Studio Bionic, follow its [Splash setup guide](https://lmstudio.ai/blog/splash-engine).

## Use the API

OpenAI Chat Completions, Responses and Completions, and Anthropic Messages,
with streaming, tool calls, JSON Schema output, and base64 images and PDFs:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M",
    "messages": [{"role": "user", "content": "Explain speculative decoding in one sentence."}]
  }'
```

Reasoning follows the model default; `"reasoning_effort": "none"` turns it off
in Chat. Anthropic Messages requests reason only when they set `thinking`.
[Reasoning settings](DEVELOPMENT.md#default-reasoning-effort) ·
[Tool calls](DEVELOPMENT.md#tool-calls) ·
[API details](DEVELOPMENT.md#api)

## Models

Splash supports these model families, with a matching DFlash2 draft selected
automatically:

| Model | GGUF example | MLX 4-bit |
| --- | --- | --- |
| Qwen3.8-27B | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` | `mlx-community/Qwen3.8-27B-4bit` |
| Qwen3.6-35B-A3B | `unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M` | `mlx-community/Qwen3.6-35B-A3B-4bit` |

Unsloth GGUF variants span **1–8 bits**, including mixed-precision UD formats;
`UD-Q8_K_XL` and BF16 targets are not supported.
[Prism ML Ternary Bonsai 2](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
is also supported in PQ2_0 (7.2 GB), including vision. Pass `OWNER/REPO:VARIANT`
to `--model`, as in the quick start. MLX targets must be affine 4-bit with
groups of 64, like the examples. Smaller variants run on
[24 GB Macs](docs/performance.md#smaller-ggufs-on-24-gb-macs).
[27B variants](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main) ·
[35B variants](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main)

Vision and the tokenizer come from the target model's source.
[Model loading and compatibility](DEVELOPMENT.md#upstream-model-loading) ·
[Supported formats](DEVELOPMENT.md#gguf-targets)

## Settings

Memory and context are sized automatically, up to the model's native context
window. To set your own limits or cache options, add these to `splash serve`:

| Option | Purpose |
| --- | --- |
| `--max-memory 28G` | Cap Metal memory use. |
| `--idle-release off` | Keep the model in memory while idle (default: release after 10m). |
| `--max-context 100K` | Set the context limit. |
| `--language-only` | Skip vision; serve text only. |
| `--kv-format bf16` | Use BF16 KV cache. Default: 8-bit (INT8). |
| `--disable-ane` | Prefill on the GPU alone. Default: the 27B also uses the Neural Engine. |
| `--max-cache-disk 16G` | Keep cached prompts on SSD when memory runs short. Off by default. |
| `--persistent-cache` | Keep the SSD cache across restarts; needs `--max-cache-disk`. Off by default. |

Use `--max-memory` to leave room for other applications.
The server listens on localhost without authentication by default. For LAN
access, authentication, browser apps on other origins, and other options, see
[server configuration](DEVELOPMENT.md#server-configuration) or
`splash serve --help`.
[KV precision](DEVELOPMENT.md#kv-cache-precision) ·
[SSD cache](DEVELOPMENT.md#ssd-cache)

## Performance

Measured with Splash 1.0 on an M5 Pro (16-core GPU, 48 GB), using its model
packages and selected SPEED-Bench coding prompts over HTTP. Ratios compare
with the next-fastest engine measured in that benchmark.

| Metric | Qwen3.6-35B-A3B | Qwen3.8-27B |
| --- | ---: | ---: |
| Decode · short prompt | 210 tok/s (1.7×) | 74 tok/s (2.0×) |
| Prefill · 32K prompt | 2,011 tok/s (1.3×) | 363 tok/s (1.2×) |
| Cached time to first token · 32K replay | 123 ms (6.6×) | 282 ms (7.3×) |
| Aggregate decode · 4 concurrent short prompts | 357 tok/s (2.0×) | 170 tok/s (3.9×) |

[Launch benchmarks](https://inco.ai/blog/splash/) ·
[Measurement details](docs/performance.md#splash-10-launch-benchmarks) ·
[Run benchmarks locally](DEVELOPMENT.md#local-benchmarks)

### GGUF against llama.cpp

Same Unsloth UD-Q4_K_M weights on Metal. Decode speed in tok/s:

| Model | Engine | M5 Pro, 20-core GPU | M3 Max, 40-core GPU |
| --- | --- | ---: | ---: |
| 27B | llama.cpp | 16 | 17 |
| | llama.cpp with MTP | 27 | 20 |
| | **Splash** | **74** | **92** |
| 35B-A3B | llama.cpp | 69 | 66 |
| | **Splash** | **175** | **209** |

That is **2.5–3.2×** as fast on the 35B and **4.5–5.3×** on the 27B
(**2.7–4.6×** against MTP).

**Closely matches llama.cpp's predictions.**

| Next-token agreement ↑ | 27B | 35B-A3B |
| --- | ---: | ---: |
| llama.cpp: CPU vs. GPU | 97.8% | 96.5–96.9% |
| llama.cpp: single-token vs. batched | 99.65–99.75% | 97.95% |
| **Splash vs. llama.cpp** | **99.30–99.45%** | **97.83–98.14%** |

Splash uses BF16 KV and `--disable-ane` in this comparison.
[Benchmark details](docs/performance.md#gguf-against-llamacpp)

## Design

Each supported model pairs a trained DFlash2 draft with Metal kernels for its
shapes. The runtime, scheduler, cache, and API are shared. Weights are
converted to the kernels' layouts as they load, with no copy on disk; kernels
ship precompiled, with no Xcode or local tuning required.
[How Splash works](https://inco.ai/blog/splash/)

## More

- [Development](DEVELOPMENT.md): configuration, API, internals, building, and testing.
- [Issues and feedback](https://github.com/incoai/splash/issues)
- [Apache-2.0](LICENSE). GGUF kernels include MIT-licensed material from
  llama.cpp; see [third-party notices](THIRD_PARTY_NOTICES). Model weights keep their own licenses.
