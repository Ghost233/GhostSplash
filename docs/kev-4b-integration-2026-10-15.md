# Kev-4B 本地接入与正确性验证

## 启动

```sh
sh /Users/ghost233/.codex/worktrees/qwen4b-quantized-ffn/GhostSplash/kev4bsplash.sh
```

模型使用本机 `Kev-4B-Q4_K_M.gguf`，位于 `/Users/ghost233/.lmstudio/models/ggml-org/Kev-4B-GGUF/`。
原始 GGUF、旧构建和历史结果保持完整。

| 参数 | 默认值 |
|---|---|
| 地址与接口 | `127.0.0.1:8012/v1/systemone` |
| 上下文 | 8192 tokens |
| Metal 内存上限 | 12 GiB |
| KV | BF16 |
| 请求队列 | 1 |
| Prefill 分块 | 编译值 2048 |
| ANE | 关闭 |
| Drafter、lookup、compact、fast-topk | 关闭 |
| 磁盘缓存 | 关闭 |

覆盖项使用启动参数。例如，`--kv-format q4h` 切换 KV，`--port 8015` 修改端口。
`--ane on` 请求启用原有 ANE FFN 路径。本轮真实模型验证使用 GPU，尚未验证 Kev 的 ANE 对照。
启动器拒绝开启 lookup、compact 和 fast-topk。Kev 不生成文本，这些 decode 功能没有执行路径。

```sh
curl http://127.0.0.1:8012/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ggml-org/Kev-4B-GGUF:Q4_K_M",
    "state": "I was charged twice. Please refund one charge.",
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Payments and refunds", "shipping": "Delivery problems"}
      }
    }
  }'
```

System One 支持 `choice`、`score` 和 `noul`，返回 `output_tokens=0`。
Chat、Completions、Responses 和 Messages 的生成请求会明确失败。

## 实现

适配器保存 Kev 类型、512 维 head、三个问题类型的温度和专用模板。
初筛要求 `cls.output.weight/bias`，检查实际形状。
原生加载器识别 `Kev-4B`，加载量化指针头，跳过普通词表输出权重。
缺少 head、形状错误、未知决策类型、非法温度和空模板都会明确失败。
安装清单允许 Kev 不带 drafter，其他家族保留原要求。

骨干复用 32 层、隐藏维度 2560 的 Qwen3.5-4B 路径。
候选项在 `<|box_end|>` 位置读出 key，最终 `<|fim_suffix|>` 位置读出 query。
512 维投影包含两份 256 维向量。FP32 bias、点积和 `sqrt(256)` 缩放保留原语义。
激活、累加和残差规则保持原有实现。

专用请求标记让 `score_tokens` 表示有序输入位置。
跨分块结果属于当前请求，批次换 lane 后也能继续使用。
缓存恢复点限制在首个候选项读出之前。普通生成和 Standard DFlash／DFlash2 保留原路径。

前端读取 GGUF 的 System One 模板，展开结构化材料，并转义用户文本中的特殊 token 写法。
Choice 按标签排序，Noul 按 no、yes 排序，Score 保留等级顺序。
概率映射回调用者标签，应用文件中的温度。
Kev 的 confidence 使用参考公式：Choice 相对均匀分布归一化最大概率，Score 相对均匀中心衡量到众数的距离。
普通模型的原有置信度行为保持不变。

## 验证

验证时的源码基点：`1fe29260f8a5c08115b2a402e1f9bf740bfb6a4a`。验证记录保存当时的工作区快照，最新保存版本以 Git 提交记录为准。
原生构建标识：`src-2c7fa8d7b59e0fd7c030eb51fb7944afe4f298469041c6532ce8b756d26166ca`。
逐文件 SHA-256、最终命令和真实退出码见 `data/kev-4b-integration-2026-10-15/verification.json`。

| 检查 | 结果 |
|---|---|
| 完整 Python | 1343 项，4 项原有跳过，退出码 0 |
| 完整安装器 | 237 项，退出码 0 |
| 原生 CPU、生产与工具构建 | 退出码 0 |
| 完整 Metal | Shader validation 开启，退出码 0 |
| ASan、UBSan、TSan | 退出码 0 |
| 真实 Q8_0 head | 4096 次 FP64 对照，最大绝对误差 0.000248704 |

数值测试沿用既有未舍入 FP64 投影误差界限。
独立算术用例检查 query／key 映射、bias、缩放和非有限结果。

真实 HTTP 输入覆盖分类、情感、Noul、评分、多问题、特殊标记文本、跨分块、255 个选项和精确 8192-token 边界。
每个 KV 完成 9 组，再验证长输入缓存重复和并发 2。
重复的 7231-token 输入实际复用 6016 tokens。
缓存及并发结果使用既有 `1e-3 × max(1, abs(value))` 一致性要求。
所有请求生成 0 tokens。Choice、Score 众数和 Noul 判断与参考端一致，confidence 公式单独核对。

参考程序为本机 llama.cpp 0.6.0-dev，commit `8bb3cf0`，使用 BF16 KV。
普通样本分配 8192 上下文；边界样本需额外预留一个位置，因此参考端分配 8193。
实际输入仍为 8192 tokens，Splash 窗口仍为 8192。
参考端 batch／ubatch 为 8192，Splash 保持 2048 分块。

| Splash KV | 最大概率绝对差 | 最终选项／判断 | Metal 分配峰值 |
|---|---:|---|---:|
| BF16 | 0.00664104（0.664 个百分点） | 全部一致 | 8.534 GiB |
| Q4H | 0.01622687（1.623 个百分点） | 全部一致 | 8.034 GiB |

跨引擎概率没有逐位等价。差异单独记录，不能替代 head 数值断言。
默认保留 BF16 KV，Q4H 可以显式选择。本轮没有性能测试，也没有更改 FFN 缓存策略。
当前 4B 路径仍保留约 4.22 GiB 的常驻 FFN FP16 缓存。

参考程序及两次 Splash 引擎均正常退出，退出码为 0。测试端口 18112、18113 已释放。

## 重建与证据

```sh
cd /Users/ghost233/.codex/worktrees/qwen4b-quantized-ffn/GhostSplash
make BUILD=build/kev-integration-20261015 BUILD_ID_PYTHON=.venv/bin/python check-native-build
```

数据目录保存原始输入、token 位置、响应、状态快照、模型绑定、源代码快照、失败复现和最终通过日志。
先前参考端的 batch／context 限制已按其要求处理，输入及数值标准保持完整。

语义参考 [Kev 模型卡](https://huggingface.co/jaredpalmer/kev-4b)，
输入和返回值参考 [llama.cpp 固定版本](https://github.com/ggml-org/llama.cpp/blob/a962b74dde72a8218b2a276d5bf4f3747ed7814b/tools/server/server-decision.cpp)。
