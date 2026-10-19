# D3 lite 与 D3 mini 本地接入

本次接入文本决策接口 `/v1/systemone`。D3 不生成聊天 token，因此不适用 DFlash、DFlash2、Lookup、Compact 或 Fast-topk。MLX decider 不在本次范围内。

最新有效对照已按用户后续指示改为独立运行 LM Studio 同版本 llama.cpp，实际输入为 4096／32767 tokens。四组平均加速为 2.00、1.78、1.63 和 1.31 倍，详见[独立对照结果](#独立对照结果)。此前的公开 API 测量和精确 32768 异常作为历史记录保留。

## 启动

在仓库根目录运行：

```sh
sh d3litesplash.sh
sh d3minisplash.sh
```

| 项目 | D3 lite | D3 mini |
| --- | --- | --- |
| 原始文件 | `ggml-org/vllm-d3-lite-GGUF/vllm-d3-lite-BF16.gguf` | `ggml-org/vllm-d3-mini-GGUF/vllm-d3-mini-Q4_K_M.gguf` |
| 默认端口 | 8016 | 8017 |
| 上下文容量 | 262144 | 262144 |
| KV 格式 | Q4H | Q4H |
| 内存预算 | 40 GiB | 40 GiB |
| 请求队列 | 1 | 1 |
| ANE 与生成专用功能 | 关闭 | 关闭 |

模型来自 `/Users/ghost233/.lmstudio/models/`。启动器复用原始 GGUF，不转换权重。40 GiB 是预算，不代表启动时立即分配 40 GiB。工作区随实际请求增长。

## 计算路径

D3 复用现有 Qwen GGUF 权重读取、量化矩阵计算和 GDN 算子。完整 attention 使用双向可见范围，GDN 仍按因果顺序递推。GDN 状态在同一层的 2048 行分块间延续，在不同层和请求之间清零。

新增的 Metal attention 使用 MPP、GQA 和在线 softmax。Q4H 保存打包的 4-bit K/V 与分组 scale；计算前读取真实打包数据，生成 BF16 整数操作数工作区。分组 scale 参与 FP32 分数与概率计算，输出恢复 V 的 H64 变换。BF16 操作数工作区仍会占用内存，不能把 Q4H 缓存大小当成 attention 的全部内存。

服务只收集当前模型输出 head 的决策代码行，最多 255 行，补齐至 256 行。读取位置是序列的最后一个 token。最后一层可只计算该位置的 query 与 FFN，避免计算其余位置的最终输出。原始模型权重保留量化格式，不创建全模型 FFN FP16 常驻缓存。

`--encoder-option-queries off` 可关闭最后一层的 query 精简，作为正确性对照。`--encoder-submit-batch-size 1/2/4` 控制一次 Metal 提交包含的短请求数；请求复用工作区，输出槽独立。两项均沿用现有 encoder 启动参数。

## 正确性验证

基线源码为 `16f15a4ff3e005973938335531443c8582ec9450` 加本次未提交改动。最终源码、二进制哈希和真实命令退出码另存验证清单。

- 独立 FP64 attention 对照覆盖完整 query、最后 query、跨块长度，以及 Q4H 打包数据和 scale。
- 真实模型 head 每个比较 2040 个投影值。Lite 最大绝对误差为 `2.6077e-08`，Mini 为 `6.79847e-05`。
- 真实模型完整与精简路径、重复运行均覆盖 37、129、2049、4096 和 32768 tokens，保持原 `1e-3 * max(1, abs(reference))` 标准。
- 两个模型的原生协议控制测试均通过，包含四并发与串行对照、无效请求恢复、取消和后续请求可用性。
- HTTP 校验覆盖 128、129、2048、2049、4096 和 32768 tokens。两个模型的 4K 结果与 LM Studio 选择一致；结构化 state 和 choice、noul、score 均通过。不同 KV 格式的完整概率不要求逐位一致。4K 及边界用例的 choice 最大概率差分别为 Lite `0.00015587`、Mini `0.00005506`。
- 归档前的中文结构化输入补测发现 Jinja 默认 JSON filter 会转义中文与 HTML，并排序对象字段。已改为与模型/HF 相同的 UTF-8 JSON 序列化。复现测试先失败、修复后通过；两个模型的中文、嵌套 criteria、结构化 instructions 和三类问题真实 API 补测均通过，token 数分别为 120、104、115，与 LM Studio 一致。
- Python 门禁为 1412 项测试、4 项跳过，退出码 0。构建、架构检查、sanitizer 和完整 Metal 门禁均通过，退出码 0。后续新增的两项测量隔离测试也单独通过。中文序列化修复后的完整服务器回归为 930 项，退出码 0。

## 性能对照条件

双方使用相同 GGUF，上下文配置值均为 262144。Splash 使用 Q4H；参考使用 Q4_0。统计 HTTP 决策请求的总延迟与输入吞吐量，不虚构 decode 速度。每组预热后交错测量三轮，并检查实际输入 token 数、答案结构及概率。下面分开保留 LM Studio 公开 API 与后续独立 llama.cpp 的记录。

LM Studio 的物理 batch 最初为 512，无法容纳 4K 非因果请求。测试将这两个模型的运行实例重新加载为 eval/physical batch 32768，保留四槽、Flash Attention 和全 GPU。未修改其他模型或保存的用户配置。原生进程已确认 K/V 均为 `q4_0`。

LM Studio 32K 请求此前多次触发固定的 30000 ms System One 协议超时。超时样本不计为完成，也不用于推导吞吐量或加速倍率。最终重试结果另列。

内存使用采样的进程 physical footprint。Splash 包含 Python 服务及原生进程；LM Studio 仅统计相应模型原生进程，不含共享 GUI。LM Studio 同时加载模型配套 mmproj，而本次 Splash 为纯文本路径。表中内存受这些范围差异影响。
LM Studio 保留原来的四槽配置，双方实际都只有一个活动请求。内存差异包含输出 head 收集、工作区设计、槽位和 mmproj，不能单独归因于 KV 格式。

### 4K 三轮结果

用户确认接受当前桌面负载。正式采样前 GPU 曾持续约 60%～70%，本轮属于诊断性对照。脚本继续检查其他推理活动，并保留 GPU 和分页计数。未检测到其他推理活动不能证明桌面负载完全一致。

吞吐量按 `4096 / 三轮平均延迟` 计算。内存取三轮中的最大采样值。

| 模型 | 后端 | 平均延迟（秒） | 输入 tokens/s | 峰值内存（GiB） |
| --- | --- | ---: | ---: | ---: |
| Lite | LM Studio | 1.0998 | 3724 | 2.83 |
| Lite | Splash | 0.3857 | 10621 | 2.17 |
| Mini | LM Studio | 3.6455 | 1124 | 4.93 |
| Mini | Splash | 1.7734 | 2310 | 3.54 |

本轮 Lite 的吞吐倍率为 2.85 倍，Mini 为 2.06 倍。原始数据见 `build/d3-integration-20261019/performance-4k-desktop/results.json`，真实退出码为 0。两者的上下文、KV 格式、实际 token 数和答案结构均已核对。
性能输入使用英文字符串，不经过上述 JSON filter。修复后重新提交原 4K 性能输入，两个模型的概率变化均为 0。原性能结果保留；修复前的源码快照另存在 `build/d3-integration-20261019/source-state-before-unicode-fix.patch`。原生二进制与 Metal 库没有变化。

最初按 GPU ≤ 50% 等待的运行没有产生样本，随后按用户确认取消，保留了原始记录。该次不计为通过，也不混入三轮结果。

### 32K 三轮结果与参考接口限制

Splash 使用同一参数完成三轮，真实退出码为 0。

| 模型 | 平均延迟（秒） | 三轮范围（秒） | 输入 tokens/s | 峰值内存（GiB） |
| --- | ---: | --- | ---: | ---: |
| Lite | 7.4663 | 7.2176–7.9560 | 4389 | 2.80 |
| Mini | 27.3057 | 26.8561–28.0052 | 1200 | 5.07 |

完成双方 4K 测量后，重新请求实际 LM Studio 的 32768-token 接口。Lite 在 30.030 秒、Mini 在 30.008 秒返回 HTTP 504，错误均为 `Engine protocol systemOne request timed out after 30000ms`。因此这份公开 API 记录没有有效的 LM Studio 32K 吞吐量或加速倍率。后续经用户授权执行的独立 llama.cpp 测量另列，不混入上述数据。

两个失败窗口的原生进程最大采样值分别约为 8.75 GiB 和 11.84 GiB。窗口包含请求附近的准备时间，且请求未完成，不能作为成功运行的完整峰值。超时后已重载这两个参考实例，恢复为 idle、queued=0；本轮 Splash 测试进程已退出，端口 18145 已释放。

可核对的汇总和原始 JSON 已保存到 [data/d3-integration-2026-10-19](data/d3-integration-2026-10-19/)。`summary.json` 保存平均值与范围，`reference-32k-retries.json` 保留两个超时及恢复结果，`verification.json` 记录源码、构建身份、文件哈希和真实退出码。

原始构建、数值日志、API 回应和性能记录保存在 `build/d3-integration-20261019/`；旧构建位于其 `baseline/` 子目录。原始 GGUF 和历史结果保持完整。

## 独立 llama.cpp 对照

用户随后明确要求直接运行 llama.cpp，沿用 LM Studio 刚才的参数，并卸载了 LM Studio 中的模型。本组使用其原二进制：`llama.cpp-mac-arm64-apple-metal-advsimd-2.60.0/llama-server`，版本输出为 `0.7.0-dev (build 1, commit 2a66703)`。不读取 LM Studio 私有认证参数，也不重新加载其模型实例。

| 参数 | 独立参考设置 |
| --- | --- |
| 主模型与 mmproj | 原始 GGUF 及配套 BF16 mmproj |
| `--ctx-size` | 262144 |
| `--parallel` | 4 |
| `--batch-size`／`--ubatch-size` | 32768／32768 |
| `--cache-type-k`／`--cache-type-v` | q4_0／q4_0 |
| `--flash-attn` | on |
| `--n-gpu-layers` | 999999 |
| `--threads` | 13 |
| 监听 | 127.0.0.1:18144，测试结束后退出 |

llama.cpp 日志明确显示四槽下 `n_ctx_slot=65536`，因此这里的 262144 是四槽总量；Splash 的单请求上限为 262144。本轮最大输入约 32K，两边都未截断。此处复现 LM Studio 的实际参数，没有把参考偷偷改成单槽。

### 精确 32768 的异常记录

公开接口单槽与四槽、`/v1/systemone` 与 `/v1/decisions` 均复现 30 秒错误。独立参考的精确 32768 请求也持续数分钟，进程采样显示等待 `ggml_metal_synchronize`／Metal command buffer；这个探针随后主动取消，驱动退出码为 1，清理使用了 SIGTERM。该次不是有效速度样本。

同配置的 32767 输入可以完成。按用户原始“32768 左右”的要求，正式长输入对照采用 **32767 个实际 tokens**，短输入仍为 4096。两个后端处理完全相同的请求，模型、量化、上下文、batch 与并发设置均不变。精确 32768 的异常保留，不能用 32767 的结果宣称它已被修复。

### 独立对照结果

正确性阶段先完成 Lite，再完成 Mini，退出码为 0。随后执行两种长度各三轮交错测量，退出码为 0。每一轮配对均为 Splash 更快；平均吞吐量按 `实际输入 tokens / 三轮平均延迟` 计算。

| 模型 | 实际输入 tokens | 后端 | 平均延迟（秒） | 输入 tokens/s | 采样峰值内存（GiB） |
| --- | ---: | --- | ---: | ---: | ---: |
| Lite | 4096 | Splash | 0.5810 | 7050 | 2.07 |
| Lite | 4096 | llama.cpp | 1.1628 | 3522 | 2.61 |
| Lite | 32767 | Splash | 10.7853 | 3038 | 2.62 |
| Lite | 32767 | llama.cpp | 19.1674 | 1710 | 7.23 |
| Mini | 4096 | Splash | 3.3897 | 1208 | 3.66 |
| Mini | 4096 | llama.cpp | 5.5090 | 744 | 4.74 |
| Mini | 32767 | Splash | 41.4563 | 790 | 5.02 |
| Mini | 32767 | llama.cpp | 54.3787 | 603 | 10.35 |

四组平均加速倍率依次为 **2.00、1.78、1.63 和 1.31 倍**。当前桌面负载影响明显，尤其 Mini 长输入的 Splash 三轮为 51.21、48.41、24.75 秒，参考为 65.62、56.60、40.92 秒。这是本轮条件下的诊断性数据，不代表机器空闲时的稳定峰值；前后两组报告的绝对速度也不宜直接混算。

性能结束后，以相同参考参数加详细日志执行短正确性检查，确认实际分配为 Lite K/V 各 432 MiB、Mini K/V 各 1152 MiB，类型均为 `q4_0`。日志也显示调用 `kernel_flash_attn_ext_*q4_0*`，两次检查退出码均为 0。此检查没有混入测速样本。

`standalone-summary.json` 保存最终平均值、每轮倍率和内存；`standalone-correctness.json`、`standalone-performance.json` 和 `standalone-commands.json` 保存可复核的请求、完整命令及真实退出码。原 LM Studio 公开 API 数据继续保留在原文件中。

独立对照的正确性与性能命令为：

```sh
.venv/bin/python -m dev.benchmarks.d3_models \
  --reference-mode llama --phase correctness \
  --models lite mini --lengths 4096 32767 --accept-desktop-load \
  --output build/d3-integration-20261019/llama-correctness-near32k

.venv/bin/python -m dev.benchmarks.d3_models \
  --reference-mode llama --phase performance --rounds 3 \
  --models lite mini --lengths 4096 32767 --accept-desktop-load \
  --correctness-report build/d3-integration-20261019/llama-correctness-near32k/results.json \
  --output build/d3-integration-20261019/llama-performance-near32k
```

运行器同时核对原生二进制与 Python 服务源码指纹，拒绝复用不同版本的正确性记录。参考完整命令、原生 `/props`、原始响应、分页统计及子进程退出码写入各组 `results.json`。

## 公开 API 复测入口

复测入口为 `.venv/bin/python -m dev.benchmarks.d3_models`。示例：

```sh
# 凭证仅放在仓库外的私有文件中，不写入源码或命令正文。
.venv/bin/python -m dev.benchmarks.d3_models \
  --phase correctness --lengths 4096 \
  --lmstudio-key-file /private/tmp/your-private-key-file \
  --output build/d3-correctness-4k

.venv/bin/python -m dev.benchmarks.d3_models \
  --phase performance --lengths 4096 --rounds 3 \
  --lmstudio-key-file /private/tmp/your-private-key-file \
  --lmstudio-lite-port LITE_NATIVE_PORT --lmstudio-mini-port MINI_NATIVE_PORT \
  --correctness-report build/d3-correctness-4k/results.json \
  --output build/d3-performance-4k
```

`LITE_NATIVE_PORT` 和 `MINI_NATIVE_PORT` 指 LM Studio 为各模型创建的原生监听端口，用于锁定内存采样进程。HTTP 请求仍发往公开 API 端口 1234。LM Studio 重载模型后需重新确认原生端口。

`--reference-mode none` 可单独验证和测量 Splash；正确性记录的模式、模型、长度和二进制哈希必须与后续性能运行匹配。此模式的结果不能当作与 LM Studio 的有效对照。

本轮诊断性测量额外传入 `--accept-desktop-load`。该参数允许高于 50% 的桌面 GPU 负载，仍等待其他推理活动结束。默认策略保持 GPU ≤ 50%，不会默认接受桌面负载。

归档中的 `lite-reference-kv-proof.log.gz` 和 `mini-reference-kv-proof.log.gz` 保存原始参考引擎日志，使用无损 gzip 保留尾部空格和终端控制字节，未清洗日志内容。未压缩的原文件仍保留在 `build/d3-integration-20261019/`；[归档校验清单](data/d3-integration-2026-10-19/raw-log-archives.json)记录原始字节与压缩文件的 SHA-256。
