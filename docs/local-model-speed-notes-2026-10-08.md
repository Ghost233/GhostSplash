# 本地模型测速与性能诊断存档（2026-10-08）

归档日期：2026-10-09。以下记录来自已完成的历史运行，本次归档没有重新发起模型推理。

## Coder390：官方 DFlash2 的 64K 以内测速

硬件：Apple M5 Pro，20 GPU cores，64 GiB 内存。
模型：用户指定本地 Q4LynnStyle-MTP GGUF；纯文本，使用官方 incoai/Qwen3.8-27B-DFlash2，未使用 GGUF 自带 MTP。
启动配置：192.168.64.1:8007；上下文 262144；Metal 上限 40G；SSD 缓存 50G；队列 4。
源码分支 codex/splash-local-compat；commit 4d8e399d647a0e3857aedd2afcec5daffe045bbc。
Native build：src-d062cfd94c98e66f72b9a775ac9afb5149ff7176f8bd2bf679523a71c9d591f1
测速复用 zx-bench/scripts/bench-context-speed.py；仅平台空闲检查改为分块读取完整 JSON。
8K/16K/32K/64K 分别为精确 8000/16000/32000/64000 输入 tokens；每请求输出 512 tokens，冷缓存，temperature=0，ignore_eos=true。
16 组、40 条请求全部完成；最终进程退出码 0；全部 cached=0；同一引擎实例；无取消、模型失败或内存采样缺口。
ANE 从开始到结束保持 split，share≈0.382；系统压力记录包含 normal 和 warning。
中途两次平台历史响应 IncompleteRead 发生在发请求之前，续跑跳过已完成组；原错误保留在 results.json/preflight_failures 和运行日志。
测速完成后 8007 服务已停止。本测试衡量速度，不评估回答质量。

### 全部实测

prefill 为整组输入 / native prefill 累计 wall time；decode 总吞吐为整组输出 / decode_cycle 时间，不是每条请求同时具有此速度。
TTFT 为客户端首输出等待，包含排队；单请求 decode 列为各请求 timing 的平均值。

| 输入 tokens | 并发 | prefill tok/s | decode 总 tok/s | 单请求 decode 均值 tok/s | TTFT 均值 s | Metal 峰值 GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 8000 | 1 | 651.55 | 55.85 | 56.28 | 12.53 | 21.78 |
| 8000 | 2 | 538.01 | 70.30 | 50.12 | 25.70 | 23.17 |
| 8000 | 3 | 427.72 | 56.37 | 33.18 | 45.99 | 25.02 |
| 8000 | 4 | 417.57 | 58.91 | 30.98 | 60.01 | 27.50 |
| 16000 | 1 | 491.33 | 58.75 | 59.30 | 32.83 | 28.23 |
| 16000 | 2 | 445.80 | 66.24 | 46.22 | 57.58 | 30.12 |
| 16000 | 3 | 480.07 | 64.71 | 39.96 | 73.41 | 32.74 |
| 16000 | 4 | 413.39 | 57.37 | 32.97 | 110.44 | 36.36 |
| 32000 | 1 | 423.47 | 57.44 | 57.98 | 76.16 | 37.59 |
| 32000 | 2 | 404.44 | 58.60 | 41.45 | 122.63 | 39.25 |
| 32000 | 3 | 420.77 | 55.59 | 33.39 | 161.55 | 39.25 |
| 32000 | 4 | 414.81 | 63.46 | 38.78 | 214.14 | 39.25 |
| 64000 | 1 | 394.12 | 70.12 | 70.27 | 163.18 | 39.25 |
| 64000 | 2 | 431.34 | 61.52 | 50.53 | 223.52 | 39.25 |
| 64000 | 3 | 389.26 | 68.42 | 52.19 | 336.95 | 39.25 |
| 64000 | 4 | 398.88 | 63.89 | 49.08 | 418.38 | 39.25 |

### 64K 单并发历史对比

| 模型 | prefill tok/s | 单请求 decode tok/s | TTFT s |
|---|---:|---:|---:|
| coder39027 | 394.12 | 70.27 | 163.18 |
| whittle27 | 310.81 | 56.07 | 207.00 |
| original27 | 238.58 | 38.46 | 269.03 |
| tiel_miniplus35 | 1076.30 | 63.36 | 59.73 |

限制：原版 27B 的历史测试第一组即发生 ANE 回退，后续为 GPU，因此其结果不是正常启用 ANE 的基准；35B 数据来自 Tiel MiniPlus。各模型权重、量化与引擎 build 不同，只代表这些实际配置的速度差异。每档仅一组，未统计跨轮波动。

原始运行目录：`build/context-speed-coder390-20261008-042003/`，完整日志仍保存在本地、未纳入 Git。
本次将各组速度、逐请求计数、采样条件和来源文件 SHA-256 归档到 [精简数据](data/local-model-speed-2026-10-08.json)。


## Coder390：作者量化 DFlash2 的 16K 以内检查

当前 `coder39027bsplash.sh` 指向作者发布的 FP8 draft 转成的 BF16 文件，
与上面 64K 测速使用的官方 draft 不同。转换程序执行 E4M3FN 解码、
乘 FP32 per-tensor scale、BF16 round-to-nearest-even；20 个权重张量转换，
61 个 BF16 张量保留。Splash 加载时仍会按其自身路径准备 draft 权重；
这些结果不代表原生 FP8 推理，也没有证明作者 draft 优于官方 draft。

历史检查：单并发、temperature=0、冷前缀、每请求输出 128 tokens。

| 输入 tokens | 单请求 decode tok/s | 完成状态 |
|---:|---:|---|
| 8000 | 38.39 | completed |
| 16000 | 57.76 | completed |

精简数据保存在同一 JSON 的 `coder390_author_draft_16k`。
源文件：`build/coder390-quantized-draft-16k-smoke-20261008/results.json`。
检查期间存在内存压力 warning，短检查与上面输出 512 tokens 的完整测速不可直接作为 draft 优劣对照。

转换入口（目标目录必须尚不存在；权重文件未提交 Git）：

```sh
.venv/bin/python dev/tools/dequantize_coder390_dflash2.py \
  install/models-local/drafts/coder390-dflash2-fp8/source \
  install/models-local/drafts/coder390-dflash2-fp8/bf16
```

原始 FP8 来源的 `source/` 需包含 `model.safetensors`、`config.json` 和
记录模型文件 SHA-256 的 `manifest.json`。当前已有转换结果时不应再次运行这个命令。

## Swift 1.5 IQ3_S：慢速与恢复后的观察

相同模型和 native build：
`src-d062cfd94c98e66f72b9a775ac9afb5149ff7176f8bd2bf679523a71c9d591f1`。
两次均为现有请求的约 8 秒只读观察，没有插入诊断生成请求。

| 指标 | 慢速观察 | 恢复后观察 |
|---|---:|---:|
| 平均每轮解码 ms | 145.49 | 88.99 |
| 完成的解码轮数 | 55 | 90 |
| 输出 tokens | 178 | 234 |
| 窗口输出 tok/s | 约 22.3（178/约 8s） | 29.32（native decode wall） |
| Draft 接收率 | 31.95% | 22.86% |
| 内存压力 | warning | normal |

相邻状态快照的可用内存从约 11.52 GiB 增至约 14.29 GiB；
恢复后活动采样时为约 14.57 GiB。内存值与速度记录的取样时刻不同。
平均每轮耗时降低约 39%，说明观察到的改善包含计算耗时下降，
但不同生成内容、上下文长度及系统负载均可能影响比较。

用户报告重启 ChatGPT 后变快。读取到的 ChatGPT 进程启动时间是 20:18，
Splash 当前实例启动于 20:28，而慢速记录来自 16:24 启动的实例。
因此这不是控制变量实验，不能确认 ChatGPT 是唯一原因、不能认定其存在内存泄漏，
也不能将性能变化归因于某个引擎修复。内存压力改善和系统资源争用是继续排查的线索。

精简数据保留原始观察数值、相邻内存快照、实例启动时间和限制说明。
