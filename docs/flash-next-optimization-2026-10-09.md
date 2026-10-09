# Flash Next 本机优化与 MTP 接入（2026-10-09）

状态：用户要求停止所有测试。8K 已完成；16K 在日志进度 10112/16000 时取消，未完成，不能记为通过。基准测试进程及测试引擎均已退出，8008 端口已释放。

已停止长测的运行构建为 `src-52a1cca194a14a6dff66ee68168d2400a56a9be7528cb6d0c5d25b3c6637c9e9`，prefill batch 为 128 行。工作区另有 512 行批次、缩小临时 logits 缓冲，以及 MTP teacher-only KV/index priming（跳过 query/output/FFN）的改动，已通过 C++ 语法检查，尚未编译运行和验证，不能将本轮结果记为这版改动的验收。

## 本机和配置

- Apple M5 Pro，20 GPU cores，64 GiB 内存。
- 目标：用户本地 Swift-1.5-Qwen3.8-Flash-Next GSQ-RCO IQ3_S，两个 GGUF 分片。
- Draft：用户本地 sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16，32 个 BF16 tensors，单层、512 专家。
- 端口 8008，上下文 262144，Metal 上限 40G，请求队列容量 4。
- 专家缓存：包含 MTP 时 18 GiB；BF16 head 在现有 float 投影接口中展开为 F32，约 9.7 GiB。
- `SPLASH_MTP=0 sh swift15flashnextsplash.sh` 关闭推测解码，权重和内存配置仍相同。
- 现有实验路径不支持前缀状态缓存与工具约束；本轮检查为纯文本基础推理。

## 已修改

1. 允许后续 GGUF 分片省略第一片已有的架构字段，明确架构冲突仍拒绝。
2. Flash Next 不再被通用入口强制要求 DFlash `draft/`；明确指定 draft 时只接受匹配的 qwen4_exp_mtp。
3. 层优先、有界多行 prefill；按专家分桶处理真实行，并按原 routing rank 稳定组合。
4. 修正逻辑宽度与物理 padding，FP32 quantized projection 的 rows 取 8 的倍数，大批次分为最多 32 行的 FP32 子投影。
5. SSD native expert rows 在最多 16 MiB staging 中成组 repack，减少逐 256 行提交及等待。
6. MTP 使用前一位置的完整四条 HC streams：global RMS 后每条 stream 的 fc_hidden，加 fc_embedding。
7. MTP teacher cache、共享 target embedding/head，批量 target verification；拒绝时回放 GDN recurrence operands、恢复 PLE/history/hash 和逻辑 KV 长度。
8. Draft 深度按接收情况在 1–3 间反馈调整；非零温度由 target sampler 实际抽样，接受相同 draft token，未接受的 token 使用 target 的结果。
9. 有界工作通过 async ticket 提交，状态通道继续响应；状态内存报告按已分配对象计量，完成的 counters 在同步点发布。

来源与借鉴边界见 [PR 调研](flash-next-pr-research-2026-10-09.md)。没有直接启用 M5 Ultra 专用 profile，也没有引入全权重常驻方案。

## 已完成的真实短测

| 模式 / 输入 | prefill 时间 | decode 时间 | 说明 |
|---|---:|---:|---|
| 原串行 AR，128 tokens / 输出 16 | 104.87 s | 10.09 s | 优化前源码与构建记录保存在原 results.json |
| 优化 AR，128 / 输出 16 | 41.20 s | 6.21 s | 已加载 draft 但 MTP 关闭；暖 expert cache，与首行缓存条件不同 |
| 最终 async + MTP，128 / 输出 16 | 45.84 s | 10.18 s | 冷 UUID 前缀，包含 teacher prefill，内存峰值 35.40 GiB |

这些数据是不同运行窗口，不是严格 ABBA 平均，不应自动外推为 16K 提速比例。

输出对照：计数和 Python 函数两组 greedy 与优化前 AR 一致。中文解释和 seed=17、temperature=0.7 的样本也在 MTP on/off 间逐字一致；对应接收率为 27/60、22/75。低接收率时 MTP 可能更慢，所以并不承诺所有请求提速。

中间的 128-row 同步版本曾因 native 主循环未在 30 s 内回应 status 而失败；此错误保留在日志，未记为通过。修复 async ticket 后同协议测试完成。

## 检查记录

以下通过项来自之前的运行版本。最终工作区的 512 行与 teacher-only 修改只做了 C++ 语法检查；用户随后要求停止所有测试，因此作为待验证开发检查点保存，未重跑最终运行门禁。

- 原生目录和真实 BF16 MTP config 检查通过，包括 decimal RMS epsilon。
- Flash Next Metal 数值、行 padding、PLE retained-prefix commit 和 runtime admission/suspend/end tests 通过。
- 完整 Python gate：1243 tests，7 skipped；compileall、ruff check、ruff format 通过。
- check-source、完整 native CPU gate 通过。
- 8K：prefill 3606.74 s，decode 64.99 s / 128 输出 tokens，Metal 峰值 35.48 GiB；16K 已按用户要求取消，测试服务已关闭。

8K 这一组请求的专家缓存增量：141093 hits、2039415 misses、2038727 evictions，miss 率约 93.5%；`expert_bytes_read` 增量 1392652800000 bytes（约 1.27 TiB）。这是软件计数的逻辑读取量，不是 SSD 物理 I/O 测量。prefill GPU 计时 503.69 s，占 command wall time 3606.74 s 的约 14%；剩余时间包括 host、权重读取/repack、提交与同步，不能全部归因于物理 SSD。

16K 在约 9.6K 位置的 `iostat -d -w 1 -c 6` 短采样：排除首行启动以来平均值，disk0 的五个 1 s 样本为 384.90、470.66、307.96、318.02、320.16 MB/s。这是全机磁盘吞吐，未做进程归因或全程积分，不能据此断言 SSD 已饱和。权重 repack 的 GPU 命令也不计入上面的 model GPU timer；14% 不是 GPU 总利用率。原始记录：`build/flash-next-128-disk-sample-20261009.log`。

## 本地证据

`build/flash-next-ar-baseline-128-20261009/`、`build/flash-next-optimized-ar-128-20261009/`、`build/flash-next-async128-short-20261009/`；
`build/flash-next-mtp-{on,off}-greedy-20261009.json`、`build/flash-next-mtp-on-varied-20261009.json`、`build/flash-next-varied-parity-20261009.json`；
`build/flash-next-mtp-8k16k-20261009/` 保存本轮精确 8000/16000 输入、单并发、每请求输出 128、temperature=0、ignore_eos=true 的测试结果。

停止前的状态快照：16K 请求已完成部分为 9984 tokens / 4179.49 s，即 2.3888 tok/s；随后最后日志进度为 10112 tokens。该速度是部分 prefill 计时，既非完整 16K 结果，也不包含工作区尚未运行的 512 行与 teacher-only 优化。原始快照：`build/flash-next-16k-progress-speed-20261009.json`。

Git 中归档的精简测量记录（含运行 build ID、源结果 SHA-256、8K 完成结果与 16K 取消状态）：[speed checkpoint](data/flash-next-speed-checkpoint-2026-10-09.json)。原始内存 trace 继续保留在本机 build 目录。
