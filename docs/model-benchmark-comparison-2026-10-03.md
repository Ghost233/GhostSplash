# Splash 模型历史评测对照

查阅日期：2026-10-03。比较对象为 Qwen3.8-27B、Swift 1.5 Qwen3.8-27B、Bonsai 2 27B 与 Tiel-Coder-35B-A3B。以下是公开历史结果与本机记录的证据整理，没有重新运行评测。

## 口径

Splash 是运行时。模型能力分数取决于权重、量化、采样、思考预算和评分协议；吞吐还取决于机器、后端、上下文和并发。不同发布者的分数不能合并成统一排行榜。BF16 模型公开成绩也不等于本机 Splash 量化版本的实测成绩。

公开资料的共同结论是：Swift 1.5 主要减少思考 token、改善部分编码与代理任务；Bonsai 2 主要节省权重内存，Splash 实测不保证比普通 Q4 更快；Tiel 主要在 MoE 结构和更短任务耗时上有优势，但小样本代码修复成绩低于原版 27B。这是各来源成对结果的综合，不是一次四模型统一评测。

| 对象 | 本次所指版本 | 主要证据 |
| --- | --- | --- |
| 原版 27B | incoai/Qwen3.8-27B-Splash，另有 UD-Q4_K_M 性能对照 | [Splash 官方性能说明](https://github.com/incoai/splash/blob/main/docs/performance.md) |
| Swift 1.5 | ukisai/Swift-1.5-Qwen3.8-27B；用户文件为 GSQ-RCO IQ3_S-mtp | [UkisAI 模型卡](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b)、[GSQ-RCO 卡](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF) |
| Bonsai 2 | prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0 | [Prism 模型卡](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)、[Splash PR166](https://github.com/incoai/splash/pull/166) |
| Tiel Coder | peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF:UD-Q4_K_XL | [量化发布者模型卡](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF) |

## Swift 1.5 Qwen3.8-27B：同协议对照

UkisAI 同协议 BF16、vLLM 0.27.1、xhigh、五 seeds（0–4）；通常上下文 262144，Terminal-Bench 两者上下文 131072、同 Harbor 构建。采样 temperature=1、top_p=.95、top_k=20。表中差值为百分点，token 是平均思考 token。原始来源：[Swift 1.5 模型卡](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b)。

| 基准 | 原版 Qwen3.8 | Swift 1.5 | 差值 | 平均思考 token 减少 |
| --- | ---: | ---: | ---: | ---: |
| GPQA-Diamond | 88.28 | 88.59 | +0.31 | 41.9% |
| C-Eval | 90.00 | 90.92 | +0.92 | 45.1% |
| IFBench | 73.53 | 72.07 | -1.46 | 38.5% |
| ERQA | 67.45 | 65.40 | -2.05 | 53.9% |
| AIME 2026 | 98.67 | 96.00 | -2.67 | 40.0% |
| HMMT November 2025 | 99.33 | 97.33 | -2.00 | 32.1% |
| LiveCodeBench v6 | 76.76 | 81.71 | +4.95 | 24.5% |
| Terminal-Bench 2.1 | 69.21 | 72.13 | +2.92 | 16.3% |

输出上限不是本机默认值：GPQA/ERQA 100000、IFBench 81920、AIME/HMMT 250000、LiveCodeBench 32768、C-Eval 16384。这些分数不能证明当前 GSQ-RCO IQ3_S 文件具有同样准确率。[协议来源](https://ukisai.com/swift-1-5-27b)

Swift 的优势来自更短的思考链和后训练，公开表中编码、终端代理分数提高；数学、指令遵循、ERQA 有小幅下降。58.5% 是 GPQA **中位数**思考 token 减少；该任务平均减少为 41.9%，不是所有任务统一减少 58.5%。[指标来源](https://ukisai.com/swift-1-5-27b)

### Swift 的“速度”不能直接转换成 Splash tok/s

BF16 模型卡概述写 1.95×；官网/GSQ 卡写 9.18×，但同页只给出一个建游戏演示：104.6 分钟降为 11.39 分钟，恰为 9.18×。缺少统一吞吐、TTFT 与硬件信息，因此两种宣传倍数不能当成本机 Splash 每秒 token 加速。公开各任务更可信的是上述思考 token 降幅。[模型卡](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b)、[官网演示及协议](https://ukisai.com/swift-1-5-27b)

转换作者 SiliconSpecies 实测了另一种 **Swift 1.5 的 Splash 4-bit 包**，其 17.4 GB 权重为 `ukisai/Swift-1.5-4bit-MLX` 的精确 repack，DFlash 2 沿用原版。机器 M5 Max（40 核 GPU、128 GB）、插电，Splash 1.0.2，greedy，prompt 加盐以避免前缀缓存。[转换作者记录](https://huggingface.co/SiliconSpecies/Swift-1.5-Qwen3.8-27B-Splash)

| 测量项 | Swift 1.5 Splash 4-bit | Swift 1.0 对照 |
| --- | ---: | ---: |
| 1024-token 回复 decode | 75–84 tok/s | 74–85 tok/s |
| prefill 2K / 8K | 993 / 920 tok/s | 1045 / 931 tok/s |
| prefill 31K / 62K | 825 / 723 tok/s | 831 / 730 tok/s |
| 四请求总吞吐 | 179 tok/s | 195 tok/s |
| draft acceptance | 0.44 | 0.44 |

该表可作为历史 Splash 速度参考，但既没有原版 Qwen 对照，也不是用户指定的 GSQ-RCO IQ3_S-mtp 文件。[测量来源](https://huggingface.co/SiliconSpecies/Swift-1.5-Qwen3.8-27B-Splash#speed)

### 用户指定 GSQ-RCO IQ3_S-mtp 文件

发布者列出的 IQ3_S 标准文件为 11.77 GB，含 MTP 为 12.12 GB。分层名称代表混合精度，不代表每个 tensor 都是 IQ3_S。标准文件 held-out 测量采用 512-token context；C4 KLD=.041748、CodeParrot KLD=.035447，均为相对 Swift BF16 的概率分布差异，不能当准确率。MTP 文件的解码速度与质量没有单独评测；这些短文本测试也没有验证 32K/262K 长上下文能力。[GSQ-RCO 模型卡](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF)

## Bonsai 2 27B：当前 Prism 第一方对照

GGUF 模型卡当前用 EvalScope + vLLM、H100、相同基础设施/解码/评分、thinking 模式比较 14 项基准。下表在这套协议内可比较，与 UkisAI Swift 表不能直接横向排名。[当前 GGUF 模型卡](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)

| 变体 | 14 项平均 | 与 FP16 差值 | 标称语言权重文件 |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B FP16 | 86.32 | — | 54 GB |
| Qwen3.8-27B UD-Q4_K_XL | 85.18 | -1.14 | 17.6 GB |
| Qwen3.8-27B IQ2_XXS | 72.59 | -13.73 | 7.27 GB |
| Bonsai 2 27B | 84.78 | -1.54 | PTQ1_0 5.95 GB / PQ2_0 7.21 GB |

| 能力类别 | 原版 FP16 | Bonsai 2 | 差值 |
| --- | ---: | ---: | ---: |
| 知识与推理 | 85.55 | 79.86 | -5.69 |
| 数学 | 97.06 | 96.57 | -0.49 |
| 编码 | 89.07 | 89.42 | +0.35 |
| 指令遵循 | 81.25 | 82.66 | +1.41 |
| 代理 / 工具调用 | 76.74 | 74.92 | -1.82 |
| 视觉 | 71.36 | 66.19 | -5.17 |

Bonsai 优势是很小的权重体积下保持数学/编码成绩；此套基准的损失主要在知识推理与视觉。98.2% 是平均成绩相对 FP16 的保留率，不是所有任务准确率，也不等于错误率只增加 1.8%。[结果来源](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)

具体单项 FP16 → Bonsai：LiveCodeBench 90.05 → 90.07、AIME26 94.58 → 95.83、MuSR 79.63 → 70.63、BFCL v3 76.74 → 74.92。该 14 项表没有 GPQA、SWE-bench；LCB 协议也不能视作与 Swift 的 v6 五次重复相同。[单项来源](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf#full-per-benchmark-results)

### Bonsai 发布历史与速度口径

2026-09-17 官方发布新闻给出 **20 项**聚合：原版 85.4、Bonsai 83.9，RTX5090 最高 143 tok/s。这与当前 14 项卡的 86.32/84.78 属于不同结果版本；不能拼成一套表。[官方发布新闻](https://prismml.com/news/prismml-launches-bonsai-2-27b)

当前 GGUF 卡的 llama-bench 使用 batch=1、depth=0、无视觉塔；PQ2_0 M5 Pro Metal TG128=28.1 tok/s、PP512=387 tok/s。卡中 M5 Max 47/765 和 M4 Pro 18/125 的 TG/PP 来自 pre-rotation 旧构建，尚待新栈重测。这些是 llama.cpp Metal 结果，不是 Splash 结果。[吞吐与测量条件](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf#cross-platform-throughput)

## 补充：Swift Bonsai 2 不等于 Swift 1.5 Qwen

现有 Swift Bonsai 启动器指向另一个模型；用户本次确认的 Swift 指 Swift 1.5 Qwen3.8-27B。Swift Bonsai 的 JSON 显示 GPQA/C-Eval 是历史运行时修正评测，IFBench/AIME 是当前 PQ2_0；全套并未在当前分发权重上重跑。GPQA/C-Eval 的平均思考 token 减少约 19%/18%，AIME 完整输出反增 4.3%；39.8% 是 GPQA 中位数减少。作者明确称其为实验研究发布，并承认开放式编码/工具等行为不稳定。[模型卡](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF)、[原始评测 JSON](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF/blob/main/benchmark_results.json)

## Splash 原版 27B 的官方历史性能

Splash 1.0、2026 年 9 月的官方 SPEED-Bench 选定编码提示，1024 输出上限、reasoning medium。机器 M5 Pro 16 核 GPU、48 GB。该段发布基准没有单独注明 KV 格式；同文后面的 GGUF/llama.cpp 精度对照明确使用 BF16 KV，不能将该说明扩展到所有发布基准。[官方性能说明](https://github.com/incoai/splash/blob/main/docs/performance.md)

| 指标 | 原版 Qwen3.8-27B Splash packed Q4 | 来源 |
| --- | ---: | --- |
| decode | 74 tok/s | [官方说明](https://github.com/incoai/splash/blob/main/docs/performance.md) |
| 32K prefill | 363 tok/s | [官方说明](https://github.com/incoai/splash/blob/main/docs/performance.md) |
| cached TTFT | 282 ms | [官方说明](https://github.com/incoai/splash/blob/main/docs/performance.md) |
| 四请求合计 | 170 tok/s | [官方说明](https://github.com/incoai/splash/blob/main/docs/performance.md) |

同文的 GGUF UD-Q4_K_M 单请求 decode：M5 Pro 20 核为 74 tok/s，M3 Max 40 核为 92 tok/s。Qwen3.6-35B-A3B 原版 Splash 的 175/209 tok/s 属于另一个模型，不能作为 Tiel 速度。以上都不是当前 a105662 引擎与本机 INT8 KV 的重新测量。[官方历史说明](https://github.com/incoai/splash/blob/main/docs/performance.md)

## Bonsai 与普通 Q4：Splash 中真正匹配的速度对照

2026-09-26 合并的 Splash PR166 使用默认 DFlash、INT8 KV、greedy、1024-token 代码输出。Bonsai 单请求在 M5 Pro 20 核/48 GB 为 72.7 tok/s；M3 Max/128 GB 为 78.7 tok/s。8K cold TTFT 分别为 15.1/33.6 秒，缓存命中为 .16/.24 秒。[PR166 测量](https://github.com/incoai/splash/pull/166)

更有比较意义的是同一 M3 Max、8 条 SPEED-Bench 提示、相同 drafter、greedy/medium/text-only 对照。TPOT 越低越快。

| 上下文 | Bonsai PQ2_0 TPOT | 原版 UD-Q4_K_M TPOT | Bonsai 相对耗时 | drafter 接受率 Bonsai / Q4 | 来源 |
| --- | ---: | ---: | ---: | ---: | --- |
| 短提示 | 11.4 ms | 11.1 ms | +2.7% | 61% / 67% | [PR166](https://github.com/incoai/splash/pull/166) |
| 8K | 16.9 ms | 14.2 ms | +19.0% | 40% / 52% | [PR166](https://github.com/incoai/splash/pull/166) |
| 16K | 17.5 ms | 15.2 ms | +15.1% | 41% / 53% | [PR166](https://github.com/incoai/splash/pull/166) |

由 TPOT 取倒数推算，Bonsai 的生成吞吐在 8K/16K 约低 16.0%/13.1%。该对照里小权重没有变成更高整体生成速度；原版 drafter 对 Bonsai 的接受率更低，抵消部分权重带宽优势。主要收益仍是约 7.2 GB PQ2_0 权重。[PR166](https://github.com/incoai/splash/pull/166)

早期 PR162 有轨迹 replay 的 8K/16K/32K 速度与 GSM8K 5 题 5/5 试验。这不是完整能力评测；最终 PR166 明确没有重跑 SWE-rebench/GSM 表，不能据此宣称 Bonsai 与原版完整任务分数相同。[PR162](https://github.com/incoai/splash/pull/162)、[PR166](https://github.com/incoai/splash/pull/166)

## Tiel-Coder-35B-A3B：任务成绩与耗时

量化发布者的 Tiel 由 Ornith 1.5 Qwen3.6 MoE 35B-A3B、Sharp 模板及动态量化组成。用户用的 UD-Q4_K_XL 文件约 22.4 GB。MoE 总权重需驻留，但每个 token 只激活部分专家；不能用总参数数目直接推算每 token 工作量。[发布者模型卡](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF)

| 发布者的成对结果 | 原版 / 参考模型 | Tiel | 差值 | 来源 |
| --- | ---: | ---: | ---: | --- |
| SWE-bench-Live，同 25 题 | Qwen3.8-27B 16/25 = 64% | 12/25 = 48% | -16 pp | [Tiel 卡](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF) |
| SWE-bench-Live 任务耗时 | 原版 50.2 分钟/attempt | 8.6 分钟（正文称中位数） | 约 5.8×；耗时口径有限 | [Tiel 卡](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF) |
| MMLU-Pro | Ornith 78.0；原版 Qwen3.6 MoE 85.3 | 73.7 | -4.3 / -11.6 pp | [Tiel 卡](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF) |
| ClawEval 多轮，114 对话 | Ornith 65.3 | 67.2 | +1.9 pp | [Tiel 卡](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF) |

SWE 只有每题单次、小样本；硬件信息缺失，耗时含模型完整输出和代理循环，不能换算成单纯 tok/s。ClawEval 中回答/澄清比例为 4:1。上述结果支持“更快完成尝试，但这组代码修复成功率更低”，不支持把 Tiel 的 48% 与另一作者 Terminal-Bench 的 72.13% 直接排名。[协议与结果](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF)

## 本机历史证据与缺失项

主线程检查当前仓库未找到完整四模型同条件评测产物；现有 `build/build.log`、`build/update-build.log` 属于构建日志。当前机器为 M5 Pro 20 核 GPU、64 GB，启动器设置 INT8 KV、40 GiB Metal、50 GiB disk、queue=4、context=262144。这是本地配置证据，不是公开性能测试的机器/设置。

用户贴出的 Tiel 十条完成日志 decode 为 149.9、128.4、100.8、87.8、71.6、105.5、105.4、83.7、198.4、92.7 tok/s，中位数 103.1，范围 71.6–198.4；TTFT .2–.6 秒，输入 243–1229、输出 16–3183 tokens。短回复、小输入、缓存、采样和当时并发未固定，不能把 198.4 当作稳定大上下文速度，也不能与 1024-token 官方跑分直接横比。来源：本次会话中用户提交的 Tiel 日志。

主线程检查用户指定 Swift GSQ-RCO IQ3_S-mtp GGUF，`token_embd.weight` 实际为 IQ2_S，当前 Splash 不支持该类型，因此此文件没有本机 Splash 成功速度。17.4 GB 的 SiliconSpecies/MLX 4-bit 路径是不同权重格式；其 75–84 tok/s 不代表这个 12.12 GB GSQ-RCO 文件的性能。来源：主线程本地 GGUF 与引擎加载支持范围检查；公开混合精度规则见 [GSQ-RCO 卡](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF)。

## 可验证的结论

- 原版：最适合当同机器/协议对照基线；当前公开数字覆盖编码和一般推理，但本机完整准确率未知。
- Swift 1.5：公开同协议有编码与终端代理提升，并减少思考 token；总完成时间可能缩短，不能保证每秒生成 token 显著增加。数学、指令遵循存在小幅取舍。
- Bonsai 2：权重占用显著减少，Prism 的成对成绩中数学/编码接近原版，知识推理与视觉损失较明显；Splash 同机长上下文对照稍慢于普通 Q4。
- Tiel：用户本机短输入记录的中位速度为 103.1 tok/s，发布者小样本任务耗时更短；同组 SWE 代码修复成功率和知识分数下降。不能据现有证据称其是四者中的最强编码模型。

最终缺少的是当前引擎、同一台 Mac、相同提示/输出上限/采样/缓存与并发条件下的四模型统一速度和准确率测试。

## 2026-10-04 实际请求耗时补充

这次补查找到了实际计时文件，下面的秒数是评测程序记录的请求耗时，**没有用 token 数推算时间**。记录测量于 2026-09-27/28；由独立转换作者 daavidhauser 发布，并非 UkisAI 官方 BF16 五次重复结果，也不是本机 Splash 测试。[转换作者说明](https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen)

环境为单张 RTX 3090 24 GB（功耗限制 250 W）、HyperQwen commit `253c76aea0a240bf7cb2bd3ed92b672c0f258d0d`、patched vLLM 0.27.1、FP8 KV、MTP。配置上下文 150000，单调用输出上限 128000。任务请求并发为 2；思考任务用 xhigh、temperature=1、top_p=.95、top_k=20、seed15027；GSM8K/工具测试禁用思考且 greedy。Qwen 参考是 HyperQwen fast W4A16 AutoRound；Swift 1.5 是 UkisAI AWQ INT4 主体再转 INT8 embeddings/head/MTP linear，两个权重导出并不相同。[环境 JSON](https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen/blob/main/evaluation/environment.json)、[评测清单](https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen/blob/main/evaluation/manifest.json)、[模型格式说明](https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen#changes-from-upstream)

### 原版参考与 Swift 1.5 INT8 heads 的实际计时

每行耗时是所有该类任务的平均请求时间，分数是同批任务成功率；不是整套累计时间，也不是仅成功任务的平均。

| 任务与样本 | Qwen fast 平均秒 | Swift 1.5 平均秒 | 实测耗时减少 | Qwen / Swift 分数 | 原始来源 |
| --- | ---: | ---: | ---: | ---: | --- |
| 数学：GSM8K 前 200 题，关闭思考 | 3.496 | 3.220 | 7.9% | 97.5% / 98.0% | [Qwen JSON](https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen/blob/main/evaluation/qwen-fast-full/summary.json)、[Swift JSON](https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen/blob/main/evaluation/swift15-baseline-full/summary.json) |
| 指令遵循：IFBench 300 题，strict | 114.297 | 63.120 | 44.8% | 74.0% / 73.67% | [Qwen JSON](https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen/blob/main/evaluation/qwen-fast-full/summary.json)、[Swift JSON](https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen/blob/main/evaluation/swift15-baseline-full/summary.json) |
| 编程：LiveCodeBench 100 题子集 | 331.052 | 258.764 | 21.8% | 90% / 89% | [Qwen JSON](https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen/blob/main/evaluation/qwen-fast-full/summary.json)、[Swift JSON](https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen/blob/main/evaluation/swift15-baseline-full/summary.json) |
| 自定义工具调用 / JSON：30 题 | 1.032 | 1.070 | -3.6%（稍慢） | 29/30 / 30/30 | [Qwen JSON](https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen/blob/main/evaluation/qwen-fast-full/summary.json)、[Swift JSON](https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen/blob/main/evaluation/swift15-baseline-full/summary.json) |

分布较偏，LCB 平均约 5.52 → 4.31 分钟，但中位数 90.48 → 62.82 秒；IFBench 中位数 59.35 → 28.45 秒。两者都应同时看平均与中位数。GSM8K 中位数 3.07 → 2.95 秒。[Qwen 原始汇总](https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen/blob/main/evaluation/qwen-fast-full/summary.json)、[Swift 原始汇总](https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen/blob/main/evaluation/swift15-baseline-full/summary.json)

LCB 是固定 v6-era 的 100 个 Python stdin/stdout 问题（34 易、33 中、33 难），作者自定义判题，不能当作完整官方 LCB；GSM8K 也不是 AIME/HMMT 式高难思考数学。IFBench 是指令遵循，不能替代 GPQA/C-Eval 的知识推理测试。[样本与评分说明](https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen#quality)

### 秒数如何测量

评测代码 `stream()` 在发出 `/chat/completions` 请求前记 `time.monotonic()`，完整 SSE 响应结束后记结束时间，返回 `wall_seconds=end-start`。包括请求传输、prefill、等待服务处理、首 token 与完整生成；不包括在本地线程池排队等待开始执行的时间。多次模型调用的工具题把各请求 `wall_seconds` 相加为 `model_seconds`。代码执行判题发生在这个值记录之后，所以这些秒数 **不包含运行生成代码的测试、IFBench 验证等评分时间**。[计时实现](https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen/blob/main/evaluation/code/swift15/evaluate.py)

`aggregate()` 对该类别全部 rows 的 `model_seconds` 取 mean/median，包含答错和截断；截断记错。Qwen 该批两条截断（IFBench、LCB 各一），Swift INT8 一条 LCB 截断；双方请求错误数均为 0。并发 2 时每条请求秒数的总和不等于 GPU 独占计算时间，也不等于套件墙钟总时长。[聚合实现](https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen/blob/main/evaluation/code/swift15/evaluate.py)、[原始计数](https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen/blob/main/evaluation/swift15-baseline-full/summary.json)

### INT4 heads 变体作为额外对照

同作者 Swift 1.5 INT4 heads 的平均请求秒数：GSM8K 3.214、IFBench 58.225、LCB 248.377、tools .959；对应分数 97.5%、72.33%、91%、30/30。此变体恢复了曾暂停的评测（`resumed=true`），汇总包含全部 630 题；`new_run_wall_seconds` 仅覆盖恢复后 272 个新任务，不能拿来当整套 630 题运行时间。[INT4 原始汇总](https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen/blob/main/evaluation/swift15-fast-full/summary.json)

目前找到的这批原始文件有实际编程、基础数学与指令遵循计时；**仍没有 GPQA、C-Eval、AIME、HMMT 的逐类实际请求秒数**。UkisAI 的单游戏实测 104.6 → 11.39 分钟是一个编码演示，无法填补这些分类基准的计时缺口。[UkisAI 27B 演示](https://ukisai.com/swift-1-5-27b)

本次仅新增调研报告，保留所有启动器与模型记录。结束时本地 `main` 和现有 `origin/main` 引用均为 `a1056620e32c54ef533c2ea09910ea0787e07e8a`；实时 GitHub `main` 已变为 `5d9950c64a5d8b20cd5a886e0e470543d013cb39`，因此本地尚未同步这次新提交。调研过程中没有更新运行中的引擎。
