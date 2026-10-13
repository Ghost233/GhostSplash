# Swift 27B 的五种 KV 格式测试

## 模型和范围

模型为用户指定的 `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`。
它是 64 层 Dense 主模型，隐藏维度 5120，另有 1 层 MTP。
本轮固定同一份混合量化 GGUF 和本机缓存的 Qwen3.8-27B DFlash2，保留各张量的原始量化语义。
MTP 未参与计算。IQ3_S 是文件名标签，主模型的 192 个 FFN 矩阵使用 9 种量化格式。

仅改变 KV：`bf16`、`int8`、`q4_0`、`q4_0_unrotated` 和 `q4h`。
每种格式测 3 轮，共 15 批，顺序为正序、逆序和旋转两项。
固定 32,768 输入、16,384 实际输出、131,072 上下文容量、并发 1、40 GiB 测试内存上限。
其他计算条件为尾段 splits=16、DFlash2 block=8、ANE 请求开启（share=0.5、最小 512 行）、lookup 和 compact 开启、fast top-k 关闭。
真实 ANE 执行及回退单独记录。现有 27B 路径保留量化 FFN，不使用 4B 专用缓存实验参数。

## 正确性与已知限制

先检查五种 KV 的单请求数值，再补充 Q4H 的 ANE 数值检查。
保留原来的非有限值隔离、分块状态 cosine > 0.999，以及同形状 head 分数相对误差 <= 1e-3。
另检查同 lane 的冷请求重跑决策一致性。
速度测量继续检查实际参数、全部事实、输入和输出数量、内存账本，以及同 KV 三轮的完整 token 轨迹一致。
跨 KV 的输出轨迹可以不同。事实检查不替代数值检查。

BF16 的并发 4 相同输入用例尚未通过：首个输出 token 相同，但各 lane 接受的草稿数量不同。
原始失败保留于 `build/swift27b-kv16k-v1/numeric-bf16-diagnostic/`，原断言没有放宽。
本轮只验收用户请求的 C1，不宣称并发 4、所有采样或缓存恢复行为已通过。

性能启动要求连续 30 秒 GPU 不超过 50%，并且没有检测到其他活动推理或已知 benchmark。
运行中检测到外部活动则取消该次并重试，取消的尝试不会计为通过。
监控无法覆盖所有桌面 GPU 任务，原始慢样本和背景记录全部保留。

## 运行与日志

先安装指定本地目标及配套 drafter。当前性能测量已按用户要求停止，后续先执行独立的正确性入口：

```sh
.venv/bin/python -m dev.benchmarks.swift27b_correctness
```

正确性结果保存于 `build/swift27b-kv-correctness-v1/`。
主日志为 `build/swift27b-kv-correctness-v1.controller.log`，真实退出码保存在同名 `.controller-exit.json` 中。
五种 KV 分别验证 32K 输入：关闭 lookup／compact 时生成 512 个参照 token，开启时实际生成 16K token，严格比较前 512 个 token。
Q4H 再完整重跑 16K 输出，检查全部 token 一致性。所有请求均使用 correctness 模式，结果不输出阶段计时。
数值检查只在命令、源码、产物和权重身份未变时复用。短输入的 ANE 准备检查不证明真实执行，长输入还要检查 ANE 的实际执行次数。
正确性脚本不会恢复性能测量。并发 4 的失败仍单独保留。

原性能准备与测量入口如下，`run` 会进入性能测量：

```sh
.venv/bin/python -m dev.benchmarks.swift27b_kv prepare
.venv/bin/python -m dev.benchmarks.swift27b_kv run
```

默认目录是 `build/swift27b-kv16k-c1-v2`，准备时不会覆盖已有目录。
`plan.json` 保存权重与产物 SHA-256、输入和固定参数；`weight-formats.json` 保存实际权重量化分布。
本轮后台控制输出写入 `controller.log`，进度写入 `progress.json`，各批 token 进度写入对应的 `stderr.log`。
`controller-exit.json` 保存真实退出码，全部成功后生成 `comparison-report.json` 和 `comparison-report.md`。
当前标准参数不会自动修改。

`ignore_eos=true`、greedy 和冷前缀用于固定长度对照。业务请求按任务设置停止条件及输出预算。
阶段耗时排除模型加载和固定预热，首 token、prefill、decode 和请求总耗时分别记录。
Device 和 backend 内存峰值与进程 RSS、整机内存是不同指标。
