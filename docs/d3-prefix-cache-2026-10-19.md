# D3 lite / mini 的 KV 前缀缓存验证

D3 lite 和 D3 mini 目前都没有接入现有的跨请求 KV 前缀缓存。两个正式实例和隔离验证进程均报告 `features.causal_prefix_cache=false`；重复完整输入也没有缓存命中。Q4H 仍然正常用于一次前向计算中的 K/V 量化，这与跨请求前缀复用是两个功能。

本次只验证能力和正确性，没有修改引擎、切换模型参数、重启正式服务或进行性能测试。

## 实际验证

使用当前 `build/splash`、原始 Lite BF16 / Mini Q4_K_M 权重、Q4H、上下文容量 32768、单请求串行提交。每个模型、每种长度做六次请求：首次 A、重复 A、同前缀换后缀 B、切回 A、A 更换 cache salt，以及独立新进程中的 B。A / B 只修改末尾的退款／配送内容，其余模板与输入保持一致。

| 模型 | 实际输入 tokens | A/B 公共前缀 tokens | 请求数 | 实际命中 tokens | 跨请求去重 tokens | 重复／独立进程最大分数误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D3 lite | 4096 | 4042 | 6 | 全部 0 | 0 | 0 |
| D3 lite | 32768 | 32714 | 6 | 全部 0 | 0 | 0 |
| D3 mini | 4096 | 4042 | 6 | 全部 0 | 0 | 0 |
| D3 mini | 32768 | 32714 | 6 | 全部 0 | 0 | 0 |

总计 24 次真实请求、16 组数值对照，每组核对全部 255 个答案 code 的原始 logits。采用既有标准：`abs(actual-reference) <= 1e-3 * max(1, abs(reference))`，没有放宽容差。所有请求均正常完成，8 个隔离原生进程均正常退出，退出码 0。改变后缀确实改变了输出分数，不是重复了同一个输入。

这里的“数值校验通过”表示重复执行、切换请求和独立进程的结果一致；不能解释为“KV 前缀缓存校验通过”。实际缓存命中为零。

## 不能直接复用的原因

原始两个 GGUF 均保存 `qwen35.attention.causal=false`。安装器与原生模型加载器都检查这一语义；D3 attention shader 对每个 query 遍历全部有效 key，没有按照 query 位置屏蔽未来 token。

现有生成模型缓存按照前缀 token、图像和 cache salt 匹配 KV 块，再恢复对应的模型状态快照。它依赖“相同前缀的状态不受后缀影响”。D3 从第一个非因果全注意力层输出开始不满足这个条件：同一前缀接上不同后缀，前缀隐藏状态可能改变，随后层的 K/V 和 GDN 状态也可能改变。GDN 自身的递归因果性不能消除前面全注意力层已经引入的后缀依赖。

另外运行了一个实际 Metal 算子检查，覆盖 Lite 的 8 个 query heads 和 Mini 的 16 个 query heads。Q/K 全零，前 64 个 token 的 V 保持零，只把后 64 个 token 的 V 从零改成一。非因果注意力的前缀输出从精确的 0 变为精确的 0.5，所有前缀行／通道均做精确相等检查，退出码 0。这个 BF16 算子检查用于证明后缀依赖；真实模型请求的 K/V 使用 Q4H。

当前实现没有冒险恢复这类状态：D3 的服务直接发送 `matched_prompt_tokens=0`；模型从完整输入重新计算，每层重建 K/V，并清零该层的 GDN 工作状态。通用服务的持久化缓存参数也没有转交给 D3 原生进程。仅添加 `--persistent-cache` 或增加缓存预算不能启用这一能力。

相关源码：

- `runtime/engine/Cache.cpp`：`Cache::extendMatch` 按前缀匹配。
- `runtime/engine/Engine.cpp`：`Engine::completeAdmission` 从匹配状态恢复并记录命中。
- `runtime/model/D3.mm`：完整前向计算和工作区复用。
- `runtime/metal/kernels/shared/d3.metal`：非因果注意力的 key 可见性。
- `runtime/encoder/Service.cpp`：缓存能力标记和零命中的 StartEvent。
- `server/server.py`：D3 原生启动参数分支。

## 后续选择

保持现有 KV 前缀缓存关闭。不能通过把 D3 改为因果注意力来取得缓存命中，这会改变模型语义。

若要进一步开发缓存，可以另行评估完整请求结果缓存：精确匹配完整输入、模型身份和影响结果的选项。它只能加速重复完整请求。第一个非因果注意力层之前的因果计算也存在局部复用空间，但需要单独设计分层缓存与验证，不能宣称现有完整 KV / 状态缓存已支持。

## 复现与记录

源码基点为 `ghost` 的 `16f15a4ff3e005973938335531443c8582ec9450`，包含尚未提交的 D3 实现。实际引擎 SHA-256 为 `311ead4c092fb33eb0eb0fe1d6d91934ef8d1e43b45f31814f3049cad7b2148a`，Build ID 为 `src-aa0e660dbe4edc1c4dfd65a7512240f10944227ed26fb0939b5df8bc6fea4b30`。

实际执行命令：

```sh
.venv/bin/python -u -c 'import runpy; runpy.run_path("build/d3-prefix-cache-20261019/probe.py", run_name="__main__")' \
  --binary build/splash \
  --output docs/data/d3-prefix-cache-2026-10-19/results.json

build/d3-prefix-cache-20261019/suffix-dependency build/splash.metallib

.venv/bin/python -m unittest \
  dev.tests.server.test_d3 dev.tests.install.test_d3_metadata
```

原样执行的 Python 探针文本保存在 `data/d3-prefix-cache-2026-10-19/probe-executed.txt`，SHA-256 与结果内记录一致；`probe.py` 为只整理了导入格式和空白的复现入口。结果保存了每个原生进程的完整命令、启动／结束状态、StartEvent、255 个 logits 和真实退出码。算子源文件、构建日志、正式实例状态、单元测试日志和验证清单一起保留。单元测试共 4 项，退出码 0；本次不重复整仓引擎测试，因为没有修改生产代码。

- [汇总数据](data/d3-prefix-cache-2026-10-19/summary.json)
- [完整请求结果](data/d3-prefix-cache-2026-10-19/results.json)
- [验证命令、来源和退出码](data/d3-prefix-cache-2026-10-19/verification.json)
- [原始执行日志](data/d3-prefix-cache-2026-10-19/probe.log)
