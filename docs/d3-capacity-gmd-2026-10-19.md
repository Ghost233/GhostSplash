# D3 容量发现兼容修复

已修复 Splash D3 状态接口遗漏上下文容量的问题。正式 GMD 中的 D3 lite 与 D3 mini 均重新发布 `context_length=32768`，ConsoKnowledgeLauncher 原有探测已恢复。本次未修改 GMD 或 Launcher 源码，也未放宽容量保护。

## 原因与修改

D3 的原生 Ready 消息已有实际上下文，但 `/status` 缺少 `maximum_context_tokens`。GMD 无法据此建立运行容量，于是在 `/api/v1/models` 中省略这两个决策模型。Launcher 在推理前查询容量，返回 `jev-context-unavailable`。

`runtime/encoder/Service.cpp` 现在从同一个 `context` 变量输出 `maximum_context_tokens`。该变量也用于模型构造、协议输入限制和 Ready 消息。没有读取 GMD 保存配置充当运行状态，也没有硬编码 32768。

真实原生协议回归增加了“状态容量等于配置及 Ready 容量”的断言，并允许指定验证上下文。Lite 的 32768、4096，以及 Mini 的 32768 均通过，包含原有并发、无效请求、取消和恢复检查。

## 并发语义

| 字段或限制 | 当前值 | 含义 |
| --- | ---: | --- |
| `http.requests.capacity` | 32 | HTTP 接纳上限 |
| 原生 Ready `max_concurrent_requests` | 4 | 原生同时接纳的请求上限 |
| `features.encoder_submit_batch_limit` | 4 | 短请求一次 Metal 提交的批次上限 |
| 长请求一次提交 | 1 | 超过 1024 行时不与其他长请求合批 |
| Launcher `maxConcurrentRequests` | 4 | 原适配代码对 GMD 值做上限约束后的调用并发 |

GMD 目前沿用 HTTP 接纳数发布 `max_concurrent_requests=32`；这不能解释为 32 路 GPU 执行宽度。本次保留该已有行为，并实际验证 Launcher 的 4 路并发。没有给 encoder 虚构 `memory_plan` 或原生执行宽度。

## 正式实例更新

先在 `build/d3-capacity-20261019/engine/` 隔离编译和校验。新旧 Metal 库 SHA-256 相同，计算算子没有变化。

说明重启影响并确认两个 D3 无活动请求后，原子替换本地标准构建的引擎文件，通过正式 GMD 实例界面依次停止、启动 Lite 和 Mini。保存参数保持不变。Qwen 的 PID 与实例身份未变化，embedding 仍在运行。

- 旧构建：`src-c1a598fd103a033d882f1e60723725bbd7560d58493c66e86753418d61615903`。
- 新构建：`src-aa0e660dbe4edc1c4dfd65a7512240f10944227ed26fb0939b5df8bc6fea4b30`。
- 旧引擎、Metal 库与构建身份保存在 `build/d3-capacity-20261019/previous-c1/`。

旧性能记录保持原样；本轮没有执行性能测试。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| 修复前真实 Launcher probe | 退出 1，`jev-context-unavailable` |
| 修复前原生状态断言 | 退出 1，缺少 `maximum_context_tokens` |
| 隔离构建 | 退出 0，arm64 |
| Lite，context=32768／4096 | 各退出 0 |
| Mini，context=32768 | 退出 0 |
| 正式 GMD 容量发现 | 两模型均为 32768 |
| 原 Launcher 单次 probe | 两模型均成功，退出 0 |
| 每模型 4 路并发 probe | 8 个结果全部成功，退出 0 |
| Ruff、Python 语法、架构及 Git 差异检查 | 均退出 0 |

真实 Launcher 反馈循环仍为原来的调用：

```js
import {createModelAdapters} from '/Users/ghost233/code/conso-wiki-mono/conso-knowledge/scripts/model-adapters.mjs';
const result = await createModelAdapters().probe({
  target: 'jev',
  config: {
    url: 'http://127.0.0.1:54841/v1/systemone',
    model: 'vllm-d3-mini-q4-k-m',
    apiKey: '',
  },
});
console.log(result);
process.exitCode = result.ok ? 0 : 1;
```

返回 `ok=true`、`contextLength=32768`、`maxInputTokens=32768`、`maxConcurrentRequests=4`。Lite 使用 `vllm-d3-lite-bf16` 同样通过。

完整前后状态、原生回归、部署身份、真实命令及退出码见 [验证数据](data/d3-capacity-2026-10-19/verification.json)。服务端口在重启后已变化，应通过 GMD 公开入口访问，不依赖记录中的旧内部端口。
