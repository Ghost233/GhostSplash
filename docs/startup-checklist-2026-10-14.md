# 启动参数与实际状态清单

服务启动时自动输出两份清单，不需要新增开关。

1. `Startup configuration`：在加载 tokenizer 和模型前列出全部服务启动参数，包括默认值、文件路径和开关。参数名可直接用于修改启动命令；API key 只显示开启或关闭，不打印密钥。
2. `Startup effective state`：在 `Ready` 后读取一次原生状态，确认实际上下文、并发容量、KV、ANE、功耗恢复、drafter、prefill 块、FFN 缓存及可选功能。不会为了报告加载排名文件，也不会运行额外的模型投影或性能测试。

## 状态含义

| 日志状态 | 含义 |
| --- | --- |
| `on` | 功能已启用 |
| `on (conditional)` | 路径已启用，是否使用取决于请求条件；不代表每轮都会使用 |
| `off` | 已关闭 |
| `validated` | Compact 已构建并通过数值与映射校验 |
| `fallback` | ANE 已停止，回到 GPU，后面附原因 |
| `unavailable` | 配置已启用，但当前条件不能满足，例如相对功耗恢复缺少有效参考 |
| `error` | 启动或验证出错，保留原错误及退出行为 |
| `unknown` | 当前构建未报告该字段、状态请求失败或快照过期；不会据此宣称功能开启 |

`Native concurrent capacity` 是引擎支持的并发容量，`--queue-size` 是前端允许在运行或等待的请求数量，两者分别列出。

Compact 覆盖率回退和请求条件回退分别列为 `coverage_fallbacks` 与 `request_fallbacks`，不当作文件错误或数值错误。关闭时实际 capacity 为 0；配置中的 capacity 仍显示预设值。

ANE 相对功耗恢复只有有效参考才能使用；绝对功耗阈值不依赖该参考。恢复日志保留原生阶段，例如 `running`、`cooldown`、`reference_unavailable`、`unknown_power`、`waiting_low_power` 或 `fault_stopped`。

## 启动和调整

下面三个启动器默认使用新构建 `build/startup-checklist-20261014/splash`，其他参数沿用各自已有值。旧构建保留。

```sh
sh qwen38miniplusv3splash.sh
sh qwen38valapexminiplusv3splash.sh
sh swift1527biq3ssplash.sh
```

Saluki 启动器已按用户要求删除。其历史验证记录与公共排名保留。

改变参数仍通过启动命令。例如：

```sh
sh qwen38miniplusv3splash.sh --ane-ffn-share 0.5
sh qwen38miniplusv3splash.sh --disable-ane
sh qwen38miniplusv3splash.sh --compact-draft-head on \
  --compact-draft-ranking compact-rankings/qwen35-248320-dev-seed.splash
```

以下为格式示例，不是此次运行大模型产生的状态：

```text
Startup configuration · on/off are configured flags; effective state follows Ready
  --ane-ffn-share = auto
  --compact-draft-head = off
  --compact-draft-ranking = off / unset
...
Startup effective state · snapshot at Ready; fallback is distinct from error
  ANE = on; share=0.4118; minimum_rows=512; reason=...
  ANE power reference = unavailable; watts=0; reason=background ANE load was not quiet
  ANE recovery = unavailable; phase=running; reason=not sampling
  Drafter = on; kind=dflash2; layers=5; block=8; tokens=7
  Prefill block = 2048 tokens (compiled)
  Prompt lookup = on (conditional); reason=none
  Compact = off; capacity=0; ...
```

启动失败会输出 `Startup state = error; stage=...`，明确失败发生在 HTTP 绑定、tokenizer、模板、原生引擎初始化还是前端准备阶段。原生引擎的具体错误继续保留，例如 compact 排名损坏或数值验证失败。失败后尚未完成的阶段标为未确认。

## 验证记录

本轮只做正确性、编译和状态展示验证，没有启动生产模型或性能测试。
完整命令、源码及构建身份、真实退出码和日志见 [验证记录](data/startup-checklist-2026-10-14/verification.json)。

- 日志验收覆盖完整参数、密钥隐藏、日志换行转义、实际关闭、ANE 回退、无效功耗参考、绝对阈值恢复、compact 校验失败、覆盖率回退和状态未知。
- 服务启动集成验收覆盖 `Ready` 后展示实际状态及原生启动失败后的资源释放。
- 原生状态验收覆盖加载后的 drafter 信息、compact 数值验证与回退计数；旧的 JSON golden 保持不变。
- 原生输出新增 `startup_features` JSON，服务运行后也可从既有 `/status` 查询实时状态。控制台清单是启动时快照，之后变化以 `/status` 和原有回退日志为准。
