# 增量预填充内部接口

这份接口供 Ghost Model Deck（GMD）连接 GhostSplash。当前实现仍在开发中，不能据此宣布 `gmd-prefill/1` 已可用。真实 Tiel Coder 与 DFlash2 的验证尚未执行。

## 服务身份和请求绑定

`GET /v1/splash/runtime` 返回当前 HTTP 服务实例、原生运行代次和预填能力。未就绪时 `runtime_generation` 为 `null`，各项能力为 `false`。这个接口遵循服务原有认证。

```json
{
  "service_instance": "e35c5214a396a14bdc468e12",
  "runtime_generation": 1,
  "prefill_capabilities": {
    "schema": "splash-prefill/1",
    "operation_tracking": true,
    "chat_zero_budget": true,
    "systemone_state": false
  }
}
```

服务实例由每次 HTTP 服务启动生成。原生进程重启时，运行代次变化。GMD 应把两者与自己的运行代次绑定，且在请求前核对。

`chat_zero_budget` 表示当前实例提供文本零预算入口。`systemone_state` 表示实例支持因果状态预填，且当前 tokenizer 和模板的探针验证了状态输入与后续问题共享部分实际 token。D3 和双向编码器返回 `false`。探针不证明所有输入都会命中，也不证明决策准确率或草稿模型活动。客户端仍须核对实际模型能力与每次原生结果。

探针失败时，`systemone_reason` 区分 `prefix_not_shared`、`probe_error`、`noncausal_model` 和 `not_ready`。探测异常只记录异常类型，不记录输入。为保持同一运行实例的能力稳定，探测结果在该 HTTP 服务内不变；需要重启服务才重新探测。

要跟踪一次 Chat Completions 或 System One 操作，提交请求时携带：

```http
X-Splash-Operation-ID: operation-123
X-Splash-Operation-Token: <64 位十六进制随机凭证>
X-Splash-Service-Instance: e35c5214a396a14bdc468e12
X-Splash-Runtime-Generation: 1
X-Splash-Operation-Kind: prefill
```

操作 ID 长度为 1～128 个字符，只接受字母、数字、`.`、`_`、`~` 和 `-`。凭证须为 256 位随机值，以 64 个十六进制字符表示。服务只保存凭证摘要，不记录凭证或请求正文。每次操作使用新 ID 和新凭证。

服务在读取请求正文前注册操作。重复 ID 返回 409，不能再次执行；GMD 负责公开接口的幂等重放。无效身份返回 400；服务实例不匹配返回 409；原生代次不匹配时，在写入原生请求之前返回 409。跟踪登记表满时返回 503。未登记时，服务仍按原有接口处理普通请求。

操作类型为 `prefill` 或 `consume`。文本零预算和 System One 的 `prefill_only: true` 使用 `prefill`；最终生成和判断使用 `consume`。服务读取正文后、提交原生请求前核对类型，不一致时返回 400。

## 预填充与结果

文本请求使用 `POST /v1/chat/completions`，把 `max_tokens` 或 `max_completion_tokens` 设为整数 `0`。两个字段同时出现时，零预算须一致。零预算暂不支持 `stream: true`。后续每次都发送完整累计 `messages`；需要生成时使用新的操作 ID 和正数输出预算。

支持的因果决策模型使用 `POST /v1/systemone`，提交完整累计 `state` 和 `"prefill_only": true`。预填请求可以省略 `questions` 或传空对象；非空 `questions` 返回 422。需要判断时发送完整状态和正常的 `questions`。D3 lite、D3 mini 和双向编码器拒绝状态预填，普通判断仍可使用。

探针只能覆盖样例。对使用操作跟踪且声明支持状态预填的判断请求，服务还会核对每道题的实际 token 前缀。题目改变了状态前缀时，服务在提交任何原生题目前返回 422。未使用操作跟踪的普通判断保留原有行为。

同一次输入应保持模型、缓存盐和影响模板的参数一致。缓存按照实际 token 前缀查找，追加文本可能改变连接处的分词。预填不代表整个输入都有可复用快照，也不代表缓存已经写入 SSD。

## 查询和取消

使用相同凭证查询 `GET /v1/splash/operations/{operation_id}`，或请求取消 `POST /v1/splash/operations/{operation_id}/cancel`。查询和取消同样遵循服务原有认证。缺少凭证、凭证不符或找不到记录时统一返回 404。404 不能证明原操作已停止。

以下是完成后查询的结构示例，数值不代表实测：

```json
{
  "operation_id": "operation-123",
  "service_instance": "e35c5214a396a14bdc468e12",
  "state": "completed",
  "terminal": true,
  "lease_released": true,
  "children": [{
    "question_id": null,
    "native_request_id": 42,
    "runtime_generation": 1,
    "state": "length",
    "prompt_tokens": 1200,
    "completion_tokens": 0,
    "matched_prompt_tokens": 900,
    "deduplicated_tokens": 0,
    "lease_released": true
  }]
}
```

`native_request_id` 来自原生运行时，不是 HTTP Job ID。`question_id` 对应 System One 的题目；每道实际执行的题目各有一个子请求。无需原生执行的确定性题目没有子请求。

`prompt_tokens` 是原生请求的输入长度。`matched_prompt_tokens` 来自原生 Start 事件；未收到 Start 时省略。`deduplicated_tokens` 来自 Done 事件。两者各自保留，不把输入长度减去命中数写成实际新计算量。未知的草稿命中、可复用边界和 token 摘要也不填写。

取消响应表示取消请求已受理，不保证计算已经结束。只有查询返回 `terminal: true` 和 `lease_released: true`，才能据此确认已观察到该操作的原生终态并释放相关请求占用。`state: unknown` 表示缺少终态证据，不能接续同一缓存。HTTP 断连、超时、GMD 本地 permit 释放和服务记录消失都不能代替这项确认。

已确认结束的记录在保留至少 1 小时后可因容量压力移除。进行中和未知状态的记录不会因为保留期到期而移除。进程重启会丢失所有记录；调用方必须把旧操作视为未知。查询返回 404 时不要盲目重试旧操作。

GMD 保存完整终态证据后，可携带同一凭证调用 `DELETE /v1/splash/operations/{operation_id}`，确认回收记录。成功返回 `{"acknowledged": true}`。在途或未知记录返回 409，凭证错误或记录缺失返回 404。确认回收后，GMD 不得重用该操作 ID；公开接口的幂等记录由 GMD 保管。

登记表最多保留 4096 条。`prefill` 操作最多占用 3584 条，其余 512 条留给 `consume` 操作。持续预填达到上限后，最终生成和判断仍可使用预留容量。无法回收且各自容量耗尽时返回 503，不得将未知操作当作已释放。

本次源码把原生终态事件放在模型 lane 和缓存请求释放之后。受控测试覆盖完成、在途命令取消、恢复中取消和错误凭证。原生协议版本从 11 升为 12，旧原生二进制不能与新服务通信。部署时必须用同一源码构建服务和原生二进制。单靠 HTTP 200、`state: completed` 或零输出，也不能证明后续请求命中用户材料。

请求取消后，已提交的共享 KV 磁盘读取可能继续由缓存管理。请求占用释放不表示所有共享后台读取已经结束。隔离调度测试覆盖了读取仍在途时立即提交后继请求；后继请求等待共享读取，而不是提前使用未完成的数据。该测试使用模拟存储，不能代替真实磁盘和 GPU 验证。
