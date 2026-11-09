# H2O-Lightning-4B 文本决策

Splash 支持 `h2oai/h2o-lightning-4b-gguf` 的文本 System One 请求，包括 `choice`、`noul` 和 `score`。当前接入不包含图像和 `box` 问题。

## 启动

在仓库根目录运行：

```sh
sh h2olightning4bsplash.sh
```

启动器直接读取本机文件：

```text
/Users/ghost233/.lmstudio/models/h2oai/h2o-lightning-4b-gguf/h2o-lightning-4b-Q8_0.gguf
```

默认监听 `127.0.0.1:8037`，上下文容量为 40,960 tokens，内存预算为 16 GiB。预算是上限，工作区按实际输入增长。默认使用 Q4H KV，也可指定 BF16：

```sh
sh h2olightning4bsplash.sh --port 8037 --kv-format bf16
```

## 请求

```sh
curl http://127.0.0.1:8037/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "h2oai/h2o-lightning-4b-gguf:Q8_0",
    "state": "客户已经付款，但订单显示未支付，请求退款。",
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "哪个部门应处理这条消息？",
        "criteria": {"billing": "付款和退款", "shipping": "配送和签收"}
      },
      "paid": {"type": "noul", "instructions": "客户已经付款了吗？"},
      "urgency": {
        "type": "score",
        "instructions": "评估处理紧迫程度。",
        "criteria": ["低", "中", "高"]
      }
    }
  }'
```

服务从 GGUF 读取 System One 模板、255 个答案标签和三类问题的校准温度。`noul` 按模型要求先呈现 true，再呈现 false。文本和结构化输入中的特殊 token 拼写会转义为普通文本。

本地发布文件默认使用因果注意力。服务只计算答案标签对应的输出权重行；共享词嵌入的模型从词嵌入表读取这些行。常规决策不执行连续文本生成，DFlash、ANE、Lookup、Compact 和 Fast-topk 均关闭。返回概率不应用额外的 yes/no floor。

图像、`box`、独立的 state prefill 和聊天生成均不在当前支持范围内。输入超过配置的上下文容量时，服务返回长度错误，不自动截断记录。

模型及原始服务约定见 [H2O 模型说明](https://huggingface.co/h2oai/h2o-lightning-4b-gguf)。
