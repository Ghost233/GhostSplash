# Tiel MiniPlus 的本地模板存档

`tiel-miniplus-v2.1.original.jinja` 是用户在 2026-10-08 提供的原始文件副本。

- 模型：`IsValorum/Tiel-Coder-35B-A3B-APEX-I-MiniPlus-V2.1-GGUF`。
- 用户提供的来源地址：https://huggingface.co/IsValorum/Tiel-Coder-35B-A3B-APEX-I-MiniPlus-V2.1-GGUF/raw/main/chat_template.jinja
- 本地来源：`/Users/ghost233/.lmstudio/models/IsValorum/Tiel-Coder-35B-A3B-APEX-I-MiniPlus-V2.1-GGUF/chat_template.jinja`。
- SHA-256：`c12441d835690f5099793812ff62b2b7ff5e08049ebdee7a49763eb806959a47`。
- 未核实对应的 Hugging Face commit；这个副本以本地文件及其摘要为准。

`tielminiplus35ba3bsplash.sh` 每次启动读取上述本地来源，在
`build/tokenizer-overrides/tiel-miniplus/` 生成独立的 tokenizer 副本。
原模板要求 JSON 工具调用；当前 Splash 的解析器及工具约束使用
`<function=…>` / `<parameter=…>`，启动器因此改写调用示例、格式说明和历史调用渲染，
并将拒绝后续 system 消息的分支改为原位渲染。其余作者规则保留。

如果本地原始文件丢失，可以从本目录复制回上述路径，也可以将启动器的
`chat_template` 指向本存档。以后作者修改模板时，需要重新核对适配位置；
启动器遇到不匹配的内容会报错。

离线验证覆盖普通对话、工具历史、五档 reasoning effort、后续 system 消息、
六种工具参数类型的流式解析及词表一致性。此次未启动真实模型验证生成质量。
