# VAL APEX I MiniPlus V3 启动器

`qwen38miniplusv3splash.sh` 加载用户指定的模型：

```text
/Users/ghost233/.lmstudio/models/IsValorum/Qwen3.8-27B-VAL-APEX-I-MiniPlus-V3-GGUF/Qwen3.8-27B-VAL-APEX-I-MiniPlus-V3.gguf
```

主入口已修正为 `qwen38miniplusv3splash.sh`，默认端口 8008。另一个启动器 `qwen38valapexminiplusv3splash.sh` 也指向同一模型，默认端口 8009。原 EfficientThink 脚本内容作为历史记录保留。
Saluki 启动器 `saluki27biq2mixsplash.sh` 已删除；其原始模型此前已由用户删除，目前没有 Saluki 引擎进程。历史测试、旧排名及公共排名文件保留。

## 启动

在本 worktree 中执行：

```sh
sh qwen38miniplusv3splash.sh
```

| 参数 | 默认值 |
| --- | --- |
| 端口 | `8008` |
| API | `http://127.0.0.1:8008/v1` |
| 监听地址 | `0.0.0.0` |
| 上下文容量 | `262144` |
| KV | `q4h` |
| 请求队列 | `1` |
| Metal 内存预算 | `40G` |
| 磁盘缓存预算 | `50G`，跨启动持久化默认关闭 |
| ANE | 自动探测 share；功耗参考和恢复默认关闭 |
| Drafter | 已转换的 Qwen3.8-27B DFlash2，5 层，block=8 |
| Lookup | 开启 |
| Compact | 默认关闭，显式开启需指定公共排名 |
| Fast top-k | 关闭 |
| 模态 | 纯文本，未加载目录中的 Q8_0 mmproj |
| 原生构建 | `build/startup-checklist-20261014/splash` |

启动后自动显示完整参数清单和实际状态，参数仍可通过命令行覆盖：

```sh
sh qwen38miniplusv3splash.sh --port 8010 --max-memory 32G
sh qwen38miniplusv3splash.sh --ane-ffn-share 0.5
sh qwen38miniplusv3splash.sh --disable-ane
sh qwen38miniplusv3splash.sh --help
```

## Drafter 来源

新目录没有 drafter 文件。启动器复用现有转换产物，没有重新转换或复制：

```text
install/models-local/drafts/miniplus-v3-dflash2-q4km/bf16/
```

原始 drafter 仍在旧目录：

```text
/Users/ghost233/.lmstudio/models/IsValorum/Qwen3.8-27B-EfficientThink-VAL-APEX-I-MiniPlus-V2.1-GGUF/dflash2-qwen38-27b-Q4_K_M.gguf
```

新主模型与旧 EfficientThink 的词表 token、类型及 merge 序列完全一致；主干均为 64 层、隐藏宽度 5120。安装器和原生模型检查接受这份 drafter 配置。这个检查证明结构兼容，未测量新模型上的草稿接受率或加速效果。原转换报告保持原样，没有伪造绑定新模型的转换记录。

## 验证

- Shell 语法、真实 `--help` 和完整安装回归通过。安装测试 232 项，其中 4 项依赖外部客户端的集成测试跳过。
- 新模型安装已离线准备，完整内容校验通过。安装只创建元数据和源文件链接，原 GGUF 与转换权重保持完整。
- 原生 CPU 规划通过：主模型 66 个 image、drafter 6 个 image、81 个 drafter tensor。当前引擎支持新模型的量化布局，无需修改算子。
- 本轮没有启动模型服务、运行生成或测量性能。

真实安装绑定、命令、退出码及源码哈希见 [验证记录](data/miniplus-val-apex-launcher-2026-10-14/verification.json)。

短文件名修正与 8008 释放状态见 [入口修正记录](data/miniplus-canonical-launcher-2026-10-14/verification.json)。本轮没有自动重新启动服务。
