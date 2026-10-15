# 合回 ghost 与清理 worktree（2026-10-15）

将 `codex/qwen4b-quantized-ffn` 的 `b20accfc5df6f72ed03425f8c388f24f092703ef` 合并回主工作区的 `ghost`。原 `ghost` 为 `9d792904949aece261420f76e01253253518d101`。

三处冲突分别涉及 Q2_0 的支持列表、README 的上游能力归属和 4B tied head 测试，均保留已验证的实验分支内容。整合后的 `runtime/`、`server/`、`install/`、`dev/` 与 Makefile 逐字节等同于 `b20accfc`。

8 个此前指向实验构建的启动器改为默认使用当前 checkout 的 `build/splash`。模型路径、上下文、内存、KV、ANE、lookup、compact、fast-topk 等参数不变；已有 `--binary` 显式选择能力保留。

## 主工作区产物

- 当前引擎：`build/splash`、`build/splash.metallib`。
- 验证构建：`build/merge-upstream-20261015/`。
- Native build ID：`src-d22fd38e821706a8290ca95f7d89c1be5a97b764ef5f3728cd0aa9c52fc29704`。
- Standard DFlash：`install/models-local/drafts/qwen35-4b-dflash/`。
- MiniPlus DFlash2：`install/models-local/drafts/miniplus-v3-dflash2-q4km/`。

MiniPlus 转换产物仍为原先的 BF16 safetensors，SHA-256 为 `db45fb5626822cdcad6190a944221b62d805aab7b175892a89b2936031d0d5c5`。原有 Coder390 draft 保留。

## worktree 数据保全

本仓库两个附加 worktree 通过 Codex 的可恢复归档机制移除，保留主工作区 `/Users/ghost233/Ghost233Code/GhostSplash`。Git 分支与历史保留。

删除 checkout 前，已把主工作区原构建／模型安装目录，以及两个 worktree 的构建、模型安装目录、虚拟环境和未跟踪基准文件保存在：

```text
build/worktree-preserved/20261015-ghost-integration/
├── ghost-before-integration/
├── qwen35-4b-optimization/
├── qwen4b-quantized-ffn/
├── preservation.json
├── file-sha256.json
└── active-assets.json
```

使用 APFS clone 保全，59,843 个普通文件共 16,416,252,446 逻辑字节逐项通过 SHA-256 对照，符号链接原值也已核对。逻辑字节数不等于新增物理占用。

旧优化 worktree 的三个未跟踪文件位于对应备份的 `untracked/` 下：`dev/tools/qwen4b_baseline.py`、`dev/tests/tools/test_qwen4b_baseline.py`、`qwen4b-baseline.sh`。主目录原有三个未跟踪文档保持原路径和内容。

历史安装记录和日志保留当时的路径，供复核与恢复。当前运行使用主目录中新准备的安装记录。保全目录属于本机资产，不上传 Git；Git 管理的源码、原始测试数据和性能记录随分支整合。

## 验证

之前在相同机器完成的完整原生 CPU／Metal、ASan／UBSan／TSan 数值验证，仍对应相同的运行时和测试输入；整合没有改动这些源码。保留该轮完整日志，并在主工作区重新构建、核对 native build identity，执行完整 Python 发现、启动器测试、静态检查与实际模型迁移检查。

实际模型检查使用默认启动器，覆盖 Kev-0.8B、Julia-1、Laya 的启动、回答、重复请求、原数值容差和正常退出，确认实际使用主目录的 `build/splash`。本轮没有运行模型性能矩阵。

命令、退出码与源码一致性说明见 [验证记录](data/ghost-worktree-integration-2026-10-15/verification.json)。归档工具返回的恢复信息及最终分支同步结果保存在本机保全目录中。

最终主目录验证：完整 Python 发现 1,408 项、4 项既有跳过，退出 0；启动器 13 项、全部 shell 语法、compileall、Ruff、架构、空白、重建及构建身份检查均退出 0；三个真实模型的默认启动、回答、重复请求和正常退出检查均通过。

首次 Python 运行因主目录旧开发环境缺少锁文件已有的 `gguf==0.19.0`，转换工具测试模块导入失败。通过仓库的 `make install-development` 同步后，先重跑该模块 4 项测试，再完成全量重跑；没有改动依赖锁文件或测试标准。首次失败与最终成功日志均保留。
