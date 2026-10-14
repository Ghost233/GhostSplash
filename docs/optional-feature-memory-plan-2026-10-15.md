# Compact 与 fast-topk 启动内存规划修复

Qwen3.8 27B MLX 4-bit 同时开启 compact 和 fast-topk 时，启动审计报告 `category_exceeds_plan`。实际 shared decode 为 508.891 MiB，计划只有 238.391 MiB。差额 270.500 MiB 与可选缓冲的实际分配完全一致。

规划入口现接收实际开关。Compact 按当前 head 的量化格式和候选容量计算 plane、metadata 和 token map。Fast-topk 单独预留采样工作区。两项均纳入固定预算，再计算剩余 KV／状态容量。关闭功能时不预留对应缓冲。

计划按页对齐提供分配上限。实际报告使用 `MTLResource.allocatedSize`。小缓冲的实际分配可以小于页对齐上限，审计仍要求实际分配不超过计划。40 GiB 内存上限、系统预留、数值容差和分类审计规则保持不变。

## 回归和实模校验

原生产资源组装与运行时创建在最小测试中复现失败，真实退出码为 1。修复后，全关、compact 单开、fast-topk 单开和同时开启均通过。CPU 规划检查覆盖三个候选容量、27B 的 270.5 MiB 差额及小词表。

完整原生 CPU、所有原生工具编译、源码和构建身份检查通过。ASan／UBSan／TSan 门禁也通过，真实退出码为 0。Metal 检查覆盖 40 种量化格式，保留原 plane 字节、投影精确相等和 FP64 界限。安装模块完成 233 项测试，4 项跳过；启动器与状态报告完成 20 项测试。最终命令和真实退出码见 [验证记录](data/optional-feature-memory-plan-2026-10-15/verification.json)。

真实 MLX 模型在隔离端口 18110 启动，实际上下文 262,144、Q4H、队列 1、内存 40 GiB。Compact、fast-topk 和 lookup 开启，ANE 份额 0.4182、功耗参考自动测量、恢复阈值 50%。预热后到达 Ready，内存审计通过。Shared decode 计划为 533,610,880 bytes。Compact 完成 786,432 次比较，最大误差为 0。

一次 32-token 短采样请求成功，日志记录 fast-topk 执行 11 次。本次 ANE 功耗参考有效，为 3.382349629 W；恢复配置开启。这个值是工作负载参考功耗，不能解释为硬件最高功耗。没有强制触发退避恢复，也没有性能比较。

## 启动方式

启动器已改为使用 `build/optional-memory-fix-20261015/splash`。原构建 `build/general-integration-20261014` 保留。

```sh
sh qwen3827bmlx4bitsplash.sh
```

测试服务正常退出，真实退出码为 0，18110 端口已释放。原模型、drafter、公共排名和旧测试结果保持完整。原始失败、修复后日志、状态快照和源码补丁均保存在同一证据目录。
