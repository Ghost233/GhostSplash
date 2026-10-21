# Splash 1.4.0 合并记录

本次将官方 `incoai/splash` 的 `1.4.0` 合入当前 `ghost`，本地 `main`
保持官方基线。合并采用上游已经覆盖的实现，保留本地仍独有的功能。
本次不做性能测试，不能据此判断升级后的速度变化。

## 源码基线

| 项目 | 完整提交 |
| --- | --- |
| 原上游基线 | `be3b083c5f413673d44e44881042e875e674e333` |
| 合并前 ghost | `456ee426dc436fcffd890ac291b8492c5eeb9c24` |
| 官方 1.4.0 / 本地 main | `1482f6bbc2abf12375332725302019594150b6cb` |

合并前先将已有的量化 FFN 实验开关及验证记录保存为 `456ee426`。
旧构建保存在 `build/upstream-140-merge-20261021/retained-before-140/`，
原始权重、转换产物和历史测试记录保留。

## 实现取舍

- 使用上游的 64 行量化权重布局、动态 drafter 配置、统一模型内存规划、
  Flash Next、YaRN、SSD 专家加载、稀疏 KV 溢出和 HostStep 提交流程。
- 本地模型启动器改用上游公开的 `prepare_local_model` 和 assembly 生命周期。
  服务不再接受 `--tokenizer`；Tiel 的独立模板通过 `--chat-template` 指定。
- 保留普通 Qwen3.5-4B、Kev、D3、ModernBERT、dense Q4H KV、ANE 自动恢复、
  lookup、公共 compact 排名和 fast-topk。功能默认值沿用各入口已有设置。
- 保留 `--experimental-quantized-ffn on|off`，默认 `off`，仅用于普通
  Qwen3.5-4B 生成模型。开启时不分配全量常驻 FP16 FFN 缓存。
- DFlash 服务块大小固定为 8。Standard DFlash 的训练配置记录 16，
  不等于服务时使用的块大小。

## 明确限制

Flash Next 的稀疏 KV 路径仅支持上游的 INT8/BF16，不将 dense Q4H
编码送入稀疏 Q8 算子。未验收的 `Q4KGroupScale + ANE` 显式边界组合
拒绝启用；默认量化 FFN 路径和普通 ANE 不受此限制。

Kev、Julia 和 Laya 的真实权重当前不在本机，使用配置和合成算子测试覆盖。
4B、D3 lite 和 D3 mini 另做真实权重检查。此次未做官方全部模型的发布性能验收。

## 验证记录

原始运行日志位于 `build/upstream-140-merge-20261021/`。
最终命令、真实退出码、源码输入摘要及精简原始结果保存在
`docs/data/upstream-main-merge-2026-10-21/verification.json` 及同目录文件。
失败的早期尝试保留在构建目录，修复后先定点重测，再运行所属完整门禁。

首次完整 Metal 命令退出 2：本地 Q4 旋转测试错误地覆盖了不支持 Q4 的
稀疏布局。修正测试范围并增加明确拒绝断言后，五种格式分别通过。
后续 fast-topk 检查还发现上游已将 `top-k=1` 归为 greedy；本地断言按该
语义修正，继续检查 argmax 和 `top-k=2..32` 的原数值标准，定点退出 0。
其余未执行的 Metal 项目逐项补齐，均退出 0。未重复执行不受这两处测试
修改影响的已通过长项；原始非零退出记录保留，不能读作首次全量命令通过。

后续测试观测工具完成后，已重新执行最终版本的完整门禁，真实退出码 0，
不以本次分项补测替代最终验收。见
[观测与验收记录](metal-test-observation-2026-10-21.md)。
