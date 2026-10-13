# Swift 27B：全局 prefill 块大小与 drafter 开关对照

A 继续作为基准：全局 prefill 块大小 2048、DFlash2 开启。测试 512／1024／2048 与 drafter 开／关，共 6 组，各做 3 轮性能测量。固定同一份混合 GGUF、Q4H、32,768 输入、16,384 实际输出、131,072 上下文容量、并发 1、ANE share=0.5／阈值 512、尾段 splits=16。

Lookup 和 compact 始终配置为开启。NoDraft 会停用依赖 drafter 的 lookup 验证和 compact 草稿头，并记录回退原因；该组实际不加载 drafter 权重，不分配草稿状态，也不提出草稿 token。Drafter 开启时始终使用匹配的 DFlash2 block=8。

三个块大小分别从冻结源码快照构建。Host、shader、工作区、ANE 行数范围与预热保持一致。Drafter 的上下文窗口仍为 2048，整窗恢复按各自块大小拆分，避免小块工作区溢出。原 checkout 的源码、原 A 构建及权重不改。

数值门禁覆盖真实预算大小的 attention CPU 标量参考、页写入和 canary；同路径分数相对误差不超过 1e-3；GDN 分块状态 cosine 大于 0.999；两次冷 decode 决策一致。开启 drafter 的组还校验完整 2048 窗口恢复的状态与 ring 样本。ANE 启动 FFN 贡献 RMS 误差保持不超过 10%。

每组先做两次完整输出正确性验证，再做正序、反序、旋转顺序的三轮性能测量。性能输出必须与该组正确性输出逐 token 一致。加载、ANE 准备及对应块大小的统一预热不计入请求时间。

本轮未完成，控制器真实退出码为 1。原始错误、已完成结果和日志已保存，缺失退出证据及取消的尝试不记为通过。

原始日志：[controller.log](../build/swift27b-q4h-budget-draft-v1/controller.log)。归档：[数据目录](data/swift27b-q4h-budget-draft-2026-10-12/)。

本轮仅验收 C1，之前的 B4 决策一致性失败仍未解决。
