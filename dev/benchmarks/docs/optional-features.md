# 可选 proposal 与采样功能

本轮实现链接方案的第一批三项。所有开关默认关闭，在服务启动时读取；修改后重启服务。
第二批的宽 lookup 验证、draft ahead、streamed submit 尚未实施。

| 环境变量 | 默认 | 支持条件和回退 |
| --- | --- | --- |
| `SPLASH_PROMPT_LOOKUP=0/1` | 0 | 已安装 drafter 的单个非语法约束文本请求；查找输入和已提交输出中的 2～4 token 后缀，复制完整的一个草稿块。未命中、并发、图像或语法约束走原 DFlash 路径。 |
| `SPLASH_COMPACT_DRAFT_HEAD=0/1` | 0 | Standard DFlash 和 DFlash2；单个文本请求、一个未旋转的量化输出矩阵、匹配模型的排名文件。其他条件回退完整 head。 |
| `SPLASH_FAST_TOPK=0/1` | 0 | 采样温度大于 0，所有采样 lane 的有效 top-k 为 1～32，min-p 为 0。支持已有 penalties、top-p、token mask 和 EOS 排除；其他设置回退通用采样。 |
| `SPLASH_COMPACT_DRAFT_RANKING=路径` | 模型目录下的 `head-ranked.splash` | 只在 compact head 开启时使用；文件缺失、格式错误或模型指纹不匹配均记录回退原因。 |

环境变量只接受 `0` 或 `1`，非法值拒绝启动。全关时不创建历史副本、compact 权重副本或 fast top-k 工作区。
开关相互独立。lookup 命中时跳过本轮 draft 前向，target 验证和已接受前缀的 GDN／KV／draft 上下文提交仍走现有路径。

## 数值和状态规则

- Lookup 的采样提议为 point-mass Q：复制 token 的概率为 1，其余候选为 0。复用 target 的接受概率和拒绝后 residual correction，不把复制 token 直接当作已接受输出。
- Compact head 保留原量化编码和 scale 元数据，按原格式收集行。target head 保持完整词表。Standard DFlash 在提议后映射 token ID；DFlash2 在 codebook 查找之前映射 ID。
- Compact head 通常有 98,048 个静态行和最多 256 个动态行；小词表相应缩小。动态补充输入、已提交输出和 pending anchor 中的稀有 token。稀有 token 超过历史 token 的 1%，或不同稀有 token 超过 256 个，则使用完整 head。填充行的 logits 被置为负无穷，不能进入 proposal Q。
- Fast top-k 保留精确的 logits 降序、token ID 升序排序和 top-p 边界，继续使用原有抽样与残差修正 kernel。
- Q4_K group scale 后移仍是独立的 FFN 实验。本轮正确性使用已通过的 `quantized + baseline` FFN 路径，不启用该未通过长输出校验的候选。

## 排名文件

排名必须是与当前 target 权重指纹绑定的完整 token ID 排列。不能因为两个模型的词表大小相同就复用排名。
文件包含 8 字节 `SPLCHD01`、64 字节 target manifest SHA-256、little-endian uint32 词表大小以及完整 uint32 排列。

通过真实语料准备 token 排序 JSON 后，可使用：

```sh
.venv/bin/python -m dev.tools.compact_draft_ranking \
  --target-manifest TARGET_SHA256 \
  --token-order token-order.json \
  --output head-ranked.splash
```

工具不会猜测 token 流行度，也不会覆盖已有文件。
正确性驱动自动生成的排名只用于验证收集、映射和 target 校验，标记为 synthetic correctness fixture；它不构成可推荐的生产排名。

## 正确性入口

```sh
make -j4 BUILD=build/optional-features/block-8 check-source check-native-build check-native-cpu check-python test-sanitizers

MTL_SHADER_VALIDATION=1 build/optional-features/block-8/engine-tests/compact-projection \
  build/optional-features/block-8/engine-tests/production-and-test.metallib
MTL_SHADER_VALIDATION=1 build/optional-features/block-8/engine-tests/target-sampling \
  build/optional-features/block-8/splash.metallib --fast-topk --correctness-only

.venv/bin/python -m dev.benchmarks.qwen_optional_features \
  --model-root MODEL_ROOT --build build/optional-features/block-8 \
  --reference-payload PRIOR_32K_PAYLOAD --output NEW_RESULTS_DIRECTORY \
  --allow-concurrent-inference
```

真实模型的 ANE 校验使用 `MTL_SHADER_VALIDATION=0`；否则验证层包装共享事件会导致 ANE 回退。驱动会校验 ANE 实际启用、share 和阈值。算子与采样测试保留 shader validation，数值和 token 标准不变。

追加 `--long` 后检查原有 32K 输入，固定实际输出 512／2048。
驱动只支持 correctness，保存全关、单项、组合的原始 token 输出及首次分歧。短输入额外检查组合采样的固定 seed 重复性。
推理服务并行授权只用于正确性；其他 benchmark 仍由已有隔离门禁拦截。结果不可作为性能数据。

`Runtime::optionalFeatureReport()`、native 正确性 JSON 的 `optional_features` 和请求结束日志包含配置、命中／实际执行次数、回退原因及新增 Metal 字节。
新增 compact 权重和采样工作区计入实际共享 decode 内存报告，不将其隐去。
