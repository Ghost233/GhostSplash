# 可选 proposal 与采样功能

本轮实现链接方案的第一批三项。所有开关默认关闭，在服务启动时读取；修改后重启服务。
第二批的宽 lookup 验证、draft ahead、streamed submit 尚未实施。

| 启动参数 | 默认 | 支持条件和回退 |
| --- | --- | --- |
| `--prompt-lookup off\|on` | off | 已安装 drafter 的单个非语法约束文本请求；查找输入和已提交输出中的 2～4 token 后缀，复制完整的一个草稿块。未命中、并发、图像或语法约束走原 DFlash 路径。 |
| `--compact-draft-head off\|on` | off | Standard DFlash 和 DFlash2；必须显式指定公共排名、支持的未旋转量化 FP32 head，并通过一次性数值检查。文件、布局或数值错误拒绝启用；多 lane／图像使用完整 head。 |
| `--fast-topk off\|on` | off | 采样温度大于 0，所有采样 lane 的有效 top-k 为 1～32，min-p 为 0。支持已有 penalties、top-p、token mask 和 EOS 排除；其他设置回退通用采样。 |
| `--compact-draft-ranking 文件` | 未设置 | 只在显式开启时读取，路径本身不会开启功能。不扫描模型目录或测试目录。文件缺失、格式损坏、词表或排列错误拒绝启用。 |

功能开关只接受 `off` 或 `on`，非法值拒绝启动。旧环境变量不再读取。全关时不创建历史副本、compact 权重副本或 fast top-k 工作区。
开关相互独立。lookup 命中时跳过本轮 draft 前向，target 验证和已接受前缀的 GDN／KV／draft 上下文提交仍走现有路径。

## 数值和状态规则

- Lookup 的采样提议为 point-mass Q：复制 token 的概率为 1，其余候选为 0。复用 target 的接受概率和拒绝后 residual correction，不把复制 token 直接当作已接受输出。
- Compact head 保留原量化编码和 scale 元数据，按原格式收集行。target head 保持完整词表。Standard DFlash 在提议后映射 token ID；DFlash2 在 codebook 查找之前映射 ID。
- Compact head 通常有 98,048 个静态行和最多 256 个动态行；小词表相应缩小。动态补充输入、已提交输出和 pending anchor 中的稀有 token。稀有 token 超过历史 token 的 1%，或不同稀有 token 超过 256 个，则使用完整 head。填充行的 logits 被置为负无穷，不能进入 proposal Q。
- Fast top-k 保留精确的 logits 降序、token ID 升序排序和 top-p 边界，继续使用原有抽样与残差修正 kernel。
- Q4_K group scale 后移仍是独立的 FFN 实验。本轮正确性使用已通过的 `quantized + baseline` FFN 路径，不启用该未通过长输出校验的候选。

## 公共排名文件

当前格式为 `SPLCHD02`，包含 8 字节版本标记、little-endian uint32 词表大小，以及完整 uint32 token ID 排列。
文件不包含或校验主模型权重 SHA。实际 compact 权重始终从当前模型的输出 head 收集。
旁边的 JSON 可以记录 tokenizer profile、tokenizer SHA 和语料来源，供人工选择；这些信息不参与运行时绑定。

原生服务和启动器默认关闭 compact。显式开启时必须同时指定文件：

```sh
sh qwen38miniplusv3splash.sh \
  --compact-draft-head on \
  --compact-draft-ranking compact-rankings/qwen35-248320-dev-seed.splash
```

仅指定文件仍保持关闭；关闭时不读取排名、不创建 compact 副本、映射或数值校验工作区。
显式开启时，文件／词表／完整排列／ID 范围／重复与遗漏／head 布局检查必须通过。
构建 compact 后，用相同 BF16 输入比较完整 head 的对应行，复用原 staged decode 精确相等标准。
首次检查同时覆盖初始行收集和动态行更新路径；失败报错，拒绝启用。校验工作区检查后释放，不在每轮生成中计算完整 head。

稀有 token 超过历史的 1%，或不同稀有 token 超过 256 个，仍正常使用完整 head。
覆盖率回退和请求条件回退分别计数；文件、布局和数值错误会中止启用。
主模型使用完整词表验证，接受概率、残差修正和状态提交规则保持原逻辑。
草稿与完整 drafter 不同、接受率下降或回答不同，需要另行评测，不作为这项数值检查的失败条件。

一次性导出旧 `SPLCHD01` 文件，保留原文件及报告：

```sh
.venv/bin/python -m dev.tools.compact_draft_ranking \
  --import-legacy OLD_HEAD_RANKED.splash \
  --tokenizer-profile qwen35-vocab248320-nfc-bytelevel-bpe \
  --output public-ranking.splash > public-ranking.json
```

新加载器只读取公共格式。旧格式必须离线转换，不增加运行时兼容模式。
也可以从完整 token 排序 JSON 或本地 UTF-8 语料生成公共文件：

```sh
.venv/bin/python -m dev.tools.compact_draft_ranking \
  --token-order token-order.json --output public-ranking.splash

.venv/bin/python -m dev.tools.compact_draft_ranking \
  --tokenizer MODEL_DIRECTORY/tokenizer \
  --corpus corpus-a.txt corpus-b.txt \
  --validation-corpus heldout.txt \
  --tokenizer-profile PROFILE_NAME \
  --output public-ranking.splash > public-ranking.json
```

语料模式只使用 CPU 分词，保留特殊控制 token，按频率降序、同频按 ID 升序排列。
覆盖率报告描述语料整体，不保证每个请求都满足覆盖率条件。工具拒绝覆盖已有文件。
本仓库公共种子文件是 `compact-rankings/qwen35-248320-dev-seed.splash`，没有默认路径或自动选择规则。
来源是仍然存在的 Saluki 旧排名；先前取消的是启动器默认路径，不是删除排名文件。
完整变更和验收记录见 [公共排名加载说明](../../../docs/compact-public-ranking-2026-10-14.md)。

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
