# 实验报告：Fused sigmoid gating delta rule

## 1. 问题与场景

融合门控、Q/K normalization 与递归更新，比较 output、状态池及非 inplace 快照，覆盖 accepted-token 初始状态索引。

- 输入 dtype：本次性能矩阵为 FP16，正确性另覆盖 FP32/BF16。
- 精确 shape 与输入生成：见 [cases.py](../qwen35_lab/cases.py)。
- 实际硬件、软件、源码提交与代码摘要：见 [测量元数据](../results/performance.metadata.json)。

## 2. 基线与瓶颈假设

基础版为门控、归一化、递推三个 kernel。初始假设一次融合能减少 launch 和中间数据，但直接融合实测退化。为检验重复计算是否重要，继续拆分共享计算、行分工与 K 循环三个因素；计时本身不能确认具体硬件瓶颈。

## 3. 方案与取舍

保留直接融合 `fused`。`shared` 只共享 normalization/门控，保持每 warp 一行、每 lane 8 个 K 槽；`tiled` 改为每 warp 4 行、每 CTA 16 行；`optimized` 再针对 K 不超过 128 的情况，把 K 槽缩为 4。各阶段都以 FP32 递推，并对照相同数学 reference。

代码见 [CUDA 实现](../csrc/kernels.cu)，数学定义见 [独立 reference](../qwen35_lab/reference.py)。

## 4. 结果

| case | 实现 | 目标区间 (us) | 相对基础版 |
|---|---|---:|---:|
| delta_decode | baseline | 29.082 | 1.00x |
| delta_decode | fused | 36.250 | 0.80x |
| delta_decode | shared | 38.202 | 0.76x |
| delta_decode | tiled | 29.184 | 1.00x |
| delta_decode | optimized | 24.576 | 1.18x |
| delta_spec | baseline | 20.787 | 1.00x |
| delta_spec | fused | 24.678 | 0.84x |
| delta_spec | shared | 25.414 | 0.82x |
| delta_spec | tiled | 20.992 | 0.99x |
| delta_spec | optimized | 17.715 | 1.17x |

结果来自 [performance.csv](../results/performance.csv)，5 次 warmup，每组 10 次、共 3 组；表中为各组均值的中位数，样本 p10/p90 保存在 CSV。采用 Graph 外部 event 目标区间计时，重置拷贝位于 event 之外。数据在重置后较热，测量结果受该条件影响。

Systems 原始报告本地保存在 `profiles/delivery/`，文件名包含 case、实现和 dtype；见 [采集清单与摘要](../results/README.md)。Compute 报告与硬件指标留待后续实践，本报告不提供未采集的指标。

TTFT、TPOT、端到端吞吐和模型峰值显存：未测。本次相对速度仅与本实验基础版比较。

## 5. 适用边界与失败尝试

直接融合和单独共享计算在 decode/spec 两种 case 均退化；单独共享新增了 barrier，也改变了计算分工。行分块和 K 专用化随后改善了结果。当前计时支持组合策略在这些 case 更快，无法把收益全部归因于共享；没有 Compute 数据时，也不能把寄存器或某种 stall 宣称为已证实瓶颈。

普通复制、动态量化和状态的测试判据见 [课程入口](../README.md#代码与输入契约)。上游辅助对照与独立数学 reference 分开判断。

## 6. 可迁移结论

少一次 launch 不保证更快。用一次真实退化提出新假设，再用控制变量修改和相同输入重测，保留失败版本。

## 附：复现

```bash
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py bench --case delta_decode --impl optimized --iters 10 --repeats 3
.venv/bin/python 02-triton-kernels/qwen35-cuda/profile.py nsys --case delta_decode --impl optimized
```

完整矩阵、sanitizer 和数据记录方法见 [结果索引](../results/README.md)。初始退化实验保存在 [performance_initial.csv](../results/performance_initial.csv)；其中 delta 的历史 `optimized` 标签指当时的直接融合阶段，对应当前 `fused` 实现。
