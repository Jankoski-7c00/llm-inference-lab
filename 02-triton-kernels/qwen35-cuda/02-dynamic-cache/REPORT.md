# 实验报告：动态量化分页缓存

## 1. 问题与场景

每个 token/head 独立计算 K/V absmax、scale、量化值并写入分页缓存。K=128，V=64 的预设覆盖不同维度。

- 输入 dtype：本次性能矩阵为 FP16，正确性另覆盖 FP32/BF16。
- 精确 shape 与输入生成：见 [cases.py](../qwen35_lab/cases.py)。
- 实际硬件、软件、源码提交与代码摘要：见 [测量元数据](../results/performance.metadata.json)。

## 2. 基线与瓶颈假设

基础版单线程扫描一个 head。假设把独立元素分给一个 warp，并用 shuffle 归约，可减少串行扫描成本。

## 3. 方案与取舍

一个 warp 负责一个 token/head，独立归约 K/V；lane 0 写 scale，各 lane 完成自己的量化和缓存写入。

代码见 [CUDA 实现](../csrc/kernels.cu)，数学定义见 [独立 reference](../qwen35_lab/reference.py)。

## 4. 结果

| case | 实现 | 目标区间 (us) | 相对基础版 |
|---|---|---:|---:|
| dynamic_int8 | baseline | 18.944 | 1.00x |
| dynamic_int8 | optimized | 5.018 | 3.78x |
| dynamic_fp8 | baseline | 16.589 | 1.00x |
| dynamic_fp8 | optimized | 4.915 | 3.38x |

结果来自 [performance.csv](../results/performance.csv)，5 次 warmup，每组 10 次、共 3 组；表中为各组均值的中位数，样本 p10/p90 保存在 CSV。采用 Graph 外部 event 目标区间计时，重置拷贝位于 event 之外。数据在重置后较热，测量结果受该条件影响。

Systems 原始报告本地保存在 `profiles/delivery/`，文件名包含 case、实现和 dtype；见 [采集清单与摘要](../results/README.md)。Compute 报告与硬件指标留待后续实践，本报告不提供未采集的指标。

TTFT、TPOT、端到端吞吐和模型峰值显存：未测。本次相对速度仅与本实验基础版比较。

## 5. 适用边界与失败尝试

INT8 与独立 reference 精确一致；上游除法在半整数附近可导致相邻整数。FP8 原生转换与上游中间 FP16 路径也可差一个编码，仅上游辅助对照采用该边界。

普通复制、动态量化和状态的测试判据见 [课程入口](../README.md#代码与输入契约)。上游辅助对照与独立数学 reference 分开判断。

## 6. 可迁移结论

量化测试应分别检查 scale、离散存储与反量化误差。先确认算术路径，再判断差异是实现错误还是定义明确的舍入差异。

## 附：复现

```bash
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py bench --case dynamic_int8 --impl optimized --iters 10 --repeats 3
.venv/bin/python 02-triton-kernels/qwen35-cuda/profile.py nsys --case dynamic_int8 --impl optimized
```

完整矩阵、sanitizer 和数据记录方法见 [结果索引](../results/README.md)。初始退化实验保存在 [performance_initial.csv](../results/performance_initial.csv)；其中 delta 的历史 `optimized` 标签指当时的直接融合阶段，对应当前 `fused` 实现。
