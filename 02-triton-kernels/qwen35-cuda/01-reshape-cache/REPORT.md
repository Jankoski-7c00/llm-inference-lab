# 实验报告：普通分页缓存

## 1. 问题与场景

K/V 根据 slot mapping 原地写入分页缓存，覆盖原 dtype 与固定 scale FP8，支持 NHD 和源码中的 head-major 地址映射。

- 输入 dtype：本次性能矩阵为 FP16，正确性另覆盖 FP32/BF16。
- 精确 shape 与输入生成：见 [cases.py](../qwen35_lab/cases.py)。
- 实际硬件、软件、源码提交与代码摘要：见 [测量元数据](../results/performance.metadata.json)。

## 2. 基线与瓶颈假设

基础版每线程复制一个元素。假设在对齐的 NHD 路径中，向量化能够减少索引与访存指令；小 token 数和 head-major fallback 不一定有收益。

## 3. 方案与取舍

使用每线程 4 元素的 Pack4 搬运，并按 token/tile 分配 CTA；不满足连续性、对齐或布局条件时保留标量路径。

代码见 [CUDA 实现](../csrc/kernels.cu)，数学定义见 [独立 reference](../qwen35_lab/reference.py)。

## 4. 结果

| case | 实现 | 目标区间 (us) | 相对基础版 |
|---|---|---:|---:|
| cache_decode | baseline | 4.096 | 1.00x |
| cache_decode | optimized | 3.584 | 1.14x |
| cache_prefill | baseline | 19.046 | 1.00x |
| cache_prefill | optimized | 8.397 | 2.27x |
| cache_head_major | baseline | 3.994 | 1.00x |
| cache_head_major | optimized | 4.096 | 0.98x |

结果来自 [performance.csv](../results/performance.csv)，5 次 warmup，每组 10 次、共 3 组；表中为各组均值的中位数，样本 p10/p90 保存在 CSV。采用 Graph 外部 event 目标区间计时，重置拷贝位于 event 之外。数据在重置后较热，测量结果受该条件影响。

Systems 原始报告本地保存在 `profiles/delivery/`，文件名包含 case、实现和 dtype；见 [采集清单与摘要](../results/README.md)。Compute 报告与硬件指标留待后续实践，本报告不提供未采集的指标。

TTFT、TPOT、端到端吞吐和模型峰值显存：未测。本次相对速度仅与本实验基础版比较。

## 5. 适用边界与失败尝试

head-major 仍走相同标量路径，本次的小幅时间差不能解释为有效优化。重置后的 cache 较热，逻辑字节数除以 latency 不等同于实际显存带宽。

普通复制、动态量化和状态的测试判据见 [课程入口](../README.md#代码与输入契约)。上游辅助对照与独立数学 reference 分开判断。

## 6. 可迁移结论

先明确地址契约，再优化访存；保留正确的尾部与布局 fallback，比假定所有张量连续更可靠。

## 附：复现

```bash
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py bench --case cache_prefill --impl optimized --iters 10 --repeats 3
.venv/bin/python 02-triton-kernels/qwen35-cuda/profile.py nsys --case cache_prefill --impl optimized
```

完整矩阵、sanitizer 和数据记录方法见 [结果索引](../results/README.md)。初始退化实验保存在 [performance_initial.csv](../results/performance_initial.csv)；其中 delta 的历史 `optimized` 标签指当时的直接融合阶段，对应当前 `fused` 实现。
