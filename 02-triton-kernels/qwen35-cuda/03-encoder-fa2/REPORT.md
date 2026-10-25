# 实验报告：MM encoder FA2

## 1. 问题与场景

非因果、零 dropout 的变长 encoder attention；预设包含 D=72 和两个不同长度的独立序列。

- 输入 dtype：本次性能矩阵为 FP16，正确性另覆盖 FP32/BF16。
- 精确 shape 与输入生成：见 [cases.py](../qwen35_lab/cases.py)。
- 实际硬件、软件、源码提交与代码摘要：见 [测量元数据](../results/performance.metadata.json)。

## 2. 基线与瓶颈假设

基础版每个 CTA 处理一个 query，逐 key 同步。假设 Q tile 复用、Tensor Core 和搬运重叠可逐步降低目标区间时间。

## 3. 方案与取舍

保留 baseline、普通 SIMT 分块、WMMA、cp.async 四阶段；warp 拥有独立 Q 行，共享 K/V，使用 online softmax 保存未归一化输出。

代码见 [CUDA 实现](../csrc/kernels.cu)，数学定义见 [独立 reference](../qwen35_lab/reference.py)。

## 4. 结果

| case | 实现 | 目标区间 (us) | 相对基础版 |
|---|---|---:|---:|
| encoder_short | baseline | 42.230 | 1.00x |
| encoder_short | simt | 54.720 | 0.77x |
| encoder_short | wmma | 25.088 | 1.68x |
| encoder_short | async | 20.400 | 2.07x |
| encoder_long | baseline | 2663.798 | 1.00x |
| encoder_long | simt | 609.373 | 4.37x |
| encoder_long | wmma | 231.888 | 11.49x |
| encoder_long | async | 163.219 | 16.32x |

结果来自 [performance.csv](../results/performance.csv)，5 次 warmup，每组 10 次、共 3 组；表中为各组均值的中位数，样本 p10/p90 保存在 CSV。采用 Graph 外部 event 目标区间计时，重置拷贝位于 event 之外。数据在重置后较热，测量结果受该条件影响。

Systems 原始报告本地保存在 `profiles/delivery/`，文件名包含 case、实现和 dtype；见 [采集清单与摘要](../results/README.md)。Compute 报告与硬件指标留待后续实践，本报告不提供未采集的指标。

TTFT、TPOT、端到端吞吐和模型峰值显存：未测。本次相对速度仅与本实验基础版比较。

## 5. 适用边界与失败尝试

短序列的 SIMT 分块版实际比基础版慢。WMMA 教学版仍通过 shared memory 交换 accumulator，不能声称达到库实现效率。较长序列下的收益也只相对本实验基础版。

普通复制、动态量化和状态的测试判据见 [课程入口](../README.md#代码与输入契约)。上游辅助对照与独立数学 reference 分开判断。

## 6. 可迁移结论

分块、Tensor Core 和异步搬运是不同机制，需要逐阶段测量。非整齐维度和序列尾块必须在每一步继续验证。

## 附：复现

```bash
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py bench --case encoder_short --impl async --iters 10 --repeats 3
.venv/bin/python 02-triton-kernels/qwen35-cuda/profile.py nsys --case encoder_short --impl async
```

完整矩阵、sanitizer 和数据记录方法见 [结果索引](../results/README.md)。初始退化实验保存在 [performance_initial.csv](../results/performance_initial.csv)；其中 delta 的历史 `optimized` 标签指当时的直接融合阶段，对应当前 `fused` 实现。
