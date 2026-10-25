# 第 12 课：融合的收益与代价

## 对比对象

`baseline` 分为三次 launch：门控计算、Q/K normalization、递推更新。`fused` 直接合在一次 launch。之后分三步改变实现：

| 阶段 | 相对上一步只改变的因素 |
|---|---|
| `shared` | CTA 内共享 normalization/门控；保持每 warp 一行和 8 个 K 槽 |
| `tiled` | 每 warp 改为 4 行，CTA 从 4 行改为 16 行 |
| `optimized` | K 不超过 128 时，将每 lane 的 K 槽从 8 缩为 4 |

直接融合减少中间张量的写回和再读取，也减少 launch。但不同 V 行重复 normalization 与门控工作。第一轮实际出现融合退化；拆开后，单独共享计算也没有收益，后续行分块和 K 循环专用化才改善了测量结果。该结果说明不能只凭重复计算就认定主瓶颈。

## 状态驻留

`fused/shared` 每个 warp 负责一个 V 行、CTA 负责 4 行。`tiled/optimized` 每个 warp 负责 4 行、CTA 负责 16 行。共享阶段由首个 warp 计算 normalization，通过 shared memory 提供给整个 CTA，门控也只计算一次。时间循环保持 FP32 局部状态，每步按要求写出，不把刚写出的低精度状态重新读回来作为下一步输入。

局部数组是否驻留寄存器取决于编译结果。只能从代码估算需求，最终要结合 ptxas 或后续 Compute 数据确认是否 spill。

## 实验

```bash
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py bench --case delta_decode --impl baseline
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py bench --case delta_decode --impl fused
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py bench --case delta_decode --impl shared
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py bench --case delta_decode --impl tiled
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py bench --case delta_decode --impl optimized
.venv/bin/python 02-triton-kernels/qwen35-cuda/profile.py nsys --case delta_decode --impl baseline
.venv/bin/python 02-triton-kernels/qwen35-cuda/profile.py nsys --case delta_decode --impl optimized
```

计时前必须从相同的初始状态开始。微基准中的重置不放入被测区间；它也会让状态比较热，需要在报告中说明。反复更新同一状态而不记录其变化，无法作为固定输入的公平比较。

## 练习

1. 计算基础版 Qn/Kn 两个 FP32 中间张量的读写字节数。
2. 在 Systems 中验证基础版与优化版 launch 次数，而不只看一个总时间。
3. 共享版新增了 barrier 和更多局部状态。若它更慢，列出重复计算、寄存器压力、并行度三个可验证假设，并区分哪些需要 Compute。

过关：能说明融合消除了什么、增加了什么，并用真实结果而不是 kernel 名字判断收益。
