# 第 8 课：FA2 的分工与 Tensor Core

## 并行策略

本实验按 `(sequence,head,Q tile)` 分配 CTA。CTA 内两个 warp 分别拥有 16 个完整 query 行，共用 K/V tile。每个 warp 对自己负责的 query 行累加全部 key tile，因此不需要跨 warp 合并部分输出。

这体现 FA2 的 Q 行并行和 warp 内任务划分。Online softmax 只是其中一部分，还要考虑非矩阵指令、并行度和线程间通信。

## WMMA 的两个 GEMM

1. `Q_tile @ K_tile.T`：Q 片段为 row-major；以 col-major 视图加载转置后的 K。
2. `P_tile @ V_tile`：P/V 为 row-major，输出使用 FP32 accumulator。

WMMA fragment 的元素到 lane 映射是 opaque 的。本实现把 accumulator 存到每个 warp 独立负责的 shared memory 行，再做逐行 softmax 和 alpha 重缩放。这个选择便于推导，代价是额外 shared memory 读写；不能据此宣称达到库实现效率。

`D=72` 时使用填零后的片上宽度 80。全局输入仍是 72 个有效元素，输出也只能写 72 个。不要读取输入行后面的数据来填 padding。

## 实验

```bash
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py bench --case encoder_short --impl simt
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py bench --case encoder_short --impl wmma
```

对照 [kernels.cu](../csrc/kernels.cu) 的 `attention_tiled`：标记 QK、softmax、旧输出重缩放、PV、最终归一化五个阶段。指出每个 barrier 保证了哪些数据可见。

## 练习

1. 为什么 K 的物理 `[keys,D]` 存储可以作为 WMMA 的 col-major `[D,keys]` 视图？
2. Q tile 翻倍会同时增加哪些资源需求？
3. 把当前 shared-memory 中间 accumulator 换成寄存器，需要解决哪些 fragment 映射问题？

过关：能解释 FA2 的并行划分，也能指出教学版的性能代价。参考：[FA2 论文](https://arxiv.org/abs/2307.08691)。
