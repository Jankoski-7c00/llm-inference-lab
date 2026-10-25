# 第 3 课：从标量搬运到向量化

## 先计算工作量

原 dtype 复制的逻辑数据量为 `2*T*H*D` 个读元素和相同数量的写元素，即 `4*T*H*D*sizeof(dtype)` 字节，再加 slot metadata。这个值是逻辑流量估算，不是硬件实际 DRAM 事务数；L2 命中、重复 slot 读取和非合并事务都会使两者不同。

基础版一个线程搬一个元素。优化版在适合的 NHD 布局中，用一个线程搬 4 个连续元素，CTA 按 token/tile 分工。它减少索引和指令开销，但不保证在小 token 数上更快。

`Pack4<T>` 的 load/store 必须满足对齐和连续性。源码同时检查起始地址、D、外层 stride 和目标布局，其余情况保留标量实现。特别注意：从第一个元素之后开始切片，可能保持所有 stride 不变却使起点不对齐。不要用强制转换掩盖不对齐。

## 实验

```bash
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py bench --case cache_prefill --impl baseline --output 02-triton-kernels/qwen35-cuda/outputs/cache_base.json
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py bench --case cache_prefill --impl optimized --output 02-triton-kernels/qwen35-cuda/outputs/cache_opt.json
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py bench --case cache_decode --impl optimized --mode eager
```

前两个在相同 Graph 模式下比较。最后一个用于观察主机提交影响，不能直接拿它与 Graph 数字计算加速比。

## 练习

1. 输入 FP16、输出 FP8 时重新计算逻辑字节数。
2. `D=33` 为什么不适合直接搬 4 个元素？怎样保留正确的尾部路径？
3. 同一实现下比较 `cache_decode/cache_prefill`，提出一个能被 Systems 时间线验证的假设。
4. 记录修改前后 latency、有效逻辑带宽与适用 shape，写出一次无收益的尝试。

过关：能把收益归因到明确的修改，并解释逻辑带宽与硬件带宽的区别。报告入口：[普通缓存](../01-reshape-cache/REPORT.md)。
