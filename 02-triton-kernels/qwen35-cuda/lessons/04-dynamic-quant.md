# 第 4 课：动态量化是一次归约加写入

## 定义

每个 `(token,head)` 独立处理 K 和 V：

\[
s=\max(\operatorname{absmax}(x)/Q_{max},10^{-6}),\quad y=x(1/s).
\]

INT8 对 `y` 使用 ties-away-from-zero 舍入，再截断到 `[-128,127]`。这里 `torch.round` 的 ties-to-even 不满足定义。FP8 对有界值执行浮点格式转换。反量化为 `x_hat=stored*s`，不能只看 cache 而忘记 scale。

K/V 拥有各自的 absmax、dimension 和 scale，地址使用各自 stride。零向量的 scale 仍为正，从而不会发生除零。

## 两个 CUDA 阶段

基础版用一个线程扫描整条 head 向量，作为清晰的正确性起点。优化版用一个 warp 扫描，再进行 `warp_max`，最后各 lane 负责自己的量化元素。每个 head 的 FP32 scale 只写一次。

## 动手

```bash
PYTHONPATH=02-triton-kernels/qwen35-cuda .venv/bin/python -m qwen35_lab.exercise dynamic
.venv/bin/python -m pytest 02-triton-kernels/qwen35-cuda/tests/test_cache.py -q -k dynamic
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py check --case dynamic_fp8 --impl optimized
```

## 手算

给定 `x=[127,0.5,-0.5,1.5,-1.5,0,-127]` 和 INT8 量化，求 scale 与存储值。再用全零向量说明 scale 下限的作用。

说明为什么对 scale 使用浮点容差、对 INT8 存储值要求整数一致、对反量化误差单独统计。下一课分析 FP8 舍入路径与相邻编码的例外。

过关：能独立写出 scale、舍入、饱和、写入，并通过不同 K/V dimension 和非连续 head stride 测试。
