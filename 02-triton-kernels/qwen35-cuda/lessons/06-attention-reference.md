# 第 6 课：先实现一个明确正确的 attention

## 数学与序列边界

对每个独立序列和 head：

\[
S=QK^T\,scale,\quad P=\operatorname{softmax}(S),\quad O=PV.
\]

本课对应 MM encoder：非因果、dropout 为 0。每个 query 能看见同一序列内的所有 key；不能跨 `cu_seqlens` 边界。

数学 reference 显式用高精度矩阵乘法和 softmax，避免 reference 自动选择与被测实现相同的 fused backend。独立性比接口名字更重要。

## 基础 CUDA

`attention_naive` 用一个 CTA 处理一个 query/head，线程沿 D 合作计算 dot。CTA 循环访问该序列的 K/V，保存自己的输出累加值。这样容易验证，但不同 query 重复读取 K/V，且每个 key 都有同步成本。

为了提前建立数值稳定性，基础版已使用逐 key 的 online 更新。第 7 课将它推广到 tile；第 8 课再改变 warp 分工与矩阵乘法实现。

## 手算题

`Q=K=0`，同一序列的两个 V 为 2 和 6，则两个 query 的输出都为 4。若误加 causal mask，第一个输出会变成 2。已有 CPU 测试使用这个例子守住语义。

```bash
.venv/bin/python -m pytest 02-triton-kernels/qwen35-cuda/tests/test_reference.py -q
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py check --case encoder_short --impl baseline
```

## 练习

1. 画出长度为 2 和 3 的两个 packed 序列各自的 attention 矩阵。
2. 为什么只测试一个序列发现不了边界泄漏？
3. 数学 reference 的完整 S 是 `L*L`。输出则是 `L*D`，这给优化提供了什么机会？

过关：能区分数学语义、reference 精度和 CUDA 执行策略。
