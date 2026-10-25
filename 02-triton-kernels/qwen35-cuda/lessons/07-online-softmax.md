# 第 7 课：online softmax 的三个状态

## 递推公式

对于一个 query 行，保存最大值 `m`、分母 `l` 和未归一化输出向量 `u`。当前 tile 的 scores 为 `s_j`：

\[
m'=\max(m,\max_j s_j),\quad \alpha=e^{m-m'},\quad p_j=e^{s_j-m'}.
\]

\[
l'=\alpha l+\sum_j p_j,\quad u'=\alpha u+\sum_j p_jv_j.
\]

全部 tile 完成后才计算 `u/l`。初始化 `m=-inf,l=0,u=0`；遇到首个有效 tile，旧状态的系数自然为 0。

这使得 S/P 只在片上保存一个 tile，不需要把完整 `L*L` 中间矩阵写入显存。Q/K/V 本身仍在显存，不能说 attention 不访问显存。

## 关键正确性条件

- 旧分母和旧输出必须使用同一个 alpha 重缩放。
- 尾部 key 的 score 为 `-inf`，概率为 0。
- padding query 行单独处理，避免 `-inf-(-inf)` 产生 NaN。
- 允许数学等价的累加顺序有浮点误差，但不能省掉重缩放步骤。

## 动手

```bash
PYTHONPATH=02-triton-kernels/qwen35-cuda .venv/bin/python -m qwen35_lab.exercise softmax
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py check --case encoder_short --impl simt
.venv/bin/python -m pytest 02-triton-kernels/qwen35-cuda/tests/test_attention.py -q
```

手算两个单元素 tile：scores 为 0、`ln(2)`，V 为 2、8。第一步 `m=0,l=1,u=2`；第二步求 alpha、l、u 和最终输出。再说明把两个 tile 反过来处理为什么数学结果相同。

过关：能闭卷写出三个状态和更新顺序，并解释大 logits 为什么仍可稳定计算。
