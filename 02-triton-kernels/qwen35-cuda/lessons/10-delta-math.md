# 第 10 课：把 fused delta rule 拆回数学

## 记号

每个 value head 的状态 `S` 形状为 `[V,K]`，不是 `[K,V]`。Q/K head 数可以少于 value head 数，使用 `h_qk=floor(h_v/(HV/H))` 分组映射。

对一个 token：

\[
d_t=\exp[-\exp(A_{log})\,softplus(a_t+dt_{bias})],\quad \beta_t=\sigma(b_t).
\]

若启用 normalization：`q=q/sqrt(sum(q*q)+eps)`，K 同理；再把 Q 乘以 scale。递推为：

\[
\bar S_t=d_t S_{t-1},\quad r_t=\beta_t(v_t-\bar S_t k_t),
\]

\[
S_t=\bar S_t+r_t k_t^T,\quad o_t=S_t q_t.
\]

注意 prediction 使用衰减后的状态，输出使用更新后的状态。把这两个顺序写错，仍可能在特殊输入上“看起来合理”。

`softplus` 的 beta 参数与门控 `beta_t` 不是同一含义；实现中前者是标量参数，后者来自 sigmoid。

## 手算

取 `K=V=1`，关闭 normalization，`S_old=2,q=k=1,v=3`，所有 gate 参数为 0：`d=1/2,beta_t=1/2`。先得到 `S_bar=1`，prediction 为 1，残差为 1，最终 `S=2,o=2`。

现有 CPU reference 测试使用这个例子，能区分“先衰减再预测”和错误的更新顺序。

## 动手

```bash
.venv/bin/python -m pytest 02-triton-kernels/qwen35-cuda/tests/test_reference.py -q -k delta
PYTHONPATH=02-triton-kernels/qwen35-cuda .venv/bin/python -m qwen35_lab.exercise delta
```

在 [reference.py](../qwen35_lab/reference.py) 标出每一项公式对应的语句，再在 CUDA 的 `delta_kernel` 找到它们。

过关：能闭卷写出递推，并解释门控、衰减、纠错残差和输出各自做了什么。
