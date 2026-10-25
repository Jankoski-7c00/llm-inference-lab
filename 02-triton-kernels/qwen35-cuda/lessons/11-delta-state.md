# 第 11 课：状态池、单步 decode 与多步 spec

## 状态布局

状态池为 `[pool,HV,V,K]`。连续 K 维由 warp 的 lane 合作处理，每个 warp 负责一个 V 行。输入使用 packed token，`cu_seqlens` 划分独立序列。

普通单步 decode 通过 `indices[n,0]` 选择初始状态。Spec 路径通过 `indices[n,accepted[n]-1]` 选择起始状态；递推第 i 步可写到 `indices[n,i]`。不要把“接受了几个 token”和“本轮处理几个 token”混为一谈。

同一序列的时间步有依赖，CUDA 内按时间顺序执行；独立序列、value heads 和 V 行并行。使用 atomic 并不能自动消除递推的时序要求。

## Inplace 与 trace

- `inplace=True`：只修改有效目标状态项，返回状态池。
- `inplace=False`：初始状态池保持不变，返回每个 token 的 `[HV,V,K]` 快照。
- 初始索引 0 为 padding；本实现输出零且不写状态。原版在这类输出上不提供数值保证。
- 不同序列的状态写入不能重叠；构造输入时先运行 debug metadata 校验。

## 测试设计

用 `K=65,V=33` 暴露状态转置与 tail bug，用不同初始状态暴露错误的 accepted index。不能只用方形状态测试。

```bash
.venv/bin/python -m pytest 02-triton-kernels/qwen35-cuda/tests/test_delta.py -q
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py check --case delta_spec --impl optimized
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py check --case delta_spec --impl upstream
```

练习：给 `indices=[6,7,8],accepted=2` 标出初始读取与三步写入的地址。把 index 7 放入另一条序列的写入列表，说明为什么出现竞态。

过关：能同时验证输出、有效状态、未写状态以及非 inplace 下的初始状态池。
