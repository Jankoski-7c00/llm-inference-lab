# 第 2 课：把 slot 变成物理地址

## 推导

设页大小为 `B`，逻辑 slot 为 `s`：

\[
page=\lfloor s/B\rfloor,\quad offset=s\bmod B.
\]

连续 NHD cache 的形状为 `[pages,B,H,D]`：

\[
addr=((page\,B+offset)H+h)D+d.
\]

有 padding 或视图时用实际 stride：`page*s0 + offset*s1 + h*s2 + d*s3`。stride 的单位是元素，乘指针类型后才转换为字节地址。

例：`B=4,H=2,D=4,slot=5,h=1,d=2`，页号为 1、页内位置为 1，连续偏移为 46。

Head-major K 的 `[pages,H,D/x,B,x]` 地址为：

`page*s0 + h*s1 + (d/x)*s2 + offset*s3 + (d%x)*s4`。

对应 V 为 `[pages,H,D,B]`，地址为 `page*s0 + h*s1 + d*s2 + offset*s3`。两种布局的同一个逻辑元素拥有不同地址。

## 从线程索引开始

基础版把 `z` 展开为 `(token,head,dim)`。每个线程独立读 slot、判断 padding，然后搬运一个 K 和一个 V。输入需要考虑 token/head stride，不能默认 `token*H*D`。

在 [kernels.cu](../csrc/kernels.cu) 中阅读 `CacheDesc`、`cache_offset`、`cache_kernel<...,false>`。先手算地址，再对照代码。

## 练习与验证

```bash
PYTHONPATH=02-triton-kernels/qwen35-cuda .venv/bin/python -m qwen35_lab.exercise cache
.venv/bin/python -m pytest 02-triton-kernels/qwen35-cuda/tests/test_cache.py -q -k 'cache_random or tail_dimension'
```

1. 将 `slot=5` 改为 `slot=7`，手算页号、页内位置和地址。
2. 为什么使用随机不重复 slot，而不只测试连续 slot？
3. 把 cache 预填为 sentinel。除目标位置外，所有元素应该发生什么变化？
4. 为什么两个 token 同时写同一 slot 是竞态，不能通过“最后一个 token 赢”来定义结果？

过关：可以独立补出索引，并通过跨页、负 slot、tail dimension、未写区域测试。答案见 [第 2 课](ANSWERS.md#第-2-课)。
