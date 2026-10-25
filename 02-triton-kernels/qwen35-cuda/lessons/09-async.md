# 第 9 课：cp.async 与双缓冲

## 时间线

同步版是 `load tile i → compute tile i → load tile i+1`。双缓冲希望在计算 tile i 时提交 tile i+1 的搬运：

```text
buffer 0: load 0 | compute 0          | load 2 | compute 2
buffer 1:        | load 1 | compute 1 |        | load 3
```

本实验使用 16-byte `cp.async`。提交 group 后，使用该 buffer 前执行 wait，再由 CTA barrier 保证所有消费者能读到数据。覆盖旧 buffer 前，还必须确保上一轮消费者已经结束。

越界的 key 或 padding dimension 采用零字节 source-size 的零填充，并提供安全的全局指针。对齐检查既包含 stride，也包含 K/V 的实际起始地址；切片 offset 可能改变后者。不能满足时，API 要求选择同步路径。

## 控制变量

同步 WMMA 与异步版使用相同 tile 和双缓冲 shared memory 预算，先隔离“搬运重叠”的影响。异步操作、commit/wait 和屏障也有开销，短序列可能没有收益。

```bash
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py bench --case encoder_long --impl wmma --iters 10 --repeats 3
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py bench --case encoder_long --impl async --iters 10 --repeats 3
compute-sanitizer --tool synccheck --error-exitcode 99 .venv/bin/python 02-triton-kernels/qwen35-cuda/sanitize.py
```

## 练习

1. 删除 wait 和删除 CTA barrier 分别可能造成什么错误？
2. 为什么异步搬运不能弥补所有不合并访存问题？
3. 记录 short/long 两种 case 的变化。如果 async 更慢，检查工作量和同步成本，不把实现名字当成性能结论。

过关：能画出两个 buffer 的生产者/消费者关系，并能用实测结果说明收益边界。
