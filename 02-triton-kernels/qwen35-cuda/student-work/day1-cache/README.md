# 第一天 Cache 练习源码

这里保存补写完成的 Cache 练习。绑定、reference 和测试继续复用课程目录；其他算子的代码保留课程骨架中的实现。

在远端 GPU 服务器的仓库根目录执行：

```bash
export QWEN35_KERNEL_SOURCE="$PWD/02-triton-kernels/qwen35-cuda/student-work/day1-cache/kernels.cu"
.venv/bin/python -m pytest 02-triton-kernels/qwen35-cuda/tests/test_cache.py -q

.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py bench \
  --case cache_prefill --impl optimized --dtype bfloat16 \
  --mode graph --warmup 5 --iters 20 --repeats 5

.venv/bin/python 02-triton-kernels/qwen35-cuda/profile.py ncu \
  --case cache_prefill --impl optimized --dtype bfloat16 --set full
```

性能与 profiling 显式使用 BF16；正确性测试保留课程原有多 dtype 覆盖。

实验数据、测量元数据、日志和完整个人笔记保存在本地忽略目录。此次实验笔记位于本地 `02-triton-kernels/qwen35-cuda/results/student-day1-20261025/REPORT.md`，原始报告位于 `profiles/`，离线包位于 `outputs/archives/`。
