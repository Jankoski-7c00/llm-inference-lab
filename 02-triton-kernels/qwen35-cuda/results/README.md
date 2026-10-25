# 实验结果与复现索引

这里保存可控合成输入上的精简实测数据；原始 profiling 文件和完整日志保存在本地忽略目录。学习者的能力验收仍由 [进度表](../notes/PROGRESS.md) 单独记录。

## 已完成的验证

| 项目 | 结果 | 证据 |
|---|---|---|
| 正确性单测 | 69 项通过 | [pytest 摘要](pytest_summary.txt) |
| 预设与原版对照 | 35 个 case/实现组合通过 | [correctness.csv](correctness.csv) |
| Memcheck | 26 个定向组合，0 errors | [sanitizer 摘要](sanitizer_summary.txt) |
| Synccheck | 26 个定向组合，0 errors | 同上 |
| Racecheck | 26 个定向组合，0 errors / 0 warnings | 同上 |
| Systems | 11 份报告已生成并确认目标 launch 数 | [kernel 摘要](nsys_kernel_summary.csv) |
| Compute | 后续补充实践，尚未采集报告 | [验证记录](validation.json) |

完整状态、复现命令、日志摘要和文件校验值见 [validation.json](validation.json)。

## 性能结果

[performance.csv](performance.csv) 记录 Graph 目标区间；[performance_eager.csv](performance_eager.csv) 单独记录 eager，包括主机提交间隙，两者不能混合计算加速比。

当前 Graph 数据使用 5 次 warmup，每组 10 次，共 3 组。表中 latency 为各组均值的中位数，CSV 同时记录样本 p10/p90。每次恢复相同初始状态，重置位于计时区间之外，数据较热。

| case | 基础版 (us) | 本次优化阶段 | 目标区间 (us) |
|---|---:|---|---:|
| cache_prefill | 19.046 | 向量化 | 8.397 |
| dynamic_int8 | 18.944 | warp 归约 | 5.018 |
| encoder_short | 42.230 | WMMA + async | 20.400 |
| delta_decode | 29.082 | 共享、行分块及 K 专用化 | 24.576 |

这些数字只描述本次输入和测量方法，不代表完整模型速度，也没有与生产库作性能比较。几微秒 case 的微小变化可能落在计时粒度或波动范围内。

实际硬件、软件、源码提交及 SHA256 分别保存在 [Graph 元数据](performance.metadata.json)、[eager 元数据](performance_eager.metadata.json)、[正确性元数据](correctness.metadata.json)。计划和讲义保持不绑定具体环境。

## 保留的失败尝试

- Head-major cache 没有进入向量路径，本次约 3.994→4.096 us，不视为有效优化收益。
- 短序列 SIMT 分块 attention 约 54.720 us，慢于基础版 42.230 us；引入 tile 不自动带来收益。
- Delta 直接融合约 36.250 us，慢于三次 launch 基线 29.082 us。仅共享计算为 38.202 us；随后行分块得到 29.184 us，再专用化 K 循环得到 24.576 us。不能把全部收益归因于共享 normalization；硬件瓶颈仍待 Compute 复核。
- [初始性能表](performance_initial.csv) 与 [元数据](performance_initial.metadata.json) 保留修改前结果。其中 delta 的历史 `optimized` 名称指当时的直接融合实现，对应当前 `fused`。
- 最初把完整 GPU reference 测试放入 memcheck 时，测试数值通过，但工具报告了 48 次 cuBLAS 内部 kernel 查询相关 CUDA API Error，因此该次运行不记为干净通过。随后使用 CPU goldens 的定向 worker，保持 CUDA API 错误报告开启，分别完成三个 sanitizer 检查。完整日志在 `outputs/memcheck_full.log`。

## Systems 报告

每份报告执行 3 次目标调用。Delta 基础版每次 3 个 kernel，所以总计 9 次 launch；其余报告总计 3 次。摘要由真实报告的 `nsys stats` 导出，并校验了这些数量。

| case | 实现 | 本地原始报告 |
|---|---|---|
| cache_prefill | baseline | [报告](../profiles/delivery/cache_prefill_baseline_float16.nsys-rep) |
| cache_prefill | optimized | [报告](../profiles/delivery/cache_prefill_optimized_float16.nsys-rep) |
| dynamic_int8 | baseline | [报告](../profiles/delivery/dynamic_int8_baseline_float16.nsys-rep) |
| dynamic_int8 | optimized | [报告](../profiles/delivery/dynamic_int8_optimized_float16.nsys-rep) |
| encoder_short | baseline | [报告](../profiles/delivery/encoder_short_baseline_float16.nsys-rep) |
| encoder_short | async | [报告](../profiles/delivery/encoder_short_async_float16.nsys-rep) |
| delta_decode | baseline | [报告](../profiles/delivery/delta_decode_baseline_float16.nsys-rep) |
| delta_decode | fused | [报告](../profiles/delivery/delta_decode_fused_float16.nsys-rep) |
| delta_decode | shared | [报告](../profiles/delivery/delta_decode_shared_float16.nsys-rep) |
| delta_decode | tiled | [报告](../profiles/delivery/delta_decode_tiled_float16.nsys-rep) |
| delta_decode | optimized | [报告](../profiles/delivery/delta_decode_optimized_float16.nsys-rep) |

这些原始文件不进入 Git，新 checkout 请重新生成。报告中的 profiler 时间和 Graph 微基准采用不同采集方式，不能逐行混合作为同一组 latency 数据。

## 复现

从仓库根目录执行：

```bash
.venv/bin/python -m pytest 02-triton-kernels/qwen35-cuda/tests -q
.venv/bin/python 02-triton-kernels/qwen35-cuda/suite.py --check-only --include-upstream --cases all
.venv/bin/python 02-triton-kernels/qwen35-cuda/suite.py --cases all --iters 10 --repeats 3
.venv/bin/python 02-triton-kernels/qwen35-cuda/suite.py --mode eager --cases cache_decode,dynamic_int8,encoder_short,delta_decode --iters 10 --repeats 3 --output 02-triton-kernels/qwen35-cuda/outputs/eager.csv
compute-sanitizer --tool memcheck --error-exitcode 99 .venv/bin/python 02-triton-kernels/qwen35-cuda/sanitize.py
compute-sanitizer --tool synccheck --error-exitcode 99 .venv/bin/python 02-triton-kernels/qwen35-cuda/sanitize.py
compute-sanitizer --tool racecheck --error-exitcode 99 .venv/bin/python 02-triton-kernels/qwen35-cuda/sanitize.py
.venv/bin/python 02-triton-kernels/qwen35-cuda/profile.py nsys --case delta_decode --impl optimized --output-dir 02-triton-kernels/qwen35-cuda/profiles/reproduction
```

相同输出目录中的原始报告不会被采集脚本覆盖；重新采集时选新目录。
