# 第 13 课：从报告到一次可复盘的优化

本课 Compute 实操尚待补充，已有材料与未完成项统一见[课程入口的状态说明](../README.md#ncu-实操待补充)。

## 1. 建立可信计时

正确性检查、构建和 warmup 完成后再计时。Graph 微基准使用外部 event 节点，重置拷贝在每对 event 之外。eager 模式额外包含主机提交间隙；两种结果分别记录。

```bash
.venv/bin/python 02-triton-kernels/qwen35-cuda/suite.py --output 02-triton-kernels/qwen35-cuda/outputs/suite.csv
```

记录各组均值的中位数、样本 p10/p90 与重复数。微小差异应重测同一 case，不能把所有小波动都写成收益。

## 2. Systems：当前可完成的实践

```bash
.venv/bin/python 02-triton-kernels/qwen35-cuda/profile.py nsys --case encoder_short --impl async
nsys stats --report cuda_gpu_kern_sum 02-triton-kernels/qwen35-cuda/profiles/encoder_short_async_float16.nsys-rep
nsys-ui 02-triton-kernels/qwen35-cuda/profiles/encoder_short_async_float16.nsys-rep
```

在 GUI 中找到 `qwen35_target` NVTX 区间：选择对应 CUDA stream，查看 kernel 名称、持续时间、前后空隙和拷贝。重置操作在 NVTX 区间之外；整个 capture 内仍可见它们，分析时不要合并成算子开销。

远程机器采集 CLI 报告后，可下载到本地用兼容的 Nsight Systems GUI 打开。命令行先确认目标 kernel 确实存在，空报告不算完成。

## 3. Compute：后续补充实践

```bash
.venv/bin/python 02-triton-kernels/qwen35-cuda/profile.py ncu --case encoder_short --impl async
ncu --import 02-triton-kernels/qwen35-cuda/profiles/encoder_short_async_float16.ncu-rep --page details
ncu-ui 02-triton-kernels/qwen35-cuda/profiles/encoder_short_async_float16.ncu-rep
```

首次使用 basic；需要全面分析时另选输出目录并加 `--set full`。采集脚本预热后才启用 profiler，使用 NVTX 过滤目标区间。Compute 可能多次 replay，不能把被插桩过程的墙钟时间当成正常运行 latency；内核写状态时要保持 replay 输入一致。

若报 `ERR_NVGPUCTRPERM`，保留失败记录并将该实践标为待完成。[NVIDIA 权限说明](https://developer.nvidia.com/ERR_NVGPUCTRPERM)提供宿主机/容器授权方法，操作应由有权限的平台管理员处理。

## 4. 由问题选择指标

| 问题 | 观察方向 | 可验证的修改 |
|---|---|---|
| 小 kernel 之间空隙多 | Systems launch 与同步 | 融合、减少同步、Graph |
| 访存效率低 | Memory Workload、实际事务与缓存命中 | 合并访问、向量化、布局 |
| occupancy 受限 | Launch/Occupancy、寄存器和 SMEM 限制 | tile 大小与资源预算 |
| 片上通信多 | shared memory、bank conflict | warp 分工与布局 |
| 指令等待多 | Scheduler/Warp State，结合指令依赖 | 复用、流水线、减少依赖 |
| Tensor Core 使用不足 | Compute Workload、指令与矩阵尺寸 | WMMA、对齐与 tile |

具体 metric 名称通过 `ncu --query-metrics` 和 section 列表确认。不能只看到 low occupancy 就认定它是瓶颈，也不能把逻辑字节数除 latency 叫作实测 DRAM 吞吐。

## 5. 实验记录

每次只改变一个因素：先写预测，再运行，最后写支持/否定假设的数据。使用算子目录中的 REPORT，保留失败尝试。参阅 [Systems 官方指南](https://docs.nvidia.com/nsight-systems/UserGuide/) 和 [Compute 官方指南](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。
