# 03 · End-to-End Benchmark

**学习方向**:端到端性能验证——所有 kernel / Runtime / 编译器层的改动,最终都要在端到端指标上说话。本目录沉淀统一的 benchmark 方法与结果档案,避免"自我感觉变快了"。

## 核心指标

- **TTFT**(Time To First Token):prefill 延迟,影响首 token 体验。
- **TPOT**(Time Per Output Token):decode 单 token 延迟,决定流式体感。
- **吞吐**(throughput):tokens/s 或 requests/s,在固定并发与负载下测量。
- **峰值显存**:权重 + KV cache + 激活的实测峰值。

## 记录字段约定

每条 benchmark 记录必须包含:

- 模型与权重来源(公开模型)、dtype、量化方式
- vLLM 版本与启动参数(block size / chunked prefill / prefix caching 等开关)
- 输入分布:prompt 长度、输出长度、并发数、到达模式
- 硬件:GPU 型号、数量、显存、驱动 / CUDA 版本
- 结果:TTFT、TPOT、吞吐、峰值显存(注明单位与统计口径:均值 / P50 / P99)
- 复现命令与 commit

## 状态

⬜ 未开始(配合 11 月 vLLM 控制变量实验启用)
