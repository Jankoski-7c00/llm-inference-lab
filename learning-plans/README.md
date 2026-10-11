# Learning Plans

inference-lab 的季度学习计划，包含目标、时间安排、里程碑和验收标准。计划与[实验及阅读笔记](../README.md)在同一 Git 仓库中维护。

## 2026 Q4

覆盖 2026 年 10–12 月，当前主线是 Attention 子系统基础与 paged KV decode attention 原型。

| 文档 | 内容 | 时间占比 |
|---|---|---|
| [00_季度总览与执行机制.md](./2026Q4/00_季度总览与执行机制.md) | 季度总览、周计划、里程碑与验收 | — |
| [01_vLLM与推理Runtime.md](./2026Q4/01_vLLM与推理Runtime.md) | 请求生命周期、调度与 KV 管理 | 30% |
| [02_CUDA与Triton_GPU性能.md](./2026Q4/02_CUDA与Triton_GPU性能.md) | 算子实现、profiling 与季度主项目 | 30% |
| [03_编译器横向_MLIR与Inductor.md](./2026Q4/03_编译器横向_MLIR与Inductor.md) | 编译器概念与 lowering 路径 | 15% |
| [04_Attention与新模型跟踪.md](./2026Q4/04_Attention与新模型跟踪.md) | 模型结构卡片与适配清单 | 15% |
| [05_异构与分布式基础.md](./2026Q4/05_异构与分布式基础.md) | 异构硬件与分布式基础 | 10% |

## 计划与实践的对应关系

- [Runtime 笔记](../01-vllm-runtime-trace/README.md)：从[v0.31.0 起步问题清单](../01-vllm-runtime-trace/v0.31.0/00-first-request-reading-notes.md)开始。
- [Triton 实验](../02-triton-kernels/README.md)：每个实验单独建目录，复用报告模板。
- [端到端 benchmark](../03-e2e-benchmark/README.md)：统一记录环境、负载和性能指标。
- [编译器笔记](../04-compiler-notes/README.md)与[模型结构卡片](../05-model-cards/README.md)：沉淀阅读结论和适配问题。

本分支以 Mac 源码阅读、tokenizer 和 CPU 张量练习为重点。GPU 相关事项保留为季度目标，不作为本机已完成的实验。环境恢复与 submodule 下载命令见[项目首页](../README.md)。

计划里的版本、源码路径和环境示例以实际 checkout 与运行配置为准。完成事项应关联笔记、实验报告或可复现结果；新增季度时建立 `YYYYQn/` 目录并更新本索引。
