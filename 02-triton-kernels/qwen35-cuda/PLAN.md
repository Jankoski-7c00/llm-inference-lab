# Qwen3.5 CUDA 算子面试学习计划

## 1. 目标与学习方式

围绕仓库中的 vLLM 源码，完成普通缓存、动态量化缓存、MM encoder FA2 和 fused sigmoid gating delta rule 的 CUDA 实现、正确性验证、性能分析与复习笔记。

每课按照“读懂输入输出 → 推导公式和索引 → 手算 → 补写 CUDA → 验证 → 分析性能 → 整理笔记”推进。先自己完成练习，再对照答案，能讲清楚后再进入下一课。

## 2. 课程安排

| 课程 | 内容 | 验收 |
|---|---|---|
| 1 | 模型三条调用链与四个算子契约 | 数据流图、shape/stride/副作用表 |
| 2–3 | 分页缓存寻址、基础 CUDA、向量化 | 基础版与优化版通过测试，完成一次性能对比 |
| 4–5 | 动态 scale、量化、warp 归约和舍入 | INT8/FP8 实现、边界测试和归约实验 |
| 6–9 | Attention、online softmax、FA2 分工、WMMA、异步搬运 | 支持非因果变长输入的 CUDA FA2，逐阶段对比 |
| 10–12 | 门控、状态递推、多步更新与融合 | 同时验证输出和状态，解释融合取舍 |
| 13–14 | Profiling、实验复盘、闭卷练习和模拟面试 | 四份实验报告与可复习讲解稿 |

普通缓存覆盖 NHD、head-major 与固定 scale FP8；动态缓存覆盖独立 K/V scale、INT8/FP8 与不同 K/V dimension。FA2 覆盖尾块和非整齐 head dimension。Delta rule 覆盖 normalization、状态索引、短序列 spec 与 FP32 累加。

## 3. 实验与验证

- 每个算子独立报告，共享编译、测试和测量工具；保留数学 reference、CUDA 基础版和优化版。
- 缓存检查随机 slot、负 slot、跨页和未写区域；量化检查全零、极值、scale 下限、舍入和饱和。
- FA2 检查短序列、变长、尾块、大 logits；Delta rule 检查单步、多步、初始状态、无效索引及非方形状态。
- 普通复制精确一致；浮点计算固定 dtype 容差；量化分别验证 scale、存储值和反量化误差。
- 每次优化先通过正确性，再计时；使用 Compute Sanitizer 检查越界和同步。

## 4. Profiling 安排

当前先完成 CUDA Events 计时与 Nsight Systems 报告的生成、阅读和对比。排除编译、分配、状态重置开销，分别记录 eager 和 CUDA Graph 结果。

Nsight Compute 报告采集与硬件指标分析列为后续补充实践，不作为前半段学习的前置条件。届时使用访存、寄存器、occupancy、shared memory、warp stall 和 Tensor Core 指标复核之前的瓶颈假设。

每次只修改一个因素，按“现象 → 证据 → 假设 → 修改 → 结果”记录，包括失败尝试与收益边界。

## 5. 笔记与最终验收

每课留下公式、布局图、线程分工、核心代码、测试、复现命令、结果、错题及复习答案。每个算子准备 30 秒与 3 分钟讲解稿。

最终要求能独立补全核心 CUDA 代码、解释正确性边界、读懂自己的报告，并用实验数据说明优化为什么有效或无效。完整引擎替换、反向传播、INT4 和 KDA 不纳入本轮。
