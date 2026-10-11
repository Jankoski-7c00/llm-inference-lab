# 2026 Q4 学习计划 · 02 CUDA / Triton 与 GPU 性能工程

> 本文档为个人学习规划,基于公开技术生态(vLLM / Triton / PyTorch / 开源模型),不包含任何雇主相关信息。

> **季度区间**:2026-10-01 至 2026-12-31(共 13 周)
> **时间投入**:占季度学习总投入的 30%,约每周 3 小时
> **季度主项目**:从零实现一个正确可测的 paged KV decode attention 原型(Triton)
> **适用背景**:AI 编译器工程师(SIMT 与 SIMD 并存的异构加速器环境,TVM 系编译路线),工作目标聚焦 Attention(FlashAttention 类算子、分块 Attention、动态形状、SWA、压缩 KV、分页 KV Cache),正在补强 SIMT GPU 通用视角

---

## 1. 方向定位与季度目标

### 1.1 为什么 CUDA / Triton 是"通用坐标系"

实际工作场景中的 SIMT 加速器,其编程模型、工具链、性能语言很可能是自研或深度定制的:Kernel 怎么写、profiling 看什么、性能瓶颈怎么描述,都绑定在特定平台上。这套经验在其原生平台上有效,但在外部市场上缺少可移植的"度量衡"——面试官、开源社区、论文作者谈论 GPU 性能时,用的都是同一套公共词汇:warp / CTA / SM、occupancy、coalescing、shared memory bank conflict、register pressure、nsight 报告、TFLOPs 利用率。

CUDA 和 Triton 就是这套公共词汇的载体:

- **CUDA** 是行业事实标准的 GPU 编程模型。即使日常不写 CUDA C++,也必须能读懂 CUDA 层面的概念,否则 FlashAttention 论文、CUTLASS 代码、Nsight 报告都无法进入。
- **Triton** 是 LLM Inference 领域最实用的 Kernel 开发语言:vLLM、SGLang、FlashInfer、unsloth 大量核心 Kernel 用 Triton 写成,且 Triton 本身也是编译器——与自己的 TVM 背景直接互译,学习曲线最平缓,产出最容易展示(GitHub 可跑)。

**一句话定位**:本方向不是"转岗 CUDA 工程师",而是把已有的 NPU Attention Kernel 经验翻译到行业通用坐标系里,并沉淀为外部可识别的、带性能分析的公开作品。

### 1.2 与 NPU 经验的映射关系

已有的 SIMD NPU Attention Kernel 实践不是从零开始,概念可以大面积迁移,本季度的学习应主动做"对表":

| 已熟悉的 NPU / TVM 概念 | CUDA / SIMT GPU 对应物 | 需要补强的差异点 |
| --- | --- | --- |
| 片上 SRAM / Local Memory 分块 | shared memory(SMEM) | bank conflict、SMEM 与 register 的分配权衡 |
| DMA 异步搬运 / double buffer | cp.async / TMA、num_stages 流水 | warp 级流水线调度由编译器和硬件共同完成 |
| 向量 lane 并行 | warp(32 线程)内 SIMT 执行 | 分支分化(divergence)、warp shuffle reduction |
| tile / loop schedule(TVM tir) | Triton 的 BLOCK_M / BLOCK_N、tl.dot | occupancy 由 register/SMEM 占用反推 |
| 片上计算利用率 | SM busy / compute utilization | roofline 视角:先判断 memory-bound 还是 compute-bound |
| 平台自带 profiler | Nsight Compute(ncu)/ Nsight Systems(nsys) | metric 名称与解读方法需要重新建立直觉 |

**方法论**:每学一个 GPU 概念,先写下"这在 NPU 上对应什么、差异是什么",避免把 GPU 当全新领域重学一遍。

### 1.3 12 月底应达到的状态

- 能用 Triton 独立写出 reduction、softmax、tiled attention、decode attention(含 paged KV 地址映射)四类 Kernel,全部与 PyTorch reference 数值对齐。
- 每个 Kernel 都有 ncu profiling 记录,能说清它是 memory-bound 还是 compute-bound、瓶颈在哪个硬件单元。
- 能回答验收问题:"某个 Kernel 优化为什么能够或不能够改善 TPOT"——即能把 Kernel 级指标(latency、effective bandwidth)与端到端指标(TPOT、throughput)连接起来。
- GitHub 上有一个可运行的 paged KV decode attention 原型仓库,带 README、benchmark 脚本和 profiling 报告——这就是本季度的"可展示项目"。
- 季度至少产出 **3 个带 profiling 记录的 Triton Kernel**(对齐"每 6 个月 4 个以上 CUDA/Triton Kernel"的节奏要求,本季度 3 个,下季度补齐并超额)。

---

## 2. 每周时间安排

每周约 3 小时,切分为两块,职责分离:

### 2.1 工作日练习夜:90 分钟(建议固定在某一天晚上,如周三 20:30–22:00)

**只做小颗粒度 Kernel 练习,不碰主项目**。90 分钟的标准结构:

| 时间段 | 内容 |
| --- | --- |
| 0–15 min | 回顾上次练习的 TODO 和遗留问题 |
| 15–60 min | 写 / 改 Kernel 代码,跑通正确性验证 |
| 60–80 min | 跑 ncu,记录 profiling 字段(见 5.5 模板) |
| 80–90 min | 写 3–5 行练习日志:今天做了什么、瓶颈是什么、下次做什么 |

练习顺序按月度路线图(第 3 节)推进;卡住超过两次练习夜的问题,记入"风险与求助清单"(第 7 节),去 GPU MODE Discord 或向同行请教,不要死磕。

### 2.2 周末主项目时段:半天(3–4 小时)中的约 1.5–2 小时划给本方向

周末半天是季度主项目(paged KV decode attention 原型)的承载时间,本方向与其他方向共享这半天。本方向内的建议切分:

- **10 月**:周末时间主要用于环境搭建、profiler 入门和把练习夜的 reduction/softmax 整理成可展示形态(测试、README、profiling 记录)。
- **11 月**:周末时间全部投入 tiled attention + online softmax 原型——这是主项目的主体部分,单点连续时间最适合做这种有状态累积的工作。
- **12 月**:周末时间投入 decode attention、paged KV 地址映射、整仓整理(测试矩阵、benchmark、profiling 报告、README)。

**纪律**:练习夜不追大目标(写不出完整 Kernel 不焦虑,能跑通一个变体+一条 profiling 记录就算完成);周末不碎片化(不开新坑,只推进主项目当前里程碑)。

---

## 3. 月度路线图

### 3.1 10 月:基础 Kernel(reduction / softmax)+ profiler 入门

**目标**:建立 Triton 手感 + ncu 工作流,产出本季度前 2 个带 profiling 记录的 Kernel。

- 第 1 周(10/1–10/4,注意国庆假期可弹性挪用)
  - 搭建环境(见 5.2),跑通 Triton 官方 tutorials 01 vector-add 与 02 fused-softmax。
  - 用 Triton Puzzles 做热身(可在 CPU interpreter 模式下做,无需 GPU)。
- 第 2 周
  - 练习 Kernel 1:**reduction(sum/max)**,从单 block 版本起步,再写多 block 两级归约版本。
  - 正确性:对齐 `torch.sum` / `torch.max`,容差见 5.4。
  - 里程碑 M1:reduction Kernel 正确 + 第一条完整 ncu 记录落盘。
- 第 3 周
  - 练习 Kernel 2:**softmax(行归一化)**,理解"一次遍历 online max/sum"与"多次遍历 naive"的访存差异。
  - 里程碑 M2:softmax Kernel 正确 + profiling 记录;能口述两种实现的 effective bandwidth 差异来源。
- 第 4 周
  - 用 ncu 对 reduction 和 softmax 做系统分析:访存(coalescing)、occupancy、寄存器占用,记录到性能表(5.5 模板)。
  - 读 CUDA C++ Programming Guide 的硬件模型章节 + PMPP 第 2–5 章,建立 warp/CTA/SM 词汇表。
  - 月末检查点:2 个 Kernel + 2 份 profiling 报告;能解释 occupancy 受限的三个常见原因(register 用量、SMEM 用量、block 尺寸/调度粒度)。

### 3.2 11 月:tiled attention + online softmax,对齐 PyTorch reference

**目标**:完成主项目的主体——tiled attention 与 online softmax 原型,结果与 PyTorch reference 对齐。这是本季度第 3 个带 profiling 记录的 Kernel。

- 第 5–6 周
  - 精读 Triton 官方 06 fused-attention tutorial,逐段注释其 online softmax 逻辑(m_i / l_i / alpha rescale)。
  - 对照 FlashAttention 1/2 论文,把"不物化 N×N 矩阵"的 IO-aware 论证用自己的话写下来。
- 第 7–8 周
  - 练习 Kernel 3:**tiled attention(non-causal 起步)**,固定 head_dim=64/128,BLOCK_M / BLOCK_N 手动选参。
  - 正确性:对齐 PyTorch reference(`torch.nn.functional.scaled_dot_product_attention` 或手写 `softmax(QK^T/√d)V`),容差见 5.4。
  - 里程碑 M3:tiled attention 正确 + profiling 记录;画出本 Kernel 在目标卡上的 roofline 位置。
- 第 9 周
  - 加入 causal mask;尝试 num_warps / num_stages 扫描,观察 occupancy 与 register pressure 变化。
  - 里程碑 M4:causal 版本正确;完成一次"config 扫描 → ncu 对比 → 结论"的完整循环。
- 月末检查点:3 个 Kernel + 3 份 profiling 报告(达成季度硬性底线);能解释 online softmax 为什么省内存。

### 3.3 12 月:decode attention + paged KV 地址映射(主项目收口)

**目标**:实现一个简单 decode attention,并逐步加入 paged KV 地址映射,形成可展示的完整原型仓。

- 第 10 周
  - 练习 Kernel 4:**decode attention(q_len=1,连续 KV)**,理解 decode 阶段是 GEMV 型、强 memory-bound 的负载。
  - 正确性:对齐 PyTorch reference 的 decode 单步计算。
- 第 11–12 周
  - 加入 **paged KV 地址映射**:KV Cache 按固定大小 block(page)存储,Kernel 内通过 block table 间接寻址 `physical_block = block_table[seq, logical_block]`,再做页内偏移。
  - 分两步走:先单 sequence、block 内连续;再支持 batch + 变长 seqlen + 非满页(尾页 mask)。
  - 里程碑 M5:paged decode attention 在随机构造的 block table 上与连续 KV 的 reference 逐位对齐(同一逻辑序列、不同物理布局)。
  - 里程碑 M6:kernel 支持 GQA(KV head 数 < Q head 数),对齐 DeepSeek 类模型的实际需求。
- 第 13 周(12/29–12/31,收口)
  - 整理仓库:README(设计说明 + 复现命令)、测试矩阵(不同 seqlen / head 配置 / block size)、benchmark 脚本、全部 profiling 报告。
  - 写一段"Kernel 指标 → TPOT"的分析:decode attention 的访存量 ≈ KV Cache 读取量,优化 effective bandwidth 如何传导到 TPOT;什么优化(如纯 occupancy 提升)可能改善不了 TPOT。
  - 月末/季度检查点:对照第 6 节验收清单逐项自测。

---

## 4. 学习资料清单

下表中标注 ✅ 的链接已于 2026 年核实真实存在;个别未逐字核实的已注明。

| 名称 | 类型 | 用途 | 优先级 |
| --- | --- | --- | --- |
| Triton 官方 Tutorials([triton-lang.org/main/getting-started/tutorials](https://triton-lang.org/main/getting-started/tutorials/))✅ | 官方教程 | 01 vector-add、02 fused-softmax 入门;03 matmul 选读 | P0 |
| Triton 06 fused-attention([GitHub 源码](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py))✅ | 官方教程源码 | 11 月精读对象:tiled attention + online softmax 的权威参考实现 | P0 |
| Triton Puzzles([srush/Triton-Puzzles](https://github.com/srush/Triton-Puzzles))✅ | 交互练习 | 10 月热身;支持 CPU interpreter 模式,无卡也能做 | P0 |
| CUDA C++ Programming Guide([docs.nvidia.com/cuda/cuda-c-programming-guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/))✅ | 官方文档 | 硬件模型、内存层级、stream/event 章节的权威定义;按需查阅不精读 | P1 |
| Nsight Compute 文档([NsightCompute](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html) / [Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) / [CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html))✅ | 官方文档 | ncu 命令行与 metrics 解读;Profiling Guide 的 Metrics Reference 是查 metric 名的手册 | P0 |
| 《Programming Massively Parallel Processors》(PMPP,4th ed.,Kirk & Hwu,Morgan Kaufmann 2022) | 教材(纸质/电子) | 系统建立 SIMT 词汇表;本季度读第 2–5 章(架构、内存、性能) | P1 |
| FlashAttention 论文([arXiv:2205.14135](https://arxiv.org/abs/2205.14135))✅ | 论文 | IO-aware 论证与 online softmax 推导,11 月精读 | P0 |
| FlashAttention-2 论文([arXiv:2307.08691](https://arxiv.org/abs/2307.08691))✅ | 论文 | 并行度与 work partition 改进,理解 decode/prefill 差异的铺垫 | P1 |
| vLLM / PagedAttention 论文([arXiv:2309.06180](https://arxiv.org/abs/2309.06180))✅ | 论文 | 12 月 paged KV 地址映射的设计依据(block table、页式管理) | P0 |
| flash-attention 官方仓库([Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention))✅ | 开源代码 | 对照工业级实现的接口与行为;不做为本季度精读对象 | P2 |
| GPU MODE(原 CUDA MODE)lectures 仓库([gpu-mode/lectures](https://github.com/gpu-mode/lectures))✅ | 课程/讲座材料 | Lecture 1(profiling)、4(体系结构)、9(Reductions)、12(Flash Attention)、14(Triton 实践者指南)与本季度内容直接对应 | P1 |
| GPU MODE resource-stream([gpu-mode/resource-stream](https://github.com/gpu-mode/resource-stream))✅ + Discord([discord.gg/gpumode](https://discord.gg/gpumode))✅ | 社区 | 卡住时的求助渠道 + 新资料索引;leaderboard 可作为额外练习场 | P2 |
| Horace He《Making Deep Learning Go Brrrr From First Principles》([horace.io/brrr_intro.html](https://horace.io/brrr_intro.html))✅ | 博客 | compute / bandwidth / overhead 三类瓶颈的第一性原理框架,10 月必读 | P0 |
| Nsight Systems User Guide(docs.nvidia.com 站内检索 "Nsight Systems")| 官方文档 | nsys 时间线分析;**具体章节链接需自行检索确认** | P2 |
| kipp.ly《Transformer Inference Arithmetic》 | 博客 | 推理 FLOPs / KV Cache 心算,支撑 TPOT 分析;**需自行检索确认** | P2 |

> 说明:PMPP 为出版物,无官方免费链接,购买或借阅即可;其余 ✅ 链接均经检索核实。若链接失效,以名称检索同标题资料。

---

## 5. 实验设计与环境

### 5.1 硬件

- **首选**:一张 NVIDIA 消费卡即可(RTX 3060/4060 及以上,Ampere 架构起;有 Tensor Core 更利于 `tl.dot`)。性能绝对值不重要,重要的是 metrics 的相对变化。
- **无本地卡时的降级**:
  1. Triton Puzzles 等概念练习用 CPU interpreter 模式完成;
  2. 需要真实 profiling 时按小时租用云 GPU(AutoDL / RunPod 等,选带 ncu 权限的实例;注意部分云平台默认容器无 profiling 权限,需开启 `--cap-add SYS_ADMIN` 或宿主配合);
  3. 极端情况下(CUDA 完全不可用),练习改为纯 Triton 代码编写 + 静态分析(ptxas 输出的 register/SMEM 用量),profiling 记录顺延到有卡时补齐。

### 5.2 开发环境

- OS:Linux(原生或 WSL2);macOS 不支持 CUDA,不可作为本方向主环境。
- Python 3.10+,PyTorch 与 Triton 版本搭配建议:
  - 省心路径:安装官方 release 的 PyTorch(如 2.5.x 系),其自带匹配的 Triton(`torch` 的 triton 依赖),避免源码混装。
  - 若需要 Triton 新特性再单独升级 Triton,并锁定一组"PyTorch x.y + Triton a.b"组合写进 requirements,季度内不随意升级。
- CUDA Toolkit 与驱动:`nvidia-smi` 可见卡即够;ncu 版本随 Nsight Compute 独立安装,与 CUDA Toolkit 版本大致匹配即可。
- 工具:`ncu`(Nsight Compute CLI)、`nsys`(Nsight Systems)、`proton`(Triton 自带 profiler,选做)。

### 5.3 各 Kernel 的练习目标与里程碑定义

| # | Kernel | 练习目标 | 里程碑(Definition of Done) |
| --- | --- | --- | --- |
| 1 | reduction | 多级归约、warp 内/间协作、coalescing | 任意长度 fp32 输入对齐 `torch.sum`;ncu 报告落盘;能说清两级归约的访存量 |
| 2 | softmax | 行归约、online max/sum、数值稳定 | 对齐 `torch.softmax(dim=-1)`;对比 naive 两遍实现的 effective bandwidth 并记录 |
| 3 | tiled attention | tiling、online softmax、`tl.dot`、register pressure | 对齐 SDPA reference(causal + non-causal);完成一次 config 扫描的 ncu 对比 |
| 4 | paged decode attention | decode 负载特征、间接寻址、GQA | 随机 block table 下与连续 KV reference 对齐;支持 batch/变长/GQA;含 benchmark + 完整 profiling |

### 5.4 正确性验证方法

- reference:fp32 下用 PyTorch 手写表达式或 `torch.nn.functional.scaled_dot_product_attention` 计算。
- 输入构造:固定 seed 的随机张量;覆盖边界形状(非 BLOCK 整数倍的 seqlen、尾页不满的 paged KV、seqlen=1)。
- 容差约定(写入每个 Kernel 的测试文件头部):
  - fp32: `torch.testing.assert_close(rtol=1e-5, atol=1e-5)`
  - fp16/bf16: `rtol=1e-2, atol=1e-2`(online softmax 的累加顺序不同,需放宽)
  - paged 与连续布局对比:同一 fp32 数据仅物理布局不同,要求 `atol=0`(bitwise 一致)或至多 1 ulp 级差异。

### 5.5 profiling 工作流

**ncu 基本命令**(逐 Kernel 执行,建议写成 Makefile 或脚本):

```bash
# 完整 sections(首次分析)
ncu --set full -o reports/reduction_v1 python bench_reduction.py
# 只抓指定 kernel,控制开销
ncu -k regex:"softmax" --launch-count 3 --set detailed -o reports/softmax_v2 python bench_softmax.py
# 快速关注 roofline + occupancy
ncu --section SpeedOfLight --section Occupancy -o reports/attn_sol python bench_attn.py
```

**ncu 重点关注 metrics(按规划要求的记录字段映射)**:

| 记录字段 | 对应 ncu metric(名称以 Profiling Guide 的 Metrics Reference 为准) |
| --- | --- |
| latency | `gpu__time_duration.sum` |
| effective bandwidth | `dram__bytes.sum.per_second`(配合 `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` 看占比) |
| compute utilization | `sm__throughput.avg.pct_of_peak_sustained_elapsed`(SpeedOfLight 的 SM 占比) |
| occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active`(achieved occupancy);对照 `launch__occupancy_limit_*` 系列找限制源 |
| register | `launch__registers_per_thread` |
| shared memory | `launch__shared_mem_per_block_static` / `_dynamic` |
| bank conflict | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` / `..._op_st.sum` |
| warp stall 归因 | `smsp__average_warps_issue_stalled_*`(如 `long_scoreboard` = 等显存) |

**nsys 基本命令**(看 Kernel 间空隙与 launch overhead):

```bash
nsys profile -o reports/decode_timeline python bench_decode.py
nsys stats reports/decode_timeline.nsys-rep   # 命令行快速汇总
```

**性能记录表模板**(每次实验追加一行,存为 `perf_log.md`):

| 日期 | Kernel | 版本/shape | latency (µs) | effective BW (GB/s) | compute util (%) | occupancy (%) | reg/thread | SMEM/block (KB) | 主要 stall | 结论/下一步 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-XX | reduction | v1, N=1M fp32 | | | | | | | | |

### 5.6 学习范围自查(规划硬性要求)

tiling、online softmax、register pressure、memory layout、warp/CTA/SM、occupancy、coalescing、shared memory、bank conflict、stream/event——其中 stream/event 不在四个主 Kernel 的必经路径上,安排在 12 月收口周用 `torch.cuda.Event` 做精确计时、用多 stream 观察 overlap 时集中补齐。

---

## 6. 能力验收标准

### 6.1 自测问题清单(12 月底逐条口头/书面回答)

1. online softmax 为什么省内存?它把 N×N 中间矩阵的访存量从多少降到多少?
2. bank conflict 如何检测?ncu 里看哪个 metric?shared memory 的 bank 数与访问步长是什么关系?
3. occupancy 受限的三个常见原因分别是什么?如何在 ncu 里确认当前 Kernel 被哪一个限制?
4. 一个 Kernel 的 effective bandwidth 已经达到显存峰值的 90%,继续优化还有什么空间?这属于哪类 bound?
5. register pressure 过高会怎样?Triton 里哪些写法容易推高寄存器用量?
6. coalescing 失败的访存在 ncu 里表现为什么?如何修改 memory layout 修复?
7. decode attention 为什么几乎一定是 memory-bound?它的算术强度大约是多少?
8. **验收核心题**:某个 Kernel 优化(例如 tile 调大、occupancy 提升、bank conflict 消除)为什么能够或不能够改善 TPOT?请用一个本季度的真实实验数据回答—— Kernel latency、effective bandwidth 的变化如何传导到每 token 延迟;什么情况下 Kernel 优化了 30% 而 TPOT 几乎不变(例如瓶颈在别的 Kernel、launch overhead、或受限于权重读取带宽)。
9. warp、CTA、SM 三层各自对应什么硬件资源?Triton 的一个 program 实例落在哪一层?
10. stream 和 event 分别解决什么问题?为什么 `torch.cuda.synchronize()` 前后的时间不能直接当 Kernel latency?

### 6.2 交付物清单

- [ ] **代码**:≥ 3 个 Triton Kernel(reduction、softmax、tiled attention)+ paged decode attention 原型,全部带正确性测试。
- [ ] **profiling 报告**:每个 Kernel 至少一份 ncu 报告(.ncu-rep 文件)+ 按 5.5 模板填写的 perf_log.md。
- [ ] **主项目仓库**:paged KV decode attention 原型,含 README(设计、复现命令、结果)、benchmark 脚本、测试矩阵。
- [ ] **分析短文**:一段"Kernel 指标 ↔ TPOT"分析(可放主项目 README 或单独 md),对应验收核心题。
- [ ] **练习日志**:13 周的练习夜日志(每次 3–5 行即可)。

---

## 7. 风险与降级方案

| 风险 | 触发信号 | 降级方案 |
| --- | --- | --- |
| 时间不足(最常见) | 连续两周练习夜缺席或周末被工作挤占 | 按规划硬性要求收缩:**至少保留周末主项目 3 小时 + 1 小时 GPU 练习**。砍掉的顺序:P2 资料阅读 → 练习日志精简 → 10 月 reduction 与 softmax 合并为一个练习 → 主项目 scope 收缩为"单卡、fp16、无 GQA" |
| 无本地 NVIDIA 卡 | 环境搭建失败 | 启用 5.1 降级链:CPU interpreter 做概念练习 → 按小时租云 GPU 做 profiling;profiling 记录允许集中补做,不阻塞代码里程碑 |
| ncu 在租用的云实例上无权限 | `ncu` 报 permission 错误 | 换支持 profiling 的实例规格;临时用 `torch.cuda.Event` + proton 记录 latency/bandwidth,占用类 metrics 后续补 |
| Triton 版本 API 变动 | tutorial 代码与本地版本不匹配 | 锁定季度初验证过的版本组合,写进 requirements;优先用本地版本自带的 tutorials 副本而非网上最新代码 |
| online softmax / paged 地址映射卡住 | 同一问题占用两次以上练习夜 | 先写"naive 但正确"版本(多次遍历 softmax / 连续 KV)保住正确性,优化版作为后续迭代;去 GPU MODE Discord 提问 |
| 主项目 12 月做不完 | 第 11 周末 M5 未达成 | 收缩验收:GQA(M6)挪到 2027 Q1;paged 只要求单 sequence + 固定 block size;profiling 报告允许只覆盖 decode Kernel 一个 |

**降级原则**:任何情况下都不砍"正确性验证"和"至少 3 个 Kernel + 3 份 profiling 记录"这两条季度硬性底线;宁可缩小 shape 和 feature 范围,也不交付未验证的代码。

---

*制定日期:2026 Q3 末;复盘节点:每月末检查点 + 2026-12-31 季度总复盘(对照第 6 节)。*
