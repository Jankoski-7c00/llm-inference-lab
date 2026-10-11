# 2026 Q4 学习计划 · 方向一：vLLM 与 LLM 推理 Runtime

> 本文档为个人学习规划,基于公开技术生态(vLLM / Triton / PyTorch / 开源模型),不包含任何雇主相关信息。

> 覆盖周期：2026-10-01 ～ 2026-12-31（职业规划"未来九十天计划"，第一阶段 2026.9–2027.3 Attention 子系统基础期内）
> 时间预算：季度总投入占比 30%，约每周 3 小时；本季度第一优先级业余方向
> 长期目标锚点：2029 年成为懂编译器与硬件的 LLM Inference / AI Systems Performance 工程师

---

## 方向定位与季度目标

### 为什么学

我的工程实践聚焦在 Compiler / Kernel 一侧：FlashAttention 类算子、分块 Attention(blockwise attention)的编译重构、动态形状、DeepSeek 新架构适配、SWA、压缩 KV、分页 KV Cache。这些工作有一个共同盲区：**Kernel 的输入是谁生成的**——Attention metadata、block table、slot mapping 这些张量在进入 Kernel 之前，是 Runtime 层（调度器、KV manager、model runner）一步步构造出来的。不理解 Runtime，就只能"接需求"，无法参与"定义需求"。

在 Model → Runtime → Compiler → Kernel → Hardware 能力链中：

- **Kernel / Compiler**：当前主战场，持续投入（主项目承担）。
- **Runtime（本方向）**：本季度重点补齐的短板。具体欠缺：请求调度、continuous batching、KV allocation、attention metadata 的完整生成链路。
- **Model / Hardware**：本季度不主动投入，只在与 Runtime 交叉处顺带吸收（如 MLA 对 KV layout 的约束）。

vLLM 是当前开源生态中 runtime 设计最成熟、源码可读性最好的系统，且 V1 架构（2025 年起成为默认引擎）已经收敛，适合作为系统性学习的载体。

### 12 月底应达到的状态（可验收）

1. 能脱稿讲清楚一个 vLLM V1 decode request 的完整生命周期：scheduler 如何选中它、KV block 如何分配、block table / slot_mapping 如何生成、attention metadata 如何组装、最终如何调用到 FlashAttention backend 的 kernel。
2. 在 Qwen2.5-0.5B 级别的小模型上完成一次完整 trace，产出一份可复现的执行链报告。
3. 对 block size、chunked prefill、prefix caching 完成第一轮控制变量实验，手里有一份带数据的结果表。
4. 完成一次技术分享：《vLLM Attention 输入如何生成》(可发在个人仓库/博客)。
5. 画出一张"能讲清楚"的 vLLM → Attention Kernel 调用链图（精确到类与方法名）。

---

## 每周时间安排

每周 3 小时，固定节奏，融入现有作息：

| 时段 | 时长 | 内容 |
|---|---|---|
| 工作日晚上 A（建议周二） | 60–75 min | vLLM 源码精读：按当月主题读指定文件，做批注笔记 |
| 工作日晚上 B（建议周四） | 60–75 min | 源码精读续 / trace 实操：跑环境、打断点、记录数据 |
| 周末主项目时段内 | 30–60 min | 不单独占用大块时间，把本方向收获"嫁接"进主项目：例如主项目涉及分页 KV Cache 时，对照 vLLM 的 block table 设计做交叉笔记 |

规则：

- 两晚源码阅读是"雷打不动"的最低承诺；周末部分允许被主项目高峰挤占，但要在复盘时记一笔欠账。
- 每两周输出一篇短笔记（500 字以上即可），沉淀到 `2026Q4学习计划/notes/` 目录，避免"读了就忘"。
- 每月最后一个周末做月度复盘：对照月度路线图检查完成度，调整下月任务。

---

## 月度路线图

### 10 月（前 30 天）：环境与主干 trace

目标：环境可跑、主干调用链走通第一遍。

- [ ] **W1–W2：搭建可运行环境**
  - 安装 vLLM（建议 pip 安装官方 release，同时 clone 源码仓库用于阅读，版本对齐，如 `git checkout v0.11.x`）。
  - 下载 Qwen2.5-0.5B-Instruct，单卡（或降级方案，见实验章节）完成一次离线推理冒烟测试。
- [ ] **W2–W3：通读 V1 架构骨架**
  - 读 vLLM V1 官方博客与 V1 guide，建立全局图。
  - 对照源码入口：`vllm/v1/engine/`（`core.py` 的 `EngineCore`、`async_llm.py` 的 `AsyncLLM`）、`vllm/v1/engine/core_client.py`。
  - 重点理解：为什么单卡也是 2 进程（driver + worker）、EngineCore busy loop 长什么样、一步 `step()` 里发生了什么（schedule → execute_model → update_from_output）。
- [ ] **W3–W4：第一次 decode trace**
  - 对一次 decode 请求打断点/加日志，跟踪：`Scheduler.schedule()`（`vllm/v1/core/sched/scheduler.py`）→ KV 分配（`vllm/v1/core/kv_cache_manager.py`、`block_pool.py`）→ `GPUModelRunner.execute_model()`（`vllm/v1/worker/gpu_model_runner.py`）→ attention metadata 构建（`vllm/v1/attention/backends/utils.py` 的 `CommonAttentionMetadata`，`flash_attn.py` 的 `FlashAttentionMetadataBuilder`）→ kernel 调用。
  - 产出：trace 记录文档初稿（每经过一层记下关键数据结构的字段值）。

### 11 月（中 30 天）：机制深挖 + 控制变量实验

目标：把 10 月的"走一遍"升级为"讲得清机制"，并完成实验矩阵。

- [ ] **W5–W6：KV 管理机制精读**
  - block table 的语义：逻辑 block → 物理 block 的映射，`BlockTable`（`vllm/v1/worker/gpu/block_table.py`）。
  - slot mapping：`slot = block_id * block_size + offset` 的推导，它在哪里生成、谁消费（`reshape_and_cache` 写 KV cache）。
  - prefix caching：block hash、命中查找（`find_longest_cache_hit`）、引用计数与驱逐。
  - 读 PagedAttention 论文（见资料清单），把论文设计与 V1 代码对应起来。
- [ ] **W7–W8：调度策略精读**
  - continuous batching：waiting / running 队列流转、抢占（preemption）策略。
  - chunked prefill：`max_num_batched_tokens` 如何约束一步内 prefill 与 decode 的混排。
  - 选读 Sarathi 论文，理解 chunked prefill 的动机。
- [ ] **W8–W9：控制变量实验**（见"实验设计"章节的实验矩阵）
  - 实验一：block size ∈ {16, 32, 64, 128}。
  - 实验二：`max_num_batched_tokens` ∈ {512, 2048, 8192}（chunked prefill 粒度）。
  - 实验三：prefix caching 开 / 关（共享前缀的多请求场景）。
  - 产出：实验数据表 + 每实验 3 条以内的结论。
- [ ] **本月产出**：调用链图 v1（可用手绘 + drawio/mermaid 均可）。

### 12 月（后 30 天）：整合输出与验收

目标：把分散的理解整合成可交付、可传播的资产，完成验收。

- [ ] **W10–W11：调用链图与执行链报告定稿**
  - 调用链图要求：从 `AsyncLLM.generate()` 到 FlashAttention kernel，标出每个环节的**类名、方法名、关键张量**（block_table、slot_mapping、query_start_loc、seq_lens、max_seq_len），能被同行用一次 walkthrough 挑战不翻车。
  - 执行链报告：trace 记录 + 图 + 实验数据，整理成可复现文档（含环境版本、启动命令、脚本）。
- [ ] **W11–W12：技术分享**
  - 主题：《vLLM Attention 输入如何生成》。
  - 素材直接来自调用链图与执行链报告；20–30 分钟，留 Q&A。
  - 分享后收集的问题记入笔记，作为 2027 Q1 的输入。
- [ ] **W13：季度复盘**
  - 对照"能力验收标准"自测，脱稿回答验收问题。
  - 盘点欠账与超额项，产出 2027 Q1 计划初稿（预计进入 Attention 子系统与 Runtime 交叉主题：SWA / 压缩 KV 在 vLLM 中的支持方式、KV connector 等）。

---

## 学习资料清单

以下资料已在本计划制定时核实存在（2026 年初检索确认）；标注"需自行检索确认"的条目未找到稳定链接。

| 资料名称 | 类型 | 用途 | 优先级 |
|---|---|---|---|
| vLLM V1 官方博客《vLLM V1: A Major Upgrade to vLLM's Core Architecture》 https://blog.vllm.ai/2025/01/27/v1-alpha-release.html | 官方博客 | V1 架构总览：EngineCore 多进程、persistent batch、prefix caching 默认开启等设计动机 | P0（10 月 W2） |
| vLLM V1 Guide（docs/usage/v1_guide.md，仓库 docs 内） https://github.com/vllm-project/vllm/blob/main/docs/usage/v1_guide.md | 官方文档 | V1 与 V0 差异、功能支持矩阵 | P0 |
| vLLM V1 架构设计 RFC（GitHub Issue #8779） https://github.com/vllm-project/vllm/issues/8779 | 设计文档（Issue） | driver + SPMD worker、async single-step scheduling 的原始设计讨论 | P1 |
| vLLM 源码 `vllm/v1/` 目录 https://github.com/vllm-project/vllm | 代码仓 | 主教材。重点子目录见下方"源码阅读地图" | P0（贯穿全季） |
| PagedAttention 论文《Efficient Memory Management for Large Language Model Serving with PagedAttention》(SOSP'23) https://arxiv.org/abs/2309.06180 | 论文 | block / block table / KV 共享的理论源头 | P0（11 月） |
| 《Efficiently Scaling Transformer Inference》(Pope et al., MLSys 2023) https://arxiv.org/abs/2211.05102 | 论文 | prefill/decode 的计算-带宽特性、latency vs throughput 心智模型 | P1（11 月选读） |
| Sarathi《Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills》 https://arxiv.org/abs/2306.03078 | 论文 | chunked prefill 动机：decode 与 prefill 混批 | P1（11 月选读） |
| 《Life of an inference request (vLLM V1)》Ubicloud 技术博客 https://www.ubicloud.com/blog/life-of-an-inference-request-vllm-v1 | 博客 | 一次请求穿越 AsyncLLM → EngineCore → Scheduler → Executor 的图文 walkthrough | P1（10 月辅助） |
| vLLM 官方文档 Optimization and Tuning https://docs.vllm.ai/en/latest/configuration/optimization.html | 官方文档 | chunked prefill、preemption 等配置项的权威解释 | P1（实验前查参数） |
| FlashAttention 论文及 flash-attn 仓库 | 论文 / 代码仓 | 理解 vLLM 默认 attention backend 的 kernel 语义（与本职衔接） | P2（有余力时） |
| 中文社区 vLLM 源码解析系列（如 CSDN / 知乎上的 V1 深度分析系列） | 博客 | 中文速查，辅助理解，**不作权威依据** | P2 |
| SGLang / TensorRT-LLM 调度器对比资料 | 代码仓 | 横向对照，暂不在本季度范围 | P3（下季度） |

### 源码阅读地图（trace 主干，版本以阅读时 checkout 的 tag 为准，路径可能随版本微调）

```
vllm/v1/
├── engine/
│   ├── async_llm.py          # AsyncLLM：前端异步入口，tokenize/detokenize
│   ├── core.py               # EngineCore：busy loop，step() = schedule + execute + update
│   ├── core_client.py        # ZMQ IPC 客户端
│   ├── input_processor.py    # 请求 → EngineCoreRequest
│   └── output_processor.py   # 输出 → RequestOutput
├── core/
│   ├── sched/scheduler.py    # Scheduler.schedule()：队列管理、token 预算、抢占
│   ├── kv_cache_manager.py   # KVCacheManager：allocate_slots / get_computed_blocks / free
│   ├── block_pool.py         # BlockPool：物理 block 的分配、hash 缓存、驱逐
│   └── sched/output.py       # SchedulerOutput：发给 worker 的调度结果
├── worker/
│   ├── gpu_model_runner.py   # GPUModelRunner：_prepare_inputs、构建 attn metadata、执行 forward
│   ├── gpu_input_batch.py    # InputBatch：persistent batch 的增量更新
│   └── gpu/block_table.py    # BlockTable：block_table 张量与 slot_mapping 计算
├── attention/backends/
│   ├── utils.py              # CommonAttentionMetadata、AttentionMetadataBuilder 基类
│   └── flash_attn.py         # FlashAttentionMetadataBuilder：metadata → FA kernel 入参
└── kv_cache_interface.py     # KVCacheSpec / FullAttentionSpec / SlidingWindowSpec 等
```

---

## 实验设计与环境搭建

### 环境准备

**首选方案（单卡 GPU）**：

- 硬件：1 张消费级 NVIDIA GPU（≥ 12 GB 显存即可，如 RTX 3060 12G / 4070 Ti；Ampere 及以上架构最省事）。
- 软件：Python 3.10+，`pip install vllm`（CUDA 版本按官网匹配），源码 `git clone https://github.com/vllm-project/vllm` 并 checkout 到与安装一致的 tag。
- 模型：`Qwen/Qwen2.5-0.5B-Instruct`（约 1 GB，fp16/bf16 直接跑；显存紧张可用 GPTQ/AWQ 量化版）。

**降级方案（无 GPU 时）**：

- 方案 A：vLLM CPU backend（`VLLM_TARGET_DEVICE=cpu` 从源码编译或使用官方 CPU 镜像）。可完整走通 scheduler / KV manager / model runner 链路，但 attention kernel 为 CPU 实现，吞吐数字没有参考意义——trace 与机制学习不受影响。
- 方案 B：云端按量实例（单卡 T4/L4/A10 小时级租用），仅在做实验矩阵的 11 月 W8–W9 租用 1–2 天，成本可控。
- 无论哪条路，**trace 任务（10 月硬性要求）不降级**，实验任务可在 11 月集中突击。

### Bench 工具

- 首选 vLLM 自带 bench：`vllm bench serve`（在线服务压测，输出 TTFT/TPOT/throughput）与 `vllm bench latency` / `vllm bench throughput`（离线）。老路径 `benchmarks/benchmark_serving.py` 等脚本仍可用，以安装版本实际提供的 CLI 为准。
- 自写脚本场景：需要固定 prompt 集合、精确控制共享前缀比例时，用离线 `LLM` 类 + `time.perf_counter()` 打点。

### 实验记录字段（每个 run 一行，CSV）

`实验编号 | vllm版本 | 模型 | dtype | GPU | block_size | max_num_batched_tokens | enable_prefix_caching | 输入长度 | 输出长度 | context length | 并发数 | TTFT均值/p50/p99 | TPOT均值 | throughput(tok/s) | 峰值显存(GB) | 备注`

### 实验矩阵（11 月执行）

**实验一：block size 对性能的影响**

1. 固定：Qwen2.5-0.5B，fp16，输入 512 / 输出 128，并发 16，prefix caching 关闭。
2. 变量：`--block-size` ∈ {16, 32, 64, 128}（注意：V1 中部分 backend 对 block size 有约束，遇报错记录现象并回退到支持值）。
3. 预期观察点：block 越大，块内碎片越多（显存占用↑、可并发请求数↓），但 block table 寻址开销↓、kernel 内 gather 效率可能↑；小模型短序列下差异可能不明显——"差异不明显"本身也是有效结论，说明瓶颈不在 KV 管理。

**实验二：chunked prefill（max_num_batched_tokens）**

1. 固定：输入长度混合（64 / 512 / 2048 混合负载），输出 128，并发 32。
2. 变量：`--max-num-batched-tokens` ∈ {512, 2048, 8192}。
3. 预期观察点：预算小 → 长 prompt 被切得更碎、TTFT 中 prefill 被 decode 让路、TPOT 更平稳；预算大 → TTFT 尾部变差。画出 TTFT/TPOT 随预算变化的曲线，与 Sarathi 论文结论对照。

**实验三：prefix caching**

1. 构造：N 个请求共享 1024-token 前缀 + 各自 128-token 后缀，并发 16。
2. 变量：`--enable-prefix-caching` / `--no-enable-prefix-caching`（V1 默认开启，显式对照）。
3. 预期观察点：开启后共享前缀只算一次 prefill，TTFT 显著下降；结合 11 月源码阅读，解释命中发生在 `KVCacheManager.get_computed_blocks()` 一层。

**实验纪律**：每个实验只改一个变量；每个配置至少跑 2 次取稳定值；异常数据不删，标注原因。

---

## 能力验收标准

### 自测问题清单（脱稿回答通过）

**核心验收问题（职业规划硬性要求）**：

1. 一个 vLLM decode request 如何生成 Attention metadata 并最终调用到 Kernel？
   - 要求能讲出完整链条：`Scheduler.schedule()` 决定本步哪些请求各跑几个 token → `KVCacheManager.allocate_slots()` 分配新 block → `SchedulerOutput` 携带 block table diff 到 worker → `GPUModelRunner` 由 `InputBatch` + `BlockTable` 计算 `slot_mapping`、`query_start_loc`、`seq_lens` → `FlashAttentionMetadataBuilder.build()` 组装 `FlashAttentionMetadata` → `set_forward_context` 注入 → Attention 层调用 backend `forward()` → FA kernel。

**机制理解问题**：

2. block table 和 slot mapping 各自解决什么问题？`slot = block_id * block_size + offset` 中每个量从哪来？
3. continuous batching 相比 static batching，吞吐提升的来源是什么？代价是什么？
4. chunked prefill 为什么能降低 TPOT 抖动？`max_num_batched_tokens` 过小会有什么问题？
5. prefix caching 的 block 命中判据是什么？为什么必须以 block 为粒度而不是 token 为粒度？
6. 一个请求被 preempt 时，它的 KV block 去哪了？（recompute vs swap 策略，V1 当前采用哪种？）
7. 为什么 V1 说"单卡也有 2 个进程"？这样设计换来什么？
8. KV cache 的显存预算在引擎启动时如何确定？（profile → determine_available_memory → num_gpu_blocks）
9.（衔接本职）vLLM 的分页 KV Cache 语义与 SIMT GPU + NPU 异构加速器环境中的分块 Attention/分页 KV 实现有哪些同与不同？至少列出 3 点。
10.（衔接本职）如果要把一种新 attention 变体（如 SWA / 压缩 KV）接入 vLLM V1，需要动哪些层？（KVCacheSpec → KVCacheManager 的 manager 类型 → attention backend metadata builder，对照 `SlidingWindowSpec` 的既有实现回答）

### 交付物清单

| 交付物 | 存放位置 | 截止时间 |
|---|---|---|
| vLLM 执行链报告（可复现：环境、命令、trace 记录、实验数据） | `2026Q4学习计划/deliverables/vllm_trace_report.md` | 12-15 |
| vLLM → Attention Kernel 调用链图（drawio / mermaid 源文件 + 导出图） | `2026Q4学习计划/deliverables/attention_call_chain.*` | 12-15 |
| 实验数据表（CSV + 一页结论） | `2026Q4学习计划/deliverables/bench_results_q4.csv` | 11-30 |
| 技术分享《vLLM Attention 输入如何生成》（slides + 讲稿要点 + 现场 Q&A 记录） | `2026Q4学习计划/deliverables/talk_vllm_attention_input.*` | 12-31 |
| 双周短笔记（≥ 6 篇） | `2026Q4学习计划/notes/` | 滚动 |

---

## 风险与降级方案

| 风险 | 触发条件 | 应对 |
|---|---|---|
| 工作任务挤压（本方向是业余方向，天然让位） | 连续两周未完成两晚源码阅读 | **保底动作**：只保留"每周至少 1 晚 60 分钟 + 双周笔记"；实验矩阵整体后移到 12 月，砍掉实验一中 block size 的 2 个取值（只测 16 / 128 两极） |
| 无 GPU 可用 | 10 月 W2 环境搭建受阻 | 立即切降级方案：CPU backend 完成 trace（10 月不受影响），11 月实验改租云端按量实例 |
| vLLM 版本变动导致资料过时 | 阅读时发现目录结构/类名与笔记不符 | 以 checkout 的固定 tag 为准做 trace；报告中显式标注版本号，不追 main 分支 |
| 陷入细节沼泽（如 CUDA kernel 实现、spec decode 等支线） | 单晚阅读超时 90 分钟仍未收尾 | 本季度只走"主干调用链"；支线一律记入"下季度候选"清单，不展开 |
| 分享延期 | 12 月 W12 素材未就绪 | 分享最晚顺延到 1 月第 1 周；调用链图与报告不顺延（它们是分享的输入，不是分享的产物） |
| 整体时间严重不足（极端情况） | 主项目进入冲刺 | 按职业规划纪律：**至少保留主项目与复盘**；本方向降级为"只读 10 月 W2–W3 的 V1 架构材料 + 保留双周笔记"，trace 与实验顺延至 2027 Q1，季度复盘如实记录欠账 |

---

## 附：资料核实状态说明

本计划制定时（2026 年初）已通过检索核实以下外部资料真实存在且链接有效：

- vLLM V1 官方博客（blog.vllm.ai 的 V1 发布文，2025-01-27）；
- vLLM V1 Guide（官方仓库 docs/usage/v1_guide.md，文中注明 V0 已废弃）；
- V1 架构设计 RFC（GitHub Issue #8779）；
- PagedAttention 论文（arXiv:2309.06180，SOSP'23）；
- Efficiently Scaling Transformer Inference（arXiv:2211.05102，MLSys 2023）；
- Sarathi / chunked prefill 论文（arXiv:2306.03078）；
- Ubicloud《Life of an inference request (vLLM V1)》博客；
- 源码阅读地图中的关键路径（`vllm/v1/engine/core.py`、`vllm/v1/core/sched/scheduler.py`、`vllm/v1/core/kv_cache_manager.py`、`vllm/v1/worker/gpu_model_runner.py`、`vllm/v1/attention/backends/utils.py`、`flash_attn.py` 等）已与仓库实际结构比对确认。

注意：vLLM 迭代极快，目录结构与类名可能随版本变化；所有 trace 与实验均以实际 checkout 的 release tag 为准，报告中标明版本号。未列出的资料（如中文社区解析文章）请自行检索确认时效性。

---

*本计划为季度滚动文档：每月复盘后直接在本文件相应章节勾选 / 修订，修订记录写在月度复盘笔记中。*
