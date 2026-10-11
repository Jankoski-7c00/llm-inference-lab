# 04 · Attention 变体与新模型架构跟踪 —— 2026 Q4 学习计划

> 本文档为个人学习规划,基于公开技术生态(vLLM / Triton / PyTorch / 开源模型),不包含任何雇主相关信息。

> 季度区间:2026-10-01 至 2026-12-31
> 时间预算:季度总投入的 15%,约 **每周 1.5 小时**(机动 60–90 分钟)
> 主线:**模型结构变化 → Runtime / Compiler / Kernel 三层改动**
> 产出形式:模型结构卡片 + 模型适配清单 + Attention workload 图谱

---

## 1. 方向定位与季度目标

### 1.1 为什么是"跟踪"而不是"精读"

本方向不是泛读论文,而是为工程实践主线(FlashAttention 类算子、分块 Attention(blockwise attention)的编译重构、动态形状、DeepSeek 模型结构分析、SWA、压缩 KV、分页 KV Cache)提供**前瞻雷达**:

- 论文精读是输入导向的(论文写什么就读什么);跟踪是**输出导向**的——只关心一件事:这个结构变化落到 Runtime、Compiler、Kernel 三层分别要改什么。
- 每周只有 60–90 分钟,客观上不可能精读;强迫自己走"结构图 → KV 语义 → 与现有实现 diff"的最小流程,反而能训练出职业目标要求的核心能力:**快速读懂新模型的 Attention、KV、RoPE、MoE 和量化变化**。
- 判断标准不是"读了几篇论文",而是"新模型出现时,我能否在半天内给出三层改动预判"。

### 1.2 与工程实践的互相强化关系

| 工程实践任务 | 本方向的反哺 |
| --- | --- |
| 分块 Attention 的编译重构 | 理解 MLA/SWA 等新结构对 Legalize 规则的新约束(哪些算子组合必须保留、哪些可下沉到公共骨架) |
| DeepSeek 模型结构分析(基于公开模型) | 直接产出适配清单,把"跟踪结果"变成工程实践的输入文档 |
| SWA / 压缩 KV / 分页 KV Cache | 数据流梳理直接服务于 Kernel 边界划分与 KV block 管理策略 |
| Attention workload 图谱 | 分类矩阵(prefill/decode/mixed batch、GQA、KV dtype、context length、backend)复用实践中的真实 case |

反向也成立:实践中遇到的每一个真实 case(动态形状、分块 Attention 与 FlashAttention 的差异、Legalize 边界)都是检验"跟踪理解是否正确"的试金石。两边不脱节,是本方向不沦为碎片化的根本保证。

### 1.3 12 月底状态(Definition of Done)

- 至少 **2 张模型结构卡片**(DeepSeek-V2/V3 MLA 一张,DeepSeek-V3.2 DSA 或某 SWA 模型一张)。
- **1 份模型适配清单**:DeepSeek 的 SWA、MLA、压缩 KV、特殊索引(indexer)与实际实现的差异逐项列出,并预判 Runtime / Compiler / Kernel 三层改动。
- **1 张 Attention workload 图谱**(成稿):测试 case 能映射到具体 Kernel 路径。
- 能回答第 6 章的全部自测问题,包括终极验收问题:**相同 Attention workload 在 SIMD NPU 和 SIMT GPU 上的并行与访存差异**。

---

## 2. 每周时间安排

### 2.1 机动 60–90 分钟的使用规则

这是**机动时间**,不是固定学习时段,服务于主线而不是另开战线:

1. **只追与当前主线直接相关的内容**。判断标准:这个结构变化是否会影响我正在做的 FlashAttention 类算子 / 分块 Attention、Legalize、KV 管理?不会 → 记入"待看清单",本周不看。
2. **触发式使用,而非填满式使用**。典型触发场景:
   - 工作中遇到 DeepSeek 模型相关的新需求 → 本周机动时间用来读对应模型卡/config;
   - 某篇新论文/新模型发布且涉及 Attention/KV 结构变化 → 本周用来走一遍最小读论文流程;
   - 无事触发 → 用来维护 workload 图谱或补结构卡片,**不许顺手泛读**。
3. **单次不超过 90 分钟**。到点就停,未读完的内容写成 3 行备忘(读到哪里、卡在哪、下次从哪继续),下周继续。防止"一篇论文读三周"。
4. **每周结束留 5 分钟记一行日志**:本周跟踪了什么 → 对应三层中的哪一层 → 是否产生了卡片/清单/图谱的增量。没有增量是允许的,但要能说出为什么(与主线无关)。

### 2.2 防碎片化纪律

- 同一时刻最多跟踪 **1 个模型/主题**;新开一个必须先关闭(成卡片)或显式挂起(写明挂起原因)当前这个。
- 社交媒体/技术文章的" Attention 新进展"一律先进入待看清单,只有被实践需求命中才捞出来。
- 每两周回顾一次待看清单,超过一个月未被命中的条目直接删除——未被主线命中的跟踪没有价值。

---

## 3. 月度路线图

### 3.1 10 月:DeepSeek MLA / 压缩 KV 结构卡片

- **目标**:完成第一张模型结构卡片(DeepSeek-V2/V3 的 MLA)。
- 从 DeepSeek-V2 技术报告与 DeepSeek-V3 Technical Report 的 Architecture 章节入手,提取:latent vector `c_t^KV` 与 decoupled RoPE key `k_t^R` 的缓存语义、`kv_lora_rank` / `qk_rope_head_dim` / `qk_nope_head_dim` / `v_head_dim` 的真实数值(对照 HuggingFace config)。
- 与工程实践中的压缩 KV 实现逐项 diff:缓存布局、decode 时是否需要 up-projection、RoPE 部分是否单独缓存。
- 对照 vLLM 的 `deepseek_v2.py` 与 DeepSeek 官方 FlashMLA,看 MLA 在成熟推理栈中的 Kernel 切分方式(absorbed 与非 absorbed 两种执行方案的取舍)。
- **月底检查**:能脱口而出 MLA 相对 MHA / GQA 的 KV cache 大小倍数及推导过程。

### 3.2 11 月:SWA 与 paged KV 数据流梳理

- **目标**:梳理 Sliding Window Attention 与分页 KV Cache 的数据流,落到 KV block 管理语义。
- 核心问题链:SWA 的窗口语义 → 哪些 KV block 可以驱逐/复用 → paged KV Cache 的 block table 如何表达窗口 → prefill 与 decode 路径的差异 → mixed batch 下不同请求窗口不一致时 block 管理怎么做。
- 对照实际工程中的分页 KV Cache 实现,画出"逻辑窗口 → 物理 block"的映射图。
- 若 11 月实践主线命中 DeepSeek-V3.2 的 DSA( lightning indexer + top-k token selection),优先把 DSA 当作"动态窗口"的极端情形纳入本主题——稀疏索引对 KV block 管理的约束比固定窗口更强。
- **月底检查**:能回答"SWA 对 KV block 管理意味着什么",并给出至少一个实际实现中的具体改动点。

### 3.3 12 月:模型适配清单 + Attention workload 图谱成稿

- **目标 1:模型适配清单成稿**。以前两个月的卡片为输入,按第 5 章模板整理:每一项结构差异 → Runtime 改动 / Compiler(Legalize)改动 / Kernel 改动,标注"已支持 / 需小改 / 需新增"。
- **目标 2:Attention workload 图谱成稿**。按 prefill/decode/mixed batch × GQA 组数 × KV dtype × context length × backend 的分类矩阵整理,把实际测试 case 逐个映射到具体 Kernel 路径,找出覆盖空洞。
- **目标 3:季度复盘**。用第 6 章自测问题做闭卷自测,答不出的问题列为 2027 Q1 的输入。
- **月底检查**:两个交付物可以直接作为技术文档分享(可发在个人仓库/博客);验收问题能写出结构化答案。

---

## 4. 学习资料清单

> 优先级:P0 = 季度内必读(与主线直接命中);P1 = 命中时读;P2 = 仅入待看清单。
> 已标注链接的资料均经检索核实;标注"需自行检索确认"的为方向确定、具体 URL 未当场核实的条目。

| 名称 | 类型 | 用途 | 优先级 |
| --- | --- | --- | --- |
| DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model([arXiv:2405.04434](https://arxiv.org/abs/2405.04434)) | 论文 | MLA 的首次完整定义:latent 压缩 + decoupled RoPE;10 月卡片的主输入 | P0 |
| DeepSeek-V3 Technical Report([arXiv:2412.19437](https://arxiv.org/abs/2412.19437)) | 官方技术报告 | V3 的 MLA 超参(`d_c=512`, `d_h^R=64`, `n_h=128`)、FP8 训练、MTP;结构卡片的关键数值来源 | P0 |
| DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models([arXiv:2512.02556](https://arxiv.org/abs/2512.02556)) | 官方技术报告 | DSA(lightning indexer + fine-grained token selection)——稀疏索引/特殊索引的最新形态,适配清单的重点 | P1 |
| DeepSeek-V3.2-Exp 模型卡([HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp)) | 模型卡 | DSA 的工程化说明、config 字段对照 | P1 |
| GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints([arXiv:2305.13245](https://arxiv.org/abs/2305.13245)) | 论文 | GQA/MQA 的原始定义;workload 图谱中 GQA 维度的理论基线 | P0 |
| FlashAttention(NeurIPS 2022,arXiv:2205.14135) | 论文 | IO-aware tiling 与 online softmax 的源头;理解 FlashAttention 与分块 Attention 差异的理论锚点 | P0 |
| FlashAttention-2([arXiv:2307.08691](https://arxiv.org/abs/2307.08691)) | 论文 | 并行度与 warp 间工作划分(split-Q vs split-K);SIMD/SIMT 差异讨论的 GPU 侧参照 | P1 |
| FlashAttention-3(arXiv:2407.08608) | 论文 | 异步与低精度(FP8)下的 attention;KV dtype 维度的参照 | P2 |
| MQA: Fast Transformer Decoding: One Write-Head is All You Need([arXiv:1911.02150](https://arxiv.org/abs/1911.02150)) | 论文 | MQA 源头,与 GQA 一起构成 KV head 压缩的谱系端点 | P2 |
| Native Sparse Attention(NSA,arXiv:2502.11089) | 论文 | 硬件对齐的可训练稀疏 attention;DSA 的思想前驱,理解"稀疏索引"设计空间 | P1 |
| vLLM 模型实现:`vllm/model_executor/models/deepseek_v2.py`([GitHub](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_v2.py)) | 代码仓 | MLA 在成熟推理栈中的落地:`qk_nope_head_dim`/`qk_rope_head_dim` 拼接、head 补齐到 256、Attention 层接口;与实际实现 diff 的直接对象 | P0 |
| vLLM `model_executor/models/` 下其余 deepseek 相关文件(deepseek_mtp.py、deepseek_vl2.py 等,目录随版本演进,需自行检索确认当期文件名) | 代码仓 | MTP、多模态变体的结构增量 | P2 |
| FlashMLA([github.com/deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)) | 代码仓 | DeepSeek 官方 MLA decode kernel;理解 absorbed 执行方案与 paged KV 的 kernel 切分 | P1 |
| HuggingFace DeepSeek-V3 config([config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)) | 模型卡/config | `kv_lora_rank=512`、`q_lora_rank=1536`、`qk_rope_head_dim=64` 等字段与论文符号的对照表 | P0 |
| HuggingFace Transformers `modeling_deepseek_v3.py`([GitHub](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py)) | 代码仓 | 参考实现的"教科书版本",适合快速建立结构图 | P1 |
| vLLM Blog:DeepSeek-V3.2-Exp in vLLM(Fine-Grained Sparse Attention in Action,2025-09,具体 URL 需自行检索确认) | 工程博客 | DSA 在推理栈中的适配实录,适配清单的直接参照 | P1 |
| Hardware-Centric Analysis of DeepSeek's Multi-Head Latent Attention(arXiv:2506.02523) | 论文 | MLA 的硬件向分析(compute-bound 转移、两种执行方案的访存/算力权衡);验收问题的直接素材 | P1 |
| SWA 代表模型资料(Mistral / Gemma 2/3 的 sliding window 实现,具体模型卡需自行检索确认) | 模型卡/代码仓 | 11 月 SWA 数据流梳理的对照样本 | P1 |

**选读纪律**:P0 共 6 项,对应每周约 1.5 小时 × 13 周 ≈ 20 小时的预算,平均每项 3 小时以内——这本身就强制了"最小流程"读法(见 5.4),读不完说明流程执行走样,而不是资料太多。

---

## 5. 实验与产出方法

### 5.1 模型结构卡片模板

每张卡片一页纸以内,字段固定:

```
# 模型结构卡片:<模型名 / 结构名>

## Attention 类型
- MHA / GQA(n_kv=?) / MQA / MLA / SWA(window=?) / 稀疏(indexer 类型)
## KV 语义
- 缓存什么(per token 的字节数推导)、KV dtype、是否分页、是否可驱逐
## RoPE
- 维度、是否 decoupled、scaling 方式(YaRN 等)、对 position 的依赖
## MoE
- 专家数 / 激活数 / 共享专家;对 attention 层的间接影响(无则写"无")
## 量化
- 权重/激活/KV cache 各自的精度方案
## 对 Runtime 的要求
- KV block 管理、调度、batch 组织、cache 分配策略的改动
## 对 Compiler 的要求
- Legalize 规则、算子拆分/融合、动态形状约束、新算子引入
## 对 Kernel 的要求
- 新 kernel / 现有 kernel 的 shape·layout·mask 变化、FlashAttention 类 / 分块 Attention 实现各自是否命中
## 与当前实现的 diff(一句话级)
- 已支持 / 需小改 / 需新增,各列条目
```

### 5.2 适配清单模板

```
| # | 结构差异 | Runtime 改动 | Compiler 改动 | Kernel 改动 | 状态(已支持/需小改/需新增) | 依据(卡片/源码位置) |
```

填写规则:每一行必须能指回一张结构卡片或一处源码位置;三层中无改动的层写"无",不许留空——"确认无改动"本身就是适配结论。

### 5.3 Workload 图谱的分类矩阵示例

主轴:**执行阶段 × KV 结构 × 数值 × 规模 × 后端**,每个 case 映射到具体 Kernel 路径:

| 维度 | 取值示例 |
| --- | --- |
| 执行阶段 | prefill / decode / mixed batch |
| Attention 结构 | MHA / GQA(g=1,2,4,8…) / MLA / SWA(window 大小) |
| KV dtype | fp16 / bf16 / fp8 / int8 |
| context length | 短(<4K)/ 中(4K–32K)/ 长(>32K) |
| backend | SIMD NPU / SIMT GPU |

映射示例(格式,数值为占位):

```
case: decode, GQA g=8, KV fp8, ctx 32K, NPU
  → Runtime: paged KV block table, mixed batch 调度
  → Kernel 路径: blockwise-attn-decode-fp8-g8(分块策略 X)
  → 覆盖状态: 已有测试 case #123 / 空洞
```

图谱成稿的验收:任意一个实际测试 case,能在 30 秒内定位到矩阵坐标与 Kernel 路径;反之,矩阵中每个被声明"已支持"的坐标,能指出至少一个测试 case。

### 5.4 读论文的最小流程(≤ 90 分钟)

1. **结构图(20 分钟)**:只画 attention 相关的张量流——Q/K/V 从哪来、维度多少、缓存什么。画不出来 = 没读懂,回到原文。
2. **KV 语义(20 分钟)**:回答三个问题:每 token 缓存多少字节?decode 时读哪些?RoPE 信息存在哪?
3. **与现有实现 diff(30 分钟)**:对照当前 FlashAttention 类 / 分块 Attention 实现与 vLLM/HF 参考实现,按三层列出改动预判。
4. **落盘(20 分钟)**:增量写入结构卡片或适配清单;写不出增量说明本次跟踪与主线无关,记入日志并停止。

---

## 6. 能力验收标准

### 6.1 自测问题清单(12 月底闭卷自测)

**KV cache 定量:**

1. MLA 与普通 MHA 的 KV cache 大小差多少倍、为什么?(要求能从 `d_c`、`d_h^R`、`n_h`、`d_h` 推导出倍数,而不是背结论)
2. 同等条件下 GQA(g=8)与 MLA 的 KV cache 各是多少?谁更小、代价是什么?
3. KV dtype 从 bf16 降到 fp8,对 Runtime 的 block 管理和 Kernel 的累加精度各意味着什么?

**结构语义:**

4. SWA 对 KV block 管理意味着什么?(窗口外 block 的驱逐/复用、prefill 与 decode 的差异、mixed batch 下不同请求窗口不一致怎么办)
5. DeepSeek-V3.2 的 DSA(lightning indexer + top-k selection)相比固定窗口 SWA,对 KV 管理多出了哪些约束?(索引结果动态 → block 不可预先驱逐;indexer 自身的 KV 需要第二套缓存)
6. MLA 的 decoupled RoPE key 为什么不能并进 latent vector 一起压缩?

**适配方法论:**

7. 一个新模型的适配要先看哪三个地方?(参考答案方向:HF config 的 attention/KV 字段 → 参考实现的 attention forward → 推理栈(vLLM)中的模型文件与 kernel 选择逻辑)
8. 给定一个新结构变化,如何区分它属于 Runtime 改动、Compiler 改动还是 Kernel 改动?各举一个本季度遇到的例子。

**终极验收(职业规划验收问题):**

9. 相同 Attention workload 在 SIMD NPU 和 SIMT GPU 上的并行与访存差异是什么?(要求覆盖:并行轴选择的差异、tiling/block 粒度的差异、online softmax 的 rescale 在两边的代价、KV 访存模式与带宽瓶颈的差异、GQA/MLA 下 KV 复用对两边收益的不对称)

### 6.2 交付物清单

| 交付物 | 数量 | 完成时间 | 验收方式 |
| --- | --- | --- | --- |
| 模型结构卡片 | ≥ 2 张(MLA 一张;DSA 或 SWA 模型一张) | 10 月底 1 张,12 月中旬 1 张 | 字段完整,数值可溯源到论文/config |
| 模型适配清单 | 1 份 | 12 月底 | 每项差异三层改动齐全、状态明确、有依据链接 |
| Attention workload 图谱 | 1 张(成稿) | 12 月底 | 测试 case ↔ Kernel 路径双向可映射 |
| 周日志 | 连续 13 周 | 每周 | 每周一行,格式:跟踪对象 → 层 → 增量 |

---

## 7. 风险与降级方案

### 7.1 主要风险:论文泛读失控

本方向最大的风险不是学不会,而是**读太多**——Attention 方向新论文极多,一旦从"跟踪"滑向"泛读",15% 的预算会迅速吃掉主方向时间。

### 7.2 时间盒策略(硬性规则)

1. **每周 90 分钟硬顶**:到点即停,未读完写 3 行备忘下周继续;连续两周读同一篇 → 第三次要么关闭成卡片,要么显式放弃。
2. **主线命中测试**:开始读任何资料前,先写一句话回答"这与当前 FlashAttention 类算子 / 分块 Attention / Legalize / KV 相关实践的哪一点相关"。写不出 → 不入本周,只入待看清单。
3. **待看清单月度清理**:超过一个月未被实践需求命中的条目直接删除。
4. **产出倒逼**:某周没有产生任何卡片/清单/图谱增量时,在日志里必须写明原因(无关 / 未到产出阶段 / 时间被工作挤占),连续三周无产出 → 触发降级。

### 7.3 降级方案

| 触发条件 | 降级动作 |
| --- | --- |
| 实践主线挤压,连续 2 周无法投入 | 当周机动时间缩到 30 分钟,只做"日志 + 待看清单维护",路线图顺延 |
| 10 月底 MLA 卡片未完成 | 砍掉 P1/P2 资料,只保留 DeepSeek-V2/V3 报告 + HF config + vLLM 源码四项,卡片字段允许"MoE/量化"两项标注"略" |
| 11 月 SWA 主题与实践主线脱钩 | 用实际工程中的分页 KV Cache 实际问题替换论文跟踪,图谱照常推进 |
| 12 月时间不足 | 保两张结构卡片 + workload 图谱;适配清单允许只覆盖 MLA 与 SWA 两条线,DSA 降级为待看项 |
| 整个季度严重挤占 | 最低交付:1 张 MLA 结构卡片 + 自测问题 1/4/7/9 的答案。低于此线则本方向当季记为失败,2027 Q1 重新规划 |
