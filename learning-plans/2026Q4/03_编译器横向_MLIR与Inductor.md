# 2026 Q4 学习计划 · 编译器横向:MLIR 与 TorchInductor

> 本文档为个人学习规划,基于公开技术生态(vLLM / Triton / PyTorch / 开源模型),不包含任何雇主相关信息。

> 周期:2026-10-01 至 2026-12-31(共 13 周)
> 时间预算:每周约 1.5 小时(占整体学习时间的 15%)
> 定位:理解设计,不追求短期成为 MLIR Backend 开发者
> 锚点:以实际工程中的分块 Attention(blockwise attention)编译重构为主线,所有概念都映射回工程实践场景

---

## 一、方向定位与季度目标

### 为什么横向学编译器

我的工程实践深度绑定一套 TVM 系生产编译栈:模型图、算子 Legalize、IR 变换、算子库、动态形状处理。这些经验是真实的,但表达方式高度依赖 TVM 的 API 与术语(Relay/TIR、Pass、Strategy、Legalize 注册机制)。风险在于:

- **经验被单一框架绑定**:换一个编译栈(MLIR 系、Inductor、XLA)时,能迁移的不是 API 记忆,而是对"编译器分层、IR 设计、合法性判定、Lowering 边界"的判断力。
- **设计讨论缺共同语言**:社区(以及业内 AI Systems 团队)讨论编译器架构时,通用词汇来自 MLIR(Dialect、SSA、Region、Pattern Rewrite)和 PyTorch 2.x(Dynamo、FX、Inductor、Triton)。横向学习是把实践中的隐性知识翻译成行业通用语言的过程。
- **为长期目标铺路**:职业目标是懂编译器与硬件的 LLM Inference / AI Systems Performance 工程师。未来 7–12 个月计划用同一算例系统比较 TVM 与 Inductor 的处理方式,本季度是为那次比较打概念基础。

### 本方向不做什么

- 不做 MLIR Backend 开发(不写生产级 Dialect、不提交上游 patch)。
- 不系统读 TVM 源码(TVM 是日常工具,不是本方向的学习对象;本方向学的是 TVM 的"对照物")。
- 不追求量化性能结论,只做结构性、设计层面的对比。

### 12 月底应达到的状态

1. 能用不依赖 TVM API 的语言描述一个 DL 编译器的分层结构:Graph Capture → Graph IR → Loop-level IR → Kernel Codegen → Runtime,并指出每一层在 MLIR 生态、Inductor 路径、TVM 系编译栈中的对应物。
2. 对 MLIR 核心概念(Dialect / Operation / Region / Block / SSA / Pattern Rewrite / Canonicalization / Dialect Conversion / Type Conversion)能用自己的话解释设计动机与分工。
3. 能画出 Dynamo → FX → Inductor → Triton 的完整数据流,标注 graph capture、fusion、dynamic shape、layout、codegen、autotuning 各自发生在哪一环。
4. 能回答本季度的验收问题:**分块 Attention 的 Legalize 在完整编译栈中的职责是什么?哪些逻辑应该留给 Runtime,哪些应该留给 Kernel?**
5. 交付一份《编译器分层架构笔记(不依赖 TVM API)》、一份 Inductor 路径笔记、一张 TVM vs Inductor 差异表初稿。

---

## 二、每周时间安排(1.5 小时如何切分)

原则:工作日只做"输入",周末只做"整理输出",避免碎片时间硬啃论文。

| 时段 | 时长 | 内容 | 说明 |
|---|---|---|---|
| 工作日某个晚上(固定,如周三) | 60–75 分钟 | 精读:MLIR 官方文档章节 / Inductor 资料 / 论文小节 | 一次只读一个概念,读到能复述为止;读不完下周继续,不赶进度 |
| 周末复盘时(并入周复盘,不额外占块) | 15–30 分钟 | 整理笔记:把本周读的概念写成 3–5 行,并回答"TVM 里的对应物是什么" | 笔记直接写进架构笔记草稿文件,季度内持续生长 |

锚点学习法(贯穿全季度的自问清单):

- 每学一个 MLIR 概念,问:**TVM 里的对应物是什么?**(例如 Dialect ≈ Relay/TIR 的 IR 层级?Dialect Conversion ≈ Legalize?)
- 每学一个 Inductor 环节,问:**分块 Attention 重构中这一环是谁在承担?**(fusion 是 codegen 前还是 schedule 时?dynamic shape 的 guard 在哪?)
- 如果答不上来,不查 TVM 源码(避免超时),先把问题记录在笔记末尾的"待验证问题"清单,12 月集中对照回答。

---

## 三、月度路线图

### 10 月:MLIR 核心概念精读

目标:建立"可移植的 IR 设计词汇表"。

| 周 | 主题 | 具体内容 | 锚点任务 |
|---|---|---|---|
| W1(10/01–10/04,假期周) | 导论 | MLIR CGO 2021 论文前半(Motivation、设计原则:Progressive Lowering、SSA + Regions) | 用 3 行话总结:为什么"一层 IR 不够",TVM 的 Relay→TIR→LLVM 分层是否符合同一哲学 |
| W2 | IR 结构 | LangRef:Dialect、Operation、Region、Block、SSA、Attribute/Type 系统 | 画一张 Region/Block 嵌套图,对照 TVM 里 SeqStmt / Block 的嵌套表达 |
| W3 | 变换机制 | Pattern Rewrite 文档:RewritePattern、Greedy Driver、Walk Driver;Canonicalization 的概念 | 分块 Attention 的 Legalize 若用 Pattern Rewrite 语言表达,match 条件和 rewrite 结果分别是什么 |
| W4 | 转换机制 | Dialect Conversion 文档:ConversionTarget(Legal/Illegal/Dynamic)、三种 Conversion 模式、Type Conversion(TypeConverter、Materialization) | 写下:Dialect Conversion 的 legality 与 TVM Legalize 的语义差异(这是本月最重要的一个对照点) |

10 月产出:MLIR 概念速查表(每个概念 ≤5 行 + TVM 对应物一栏),并入架构笔记第 1 章。

### 11 月:TorchInductor 路径追踪

目标:看懂 Dynamo → FX → Inductor → Triton 全链路,建立与 TVM 系路线的差异意识。

| 周 | 主题 | 具体内容 | 锚点任务 |
|---|---|---|---|
| W5 | Graph Capture | PyTorch 2.x 官方页面 + torch.compiler 文档:Dynamo 的 Frame Evaluation 机制、graph break、guard | 分块 Attention 所在模型的 capture 边界:哪些东西进不了 FX 图,实际工作中的编译栈里对应阶段叫什么 |
| W6 | 算子规范化 | PrimTorch / decomposition:2000+ 算子收敛到 ~250 prim 的思路 | 对照 TVM 的算子库 + Legalize:殊途同归的"收敛算子面"问题 |
| W7 | Loop-level IR 与 Fusion | Inductor 的 define-by-run IR(~50 个算子)、调度与 fusion、layout 决策 | 分块 Attention 重构中的 fusion 决策若交给 Inductor 会发生在哪一步 |
| W8 | Codegen 与 Autotuning | Triton codegen、autotuning(mode="max-autotune")、dynamic shape 的符号化与 guard | 对照 TVM 的 MetaSchedule / 手工 schedule:autotuning 的搜索空间设计差异 |

11 月产出:《Inductor 路径笔记》初稿(一张数据流图 + 每环节 3–5 行注解 + 每环节的 TVM 对照物)。

### 12 月:分块 Attention 重构对照 + 架构笔记成稿

目标:把前两个月的概念落到工程锚点上,完成全部交付物。

| 周 | 主题 | 具体内容 |
|---|---|---|
| W9 | 分块 Attention 分层对照(上) | 用 10 月的 MLIR 词汇重述某类分块 Attention 的 Legalize 边界:输入/输出/合法性判定;列出"待验证问题"清单中所有问题的答案 |
| W10 | 分块 Attention 分层对照(下) | 用 11 月的 Inductor 视角审视分块 Attention:capture、decomposition、fusion、codegen、runtime 各环节,分块 Attention 的职责被如何切分 |
| W11 | 验收问题攻关 | 完整回答:分块 Attention 的 Legalize 在完整编译栈中的职责;哪些逻辑应留给 Runtime(如运行时形状/掩码决策),哪些应留给 Kernel(如分块循环、数值细节);写成笔记最后一章 |
| W12(至 12/31) | 成稿与自测 | 三份交付物定稿;按第六章自测问题清单逐条自答,答不出的标记为 Q1 遗留项 |

12 月产出:全部交付物定稿 + 自测记录 + 下季度(Q1)的遗留问题清单。

---

## 四、学习资料清单

优先级:P0 = 必读(构成季度主线);P1 = 选读(主线卡住或需加深时);P2 = 备查(用时再查)。

| 名称 | 类型 | 用途 | 优先级 |
|---|---|---|---|
| MLIR Language Reference(mlir.llvm.org/docs/LangRef/) | 官方文档 | W2 主材料:Dialect/Operation/Region/Block/SSA 的权威定义 | P0 |
| MLIR 设计论文:"MLIR: Scaling Compiler Infrastructure for Domain Specific Computation"(CGO 2021,DOI: 10.1109/CGO51591.2021.9370308;arXiv 预印本 arXiv:2002.11054) | 论文 | W1 主材料:理解 Progressive Lowering 与整体设计动机 | P0 |
| Pattern Rewriting: Generic DAG-to-DAG Rewriting(mlir.llvm.org/docs/PatternRewriter/) | 官方文档 | W3 主材料:RewritePattern、Greedy/Walk Driver、PatternBenefit | P0 |
| Dialect Conversion(mlir.llvm.org/docs/DialectConversion/) | 官方文档 | W4 主材料:ConversionTarget、Partial/Full/Analysis 三种模式、Type Conversion 与 Materialization | P0 |
| PyTorch 2.x 官方介绍页(pytorch.org/get-started/pytorch-2-x/) | 官方博客/介绍 | W5 主材料:Dynamo / AOTAutograd / PrimTorch / Inductor 四组件总览 | P0 |
| torch.compiler 文档(docs.pytorch.org/docs/stable/torch.compiler.html) | 官方文档 | W5 辅助:torch.compile 行为、backend 列表(注意其中有 TVM backend,可作对照) | P0 |
| torch.compile 入门教程(docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) | 官方教程 | 第 5 章实验 2 的操作依据:跑通 capture、观察 FX 图 | P0 |
| PyTorch Logging 文档(docs.pytorch.org/docs/stable/logging.html) | 官方文档 | 实验 2 的 dump 手段:TORCH_LOGS 的 inductor/output_code/schedule 等组件与 artifact | P0 |
| PyTorch 2.0 Release Blog(pytorch.org/blog/pytorch-2-0-release/) | 官方博客 | W5 背景材料:2.0 技术全景与 SDPA 等推理相关特性 | P1 |
| MLIR Toy Tutorial(mlir.llvm.org/docs/Tutorials/Toy/Ch-2/ 等) | 官方教程 | 实验 1 的参考:Emitting Basic MLIR 与后续章节的 pattern rewrite 示例 | P1 |
| MLIR 相关论文列表(mlir.llvm.org/pubs/) | 索引页 | 找延伸阅读的入口 | P2 |
| Inductor 源码(pytorch/pytorch 仓库 torch/_inductor/) | 代码 | 笔记卡住时定点阅读(loop-level IR 定义、fusion 决策处);不系统读 | P2 |
| "MLIR: A Compiler Infrastructure for the End of Moore's Law"(arXiv:2002.11054,与 CGO 论文同源的 arXiv 预印本) | 论文预印本 | CGO 论文的开放获取版本;注意官方建议正式引用以 CGO 版本为准 | P1 |

说明:以上 URL 均已在 2026 年初核实可访问(见文末"外部资料核实记录")。若访问失效,以 mlir.llvm.org 与 docs.pytorch.org 站内导航为准重新定位。

---

## 五、实验与小练习(轻量,不占额外时间块)

本方向以阅读 + 笔记为主,实验只做两个,每个控制在 2 个"工作日晚"内完成,写不进时间预算就砍掉。

### 实验 1:观察一次 Pattern Rewrite(10 月 W3–W4,约 1 小时)

目的:把 Greedy Driver / Canonicalization 从纸面概念变成"看过的现象"。

步骤:

1. 准备一个最小 MLIR 文本(如 `arith` Dialect 中 `x + 0` 或常量折叠场景);环境允许则用 MLIR Python bindings,否则用 `mlir-opt` 命令行。
2. 执行 canonicalize pass,开启 `-debug-only=greedy-rewriter`(Pattern Rewriting 文档 Debugging 一节给出的调试方法)。
3. 观察输出树:op 被处理顺序、pattern match 成功/失败、Insert/Replace 动作。

观察点(记入笔记):

- worklist 的初始化和遍历顺序与 TVM 的 Pass 遍历有何异同;
- pattern 的 benefit 是否影响应用顺序;
- 这个结果对理解"分块 Attention 的 Legalize 本质是一次 pattern-driven 重写"有何帮助。

降级方案:若环境搭不起来(无 mlir-opt / bindings 装不上),直接读 Pattern Rewriting 文档中的示例调试输出(`cf.cond_br` 那个例子),把每一步口头复述一遍,同样计入完成。

### 实验 2:torch.compile 一个小模型并 dump Inductor 产物(11 月 W7–W8,约 1.5 小时)

目的:亲眼看到 FX 图 → Inductor schedule → Triton 代码的物化过程。

步骤:

1. 写一个 10–20 行的小模型(建议:线性层 + GELU 的 pointwise 链,或一个简化的 attention 片段),`torch.compile` 之。
2. 用 `TORCH_LOGS="graph_code,output_code,schedule"` 运行(组件与 artifact 名称以 PyTorch Logging 文档为准),收集:Dynamo 捕获的 FX 图、Inductor 的 schedule、生成的 Triton kernel。
3. 对照同一算例在一套 TVM 系生产编译栈中的 lowering 路径(Relax/Relay 图 → TIR → 最终 kernel),逐层对齐。

观察点(记入 Inductor 路径笔记):

- fusion 发生在哪一层,融合边界由什么决定;
- dynamic shape 是以符号形式进 Triton,还是触发 guard/重编译;
- Inductor 的 loop-level IR(define-by-run)与 TIR 的"循环 + buffer"表达各自的优缺点;
- 若分块 Attention 走 Inductor 路径,Legalize 这个位置大致对应哪一步(或根本不存在——这也是一个答案)。

降级方案:无 GPU 环境时改用 CPU backend 观察 C++/OpenMP codegen,观察点不变;连编译环境都没有时,以 torch.compile 官方教程中展示的 FX 图示例为素材做纸面分析。

---

## 六、能力验收标准

### 自测问题清单(12 月底逐条自答,写进笔记附录)

MLIR 部分:

1. 用自己的话解释 Dialect 存在的意义:为什么 MLIR 不设计一个"足够好的统一 IR"?
2. Operation / Region / Block / SSA 各自的职责边界是什么?Region 和 Block 的区别?
3. Pattern Rewrite 的 Greedy Driver 与 Walk Driver 在遍历顺序和终止条件上有何差异?
4. Canonicalization 和普通的变换 pass 有什么约定上的区别?
5. **Dialect Conversion 与 Type Conversion 的分工是什么?**(Partial / Full / Analysis 三种模式分别解决什么问题;TypeConverter 的 conversion 与 materialization 有何不同)
6. Progressive Lowering 思想在 TVM 的 Relay/Relax → TIR → LLVM 路径上是否成立?哪里成立、哪里不成立?

Inductor 部分:

7. 画出 Dynamo → FX → Inductor → Triton 的完整数据流图,标注每一环的输入输出 IR 形态。
8. graph break 和 guard 分别是什么?dynamic shape 在两条路径上的处理差异?
9. PrimTorch 的 decomposition 与 TVM 的 Legalize 在动机和机制上的异同?
10. Inductor 的 fusion 决策、layout 决策、autotuning 各发生在哪一层?对应 TVM 栈的什么位置?

分块 Attention 锚点部分(核心验收题):

11. **分块 Attention 的 Legalize 在完整编译栈中的职责是什么?**用不依赖 TVM API 的语言描述其输入、输出、合法性判定标准。
12. **分块 Attention 相关的哪些逻辑应留给 Runtime、哪些应留给 Kernel?**(提示方向:运行时才能确定的形状/掩码/变长信息 vs. 编译期必须固化的分块与数值策略 vs. Legalize 应完成的图级语义规范化)
13. 如果明天把分块 Attention 移植到 Inductor 生态,现有 Legalize 逻辑里哪些会成为 decomposition、哪些会成为自定义 kernel、哪些会消失?

### 交付物清单

| 交付物 | 完成时间 | 内容要求 |
|---|---|---|
| 《编译器分层架构笔记(不依赖 TVM API)》 | 12/31 前定稿 | 分层图 + 每层在 MLIR / Inductor / TVM 三个体系中的对应物 + 第 11、12 题的书面回答 |
| 《Inductor 路径笔记》 | 11 月底初稿、12 月修订 | Dynamo→FX→Inductor→Triton 数据流图 + 每环节 3–5 行注解 + 实验 2 观察记录 |
| 《TVM vs Inductor 差异表》初稿 | 12/31 前 | 至少覆盖:graph capture、算子面收敛、IR 层级、fusion、dynamic shape、autotuning 六行;标注"待验证"的格子允许存在 |
| 自测记录 | 12/31 前 | 13 道题逐条自答,答不出的标记为 Q1 遗留项 |

---

## 七、风险与降级方案

### 定位声明

本方向在整体规划中占比 15%,是**时间紧张时最先被压缩的方向**。压缩不是失败:本方向的价值在于"长期判断力",一个季度少读几页文档不会立刻损害主线进展,但主线(分块 Attention 编译重构、Inference 方向)被耽误的代价更高。因此压缩策略要果断,不内耗。

### 分级压缩策略

- **轻度紧张(某周只剩 30 分钟)**:停掉当周阅读,只保留周末 15 分钟的笔记整理——把脑子里已有的概念和锚点问题写下来。规则:**笔记不停,新资料可以停**。
- **中度紧张(连续 2–3 周紧张)**:当月主题缩水。10 月缩水方案:只保留 LangRef + Dialect Conversion 两个文档,论文降为 P1;11 月缩水方案:只做数据流图 + 实验 2 降级为纸面分析;实验 1 永远第一个被砍。
- **重度紧张(整月顾不上)**:当月主题整体顺延到下月,12 月"成稿月"不可顺延——架构笔记哪怕只有 2 页也要成稿,因为交付物是本方向存在的证据。
- **底线交付**:无论压缩到哪一级,12/31 必须存在三份文件:架构笔记(可薄)、Inductor 路径笔记(可薄)、差异表(可留"待验证"格子)。薄的成稿 > 厚的草稿。

### 其他风险

| 风险 | 应对 |
|---|---|
| MLIR 文档偏实现细节,容易陷进 C++ API | 每月开头重读一遍定位:"理解设计,不做 MLIR Backend 开发";遇到 API 细节一律跳过,只取设计动机 |
| 读 Inductor 时滑向源码深读,时间失控 | 源码只在笔记卡住时定点查阅(P2),不设"读源码"的时间块 |
| 锚点问题答不上来,挫伤节奏 | 答不上来是正常的,记入"待验证问题"清单即算完成当周的锚点任务 |
| 分块 Attention 重构需求变化,锚点失效 | 锚点切换为重构后的最新形态;概念层的学习不受影响 |

---

## 附:外部资料核实记录

以下链接于撰写本计划时逐一访问核实(2026 Q4 计划编制期间):

- MLIR Language Reference — https://mlir.llvm.org/docs/LangRef/
- MLIR CGO 2021 论文 — DOI: 10.1109/CGO51591.2021.9370308,官方论文列表页 https://mlir.llvm.org/pubs/ 确认其存在并指向 arXiv 预印本
- Pattern Rewriting 文档 — https://mlir.llvm.org/docs/PatternRewriter/
- Dialect Conversion 文档 — https://mlir.llvm.org/docs/DialectConversion/
- MLIR Toy Tutorial 第 2 章 — https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/
- PyTorch 2.x 官方介绍 — https://pytorch.org/get-started/pytorch-2-x/
- torch.compiler 文档 — https://docs.pytorch.org/docs/stable/torch.compiler.html
- torch.compile 入门教程 — https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- PyTorch Logging(TORCH_LOGS)文档 — https://docs.pytorch.org/docs/stable/logging.html
- PyTorch 2.0 Release Blog — https://pytorch.org/blog/pytorch-2-0-release/

若后续发现失效:MLIR 侧从 https://mlir.llvm.org 的 Docs 导航重新定位;PyTorch 侧从 https://docs.pytorch.org 站内搜索 "torch.compile" / "TORCH_LOGS" 重新定位。
