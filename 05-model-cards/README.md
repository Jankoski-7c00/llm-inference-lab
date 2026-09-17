# 05 · Model Cards

**学习方向**:跟踪 Attention 演进与新模型结构。每篇结构卡片回答一个问题:**这个结构对推理系统提出了什么要求**(KV cache 形态、attention 计算模式、并行与显存策略)。

## 计划卡片

- **DeepSeek MLA(Multi-head Latent Attention)**:latent KV 压缩的结构拆解,对 KV cache 显存与 decode 计算的影响;结合已有的 DeepSeek 适配经验整理(季度里程碑:MLA 结构卡片)。
- **SWA(Sliding Window Attention)**:滑窗 attention 的数据流梳理,对 KV block 管理与调度策略的影响。

## 适配清单

针对每个结构,维护一份"推理适配 checklist":KV cache layout、attention backend 需求、chunked prefill / prefix caching 兼容性、已知坑位。

## 状态

⬜ 未开始(10 月产出 MLA 结构卡片初稿)
