# 01 · vLLM Runtime Trace

**学习方向**:读懂 vLLM 推理运行时——从一次 decode 请求进入系统,到 scheduler、KV cache 管理、model runner、attention backend 的完整调用链,建立"推理系统在每一微秒做什么"的直觉。

**为什么排第一**:推理系统的性能问题最终都落在 Runtime 的调度与内存决策上;先有调用链全局图,后面的 kernel 优化才知道改在哪。

## 计划产出

- **decode 调用链图**:一张图串起 decode 路径上的关键入口与模块(入口 → scheduler → block manager → model runner → attention backend → kernel),标注 hot path。
- **trace 笔记**:按版本记录关键代码位置、数据结构(paged KV block table 等)与阅读过程,按目录分文件。
- **控制变量实验**:对 block size、chunked prefill、prefix caching 等开关做单变量对比,记录配置、结果与解读(配合 `03-e2e-benchmark` 的记录字段约定)。

## 状态

🚧 进行中
