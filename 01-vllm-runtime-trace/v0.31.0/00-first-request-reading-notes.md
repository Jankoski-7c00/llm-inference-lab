# vLLM V1 起步笔记：带着问题追踪一次推理请求

> 本笔记基于公开源码，不包含任何雇主代码与数据。
> 创建日期：2026-10-11。状态：阅读提纲，尚未完成运行 trace；下面的空表不是实测结果。

## 1. 本次目标与阅读边界

对应[季度 Runtime 学习计划](../../../learning-plans/2026Q4/01_vLLM与推理Runtime.md)的 10 月任务：建立 V1 骨架，走通一次请求，为后续「Attention 输入如何生成」的完整 trace 打基础。

第一遍只追一条主线：**短 prompt → 请求入队 → 首次模型计算 → 后续 decode → 请求结束 → 输出文本**。先使用离线 `LLM.generate()`，之后再补 `AsyncLLM` 在线入口。多模态、分布式、LoRA、推测解码留待主线读通后学习。

源码版本：`v0.31.0`，submodule commit：`db9527a46873454610df6dbedf79a36d6bf1a7f6`。季度计划中的旧版本示例与路径仅作导航，以当前源码和实际执行分支为准。

### 待记录的运行环境

| 项目 | 记录 |
|---|---|
| 模型 | 计划：`Qwen/Qwen2.5-0.5B-Instruct`；实际：待填 |
| 安装的 vLLM 版本与源码位置 | 待填；确认运行代码与阅读版本一致 |
| GPU / 显存 / 驱动 / CUDA | 待填 |
| PyTorch / Python / dtype | 待填 |
| prompt / 实际 token IDs / 输出上限 | 待填 |
| prefix caching / chunked prefill / async scheduling | 待填；记录实际生效配置 |
| runner / executor / Attention backend / 进程 | 待填；从日志或调用路径确认 |
| 运行命令与脚本位置 | 待填；无 GPU 时先做静态阅读，GPU trace 另行安排 |

## 2. 第一晚：先抓住引擎循环（60–75 分钟）

先读 [EngineCore.step()](../../third_party/vllm/vllm/v1/engine/core.py)，再向两端扩展。定位 `schedule → execute_model → sample_tokens（按分支）→ update_from_output`。暂时只跟调用关系与数据流。

- [ ] 一次 `step()` 对应一次请求，还是一批请求的一轮计算？
- [ ] `scheduler_output` 和 `model_output` 分别包含什么？谁生产、谁消费？
- [ ] 为什么 `execute_model()` 返回后还可能调用 `sample_tokens()`？本次配置走哪个分支？
- [ ] `update_from_output()` 更新了请求的哪些状态？新 token 在哪里加入请求？
- [ ] `step_with_batch_queue()` 与 `step()` 如何被选择？实际运行是否走我阅读的路径？

**本晚产出**：用自己的话写 5–8 行，解释一轮推理。证据至少包含一组真实的调用方与被调用方。

我的解释：待填。

## 3. 第二晚：请求如何进入、结果如何返回（60–75 分钟）

阅读顺序：

1. [LLM.generate()](../../third_party/vllm/vllm/entrypoints/llm.py)。
2. [OfflineInferenceMixin](../../third_party/vllm/vllm/entrypoints/offline_utils.py)：`_run_completion()`、`_add_request()`、`_run_engine()`，补查 rendering 路径。
3. [LLMEngine](../../third_party/vllm/vllm/v1/engine/llm_engine.py)：`add_request()`、`step()`。
4. [InputProcessor](../../third_party/vllm/vllm/v1/engine/input_processor.py)、[OutputProcessor](../../third_party/vllm/vllm/v1/engine/output_processor.py)、[EngineCoreClient](../../third_party/vllm/vllm/v1/engine/core_client.py)。

- [ ] 文本在哪一层变成 token IDs？rendering 与 `process_inputs()` 的职责如何分工？
- [ ] `EngineCoreRequest` 与 Scheduler 使用的 [Request](../../third_party/vllm/vllm/v1/request.py) 有什么区别？转换在哪里发生？
- [ ] 请求 ID 如何贯穿输入、调度、输出？能否用它串起日志？
- [ ] `LLMEngine.step()` 与 `EngineCore.step()` 如何连接？哪些调用跨进程，哪些在当前进程？记录实际进程，不预设固定数量。
- [ ] token ID 在哪里转回文本？EOS、输出长度、stop string 分别由谁判断？请求结束后如何通知其他组件？

**本晚产出**：画出「输入侧 → 引擎核心 → 输出侧」，标出对象类型与通信边界。

## 4. 第三次：调度器为什么让它计算这些 token？

读 [Scheduler.schedule() / update_from_output()](../../third_party/vllm/vllm/v1/core/sched/scheduler.py)、[SchedulerOutput](../../third_party/vllm/vllm/v1/core/sched/output.py) 和 [KVCacheManager.allocate_slots()](../../third_party/vllm/vllm/v1/core/kv_cache_manager.py)。

- [ ] 请求何时进入 waiting，何时进入 running，何时移除？
- [ ] `num_computed_tokens`、prompt 长度、已输出 token 数各表示什么？生成了一个 token，是否表示它的 KV 已经计算？
- [ ] 阅读 `schedule()` 开头的统一 token 进度注释：prefill 与 decode 如何通过相同机制安排？
- [ ] `num_scheduled_tokens[request_id]` 如何决定？`max_num_batched_tokens` 如何约束本轮预算？
- [ ] 首次请求与已运行请求在 `SchedulerOutput` 中有何区别？worker 为何需要保存状态？
- [ ] KV block 在调度的哪个位置分配？分配失败和请求结束分别触发什么？先记录分支，抢占机制留待第二遍展开。

### 手推练习：8-token prompt，最多生成 3 个 token

假设单请求、无缓存命中、无推测解码、预算足够，忽略提前 EOS。**8 个 token 是手推条件，不代表输入 8 个字。**先预测，再用实际 trace 验证；异步调度下同时注明观察时点。

| 轮次 | 计算前 num_computed_tokens | 本轮输入 token / position | 调度 token 数 | 新采样 token | 计算后有效 KV 范围 | 是否结束 / 原因 |
|---|---|---|---|---|---|---|
| 首轮 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 后续轮 1 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 后续轮 2 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 5. 主线读通后：追到 Attention 输入

入口：[GPUModelRunner](../../third_party/vllm/vllm/v1/worker/gpu_model_runner.py) 的 `execute_model()`、`_prepare_inputs()`、`_model_forward()`、`sample_tokens()`。

- [ ] 调度结果如何更新 persistent `InputBatch`？本轮 `input_ids` 与 `positions` 从哪里取？
- [ ] block table 表达「逻辑 block → 物理 block」的哪部分映射？slot mapping 的每个元素对应哪个输入 token？
- [ ] 在简单分页布局中，`slot = block_id * block_size + offset` 的三个量从哪里来？当前路径是否有额外布局约束？
- [ ] `query_start_loc`、`seq_lens`、`slot_mapping` 的 shape、dtype、数值分别是什么？首轮与 decode 有什么变化？
- [ ] metadata 在哪里构造、如何传入模型 Attention 层？实际 backend 是什么，最终调用哪个 kernel？

导航：当前 `GPUModelRunner` 使用的 [worker/block_table.py](../../third_party/vllm/vllm/v1/worker/block_table.py)；仓库另有 `worker/gpu/block_table.py`，先确认 runner 再选择路径。若实际 backend 为 FlashAttention，再追 [FlashAttentionMetadataBuilder.build()](../../third_party/vllm/vllm/v1/attention/backends/flash_attn.py)。不要把计划中的 backend 当成运行结果。

## 6. 每次阅读的记录模板

每次选 2–3 个问题深入，未理解的地方保留为问题。

| 问题 | 当前解释或假设 | 源码证据（文件、函数、行号、commit） | 运行观察 | 状态 |
|---|---|---|---|---|
| 待填 | 待填 | 待填 | 未运行 | 待确认 |

- 日期 / 本次范围：待填。
- 最关键的数据流：谁生成什么对象或张量，谁消费？待填。
- 与原先理解不同的地方：待填。
- 下一次只追的两个问题：待填。

## 7. 第一轮完成标准

- [ ] 能不看源码讲清一次请求从输入到结束的主干。
- [ ] 能解释首轮与后续轮输入、token 进度和 KV 状态的差异。
- [ ] 能画出带类名、方法名和数据对象的初版调用链；跨进程连线单独标注。
- [ ] 有一份真实 trace，或明确标为静态阅读的记录；不混写推测与观测。

后续问题停车场：continuous batching、抢占、prefix caching、chunked prefill、KV 显存预算，以及它们对 TTFT / TPOT 的影响。按计划放到 11 月机制阅读与控制变量实验中展开。
