# Qwen3.5 CUDA 算子面试实战

从 vLLM 的算子契约出发，练习四个算子的 CUDA 前向实现，保留数学 reference、基础实现、优化阶段和实验记录。按 [学习计划](PLAN.md) 逐课推进，第一次从 [第 1 课](lessons/01-contracts.md) 开始。

## ncu 实操待补充

**第 13 课的 Nsight Compute（ncu）实操需要之后在 PRO 6000 上额外完成，目前未完成实机验收。** 当前环境无法访问 GPU 性能计数器，尚未成功采集 `.ncu-rep`。

当前已有四个算子的实际 CUDA 实现、正确性测试、微基准、Systems 报告摘要，以及 [ncu 采集脚本](profile.py)和[报告阅读教程](lessons/13-profiling.md)。尚未提供真实 Compute 报告、基于报告的指标分析案例，以及根据这些指标新增并验证的调优代码。已有优化阶段的计时结果不能表述为已经完成 ncu 分析或证实了具体硬件瓶颈。

后续按以下顺序补齐，并在 [学习进度](notes/PROGRESS.md) 和各算子 REPORT 中留下证据：

1. 在目标显卡重新构建并通过正确性检查，确认 ncu 支持具体型号且具备计数器访问权限。
2. 采集基础版和优化版的真实 Compute 报告，确认目标 kernel 和需要分析的指标存在。
3. 根据报告提出一个瓶颈假设，修改一个因素，重新验证正确性、独立计时并采集报告。
4. 保存前后指标、代码差异和复盘笔记，记录假设得到支持还是被否定。

CUDA 源码已通过 Blackwell `sm_120` 交叉编译；这只验证编译兼容性，PRO 6000 上的运行、数值结果、性能和 ncu 采集仍需实卡验证。现有测量值与优化排名需要在新显卡上重新测量。

## 先运行一个实验

以下命令均从仓库根目录执行。实验直接使用 `.venv/bin/python`，避免基础学习锁文件重新解析已经准备好的 GPU 依赖。

```bash
# 已有可用的 CUDA PyTorch 后，补充编译/测试依赖并构建
bash 02-triton-kernels/qwen35-cuda/setup_lab.sh

# 查看预设，再验证基础实现
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py list
.venv/bin/python 02-triton-kernels/qwen35-cuda/run.py check --case cache_decode --impl baseline

# 完整正确性矩阵（包含独立 reference 和原版 Triton 对照）
.venv/bin/python -m pytest 02-triton-kernels/qwen35-cuda/tests -q
.venv/bin/python 02-triton-kernels/qwen35-cuda/suite.py --check-only --include-upstream --cases all
```

准备脚本要求已有 `uv`、`nvcc` 和兼容的 C++ 编译器。没有 `.venv` 时，它从 `QWEN35_PYTHON` 指定的已有解释器建立可见系统包的环境；已有 `.venv` 时保留它。脚本只补充 Ninja 和 pytest，不自动替换 PyTorch。数学 reference 的 CPU 测试可单独运行 `tests/test_reference.py`。

Sanitizer 使用 CPU goldens，单独验证本实验 CUDA 路径，避免 GPU BLAS reference 初始化的内部查询干扰结果；没有关闭 CUDA API 错误报告：

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 .venv/bin/python 02-triton-kernels/qwen35-cuda/sanitize.py
compute-sanitizer --tool synccheck --error-exitcode 99 .venv/bin/python 02-triton-kernels/qwen35-cuda/sanitize.py
compute-sanitizer --tool racecheck --error-exitcode 99 .venv/bin/python 02-triton-kernels/qwen35-cuda/sanitize.py
```

## 课程入口

| 课 | 主题 | 课 | 主题 |
|---|---|---|---|
| [01](lessons/01-contracts.md) | 调用链与算子契约 | [08](lessons/08-fa2-wmma.md) | FA2 分工与 WMMA |
| [02](lessons/02-cache-indexing.md) | 分页缓存地址 | [09](lessons/09-async.md) | 异步搬运与双缓冲 |
| [03](lessons/03-cache-optimization.md) | 向量化与带宽估算 | [10](lessons/10-delta-math.md) | Delta rule 数学 |
| [04](lessons/04-dynamic-quant.md) | 动态量化 | [11](lessons/11-delta-state.md) | 状态索引与多步递推 |
| [05](lessons/05-warp-and-rounding.md) | Warp 归约与舍入 | [12](lessons/12-delta-fusion.md) | 融合与寄存器状态 |
| [06](lessons/06-attention-reference.md) | Attention reference | [13](lessons/13-profiling.md) | 报告采集、阅读与调优 |
| [07](lessons/07-online-softmax.md) | Online softmax | [14](lessons/14-review.md) | 复盘与模拟面试 |

先完成每课练习，再查看 [答案与口述稿](lessons/ANSWERS.md)。用 [进度表](notes/PROGRESS.md) 记录你实际掌握的内容，代码测试通过不代表学习里程碑已经完成。

## 代码与输入契约

四个 Python 入口位于 [api.py](qwen35_lab/api.py)，CUDA 在 [kernels.cu](csrc/kernels.cu)。每个实验目录有独立报告，共享编译、测试、case 和测量工具。

| 入口 | 输入与输出 | 原版源码 |
|---|---|---|
| `reshape_cache` | K/V `[T,H,D]`；原地写 cache | `reshape_and_cache_kernel_flash` |
| `dynamic_cache` | K/V 可有不同 D；原地写 INT8/FP8 cache 和 FP32 scales | `_reshape_cache_per_token_head` |
| `attention` | Q/K/V `[T,H,D]`，`cu_seqlens` 与 Python `max_seqlen`；输出 `[T,H,D]` | MM encoder 的非因果 FlashAttention 路径 |
| `delta_rule` | packed Q/K/V、门控和 `[pool,HV,V,K]` 状态；返回 output 和 state/trace | `fused_sigmoid_gating_delta_rule_update_kernel` |

- 输入支持 FP32/FP16/BF16；Tensor Core attention 使用 FP16/BF16。`baseline`、`simt`、`wmma`、`async` 是逐步可比较的实现阶段。
- Attention 覆盖 `1 <= D <= 256`；异步路径要求 K/V 起始地址、D 与外层 stride 满足 16-byte 对齐条件，不满足时显式选择 `wmma` 或 `simt`。
- Delta rule 覆盖 `1 <= K <= 256`、grouped value heads、FP32/FP16/BF16 状态、单步及短序列 spec、多步快照。`A_log/dt_bias` 使用 FP32。KDA、INT4、反向传播不在本实验范围。
- Delta 的可比较阶段为 `baseline`（三个 kernel）、`fused`（直接融合）、`shared`（仅共享计算）、`tiled`（再改变行分工）和 `optimized`（再专用化 K 循环）。保留实际发生过的退化，逐步控制变量。
- 非负 slot 必须有效且唯一；不同序列不能同时写同一个状态。`validate_metadata` 在构造输入时检查这些条件，会同步到 CPU，不能放进计时区间。
- Delta 索引 0 保留为空状态。本实现为空初始状态定义零输出；原版 Triton 对应 padding 输出未定义，因此上游比较只适用于有效 token。无效状态不被写入。
- 原生 CUDA 的 FP8 转换直接使用 FP32→FP8 RNE。当前原版 Triton 部分路径包含 FP32→FP16 RTZ→FP8 RNE，边界上可以差一个相邻 FP8 编码。对独立 reference 仍要求逐字节一致，仅对这类上游 FP8 比较允许相邻编码；见 [第 5 课](lessons/05-warp-and-rounding.md)。本实验不承诺 FP8 与上游逐位等价。
- INT8 对独立 reference 要求精确一致。上游 `div.full.f32` 与原生 FP32 除法在半整数边界也可能得到相邻整数，上游辅助比较允许 1 个整数单位；scale 仍按固定浮点容差检查，不放宽本实验实现与数学 reference 的要求。

原版 Triton 函数从仓库固定源码原文抽取到忽略的 `.cache/`，保留许可证、路径和源码摘要；未修改 vLLM 子模块，也不需要先安装完整 vLLM。抽取函数的 launch adapter 用于正确性，不能当作公平微基准。

## 动手补写 CUDA

```bash
PYTHONPATH=02-triton-kernels/qwen35-cuda .venv/bin/python -m qwen35_lab.exercise cache
# 按输出提示编辑 .cache/exercises/cache/kernels.cu，再运行输出的测试命令
# 其他练习：dynamic、softmax、delta
PYTHONPATH=02-triton-kernels/qwen35-cuda .venv/bin/python -m qwen35_lab.exercise cache --answer
```

练习生成器只在 `.cache` 中制作带空白的副本，保留已验证的实现。`QWEN35_KERNEL_SOURCE` 可选择你的副本；已有学生文件不会被覆盖。

## 计时与 profiling

```bash
.venv/bin/python 02-triton-kernels/qwen35-cuda/suite.py --output 02-triton-kernels/qwen35-cuda/outputs/suite.csv
.venv/bin/python 02-triton-kernels/qwen35-cuda/profile.py nsys --case encoder_short --impl async
# 后续具备性能计数器权限时运行；采集失败会保留失败状态并返回非零
.venv/bin/python 02-triton-kernels/qwen35-cuda/profile.py ncu --case encoder_short --impl async
```

Graph 微基准把重置拷贝放在每对外部 event 节点之外，避免递归状态漂移，也减少 Python 提交间隙。`--mode eager` 单独记录主机提交影响；两种模式不能混合比较。这些是重置后数据较热的目标区间测量，不代表完整模型的 TTFT/TPOT。

原始报告在忽略的 `profiles/`，临时数据在 `outputs/`；精简复现结果在 [results/](results/README.md)。计划和讲义不绑定具体环境，实际实验硬件、软件、源码摘要在结果元数据中单独记录。

## 阅读来源

- [仓库缓存算子](../../third_party/vllm/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py)
- [MM encoder 调用](../../third_party/vllm/vllm/model_executor/layers/attention/mm_encoder_attention.py) 与 [非因果 varlen wrapper](../../third_party/vllm/vllm/v1/attention/ops/vit_attn_wrappers.py)
- [Delta rule 算子](../../third_party/vllm/vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py)
- [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691)
- [NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/) 与 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
