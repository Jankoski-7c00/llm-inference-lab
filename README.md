# inference-lab

## About (EN)

A public learning lab tracking my journey from AI compiler engineer (TVM stack, SIMT GPU + SIMD NPU heterogeneity, Attention operators, FA/BFA legalize, DeepSeek adaptation) toward an **LLM Inference / AI Systems Performance** role by 2029. This repo collects reproducible experiments and notes along the full capability chain: vLLM inference runtime tracing, CUDA/Triton GPU kernel work, MLIR/TorchInductor compiler notes, model architecture cards (MLA, SWA), and end-to-end benchmarking (TTFT / TPOT / throughput / memory). Everything here is built on the open-source ecosystem — vLLM, Triton, PyTorch, public models — and documents plans and progress honestly, with no fabricated results. Current milestone: a correct, measurable **paged KV decode attention prototype in Triton** (2026 Q4).

---

## 这是什么

**个人定位**:AI 编译器工程师(TVM 编译栈 / SIMT GPU + SIMD NPU 异构 / Attention 算子 / FA·BFA Legalize / DeepSeek 适配),目标是在 2029 年前成为**懂编译器与硬件的 LLM Inference / AI Systems Performance 工程师**。

本仓库是学习作品集:所有里程碑都是**进行中的计划**,而不是已完成的成果。

### 能力链地图

```
Model(结构卡片/适配清单)
  → Runtime(vLLM 推理运行时 / 调度与 KV 管理)
    → Compiler(MLIR / TorchInductor / 分层架构笔记)
      → Kernel(CUDA / Triton:从 reduction 到 paged decode attention)
        → Hardware(HBM 带宽 / compute 利用率 / SIMT 执行模型)
          → 端到端性能验证(TTFT / TPOT / 吞吐 / 峰值显存)
```

### 目录导览

| 目录 | 学习方向 | 状态 |
|---|---|---|
| `01-vllm-runtime-trace/` | vLLM 推理 Runtime:decode 调用链 trace、控制变量实验 | 🚧 进行中 |
| `02-triton-kernels/` | CUDA/Triton GPU 性能:从 reduction 到 paged decode attention | 🚧 进行中 |
| `03-e2e-benchmark/` | 端到端 benchmark:TTFT / TPOT / 吞吐 / 显存 | ⬜ 未开始 |
| `04-compiler-notes/` | MLIR / TorchInductor 编译器横向,与 TVM 对比 | ⬜ 未开始 |
| `05-model-cards/` | 模型结构卡片(MLA、SWA)与适配清单 | ⬜ 未开始 |
| `learning-plans/` | 季度学习计划与验收标准 | 持续更新 |
| `scripts/` | 本地环境初始化脚本 | — |
| `assets/` | 图、调用链图、曲线等公共素材 | — |

### 学习计划与本地环境

季度计划已合并到 [`learning-plans/`](./learning-plans/README.md)，与实验和阅读笔记一起维护。

本地项目根目录：`/Users/jankoski/inference-lab`。安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后运行：

```bash
cd /Users/jankoski/inference-lab
bash scripts/setup_env.sh
uv run python
```

脚本创建项目内的 `.venv/`，使用 Python 3.12，安装 IPython、NumPy、PyTorch 和 Transformers。首次安装生成 `uv.lock`；重复运行按锁文件恢复环境。此环境用于源码阅读、tokenizer 和 CPU 张量练习，完整 vLLM / CUDA / Triton 环境另行配置。初次安装需要网络，不下载模型权重。

从其他目录也可运行 `bash /Users/jankoski/inference-lab/scripts/setup_env.sh`。配置与锁文件提交到 Git，虚拟环境不提交。仓库远程地址仍为原 `llm-inference-lab` 地址。

### 2026 Q4 里程碑

- [ ] vLLM decode 全链路 trace + 调用链图
- [ ] Triton reduction / softmax + 首份 profiling
- [ ] tiled attention + online softmax,对齐 reference
- [ ] vLLM 控制变量实验(block size / chunked prefill / prefix caching)
- [ ] paged KV decode attention 原型(Triton,正确可测)
- [ ] 编译器分层架构笔记
- [ ] DeepSeek MLA 结构卡片
- [ ] 季度复盘(12 月底)

### 脱敏声明

本仓库只含基于开源生态(vLLM / Triton / PyTorch / 公开模型)的学习与实验内容,不包含任何雇主代码与数据。

### License

MIT — 详见 [LICENSE](./LICENSE)。
