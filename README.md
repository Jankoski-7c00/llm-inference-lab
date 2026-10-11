# inference-lab

面向 Linux / NVIDIA CUDA 环境的 LLM 推理实验仓库，维护 vLLM Runtime trace、CUDA/Triton 算子、编译器笔记、模型结构卡片与端到端 benchmark。

当前目标是运行小模型，追踪调度、KV 分配、Attention metadata 到 GPU kernel 的执行链，并实现正确可测的 paged KV decode attention 原型。GPU 环境与实验数据尚待目标机器验证。

Mac 上的源码阅读与 CPU 练习见 [MacOS 分支](https://github.com/Jankoski-7c00/llm-inference-lab/tree/MacOS)。

## 项目结构

| 路径 | 内容 |
|---|---|
| [learning-plans/](./learning-plans/README.md) | 季度学习计划、时间分配与验收标准 |
| [01-vllm-runtime-trace/](./01-vllm-runtime-trace/README.md) | vLLM 调用链、阅读问题和 trace 记录 |
| [02-triton-kernels/](./02-triton-kernels/README.md) | CUDA/Triton 算子实验与报告模板 |
| [03-e2e-benchmark/](./03-e2e-benchmark/README.md) | TTFT、TPOT、吞吐和显存测量约定 |
| [04-compiler-notes/](./04-compiler-notes/README.md) | MLIR、TorchInductor 与编译器架构笔记 |
| [05-model-cards/](./05-model-cards/README.md) | MLA、SWA 等结构卡片与推理适配清单 |
| `assets/` | 共享图表与调用链图 |
| [scripts/setup_env.sh](./scripts/setup_env.sh) | 基础学习依赖脚本，尚未配置 CUDA Runtime |
| `third_party/vllm/` | vLLM 源码 submodule，固定为 `v0.31.0` |
| `pyproject.toml` / `uv.lock` / `.python-version` | 基础学习依赖与 Python 版本；尚未锁定 GPU 环境 |
| `.venv/` | 本地虚拟环境，不提交到 Git |

## 准备 CUDA 环境

目标环境为 Linux、NVIDIA GPU、可用的 NVIDIA 驱动和 Python 3.12。安装 Git 和 uv，并确认 `nvidia-smi` 能识别 GPU。根据固定版本的 [GPU 安装说明](./third_party/vllm/docs/getting_started/installation/gpu.md)和 [CUDA 安装细节](./third_party/vllm/docs/getting_started/installation/gpu.cuda.inc.md)，按硬件与驱动选择兼容的 wheel。

```bash
git clone --branch main --recurse-submodules \
  https://github.com/Jankoski-7c00/llm-inference-lab.git \
  ~/inference-lab
cd ~/inference-lab
nvidia-smi
```

### 安装 vLLM 并验证 GPU

下面是全新 checkout 上的 release-wheel 安装起点，需在目标 GPU 机器上验证。使用干净环境，让 vLLM 解析配套 PyTorch 依赖，不预先安装任意版本的 PyTorch。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "vllm==0.31.0" --torch-backend=auto
.venv/bin/python - <<'PY'
import torch
import vllm
print("vLLM:", vllm.__version__)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA is not available"
print("GPU:", torch.cuda.get_device_name(0))
x = torch.tensor([1, 2, 3], device="cuda")
print("GPU sum:", x.sum().item())
PY
```

当前根目录的 `uv.lock` 与 `setup_env.sh` 只覆盖基础学习依赖，尚未包含 vLLM/CUDA 的版本组合。此 GPU 环境直接使用 `.venv/bin/python`；不要用 `uv sync` 或 `uv run` 按基础锁文件重建它。验证后将实际驱动、CUDA、PyTorch、Triton 和 vLLM 版本写入实验报告，后续再维护 GPU 专用依赖与安装脚本。

若需修改 submodule 中的 Python 代码或 CUDA kernel，按固定版本安装文档选择 editable 安装或源码编译。安装 release wheel 时，运行的是安装包，并不会自动运行 submodule 中修改的代码。编译 CUDA 源码还需匹配的 CUDA Toolkit 和 `nvcc`。

### 小模型冒烟测试

GPU 检查通过后，用季度计划中的小模型验证离线推理。首次运行会下载模型权重；按可用显存选择 dtype 与配置。

```bash
.venv/bin/python - <<'PY'
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")
outputs = llm.generate(
    ["Explain the purpose of a KV cache in one sentence."],
    SamplingParams(temperature=0, max_tokens=32),
)
print(outputs[0].outputs[0].text)
PY
```

## 恢复 vLLM submodule

如果克隆时没有使用 `--recurse-submodules`，在项目根目录执行：

```bash
git submodule update --init --recursive
git submodule status
```

该命令检出主仓库记录的固定提交，不追踪 upstream 最新版本。当前提交为 `db9527a46873454610df6dbedf79a36d6bf1a7f6`（`v0.31.0`）。下载源码后即可阅读；运行 vLLM 推理需要另行配置对应平台的依赖。

## GPU 实验路线

1. 阅读[季度总览](./learning-plans/2026Q4/00_季度总览与执行机制.md)，明确学习目标。
2. 完成 GPU 检查与小模型冒烟测试，按[vLLM 起步阅读笔记](./01-vllm-runtime-trace/v0.31.0/00-first-request-reading-notes.md)记录首次 prefill 和后续 decode。
3. 追踪实际 runner 与 Attention backend，记录 block table、slot mapping、metadata 与 kernel 调用，再开展 Triton reference 对齐、profiling 和端到端测量。

## 2026 Q4 里程碑

- [ ] vLLM decode 全链路 trace 与调用链图
- [ ] Triton reduction / softmax 与首份 profiling
- [ ] tiled attention / online softmax 对齐 reference
- [ ] block size / chunked prefill / prefix caching 控制变量实验
- [ ] 正确可测的 paged KV decode attention 原型
- [ ] 编译器分层架构笔记与 DeepSeek MLA 结构卡片
- [ ] 季度复盘

## License

MIT，详见 [LICENSE](./LICENSE)。vLLM submodule 使用其自身许可证。
