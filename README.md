# inference-lab

Mac 上的 LLM 推理源码阅读与学习环境，集中维护季度计划、vLLM V1 阅读笔记、模型结构卡片和 CPU 练习。

当前重点是通过静态阅读和小脚本理解请求生命周期、调度、KV cache、tokenizer 与张量布局。CUDA/Triton 目录保留实验规划与报告模板，GPU trace 和性能数据尚待实际验证。

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
| [scripts/setup_env.sh](./scripts/setup_env.sh) | 创建并恢复项目学习环境 |
| `third_party/vllm/` | vLLM 源码 submodule，固定为 `v0.31.0` |
| `pyproject.toml` / `uv.lock` / `.python-version` | Python 依赖声明、锁文件与解释器版本 |
| `.venv/` | 本地虚拟环境，不提交到 Git |

## 在新 Mac 上恢复环境

先安装 Git 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)，然后执行：

```bash
git clone --branch MacOS --recurse-submodules \
  https://github.com/Jankoski-7c00/llm-inference-lab.git \
  ~/inference-lab
cd ~/inference-lab
bash scripts/setup_env.sh
```

项目目录名使用 `inference-lab`，GitHub 仓库地址仍为 `llm-inference-lab`。脚本自动准备 Python 3.12，在项目内创建 `.venv/`，按 `uv.lock` 恢复依赖并验证导入和 CPU 计算。重复运行不会升级锁定的依赖。初次安装需要网络，不下载模型权重。

```bash
uv run python                          # 进入 Python
uv run ipython                         # 进入交互式学习环境
uv run python path/to/script.py         # 替换为实际脚本路径
source .venv/bin/activate               # 可选：手动激活环境
```

脚本也可以从其他目录调用：`bash ~/inference-lab/scripts/setup_env.sh`。新机器上运行脚本重建环境，无需复制旧机器的 `.venv/`。

## 恢复 vLLM submodule

如果克隆时没有使用 `--recurse-submodules`，在项目根目录执行：

```bash
git submodule update --init --recursive
git submodule status
```

该命令检出主仓库记录的固定提交，不追踪 upstream 最新版本。当前提交为 `db9527a46873454610df6dbedf79a36d6bf1a7f6`（`v0.31.0`）。此 submodule 用于阅读源码；当前脚本未安装完整 vLLM，也不执行 GPU 推理。

## Mac 上的学习路线

1. 阅读[季度总览](./learning-plans/2026Q4/00_季度总览与执行机制.md)，明确学习目标。
2. 从[vLLM 起步阅读笔记](./01-vllm-runtime-trace/v0.31.0/00-first-request-reading-notes.md)开始，带着问题追踪单次请求。
3. 用 IPython、NumPy 和 PyTorch CPU 小脚本验证 token、索引与张量概念；阅读 KV 管理和 Attention metadata，明确区分静态推导与运行观测。

## 2026 Q4 学习方向

- [ ] vLLM decode 全链路 trace 与调用链图
- [ ] Triton reduction / softmax 与首份 profiling
- [ ] tiled attention / online softmax 对齐 reference
- [ ] block size / chunked prefill / prefix caching 控制变量实验
- [ ] 正确可测的 paged KV decode attention 原型
- [ ] 编译器分层架构笔记与 DeepSeek MLA 结构卡片
- [ ] 季度复盘

## License

MIT，详见 [LICENSE](./LICENSE)。vLLM submodule 使用其自身许可证。
