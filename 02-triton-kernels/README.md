# 02 · Triton Kernels

**学习方向**:GPU 性能的立身之本。从最基本的 memory-bound kernel 写起,逐级进阶到本季度的主项目:**正确可测的 paged KV decode attention 原型(Triton)**。

## 进阶路线

1. **reduction / softmax**:建立 block、mask、reduction 的基本功,产出首份 profiling(带宽利用率、瓶颈判定)。
2. **tiled attention + online softmax**:实现标准 FlashAttention 风格的 tiled attention,数值上对齐 PyTorch reference。
3. **paged KV decode attention 原型**(2026 Q4 主项目):支持 paged KV layout 的 decode attention,正确性对齐 reference,性能有 profiling 数据。

## 组织约定

- 每个实验一个子目录,报告统一复用 [`_template/REPORT.md`](./_template/REPORT.md) 模板。
- 数值正确性一律对齐 reference(PyTorch 或已知正确实现),profile 数据写入报告第 4 节。
- 微基准关注 kernel latency、HBM 带宽利用率、compute 利用率;端到端指标放 `03-e2e-benchmark`。

## 状态

🚧 进行中(第 1 阶段:reduction / softmax)
