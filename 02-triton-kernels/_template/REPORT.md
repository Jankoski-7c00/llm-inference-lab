# 实验报告:<实验名>

## 1. 问题与场景

- 模型 / 算子:
- dtype:
- batch 或 token 数:
- context length:
- 并发:
- 硬件:

## 2. 基线与瓶颈假设

- 基线(实现 + 数据):
- 瓶颈假设:
  - Kernel latency:
  - HBM 带宽(理论 vs 实测):
  - compute 利用率:
  - transfer 与 sync 开销:

## 3. 方案与取舍

- 方案:
- 为什么改这一层(Runtime / Compiler / layout / Kernel):
- 被放弃的方案及原因:

## 4. 结果

- 微基准:
- TTFT / TPOT:
- throughput:
- 峰值显存:
- (原始数据 / 图表链接)

## 5. 适用边界与失败尝试

- 在哪些 shape / 场景中有效:
- 在哪些 shape / 场景中无效或回退:
- 失败尝试与原因:

## 6. 可迁移结论

- 抽象出的通用方法:

---

## 附:环境与复现

- GPU 型号:
- CUDA / Triton / PyTorch 版本:
- 复现命令:
- commit / 数据存档位置:
