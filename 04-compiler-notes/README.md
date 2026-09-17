# 04 · Compiler Notes

**学习方向**:编译器横向。以本职的 TVM 编译栈为参照系,精读 MLIR 概念与 TorchInductor 的 lowering / codegen 路径,理解"PyTorch 图如何变成 GPU kernel",反哺对 vLLM 编译路径与 kernel 生成的理解。

## 计划内容

- **MLIR 概念精读**:dialect、pass、lowering 链条的最小工作模型。
- **TorchInductor 路径笔记**:从 FX graph 到 Triton codegen 的关键环节,结合实际模型 dump 中间产物。
- **与 TVM 的对比**:分层抽象、schedule 表达、算子融合的异同——用已有的 TVM 经验加速对 Inductor/MLIR 的建模。
- **编译器分层架构笔记**(季度里程碑):一张从 frontend 到 codegen 的分层图,标注各层职责。

## 状态

⬜ 未开始(10 月先做 MLIR 概念精读)
