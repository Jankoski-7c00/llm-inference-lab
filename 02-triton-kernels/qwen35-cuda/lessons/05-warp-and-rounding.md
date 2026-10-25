# 第 5 课：warp 归约与量化舍入路径

## Warp 并行

一个 warp 处理一个 `(token,head)`。lane `l` 访问 `l,l+32,l+64,...`。每条 lane 先计算局部 absmax，再以 16、8、4、2、1 的距离 shuffle 合并。最终 lane 0 的值广播给全 warp。

尾部维度用 0 参与 absmax，所有 lane 仍执行 shuffle；不能因为 `lane>=D` 就提前退出而让其他 lane 使用 full mask。K 和 V 分别循环、分别归约。

## 为什么数值相近不保证编码相同

主测试要求 CUDA 与独立 reference 的 INT8/FP8 存储精确一致。独立 reference 使用高精度临时除法再显式舍入至 FP32，以避免张量库的除以标量优化隐式改变 scale。

普通 FP8 写入的非二进制 scale 同样要注意这一点：先把 scale 按实际参数类型舍入为 FP32，再用高精度除法构造 FP32 商。测试用 `0.13/0.21` 附近的 FP8 中点与相邻 FP32 输入，防止 reference 的倒数乘法优化改变最终编码。

上游对照有两个已定位的边界差异：

1. 某些 FP8 转换路径先执行 `cvt.rz.f16.f32`，再执行 `cvt.rn.satfinite.e4m3x2.f16x2`。例如 `-5.252205...`，直接 FP8 RNE 得到 `-5.5`；中间向零截断到 FP16 后落在 FP8 中点，结果为 `-5.0`。
2. 动态量化的上游 PTX 使用 `div.full.f32`。原生除法算出的 `63.499996...` 可能在另一条计算路径上跨过 `63.5`，INT8 结果相差 1。即使 scale 看起来相同，reciprocal 也可能有不同舍入。

因此，仅对上游辅助比较允许一个 INT8 单位或一个相邻有限 FP8 编码，scale 容差为 `rtol=1e-6,atol=1e-8`。这不改变普通复制与本实验数学 reference 的严格比较，也不意味着任意数值误差都可接受。

## 实际查看 PTX

```bash
.venv/bin/python 02-triton-kernels/qwen35-cuda/inspect_upstream.py cache --output 02-triton-kernels/qwen35-cuda/outputs/cache.ptx
.venv/bin/python 02-triton-kernels/qwen35-cuda/inspect_upstream.py dynamic --output 02-triton-kernels/qwen35-cuda/outputs/dynamic.ptx
rg 'cvt\.|div\.|rcp' 02-triton-kernels/qwen35-cuda/outputs/*.ptx
```

读取你实际生成的 PTX；不同 lowering 路径可能变化。不要把本课观察写成所有 GPU/编译器的固定事实。

## 实验与过关

比较 `dynamic_int8` 的 baseline/optimized，并记录一个 warp 每个 lane 实际处理多少元素。再比较 `dynamic_fp8`。解释归约指令减少和 warp 利用率的取舍。

过关：能写出 warp max，并解释“浮点 scale 容差、整数存储差异、FP8 编码间隔”为什么是三种不同判据。
