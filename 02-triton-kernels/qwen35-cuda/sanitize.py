#!/usr/bin/env python3
"""Target the lab kernels with CPU goldens, avoiding GPU BLAS probe traffic.

Usage: compute-sanitizer --tool memcheck --error-exitcode 99 \
    .venv/bin/python 02-triton-kernels/qwen35-cuda/sanitize.py
The same worker can be run with synccheck or racecheck. No API errors are hidden.
"""

import argparse

import torch

from qwen35_lab import api
from qwen35_lab.cases import Case, make_case


def cpu_copy(case):
    args = {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in case.args.items()}
    kwargs = {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in case.kwargs.items()}
    return Case(case.name, case.op, args, kwargs, case.mutable).freeze()


def compare(a, b, *, op):
    a = a.cpu()
    if a.dtype == api.FP8:
        torch.testing.assert_close(a.view(torch.uint8), b.view(torch.uint8), rtol=0, atol=0)
    elif op in ("cache", "dynamic") and a.dtype != torch.float32:
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    else:
        tol = (2e-4 if a.dtype == torch.float32 else (5e-3 if a.dtype == torch.float16 else 2e-2)) if op in ("attention", "delta") else 1e-6
        torch.testing.assert_close(a, b, rtol=tol, atol=tol)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    args = parser.parse_args()
    names = ["cache_decode", "cache_head_major", "dynamic_int8", "dynamic_fp8", "encoder_short", "delta_spec"]
    cases = [make_case(name, dtype=getattr(torch, args.dtype)) for name in names]
    tail = make_case("delta_spec", dtype=getattr(torch, args.dtype))
    tail.name = "delta_non_square_padding_trace"
    for name in ("q", "k"):
        tail.args[name] = tail.args[name][:, :2, :65].contiguous()
    tail.args["v"] = tail.args["v"][:, :4, :33].contiguous()
    for name in ("a", "b"):
        tail.args[name] = tail.args[name][:, :4].contiguous()
    for name in ("A_log", "dt_bias"):
        tail.args[name] = tail.args[name][:4].contiguous()
    tail.args["initial_state"] = tail.args["initial_state"][:, :4, :33, :65].contiguous().to(torch.bfloat16)
    tail.args["state_indices"][1, 1] = 0
    tail.kwargs["inplace"] = False
    tail.freeze()
    cases.append(tail)
    unaligned = make_case("cache_decode", dtype=getattr(torch, args.dtype))
    unaligned.name = "cache_misaligned_origin"
    for name in ("key", "value", "key_cache", "value_cache"):
        old = unaligned.args[name]
        shifted = torch.empty(old.numel()+1, dtype=old.dtype, device=old.device)[1:].view(old.shape)
        shifted.copy_(old)
        unaligned.args[name] = shifted
    unaligned.freeze()
    cases.append(unaligned)
    static_fp8 = make_case("cache_decode", dtype=getattr(torch, args.dtype))
    static_fp8.name = "cache_static_fp8_nonbinary_scales"
    for name in ("key_cache", "value_cache"):
        static_fp8.args[name] = static_fp8.args[name].to(api.FP8)
    static_fp8.kwargs = dict(k_scale=0.13, v_scale=0.21)
    static_fp8.freeze()
    cases.append(static_fp8)
    for case in cases:
        cpu = cpu_copy(case)
        expected = cpu.run("reference")
        impls = ["baseline", "optimized"]
        if case.op == "attention":
            impls = ["baseline", "simt"] if args.dtype == "float32" else ["baseline", "simt", "wmma", "async"]
        elif case.op == "delta":
            impls = ["baseline", "fused", "shared", "tiled", "optimized"]
        for impl in impls:
            case.reset()
            actual = case.run(impl)
            torch.cuda.synchronize()
            if actual is not None:
                actual = actual if isinstance(actual, tuple) else (actual,)
                expected_seq = expected if isinstance(expected, tuple) else (expected,)
                for a, b in zip(actual, expected_seq):
                    compare(a, b, op=case.op)
            for name in case.mutable:
                compare(case.args[name], cpu.args[name], op=case.op)
            print(f"PASS {case.name}/{impl} (CPU golden)", flush=True)


if __name__ == "__main__":
    main()
