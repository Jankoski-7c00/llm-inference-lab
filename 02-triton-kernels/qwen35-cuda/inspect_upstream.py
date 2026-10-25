#!/usr/bin/env python3
"""Export actually compiled upstream PTX for source-level comparison."""

import argparse
from pathlib import Path

import torch

from qwen35_lab import upstream
from qwen35_lab.cases import make_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("op", choices=["cache", "dynamic", "delta"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    case = make_case({"cache": "cache_decode", "dynamic": "dynamic_fp8", "delta": "delta_spec"}[args.op], dtype=torch.float32)
    if args.op == "cache":
        case.args["key_cache"] = case.args["key_cache"].to(torch.float8_e4m3fn)
        case.args["value_cache"] = case.args["value_cache"].to(torch.float8_e4m3fn)
    case.run("upstream")
    source = upstream.DELTA_SOURCE if args.op == "delta" else upstream.CACHE_SOURCE
    name = {"cache": "reshape_and_cache_kernel_flash", "dynamic": "_reshape_cache_per_token_head", "delta": "fused_sigmoid_gating_delta_rule_update_kernel"}[args.op]
    kernel = upstream._kernel(source, name)
    while not hasattr(kernel, "device_caches") and hasattr(kernel, "fn"):
        kernel = kernel.fn
    if not hasattr(kernel, "device_caches"):
        raise RuntimeError("Triton introspection API changed; inspect the installed JITFunction cache API")
    entries = kernel.device_caches[torch.cuda.current_device()][0]
    ptx = "\n\n".join(entry.asm["ptx"] for entry in entries.values())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(ptx)
        print(args.output)
    else:
        print(ptx)


if __name__ == "__main__":
    main()
