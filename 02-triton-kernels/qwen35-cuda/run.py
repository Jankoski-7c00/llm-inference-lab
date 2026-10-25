#!/usr/bin/env python3
"""Run from any directory using the repository's Python interpreter."""

import argparse
import csv
import json
from pathlib import Path

import torch

from qwen35_lab.cases import PRESETS, make_case
from qwen35_lab.extension import extension
from qwen35_lab.measure import benchmark, metadata, profile_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "build", "check", "bench", "profile"])
    parser.add_argument("--case", choices=PRESETS, default="cache_decode")
    parser.add_argument("--impl", choices=["baseline", "fused", "shared", "tiled", "optimized", "simt", "wmma", "async", "upstream"], default="optimized")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--mode", choices=["graph", "eager"], default="graph")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.warmup, args.iters, args.repeats) <= 0:
        parser.error("warmup, iters and repeats must be positive")
    if args.command == "list":
        for name, (op, description) in PRESETS.items():
            print(f"{name:20} {op:10} {description}")
        return
    if args.command == "build":
        extension()
        print("CUDA extension built")
        return
    case = make_case(args.case, dtype=getattr(torch, args.dtype), seed=args.seed)
    if args.command == "check":
        case.check(args.impl)
        torch.cuda.synchronize()
        result = dict(case=case.name, impl=args.impl, dtype=args.dtype, status="passed")
    elif args.command == "profile":
        profile_target(case, args.impl, warmup=args.warmup, iters=args.iters)
        result = dict(case=case.name, impl=args.impl, dtype=args.dtype, status="target completed; inspect profiler output separately")
    else:
        if args.impl == "upstream":
            parser.error("upstream launch adapters are correctness oracles; their allocations are not a comparable microbenchmark")
        result = benchmark(case, args.impl, warmup=args.warmup, iters=args.iters, repeats=args.repeats, mode=args.mode)
        result.update(dtype=args.dtype, seed=args.seed)
    record = dict(result=result, metadata=metadata())
    print(json.dumps(record, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix == ".csv":
            exists = args.output.exists() and args.output.stat().st_size > 0
            with args.output.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(result))
                if not exists:
                    writer.writeheader()
                writer.writerow(result)
            args.output.with_suffix(".metadata.json").write_text(json.dumps(record["metadata"], indent=2))
        else:
            args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    main()
