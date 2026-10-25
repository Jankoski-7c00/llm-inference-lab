#!/usr/bin/env python3
"""Reproduce the curated correctness and performance matrix."""

import argparse
import csv
import json
from pathlib import Path

import torch

from qwen35_lab.cases import PRESETS, make_case
from qwen35_lab.measure import benchmark, metadata


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="cache_prefill,dynamic_int8,encoder_short,delta_decode")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--include-upstream", action="store_true")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--mode", choices=["graph", "eager"], default="graph")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs" / "suite.csv")
    args = parser.parse_args()
    names = list(PRESETS) if args.cases == "all" else args.cases.split(",")
    if any(name not in PRESETS for name in names):
        parser.error("unknown case")
    if min(args.iters, args.repeats) <= 0:
        parser.error("iters/repeats must be positive")
    if args.include_upstream and not args.check_only:
        parser.error("upstream adapters are correctness oracles; use --check-only")
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in names:
        case = make_case(name, dtype=getattr(torch, args.dtype))
        impls = ["baseline", "optimized"]
        if case.op == "attention":
            impls = ["baseline", "simt"] if args.dtype == "float32" else ["baseline", "simt", "wmma", "async"]
        elif case.op == "delta":
            impls = ["baseline", "fused", "shared", "tiled", "optimized"]
            if args.include_upstream:
                impls.append("upstream")
        elif args.include_upstream:
            impls += ["upstream"]
        for impl in impls:
            if args.check_only:
                case.check(impl)
                torch.cuda.synchronize()
                row = dict(case=name, impl=impl, status="passed", dtype=args.dtype)
                print(f"PASS {name}/{impl}", flush=True)
            else:
                row = benchmark(case, impl, iters=args.iters, repeats=args.repeats, mode=args.mode)
                row.update(dtype=args.dtype, seed=17)
                print(f"{name}/{impl}: {row['target_interval_us']:.3f} us ({args.mode})", flush=True)
            rows.append(row)
            # Preserve partial progress if a later case fails.
            with args.output.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata(), indent=2)+"\n")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
