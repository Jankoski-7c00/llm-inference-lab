#!/usr/bin/env python3
"""Collect a real report, or fail with the profiler's original error."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from qwen35_lab.cases import PRESETS
from qwen35_lab.extension import LAB_ROOT
from qwen35_lab.measure import metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", choices=["nsys", "ncu"])
    parser.add_argument("--case", choices=PRESETS, default="cache_decode")
    parser.add_argument("--impl", default="optimized")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--output-dir", type=Path, default=LAB_ROOT / "profiles")
    parser.add_argument("--set", choices=["basic", "full"], default="basic")
    args = parser.parse_args()
    if not shutil.which(args.tool):
        parser.error(f"{args.tool} not found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir.resolve() / f"{args.case}_{args.impl}_{args.dtype}"
    report = stem.with_suffix(f".{args.tool}-rep")
    if report.exists():
        parser.error(f"report already exists: {report}; use another output directory")
    program = [sys.executable, str(LAB_ROOT / "run.py"), "profile", "--case", args.case,
               "--impl", args.impl, "--dtype", args.dtype, "--iters", "3"]
    if args.tool == "nsys":
        command = ["nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                   "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "-o", str(stem), *program]
    else:
        command = ["ncu", "--profile-from-start", "off", "--nvtx", "--nvtx-include", "qwen35_target/",
                   "--launch-count", "8", "--set", args.set, "-o", str(stem), *program]
    print("Running:", " ".join(command), flush=True)
    completed = subprocess.run(command)
    success = completed.returncode == 0 and report.exists() and report.stat().st_size > 0
    manifest = dict(tool=args.tool, command=command, status="collected" if success else "failed",
                    report=str(report), exit_code=completed.returncode, metadata=metadata())
    stem.with_suffix(f".{args.tool}.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if not success:
        print("No successful report recorded. ERR_NVGPUCTRPERM requires platform/driver profiling permission; keep the Compute milestone pending.", file=sys.stderr)
        sys.exit(completed.returncode or 1)
    print("Report:", report)


if __name__ == "__main__":
    main()
