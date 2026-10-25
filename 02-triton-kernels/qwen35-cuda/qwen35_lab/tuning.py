"""Prepare the two single-factor candidates without replacing the course baseline.

Run after predicting the result and attempting the edit yourself. Generated
sources and patches are references, not evidence of performance or mastery.
"""

import argparse
import difflib

from .extension import LAB_ROOT


CANDIDATES = {
    "cache": (
        "cache_cta256",
        "cache_kernel<T,Dst,true><<<dim3(t,(h*d+511)/512),128,0,stream>>>",
        "cache_kernel<T,Dst,true><<<dim3(t,(h*d+1023)/1024),256,0,stream>>>",
        "tests/test_cache.py",
        "cache_decode,cache_prefill,cache_head_major",
    ),
    "delta": (
        "delta_rows2",
        "if(dk<=128) delta_shared_kernel<scalar_t,state_t,4,4><<<dim3((dv+15)/16,hv,cu.numel()-1),128,0,stream>>>",
        "if(dk<=128) delta_shared_kernel<scalar_t,state_t,4,2><<<dim3((dv+7)/8,hv,cu.numel()-1),128,0,stream>>>",
        "tests/test_delta.py",
        "delta_decode,delta_spec",
    ),
}


def prepare(task):
    name, original, replacement, _, _ = CANDIDATES[task]
    source_path = LAB_ROOT / "csrc" / "kernels.cu"
    source = source_path.read_text()
    if source.count(original) != 1:
        raise RuntimeError("candidate anchor changed; review the intended single-factor edit")
    target = LAB_ROOT / ".cache" / "tuning" / name / "kernels.cu"
    patch = target.with_suffix(".patch")
    if target.exists() or patch.exists():
        raise FileExistsError(f"preserving existing candidate: {target.parent}")
    candidate = source.replace(original, replacement)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(candidate)
    patch.write_text("".join(difflib.unified_diff(
        source.splitlines(keepends=True), candidate.splitlines(keepends=True),
        fromfile="csrc/kernels.cu", tofile=f"{name}/kernels.cu")))
    return target, patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=CANDIDATES)
    args = parser.parse_args()
    target, patch = prepare(args.task)
    name, _, _, test, cases = CANDIDATES[args.task]
    print("Source:", target)
    print("Patch:", patch)
    print("Run from repository root; use this source for tests, timing and profiling:")
    print(f'export QWEN35_KERNEL_SOURCE="{target}"')
    print(f'.venv/bin/python -m pytest "{LAB_ROOT / test}" -q')
    print(f'.venv/bin/python "{LAB_ROOT / "suite.py"}" --cases {cases} --dtype bfloat16 '
          f'--mode graph --iters 20 --repeats 5 --output "{LAB_ROOT / "outputs" / name / "graph.csv"}"')


if __name__ == "__main__":
    main()
