"""Prepare isolated CUDA blanks; the working solution stays untouched."""

import argparse

from .extension import LAB_ROOT

EXERCISES = {
    "cache": (
        "int64_t block = slot / c.block_size, off = slot % c.block_size;",
        "int64_t block = /* TODO: page number */, off = /* TODO: slot within page */;",
        "Write page quotient/remainder; then derive the NHD address from strides.",
        "tests/test_cache.py",
    ),
    "dynamic": (
        "float sk=fmaxf(km/qmax,1e-6f), sv=fmaxf(vm/qmax,1e-6f);",
        "float sk=/* TODO: K scale */, sv=/* TODO: V scale */;",
        "Use independent K/V absmax and a strictly positive minimum scale.",
        "tests/test_cache.py",
    ),
    "softmax": (
        "float next=valid ? fmaxf(maxima[r],m) : 0;",
        "float next=/* TODO: online maximum; padded rows must remain finite */;",
        "Merge previous and current tile maxima before computing alpha and P.",
        "tests/test_attention.py",
    ),
    "delta": (
        "for(int u=0;u<8;++u) {s[u]+=delta*kk[u];o+=s[u]*qq[u];}",
        "for(int u=0;u<8;++u) {s[u]+=/* TODO: outer-product update */;o+=/* TODO: output */;}",
        "The output uses the updated state, after the decayed-state prediction.",
        "tests/test_delta.py",
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=EXERCISES)
    parser.add_argument("--answer", action="store_true")
    args = parser.parse_args()
    original, blank, hint, test = EXERCISES[args.task]
    if args.answer:
        print(original)
        return
    source = (LAB_ROOT / "csrc" / "kernels.cu").read_text()
    if source.count(original) != 1:
        raise RuntimeError("exercise anchor changed; update the worksheet before generating")
    target = LAB_ROOT / ".cache" / "exercises" / args.task / "kernels.cu"
    if target.exists():
        raise FileExistsError(f"preserving existing student work: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.replace(original, blank))
    print(hint)
    print("Edit:", target)
    print("Then run from repository root:")
    print(f'QWEN35_KERNEL_SOURCE="{target}" .venv/bin/python -m pytest "{LAB_ROOT / test}" -q')


if __name__ == "__main__":
    main()
