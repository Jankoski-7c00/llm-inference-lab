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

EXTRA_BLANKS = {
    "cache": [
        ("auto kk = *reinterpret_cast<const Pack4<T>*>(k+t*ks.a+h*ks.b+d);",
         "auto kk = /* TODO: aligned Pack4 K load using token/head strides */;"),
        ("auto vv = *reinterpret_cast<const Pack4<T>*>(v+t*vs.a+h*vs.b+d);",
         "auto vv = /* TODO: aligned Pack4 V load using token/head strides */;"),
        ("*reinterpret_cast<Pack4<Dst>*>(kc+cache_offset(kd,slot,h,d)) = ko;",
         "/* TODO: store ko to the paged K cache as an aligned Pack4 */"),
        ("*reinterpret_cast<Pack4<Dst>*>(vc+cache_offset(vd,slot,h,d)) = vo;",
         "/* TODO: store vo to the paged V cache as an aligned Pack4 */"),
    ],
    "delta": [
        ("for(int u=0;u<8;++u) {s[u]*=decay;pred+=s[u]*kk[u];}",
         "for(int u=0;u<8;++u) {s[u]*=/* TODO: state decay */;pred+=/* TODO: prediction */;}"),
        ("float delta=(as_float(v[(int64_t(t)*hv+h)*dv+vr])-pred)*beta_t, o=0;",
         "float delta=/* TODO: gated residual from V minus the reduced prediction */, o=0;"),
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=EXERCISES)
    parser.add_argument("--answer", action="store_true")
    args = parser.parse_args()
    original, blank, hint, test = EXERCISES[args.task]
    replacements = [(original, blank), *EXTRA_BLANKS.get(args.task, [])]
    if args.answer:
        print("\n".join(original for original, _ in replacements))
        return
    source = (LAB_ROOT / "csrc" / "kernels.cu").read_text()
    for original, blank in replacements:
        if source.count(original) != 1:
            raise RuntimeError("exercise anchor changed; update the worksheet before generating")
        source = source.replace(original, blank)
    target = LAB_ROOT / ".cache" / "exercises" / args.task / "kernels.cu"
    if target.exists():
        raise FileExistsError(f"preserving existing student work: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)
    print(hint)
    print("Edit:", target)
    print("Then run from repository root:")
    print(f'QWEN35_KERNEL_SOURCE="{target}" .venv/bin/python -m pytest "{LAB_ROOT / test}" -q')


if __name__ == "__main__":
    main()
