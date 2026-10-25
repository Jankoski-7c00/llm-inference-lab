import functools
import hashlib
import os
from pathlib import Path

import torch

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parents[1]


@functools.lru_cache(maxsize=1)
def extension():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the CUDA implementations")
    import ninja
    from torch.utils.cpp_extension import load

    os.environ["PATH"] = str(ninja.BIN_DIR) + os.pathsep + os.environ["PATH"]
    os.environ.setdefault("MAX_JOBS", "1")
    custom = os.environ.get("QWEN35_KERNEL_SOURCE")
    kernel_source = Path(custom).resolve() if custom else LAB_ROOT / "csrc" / "kernels.cu"
    tag = "_"+hashlib.sha256(str(kernel_source).encode()).hexdigest()[:8] if custom else ""
    build = LAB_ROOT / ".cache" / ("extension"+tag)
    build.mkdir(parents=True, exist_ok=True)
    return load(
        name="qwen35_cuda_lab"+tag,
        sources=[str(LAB_ROOT / "csrc" / "bindings.cpp"), str(kernel_source)],
        build_directory=str(build),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=os.environ.get("QWEN35_VERBOSE_BUILD") == "1",
    )
