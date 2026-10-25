import importlib.metadata
import hashlib
import platform
import statistics
import subprocess

import torch

from .extension import LAB_ROOT, REPO_ROOT


def metadata():
    def git(*args):
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    packages = {}
    for name in ("torch", "triton", "ninja", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    data = dict(python=platform.python_version(), packages=packages,
                torch_cuda=torch.version.cuda, repository_commit=git("rev-parse", "HEAD"),
                dirty=bool(git("status", "--porcelain")),
                upstream_commit=git("-C", "third_party/vllm", "rev-parse", "HEAD"))
    data["source_sha256"] = {str(path.relative_to(LAB_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in [*sorted((LAB_ROOT/"csrc").glob("*")),
                                          *sorted((LAB_ROOT/"qwen35_lab").glob("*.py"))] if path.is_file()}
    if torch.cuda.is_available():
        data.update(gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability())
    return data


@torch.inference_mode()
def benchmark(case, impl, *, warmup=5, iters=20, repeats=5, mode="graph"):
    case.check(impl)
    for _ in range(warmup):
        case.reset()
        result = case.run(impl)
    torch.cuda.synchronize()
    pairs = [(torch.cuda.Event(enable_timing=True, external=mode == "graph"),
              torch.cuda.Event(enable_timing=True, external=mode == "graph")) for _ in range(iters)]
    graph = None
    if mode == "graph":
        graph = torch.cuda.CUDAGraph()
        # Reset copies are graph nodes OUTSIDE each timed interval. External
        # event nodes measure GPU execution without Python submission gaps.
        with torch.cuda.graph(graph):
            for start, end in pairs:
                case.reset()
                start.record()
                result = case.run(impl)
                end.record()
        torch.cuda.synchronize()
    groups = []
    all_samples = []
    for _ in range(repeats):
        if graph is not None:
            graph.replay()
        else:
            for start, end in pairs:
                case.reset()
                start.record()
                result = case.run(impl)
                end.record()
        torch.cuda.synchronize()
        samples = [start.elapsed_time(end)*1000 for start, end in pairs]
        all_samples.extend(samples)
        groups.append(statistics.mean(samples))
    case.reset()
    ordered = sorted(all_samples)
    return dict(case=case.name, op=case.op, impl=impl, mode=mode,
                target_interval_us=statistics.median(groups),
                p10_us=ordered[int((len(ordered)-1)*0.1)],
                p90_us=ordered[int((len(ordered)-1)*0.9)],
                warmup=warmup, iters=iters, repeats=repeats,
                reset_in_timing=False,
                note="eager intervals include CPU submission gaps" if mode == "eager" else "external graph event intervals; reset copies excluded; warm-state measurement")


@torch.inference_mode()
def profile_target(case, impl, *, warmup=5, iters=3):
    case.check(impl)
    for _ in range(warmup):
        case.reset()
        result = case.run(impl)
    case.reset()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iters):
        case.reset()
        torch.cuda.nvtx.range_push("qwen35_target")
        result = case.run(impl)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    return result
