#!/usr/bin/env bash
set -euo pipefail
qwen35_lab_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
qwen35_repo_root="$(cd -- "$qwen35_lab_root/../.." && pwd)"
qwen35_uv="$(command -v uv || true)"
if [[ -z "$qwen35_uv" && -x "$qwen35_repo_root/.venv/bin/uv" ]]; then
    qwen35_uv="$qwen35_repo_root/.venv/bin/uv"
fi
if [[ -z "$qwen35_uv" ]]; then
    printf '%s\n' 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2
    exit 1
fi
if ! command -v nvcc >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1; then
    printf '%s\n' 'nvcc and a compatible C++ compiler are required.' >&2
    exit 1
fi
if [[ ! -x "$qwen35_repo_root/.venv/bin/python" ]]; then
    "$qwen35_uv" venv --python "${QWEN35_PYTHON:-python3}" --system-site-packages "$qwen35_repo_root/.venv"
fi
"$qwen35_repo_root/.venv/bin/python" -c 'import torch; assert torch.cuda.is_available(), "Provide a CUDA-capable PyTorch installation first"'
"$qwen35_uv" pip install --python "$qwen35_repo_root/.venv/bin/python" -r "$qwen35_lab_root/requirements.txt"
"$qwen35_repo_root/.venv/bin/python" "$qwen35_lab_root/run.py" build
