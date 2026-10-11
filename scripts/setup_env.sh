#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/' >&2
    exit 1
fi
cd -- "$project_root"
export UV_PROJECT_ENVIRONMENT="$project_root/.venv"

if [[ ! -f pyproject.toml ]]; then
    uv init --bare --no-package --vcs none --no-workspace --python 3.12 --name inference-lab
fi
if [[ ! -f .python-version ]]; then
    uv python pin 3.12
fi
if [[ ! -f uv.lock ]]; then
    uv add ipython numpy torch transformers
else
    uv sync --locked
fi

"$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY'
import sys
from importlib.metadata import version
import IPython
import numpy as np
import torch
import transformers
assert sys.version_info[:2] == (3, 12), sys.version
assert np.array([1, 2, 3]).sum() == 6
assert torch.tensor([1, 2, 3], device='cpu').sum().item() == 6
print('Python:', sys.version.split()[0])
print('Interpreter:', sys.executable)
for name in ('ipython', 'numpy', 'torch', 'transformers'):
    print(f'{name}: {version(name)}')
print('Imports and NumPy/PyTorch CPU checks passed.')
PY
printf '\nEnvironment ready.\nActivate: source "%s/.venv/bin/activate"\nRun: uv run --project "%s" python\n' "$project_root" "$project_root"
