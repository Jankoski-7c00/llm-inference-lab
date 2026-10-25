import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def deterministic_inputs():
    torch.manual_seed(17)


@pytest.fixture(scope="session")
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; mathematical reference tests still run")
    from qwen35_lab.extension import extension

    extension()
    return torch.device("cuda")
