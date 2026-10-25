"""CPU regressions for report collection and isolated student/candidate sources."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from qwen35_lab import exercise, tuning
from qwen35_lab.extension import LAB_ROOT
from qwen35_lab.measure import metadata


@pytest.fixture
def profile_tool():
    spec = importlib.util.spec_from_file_location("lab_profile_tool", LAB_ROOT / "profile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("tool,suffix", [("ncu", ".ncu-rep"), ("ncu", ".ncu-repz"), ("nsys", ".nsys-rep")])
def test_profile_records_actual_report_and_refuses_overwrite(profile_tool, monkeypatch, tmp_path, tool, suffix):
    monkeypatch.setattr(sys, "argv", ["profile.py", tool, "--dtype", "bfloat16", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(profile_tool.shutil, "which", lambda _: "/fake/profiler")
    monkeypatch.setattr(profile_tool, "metadata", lambda: {})
    calls = []

    def collect(command):
        calls.append(command)
        Path(command[command.index("-o")+1]).with_suffix(suffix).write_bytes(b"test report")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(profile_tool.subprocess, "run", collect)
    profile_tool.main()
    manifest = json.loads(next(tmp_path.glob(f"*.{tool}.json")).read_text())
    assert manifest["status"] == "collected"
    assert manifest["report"].endswith(suffix)
    assert Path(manifest["report"]).read_bytes() == b"test report"
    with pytest.raises(SystemExit) as error:
        profile_tool.main()
    assert error.value.code == 2
    assert len(calls) == 1


@pytest.mark.parametrize("exit_code,content", [(0, None), (0, b""), (7, b"partial report")])
def test_profile_preserves_failures(profile_tool, monkeypatch, tmp_path, exit_code, content):
    monkeypatch.setattr(sys, "argv", ["profile.py", "ncu", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(profile_tool.shutil, "which", lambda _: "/fake/ncu")
    monkeypatch.setattr(profile_tool, "metadata", lambda: {})

    def collect(command):
        if content is not None:
            Path(command[command.index("-o")+1]).with_suffix(".ncu-repz").write_bytes(content)
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(profile_tool.subprocess, "run", collect)
    with pytest.raises(SystemExit) as error:
        profile_tool.main()
    assert error.value.code == (exit_code or 1)
    manifest = json.loads(next(tmp_path.glob("*.ncu.json")).read_text())
    assert manifest["status"] == "failed"
    assert manifest["exit_code"] == exit_code


@pytest.mark.parametrize("task", ["cache", "delta"])
def test_isolated_exercise_preserves_work(monkeypatch, tmp_path, task):
    original = (LAB_ROOT / "csrc" / "kernels.cu").read_text()
    (tmp_path / "csrc").mkdir()
    source = tmp_path / "csrc" / "kernels.cu"
    source.write_text(original)
    monkeypatch.setattr(exercise, "LAB_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["exercise", task])
    exercise.main()
    target = tmp_path / ".cache" / "exercises" / task / "kernels.cu"
    assert "TODO:" in target.read_text()
    assert source.read_text() == original
    target.write_text("student work")
    with pytest.raises(FileExistsError):
        exercise.main()
    assert target.read_text() == "student work"


@pytest.mark.parametrize("task", ["cache", "delta"])
def test_candidate_preserves_baseline_and_tracks_effective_source(monkeypatch, tmp_path, task):
    original = (LAB_ROOT / "csrc" / "kernels.cu").read_text()
    (tmp_path / "csrc").mkdir()
    source = tmp_path / "csrc" / "kernels.cu"
    source.write_text(original)
    monkeypatch.setattr(tuning, "LAB_ROOT", tmp_path)
    target, patch = tuning.prepare(task)
    _, before, after, _, _ = tuning.CANDIDATES[task]
    assert target.read_text() == original.replace(before, after)
    assert source.read_text() == original
    assert before in patch.read_text() and after in patch.read_text()
    monkeypatch.setenv("QWEN35_KERNEL_SOURCE", str(target))
    record = metadata()["kernel_source"]
    assert record == dict(path=str(target.resolve()), sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    with pytest.raises(FileExistsError):
        tuning.prepare(task)
