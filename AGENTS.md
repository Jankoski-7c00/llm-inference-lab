# Repository Guidelines

## Project Structure & Module Organization

This unified learning lab contains documentation, quarterly plans in `learning-plans/`, and experiment scaffolding. Use `01-vllm-runtime-trace/` for runtime traces, `02-triton-kernels/` for kernel experiments, `03-e2e-benchmark/` for end-to-end measurements, `04-compiler-notes/` for compiler notes, and `05-model-cards/` for model architecture cards. Store shared diagrams in `assets/`. Keep each kernel experiment in its own subdirectory and copy `02-triton-kernels/_template/REPORT.md` for its report. `third_party/vllm/` is a pinned upstream submodule.

## Build, Test, and Development Commands

Run these commands from the repository root:

- `git submodule update --init --recursive`: download dependencies at their recorded commits.
- `git submodule status`: inspect the checked-out dependency revision.
- `bash scripts/setup_env.sh`: initialize Python 3.12 and install the four learning dependencies; repeat runs use `uv.lock`.
- `uv run python`: run Python in the project-local `.venv/`.
- `git diff --check`: catch whitespace errors before committing.

The root `pyproject.toml` and `uv.lock` manage IPython, NumPy, PyTorch, and Transformers for CPU learning. Commit these files and `.python-version`; exclude `.venv/`. No repository-wide build, test runner, formatter, or linter is configured. Document exact setup and execution commands in each experiment's README and report. Treat changes to the vLLM submodule revision as explicit dependency updates.

## Coding Style & Naming Conventions

Preserve the numbered, topic-based directory structure. Use descriptive experiment names such as `reduction/` or `paged-decode-attention/`. Follow existing Markdown heading, table, and checklist conventions; retain Chinese explanations and standard English technical terms where appropriate. For new Python/Triton code, use four-space indentation, `snake_case` functions and variables, and uppercase constants. These are contributor conventions; no automated style enforcement currently exists.

## Testing Guidelines

No test framework or coverage threshold exists yet. Validate every kernel against PyTorch or another known-correct reference before reporting performance. Document shapes, dtypes, numerical tolerances, and failure cases. If introducing pytest, use `test_*.py` names and record the runnable command. Reports must include hardware, software versions, reproduction commands, and the commit. End-to-end records must also specify workload, concurrency, TTFT, TPOT, throughput, peak memory, units, and statistics. Label unrun experiments clearly.

## Commit & Pull Request Guidelines

Existing commits use concise, descriptive subjects, such as `Add vLLM v0.31.0 as a pinned submodule`; no strict prefix scheme is established. Keep changes focused. PRs should explain the affected experiment or note, link relevant issues when available, and state validation performed. Include plots or diagrams when they substantiate conclusions.

## Security & Data Hygiene

Use public models and open-source materials only. Exclude employer code, private data, model weights, and raw profiling artifacts covered by `.gitignore`. Publish concise results with reproducible evidence; never invent measurements or mark planned milestones complete without evidence.
