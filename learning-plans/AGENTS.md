# Repository Guidelines

## Project Structure & Module Organization

This subtree of inference-lab archives sanitized personal learning plans as Markdown documents. The local `README.md` indexes the available quarters. `2026Q4/` contains the quarter overview (`00_季度总览与执行机制.md`) and five numbered topic plans covering vLLM, CUDA/Triton, compilers, Attention/model tracking, and heterogeneous/distributed systems. Executable experiments belong in the parent inference-lab project; link to their evidence from plans.

For a new quarter, use a directory such as `2027Q1/`, start with a `00_` overview, and add topic documents following the existing `NN_主题.md` pattern. Update the root index when adding or renaming documents.

## Development & Validation Commands

Run these commands from the parent Git repository root:

- `git diff --check`: check edited documents for whitespace errors.
- `git diff --stat`: review the scope of a proposed change.
- `rg --files learning-plans -g '*.md'`: list documents when checking indexes and links.

These plans have no build process or automated test suite. The parent project manages its learning environment with `scripts/setup_env.sh`, `pyproject.toml`, and `uv.lock`; do not create a separate Python environment here. Preview changed Markdown before submitting it and manually verify relative links.

## Writing Style & Naming Conventions

Write primarily in Chinese, preserving established English technical terms such as TTFT, TPOT, Runtime, and Kernel. Use one top-level title per document, numbered sections where the existing document does, tables for comparable schedules or criteria, and `- [ ]` checkboxes for unfinished milestones. Keep filenames descriptive and ordered with two-digit prefixes. Match surrounding list indentation and heading levels. Preserve the public-information disclaimer at the start of each quarterly document.

## Review Guidelines

Check dates against the quarter, ensure time allocations and percentages agree with the overview, and keep shared milestones consistent across topic plans. Distinguish planned work from completed results. Mark a milestone complete only when supporting evidence exists, and link the associated report or experiment where possible. No coverage target or testing framework applies to this documentation-only repository.

## Commit & Pull Request Guidelines

The single existing commit uses a descriptive subject: `Add 2026 Q4 learning plan (sanitized, 6 docs)`. Follow that concise style; no formal commit-prefix convention is established. Keep changes scoped to a quarter or related planning update. PRs should explain changed goals, schedules, or acceptance criteria and describe consistency and link checks. Link relevant issues if available.

## Privacy & Content Boundaries

Use public technical sources and generalized industry descriptions. Exclude employer names, non-public project identifiers, proprietary code, and internal data. Preserve the distinction between personal plans and demonstrated accomplishments.
