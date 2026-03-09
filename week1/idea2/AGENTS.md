# Agent Instructions (week1/idea2)

UAV multi-modal fusion + RL experiments (PyTorch + SB3-style scripts). Python 3.10+.

Key dirs: `networks/`, `envs/`, `tests/` (script-style), `experiments/`, `docs/`.

## Local Sources of Truth

- Code style (authoritative): `docs/CODE_STYLE.md`
- Engineering standards + external references: `docs/ENGINEERING_STANDARDS.md`

If guidance conflicts, follow `docs/CODE_STYLE.md`.

## Cursor / Copilot Rules

No `.cursorrules`, `.cursor/rules/`, or `.github/copilot-instructions.md` found.

## Build / Lint / Test Commands

No packaging/CI config (no `pyproject.toml`, no `Makefile`). Use the commands below.

### Environment Setup (recommended)

This project is currently developed using the conda environment `sb3_idea2`.

```bash
python -m venv .venv
source .venv/bin/activate

# Install runtime deps (versions are project-dependent; pin if you add a lockfile later)
python -m pip install -U pip
python -m pip install torch gymnasium stable-baselines3 numpy
```

### Format / Lint

Preferred (as documented in `docs/CODE_STYLE.md`):

```bash
ruff format .
ruff check .
```

Alternative: `black .`

### “Build” / Smoke Check

There is no build step; use a syntax/import smoke check:

```bash
python -m compileall .
```

### Tests (current reality = script-style)

Run a single test script:

```bash
python tests/integration/test_basic_setup.py
```

Run all current test scripts:

```bash
python tests/integration/test_basic_setup.py
python tests/training/train_minimal_test.py
```

Important: `tests/integration/test_basic_setup.py` and `tests/training/train_minimal_test.py` import modules that are currently missing in this repo:

- `envs/uav_multimodal_env.py`
- `networks/uav_multimodal_extractor.py`

Expect these scripts to fail until those files are added or imports are corrected.

### Pytest (optional / future-facing)

If you add pytest unit tests under `tests/unit/`:

```bash
pytest
pytest tests/unit/test_file.py
pytest tests/unit/test_file.py::test_name
pytest -k "keyword" tests/unit
```

## Runnable Module Self-Tests (existing pattern)

Many modules include local `test_*()` + `if __name__ == "__main__":`. Examples:

```bash
python networks/dynamic_weighting_layer.py
python envs/simple_navigation_env.py
python envs/simple_2d_env.py
```

## Code Style (agent-facing summary)

Follow `docs/CODE_STYLE.md` (authoritative). Highlights:

### Imports

- Order: standard library, third-party, local modules.
- Use absolute imports from repo root (e.g., `from networks...`, `from envs...`).
- One import per line; avoid wildcard imports.

### Formatting

- Indent: 4 spaces.
- Keep lines reasonably short (local doc targets <100 chars).
- Run `ruff format .` before finishing a change.

### Types

- Add type hints for new/edited public functions (params + returns).
- Prefer standard `typing` constructs (`Dict`, `Optional`, etc.).
- Do not use typing to hide errors (avoid `Any` unless unavoidable).

### Naming

- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_CASE`
- Tests: `test_*`

### Docstrings

- Use triple double quotes.
- Use Google-style sections (`Args:`, `Returns:`, `Raises:`) as in `docs/CODE_STYLE.md`.
- For tensor code, document shapes explicitly (`(B, N, 3)`, `(B, 3, H, W)`).

### Error Handling

- Never use bare `except:` or empty handlers.
- Validate inputs where shape/type errors are likely; raise `TypeError`/`ValueError` with actionable messages.
- Don’t swallow exceptions in training loops; re-raise after logging.

### PyTorch Module Conventions

- Define layers in `__init__`, not inside `forward()`.
- Keep `forward()` free of side effects.
- Use deterministic practices only when needed (see `docs/ENGINEERING_STANDARDS.md`).

## Testing Expectations for New Code

If you add a new module under `networks/` or `envs/`, include a minimal `test_*()` that checks shapes, ranges, and invariants (CPU-only unless explicitly GPU-specific).

Workflow: read `docs/CODE_STYLE.md` first; make the smallest correct change; run `ruff format .` + `ruff check .`; run the closest module self-test or script test.
