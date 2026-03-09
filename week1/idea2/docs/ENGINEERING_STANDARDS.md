# Engineering Standards (Idea2)

**Created**: 2026-02-04

This document defines project-wide engineering standards for `week1/idea2`, based on widely-used, authoritative public standards and the current repo state.

## 1. Scope

- Language: Python 3.10+
- ML stack: PyTorch
- Experiments/training: (SB3-style usage exists in scripts)
- Target: research code that remains reproducible, debuggable, and reviewable

## 2. Primary References (authoritative)

- PEP 8 (Python style): https://peps.python.org/pep-0008/
- PEP 257 (docstrings): https://peps.python.org/pep-0257/
- PEP 484 (type hints): https://peps.python.org/pep-0484/
- PEP 526 (variable annotations): https://peps.python.org/pep-0526/
- Python typing docs: https://docs.python.org/3/library/typing.html
- pytest docs: https://docs.pytest.org/en/stable/
- pytest good practices: https://docs.pytest.org/en/stable/explanation/goodpractices.html
- pytest fixtures: https://docs.pytest.org/en/stable/how-to/fixtures.html
- pytest markers: https://docs.pytest.org/en/stable/how-to/mark.html
- pytest parametrization: https://docs.pytest.org/en/stable/how-to/parametrize.html
- pytest assertions: https://docs.pytest.org/en/stable/how-to/assert.html
- Ruff docs: https://docs.astral.sh/ruff/
- Black docs: https://black.readthedocs.io/en/stable/
- PyTorch Modules note: https://pytorch.org/docs/stable/notes/modules.html
- PyTorch Reproducibility note: https://pytorch.org/docs/stable/notes/randomness.html
- Python Packaging User Guide (pyproject): https://packaging.python.org/en/latest/tutorials/packaging-projects/
- Python venv: https://docs.python.org/3/library/venv.html

## 3. Repo Reality (current state)

- The repo contains `docs/CODE_STYLE.md` and uses it as the local style authority.
- As of a repository scan on 2026-02-04, no tool config files were found at the repo root (e.g. `pyproject.toml`, `ruff.toml`, `setup.cfg`, `.pre-commit-config.yaml`).
- Tests under `tests/` currently look like runnable scripts (not `pytest`-collected unit tests).

If a rule below conflicts with `docs/CODE_STYLE.md`, follow `docs/CODE_STYLE.md`.

## 4. Baseline Coding Rules (Python)

- Follow PEP 8 for naming, whitespace, and imports.
- Prefer explicit code over clever code (PEP 20): https://peps.python.org/pep-0020/
- Keep files and identifiers ASCII-only unless the file already uses non-ASCII and it is justified.

### Imports

- Use absolute imports within this repo (project-root based).
- Import groups in this order (PEP 8): standard lib, third-party, local.
- One import per line; avoid wildcard imports.

### Functions and error handling

- Validate tensor shapes/dtypes early when shape mismatch is likely.
- Never use empty `except:` blocks.
- Raise specific exceptions; keep error messages actionable.

## 5. Typing Rules

- New/edited functions should have type hints for parameters and return values.
- Use `typing` constructs from the standard library (PEP 484/526):
  - `Dict[str, torch.Tensor]`, `Optional[torch.Tensor]`, etc.
- Do not use type hints as runtime checks; they are for tooling and readability.

## 6. PyTorch Module Rules

- `nn.Module` design follows PyTorch guidance: https://pytorch.org/docs/stable/notes/modules.html

Required conventions for new `nn.Module`s:

- Define all submodules in `__init__`; avoid creating layers inside `forward`.
- Keep `forward()` pure (no hidden global state mutations).
- Document tensor shapes in docstrings (`(B, N, 3)`, etc.).
- Use `register_buffer` for non-parameter tensors that should be saved in `state_dict`.
- Be explicit about expected `dtype` and device assumptions where relevant.

Reproducibility guidance (opt-in, because it may reduce performance):

- Seeding: `random.seed`, `np.random.seed`, `torch.manual_seed`.
- Determinism: `torch.use_deterministic_algorithms(True)` where feasible.
- cuDNN benchmarking: `torch.backends.cudnn.benchmark = False` for repeatable runs.
  - Reference: https://pytorch.org/docs/stable/notes/randomness.html

## 7. Testing Rules

### Recommended direction (pytest)

- Prefer pytest-style unit tests and default discovery conventions:
  - collection: starts at current directory or `testpaths` if configured
  - files: `test_*.py` or `*_test.py`
  - functions/methods: `test_*`
  - classes: `Test*` (no `__init__`)
  - reference: https://docs.pytest.org/en/stable/explanation/goodpractices.html

Recommended pytest hygiene (when/if you adopt pytest config):

- Register custom markers to avoid silent typos (and enable strict markers): https://docs.pytest.org/en/stable/how-to/mark.html
- Prefer fixtures for setup/teardown (yield-fixtures for cleanup): https://docs.pytest.org/en/stable/how-to/fixtures.html
- Use `pytest.approx` for float comparisons when appropriate: https://docs.pytest.org/en/stable/how-to/assert.html

### Practical rules for ML unit tests

- Tests should assert:
  - tensor shapes
  - value ranges (e.g., probabilities in [0, 1])
  - invariants (e.g., fusion weights sum to 1)
- Set seeds in tests when randomness affects assertions.
- Prefer CPU tensors for unit tests unless the test is explicitly GPU-only (CUDA nondeterminism can make tests flaky).
- Avoid flaky tests; if CUDA nondeterminism exists, run unit tests on CPU tensors.

## 8. Tooling (format/lint)

This repo documents `ruff format .` / `ruff check .` / `black .` in `docs/CODE_STYLE.md`, but does not include configuration files.

Recommended standard commands (when the tools are installed):

- `ruff format .`
- `ruff check .`

Optional (not required by the current repo):

- Static type checking:
  - mypy: https://mypy.readthedocs.io/
  - pyright: https://microsoft.github.io/pyright/

References:

- Ruff formatter: https://docs.astral.sh/ruff/formatter/
- Ruff config: https://docs.astral.sh/ruff/configuration/
- Black: https://black.readthedocs.io/en/stable/

## 9. Project Structure & Documentation

- Put long-form technical notes in repo root or `docs/` and cross-link them.
- Every new module should include a short module docstring and a minimal `test_*()` function if the repo continues the current script-style testing approach.
- Prefer small, composable modules under `networks/` over monolith classes.
