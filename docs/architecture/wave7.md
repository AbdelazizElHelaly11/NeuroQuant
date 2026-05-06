# Wave 7 — Packaging + docs

## Decision matrix

| ID | Item                                                 | Decision   |
| -- | ---------------------------------------------------- | ---------- |
| O1 | Project metadata in `pyproject.toml`                 | Implement  |
| O2 | `python -m build` produces a working wheel           | Implement  |
| O3 | Console-script entry point (`neuroquant` CLI)        | Implement  |
| O4 | README + per-wave architecture docs                  | Implement  |

## What shipped

### O1 · Full project metadata
- [`pyproject.toml`](../../pyproject.toml) `[project]` block extended with:
  - `description`, `readme = "README.md"`, `requires-python = ">=3.10"`, MIT license.
  - `keywords`, full `classifiers` list (Development Status :: 5 - Production/Stable, Python 3.10/3.11/3.12, etc.).
  - `dependencies` — runtime deps mirrored from `requirements.txt`, version floors aligned with what CI verifies.
  - `[project.optional-dependencies]`: `test`, `dev`, `xai` extras.
  - `[project.urls]` for Homepage / Documentation / Repository.

### O3 · Console script
- `[project.scripts]` registers `neuroquant = "main:main"` — installs `neuroquant.exe` (Windows) / `neuroquant` (POSIX) on `PATH` after `pip install`.
- `main.main()` already existed (argparse-driven entry that handles config loading, CLI overrides, pipeline execution); no main.py changes needed.

### O2 · Wheel build verified
- `python -m build --wheel` produces `dist/neuroquant-2.0.0-py3-none-any.whl`.
- Wheel structure verified:
  - Top-level modules: `config`, `data`, `main`, `models`, `quantization`, `tracking`, `utils`, `visualization`, `xai`.
  - `entry_points.txt`: `console_scripts: neuroquant = main:main`.
  - `METADATA`: name + version + description + license correctly populated.
- `pip install --no-deps dist/neuroquant-2.0.0-py3-none-any.whl` in a clean venv installs the wheel and creates the `neuroquant` console script as expected.

### O4 · README + per-wave architecture docs
- New [`README.md`](../../README.md) at project root: install / run / config / methods / tests / architecture overview, with badge row + per-wave table linking to detailed notes.
- `docs/architecture/wave{1..7}.md` — one markdown file per wave with the original decision matrix, what shipped, what tests cover it, and the production outcome.
- New [`LICENSE`](../../LICENSE) at project root (MIT, matches the `pyproject.toml` declaration).
- Pre-existing `RUN_GUIDE.md` removed — the new README is more complete, and the per-wave docs cover everything that was Windows-specific in the run guide.

## Tests

[`test_wave7_production.py`](../../test_wave7_production.py) covers:
- `pyproject.toml` exposes the right metadata fields.
- Console-script entry point points at `main:main`.
- Wheel exists in `dist/` after build (skipped if wheel not yet built).
- README + every per-wave doc exists.

## Outcomes

- The framework is now a proper installable Python package: `pip install neuroquant` would work if published.
- A new user can run `neuroquant --config config.yaml` instead of needing to know about `python main.py`.
- Each wave has a reproducible, dated record of what was decided, what was implemented, and what tests guard it — the seven-wave production-hardening process is fully documented.
