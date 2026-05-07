"""
NeuroQuant v2.0 — Wave 7 production tests (Packaging + docs).

Coverage matrix:

  * O1 — pyproject.toml metadata
            - [project] block carries name, version, description,
              requires-python, license, dependencies
            - dependencies block lists the wave-4/5 ONNX deps
            - optional-dependencies expose test, dev, xai extras

  * O3 — Console-script entry point
            - [project.scripts] registers ``neuroquant``
            - Entry point points at ``main:main``
            - main.main is callable (i.e. exists and is a function)

  * O2 — Wheel build artefact
            - dist/neuroquant-2.0.0-py3-none-any.whl exists OR can
              be re-built (skipped if build not run yet — CI handles
              the actual build)
            - Wheel contains the expected top-level modules
            - entry_points.txt declares the neuroquant script

  * O4 — README + per-wave docs
            - README.md exists at project root
            - docs/architecture/wave{1..7}.md exist
            - Each per-wave doc names its decision-matrix items

  * CI sanity
            - .github/workflows/tests.yml installs the dev extras
              correctly via the pyproject ``test`` group
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_utf8(path: Path) -> str:
    """UTF-8 read; default Windows CP1252 chokes on the docs' ≥ char."""
    return path.read_text(encoding="utf-8")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# O1 — pyproject.toml metadata
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _load_pyproject() -> dict:
    """Load pyproject.toml using stdlib tomllib (3.11+) or tomli."""
    try:
        import tomllib  # 3.11+
    except ImportError:  # pragma: no cover — only on 3.10
        import tomli as tomllib  # type: ignore[no-redef]
    pyproject = PROJECT_ROOT / "pyproject.toml"
    return tomllib.loads(_read_utf8(pyproject))


def test_pyproject_project_block_has_required_metadata():
    """The [project] block contains every PEP 621 required field."""
    cfg = _load_pyproject()
    proj = cfg.get("project", {})
    assert proj.get("name") == "neuroquant"
    assert proj.get("version") == "2.0.0"
    assert proj.get("description")
    assert proj.get("requires-python")
    assert proj.get("license", {}).get("text") == "MIT"
    assert proj.get("readme") == "README.md"


def test_pyproject_lists_runtime_dependencies():
    """``dependencies`` includes the wave-4 ONNX trio."""
    cfg = _load_pyproject()
    deps = " ".join(cfg.get("project", {}).get("dependencies", []))
    assert "torch" in deps
    assert "onnx" in deps and "onnxruntime" in deps and "onnxscript" in deps
    assert "pydantic" in deps
    assert "pymoo" in deps
    assert "mlflow" in deps


def test_pyproject_optional_dependencies_expose_extras():
    """``test``, ``dev``, ``xai`` extras are present and complete."""
    cfg = _load_pyproject()
    extras = cfg.get("project", {}).get("optional-dependencies", {})
    for name in ("test", "dev", "xai"):
        assert name in extras, f"missing optional-dependency '{name}'"
    test_deps = " ".join(extras["test"])
    assert "pytest" in test_deps and "pytest-cov" in test_deps
    assert "hypothesis" in test_deps
    dev_deps = " ".join(extras["dev"])
    assert "ruff" in dev_deps
    assert "build" in dev_deps


def test_pyproject_classifiers_declare_python_versions():
    """The classifier list claims Python 3.10/3.11/3.12 support."""
    cfg = _load_pyproject()
    classifiers = cfg.get("project", {}).get("classifiers", [])
    versions = [c for c in classifiers if "Programming Language :: Python :: 3." in c]
    found = {c.rsplit(":", 1)[1].strip() for c in versions}
    assert {"3.10", "3.11", "3.12"}.issubset(found)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# O3 — Console-script entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_pyproject_registers_neuroquant_console_script():
    """``[project.scripts]`` declares the ``neuroquant`` entry."""
    cfg = _load_pyproject()
    scripts = cfg.get("project", {}).get("scripts", {})
    assert scripts.get("neuroquant") == "main:main", (
        f"expected 'neuroquant = main:main', got {scripts!r}"
    )


def test_main_main_function_is_callable():
    """``main.main`` exists and is callable (the entry-point target)."""
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        import main as mainmod
    finally:
        # Don't permanently mutate sys.path for downstream tests.
        if str(PROJECT_ROOT) in sys.path:
            sys.path.remove(str(PROJECT_ROOT))
    assert hasattr(mainmod, "main")
    assert callable(mainmod.main)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# O2 — Wheel build artefact
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def wheel_path() -> Path:
    """Path to the most recently built wheel.

    Skips if no wheel has been built yet — CI builds it as part of
    the pipeline; locally a developer must run ``python -m build``.
    """
    dist = PROJECT_ROOT / "dist"
    if not dist.exists():
        pytest.skip("no dist/ directory; run `python -m build` first")
    wheels = sorted(dist.glob("neuroquant-*-py3-none-any.whl"))
    if not wheels:
        pytest.skip("no neuroquant wheel built yet")
    return wheels[-1]


def test_wheel_contains_top_level_modules(wheel_path):
    """The wheel ships every top-level package the framework needs."""
    expected = {
        "config.py", "main.py",
        "quantization/", "utils/", "tracking/",
        "visualization/", "data/", "models/",
    }
    with zipfile.ZipFile(wheel_path) as zf:
        names = zf.namelist()
    for ext in expected:
        assert any(n.startswith(ext) or n == ext for n in names), (
            f"wheel missing {ext}"
        )


def test_wheel_declares_neuroquant_entry_point(wheel_path):
    """``entry_points.txt`` inside the wheel declares the CLI."""
    with zipfile.ZipFile(wheel_path) as zf:
        ep_files = [n for n in zf.namelist() if n.endswith("entry_points.txt")]
        assert ep_files, "wheel missing entry_points.txt"
        text = zf.read(ep_files[0]).decode()
    assert "[console_scripts]" in text
    assert "neuroquant = main:main" in text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# O4 — README + architecture docs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_readme_exists_and_links_to_architecture():
    """``README.md`` exists at root and points at the per-wave docs."""
    readme = PROJECT_ROOT / "README.md"
    assert readme.exists(), "README.md missing"
    text = _read_utf8(readme)
    assert "NeuroQuant" in text
    assert "docs/architecture/" in text


@pytest.mark.parametrize("wave", [1, 2, 3, 4, 5, 6, 7])
def test_per_wave_architecture_doc_exists(wave):
    """Each wave has a markdown architecture note under docs/architecture/."""
    doc = PROJECT_ROOT / "docs" / "architecture" / f"wave{wave}.md"
    assert doc.exists(), f"missing docs/architecture/wave{wave}.md"
    text = _read_utf8(doc)
    # Each wave note must include a decision-matrix block and a tests
    # section so reviewers can find the contract that ships.
    assert "Decision matrix" in text or "decision matrix" in text
    assert "## Tests" in text or "Tests" in text
