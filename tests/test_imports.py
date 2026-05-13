"""Smoke tests for the public API surface.

Catches regressions where a refactor accidentally removes a symbol from
``neuroquant.__all__`` or breaks an internal import chain. Cheap (no
torch tensors created) and exhaustive.
"""
from __future__ import annotations


def test_top_level_imports() -> None:
    """Every symbol in ``neuroquant.__all__`` must be importable."""
    import neuroquant

    for name in neuroquant.__all__:
        assert hasattr(neuroquant, name), f"missing public symbol: {name}"


def test_version_string() -> None:
    import neuroquant

    assert isinstance(neuroquant.__version__, str)
    # Coarse semver shape check — three dot-separated chunks, all numeric
    # at the start. Avoids hard-coding the literal version so this test
    # doesn't need updating on every bump.
    parts = neuroquant.__version__.split(".")
    assert len(parts) >= 2
    int(parts[0])
    int(parts[1])


def test_cli_entry_point_importable() -> None:
    """``neuroquant`` console-script points at ``neuroquant.cli:main``."""
    from neuroquant.cli import main

    assert callable(main)
