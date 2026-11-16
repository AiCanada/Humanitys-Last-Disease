"""Shared pytest fixtures for foundry tests."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def gpu():
    """Fixture to check GPU availability for tests that require CUDA.

    ``torch`` is imported here rather than at module scope. The root
    ``tests/conftest.py`` imports this package, and ``testpaths`` in
    ``pyproject.toml`` makes that conftest an *initial* conftest, so pytest
    loads it before every run -- even ``pytest --version``. A module-level
    ``import torch`` therefore cost ~20s on every invocation, paid by the 123
    of 146 test files that never touch torch. Nothing outside this body needs
    it, and a run with no GPU test never imports it at all.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    return True


@pytest.fixture(autouse=True)
def data_path(monkeypatch):
    """Point ``DATA_PATH`` at nothing, so a test never reads the real asset tree.

    Autouse, and imported into the modules that want it with ``# noqa: F401``
    rather than re-exported from ``tests.helpers``. Three files each defined
    this. It must not reach ``tests/conftest.py``: an autouse fixture in a
    conftest applies to every test under it, and most of the suite needs the
    real ``DATA_PATH``.
    """
    monkeypatch.setenv("DATA_PATH", "/unused")


def get_test_data_dir(conftest_file: str) -> Path:
    """Get TEST_DATA_DIR path relative to conftest file."""
    return Path(conftest_file).resolve().parent / "data"
