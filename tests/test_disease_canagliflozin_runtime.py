"""Regressions for local solver inventory and concurrent, offline-safe installation."""

import hashlib
import os
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from rfproteina.disease import canagliflozin_runtime as runtime
from scripts import install_canagliflozin as installer

supported = pytest.mark.skipif(
    sys.platform not in installer.WHEELS or sys.version_info[:2] != (3, 12)
    or sys.implementation.name != "cpython" or platform.machine().lower() not in {"x86_64", "amd64"},
    reason="installer requires a supported wheel platform",
)


def test_cached_solver_probe_detects_missing_shared_library(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    site = root / "venv" / ("Lib" if os.name == "nt" else "lib/python3.12") / "site-packages"
    library = site / "numpy.libs" / ("openblas.dll" if os.name == "nt" else "libopenblas.so")
    for path in (
        runtime.interpreter(root), root / "venv" / "pyvenv.cfg", site / "roadrunner" / "__init__.py",
        site / "roadrunner" / "roadrunner.py", site / "numpy" / "__init__.py", library,
        *(site / f"{name}-{version}.dist-info" / "METADATA" for name, version in runtime.worker.VERSIONS.items()),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    probes = []

    def native_probe(where, data):
        assert where == root and data is None
        probes.append(where)
        if not library.is_file():
            raise runtime.CanagliflozinRuntimeError("canagliflozin_solver_unavailable")

    monkeypatch.setattr(runtime, "install_directory", lambda: root)
    monkeypatch.setattr(runtime, "model_bytes", lambda _: b"model")
    monkeypatch.setattr(runtime, "invoke", native_probe)
    assert runtime.installed() and runtime.installed()
    assert len(probes) == 1
    # A partial package move/repair can affect a shared library without touching the extension or its metadata.
    moved = library.rename(library.with_suffix(".hidden"))
    assert not runtime.installed()
    assert len(probes) == 2
    moved.rename(library)
    assert runtime.installed()


@supported
def test_missing_uv_fails_before_any_download(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "install_directory", lambda: tmp_path / "optional")
    monkeypatch.setattr(installer.shutil, "which", lambda *args, **kwargs: None)
    monkeypatch.setattr(installer, "_download", lambda *args: pytest.fail("assets fetched before checking uv"))
    with pytest.raises(installer.InstallError, match="uv is required"):
        installer.install()


@supported
def test_simultaneous_installation_stages_once_and_preserves_invalid_existing_tree(tmp_path, monkeypatch):
    destination = tmp_path / "bundle"
    model = b"test model"
    downloads, builds, validations = [], [], []
    start = Barrier(2, timeout=10)

    def uv():
        start.wait()
        return "uv"

    def download(url, size):
        downloads.append(url)
        return model

    def stage(temp, snapshot, source, executable):
        assert snapshot == model and source is None and executable == "uv"
        staged = temp / "bundle"
        (staged / "model").mkdir(parents=True)
        (staged / "model" / runtime.MODEL_FILENAME).write_bytes(snapshot)
        builds.append(staged)
        return staged

    monkeypatch.setattr(runtime, "install_directory", lambda: destination)
    monkeypatch.setattr(runtime, "MODEL_SIZE", len(model))
    monkeypatch.setattr(runtime, "MODEL_SHA256", hashlib.sha256(model).hexdigest())
    monkeypatch.setattr(runtime, "invoke", lambda root, data: validations.append((root, data)))
    monkeypatch.setattr(installer, "_uv", uv)
    monkeypatch.setattr(installer, "_download", download)
    monkeypatch.setattr(installer, "_stage", stage)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = (pool.submit(installer.install) for _ in range(2))
        assert first.result() == second.result() == destination
    assert downloads == [runtime.MODEL_URL] and len(builds) == 1
    assert validations == [(destination, model)]
    installed_model = destination / "model" / runtime.MODEL_FILENAME
    installed_model.write_bytes(b"broken")
    with pytest.raises(installer.InstallError, match="canagliflozin_model_integrity"):
        installer.install()
    assert installed_model.read_bytes() == b"broken" and len(builds) == 1
