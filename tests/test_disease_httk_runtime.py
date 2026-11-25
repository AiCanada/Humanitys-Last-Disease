"""Native R startup when the system's temp directory contains spaces."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from rfproteina.disease import httk


def test_spaced_temp_path_stays_private_and_is_removed(tmp_path, monkeypatch):
    parent = tmp_path / "R scratch with spaces"
    parent.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(parent))
    observed = []

    def run(_cmd, **options):
        working = Path(options["cwd"])
        assert working.is_dir() and working.parent == parent
        for name in ("TMPDIR", "TMP", "TEMP"):
            value = options["env"][name]
            assert " " not in value and (working / value).resolve() == working.resolve()
        observed.append(working)
        return SimpleNamespace(stdout=b"httk\t2.7.4\n")

    monkeypatch.setattr(httk.subprocess, "run", run)
    assert httk._run(Path("test-Rscript"), "probe", b"", timeout=12).strip() == "httk\t2.7.4"
    assert len(observed) == 1 and not observed[0].exists()


def test_installed_httk_probe_works_with_spaced_temp_path(tmp_path, monkeypatch):
    executable = httk.runtime_path()
    if executable is None or not httk.asset_available():
        pytest.skip("Requires the pinned local R runtime")
    parent = tmp_path / "R startup path with spaces"
    parent.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(parent))
    assert httk._run(executable, "probe", b"", timeout=12).strip() == f"httk\t{httk.VERSION}"
    assert not list(parent.iterdir())
