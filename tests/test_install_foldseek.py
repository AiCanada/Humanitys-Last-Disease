"""Behaviour of scripts/install_foldseek.py, and the resolution it depends on.

The cap is the part worth pinning. ``Alphafold/UniProt50`` is 114.1 GB and its
name says nothing about that, so the refusal has to come from the recorded
figure rather than from an operator noticing.
"""

import importlib.util
import os
import stat
from pathlib import Path

import pytest

from rfproteina.disease import novelty
from rfproteina.disease.novelty import FOLDSEEK_DATABASES
from tests.helpers import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "install_foldseek.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("install_foldseek", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INSTALL = _load_script()


def _executable(path: Path, body: str = "#!/bin/sh\necho stub\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def data_path(tmp_path, monkeypatch):
    """A DATA_PATH of its own, and no FOLDSEEK pointing anywhere real."""
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    monkeypatch.delenv("FOLDSEEK", raising=False)
    return tmp_path


# ------------------------------------------------------------------- the cap


def test_the_oversized_database_is_refused_by_its_recorded_size():
    assert INSTALL.declared_gb("Alphafold/UniProt50") == pytest.approx(114.1)
    assert INSTALL.declared_gb("Alphafold/UniProt50") > INSTALL.DOWNLOAD_CAP_GB


def test_every_small_database_is_inside_the_cap():
    for name in ("PDB", "Alphafold/Swiss-Prot", "Alphafold/Proteome"):
        size = INSTALL.declared_gb(name)
        assert size is not None, f"{name} records no readable size"
        assert size <= INSTALL.DOWNLOAD_CAP_GB


def test_a_size_that_cannot_be_read_refuses_rather_than_assuming_it_is_small(monkeypatch):
    """Fail closed. The cap exists because the names do not warn you."""
    monkeypatch.setitem(
        FOLDSEEK_DATABASES, "Mystery", {"command": "foldseek databases Mystery m tmp", "download": "m.tar.gz"}
    )
    assert INSTALL.declared_gb("Mystery") is None


def test_a_version_number_in_the_file_name_is_not_read_as_the_size(monkeypatch):
    monkeypatch.setitem(
        FOLDSEEK_DATABASES,
        "Versioned",
        {"command": "foldseek databases Versioned v tmp", "download": "pdb100_v2.tar.gz 3.5 GB"},
    )
    assert INSTALL.declared_gb("Versioned") == pytest.approx(3.5)


def test_an_oversized_database_downloads_nothing(data_path, tmp_path, capsys):
    binary = _executable(tmp_path / "foldseek")
    with pytest.raises(SystemExit) as raised:
        INSTALL._install_database(binary, "Alphafold/UniProt50")
    message = str(raised.value)
    assert "114.1 GB" in message
    assert "over this project's 25 GB download limit" in message
    assert "Nothing was downloaded" in message
    assert not (data_path / "foldseek" / "databases").exists()


def test_an_unknown_database_is_named_against_the_recorded_ones(data_path, tmp_path):
    binary = _executable(tmp_path / "foldseek")
    with pytest.raises(SystemExit, match="not one of the recorded databases"):
        INSTALL._install_database(binary, "NotADatabase")


def test_listing_marks_each_database_allowed_or_refused(capsys):
    assert INSTALL.main(["--list-databases"]) == 0
    printed = capsys.readouterr().out
    assert "cap  25 GB per download" in printed
    assert "Alphafold/UniProt50    REFUSED (114.1 GB, over the cap)" in printed
    assert "PDB                    allowed (2.2 GB)" in printed


def test_the_alias_comes_from_the_recorded_command():
    """So the directory on disk matches what the handoff tells an operator to type."""
    assert INSTALL.database_alias("PDB") == "pdb"
    assert INSTALL.database_alias("Alphafold/Proteome") == "afdb_prot"
    assert INSTALL.database_alias("NotADatabase") is None


# ------------------------------------------------------- resolution and --have


def test_have_reports_absent_before_anything_is_installed(data_path, capsys):
    assert INSTALL.main(["--have"]) == 1
    assert "status  absent" in capsys.readouterr().out


def test_the_screen_resolves_an_installed_binary_that_is_not_on_path(data_path):
    """The defect this closes.

    ``foldseek_available`` asked ``shutil.which`` alone, so a Foldseek
    installed under DATA_PATH -- where this script puts it, and where the
    MMseqs2 installer puts its own binary -- left the novelty axis reporting
    ``foldseek_not_installed`` with the executable on disk.
    """
    assert novelty.foldseek_command() is None
    installed = _executable(data_path / "foldseek" / "foldseek")
    assert novelty.foldseek_command() == str(installed)
    assert novelty.foldseek_available() is True


def test_a_binary_unpacked_to_bin_is_resolved_too(data_path):
    """The release tarball's own layout, in case a copy is used where it fell."""
    installed = _executable(data_path / "foldseek" / "bin" / "foldseek")
    assert novelty.foldseek_command() == str(installed)


def test_the_environment_variable_wins_over_an_installed_copy(data_path, tmp_path, monkeypatch):
    _executable(data_path / "foldseek" / "foldseek")
    chosen = _executable(tmp_path / "elsewhere" / "foldseek")
    monkeypatch.setenv("FOLDSEEK", str(chosen))
    assert novelty.foldseek_command() == str(chosen)


def test_a_directory_named_like_the_binary_is_not_mistaken_for_it(data_path):
    (data_path / "foldseek" / "foldseek").mkdir(parents=True)
    assert novelty.foldseek_command() is None


def test_a_file_that_is_not_executable_is_not_offered(data_path):
    path = data_path / "foldseek" / "foldseek"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not executable", encoding="utf-8")
    path.chmod(0o644)
    assert novelty.foldseek_command() is None


def test_have_reports_the_binary_and_the_databases_once_installed(data_path, capsys, monkeypatch):
    _executable(data_path / "foldseek" / "foldseek", "#!/bin/sh\necho 1.2.3\n")
    (data_path / "foldseek" / "databases" / "pdb").mkdir(parents=True)
    assert INSTALL.main(["--have"]) == 0
    printed = capsys.readouterr().out
    assert "1.2.3" in printed
    assert "pdb" in printed


def test_a_second_run_keeps_the_binary_that_is_there(data_path, capsys, monkeypatch):
    """Nothing is re-downloaded; an installer an operator fears is one they skip."""
    _executable(data_path / "foldseek" / "foldseek")

    def _refuse(*_args, **_kwargs):
        raise AssertionError("the binary was downloaded again")

    monkeypatch.setattr(INSTALL, "_install_binary", _refuse)
    assert INSTALL.main([]) == 0
    assert "have   " in capsys.readouterr().out


def test_the_screen_runs_the_resolved_binary_not_the_bare_name(data_path, tmp_path, monkeypatch):
    """``novelty_search`` has to execute the copy it found, or resolving it
    achieved nothing: ``foldseek`` alone is not on PATH here."""
    installed = _executable(data_path / "foldseek" / "foldseek")
    (tmp_path / "d1.pdb").write_text("x", encoding="utf-8")
    database = tmp_path / "db"
    database.write_text("db", encoding="utf-8")
    seen: list[str] = []

    def fake_run_tool(command, **_kwargs):
        from rfproteina.disease.screening_tools import ToolResult

        seen.append(command[0])
        Path(command[4]).write_text("d1\t8bbb\t0.95\t0.77\t1e-20\t400\n", encoding="utf-8")
        return ToolResult(True, 0, "", "")

    monkeypatch.setattr(novelty, "run_tool", fake_run_tool)
    novelty.novelty_search(tmp_path, database, tmp_path / "work")
    assert seen == [str(installed)]


def test_an_explicit_binary_argument_is_left_alone(data_path, tmp_path, monkeypatch):
    """A caller that named its own Foldseek keeps it, resolver or no resolver."""
    _executable(data_path / "foldseek" / "foldseek")
    (tmp_path / "d1.pdb").write_text("x", encoding="utf-8")
    database = tmp_path / "db"
    database.write_text("db", encoding="utf-8")
    seen: list[str] = []

    def fake_run_tool(command, **_kwargs):
        from rfproteina.disease.screening_tools import ToolResult

        seen.append(command[0])
        Path(command[4]).write_text("", encoding="utf-8")
        return ToolResult(True, 0, "", "")

    monkeypatch.setattr(novelty, "run_tool", fake_run_tool)
    novelty.novelty_search(tmp_path, database, tmp_path / "work", foldseek="/opt/mine/foldseek")
    assert seen == ["/opt/mine/foldseek"]


def test_the_two_published_builds_are_the_ones_selected_between():
    """There is no SSE2 build; selecting one would 404 on a non-AVX2 machine."""
    assert INSTALL._AVX2.endswith("foldseek-linux-avx2.tar.gz")
    assert INSTALL._SSE41.endswith("foldseek-linux-sse41.tar.gz")
    assert os.path.dirname(INSTALL._AVX2) == os.path.dirname(INSTALL._SSE41)
