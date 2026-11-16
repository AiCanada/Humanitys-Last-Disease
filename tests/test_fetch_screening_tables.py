"""The command that fetches the screening tables.

Thin by design -- the registry, the limit and the download are in
``rfproteina.disease.tables`` and are tested in ``test_disease_tables.py``. What
is tested here is what the command *says*, because every line of it is a claim
about files on somebody's disk.

The case that prompted this: ``--dry-run`` asked each server its size, fetched
nothing, and then printed "All requested tables arrived." A dry run reporting
arrival is worse than no dry run, because the next step is to go looking for
files that were never written.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from rfproteina.disease import tables

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fetch_screening_tables.py"


def _load():
    """The script as a module, loaded by path rather than by package name."""
    spec = importlib.util.spec_from_file_location("fetch_screening_tables", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fetch_tables = _load()


class TestItNeverClaimsMoreThanItDid:
    def test_a_dry_run_does_not_report_arrival(self, capsys, monkeypatch) -> None:
        """The defect. Nothing was downloaded, so nothing arrived."""
        monkeypatch.setattr(fetch_tables, "declared_size", lambda url: 1234)
        monkeypatch.setattr(
            fetch_tables, "fetch", lambda *a, **k: pytest.fail("a dry run must not download")
        )
        assert fetch_tables.main(["--dry-run"]) == 0
        printed = capsys.readouterr().out
        assert "nothing downloaded" in printed
        assert "arrived" not in printed.replace("did not arrive", "")

    def test_a_real_run_reports_arrival_only_when_everything_landed(self, capsys, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(fetch_tables, "declared_size", lambda url: 10)
        monkeypatch.setattr(fetch_tables, "fetch", lambda source, dest, **k: dest / "x")
        assert fetch_tables.main(["--to", str(tmp_path)]) == 0
        assert "All requested tables arrived." in capsys.readouterr().out

    def test_a_refusal_is_counted_and_exits_nonzero(self, capsys, monkeypatch, tmp_path) -> None:
        """A partial fetch must not look like a whole one."""

        def refuse(source, dest, **_kwargs):
            raise fetch_tables.TooLargeError(f"{source.name} is too large")

        monkeypatch.setattr(fetch_tables, "declared_size", lambda url: 10)
        monkeypatch.setattr(fetch_tables, "fetch", refuse)
        assert fetch_tables.main(["--to", str(tmp_path)]) == 1
        printed = capsys.readouterr().out
        assert "REFUSED" in printed
        assert "did not arrive" in printed


class TestWhatItPrints:
    def test_list_shows_the_downloads_the_queries_and_the_refusals(self, capsys) -> None:
        """All three, because the refusals are the reason the limit exists."""
        assert fetch_tables.main(["--list"]) == 0
        printed = capsys.readouterr().out
        assert "Downloaded" in printed and "Queried, not downloaded" in printed and "Refused" in printed
        assert "STRING interaction partners" in printed
        assert "1.27 TB" in printed, "the largest refusal should be legible as TB"

    def test_list_fetches_nothing(self, capsys, monkeypatch) -> None:
        monkeypatch.setattr(fetch_tables, "fetch", lambda *a, **k: pytest.fail("--list must not download"))
        monkeypatch.setattr(
            fetch_tables, "declared_size", lambda url: pytest.fail("--list must not touch the network")
        )
        assert fetch_tables.main(["--list"]) == 0

    def test_every_run_says_which_axes_still_go_online(self, capsys, monkeypatch, tmp_path) -> None:
        """Having the tables is not being offline, and the command says so."""
        monkeypatch.setattr(fetch_tables, "declared_size", lambda url: 10)
        monkeypatch.setattr(fetch_tables, "fetch", lambda source, dest, **k: dest / "x")
        fetch_tables.main(["--to", str(tmp_path)])
        printed = capsys.readouterr().out
        assert "asked online on every run" in printed
        for axis in ("paralog", "plasma_abundance", "plasma_hub"):
            assert axis in printed


class TestCoverage:
    def test_an_empty_run_is_incomplete_and_exits_nonzero(self, capsys, tmp_path) -> None:
        assert fetch_tables.main(["--coverage", str(tmp_path)]) == 1
        printed = capsys.readouterr().out
        assert "MISSING" in printed and "incomplete" in printed

    def test_a_stocked_run_is_complete_and_still_names_the_online_lookups(self, capsys, tmp_path) -> None:
        for stem in ("mhc_i", "immunogenicity", "ada", "plasma_fold"):
            (tmp_path / f"{stem}.tsv").write_text("x", encoding="utf-8")
        assert fetch_tables.main(["--coverage", str(tmp_path)]) == 0
        printed = capsys.readouterr().out
        assert "MISSING" not in printed
        assert "Still asked online" in printed


class TestSelection:
    def test_an_unknown_source_name_is_refused_rather_than_silently_fetching_nothing(
        self, capsys, tmp_path
    ) -> None:
        """--only with a typo used to be an empty selection and a cheerful exit."""
        assert fetch_tables.main(["--only", "IEDB mhc", "--to", str(tmp_path)]) == 2
        assert "No such source" in capsys.readouterr().err

    def test_a_known_source_name_selects_just_that_one(self, capsys, monkeypatch, tmp_path) -> None:
        taken: list[str] = []
        monkeypatch.setattr(fetch_tables, "declared_size", lambda url: 10)
        monkeypatch.setattr(
            fetch_tables, "fetch", lambda source, dest, **k: (taken.append(source.name), dest / "x")[1]
        )
        assert fetch_tables.main(["--only", "IDC anti-drug antibody incidence", "--to", str(tmp_path)]) == 0
        assert taken == ["IDC anti-drug antibody incidence"]


def test_the_default_destination_follows_data_path(monkeypatch, tmp_path) -> None:
    """Reference data belongs beside the checkpoints, not in the checkout.

    The helper moved into ``rfproteina.disease.tables`` so that the library and
    this script cannot disagree about where downloads live; the script had its
    own copy. Asserted through the script's own import so that a script that
    stops using the shared one fails here.
    """
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    assert fetch_tables.default_directory() == tmp_path / "screening_tables"
    monkeypatch.setenv("DATA_PATH", "")
    assert fetch_tables.default_directory().name == "screening_tables"
    assert fetch_tables.default_directory is tables.default_directory
