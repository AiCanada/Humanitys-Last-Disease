"""Behaviour of scripts/install_gtopdb.py, run as the operator would run it.

The reductions themselves are covered in ``test_disease_tables.py``. What is
pinned here is the script's own contract: that it fetches all three files, that
it reports rather than rewrites a BindingDB subset it cannot merge, and that a
release which has moved on is called out instead of quietly breaking the size
a registry test asserts.
"""

import csv
import importlib.util
from pathlib import Path

import pytest

from rfproteina.disease.tables import BINDINGDB_PANEL_SUBSET_FILE, GTOPDB_FILES, PANEL_SUBSET_COLUMNS
from tests.helpers import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "install_gtopdb.py"

VERSION_LINE = '"# GtoPdb Version: 2026.3.1 - published: 2026-10-12"'
SMILES = "CC(=O)Oc1ccccc1C(=O)O"

INTERACTION_COLUMNS = (
    "Target Gene Symbol",
    "Target Species",
    "Ligand",
    "Ligand ID",
    "Type",
    "Original Affinity Units",
    "Original Affinity Median nm",
    "Original Affinity Relation",
    "PubMed ID",
    "Target",
    "Target ID",
    "Selectivity",
    "Primary Target",
)
TARGET_COLUMNS = ("Type", "Family id", "Family name", "Target id", "Target name", "HGNC symbol")


def _load_script():
    spec = importlib.util.spec_from_file_location("install_gtopdb", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INSTALL = _load_script()


def _csv_with_comment(path: Path, columns, rows: list[dict[str, str]]) -> Path:
    """A GtoPdb CSV: a version comment, then the header. The comment is the
    reason the readers cannot just hand the file to ``csv.DictReader``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(VERSION_LINE + "\n")
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_release(directory: Path, only: str | None = None) -> None:
    """The three files, with one Human hERG interaction that survives every filter.

    ``only`` writes a single named file, which is what the stubbed download
    needs: writing all three on the first call would leave the other two
    already present, and the script would then rightly skip fetching them.
    """
    if only is not None:
        written = {name: directory / name for name in GTOPDB_FILES}
        staging = directory / "_staging"
        _write_release(staging)
        written[only].parent.mkdir(parents=True, exist_ok=True)
        written[only].write_bytes((staging / only).read_bytes())
        for leftover in staging.iterdir():
            leftover.unlink()
        staging.rmdir()
        return
    _csv_with_comment(
        directory / "interactions.csv",
        INTERACTION_COLUMNS,
        [
            {
                "Target Gene Symbol": "KCNH2",
                "Target Species": "Human",
                "Ligand": "elacridar",
                "Ligand ID": "13578",
                "Type": "Inhibitor",
                "Original Affinity Units": "IC50",
                "Original Affinity Median nm": "40.0",
                "Original Affinity Relation": "=",
                "PubMed ID": "12636153",
                "Target": "K<SUB>v</SUB>11.1",
                "Target ID": "572",
                "Selectivity": "Not Determined",
                "Primary Target": "false",
            }
        ],
    )
    _csv_with_comment(
        directory / "ligands.csv",
        ("Ligand ID", "Name", "SMILES", "InChIKey", "Approved", "Withdrawn"),
        [
            {
                "Ligand ID": "13578",
                "Name": "elacridar",
                "SMILES": SMILES,
                "InChIKey": "PUBLISHED-KEY-N",
                "Approved": "",
                "Withdrawn": "",
            }
        ],
    )
    _csv_with_comment(
        directory / "targets_and_families.csv",
        TARGET_COLUMNS,
        [
            {
                "Type": "vgic",
                "Family id": "68",
                "Family name": "Voltage-gated potassium channels",
                "Target id": "572",
                "Target name": "KCNH2",
                "HGNC symbol": "KCNH2",
            }
        ],
    )


def _panel_subset(path: Path, columns=PANEL_SUBSET_COLUMNS) -> Path:
    """A BindingDB-shaped panel subset, in whichever column spelling is asked for."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(list(columns))
        writer.writerow(["BDBKEY-N", "BDBKEY-N", "CCO", "herg", "Q12809", "40.0", "=", "IC50", "1", "bdb", "src"])
    return path


@pytest.fixture
def data_path(tmp_path, monkeypatch):
    """A DATA_PATH of its own, so nothing touches the real install."""
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture
def stub_fetch(monkeypatch):
    """Stand in for the network, writing the fixture release instead.

    Patched on the script, where the name is used, not on ``tables``: the
    script binds ``fetch`` at import, so patching the origin would not be seen.
    """
    calls: list[str] = []

    def _fetch(source, destination):
        destination.mkdir(parents=True, exist_ok=True)
        calls.append(source.file)
        _write_release(destination, only=source.file)
        return destination / source.file

    monkeypatch.setattr(INSTALL, "fetch", _fetch)
    return calls


def test_have_reports_every_file_absent_and_exits_nonzero(data_path, capsys):
    assert INSTALL.main(["--have"]) == 1
    printed = capsys.readouterr().out
    for name in GTOPDB_FILES:
        assert name in printed
    assert "incomplete" in printed


def test_a_run_fetches_all_three_files_and_writes_both_reductions(data_path, stub_fetch, capsys):
    """The whole point: one command leaves the release and both subsets on disk."""
    assert INSTALL.main([]) == 0
    printed = capsys.readouterr().out

    downloads = data_path / "screening_tables"
    for name in GTOPDB_FILES:
        assert (downloads / name).is_file(), f"{name} was not fetched"
    assert set(stub_fetch) == set(GTOPDB_FILES), "every registered file is asked for by name"

    subsets = data_path / "gtopdb"
    assert (subsets / "gtopdb_panel_subset.tsv").is_file()
    assert (subsets / "gtopdb_offtarget_subset.tsv").is_file()
    assert "1 rows over 1 ligands" in printed
    assert INSTALL.main(["--have"]) == 0


def test_a_second_run_keeps_what_is_there_rather_than_downloading_again(data_path, stub_fetch, capsys):
    """Idempotent: an installer an operator is afraid to re-run does not get re-run."""
    assert INSTALL.main([]) == 0
    stub_fetch.clear()
    capsys.readouterr()

    assert INSTALL.main([]) == 0
    assert stub_fetch == [], "nothing was downloaded a second time"
    assert capsys.readouterr().out.count("have   ") == len(GTOPDB_FILES)


def test_refresh_downloads_again_over_a_file_that_is_already_there(data_path, stub_fetch):
    assert INSTALL.main([]) == 0
    stub_fetch.clear()
    assert INSTALL.main(["--refresh"]) == 0
    assert set(stub_fetch) == set(GTOPDB_FILES)


def test_a_release_that_no_longer_matches_the_pinned_size_is_called_out(data_path, stub_fetch, capsys):
    """GtoPdb publishes quarterly. A bigger file is a new release, not a fault --
    but it does make ``Source.size`` wrong, and a registry test asserts that
    size, so the operator hears it here rather than from that failure."""
    assert INSTALL.main([]) == 0
    printed = capsys.readouterr().out
    assert "the registry pins" in printed
    assert "update Source.size" in printed


def test_a_stale_bindingdb_subset_is_reported_and_left_alone(data_path, stub_fetch, capsys):
    """The state found on the installed machine.

    A subset written before the column was renamed cannot be merged. The run
    still succeeds -- the GtoPdb subset stands on its own -- and the stale file
    is not edited into shape, because its other drift has not been checked.
    """
    stale_columns = ["inchikey_bindingdb" if name == "inchikey_upstream" else name for name in PANEL_SUBSET_COLUMNS]
    stale = _panel_subset(data_path / "bindingdb" / BINDINGDB_PANEL_SUBSET_FILE, stale_columns)
    before = stale.read_bytes()

    assert INSTALL.main([]) == 0, "a merge it cannot do is not a failed install"
    printed = capsys.readouterr().out
    assert "merge  refused: column_absent:inchikey_upstream" in printed
    assert "Regenerate it from BindingDB_All" in printed
    assert stale.read_bytes() == before, "the stale subset was rewritten"
    assert not (data_path / "panel_subset").exists()


def test_a_current_bindingdb_subset_is_merged_with_the_gtopdb_rows(data_path, stub_fetch, capsys):
    _panel_subset(data_path / "bindingdb" / BINDINGDB_PANEL_SUBSET_FILE)

    assert INSTALL.main([]) == 0
    merged = data_path / "panel_subset" / "panel_subset.tsv"
    assert merged.is_file()
    with merged.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 2, "BindingDB's row and GtoPdb's, both kept"
    assert {row["curation"] for row in rows} == {"bdb", "IUPHAR/BPS GtoPdb (ODbL)"}
    assert "merge  refused" not in capsys.readouterr().out


def test_no_merge_is_attempted_when_there_is_no_bindingdb_subset(data_path, stub_fetch, capsys):
    assert INSTALL.main([]) == 0
    printed = capsys.readouterr().out
    assert "merge  skipped" in printed
    assert "stands on" in printed
