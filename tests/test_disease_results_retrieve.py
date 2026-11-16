"""Getting an AlphaFold model back, and saying honestly where it came from.

Neither destination can be driven end to end. The AlphaFold Server has no
submission API and issues no credential; the AlphaFold database has a
documented API but it is GET-only and keyed by accession, serving models
already computed and accepting no sequence for folding. So retrieval is several
routes, and the person has already done part of the work by the time they get
here -- they have the file, or the text, or only an accession.

The thing these tests guard is not the plumbing but the **claim**. An overlay
captioned "AlphaFold 3 model" drawn from a file somebody pasted is the page
asserting something nobody checked, and on a page of safety verdicts that is
the one move it must not make. So only the route that fetched from the service
it names may call itself verified, and every other caption says otherwise.
"""

from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

import pytest

from rfproteina.disease.results_retrieve import (
    TEXT_LIMIT,
    from_accession,
    from_path,
    from_text,
)

_CIF = (
    "data_model\nloop_\n"
    "_atom_site.group_PDB\n_atom_site.id\n_atom_site.label_atom_id\n"
    "_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.label_seq_id\n"
    "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
    "ATOM 1 CA ALA A 1 0.000 0.000 0.000\n"
    "ATOM 2 CA GLY A 2 3.800 0.000 0.000\n"
)
_PDB = (
    "HEADER    TEST\n"
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C\n"
    "ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00 90.00           C\n"
    "END\n"
)


def prediction(**extra: object) -> dict[str, object]:
    base = {
        "entryId": "AF-P05113-F1",
        "uniprotAccession": "P05113",
        "cifUrl": "https://alphafold.ebi.ac.uk/files/AF-P05113-F1-model_v6.cif",
        "latestVersion": 6,
        "globalMetricValue": 86.69,
    }
    base.update(extra)
    return base


# ------------------------------------------------------------- paste and parse


def test_pasted_mmcif_is_accepted() -> None:
    got = from_text(_CIF)
    assert got.found
    assert got.chains == ("A",)
    assert got.route == "paste"


def test_pasted_pdb_is_accepted_too() -> None:
    """The database offers cifUrl and pdbUrl for every model, and a person
    retrieving one by hand takes whichever their browser opened."""
    got = from_text(_PDB)
    assert got.found
    assert got.chains == ("A",)


def test_a_pasted_model_is_never_called_verified() -> None:
    """The caption is the whole point: this is the file as supplied."""
    got = from_text(_CIF)
    assert got.verified is False
    assert "Not verified" in got.caption()


@pytest.mark.parametrize("junk", ["", "   \n\n", "hello world", "<html><body>nope</body></html>"])
def test_text_that_is_not_a_structure_is_refused_by_name(junk: str) -> None:
    got = from_text(junk)
    assert not got.found
    assert got.blocker in {"empty", "not_a_structure"}
    assert got.caption()


def test_something_too_large_to_be_one_model_is_not_read() -> None:
    got = from_text("ATOM  " + "x" * (TEXT_LIMIT + 10))
    assert got.blocker == "too_large"


# ------------------------------------------------------------ open a file


def test_a_downloaded_cif_is_opened(tmp_path: Path) -> None:
    path = tmp_path / "AF-P05113-F1-model_v6.cif"
    path.write_text(_CIF, encoding="utf-8")
    got = from_path(path)
    assert got.found and got.route == "file"
    assert path.name in got.caption()


def test_a_gzipped_download_is_opened(tmp_path: Path) -> None:
    path = tmp_path / "model.cif.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(_CIF)
    assert from_path(path).found


def test_a_missing_path_is_named_not_guessed(tmp_path: Path) -> None:
    got = from_path(tmp_path / "absent.cif")
    assert got.blocker == "not_found"


def test_a_file_that_is_not_gzip_but_named_gz_is_a_blocker(tmp_path: Path) -> None:
    path = tmp_path / "model.cif.gz"
    path.write_bytes(b"not gzip")
    assert from_path(path).blocker == "unreadable"


# --------------------------------------------------------- a server archive


def archive(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, body in members.items():
            bundle.writestr(name, body)
    return path


def test_the_model_is_found_beside_the_archive_s_own_paperwork(tmp_path: Path) -> None:
    """An AlphaFold Server archive carries terms_of_use and job_request, which
    are expected members rather than surprises."""
    path = archive(
        tmp_path / "fold.zip",
        {"terms_of_use.md": "terms", "job_request.json": "{}", "fold_model_0.cif": _CIF},
    )
    got = from_path(path)
    assert got.found and got.route == "archive"
    assert "fold_model_0.cif" in got.caption()


def test_an_archive_of_only_paperwork_produced_nothing(tmp_path: Path) -> None:
    path = archive(tmp_path / "fold.zip", {"terms_of_use.md": "terms", "job_request.json": "{}"})
    got = from_path(path)
    assert got.blocker == "no_model_in_archive"
    assert "produced" in got.caption()


def test_the_first_numbered_model_is_preferred(tmp_path: Path) -> None:
    path = archive(
        tmp_path / "fold.zip",
        {"fold_model_3.cif": _CIF, "fold_model_0.cif": _CIF, "fold_model_1.cif": _CIF},
    )
    assert "fold_model_0.cif" in from_path(path).caption()


def test_a_member_named_to_escape_is_skipped(tmp_path: Path) -> None:
    """An archive is untrusted input. Nothing is written out of it, but a name
    built to escape is not read either."""
    path = archive(tmp_path / "fold.zip", {"../outside.cif": _CIF})
    assert from_path(path).blocker == "no_model_in_archive"


def test_a_corrupt_archive_is_a_blocker_not_a_crash(tmp_path: Path) -> None:
    path = tmp_path / "fold.zip"
    path.write_bytes(b"PK\x03\x04 not really a zip")
    assert from_path(path).blocker == "unreadable"


# ------------------------------------------------- the database, by accession


def test_a_model_fetched_from_the_database_is_the_one_route_that_is_verified() -> None:
    """It came from the service the caption names, keyed by the accession asked
    for. That is the only basis on which this page may vouch for a structure.
    """
    got = from_accession(
        "p05113",
        read_json=lambda url: [prediction()],
        read_bytes=lambda url: _CIF.encode(),
    )
    assert got.found and got.route == "afdb"
    assert got.verified is True
    assert "AF-P05113-F1" in got.caption()
    assert "version 6" in got.caption()
    assert "86.7" in got.caption(), "the mean pLDDT belongs in the caption"
    assert "Not verified" not in got.caption()


def test_an_accession_the_database_does_not_hold_says_why_a_design_misses() -> None:
    """The important sentence on this whole page. A designed binder has no
    entry by construction, and that is not a failure to retry."""

    def absent(url: str) -> object:
        raise LookupError("no entry")

    got = from_accession("DENOVO1", read_json=absent, read_bytes=lambda url: b"")
    assert not got.found
    assert got.blocker == "afdb_unreachable"

    empty = from_accession("DENOVO1", read_json=lambda url: [], read_bytes=lambda url: b"")
    assert empty.blocker == "afdb_no_entry"
    assert "by construction" in empty.caption()
    assert "checksum" in empty.caption()


def test_a_record_naming_no_model_file_is_refused() -> None:
    got = from_accession(
        "P05113",
        read_json=lambda url: [prediction(cifUrl=None, pdbUrl=None)],
        read_bytes=lambda url: _CIF.encode(),
    )
    assert got.blocker == "afdb_no_model_url"


def test_the_pdb_url_is_used_when_there_is_no_cif() -> None:
    asked: list[str] = []

    def read_bytes(url: str) -> bytes:
        asked.append(url)
        return _PDB.encode()

    got = from_accession(
        "P05113",
        read_json=lambda url: [prediction(cifUrl=None, pdbUrl="https://example.invalid/a.pdb")],
        read_bytes=read_bytes,
    )
    assert got.found
    assert asked == ["https://example.invalid/a.pdb"]


def test_an_empty_accession_never_reaches_the_network() -> None:
    def refuse(url: str) -> object:
        raise AssertionError("no accession should mean no request")

    assert from_accession("   ", read_json=refuse, read_bytes=refuse).blocker == "no_accession"


def test_the_url_requested_is_the_one_the_record_gave() -> None:
    """Not assembled here. The record names its own file, and a URL built from
    a template would go stale the next time the database renames a version."""
    asked: list[str] = []
    from_accession(
        "P05113",
        read_json=lambda url: [prediction()],
        read_bytes=lambda url: (asked.append(url), _CIF.encode())[1],
    )
    assert asked == ["https://alphafold.ebi.ac.uk/files/AF-P05113-F1-model_v6.cif"]
