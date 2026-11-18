"""Octant CYP3A4: pIC50 becomes nM before it can fill toxicology.tsv."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from rfproteina.disease import octant
from rfproteina.disease.ligand_safety import SafetyBackendUnavailableError
from rfproteina.disease.screen import screen_generation
from rfproteina.disease.toxicology import measure_toxicology
from tests.helpers.disease import new_folder

_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
_HEADER = [
    "ocnt_batch",
    "CYP3A4_pIC50",
    "activity_status",
    "direction",
    "standardized_smiles",
]


def _write(path: Path, rows: list[dict[str, str]], header: list[str] | None = None) -> Path:
    fields = header or _HEADER
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "ocnt_batch": "OCNT-1",
        "CYP3A4_pIC50": "6",
        "activity_status": "YES",
        "direction": "DOWN",
        "standardized_smiles": _SMILES,
    }
    row.update(overrides)
    return row


def _subset_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_pic50_uses_the_molar_definition():
    """pIC50 6 is 1 µmol/L. The panel cut of 10 µmol/L is pIC50 5."""
    assert octant.pic50_to_nm(6) == pytest.approx(1000)
    assert octant.pic50_to_nm(9) == pytest.approx(1)
    assert octant.pic50_to_nm(5) == pytest.approx(10_000)
    assert octant.pic50_to_nm(7) == pytest.approx(100)
    with pytest.raises(ValueError):
        octant.pic50_to_nm(float("nan"))


def test_the_subset_stores_nanomolar_not_the_pic50(tmp_path: Path):
    table = _write(tmp_path / "inhibition.tsv", [_row(CYP3A4_pIC50="7.0")])
    result = octant.octant_cyp3a4_subset(table, tmp_path / "out")
    assert result.blocker is None
    assert result.rows == 1
    assert result.axes == ("cyp3a4",)
    assert "herg" in result.missing
    assert "cyp2j2" not in result.axes
    (row,) = _subset_rows(result.path)
    assert row["panel_axis"] == "cyp3a4"
    assert row["accession"] == "P08684"
    assert float(row["affinity_nanomolar"]) == pytest.approx(100)
    assert row["affinity_nanomolar"] != "7.0"
    assert row["relation"] == "="
    assert row["assay"] == "IC50"
    assert "CC BY 4.0" in row["curation"]
    assert "10**(9-pIC50)" in row["source"]
    assert row["inchikey"]


def test_cyp2j2_in_the_same_file_is_not_a_panel_row(tmp_path: Path):
    header = [*_HEADER, "CYP2J2_pIC50", "enzyme"]
    rows = [
        _row(CYP2J2_pIC50="8", enzyme="CYP3A4"),
        {
            "ocnt_batch": "OCNT-2",
            "CYP3A4_pIC50": "9",
            "CYP2J2_pIC50": "4",
            "enzyme": "CYP2J2",
            "activity_status": "YES",
            "direction": "DOWN",
            "standardized_smiles": "CCO",
        },
    ]
    table = _write(tmp_path / "inhibition.tsv", rows, header)
    result = octant.octant_cyp3a4_subset(table, tmp_path / "out")
    assert result.rows == 1
    assert result.axes == ("cyp3a4",)
    assert result.skipped.get("cyp2j2_not_a_panel_axis") == 2
    (row,) = _subset_rows(result.path)
    assert row["panel_axis"] == "cyp3a4"
    assert float(row["affinity_nanomolar"]) == pytest.approx(1000)


def test_a_missing_pic50_is_not_invented(tmp_path: Path):
    table = _write(
        tmp_path / "inhibition.tsv",
        [_row(CYP3A4_pIC50=""), _row(ocnt_batch="OCNT-2", CYP3A4_pIC50="not-a-number")],
    )
    result = octant.octant_cyp3a4_subset(table, tmp_path / "out")
    assert not result.written
    assert result.skipped.get("no_pic50") == 1
    assert result.skipped.get("pic50_unparsed") == 1


def test_reactivity_is_not_an_ic50_table(tmp_path: Path):
    table = _write(
        tmp_path / "reactivity.tsv",
        [{"ocnt_batch": "OCNT-1", "enzyme": "CYP2J2", "pct_remaining": "40"}],
        ["ocnt_batch", "enzyme", "pct_remaining"],
    )
    result = octant.octant_cyp3a4_subset(table, tmp_path / "out")
    assert result.blocker == "column_absent:CYP3A4_pIC50"


def test_the_raw_table_cannot_fill_toxicology(tmp_path: Path):
    """A pIC50 left in the screen's file would be read as nanomolar."""
    raw = _write(tmp_path / "inhibition.tsv", [_row(CYP3A4_pIC50="6")])
    with pytest.raises(SafetyBackendUnavailableError, match="octant_pic50_not_converted"):
        octant.measured_nm_from_octant(_SMILES, raw)
    run = tmp_path / "run"
    run.mkdir()
    (run / "toxicology.tsv").write_text(raw.read_text(encoding="utf-8"), encoding="utf-8")
    measured = measure_toxicology(run)
    assert measured["cyp3a4"]["state"] == "untested"
    assert measured["cyp3a4"]["measurement"] is None
    assert measured["cyp3a4"]["blocker"] == "toxicology_unreadable"


def test_converted_rows_fill_toxicology_in_nanomolar(tmp_path: Path):
    table = _write(
        tmp_path / "inhibition.tsv",
        [_row(CYP3A4_pIC50="7"), _row(ocnt_batch="OCNT-other", standardized_smiles="CCO", CYP3A4_pIC50="4")],
    )
    result = octant.octant_cyp3a4_subset(table, tmp_path / "out")
    measured = octant.measured_nm_from_octant(_SMILES, result.path)
    assert measured.method == octant.OCTANT_METHOD
    assert measured.axes["cyp3a4"]["value_nm"] == pytest.approx(100)
    assert measured.axes["cyp3a4"]["unit"] == "nM"
    assert "herg" not in measured.axes
    run = tmp_path / "run"
    run.mkdir()
    written = octant.write_toxicology_tsv(run / "toxicology.tsv", measured)
    assert written is not None
    text = written.read_text(encoding="utf-8")
    assert "CYP3A4_pIC50" not in text
    assert "\t100\t" in text or "\t100.0\t" in text
    record = measure_toxicology(run)["cyp3a4"]
    assert record["measurement"] == pytest.approx(100)
    assert record["state"] == "hit"
    assert record["blocker"] is None
    # A second fill must not replace a table the operator already has.
    assert octant.write_toxicology_tsv(written, measured) is None


def test_a_weak_pic50_is_a_clear_only_after_conversion(tmp_path: Path):
    """pIC50 4 is 100 µmol/L. Read unconverted, 4 would be a 4 nM hit."""
    table = _write(tmp_path / "inhibition.tsv", [_row(CYP3A4_pIC50="4")])
    result = octant.octant_cyp3a4_subset(table, tmp_path / "out")
    measured = octant.measured_nm_from_octant(_SMILES, result.path)
    run = tmp_path / "run"
    run.mkdir()
    octant.write_toxicology_tsv(run / "toxicology.tsv", measured)
    record = measure_toxicology(run)["cyp3a4"]
    assert record["measurement"] == pytest.approx(100_000)
    assert record["state"] == "clear"


def test_an_absent_ligand_does_not_write_an_empty_table(tmp_path: Path):
    table = _write(tmp_path / "inhibition.tsv", [_row()])
    result = octant.octant_cyp3a4_subset(table, tmp_path / "out")
    measured = octant.measured_nm_from_octant("CCO", result.path)
    assert measured.axes["cyp3a4"]["value_nm"] is None
    assert octant.write_toxicology_tsv(tmp_path / "toxicology.tsv", measured) is None
    assert not (tmp_path / "toxicology.tsv").exists()


def _installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pic50: str = "7") -> Path:
    table = _write(tmp_path / "inhibition.tsv", [_row(CYP3A4_pIC50=pic50)])
    result = octant.octant_cyp3a4_subset(table, tmp_path / "octant")
    assert result.path is not None
    monkeypatch.setattr(octant, "installed_subset", lambda: result.path)
    return result.path


def test_a_ligand_screen_fills_cyp3a4_from_octant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _installed(tmp_path, monkeypatch, "7")
    output = new_folder(tmp_path, "ligand")
    (output / "ligand_smiles.txt").write_text(f"{_SMILES} note\n", encoding="utf-8")
    record = screen_generation(output, binder_kind="ligand", target_id="P05113")
    group = record["toxicology"]
    assert isinstance(group, dict)
    cyp = group["cyp3a4"]
    assert isinstance(cyp, dict)
    assert cyp["state"] == "hit"
    assert cyp["measurement"] == pytest.approx(100)
    assert cyp["blocker"] is None
    assert record["states"]["cyp3a4"] == "hit"
    assert record["call"] == "axis_hit"
    text = (output / "toxicology.tsv").read_text(encoding="utf-8")
    assert "CYP3A4_pIC50" not in text
    assert "\t100" in text
    herg = group["herg"]
    assert isinstance(herg, dict)
    assert herg["blocker"] == "toxicology_not_reported"


def test_an_existing_toxicology_table_is_not_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _installed(tmp_path, monkeypatch, "7")
    output = new_folder(tmp_path, "kept")
    (output / "ligand_smiles.txt").write_text(_SMILES + "\n", encoding="utf-8")
    original = "axis\tnm\nherg\t20000\n"
    (output / "toxicology.tsv").write_text(original, encoding="utf-8")
    record = screen_generation(output, binder_kind="ligand", target_id="P05113")
    assert (output / "toxicology.tsv").read_text(encoding="utf-8") == original
    group = record["toxicology"]
    assert isinstance(group, dict)
    herg = group["herg"]
    cyp = group["cyp3a4"]
    assert isinstance(herg, dict) and isinstance(cyp, dict)
    assert herg["state"] == "clear"
    assert cyp["blocker"] == "toxicology_not_reported"


def test_a_protein_screen_does_not_write_octant_toxicology(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _installed(tmp_path, monkeypatch, "7")
    output = new_folder(tmp_path, "protein")
    (output / "ligand_smiles.txt").write_text(_SMILES + "\n", encoding="utf-8")
    record = screen_generation(output, binder_kind="protein", target_id="P05113")
    assert "toxicology" not in record
    assert not (output / "toxicology.tsv").exists()


def test_release_files_already_on_disk_are_not_fetched_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in octant.OCTANT_FILES:
        (tmp_path / name).write_text("a\tb\n", encoding="utf-8")

    def _refuse(url: str, timeout_s: int) -> bytes:
        raise AssertionError(f"refetched {url}")

    monkeypatch.setattr(octant, "_download", _refuse)
    sizes = octant.ensure_release(tmp_path)
    assert set(sizes) == set(octant.OCTANT_FILES)
    assert all(size > 0 for size in sizes.values())
