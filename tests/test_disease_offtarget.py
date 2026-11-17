"""Alignment off-target via MMseqs2: parsing, on-target exclusion, verdicts, roll-up.

MMseqs2 is not invoked; its search is mocked by writing the table the parser
reads. These run in milliseconds on CPU with no proteome download.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from rfproteina.disease import offtarget
from rfproteina.disease.screening_tools import ScreeningInputError, ToolResult


def test_accession_is_pulled_from_a_uniprot_header_id() -> None:
    assert offtarget.accession_of("sp|P69905|HBA_HUMAN") == "P69905"
    assert offtarget.accession_of("tr|A0A000|X") == "A0A000"
    assert offtarget.accession_of("plain_id") == "plain_id"


def test_parse_collects_all_hits_per_query_and_skips_bad_rows() -> None:
    table = "\n".join(
        [
            "d1\tsp|P1|A\t0.80\t180\t0.9\t0.85\t1e-40\t350",
            "d1\tsp|P2|B\t0.40\t120\t0.6\t0.6\t1e-5\t120",
            "d1\tbad\tNA\tNA\tNA\tNA\tNA\tNA",
        ]
    )
    hits = offtarget.parse_search_table(table)
    assert len(hits["d1"]) == 2


def test_worst_offtarget_excludes_the_on_target_and_applies_thresholds() -> None:
    hits = [
        offtarget.OffTargetHit("d1", "sp|P1|ON", 0.99, 200, 1.0, 1.0, 0.0, 400.0),
        offtarget.OffTargetHit("d1", "sp|P2|OFF", 0.80, 180, 0.9, 0.85, 1e-40, 350.0),
        offtarget.OffTargetHit("d1", "sp|P3|WEAK", 0.30, 60, 0.3, 0.3, 1.0, 40.0),
    ]
    worst, skipped = offtarget.worst_offtarget(
        hits, ignore_targets=frozenset({"P1"}), identity_min=0.5, coverage_min=0.5, evalue_max=1e-3
    )
    assert skipped is True
    assert worst is not None
    assert worst.target == "sp|P2|OFF"


def test_verdict_hits_on_a_qualifying_offtarget_and_clears_without_one() -> None:
    off = [offtarget.OffTargetHit("d1", "sp|P2|OFF", 0.80, 180, 0.9, 0.85, 1e-40, 350.0)]
    weak = [offtarget.OffTargetHit("d1", "sp|P3|WEAK", 0.30, 60, 0.3, 0.3, 1.0, 40.0)]
    assert offtarget.verdict(off)["state"] == "hit"
    assert offtarget.verdict(off)["measurement"] == 0.8
    assert offtarget.verdict(weak)["state"] == "clear"
    assert offtarget.verdict([])["state"] == "clear"


def test_scan_raises_without_sequences(tmp_path: Path) -> None:
    with pytest.raises(ScreeningInputError, match="no_design_sequences"):
        offtarget.offtarget_scan([("d1", "")], tmp_path / "p.fasta", tmp_path / "work")


def test_scan_is_unavailable_when_mmseqs_is_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(offtarget, "mmseqs_available", lambda: False)
    with pytest.raises(offtarget.OffTargetUnavailableError, match="mmseqs_not_installed"):
        offtarget.offtarget_scan([("d1", "ACDEFG")], tmp_path / "p.fasta", tmp_path / "work")


def test_scan_is_unavailable_when_the_proteome_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(offtarget, "mmseqs_available", lambda: True)
    with pytest.raises(offtarget.OffTargetUnavailableError, match="proteome_missing"):
        offtarget.offtarget_scan([("d1", "ACDEFG")], tmp_path / "nope.fasta", tmp_path / "work")


def test_scan_builds_a_record_per_design_from_mocked_mmseqs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proteome = tmp_path / "human.fasta"
    proteome.write_text(">sp|P2|OFF\nACDEFGHIKL\n", encoding="utf-8")

    def fake_run_tool(command: Sequence[str], **_kwargs: object) -> ToolResult:
        out = Path(command[4])
        out.write_text("d1\tsp|P2|OFF\t0.80\t180\t0.9\t0.85\t1e-40\t350\n", encoding="utf-8")
        return ToolResult(True, 0, "", "")

    monkeypatch.setattr(offtarget, "mmseqs_available", lambda: True)
    monkeypatch.setattr(offtarget, "run_tool", fake_run_tool)
    records = offtarget.offtarget_scan([("d1", "ACDEFGHIKL"), ("d2", "WWWWWWWWWW")], proteome, tmp_path / "work")
    assert records["d1"]["state"] == "hit"
    assert records["d2"]["state"] == "clear"


def test_rollup_is_a_hit_when_any_design_has_an_offtarget() -> None:
    records = {
        "d1": offtarget.verdict([offtarget.OffTargetHit("d1", "sp|P2|OFF", 0.8, 180, 0.9, 0.85, 1e-40, 350.0)]),
        "d2": offtarget.verdict([]),
    }
    assert offtarget.rollup(records)["state"] == "hit"


def test_structure_offtarget_requirements_names_the_unavailable_database() -> None:
    reqs = offtarget.structure_offtarget_requirements()
    assert "afdb_proteome" in reqs["download"]
    assert reqs["licence"] == "CC-BY-4.0"
