"""Measured red-cell evidence is attributable, bounded, and never a prediction."""

from __future__ import annotations

import hashlib
import json

import pytest

from rfproteina.disease import corpus_provenance
from rfproteina.disease import peptide_rbc as rbc

_HEADER = "\t".join(rbc.HEADER)


def _row(**fields):
    # Synthetic parser fixtures, not assay settings or measurements from a lab.
    values = {
        "candidate_id": "peptide_1", "observation_id": "read1", "formulation_id": "form_1",
        "source_id": "lab_record_1", "species": "human", "matrix": "washed_rbc", "context_id": "context1",
        "concentration": "1", "concentration_unit": "uM", "exposure_minutes": "1",
        "hemolysis_percent": "0", "controls_qc": "accepted",
    }
    return "\t".join({**values, **fields}[key] for key in rbc.HEADER)


def _write(directory, *rows):
    raw = ("\ufeff" + "\r\n".join((_HEADER, *rows)) + "\r\n").encode("utf-8")
    (directory / rbc.MANIFEST).write_bytes(raw)
    return raw


def test_keeps_measured_records_separate_and_records_source_snapshot(tmp_path):
    raw = _write(
        tmp_path,
        _row(),
        _row(observation_id="read2", formulation_id="form_2", species="rabbit", matrix="whole_blood",
             concentration_unit="mg/mL", hemolysis_percent="23.25", context_id="context2"),
        _row(observation_id="read3", hemolysis_percent="100"),
    )
    report = rbc.screen_run(tmp_path, known_candidates={"peptide_1"})
    assert report["status"] == "reported" and report["attempted"] == report["reported"] == 3
    assert report["human_reported"] == 2
    assert report["input_sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["unit"] == "percent" and report["endpoint"] == "observed_hemolysis"
    first, middle, last = report["observations"]
    assert (first["hemolysis_percent"], last["hemolysis_percent"]) == (0., 100.)
    assert (middle["species"], middle["matrix"], middle["formulation_id"], middle["context_id"]) == (
        "rabbit", "whole_blood", "form_2", "context2",
    )
    assert middle["concentration"] == 1. and middle["concentration_unit"] == "mg/mL"
    assert middle["hemolysis_percent"] == 23.25
    assert {row["source_id"] for row in report["observations"]} == {"lab_record_1"}
    assert report["corpus_provenance"]["overlap"] == corpus_provenance.USER_SUPPLIED_MEASUREMENTS
    assert not {"call", "threshold", "verdict", "risk_score", "consensus"} & report.keys()


def test_known_run_and_explicit_candidate_scope_are_not_interchangeable(tmp_path):
    _write(tmp_path, _row(), _row(candidate_id="peptide_2"), _row(candidate_id="foreign"))
    allowed = {"peptide_1", "peptide_2", "unmeasured"}
    report = rbc.screen_run(tmp_path, known_candidates=allowed)
    assert report["status"] == "partial" and report["attempted"] == 3 and report["reported"] == 2
    assert report["observations"][-1]["blocker"] == "peptide_rbc_candidate_unknown"
    selected = rbc.screen_run(tmp_path, known_candidates=allowed, candidate="peptide_2")
    assert selected["scope"] == "candidate" and selected["reported"] == 1
    assert selected["observations"][0]["candidate_id"] == "peptide_2"
    missing = rbc.screen_run(tmp_path, known_candidates=allowed, candidate="unmeasured")
    assert missing["blocker"] == "peptide_rbc_candidate_not_nominated"
    for candidate in ("", "foreign"):
        refused = rbc.screen_run(tmp_path, known_candidates=allowed, candidate=candidate)
        assert refused["scope"] == "candidate" and refused["blocker"] == "peptide_rbc_candidate_unknown"
    assert rbc.screen_run(tmp_path)["blocker"] == "peptide_rbc_candidates_unavailable"


@pytest.mark.parametrize(("changes", "code"), [
    ({"formulation_id": ""}, "identifier_invalid"),
    ({"context_id": "../local"}, "identifier_invalid"),
    ({"source_id": "https://example.invalid/result"}, "identifier_invalid"),
    ({"species": "unknown"}, "species_unsupported"),
    ({"matrix": "unspecified"}, "matrix_unsupported"),
    ({"controls_qc": "failed"}, "controls_failed"),
    ({"controls_qc": "not_reported"}, "controls_unreported"),
    ({"controls_qc": ""}, "controls_unreported"),
    ({"controls_qc": "probably"}, "controls_invalid"),
    ({"concentration_unit": "M"}, "unit_unsupported"),
    ({"concentration": "0"}, "exposure_invalid"),
    ({"exposure_minutes": "-0.01"}, "exposure_invalid"),
    ({"concentration": "1e1000"}, "exposure_invalid"),
    ({"exposure_minutes": "1e-1000"}, "exposure_invalid"),
    ({"hemolysis_percent": "NaN"}, "readout_invalid"),
    ({"hemolysis_percent": "<2"}, "readout_invalid"),
    ({"hemolysis_percent": "5%"}, "readout_invalid"),
    ({"hemolysis_percent": "-1e-1000"}, "readout_invalid"),
    ({"hemolysis_percent": "1e-1000"}, "readout_invalid"),
    ({"hemolysis_percent": "100.000000000000000000001"}, "readout_invalid"),
    ({"hemolysis_percent": "99.999999999999999999999"}, "readout_invalid"),
])
def test_ineligible_rows_are_withheld_not_analyzed_or_clamped(tmp_path, changes, code):
    _write(tmp_path, _row(**changes))
    report = rbc.screen_run(tmp_path, known_candidates={"peptide_1"})
    assert report["status"] == "untested" and report["reported"] == 0
    assert report["blocker"] == "peptide_rbc_no_qualified_observations"
    row, = report["observations"]
    assert row["blocker"] == "peptide_rbc_" + code
    assert set(row) <= {"candidate_id", "observation_id", "formulation_id", "status", "blocker"}
    assert "https://" not in json.dumps(row)


def test_duplicate_refusal_is_including_malformed_second_rows(tmp_path):
    _write(tmp_path, _row(), "peptide_1\tread1", _row(candidate_id="peptide_2"))
    report = rbc.screen_run(tmp_path, known_candidates={"peptide_1", "peptide_2"})
    assert report["status"] == "partial" and report["reported"] == 1
    assert [row.get("blocker") for row in report["observations"]] == [
        "peptide_rbc_duplicate_observation", "peptide_rbc_row_width_invalid", None,
    ]
    assert report["observations"][2]["candidate_id"] == "peptide_2"


def test_preflight_observations_use_the_same_validation(tmp_path):
    _write(tmp_path, _row(), _row(observation_id="r2", controls_qc="failed"))
    rows, blocker = rbc.read_observations(tmp_path, known_candidates={"peptide_1"})
    assert not blocker
    assert rows == rbc.screen_run(tmp_path, known_candidates={"peptide_1"})["observations"]


def test_limits_apply_even_when_requesting_one_candidate(tmp_path):
    _write(tmp_path, *[_row(observation_id=f"id{i}") for i in range(rbc.MAX_ROWS + 1)])
    report = rbc.screen_run(tmp_path, candidate="peptide_1", known_candidates={"peptide_1"})
    assert report["blocker"] == "peptide_rbc_over_limit"
    assert report["attempted"] == 0 and report["observations"] == []
    (tmp_path / rbc.MANIFEST).write_bytes(b" " * (rbc.MAX_BYTES + 1))
    assert rbc.screen_run(tmp_path)["blocker"] == "peptide_rbc_over_limit"
    assert "input_sha256" not in rbc.screen_run(tmp_path)


def test_missing_unreadable_and_wrong_schema_are_distinguished(tmp_path):
    target = tmp_path / rbc.MANIFEST
    assert rbc.screen_run(tmp_path)["blocker"] == "peptide_rbc_missing"
    target.write_bytes(b"\xff")
    assert rbc.screen_run(tmp_path)["blocker"] == "peptide_rbc_unreadable"
    target.write_text(_HEADER + "\tsequence\n", encoding="utf-8")
    assert rbc.screen_run(tmp_path)["blocker"] == "peptide_rbc_header_invalid"
    _write(tmp_path)
    assert rbc.screen_run(tmp_path)["blocker"] == "peptide_rbc_empty"


def test_manifest_symlinks_are_not_followed(tmp_path):
    outside = tmp_path / "ignored.tsv"
    outside.write_text(_HEADER + "\n" + _row(), encoding="utf-8")
    try:
        (tmp_path / rbc.MANIFEST).symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    report = rbc.screen_run(tmp_path, known_candidates={"peptide_1"})
    assert report["blocker"] == "peptide_rbc_unreadable" and "input_sha256" not in report
