"""Check attribution and scientific output for local MetFrag scoring."""

from __future__ import annotations

import csv
import io
import json
from hashlib import sha256

import pytest

from rfproteina.disease import metfrag

NICOTINE = "CN1CCC[C@H]1c1cccnc1"
NICOTINE_ALIAS = "CN2CCC[C@H]2c3cccnc3"
POSITIONAL_ISOMER = "CN1CCC[C@H]1c1ccncc1"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
SETTINGS = {
    "version": 1,
    "peaks": "measured.tsv",
    "adduct": "[M+H]+",
    "precursor_mz": 163.1229,
    "precursor_ppm": 5,
    "fragment_ppm": 5,
    "fragment_da": 0.001,
}
PEAKS = "mz\tintensity\n132.0806\t999\n70.065\t3.6\n80.0494\t115.3\n"
RESULT_FIELDS = ("Identifier", "InChIKey", "FragmenterScore", "NoExplPeaks", "NumberPeaksUsed")


def _nominate(directory, *candidates):
    table = "candidate_id\tsmiles\n" + "".join(f"{name}\t{smiles}\n" for name, smiles in candidates)
    (directory / "ligand_candidates.tsv").write_text(table, encoding="utf-8")


def _native_rows(*rows, fields=RESULT_FIELDS):
    result = io.StringIO(newline="")
    writer = csv.writer(result)
    writer.writerow(fields)
    writer.writerows(rows)
    return result.getvalue().encode("ascii")


def _candidate_rows(database):
    return list(csv.DictReader(io.StringIO(database.decode("ascii"))))


@pytest.fixture
def measured(tmp_path):
    pytest.importorskip("rdkit.Chem")
    (tmp_path / metfrag.MANIFEST).write_text(json.dumps(SETTINGS), encoding="utf-8")
    (tmp_path / SETTINGS["peaks"]).write_text(PEAKS, encoding="utf-8")
    _nominate(tmp_path, ("lead", NICOTINE))
    return tmp_path


@pytest.fixture
def no_provider(monkeypatch):
    def reject_call(*args, **kwargs):
        pytest.fail("an invalid input reached the MetFrag runtime")

    monkeypatch.setattr(metfrag.metfrag_runtime, "predict", reject_call)


def test_measured_settings_candidate_attribution_and_ties(measured, monkeypatch):
    _nominate(measured, ("other_mass", CAFFEINE), ("second", POSITIONAL_ISOMER), ("first", NICOTINE))
    peak_data = (measured / SETTINGS["peaks"]).read_bytes()

    def predict(peaks, database, **settings):
        assert settings == {
            key: SETTINGS[key] for key in ("precursor_mz", "precursor_ppm", "fragment_da", "fragment_ppm")
        }
        assert [list(map(float, row.split())) for row in peaks.splitlines()] == [
            [70.065, 3.6],
            [80.0494, 115.3],
            [132.0806, 999.0],
        ]
        first, second = _candidate_rows(database)
        assert [first["Identifier"], second["Identifier"]] == ["m002", "m003"]
        assert first["InChIKey"] != second["InChIKey"]
        assert first["MolecularFormula"] == second["MolecularFormula"] == "C10H14N2"
        assert float(first["MonoisotopicMass"]) == pytest.approx(162.115698448)
        return _native_rows(
            (first["Identifier"], first["InChIKey"], "294.7882757", 2, 3, "1.0"),
            (second["Identifier"], second["InChIKey"], "294.7882757", 2, 3, "1.0"),
            fields=(*RESULT_FIELDS, "Score"),
        )

    monkeypatch.setattr(metfrag.metfrag_runtime, "predict", predict)
    report = metfrag.screen_run(measured)
    assert report["adduct"] == "[M+H]+"
    assert report["spectrum_sha256"] == sha256(peak_data).hexdigest()
    assert report["scope"] == "candidate_table"
    assert report["status"] == "partial"
    first, second, rejected = report["candidates"]
    assert [first["candidate_id"], second["candidate_id"]] == ["first", "second"]
    assert first["candidate_sha256"] != second["candidate_sha256"]
    assert first["fragmenter_score"] == second["fragmenter_score"] == 294.7882757
    assert first["explained_peaks"] == second["explained_peaks"] == 2
    assert first["used_peaks"] == second["used_peaks"] == report["input_peak_count"] == 3
    assert rejected["candidate_id"] == "other_mass"
    assert rejected["status"] == "untested"
    assert rejected["blocker"] == "metfrag_precursor_mismatch"
    assert abs(rejected["precursor_mass_error_ppm"]) > report["precursor_ppm"]
    assert report["comparison"] == {"tested_nominations": 2, "distinct_structures": 2, "top_scoring_structures": 2}
    assert metfrag.report_view(report) == report


def test_explicit_candidate_uses_one_ligand_and_preserves_measured_zero(measured, monkeypatch):
    _nominate(measured, ("wrong_scope", CAFFEINE), ("lead", NICOTINE))

    def predict(_peaks, database, **_settings):
        (one,) = _candidate_rows(database)
        assert one["Identifier"] == "m001"
        return _native_rows(("m001", one["InChIKey"], 0, 0, 3))

    monkeypatch.setattr(metfrag.metfrag_runtime, "predict", predict)
    report = metfrag.screen_run(measured, candidate="lead")
    assert report["status"] == "complete"
    (row,) = report["candidates"]
    assert row["candidate_id"] == "lead"
    assert row["status"] == "no_matches"
    assert row["fragmenter_score"] == 0
    assert row["explained_peaks"] == 0
    assert row["used_peaks"] == 3
    assert report["comparison"] == {"tested_nominations": 1, "distinct_structures": 1, "top_scoring_structures": 0}
    assert metfrag.report_view(report) == report


@pytest.mark.parametrize(("second_score", "top_count"), [(12.0, 2), (11.99999999999, 1), (0.0, 1)])
def test_aliases_share_one_calculation_and_keep_separate_nominations(measured, monkeypatch, second_score, top_count):
    from rfproteina.disease import screen_board

    _nominate(measured, ("wrong", CAFFEINE), ("primary", NICOTINE), ("alias", NICOTINE_ALIAS),
              ("isomer", POSITIONAL_ISOMER))

    def predict(_peaks, database, **_settings):
        primary, isomer = _candidate_rows(database)
        assert [primary["Identifier"], isomer["Identifier"]] == ["m002", "m004"]
        assert primary["SMILES"] != isomer["SMILES"]
        return _native_rows(
            ("m004", isomer["InChIKey"], second_score, int(second_score > 0), 3),
            ("m002", primary["InChIKey"], 12.0, 2, 3),
        )

    monkeypatch.setattr(metfrag.metfrag_runtime, "predict", predict)
    report = metfrag.screen_run(measured)
    assert report["status"] == "partial"
    assert report["comparison"] == {
        "tested_nominations": 3, "distinct_structures": 2, "top_scoring_structures": top_count,
    }
    evidence = {row["candidate_id"]: row for row in report["candidates"]}
    assert set(evidence) == {"wrong", "primary", "alias", "isomer"}
    assert evidence["wrong"]["blocker"] == "metfrag_precursor_mismatch"
    assert evidence["alias"] == {**evidence["primary"], "candidate_id": "alias"}
    assert evidence["isomer"]["candidate_sha256"] != evidence["primary"]["candidate_sha256"]
    assert evidence["isomer"]["fragmenter_score"] == second_score
    assert metfrag.report_view(report) == report
    legacy = {field: value for field, value in report.items() if field != "comparison"}
    assert metfrag.report_view(legacy) == report
    lines = screen_board._metfrag_lines(legacy)
    assert "The 3 tested nominations represent 2 distinct molecular structures." in lines
    if top_count == 2:
        assert "2 distinct structures share the top raw FragmenterScore." in lines
    else:
        assert "The top raw score explained 2 of 3 product peaks." in lines

    for inconsistent in ({"fragmenter_score": 13.0}, {"explained_peaks": 1}):
        rows = [{**row, **inconsistent} if row["candidate_id"] == "alias" else row for row in legacy["candidates"]]
        refused = metfrag.report_view({**legacy, "candidates": rows})
        assert refused["blocker"] == "metfrag_invalid_report" and refused["candidates"] == []
        assert "comparison" not in refused


@pytest.mark.parametrize(("left", "right", "mz", "share_inchi_key"), [
    ("O=c1cccc[nH]1", "Oc1ccccn1", 96.044390, True),
    ("C[C@H](O)CC", "C[C@@H](O)CC", 75.080441, False),
])
def test_distinct_tautomers_and_stereoisomers_reach_the_provider(measured, monkeypatch, left, right, mz, share_inchi_key):
    (measured / metfrag.MANIFEST).write_text(json.dumps({**SETTINGS, "precursor_mz": mz}), encoding="utf-8")
    (measured / SETTINGS["peaks"]).write_text("mz\tintensity\n25\t10\n45\t20\n55\t30\n", encoding="utf-8")
    _nominate(measured, ("left", left), ("right", right))

    def predict(_peaks, database, **_settings):
        first, second = _candidate_rows(database)
        assert (first["InChIKey"] == second["InChIKey"]) is share_inchi_key
        assert first["SMILES"] != second["SMILES"]
        return _native_rows(*[(row["Identifier"], row["InChIKey"], 0, 0, 3) for row in (first, second)])

    monkeypatch.setattr(metfrag.metfrag_runtime, "predict", predict)
    report = metfrag.screen_run(measured)
    assert report["status"] == "complete"
    assert report["comparison"] == {"tested_nominations": 2, "distinct_structures": 2, "top_scoring_structures": 0}
    assert metfrag.report_view(report) == report


@pytest.mark.parametrize(
    "changes",
    [
        {"version": True},
        {"adduct": "[M-H]-"},
        {"precursor_mz": float("nan")},
        {"precursor_ppm": 50.1},
        {"fragment_ppm": "5"},
        {"fragment_da": 0.0, "fragment_ppm": 0.0},
        {"fragment_da": 0.0500001},
        {"peaks": "../measured.tsv"},
        {"peaks": "NUL.tsv"},
        {"provider": "PubChem"},
    ],
)
def test_invalid_manifest_never_reaches_provider(measured, no_provider, changes):
    (measured / metfrag.MANIFEST).write_text(json.dumps({**SETTINGS, **changes}), encoding="utf-8")
    report = metfrag.screen_run(measured)
    assert report["status"] == "untested"
    assert report["blocker"] == "invalid_metfrag_manifest"


def test_duplicate_json_and_missing_manifest_have_explicit_blockers(measured, no_provider):
    path = measured / metfrag.MANIFEST
    path.write_text(json.dumps(SETTINGS)[:-1] + ', "precursor_mz": 163.1229}', encoding="utf-8")
    assert metfrag.screen_run(measured)["blocker"] == "invalid_metfrag_manifest"
    path.unlink()
    assert metfrag.screen_run(measured)["blocker"] == "metfrag_manifest_missing"


@pytest.mark.parametrize(
    "peaks",
    [
        "mz\tintensity\n70.065\t1\n80.0494\t2\n",
        PEAKS + "70.0650\t12\n",
        PEAKS + "163.1229\t3\n",
        PEAKS + "110\t0\n",
        PEAKS + "110\tinfinity\n",
        "mz\tintensity\tunvalidated\n" + "110\t3\tspectrum\n" * 3,
        "mz\tintensity\n" + "".join(f"{mz / 10:g}\t1\n" for mz in range(10, 511)),
    ],
)
def test_invalid_peaks_never_reach_provider(measured, no_provider, peaks):
    (measured / SETTINGS["peaks"]).write_text(peaks, encoding="utf-8")
    assert metfrag.screen_run(measured)["blocker"] == "metfrag_invalid_peaks"


def test_accepts_exact_peak_limit_and_refuses_symlink(measured, no_provider):
    peak_file = measured / SETTINGS["peaks"]
    raw = "mz\tintensity\n" + "".join(f"{mz / 10:g}\t1\n" for mz in range(10, 510))
    peak_file.write_text(raw, encoding="utf-8")
    plan = metfrag.prepare(measured)
    assert plan.input_peak_count == 500
    assert len(plan.peaks.splitlines()) == 500
    moved = peak_file.rename(measured / "alternate.tsv")
    peak_file.symlink_to(moved)
    assert metfrag.screen_run(measured)["blocker"] == "metfrag_peaks_unreadable"


def test_candidate_table_and_explicit_run_ligand_scope_are_distinct(measured, no_provider):
    (measured / "ligand_smiles.txt").write_text(NICOTINE, encoding="utf-8")
    _nominate(measured, ("lead", NICOTINE), ("lead", NICOTINE))
    assert metfrag.screen_run(measured)["blocker"] == "metfrag_duplicate_candidate"
    assert metfrag.screen_run(measured, candidate="lead")["blocker"] == "metfrag_duplicate_candidate"
    assert metfrag.screen_run(measured, candidate="missing")["blocker"] == "metfrag_no_candidates"
    (measured / "ligand_candidates.tsv").unlink()
    assert metfrag.prepare(measured).scope == "run_ligand"
    assert metfrag.screen_run(measured, candidate="run_ligand")["blocker"] == "metfrag_candidate_scope_missing"


def test_candidate_cap_applies_to_the_selected_scope(measured, no_provider):
    _nominate(measured, *((f"lead_{number:02d}", NICOTINE) for number in range(33)))
    assert metfrag.screen_run(measured)["blocker"] == "metfrag_candidates_over_limit"
    (candidate,) = metfrag.prepare(measured, "lead_32").candidates
    assert candidate.name == "lead_32"
    assert not candidate.blocker


def test_no_compatible_candidates_keep_row_level_blockers(measured, no_provider):
    _nominate(measured, ("wrong_mass", CAFFEINE), ("salt", "CCCC[NH3+].[Cl-]"))
    report = metfrag.screen_run(measured)
    assert report["status"] == "untested"
    assert report["blocker"] == "metfrag_no_precursor_matches"
    assert "comparison" not in report
    assert {row["candidate_id"]: row["blocker"] for row in report["candidates"]} == {
        "wrong_mass": "metfrag_precursor_mismatch",
        "salt": "metfrag_unsupported_candidate",
    }
    assert metfrag.report_view(report) == report


@pytest.mark.parametrize(
    "defect",
    [
        "foreign_id",
        "wrong_inchi_key",
        "duplicate_id",
        "missing_row",
        "too_few_used_peaks",
        "too_many_explained_peaks",
        "no_peaks_nonzero_score",
        "nonfinite_score",
        "scaled_score_only",
    ],
)
def test_bad_native_output_is_rejected_as_a_whole(measured, monkeypatch, defect):
    def predict(_peaks, database, **_settings):
        (one,) = _candidate_rows(database)
        native = ("m001", one["InChIKey"], 14, 2, 3)
        if defect == "scaled_score_only":
            return _native_rows(native, fields=("Identifier", "InChIKey", "Score", "NoExplPeaks", "NumberPeaksUsed"))
        bad_rows = {
            "foreign_id": [("m002", *native[1:])],
            "wrong_inchi_key": [("m001", "not-the-candidate-key", *native[2:])],
            "duplicate_id": [native, native],
            "missing_row": [],
            "too_few_used_peaks": [(*native[:4], 2)],
            "too_many_explained_peaks": [(*native[:3], 4, 3)],
            "no_peaks_nonzero_score": [(*native[:3], 0, 3)],
            "nonfinite_score": [(*native[:2], "NaN", *native[3:])],
        }
        return _native_rows(*bad_rows[defect])

    monkeypatch.setattr(metfrag.metfrag_runtime, "predict", predict)
    report = metfrag.screen_run(measured)
    assert report["status"] == "untested"
    assert report["blocker"] == "metfrag_invalid_report"
    assert report["candidates"] == []
    assert "comparison" not in report


def test_projected_report_omits_raw_inputs_and_checks_measurement_consistency(measured, monkeypatch):
    def predict(_peaks, database, **_settings):
        (one,) = _candidate_rows(database)
        return _native_rows(("m001", one["InChIKey"], 14, 2, 3))

    monkeypatch.setattr(metfrag.metfrag_runtime, "predict", predict)
    report = metfrag.screen_run(measured)
    raw = {
        **report,
        "peaks_path": "/private/work/spectrum.tsv",
        "raw_spectrum": PEAKS,
        "candidates": [{**report["candidates"][0], "smiles": NICOTINE, "inchi_key": "private-structure-key"}],
    }
    assert metfrag.report_view(raw) == report
    raw["candidates"][0]["precursor_mass_error_ppm"] = 0.0
    projected = metfrag.report_view(raw)
    assert projected["status"] == "untested"
    assert projected["blocker"] == "metfrag_invalid_report"


def test_picker_only_recommends_installed_mass_compatible_scopes(measured, no_provider):
    from rfproteina.disease import screen_selection

    def choice(candidate=None, installed=True):
        (row,) = screen_selection.plan(measured, "ligand", example=candidate, available={metfrag.METHOD_ID: installed})
        return row

    _nominate(measured, ("outside_window", CAFFEINE), ("lead", NICOTINE))
    batch = choice()
    assert batch.recommended and "1 of 2" in batch.reason and "3 supplied peaks" in batch.reason
    assert "Only one structure is eligible" in batch.reason
    usable = choice("lead")
    assert usable.installed and usable.applicable and usable.recommended
    assert not choice("outside_window").applicable
    assert not choice("outside_window").recommended
    assert not choice("lead", installed=False).recommended
    _nominate(measured, ("lead", NICOTINE), ("alternate_label", NICOTINE_ALIAS), ("outside_window", CAFFEINE))
    aliases = choice()
    assert aliases.recommended and "2 of 3" in aliases.reason and "3 supplied peaks" in aliases.reason
    assert "Only one structure is eligible" in aliases.reason
    (measured / metfrag.MANIFEST).unlink()
    assert not choice().applicable
    assert not choice().recommended
