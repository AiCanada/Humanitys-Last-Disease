"""Badapple scaffold observations retain molecule attribution and evidence coverage."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing

import pytest

from rfproteina.disease import badapple2, badapple2_runtime

_PYRIDINE = "c1ccncc1"
_METHYLPYRIDINE = "Cc1ccncc1"
_CYCLOHEXANE = "C1CCCCC1"


def _table(directory, rows):
    (directory / "ligand_candidates.tsv").write_text(
        "candidate_id\tsmiles\n" + "".join(f"{name}\t{smiles}\n" for name, smiles in rows),
        encoding="utf-8-sig",
    )


def _match(**overrides):
    return {
        "id": 7,
        "pscore": 0.0,
        "in_drug": True,
        "nass_active": 0,
        "nass_tested": 90,
        "ncpd_active": 0,
        "ncpd_tested": 15,
        "nsub_active": 0,
        "nsub_tested": 16,
        "nsam_active": 0,
        "nsam_tested": 120,
        **overrides,
    }


def _scaffold(smiles, *matches):
    return {"scafsmi": smiles, "matches": list(matches)}


def _forbidden(*_args):
    pytest.fail("ineligible or unassigned compounds reached the Badapple lookup")


def test_named_molecules_keep_identity_even_when_lookup_order_is_reversed(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    _table(tmp_path, [("first", _METHYLPYRIDINE), ("alias", "[CH3:9]c1ccncc1"), ("other", _CYCLOHEXANE)])
    (tmp_path / "ligand_smiles.txt").write_text("C1CCCC1 fallback\n", encoding="utf-8")
    calls = []

    def lookup(smiles):
        calls.append(set(smiles))
        observations = {
            _METHYLPYRIDINE: _scaffold(_PYRIDINE, _match()),
            _CYCLOHEXANE: _scaffold(_CYCLOHEXANE, _match(id=8, in_drug=False)),
        }
        return {key: {"scaffolds": [observations[key]]} for key in reversed(smiles)}

    monkeypatch.setattr(badapple2, "_lookup", lookup)
    record = badapple2.screen_run(tmp_path)
    assert record["status"] == "complete" and calls == [{_METHYLPYRIDINE, _CYCLOHEXANE}]
    original, alias, other = record["molecules"]
    assert [row["candidate_id"] for row in record["molecules"]] == ["first", "alias", "other"]
    assert original["input_smiles"] == alias["input_smiles"] == _METHYLPYRIDINE
    assert original["scaffolds"] == alias["scaffolds"]
    assert other["input_smiles"] == _CYCLOHEXANE and other["scaffolds"] != alias["scaffolds"]
    assert original["scaffolds"][0]["pscore"] == 0.0 and original["scaffolds"][0]["status"] == "reported"
    assert {"verdict", "threshold", "safety_call", "predicted_probability"}.isdisjoint(record)

    selected = badapple2.screen_run(tmp_path, candidate="alias")
    assert selected["status"] == "complete" and selected["molecules"] == [alias]
    assert calls[-1] == {_METHYLPYRIDINE}


@pytest.mark.parametrize(
    ("matches", "state", "status"),
    [
        ([_match()], "reported", "complete"),
        ([_match(pscore=None)], "unscored", "untested"),
        ([], "not_in_corpus", "untested"),
        ([_match(), _match(id=8, nass_tested=100)], "ambiguous", "untested"),
    ],
)
def test_zero_pscore_is_distinct_from_null_absent_or_ambiguous_evidence(
    tmp_path, monkeypatch, matches, state, status,
):
    pytest.importorskip("rdkit")
    _table(tmp_path, [("candidate", _METHYLPYRIDINE)])
    monkeypatch.setattr(badapple2, "_lookup", lambda _smiles: {
        _METHYLPYRIDINE: {"scaffolds": [_scaffold(_PYRIDINE, *matches)]},
    })
    result = badapple2.screen_run(tmp_path)
    assert result["status"] == status
    (row,) = result["molecules"]
    (scaffold,) = row["scaffolds"]
    assert scaffold["scafsmi"] == _PYRIDINE and scaffold["status"] == state
    if status == "complete":
        assert scaffold["pscore"] == 0.0 and scaffold["nass_tested"] == 90
    else:
        assert row["blocker"] and scaffold.get("pscore") is None


def test_scaffold_coverage_is_reported_per_scaffold_without_pooling(tmp_path, monkeypatch):
    chem = pytest.importorskip("rdkit.Chem")
    smiles = chem.MolToSmiles(chem.MolFromSmiles("c1ccncc1-c1ccncc1"))
    _table(tmp_path, [("candidate", smiles)])
    monkeypatch.setattr(badapple2, "_lookup", lambda _smiles: {smiles: {"scaffolds": [
        _scaffold(_PYRIDINE, _match()), _scaffold(smiles),
    ]}})
    result = badapple2.screen_run(tmp_path)
    (row,) = result["molecules"]
    assert result["status"] == row["status"] == "partial"
    assert {entry["scafsmi"]: entry["status"] for entry in row["scaffolds"]} == {
        _PYRIDINE: "reported", smiles: "not_in_corpus",
    }
    assert {"pscore", "nass_tested", "verdict"}.isdisjoint(row)


@pytest.mark.parametrize(
    "damage",
    [
        {"id": True}, {"pscore": True}, {"pscore": "0.0"}, {"pscore": float("nan")},
        {"pscore": -1.0}, {"pscore": 100001.0}, {"nass_tested": False}, {"nass_tested": -1},
        {"nass_active": 91}, {"in_drug": "false"},
    ],
)
def test_malformed_database_row_does_not_hide_other_compound(tmp_path, monkeypatch, damage):
    pytest.importorskip("rdkit")
    _table(tmp_path, [("malformed", _METHYLPYRIDINE), ("valid", _CYCLOHEXANE)])
    monkeypatch.setattr(badapple2, "_lookup", lambda _smiles: {
        _METHYLPYRIDINE: {"scaffolds": [_scaffold(_PYRIDINE, _match(**damage))]},
        _CYCLOHEXANE: {"scaffolds": [_scaffold(_CYCLOHEXANE, _match(id=8))]},
    })
    result = badapple2.screen_run(tmp_path)
    assert result["status"] == "partial"
    malformed, valid = result["molecules"]
    assert malformed["status"] == "untested" and malformed["blocker"]
    assert not any(scaffold.get("pscore") is not None for scaffold in malformed.get("scaffolds", ()))
    assert valid["status"] == "complete" and valid["scaffolds"][0]["pscore"] == 0.0


@pytest.mark.parametrize(
    "compound_result",
    [
        None,
        {"scaffolds": [{"scafsmi": _PYRIDINE, "matches": None}]},
        {"scaffolds": [_scaffold(_PYRIDINE, _match()), _scaffold(_PYRIDINE, _match())]},
        {"blocker": ["wrong shape"]},
    ],
)
def test_missing_or_inconsistent_lookup_payload_never_claims_coverage(tmp_path, monkeypatch, compound_result):
    pytest.importorskip("rdkit")
    _table(tmp_path, [("affected", _METHYLPYRIDINE), ("unaffected", _CYCLOHEXANE)])
    response = {
        _CYCLOHEXANE: {"scaffolds": [_scaffold(_CYCLOHEXANE, _match(id=8))]},
        _METHYLPYRIDINE: compound_result,
    }
    monkeypatch.setattr(badapple2, "_lookup", lambda _smiles: response)
    result = badapple2.screen_run(tmp_path)
    assert result["status"] == "partial"
    affected, unaffected = result["molecules"]
    assert affected["status"] == "untested" and affected["blocker"]
    assert unaffected["status"] == "complete"


def test_lookup_with_foreign_result_identity_cannot_publish_the_batch(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    _table(tmp_path, [("nomination", _METHYLPYRIDINE)])
    monkeypatch.setattr(badapple2, "_lookup", lambda _smiles: {
        _CYCLOHEXANE: {"scaffolds": [_scaffold(_CYCLOHEXANE, _match(id=8))]},
    })
    result = badapple2.screen_run(tmp_path)
    assert result["status"] == "untested"
    (row,) = result["molecules"]
    assert row["candidate_id"] == "nomination" and row["input_smiles"] == _METHYLPYRIDINE
    assert row["blocker"] == "badapple2_invalid_report" and row["scaffolds"] == []


def test_acyclic_and_duplicate_inputs_are_attributed_without_provider(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    _table(tmp_path, [
        ("acyclic", "CCO"), ("same", _METHYLPYRIDINE), ("same", _CYCLOHEXANE),
        ("salt", _METHYLPYRIDINE + ".[Na+]"), ("large", "C" * 75 + "C1CCCCC1"),
    ])
    monkeypatch.setattr(badapple2, "_lookup", _forbidden)
    result = badapple2.screen_run(tmp_path)
    assert result["status"] == "untested"
    assert [row["candidate_id"] for row in result["molecules"]] == ["acyclic", "same", "same", "salt", "large"]
    assert all(row["status"] == "untested" and row["blocker"] for row in result["molecules"])
    assert result["molecules"][1]["blocker"] == result["molecules"][2]["blocker"]
    assert result["molecules"][0]["blocker"] != result["molecules"][1]["blocker"]


def test_selection_scope_budget_does_not_silently_drop_nominations(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    _table(tmp_path, [(f"nomination_{index}", _METHYLPYRIDINE) for index in range(33)])
    monkeypatch.setattr(badapple2, "_lookup", _forbidden)
    too_many = badapple2.screen_run(tmp_path)
    assert too_many["status"] == "untested" and too_many["blocker"]

    def lookup(smiles):
        assert smiles == [_METHYLPYRIDINE]
        return {_METHYLPYRIDINE: {"scaffolds": [_scaffold(_PYRIDINE, _match())]}}

    monkeypatch.setattr(badapple2, "_lookup", lookup)
    selected = badapple2.screen_run(tmp_path, candidate="nomination_32")
    assert selected["status"] == "complete"
    assert [row["candidate_id"] for row in selected["molecules"]] == ["nomination_32"]


def test_authoritative_malformed_table_never_falls_back_to_run_ligand(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    legacy = tmp_path / "ligand_smiles.txt"
    legacy.write_text(f"{_METHYLPYRIDINE} legacy\n", encoding="utf-8")
    table = tmp_path / "ligand_candidates.tsv"
    table.write_text("name\tsmiles\nwrong\tCC\n", encoding="utf-8")
    monkeypatch.setattr(badapple2, "_lookup", _forbidden)
    assert badapple2.screen_run(tmp_path)["status"] == "untested"
    table.unlink()
    assert badapple2.screen_run(tmp_path, candidate="run_ligand")["status"] == "untested"

    monkeypatch.setattr(badapple2, "_lookup", lambda _smiles: {
        _METHYLPYRIDINE: {"scaffolds": [_scaffold(_PYRIDINE, _match())]},
    })
    assert badapple2.screen_run(tmp_path)["molecules"][0]["input_smiles"] == _METHYLPYRIDINE


def test_cli_selection_reports_the_same_nomination_as_the_screen(tmp_path, monkeypatch, capsys):
    from scripts import screen_extras

    pytest.importorskip("rdkit")
    _table(tmp_path, [("first", _METHYLPYRIDINE), ("second", _CYCLOHEXANE)])

    def lookup(smiles):
        assert smiles == [_CYCLOHEXANE]
        return {_CYCLOHEXANE: {"scaffolds": [_scaffold(_CYCLOHEXANE, _match())]}}

    monkeypatch.setattr(badapple2, "_lookup", lookup)
    assert screen_extras.main(["badapple2", str(tmp_path), "--candidate", "second"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == badapple2.screen_run(tmp_path, candidate="second")
    assert [row["candidate_id"] for row in result["molecules"]] == ["second"]


def test_saved_view_withholds_tampered_evidence_and_ignores_claimed_verdict(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    _table(tmp_path, [("candidate", _METHYLPYRIDINE)])
    monkeypatch.setattr(badapple2, "_lookup", lambda _smiles: {
        _METHYLPYRIDINE: {"scaffolds": [_scaffold(_PYRIDINE, _match())]},
    })
    report = badapple2.screen_run(tmp_path)
    assert badapple2.report_view({**report, "verdict": "safe"}) == report
    (sample,) = report["molecules"]
    (scaffold,) = sample["scaffolds"]
    for damage in (
        {"dump_sha256": "0" * 64},
        {"molecules": [sample, sample]},
        {"molecules": [{**sample, "scaffolds": [{**scaffold, "pscore": None}]}]},
        {"molecules": [{**sample, "scaffolds": [{**scaffold, "nsam_active": 121}]}]},
    ):
        projection = badapple2.report_view({**report, **damage})
        assert projection["status"] == "untested" and projection["blocker"] == "badapple2_invalid_report"
        assert projection["molecules"] == []


@pytest.fixture
def local_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "local snapshot"
    directory = root / badapple2_runtime.SNAPSHOT
    directory.mkdir(parents=True)
    path = directory / "scaffolds.sqlite"
    metadata = {
        "schema_version": 1, "snapshot": badapple2_runtime.SNAPSHOT,
        "dump_sha256": badapple2_runtime.DUMP_SHA256, "row_count": 3,
    }
    counts = ", ".join(f"{column} INTEGER" for column in badapple2_runtime.COUNT_FIELDS)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        db.executemany("INSERT INTO metadata VALUES (?, ?)", [(key, str(value)) for key, value in metadata.items()])
        db.execute(f"CREATE TABLE scaffold (id INTEGER PRIMARY KEY, scafsmi TEXT, pscore REAL, in_drug INTEGER, {counts})")
        for molecule, row in (
            (_PYRIDINE, _match()),
            (_CYCLOHEXANE, _match(id=8, pscore=None, in_drug=None)),
            (_CYCLOHEXANE, _match(id=9, in_drug=False)),
        ):
            db.execute(
                "INSERT INTO scaffold VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (row["id"], molecule, row["pscore"], row["in_drug"],
                 *(row[column] for column in badapple2_runtime.COUNT_FIELDS)),
            )
    receipt = directory / "receipt.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt.write_text(json.dumps({**metadata, "sqlite_sha256": digest}), encoding="utf-8")
    monkeypatch.setattr(badapple2_runtime, "_root", lambda: root)
    monkeypatch.setattr(badapple2_runtime, "_chemistry_available", lambda: True)
    return path


def test_verified_local_database_keeps_native_rows_without_choosing_an_ambiguous_match(local_snapshot, monkeypatch):
    monkeypatch.setattr(badapple2_runtime, "_scaffolds", lambda _smiles: {
        _METHYLPYRIDINE: {"keys": [_PYRIDINE, "C1CCCC1"]},
        _CYCLOHEXANE: {"keys": [_CYCLOHEXANE]},
    })
    assert badapple2_runtime.installed()
    result = badapple2_runtime.lookup([_METHYLPYRIDINE, _CYCLOHEXANE])
    assert result[_METHYLPYRIDINE]["scaffolds"] == [_scaffold(_PYRIDINE, _match()), _scaffold("C1CCCC1")]
    (ambiguous,) = result[_CYCLOHEXANE]["scaffolds"]
    assert {row["id"] for row in ambiguous["matches"]} == {8, 9}
    assert {row["pscore"] for row in ambiguous["matches"]} == {None, 0.0}


def test_modified_local_database_is_unavailable_and_does_not_return_scores(local_snapshot):
    assert badapple2_runtime.installed()
    with closing(sqlite3.connect(local_snapshot)) as db, db:
        db.execute("UPDATE scaffold SET pscore = 50 WHERE id = 7")
    assert not badapple2_runtime.installed()
