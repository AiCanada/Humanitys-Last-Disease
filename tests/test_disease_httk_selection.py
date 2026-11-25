"""Standalone EPA httk exposes the same report as the finished-run screen."""

import json

import pytest

from rfproteina.disease import httk
from scripts import screen_extras


@pytest.mark.parametrize(("status", "code"), [("reported", 0), ("partial", 2), ("untested", 2)])
def test_standalone_httk_returns_coverage_and_candidate_scope(tmp_path, capsys, monkeypatch, status, code):
    def measure(directory, candidate):
        assert directory == tmp_path and candidate == "caffeine"
        return {"method": httk.METHOD, "status": status, "scope": "candidate"}

    monkeypatch.setattr(httk, "measure", measure)
    assert screen_extras.main(["httk", str(tmp_path), "--candidate", "caffeine"]) == code
    assert json.loads(capsys.readouterr().out) == {"method": httk.METHOD, "scope": "candidate", "status": status}


def test_standalone_method_id_handles_a_missing_manifest_without_a_runtime(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(httk, "runtime_path", lambda: pytest.fail("Tried to run an unprepared EPA httk scenario"))
    assert screen_extras.main(["epa_httk", str(tmp_path)]) == 2
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "untested" and record["scope"] == "run"
    assert record["attempted"] == record["reported"] == 0 and record["blocker"] == "httk_manifest_missing"


@pytest.mark.parametrize("reference_source", ["fixture", "installed"])
def test_standalone_and_picker_reject_one_mislabeled_ligand(tmp_path, capsys, monkeypatch, reference_source):
    pytest.importorskip("rdkit")
    from rfproteina.disease import httk_identity, screen_selection

    cas, dtxsid = "58-08-2", "DTXSID0020232"
    key = "RYYVLZVUVIJVGH-UHFFFAOYSA-N"
    if reference_source == "fixture":
        monkeypatch.setattr(httk_identity, "load_index", lambda: {(cas, dtxsid): key})
    else:
        if not httk_identity.index_path().exists():
            pytest.skip("The optional DSSTox reference index is not installed")
        assert httk_identity.load_index()[cas, dtxsid] == key
    monkeypatch.setattr(httk, "runtime_path", lambda: pytest.fail("A mismatched ligand reached R"))
    (tmp_path / httk.MANIFEST).write_text(json.dumps({
        "scenario": httk.SCENARIO, "candidates": {"compound": {"cas": cas, "dtxsid": dtxsid}},
    }))
    (tmp_path / "ligand_candidates.tsv").write_text("candidate_id\tsmiles\ncompound\tCC(=O)Nc1ccc(O)cc1\n")
    choice, = screen_selection.plan(tmp_path, "ligand", example="compound", available={httk.METHOD_ID: True})
    assert not choice.applicable and "httk_structure_mismatch" in choice.reason
    assert screen_extras.main(["httk", str(tmp_path), "--candidate", "compound"]) == 2
    row, = json.loads(capsys.readouterr().out)["members"]
    assert row["blocker"] == "httk_structure_mismatch" and "median" not in row
