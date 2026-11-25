"""Exercise pH-scoped ligand-state preparation through the finished-run reader."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rfproteina.disease import dimorphite


def _manifest(directory, ph_min=6.8, ph_max=7.4):
    (directory / dimorphite.MANIFEST).write_text(
        json.dumps({"version": 1, "ph_min": ph_min, "ph_max": ph_max}),
        encoding="utf-8",
    )


def _table(directory, rows):
    (directory / "ligand_candidates.tsv").write_text(
        "candidate_id\tsmiles\n" + "".join(f"{name}\t{smiles}\n" for name, smiles in rows),
        encoding="utf-8-sig",
    )


def _forbid_provider(*_args):
    pytest.fail("the invalid nomination or pH manifest reached enumeration")


def test_named_candidate_uses_assay_ph_and_preserves_aliases(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    _manifest(tmp_path, 7.0, 7.0)
    _table(tmp_path, [("base", "CN"), ("base_alias", "NC"), ("acid", "OC(C)=O")])
    (tmp_path / "ligand_smiles.txt").write_text("CCC legacy molecule\n", encoding="utf-8")
    calls = []

    def enumerate_states(ph_min, ph_max, inputs):
        calls.append((ph_min, ph_max, list(inputs)))
        examples = {"CN": ["C[NH3+]", "CN"], "CC(=O)O": ["CC(=O)[O-]"]}
        return {key: {"states": examples[key]} for key in reversed(inputs)}

    monkeypatch.setattr(dimorphite, "_enumerate", enumerate_states)
    result = dimorphite.screen_run(tmp_path)
    assert result["status"] == "complete"
    assert len(calls) == 1
    low, high, inferred = calls[0]
    assert (low, high) == (7.0, 7.0) and sorted(inferred) == ["CC(=O)O", "CN"]
    assert result["method"] == dimorphite.METHOD and result["endpoint"] == dimorphite.ENDPOINT
    assert (result["release"], result["precision"], result["max_variants"]) == ("2.1.0", 1.0, 32)
    base, alias, acid = result["molecules"]
    assert [row["candidate_id"] for row in (base, alias, acid)] == ["base", "base_alias", "acid"]
    assert base["input_smiles"] == alias["input_smiles"] == "CN"
    assert {state["smiles"]: state["formal_charge"] for state in base["states"]} == {"CN": 0, "C[NH3+]": 1}
    assert alias["states"] == base["states"]
    assert {state["smiles"]: state["formal_charge"] for state in acid["states"]} == {"CC(=O)[O-]": -1}
    assert {"verdict", "predicted_pka", "dominant_state", "populations"}.isdisjoint(result)

    selected = dimorphite.screen_run(tmp_path, candidate="base_alias")
    assert selected["status"] == "complete"
    assert [row["candidate_id"] for row in selected["molecules"]] == ["base_alias"]
    assert calls[-1] == (7.0, 7.0, ["CN"])


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "{}",
        '{"version":2,"ph_min":6.8,"ph_max":7.4}',
        '{"version":true,"ph_min":6.8,"ph_max":7.4}',
        '{"version":1,"ph_min":7.4,"ph_max":6.8}',
        '{"version":1,"ph_min":-0.1,"ph_max":7.4}',
        '{"version":1,"ph_min":6.8,"ph_max":14.1}',
        '{"version":1,"ph_min":"6.8","ph_max":7.4}',
        '{"version":1,"ph_min":false,"ph_max":7.4}',
        '{"version":1,"ph_min":6.8,"ph_max":NaN}',
        '{"version":1,"ph_min":6.8,"ph_max":1e1000}',
        '{"version":1,"ph_min":6.8,"ph_min":7.0,"ph_max":7.4}',
        '{"version":1,"ph_min":6.8,"ph_max":7.4,"precision":0.5}',
    ],
)
def test_invalid_or_missing_ph_window_blocks_even_a_selected_candidate(tmp_path, monkeypatch, payload):
    _table(tmp_path, [("nomination", "CN")])
    if payload is not None:
        (tmp_path / dimorphite.MANIFEST).write_text(payload, encoding="utf-8")
    monkeypatch.setattr(dimorphite, "_enumerate", _forbid_provider)
    result = dimorphite.screen_run(tmp_path, candidate="nomination")
    assert result["status"] == "untested" and result["blocker"]


def test_bad_named_table_never_falls_back_and_legacy_input_is_unnamed(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    _manifest(tmp_path)
    (tmp_path / "ligand_smiles.txt").write_text("CN legacy molecule\n", encoding="utf-8")
    table = tmp_path / "ligand_candidates.tsv"
    table.write_text("name\tsmiles\nnomination\tCCN\n", encoding="utf-8")
    monkeypatch.setattr(dimorphite, "_enumerate", _forbid_provider)
    assert dimorphite.screen_run(tmp_path)["status"] == "untested"
    table.unlink()
    assert dimorphite.screen_run(tmp_path, candidate="run_ligand")["status"] == "untested"

    monkeypatch.setattr(dimorphite, "_enumerate", lambda _lo, _hi, inputs: {inputs[0]: {"states": ["C[NH3+]"]}})
    result = dimorphite.screen_run(tmp_path)
    assert result["status"] == "complete"
    assert len(result["molecules"]) == 1
    assert result["molecules"][0]["input_smiles"] == "CN"


def test_invalid_duplicate_and_provider_blocked_rows_keep_attribution(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    _manifest(tmp_path)
    _table(
        tmp_path,
        [
            ("good", "CN"),
            ("duplicate", "CCN"),
            ("duplicate", "CCCN"),
            ("salt", "CN.[Na+]"),
            ("radical", "[CH3]"),
            ("dummy", "CC*"),
            ("too_large", "C" * 81),
            ("no_sites", "CCC"),
        ],
    )

    def enumerate_states(_lo, _hi, inputs):
        assert sorted(inputs) == ["CCC", "CN"]
        return {"CN": {"states": ["C[NH3+]"]}, "CCC": {"blocker": "no_recognized_sites"}}

    monkeypatch.setattr(dimorphite, "_enumerate", enumerate_states)
    result = dimorphite.screen_run(tmp_path)
    assert result["status"] == "partial"
    assert [row["candidate_id"] for row in result["molecules"]] == [
        "good", "duplicate", "duplicate", "salt", "radical", "dummy", "too_large", "no_sites",
    ]
    covered, *blocked = result["molecules"]
    assert covered["states"][0]["smiles"] == "C[NH3+]"
    assert all(row["status"] == "untested" and row["blocker"] and not row.get("states") for row in blocked)
    assert blocked[0]["blocker"] == blocked[1]["blocker"]
    assert blocked[-1]["blocker"] == "no_recognized_sites"

    monkeypatch.setattr(dimorphite, "_enumerate", _forbid_provider)
    selected = dimorphite.screen_run(tmp_path, candidate="duplicate")
    assert selected["status"] == "untested"
    assert [row["candidate_id"] for row in selected["molecules"]] == ["duplicate", "duplicate"]


def test_selection_budget_keeps_names_selectable_without_enumerating_the_table(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    _manifest(tmp_path)
    _table(tmp_path, [(f"ligand_{index}", "CN") for index in range(33)])
    monkeypatch.setattr(dimorphite, "_enumerate", _forbid_provider)
    whole = dimorphite.screen_run(tmp_path)
    assert whole["status"] == "untested" and whole["blocker"]

    def enumerate_states(_lo, _hi, inputs):
        assert inputs == ["CN"]
        return {"CN": {"states": ["C[NH3+]"]}}

    monkeypatch.setattr(dimorphite, "_enumerate", enumerate_states)
    selected = dimorphite.screen_run(tmp_path, candidate="ligand_32")
    assert selected["status"] == "complete"
    assert [row["candidate_id"] for row in selected["molecules"]] == ["ligand_32"]


def test_standalone_matches_selected_screen_and_requires_settings(tmp_path, monkeypatch, capsys):
    from scripts import screen_extras

    pytest.importorskip("rdkit")
    _table(tmp_path, [("first", "CN"), ("second", "CCN")])
    arguments = ["dimorphite", str(tmp_path), "--candidate", "second"]
    monkeypatch.setattr(dimorphite, "_enumerate", _forbid_provider)
    assert screen_extras.main(arguments) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "untested"

    _manifest(tmp_path)

    def enumerate_states(_lo, _hi, inputs):
        assert inputs == ["CCN"]
        return {"CCN": {"states": ["CC[NH3+]"]}}

    monkeypatch.setattr(dimorphite, "_enumerate", enumerate_states)
    assert screen_extras.main(arguments) == 0
    reported = json.loads(capsys.readouterr().out)
    assert reported["status"] == "complete" and reported["molecules"][0]["candidate_id"] == "second"
    assert reported == dimorphite.screen_run(tmp_path, candidate="second")


@pytest.mark.parametrize(
    ("nomination", "proposed"),
    [
        ("CN", ["C[NH3+]", "CCO"]),
        ("CN", "C[NH3+]"),
        ("CN", []),
        ("C[C@H](N)O", ["C[C@@H]([NH3+])O"]),
        ("CN", ["[13CH3][NH3+]"]),
    ],
)
def test_invalid_structure_lists_are_withheld_as_a_unit(tmp_path, monkeypatch, nomination, proposed):
    pytest.importorskip("rdkit")
    _manifest(tmp_path)
    _table(tmp_path, [("amine", nomination)])
    monkeypatch.setattr(dimorphite, "_enumerate", lambda _lo, _hi, inputs: {inputs[0]: {"states": proposed}})
    result = dimorphite.screen_run(tmp_path)
    assert result["status"] == "untested"
    assert result["molecules"][0]["blocker"] and not result["molecules"][0].get("states")


@pytest.mark.parametrize(
    "blocker", [[], {}, "untrusted provider text", "variant_budget_exceeded", "dimorphite_provider_failure"],
)
def test_bad_provider_row_never_hides_an_unaffected_nomination(tmp_path, monkeypatch, blocker):
    pytest.importorskip("rdkit")
    _manifest(tmp_path)
    _table(tmp_path, [("affected", "CN"), ("unaffected", "CCN")])
    monkeypatch.setattr(
        dimorphite, "_enumerate", lambda *_args: {"CN": {"blocker": blocker}, "CCN": {"states": ["CC[NH3+]"]}},
    )
    result = dimorphite.screen_run(tmp_path)
    assert result["status"] == "partial"
    affected, unaffected = result["molecules"]
    assert affected["status"] == "untested" and not affected["states"]
    expected = blocker if isinstance(blocker, str) and blocker in dimorphite.BLOCKERS else "dimorphite_invalid_report"
    assert affected["blocker"] == expected
    assert unaffected["states"] == [{"smiles": "CC[NH3+]", "formal_charge": 1}]


@pytest.mark.parametrize(
    "response",
    [b"not JSON", b"[]", b'{"CCN":{"states":["CC[NH3+]"]}}', b'{"CN":{},"CN":{"states":["CN"]}}'],
)
def test_worker_pipe_rejects_foreign_missing_or_duplicate_keys(monkeypatch, response):
    monkeypatch.setattr(dimorphite, "installed", lambda: True)

    def worker(argv, **kwargs):
        assert argv[-2:] == ["-m", "rfproteina.disease.dimorphite_worker"]
        assert json.loads(kwargs["input"]) == {"ph_min": 6.8, "ph_max": 7.4, "smiles": ["CN"]}
        assert kwargs["timeout"] > 0
        assert "DIMORPHITE_DL_LOG_FILE_PATH" not in kwargs["env"]
        return SimpleNamespace(stdout=response)

    monkeypatch.setenv("DIMORPHITE_DL_LOG_FILE_PATH", "relative-file.log")
    monkeypatch.setattr(dimorphite.subprocess, "run", worker)
    with pytest.raises(dimorphite.DimorphiteError, match="dimorphite_invalid_report"):
        dimorphite._enumerate(6.8, 7.4, ["CN"])


def test_saved_report_revalidates_state_identity_charge_and_duplicate_names(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    _manifest(tmp_path)
    _table(tmp_path, [("nomination", "CN")])
    monkeypatch.setattr(dimorphite, "_enumerate", lambda *_args: {"CN": {"states": ["C[NH3+]"]}})
    result = dimorphite.screen_run(tmp_path)
    assert dimorphite.report_view(result) == result
    row = result["molecules"][0]
    for damaged in (
        [{**row, "states": [{"smiles": "C[NH3+]", "formal_charge": 0}]}],
        [{**row, "states": [{"smiles": "C[NH3+]", "formal_charge": True}]}],
        [{**row, "states": [{"smiles": "CC[NH3+]", "formal_charge": 1}]}],
        [row, row],
    ):
        sanitized = dimorphite.report_view({**result, "molecules": damaged})
        assert sanitized["status"] == "untested" and sanitized["blocker"] == "dimorphite_invalid_report"
        assert sanitized["molecules"] == []


def test_optional_provider_enumerates_simple_amine_and_abstains_on_no_sites(tmp_path):
    if not dimorphite.installed():
        pytest.skip("optional pinned Dimorphite-DL provider is not installed")
    _manifest(tmp_path, 2.0, 2.0)
    _table(tmp_path, [("amine", "CN"), ("alkane", "CCC")])
    result = dimorphite.screen_run(tmp_path)
    assert result["status"] == "partial"
    amine, alkane = result["molecules"]
    assert amine["states"] == [{"smiles": "C[NH3+]", "formal_charge": 1}]
    assert alkane["status"] == "untested" and alkane["blocker"] == "no_recognized_sites"
