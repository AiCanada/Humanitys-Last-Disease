"""Contracts for site-resolved small-molecule pKa and coverage through the public screen."""

from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem

from rfproteina.disease import qupkake as pka
from rfproteina.disease import qupkake_worker as worker


def _no_sites(smiles: str, tautomerize: bool) -> dict:
    return {"status": "no_sites", "sites": [], "states": {}}


def test_activation_and_authoritative_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / pka.MANIFEST
    single = tmp_path / "ligand_smiles.txt"
    table = tmp_path / "ligand_candidates.tsv"
    single.write_text("n1ccccc1 pyridine\n")
    assert pka.screen_run(tmp_path, kind="ligand") is None
    manifest.write_text("{}")
    assert pka.screen_run(tmp_path, kind="protein")["blocker"] == "ligand_binder_kind_required"
    monkeypatch.setattr(pka, "_run", _no_sites)
    result = pka.screen_run(tmp_path, kind="ligand")
    assert result["source"] == "run_ligand" and result["attempted"] == 1
    table.write_text("name\tsmiles\nbroken\tCC\n")
    assert pka.screen_run(tmp_path, kind="ligand")["blocker"] == "invalid_candidate_table_header"
    manifest.write_text('{"input":"sdf"}')
    assert pka.screen_run(tmp_path, kind="ligand")["blocker"] == "sdf_input_unreadable"


@pytest.mark.parametrize(
    "settings",
    ["[]", '{"input":"sdf","input":"smiles"}', '{"input":"xyz"}', '{"tautomerize":1}', '{"options":{}}', "{"],
)
def test_bad_manifest_names_blocker(tmp_path: Path, settings: str) -> None:
    (tmp_path / pka.MANIFEST).write_text(settings)
    result = pka.screen_run(tmp_path, kind="ligand")
    assert result["blocker"] == "invalid_qupkake_manifest" and result["tautomerize"] is None


def test_candidate_coverage_and_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def run(smiles: str, tautomerize: bool) -> dict:
        calls.append(smiles)
        return _no_sites(smiles, tautomerize)

    monkeypatch.setattr(pka, "_run", run)
    inputs = [
        ("first", "n1ccccc1"),
        ("alias", "c1ccncc1"),
        ("salt", "[Na+].[Cl-]"),
        ("isotope", "[2H]O"),
        ("dup", "CC"),
        ("dup", "CO"),
        ("short",),
        ("", "CCC"),
        ("query", "*"),
        ("radical", "[CH3]"),
    ]
    report = pka.profile(inputs)
    assert calls == ["c1ccncc1"] and report["attempted"] == len(inputs)
    a, b, *rest = report["inputs"]
    assert a["compound_id"] == b["compound_id"] and a["source_atom_indices_by_map"][3] == 0
    assert b["source_atom_indices_by_map"][3] == 3
    assert [i["blocker"] for i in rest] == [
        "disconnected_structure_requires_nominated_microstate",
        "nonremovable_hydrogen_unsupported",
        "duplicate_candidate_id",
        "duplicate_candidate_id",
        "invalid_candidate_table_width",
        "missing_candidate_id",
        "query_or_dummy_atoms_unsupported",
        "open_shell_structure_unsupported",
    ]
    assert (
        report["status"] == "partial"
        and report["compounds_without_enumerated_sites"] == 1
        and report["predicted_sites"] == 0
    )


def test_structural_identity_resolves_stereo_and_charge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pka, "_run", _no_sites)
    molecules = ["N[C@H](C)O", "N[C@@H](C)O", "[NH3+][C@H](C)O", "[15NH2][C@H](C)O"]
    report = pka.profile([(str(i), s) for i, s in enumerate(molecules)])
    assert report["unique_compounds"] == 4
    assert len({i["compound_id"] for i in report["inputs"]}) == 4


def test_sdf_preserves_bad_records_and_source_atom_ordinals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pka, "_run", _no_sites)
    m = Chem.AddHs(Chem.MolFromSmiles("n1ccccc1"))
    order = [6, 1, 2, 3, 4, 5, 0, *range(7, m.GetNumAtoms())]
    m = Chem.RenumberAtoms(m, order)
    m.SetProp("_Name", "explicit H")
    record = Chem.MolToMolBlock(m).encode() + b"$$$$\n"
    f = tmp_path / "input.sdf"
    f.write_bytes(b"\xef\xbb\xbf" + record + b"invalid\n\xff\n$$$$\nunterminated")
    result = pka.profile_sdf(f)
    assert result["status"] == "partial" and result["attempted"] == 3
    a, b, c = result["inputs"]
    assert (
        a["source_atom_indices_by_map"][3] == 6
        and b["blocker"] == "invalid_sdf_encoding"
        and c["blocker"] == "truncated_sdf_record"
    )


def test_absent_runtime_preserves_denominator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUPKAKE_PYTHON", "/missing-qupkake-test/bin/python")
    rec = pka.profile([("valid", "c1ccccc1")])
    assert rec["status"] == "untested" and rec["compounds_without_enumerated_sites"] == 0
    assert rec["blocker"] == rec["compounds"][0]["blocker"] == "qupkake_runtime_unavailable"


def test_outside_pretraining_elements() -> None:
    report = pka.profile([("boronic acid", "OB(O)c1ccccc1")])
    assert report["inputs"][0]["blocker"] == "element_outside_qupkake_training_domain"
    assert report["attempted"] == 1 and report["unique_compounds"] == 0


def test_mapped_transfer_and_electron_count() -> None:
    m = Chem.MolFromSmiles("CC(=O)O")
    for a in m.GetAtoms():
        a.SetIntProp("_q_parent", a.GetIdx() + 1)
    base = Chem.Mol(m)
    base.GetAtomWithIdx(3).SetFormalCharge(-1)
    Chem.SanitizeMol(base)
    assert worker._reaction_site(worker._atoms(m), worker._atoms(base), "acidic") == 4
    with pytest.raises(worker.ProfileError, match="conjugate_is_not"):
        worker._reaction_site(worker._atoms(m), worker._atoms(base), "basic")
    bad = Chem.MolFromSmiles("[NH2+2]=C", sanitize=False)
    bad.UpdatePropertyCache(strict=False)
    with pytest.raises(worker.ProfileError, match="invalid_closed_shell_valence"):
        worker._check_valence(bad)
    for s in ("[NH4+]", "[O-]C=O", "c1cc[nH+]cc1", "B(O)(O)O"):
        worker._check_valence(Chem.MolFromSmiles(s))


def test_quantum_completion_and_charge() -> None:
    output = "charge : -1\nGEOMETRY OPTIMIZATION CONVERGED AFTER 6 ITERATIONS\n* finished run on"
    worker._check_xtb(output, -1, geometry=True)
    with pytest.raises(worker.ProfileError, match="xtb_charge_mismatch"):
        worker._check_xtb(output, 0, geometry=True)
    with pytest.raises(worker.ProfileError, match="xtb_calculation_incomplete"):
        worker._check_xtb("charge : -1\n* finished run on", -1, geometry=True)


def test_worker_contract_rejects_misassigned_or_nonfinite_reports() -> None:
    state = {"smiles": "N", "atom_mapped_smiles": "[NH3:1]", "formal_charge": 0}
    row = {
        "method": pka.METHOD,
        "kind": "basic",
        "endpoint": pka.ENDPOINT,
        "unit": "pKa",
        "atomic_number": 7,
        "nominated_atom_maps": [1],
        "status": "predicted",
        "micro_pka": 0.0,
        "site_atom_map": 1,
        "protonated_state_id": "acid",
        "deprotonated_state_id": "base",
    }
    base = {
        "provider": {"revision": pka.REVISION, "device": "cpu"},
        "status": "predicted",
        "states": {"base": state, "acid": {**state, "formal_charge": 1}},
        "input_state_id": "base",
        "selected_state_id": "base",
        "rejected_nominations": [],
        "failed_site_heads": [],
        "sites": [row],
    }
    assert pka._valid_report(base)
    assert pka._valid_report({**base, "status": "no_sites", "sites": []})
    failed = [{"kind": "acidic", "blocker": "invalid_site_head_output"}]
    assert pka._valid_report({**base, "status": "partial", "failed_site_heads": failed})
    assert not pka._valid_report({**base, "status": "no_sites", "sites": [], "failed_site_heads": failed})
    for update in (
        {"micro_pka": float("nan")},
        {"micro_pka": True},
        {"atomic_number": []},
        {"protonated_state_id": "missing"},
        {"kind": "acidic"},
        {"protonated_state_id": "base"},
        {"site_atom_map": 0},
        {"status": "untested", "blocker": "bad"},
    ):
        assert not pka._valid_report({**base, "sites": [{**row, **update}]})
    assert not pka._valid_report({**base, "status": "no_sites"})
    assert not pka._valid_report({})
    assert pka._valid_report({"status": "untested", "blocker": "missing", "sites": []})
