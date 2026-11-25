"""Exercise Jazzy's local molecular identity and native hydration descriptors."""

from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from rfproteina.disease import jazzy, jazzy_runtime, screen_selection

_PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
_ETHYL_ACETATE = "CCOC(C)=O"
_NATIVE = {
    "hydration_kj_mol": -54.5831,
    "apolar_kj_mol": -3.0161,
    "polar_kj_mol": -51.2688,
    "interaction_kj_mol": -0.2982,
    "ch_donor_strength": 4.2822,
    "xh_donor_strength": 1.3955,
    "acceptor_strength": 2.461,
}


def _candidates(directory, rows):
    (directory / "ligand_candidates.tsv").write_text(
        "candidate_id\tsmiles\n" + "".join(f"{name}\t{smiles}\n" for name, smiles in rows), encoding="utf-8",
    )


def _predict_native(batch):
    return [dict(_NATIVE) for _ in batch]


def _forbidden(*_args):
    pytest.fail("An unsupported nomination reached Jazzy inference")


@pytest.fixture
def nomination(tmp_path):
    pytest.importorskip("rdkit")
    _candidates(tmp_path, [("acetaminophen", _PARACETAMOL), ("ethyl_acetate", _ETHYL_ACETATE)])
    return tmp_path


def test_native_energies_and_dimensionless_strengths_are_preserved(nomination, monkeypatch):
    # Independently rounded native components differ from their total by 0.0002 kJ/mol.
    ester = {
        "hydration_kj_mol": -16.8998, "apolar_kj_mol": 6.5, "polar_kj_mol": -24.1, "interaction_kj_mol": 0.7,
        "ch_donor_strength": 3.1, "xh_donor_strength": 0.0, "acceptor_strength": 1.7,
    }

    def predict(batch):
        assert batch == [_PARACETAMOL, _ETHYL_ACETATE]
        return [dict(_NATIVE), ester]

    monkeypatch.setattr(jazzy_runtime, "predict", predict)
    result = jazzy.screen_run(nomination)
    assert result["status"] == "complete" and result["scope"] == "candidate_table"
    assert result["endpoint"] == "gas_to_water_hydration_free_energy_kj_mol"
    assert result["geometry"] == "ETDG-3D-MMFF94" and result["embedding_seed"] == 11
    assert result["candidates"] == [
        {"candidate_id": "acetaminophen", "candidate_sha256": hashlib.sha256(_PARACETAMOL.encode()).hexdigest(),
         "status": "predicted", **_NATIVE},
        {"candidate_id": "ethyl_acetate", "candidate_sha256": hashlib.sha256(_ETHYL_ACETATE.encode()).hexdigest(),
         "status": "predicted", **ester},
    ]
    assert jazzy.report_view(result) == result


@pytest.mark.parametrize("smiles", [
    "CC[NH3+]",
    "[NH3+]CC(=O)[O-]",
    "CC[NH3+].[Cl-]",
    "CC.CCO",
    "[13CH3]CCO",
    "[CH2]CCO",
    "CC(O)CC",
    "CC=CC",
    "C",
])
def test_unsupported_molecule_is_untested_and_keeps_its_candidate_id(nomination, monkeypatch, smiles):
    _candidates(nomination, [("limited", smiles)])
    monkeypatch.setattr(jazzy_runtime, "predict", _forbidden)
    result = jazzy.screen_run(nomination)
    assert result["status"] == "untested" and result["blocker"] == "jazzy_no_usable_results"
    assert result["candidates"] == [
        {"candidate_id": "limited", "status": "untested", "blocker": "jazzy_unsupported_candidate"},
    ]
    assert jazzy.report_view(result) == result


def test_named_stereoisomers_remain_attributed_to_distinct_molecules(nomination, monkeypatch):
    Chem = pytest.importorskip("rdkit.Chem")
    first, second = "C[C@H](O)CC", "C[C@@H](O)CC"
    _candidates(nomination, [("first", first), ("second", second)])
    (nomination / "ligand_smiles.txt").write_text(f"{_PARACETAMOL}\n", encoding="utf-8")
    identities = [Chem.MolToSmiles(Chem.MolFromSmiles(item), isomericSmiles=True) for item in (first, second)]

    def predict(batch):
        assert batch == identities[1:]
        return _predict_native(batch)

    monkeypatch.setattr(jazzy_runtime, "predict", predict)
    selected = jazzy.screen_run(nomination, "second")
    (row,) = selected["candidates"]
    assert selected["status"] == "complete" and row["candidate_id"] == "second"
    assert row["candidate_sha256"] == hashlib.sha256(identities[1].encode()).hexdigest()
    assert row["candidate_sha256"] != hashlib.sha256(identities[0].encode()).hexdigest()

    monkeypatch.setattr(jazzy_runtime, "predict", _forbidden)
    assert jazzy.screen_run(nomination, "SECOND")["blocker"] == "jazzy_no_candidates"
    (nomination / "ligand_candidates.tsv").unlink()
    assert jazzy.screen_run(nomination, "second")["blocker"] == "jazzy_candidate_scope_missing"


def test_duplicate_candidate_id_blocks_run_and_named_request(nomination, monkeypatch):
    _candidates(nomination, [("alias", _PARACETAMOL), ("alias", _ETHYL_ACETATE)])
    monkeypatch.setattr(jazzy_runtime, "predict", _forbidden)
    for candidate in (None, "alias"):
        result = jazzy.screen_run(nomination, candidate)
        assert result["status"] == "untested" and result["blocker"] == "jazzy_duplicate_candidate"
        assert result["candidates"] == []


@pytest.mark.parametrize("blocker", ["jazzy_geometry_unavailable", "jazzy_calculation_failed"])
def test_runtime_row_blocker_is_preserved_with_other_candidate_measurements(nomination, monkeypatch, blocker):
    _candidates(nomination, [("fragmented", "CC.CCO"), ("failed", _PARACETAMOL), ("ready", _ETHYL_ACETATE)])

    def predict(batch):
        assert batch == [_PARACETAMOL, _ETHYL_ACETATE]
        return [{"blocker": blocker}, dict(_NATIVE)]

    monkeypatch.setattr(jazzy_runtime, "predict", predict)
    report = jazzy.screen_run(nomination)
    assert report["status"] == "partial"
    unsupported, failed, ready = report["candidates"]
    assert unsupported["candidate_id"] == "fragmented" and unsupported["blocker"] == "jazzy_unsupported_candidate"
    assert failed == {"candidate_id": "failed", "candidate_sha256": hashlib.sha256(_PARACETAMOL.encode()).hexdigest(),
                      "status": "untested", "blocker": blocker}
    assert ready["candidate_id"] == "ready" and ready["status"] == "predicted"
    assert jazzy.report_view(report) == report


@pytest.mark.parametrize("problem", ["wrong_unit", "wrong_geometry", "wrong_total", "negative_strength", "duplicate_id"])
def test_saved_projection_rejects_inconsistent_values_and_strips_foreign_fields(nomination, monkeypatch, problem):
    monkeypatch.setattr(jazzy_runtime, "predict", _predict_native)
    report = jazzy.screen_run(nomination)
    raw = deepcopy(report)
    raw["private_input"] = "discard this"
    raw["candidates"][0]["smiles"] = _PARACETAMOL
    assert jazzy.report_view(raw) == report

    if problem == "wrong_unit":
        raw["endpoint"] = "gas_to_water_hydration_free_energy_kcal_mol"
    elif problem == "wrong_geometry":
        raw["geometry"] = "ETKDG-UFF"
    elif problem == "wrong_total":
        raw["candidates"][0]["hydration_kj_mol"] += 0.0003
    elif problem == "negative_strength":
        raw["candidates"][0]["acceptor_strength"] = -0.01
    else:
        raw["candidates"][1]["candidate_id"] = raw["candidates"][0]["candidate_id"]
    projected = jazzy.report_view(raw)
    assert projected["status"] == "untested" and projected["blocker"] == "jazzy_invalid_report"
    assert projected["candidates"] == []


def test_picker_recommends_only_installed_jazzy_for_compatible_chemistry(nomination, monkeypatch):
    monkeypatch.setattr(jazzy_runtime, "predict", _forbidden)

    def choice(example=None, installed=True):
        return next(row for row in screen_selection.plan(
            nomination, "ligand", example=example, available={jazzy.METHOD_ID: installed},
        ) if row.method.key == jazzy.METHOD_ID)

    eligible = choice("acetaminophen")
    assert eligible.installed and eligible.applicable and eligible.recommended
    assert not choice(installed=False).recommended
    absent = choice("unknown")
    assert not absent.applicable and not absent.recommended
    _candidates(nomination, [("unspecified", "CC=CC")])
    incompatible = choice()
    assert not incompatible.applicable and not incompatible.recommended
