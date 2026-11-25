"""Keep organic-solubility predictions tied to the nominated solvent and temperature."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from rfproteina.disease import fastsolv, fastsolv_runtime, screen_selection

_CONDITIONS = [
    {"solvent": "ethyl_acetate", "temperature_k": 312.5},
    {"solvent": "ethanol", "temperature_k": 295},
]
_PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
_BENZOIC_ACID = "O=C(O)c1ccccc1"


def _manifest(directory, conditions=_CONDITIONS):
    (directory / fastsolv.MANIFEST).write_text(
        json.dumps({"version": 1, "conditions": conditions}), encoding="utf-8",
    )


def _candidates(directory, rows):
    (directory / "ligand_candidates.tsv").write_text(
        "candidate_id\tsmiles\n" + "".join(f"{name}\t{smiles}\n" for name, smiles in rows), encoding="utf-8",
    )


def _predict_uniform(batch):
    return [{"log_s": -1.3, "ensemble_sd": 0.2} for _ in batch]


def _forbidden(*_args):
    pytest.fail("An unsupported nomination reached fastsolv inference")


@pytest.fixture
def nomination(tmp_path):
    pytest.importorskip("rdkit")
    _manifest(tmp_path)
    _candidates(tmp_path, [("acetaminophen", _PARACETAMOL), ("benzoic_acid", _BENZOIC_ACID)])
    return tmp_path


def test_two_candidates_keep_their_conditions_and_values_in_batch_order(nomination, monkeypatch):
    expected_batch = [
        {"solute_smiles": _PARACETAMOL, "solvent_smiles": "CCOC(C)=O", "temperature_k": 312.5},
        {"solute_smiles": _PARACETAMOL, "solvent_smiles": "CCO", "temperature_k": 295.0},
        {"solute_smiles": _BENZOIC_ACID, "solvent_smiles": "CCOC(C)=O", "temperature_k": 312.5},
        {"solute_smiles": _BENZOIC_ACID, "solvent_smiles": "CCO", "temperature_k": 295.0},
    ]
    scores = [(-2.9, 0.31), (-1.2, 0.22), (0.4, 0.15), (-0.6, 0.04)]

    def predict(batch):
        assert batch == expected_batch
        return [{"log_s": score, "ensemble_sd": spread} for score, spread in scores]

    monkeypatch.setattr(fastsolv_runtime, "predict", predict)
    report = fastsolv.screen_run(nomination)
    assert report["status"] == "complete" and report["conditions"] == _CONDITIONS
    assert report["scope"] == "candidate_table" and report["ensemble_members"] == 4
    assert report["endpoint"] == "organic_solid_solubility_log10_mol_l"
    first, second = report["candidates"]
    assert first["candidate_id"] == "acetaminophen" and second["candidate_id"] == "benzoic_acid"
    for index, (row, smiles) in enumerate(zip((first, second), (_PARACETAMOL, _BENZOIC_ACID), strict=True)):
        assert row["status"] == "predicted"
        assert row["candidate_sha256"] == hashlib.sha256(smiles.encode("ascii")).hexdigest()
        assert row["predictions"] == [
            {**condition, "log_s": score, "ensemble_sd": spread}
            for condition, (score, spread) in zip(_CONDITIONS, scores[index * 2:index * 2 + 2], strict=True)
        ]


@pytest.mark.parametrize("conditions", [
    [{"solvent": "water", "temperature_k": 298}],
    [{"solvent": "ethanol_water", "temperature_k": 298}],
    [{"solvent": "ethanol", "temperature_k": 294.99}],
    [{"solvent": "ethanol", "temperature_k": 323.01}],
    [{"solvent": "ethanol", "temperature_k": "312"}],
    [{"solvent": "ethanol", "temperature_k": float("nan")}],
    [{"solvent": "ethanol", "temperature_k": 312, "ph": 7.4}],
    [{"solvent": "ethanol", "temperature_k": 312}, {"solvent": "ethanol", "temperature_k": 312.0}],
    [{"solvent": "ethanol", "temperature_k": 295 + i} for i in range(9)],
    [],
])
def test_invalid_and_aqueous_conditions_are_refused_before_inference(nomination, monkeypatch, conditions):
    _manifest(nomination, conditions)
    monkeypatch.setattr(fastsolv_runtime, "predict", _forbidden)
    result = fastsolv.screen_run(nomination)
    assert result["status"] == "untested" and result["blocker"] == "invalid_fastsolv_manifest"
    assert result["candidates"] == []


def test_named_candidate_requires_the_table_and_uses_only_its_own_rows(nomination, monkeypatch):
    _manifest(nomination, [{"solvent": "methanol", "temperature_k": 323}])
    (nomination / "ligand_smiles.txt").write_text(f"{_PARACETAMOL}\n", encoding="utf-8")

    def predict(batch):
        assert batch == [{"solute_smiles": _BENZOIC_ACID, "solvent_smiles": "CO", "temperature_k": 323.0}]
        return _predict_uniform(batch)

    monkeypatch.setattr(fastsolv_runtime, "predict", predict)
    result = fastsolv.screen_run(nomination, "benzoic_acid")
    assert result["status"] == "complete" and result["scope"] == "candidate_table"
    assert [row["candidate_id"] for row in result["candidates"]] == ["benzoic_acid"]

    monkeypatch.setattr(fastsolv_runtime, "predict", _forbidden)
    assert fastsolv.screen_run(nomination, "missing")["blocker"] == "fastsolv_no_candidates"
    (nomination / "ligand_candidates.tsv").unlink()
    assert fastsolv.screen_run(nomination, "benzoic_acid")["blocker"] == "fastsolv_candidate_scope_missing"


@pytest.mark.parametrize("smiles", ["CC[NH3+]", "CC(C)[NH3+].[Cl-]", "[13CH3]CCO", "[CH2]CCO", "CCO"])
def test_unsupported_solute_is_retained_without_predictions(nomination, monkeypatch, smiles):
    _candidates(nomination, [("unsupported", smiles), ("benzoic_acid", _BENZOIC_ACID)])

    def predict(batch):
        assert [item["solute_smiles"] for item in batch] == [_BENZOIC_ACID, _BENZOIC_ACID]
        return _predict_uniform(batch)

    monkeypatch.setattr(fastsolv_runtime, "predict", predict)
    result = fastsolv.screen_run(nomination)
    assert result["status"] == "partial"
    unsupported, accepted = result["candidates"]
    assert unsupported == {
        "candidate_id": "unsupported", "status": "untested", "blocker": "fastsolv_unsupported_candidate",
    }
    assert accepted["candidate_id"] == "benzoic_acid" and accepted["status"] == "predicted"
    assert fastsolv.report_view(result) == result

    monkeypatch.setattr(fastsolv_runtime, "predict", _forbidden)
    result = fastsolv.screen_run(nomination, "unsupported")
    assert result["blocker"] == "fastsolv_no_supported_candidates"
    assert result["candidates"] == [unsupported]


def test_picker_requires_available_runtime_explicit_conditions_and_supported_solute(nomination, monkeypatch):
    monkeypatch.setattr(fastsolv_runtime, "predict", _forbidden)

    def choice(example=None, installed=True):
        return next(row for row in screen_selection.plan(
            nomination, "ligand", example=example, available={fastsolv.METHOD_ID: installed},
        ) if row.method.key == fastsolv.METHOD_ID)

    eligible = choice("benzoic_acid")
    assert eligible.installed and eligible.applicable and eligible.recommended
    assert not choice(installed=False).recommended
    absent = choice("unknown")
    assert not absent.applicable and not absent.recommended
    _manifest(nomination, [{"solvent": "water", "temperature_k": 298}])
    aqueous = choice()
    assert not aqueous.applicable and not aqueous.recommended
    _manifest(nomination)
    _candidates(nomination, [("unsupported", "CCO")])
    out_of_scope = choice()
    assert not out_of_scope.applicable and not out_of_scope.recommended


@pytest.mark.parametrize("damage", ["endpoint", "negative_spread", "nonfinite_spread", "wrong_solvent", "wrong_kelvin"])
def test_saved_projection_requires_the_correct_scale_spread_and_conditions(nomination, monkeypatch, damage):
    monkeypatch.setattr(fastsolv_runtime, "predict", _predict_uniform)
    report = fastsolv.screen_run(nomination)
    assert fastsolv.report_view({**report, "private_input": "discard this"}) == report

    raw = deepcopy(report)
    prediction = raw["candidates"][0]["predictions"][0]
    if damage == "endpoint":
        raw["endpoint"] = "organic_solid_solubility_mg_l"
    else:
        prediction.update({
            "negative_spread": {"ensemble_sd": -0.01},
            "nonfinite_spread": {"ensemble_sd": float("inf")},
            "wrong_solvent": {"solvent": "acetonitrile"},
            "wrong_kelvin": {"temperature_k": 312.0},
        }[damage])
    projected = fastsolv.report_view(raw)
    assert projected["status"] == "untested" and projected["blocker"] == "fastsolv_invalid_report"
    assert projected["candidates"] == []


@pytest.mark.parametrize("problem", ["short", "long", "bad_row", "wrong_units", "not_a_list"])
def test_malformed_provider_batch_publishes_no_partial_scores(nomination, monkeypatch, problem):
    def predict(batch):
        answers = _predict_uniform(batch)
        return {
            "short": answers[:-1],
            "long": answers + answers[:1],
            "bad_row": [*answers[:-1], {"log_s": False, "ensemble_sd": 0.2}],
            "wrong_units": [*answers[:-1], {**answers[-1], "unit": "mg/L"}],
            "not_a_list": tuple(answers),
        }[problem]

    monkeypatch.setattr(fastsolv_runtime, "predict", predict)
    report = fastsolv.screen_run(nomination)
    assert report["status"] == "untested" and report["blocker"] == "fastsolv_invalid_report"
    assert report["candidates"] == []
