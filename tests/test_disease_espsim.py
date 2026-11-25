"""Keep the two ESP-Sim scores tied to the original prealigned poses."""

from __future__ import annotations

import hashlib
import json

import pytest

from rfproteina.disease import espsim, espsim_runtime, screen_selection

_CANDIDATE = "lead_A"
_REFERENCE = "reference_A"
_CANDIDATE_FILE = "candidate.sdf"
_REFERENCE_FILE = "reference.sdf"


def _manifest(directory, **updates):
    (directory / espsim.MANIFEST).write_text(json.dumps({
        "version": 1, "candidate_id": _CANDIDATE, "reference_id": _REFERENCE,
        "candidate_structure": _CANDIDATE_FILE, "reference_structure": _REFERENCE_FILE,
        "prealigned": True, **updates,
    }), encoding="utf-8")


def _sdf(smiles, title, *, shift=0, explicit_hydrogens=True, dimensions=3, broken_bond=False):
    Chem = pytest.importorskip("rdkit.Chem")
    AllChem = pytest.importorskip("rdkit.Chem.AllChem")
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if dimensions == 3:
        assert AllChem.EmbedMolecule(molecule, randomSeed=2483) == 0
    else:
        AllChem.Compute2DCoords(molecule)
    if not explicit_hydrogens:
        molecule = Chem.RemoveHs(molecule)
    coordinates = molecule.GetConformer()
    for index, (x, y, z) in enumerate(coordinates.GetPositions()):
        coordinates.SetAtomPosition(index, (x + shift, y, z))
    if broken_bond:
        index = molecule.GetNumAtoms() - 1
        x, y, z = coordinates.GetPositions()[index]
        coordinates.SetAtomPosition(index, (x + 10, y, z))
    molecule.SetProp("_Name", title)
    return (Chem.MolToMolBlock(molecule) + ">  <pose_label>\nsource record\n\n$$$$\n").replace("\n", "\r\n").encode()


def _forbidden(*_args):
    pytest.fail("An invalid or unattributed ligand pair reached ESP-Sim")


@pytest.fixture
def pair(tmp_path):
    _manifest(tmp_path)
    (tmp_path / _CANDIDATE_FILE).write_bytes(_sdf("CCCO", _CANDIDATE))
    (tmp_path / _REFERENCE_FILE).write_bytes(_sdf("CCOC", _REFERENCE))
    return tmp_path


@pytest.mark.parametrize("shape, electrostatic", [(0.0, -1.0), (1.0, 1.0), (0.689, -0.27)])
def test_scores_and_hashes_remain_tied_to_the_original_pose_bytes(pair, monkeypatch, shape, electrostatic):
    candidate = (pair / _CANDIDATE_FILE).read_bytes()
    reference = (pair / _REFERENCE_FILE).read_bytes()

    def predict(candidate_data, reference_data):
        assert candidate_data == candidate and reference_data == reference
        return {"shape_tanimoto": shape, "electrostatic_carbo": electrostatic}

    monkeypatch.setattr(espsim_runtime, "predict", predict)
    report = espsim.screen_run(pair, _CANDIDATE)
    assert report["status"] == "complete" and report["scope"] == "prealigned_angstrom_pair"
    assert (report["candidate_id"], report["reference_id"]) == (_CANDIDATE, _REFERENCE)
    assert report["candidate_sha256"] == hashlib.sha256(candidate).hexdigest()
    assert report["reference_sha256"] == hashlib.sha256(reference).hexdigest()
    assert report["shape_tanimoto"] == shape and report["electrostatic_carbo"] == electrostatic
    assert report["shape_method"] == "rdkit_heavy_atom_shape_tanimoto"
    assert report["electrostatic_method"] == "analytic_gaussian_carbo"
    assert report["charge_model"] == "gasteiger_12_iterations"
    assert espsim.report_view(report) == report
    assert (pair / _CANDIDATE_FILE).read_bytes() == candidate
    assert (pair / _REFERENCE_FILE).read_bytes() == reference


@pytest.mark.parametrize("updates", [
    {"candidate_id": _REFERENCE},
    {"candidate_structure": _REFERENCE_FILE},
    {"candidate_structure": _REFERENCE_FILE.upper()},
    {"prealigned": False},
    {"prealigned": 1},
    {"reference_structure": "../reference.sdf"},
])
def test_manifest_requires_a_distinct_explicitly_prealigned_pair(pair, monkeypatch, updates):
    _manifest(pair, **updates)
    monkeypatch.setattr(espsim_runtime, "predict", _forbidden)
    report = espsim.screen_run(pair)
    assert report["status"] == "untested" and report["blocker"] == "invalid_espsim_manifest"


@pytest.mark.parametrize("filename, blocker", [
    (_CANDIDATE_FILE, "espsim_candidate_mismatch"),
    (_REFERENCE_FILE, "espsim_reference_mismatch"),
])
def test_each_sdf_title_must_match_its_declared_id(pair, monkeypatch, filename, blocker):
    wrong_title = _CANDIDATE.lower() if filename == _CANDIDATE_FILE else _REFERENCE.lower()
    (pair / filename).write_bytes(_sdf("CCCO", wrong_title))
    monkeypatch.setattr(espsim_runtime, "predict", _forbidden)
    report = espsim.screen_run(pair)
    assert report["status"] == "untested" and report["blocker"] == blocker


@pytest.mark.parametrize("problem, blocker", [
    ("implicit_hydrogens", "espsim_unsupported_structure"),
    ("two_dimensions", "espsim_invalid_structure"),
    ("multiple_records", "espsim_invalid_structure"),
    ("bonded_graph_mismatch", "espsim_invalid_structure"),
])
def test_sdf_requires_one_explicit_hydrogen_3d_record_and_matching_graph(pair, monkeypatch, problem, blocker):
    candidate = pair / _CANDIDATE_FILE
    cases = {
        "implicit_hydrogens": lambda: _sdf("CCCO", _CANDIDATE, explicit_hydrogens=False),
        "two_dimensions": lambda: _sdf("CCCO", _CANDIDATE, dimensions=2),
        "multiple_records": lambda: candidate.read_bytes() * 2,
        "bonded_graph_mismatch": lambda: _sdf("CCCO", _CANDIDATE, broken_bond=True),
    }
    candidate.write_bytes(cases[problem]())
    monkeypatch.setattr(espsim_runtime, "predict", _forbidden)
    report = espsim.screen_run(pair)
    assert report["status"] == "untested" and report["blocker"] == blocker


def test_joint_axis_span_depends_on_relative_translation(pair, monkeypatch):
    candidate, reference = pair / _CANDIDATE_FILE, pair / _REFERENCE_FILE
    candidate.write_bytes(_sdf("CCCO", _CANDIDATE, shift=50))
    reference.write_bytes(_sdf("CCOC", _REFERENCE, shift=50))
    monkeypatch.setattr(espsim_runtime, "predict", lambda *_: {"shape_tanimoto": 0.8, "electrostatic_carbo": -0.1})
    assert espsim.screen_run(pair)["status"] == "complete"

    reference.write_bytes(_sdf("CCOC", _REFERENCE, shift=101))
    monkeypatch.setattr(espsim_runtime, "predict", _forbidden)
    report = espsim.screen_run(pair)
    assert report["status"] == "untested" and report["blocker"] == "espsim_frame_out_of_scope"


def test_named_candidate_and_optional_table_graph_must_agree(pair, monkeypatch):
    table = pair / "ligand_candidates.tsv"
    table.write_text(f"candidate_id\tsmiles\n{_CANDIDATE}\tOCCC\n", encoding="utf-8")
    monkeypatch.setattr(espsim_runtime, "predict", lambda *_: {"shape_tanimoto": 1, "electrostatic_carbo": 0})
    assert espsim.screen_run(pair, _CANDIDATE)["status"] == "complete"

    monkeypatch.setattr(espsim_runtime, "predict", _forbidden)
    assert espsim.screen_run(pair, _CANDIDATE.lower())["blocker"] == "espsim_candidate_mismatch"
    table.write_text(f"candidate_id\tsmiles\n{_CANDIDATE}\tCCOC\n", encoding="utf-8")
    report = espsim.screen_run(pair)
    assert report["status"] == "untested" and report["blocker"] == "espsim_candidate_mismatch"
    table.write_text(f"candidate_id\tsmiles\n{_CANDIDATE}\tOCCC\n{_CANDIDATE}\tCCOC\n", encoding="utf-8")
    assert espsim.screen_run(pair)["blocker"] == "espsim_candidate_mismatch"


def test_unavailable_charge_keeps_shape_and_reported_pair_identity(pair, monkeypatch):
    monkeypatch.setattr(espsim_runtime, "predict", lambda *_: {
        "shape_tanimoto": 0.68, "blocker": "espsim_charge_unavailable",
    })
    result = espsim.screen_run(pair)
    assert result["status"] == "partial" and result["blocker"] == "espsim_charge_unavailable"
    assert result["shape_tanimoto"] == 0.68 and "electrostatic_carbo" not in result
    assert (result["candidate_id"], result["reference_id"]) == (_CANDIDATE, _REFERENCE)
    assert espsim.report_view(result) == result


@pytest.mark.parametrize("updates", [
    {"shape_tanimoto": -0.001},
    {"shape_tanimoto": 1.001},
    {"shape_tanimoto": True},
    {"electrostatic_carbo": -1.001},
    {"electrostatic_carbo": 1.001},
    {"electrostatic_carbo": float("nan")},
    {"blocker": "espsim_charge_unavailable"},
])
def test_out_of_scale_or_inconsistent_provider_result_has_no_scores(pair, monkeypatch, updates):
    monkeypatch.setattr(espsim_runtime, "predict", lambda *_: {
        "shape_tanimoto": 0.4, "electrostatic_carbo": -0.5, **updates,
    })
    result = espsim.screen_run(pair)
    assert result["status"] == "untested" and result["blocker"] == "espsim_invalid_report"
    assert {"shape_tanimoto", "electrostatic_carbo"}.isdisjoint(result)


@pytest.mark.parametrize("updates", [
    {"candidate_sha256": "wrong"},
    {"reference_id": _CANDIDATE},
    {"source_sha256": "0" * 64},
    {"shape_method": "electrostatic_tanimoto"},
    {"charge_model": "other_charge_scheme"},
    {"status": "partial"},
])
def test_saved_projection_rejects_forged_attribution_or_scale(pair, monkeypatch, updates):
    monkeypatch.setattr(espsim_runtime, "predict", lambda *_: {"shape_tanimoto": 0.8, "electrostatic_carbo": -0.1})
    report = espsim.screen_run(pair)
    assert espsim.report_view({**report, "private_coordinates": [12, 34, 56], "combined_score": 2}) == report
    projected = espsim.report_view({**report, **updates})
    assert projected["status"] == "untested" and projected["blocker"] == "espsim_invalid_report"
    assert {"shape_tanimoto", "electrostatic_carbo"}.isdisjoint(projected)


def test_picker_recommends_only_an_installed_eligible_prealigned_pair(pair, monkeypatch):
    monkeypatch.setattr(espsim_runtime, "predict", _forbidden)

    def choice(example=None, installed=True):
        return next(row for row in screen_selection.plan(
            pair, "ligand", example=example, available={espsim.METHOD_ID: installed},
        ) if row.method.key == espsim.METHOD_ID)

    eligible = choice(_CANDIDATE)
    assert eligible.installed and eligible.applicable and eligible.recommended
    assert not choice(installed=False).recommended
    wrong = choice(_REFERENCE)
    assert not wrong.applicable and not wrong.recommended
    _manifest(pair, prealigned=False)
    unaligned = choice()
    assert not unaligned.applicable and not unaligned.recommended
