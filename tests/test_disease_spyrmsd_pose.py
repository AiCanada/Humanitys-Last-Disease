"""Redocking controls test pose recovery in the original receptor coordinate frame."""

import hashlib
import json
import sys

import numpy as np
import pytest

from rfproteina.disease import corpus_provenance, mcp_api, screen, screen_inventory, screen_selection
from rfproteina.disease import spyrmsd_pose as provider
from rfproteina.disease.validation_checks import candidate_validation_plan

Chem = pytest.importorskip("rdkit.Chem")
from rdkit.Chem import AllChem


def _molecule(smiles="CC(C)O"):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(molecule, randomSeed=23) == 0
    return Chem.RemoveHs(molecule)


def _write(path, mol):
    block = Chem.MolToMolBlock(mol)
    if path.suffix == ".sdf":
        block += "$$$$\n"
    path.write_text(block, encoding="utf-8")


def _entry():
    return {
        "pose": "redocked.sdf",
        "reference": "crystal.mol",
        "experimental_reference_id": "crystal23:1:A:LIG:401",
        "pose_frame": "crystal23.model1",
        "reference_frame": "crystal23.model1",
    }


def _manifest(directory, samples):
    (directory / provider.MANIFEST).write_text(json.dumps({"samples": samples}), encoding="utf-8")


@pytest.fixture
def finished(tmp_path):
    directory = tmp_path / "run"
    samples = directory / "inference_samples"
    samples.mkdir(parents=True)
    for candidate in ("binder_0", "binder_1"):
        (samples / f"{candidate}_generated.cif").write_text("data_example\n")
    (directory / "generation.json").write_text('{"binder_kind":"ligand"}')
    reference = _molecule()
    _write(directory / "crystal.mol", reference)
    _write(directory / "redocked.sdf", reference)
    _manifest(directory, {"binder_0": _entry()})
    return directory


def _choice(directory, example=None, kind="ligand"):
    (choice,) = screen_selection.plan(directory, kind, example=example, available={provider.METHOD: True})
    return choice


@pytest.mark.parametrize(("offset", "expected"), [((0, 0, 0), 0.0), ((3, -4, 0), 5.0)])
def test_symmetry_and_atom_reordering_preserve_receptor_displacement(finished, offset, expected):
    pytest.importorskip("spyrmsd")
    ref = Chem.MolFromMolFile(str(finished / "crystal.mol"))
    pose = Chem.Mol(ref)
    points = ref.GetConformer().GetPositions()
    # Exchange the equivalent terminal methyls, then reorder the file independently.
    for i, j in enumerate((2, 1, 0, 3)):
        pose.GetConformer().SetAtomPosition(i, points[j] + offset)
        pose.GetAtomWithIdx(i).SetAtomMapNum(i + 1)
    pose = Chem.RenumberAtoms(pose, (3, 2, 1, 0))
    _write(finished / "redocked.sdf", Chem.AddHs(pose, addCoords=True))
    report = provider.measure(finished)
    assert report["status"] == "reported" and report["measured"] == report["attempted"] == 1
    (result,) = report["samples"]
    assert result["rmsd_angstrom"] == pytest.approx(expected, abs=1e-8)
    assert result["heavy_atoms"] == 4 and result["method"] == provider.METHOD
    assert result["experimental_reference_id"] == _entry()["experimental_reference_id"]
    assert result["receptor_frame_id"] == _entry()["pose_frame"]
    assert result["pose_sha256"] == hashlib.sha256((finished / "redocked.sdf").read_bytes()).hexdigest()
    assert result["reference_sha256"] == hashlib.sha256((finished / "crystal.mol").read_bytes()).hexdigest()
    assert report["provider"]["center"] is report["provider"]["minimize"] is False
    assert report["unit"] == "angstrom" and report["corpus_provenance"]["corpora"] == []
    assert report["corpus_provenance"]["overlap"] == corpus_provenance.NO_TRAINING_CORPUS


def test_rotation_about_centroid_is_retained(finished):
    pytest.importorskip("spyrmsd")
    molecule = _molecule("CCO")
    _write(finished / "crystal.mol", molecule)
    xyz = molecule.GetConformer().GetPositions()
    rotated = (xyz - xyz.mean(axis=0)) * [-1, -1, 1] + xyz.mean(axis=0)
    for i, point in enumerate(rotated):
        molecule.GetConformer().SetAtomPosition(i, point)
    _write(finished / "redocked.sdf", molecule)
    expected = np.sqrt(np.mean(np.sum((xyz - rotated) ** 2, axis=1)))
    assert expected > 1.0
    assert provider.measure(finished)["samples"][0]["rmsd_angstrom"] == pytest.approx(expected, abs=2e-4)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("CCO", "COC"),
        ("CC", "C=C"),
        ("CC(=O)O", "CC(=O)[O-]"),
        ("CCO", "[13CH3]CO"),
        ("F[C@](Cl)(Br)I", "F[C@@](Cl)(Br)I"),
        ("F/C=C/Cl", "F/C=C\\Cl"),
    ],
)
def test_element_connectivity_alone_cannot_establish_ligand_identity(finished, left, right):
    _write(finished / "redocked.sdf", _molecule(left))
    _write(finished / "crystal.mol", _molecule(right))
    assert not _choice(finished).applicable
    (row,) = provider.measure(finished)["samples"]
    assert row["blocker"] == "ligand_identity_mismatch" and "rmsd_angstrom" not in row


@pytest.mark.parametrize(
    ("change", "blocker"),
    [
        ({"experimental_reference_id": ""}, "experimental_reference_required"),
        ({"experimental_reference_id": "unknown"}, "experimental_reference_required"),
        ({"reference": ""}, "experimental_reference_required"),
        ({"reference_frame": None}, "receptor_frame_required"),
        ({"pose_frame": "unspecified", "reference_frame": "unspecified"}, "receptor_frame_required"),
        ({"reference_frame": "other.model1"}, "receptor_frame_mismatch"),
        ({"reference": "absent.sdf"}, "reference_missing"),
        ({"reference": "redocked.sdf"}, "pose_and_reference_are_same_file"),
        ({"reference": "../outside.sdf"}, "invalid_reference_filename"),
        ({"reference": "C:\\outside.mol"}, "invalid_reference_filename"),
    ],
)
def test_unmatched_or_unattributed_reference_withholds_measurement(finished, change, blocker):
    _manifest(finished, {"binder_0": {**_entry(), **change}})
    (row,) = provider.measure(finished)["samples"]
    assert row["status"] == "untested" and row["blocker"] == blocker and "rmsd_angstrom" not in row
    choice = _choice(finished, "binder_0")
    assert not choice.applicable and blocker in choice.reason


def test_manifest_and_scope_are_authoritative(finished, monkeypatch):
    monkeypatch.setattr(provider, "_rmsd", lambda *_: pytest.fail("The picker executed the symmetry search"))
    assert _choice(finished, "binder_0").recommended
    assert not _choice(finished, "binder_0", kind="peptide").applicable
    assert not _choice(finished, "binder_1").applicable
    _manifest(finished, {"foreign": _entry()})
    assert not _choice(finished).applicable
    assert provider.measure(finished)["samples"][0]["blocker"] == "unknown_generated_candidate"
    path = finished / provider.MANIFEST
    path.write_text('{"samples":{"binder_0": {}, "binder_0": {}}}')
    assert provider.measure(finished)["blocker"] == "duplicate_manifest_key"
    path.unlink()
    (finished / screen_inventory.ATTACHMENT).write_text(
        json.dumps(
            {
                provider.METHOD: [
                    {"candidate_id": "binder_0", "endpoint": provider.ENDPOINT, "value": 0, "unit": "angstrom"}
                ]
            }
        )
    )
    assert not _choice(finished).applicable
    assert provider.measure(finished)["blocker"] == "manifest_missing"


def test_incomplete_ambiguous_and_non_three_dimensional_files(finished):
    path = finished / "redocked.sdf"
    valid = path.read_text()
    for text in (valid + valid, valid.removesuffix("$$$$\n"), valid.replace("$$$$", "damaged")):
        path.write_text(text)
        assert provider.measure(finished)["samples"][0]["blocker"] == "pose_requires_one_sdf_record"
    flat = Chem.MolFromSmiles("CC(C)O")
    AllChem.Compute2DCoords(flat)
    _write(path, flat)
    assert provider.measure(finished)["samples"][0]["blocker"] == "pose_ligand_coordinates_2d"
    _write(path, _molecule("CC.O"))
    assert provider.measure(finished)["samples"][0]["blocker"] == "pose_disconnected_ligand"


def test_mapping_limit_never_returns_a_truncated_minimum(finished, monkeypatch):
    pytest.importorskip("spyrmsd")
    monkeypatch.setattr(provider, "_MAX_MAPPINGS", 1)
    report = provider.measure(finished)
    (row,) = report["samples"]
    assert report["measured"] == 0 and row["blocker"] == "symmetry_mapping_limit" and "rmsd_angstrom" not in row


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.01, False, "0"])
def test_invalid_provider_output_withholds_the_measurement(finished, monkeypatch, invalid):
    rmsd = pytest.importorskip("spyrmsd.rmsd")
    monkeypatch.setattr(rmsd, "symmrmsd", lambda *args, **kwargs: invalid)
    (row,) = provider.measure(finished)["samples"]
    assert row["status"] == "untested" and row["blocker"] == "invalid_rmsd_result" and "rmsd_angstrom" not in row


def test_unavailable_provider_never_falls_back_to_another_rmsd(finished, monkeypatch):
    monkeypatch.setitem(sys.modules, "spyrmsd", None)
    assert not screen_selection.installations({provider.METHOD})[provider.METHOD]
    report = provider.measure(finished)
    assert report["measured"] == 0 and report["samples"][0]["blocker"] == "spyrmsd_unavailable"


def test_standalone_command_accepts_a_control_outside_generation(tmp_path, capsys):
    pytest.importorskip("spyrmsd")
    from scripts.screen_pose_safety import main

    reference = _molecule("CCO")
    _write(tmp_path / "crystal.mol", reference)
    _write(tmp_path / "redocked.sdf", reference)
    _manifest(tmp_path, {"positive_control": _entry()})
    assert main(["spyrmsd", str(tmp_path), "--sample", "positive_control"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["scope"] == "standalone_controls"
    assert result["samples"][0]["candidate_id"] == "positive_control"
    assert result["samples"][0]["rmsd_angstrom"] == 0.0
    assert not _choice(tmp_path).applicable
    assert main(["spyrmsd", str(tmp_path), "--sample", "novel_pose"]) == 2
    skipped = json.loads(capsys.readouterr().out)
    assert skipped["status"] == "untested" and skipped["blocker"] == "no_nominated_redocking_control"


def test_selection_persists_mixed_coverage_and_mcp_scopes_to_one_control(finished, monkeypatch):
    pytest.importorskip("spyrmsd")
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(finished.parent))
    _manifest(finished, {"binder_0": _entry(), "binder_1": {**_entry(), "reference_frame": "wrong"}})
    choice = _choice(finished)
    assert choice.applicable and not choice.recommended
    baseline = screen.screen_generation(finished, binder_kind="ligand", target_id="", screen_selection="explicit")
    verdict = screen.screen_generation(finished, binder_kind="ligand", target_id="", screen_methods=[provider.METHOD])
    assert all(verdict[key] == baseline[key] for key in ("call", "states", "hits", "cuts"))
    saved = json.loads((finished / "screen.json").read_bytes())
    result = saved["method_selection"]["prepared_measurements"][provider.METHOD]
    assert result["status"] == "partial" and result["attempted"] == 2 and result["measured"] == 1
    assert result["samples"][0]["rmsd_angstrom"] == 0.0
    assert result["samples"][1]["blocker"] == "receptor_frame_mismatch"
    candidates, _ = candidate_validation_plan((finished,), None)
    args = {"directory": str(finished), "methods": [provider.METHOD]}
    for candidate in candidates:
        reply = mcp_api.call_tool("run_selected_screens", {**args, "candidate_id": candidate.key})
        (row,) = reply["results"]
        if candidate.example_id == "binder_0":
            assert row["status"] == "reported" and row["measurement"]["attempted"] == 1
            assert row["measurement"]["samples"][0]["rmsd_angstrom"] == 0.0
        else:
            assert row["status"] == "inapplicable" and "receptor_frame_mismatch" in row["reason"]
