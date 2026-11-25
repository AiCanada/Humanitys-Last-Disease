"""Keep CREST conformers bound to an explicitly nominated molecular state."""

from __future__ import annotations

import hashlib
import json

import pytest

from rfproteina.disease import crest, crest_runtime, screen_selection


def _manifest(directory, **updates):
    (directory / crest.MANIFEST).write_text(
        json.dumps({"version": 1, "structure": "ligand.sdf", "candidate_id": "lead_A", **updates}),
        encoding="utf-8",
    )


def _sdf(smiles, *, title="lead_A", explicit_hydrogens=True, dimension=3):
    Chem = pytest.importorskip("rdkit.Chem")
    AllChem = pytest.importorskip("rdkit.Chem.AllChem")
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if dimension == 3:
        assert AllChem.EmbedMolecule(molecule, randomSeed=2483) == 0
    else:
        AllChem.Compute2DCoords(molecule)
    if not explicit_hydrogens:
        molecule = Chem.RemoveHs(molecule)
    molecule.SetProp("_Name", title)
    return (Chem.MolToMolBlock(molecule) + "$$$$\n").encode("ascii")


def _frame(geometry, energy, *, points=None, elements=None):
    lines = geometry.decode("ascii").splitlines()
    atoms = [line.split() for line in lines[2:]]
    elements = elements if elements is not None else [atom[0] for atom in atoms]
    points = points if points is not None else [atom[1:] for atom in atoms]
    coordinates = (f"{element} {' '.join(map(str, point))}" for element, point in zip(elements, points, strict=True))
    return ("\n".join((lines[0], str(energy), *coordinates)) + "\n").encode("ascii")


def _forbid_provider(*_args):
    pytest.fail("CREST ran for a nomination outside the prepared input contract")


@pytest.fixture
def nomination(tmp_path):
    _manifest(tmp_path)
    (tmp_path / "ligand.sdf").write_bytes(_sdf("CCCO"))
    return tmp_path


def test_picker_recommends_only_installed_crest_for_the_nominated_candidate(nomination, monkeypatch):
    monkeypatch.setattr(crest_runtime, "predict", _forbid_provider)

    def choice(example=None, installed=True):
        return next(row for row in screen_selection.plan(
            nomination, "ligand", example=example, available={crest.METHOD_ID: installed},
        ) if row.method.key == crest.METHOD_ID)

    correct = choice("lead_A")
    assert correct.installed and correct.applicable and correct.recommended
    assert choice().recommended
    wrong = choice("lead_B")
    assert not wrong.applicable and not wrong.recommended
    assert not choice("lead_A", installed=False).recommended
    (nomination / crest.MANIFEST).unlink()
    absent = choice()
    assert not absent.applicable and not absent.recommended


@pytest.mark.parametrize("filename", ["../ligand.sdf", "C:/ligand.sdf", "NUL.sdf", "ligand.xyz"])
def test_manifest_requires_same_directory_sdf_name(nomination, monkeypatch, filename):
    _manifest(nomination, structure=filename)
    monkeypatch.setattr(crest_runtime, "predict", _forbid_provider)
    result = crest.screen_run(nomination)
    assert result["status"] == "untested" and result["blocker"] == "invalid_crest_manifest"


def test_linked_sdf_does_not_reach_the_provider(nomination, monkeypatch):
    linked = nomination / "ligand.sdf"
    target = linked.rename(nomination / "other.sdf")
    try:
        linked.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    monkeypatch.setattr(crest_runtime, "predict", _forbid_provider)
    result = crest.screen_run(nomination)
    assert result["status"] == "untested" and result["blocker"] == "crest_structure_unreadable"


@pytest.mark.parametrize("problem, blocker", [
    ("wrong_title", "crest_candidate_mismatch"),
    ("two_records", "crest_invalid_structure"),
    ("unterminated", "crest_invalid_structure"),
    ("flat", "crest_invalid_structure"),
    ("implicit_hydrogens", "crest_unsupported_ligand"),
    ("undefined_stereo", "crest_unsupported_ligand"),
    ("isotope", "crest_unsupported_ligand"),
    ("salt", "crest_unsupported_ligand"),
])
def test_prepared_sdf_contract_blocks_ambiguous_molecules(nomination, monkeypatch, problem, blocker):
    sdf = nomination / "ligand.sdf"
    replacements = {
        "wrong_title": lambda: _sdf("CCCO", title="lead_a"),
        "two_records": lambda: sdf.read_bytes() * 2,
        "unterminated": lambda: sdf.read_bytes().removesuffix(b"$$$$\n"),
        "flat": lambda: _sdf("CCCO", dimension=2),
        "implicit_hydrogens": lambda: _sdf("CCCO", explicit_hydrogens=False),
        "undefined_stereo": lambda: _sdf("CC=CC"),
        "isotope": lambda: _sdf("[13CH3]CCO"),
        "salt": lambda: _sdf("CCC[NH3+].[Cl-]"),
    }
    sdf.write_bytes(replacements[problem]())
    monkeypatch.setattr(crest_runtime, "predict", _forbid_provider)
    result = crest.screen_run(nomination)
    assert result["status"] == "untested" and result["blocker"] == blocker
    assert result["conformers"] == []


@pytest.mark.parametrize("problem", ["missing_atom", "wrong_order", "nonfinite_energy", "bad_count", "trailing_frame"])
def test_malformed_provider_frames_are_reported_as_invalid(nomination, monkeypatch, problem):
    plan = crest.prepare(nomination)
    valid = _frame(plan.xyz, -3.0)
    elements = list(plan.elements)
    elements[0], elements[3] = elements[3], elements[0]
    lines = valid.splitlines(keepends=True)
    invalid = {
        "missing_atom": b"".join(lines[:-1]),
        "wrong_order": _frame(plan.xyz, -3.0, elements=elements),
        "nonfinite_energy": _frame(plan.xyz, "NaN"),
        "bad_count": b"1\n" + b"".join(lines[1:]),
        "trailing_frame": valid + b"2\n",
    }[problem]
    monkeypatch.setattr(crest_runtime, "predict", lambda _xyz: valid + invalid)
    result = crest.screen_run(nomination)
    assert result["status"] == "untested" and result["blocker"] == "crest_invalid_ensemble"
    assert result["conformers"] == []


def test_changed_connectivity_is_withheld_from_energy_order(nomination, monkeypatch):
    plan = crest.prepare(nomination)
    points = plan.molecule.GetConformer().GetPositions().tolist()
    points[-1][0] += 20.0
    broken = _frame(plan.xyz, -8.0, points=points)
    viable = _frame(plan.xyz, -3.0)
    monkeypatch.setattr(crest_runtime, "predict", lambda _xyz: broken + viable)
    result = crest.screen_run(nomination)
    assert result["status"] == "partial" and result["rejected_conformers"] == 1
    assert [row["energy_hartree"] for row in result["conformers"]] == [-3.0]
    assert result["conformers"][0]["relative_energy_kcal_mol"] == 0.0

    monkeypatch.setattr(crest_runtime, "predict", lambda _xyz: broken)
    result = crest.screen_run(nomination)
    assert result["status"] == "untested" and result["blocker"] == "crest_no_valid_conformers"


def test_mirror_image_cannot_supply_a_conformer_for_nominated_stereoisomer(nomination, monkeypatch):
    (nomination / "ligand.sdf").write_bytes(_sdf("C[C@H](O)CC"))
    plan = crest.prepare(nomination)
    reflected = [(-x, y, z) for x, y, z in plan.molecule.GetConformer().GetPositions()]
    native, other_stereoisomer = _frame(plan.xyz, -4.0), _frame(plan.xyz, -6.0, points=reflected)
    monkeypatch.setattr(crest_runtime, "predict", lambda _xyz: other_stereoisomer + native)
    result = crest.screen_run(nomination)
    assert result["status"] == "partial" and result["rejected_conformers"] == 1
    assert [row["energy_hartree"] for row in result["conformers"]] == [-4.0]


def test_opposite_double_bond_stereo_cannot_supply_a_conformer(nomination, monkeypatch):
    Chem = pytest.importorskip("rdkit.Chem")
    (nomination / "ligand.sdf").write_bytes(_sdf("C/C=C/C"))
    plan = crest.prepare(nomination)
    molblock = _sdf(r"C/C=C\C").decode("ascii").removesuffix("$$$$\n")
    other = Chem.MolFromMolBlock(molblock, removeHs=False)
    points = other.GetConformer().GetPositions()
    monkeypatch.setattr(crest_runtime, "predict", lambda _xyz: _frame(plan.xyz, -6.0, points=points))
    result = crest.screen_run(nomination)
    assert result["status"] == "untested" and result["blocker"] == "crest_no_valid_conformers"


def test_rank_scale_and_shared_projection_follow_validated_local_ensemble(nomination, monkeypatch):
    plan = crest.prepare(nomination)
    xyz = plan.xyz
    energies = (-3.001, -3.004, -3.002)

    def predict(prepared):
        assert prepared == xyz
        return b"".join(_frame(xyz, energy) for energy in energies)

    monkeypatch.setattr(crest_runtime, "predict", predict)
    result = crest.screen_run(nomination, candidate="lead_A")
    assert result["status"] == "complete" and result["candidate_id"] == "lead_A"
    assert result["structure_sha256"] == hashlib.sha256((nomination / "ligand.sdf").read_bytes()).hexdigest()
    rows = result["conformers"]
    assert [row["rank"] for row in rows] == [1, 2, 3]
    assert [row["energy_hartree"] for row in rows] == sorted(energies)
    assert [row["relative_energy_kcal_mol"] for row in rows] == pytest.approx([0, 1.255018948, 1.882528422])
    assert all(len(row["coordinates_angstrom"]) == len(plan.elements) for row in rows)

    exported = crest.report_view({**result, "private_path": "/private/ligand.sdf"})
    assert "elements" not in exported and "private_path" not in exported
    assert exported["status"] == "complete" and exported["candidate_id"] == "lead_A"
    assert exported["conformers"] == [{
        "rank": row["rank"], "energy_hartree": row["energy_hartree"],
        "relative_energy_kcal_mol": row["relative_energy_kcal_mol"],
    } for row in rows]
    invalid = crest.report_view({**result, "conformers": [{**rows[0], "relative_energy_kcal_mol": 8.0}, *rows[1:]]})
    assert invalid["status"] == "untested" and invalid["blocker"] == "crest_invalid_ensemble"


def test_oversized_valid_ensemble_retains_its_lowest_32_with_visible_coverage(nomination, monkeypatch):
    monkeypatch.setattr(crest_runtime, "predict", lambda xyz: b"".join(
        _frame(xyz, energy) for energy in reversed(range(-34, 0))
    ))
    result = crest.screen_run(nomination)
    assert result["status"] == "partial"
    assert (result["rejected_conformers"], result["omitted_conformers"]) == (0, 2)
    assert [row["energy_hartree"] for row in result["conformers"]] == list(range(-34, -2))
    view = crest.report_view(result)
    assert view["status"] == "partial" and view["omitted_conformers"] == 2
