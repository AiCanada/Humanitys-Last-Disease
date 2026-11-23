"""Tests for rfproteina.disease.pose_validation (CPU-only, fast)."""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from rfproteina.disease import pose_validation as pv


def _ethanol():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _write_pdb(path, atoms):
    lines = []
    for i, (name, resname, chain, resid, x, y, z, element) in enumerate(atoms, start=1):
        lines.append(
            f"ATOM  {i:>5} {name:<4} {resname:>3} {chain}{resid:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {element:>2}"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_validate_pose_on_clean_ligand_has_no_failures():
    result = pv.validate_pose(_ethanol())
    assert result.posebusters_mode == "mol"
    assert result.posebusters_failures == 0
    assert isinstance(result.internal_strain_kcal_mol, float)
    verdicts = pv.verdicts(result)
    assert verdicts["posebusters_failures"]["state"] == "clear"
    assert verdicts["internal_strain_kcal_mol"]["state"] == "no_cut"
    assert verdicts["protein_clashes"]["state"] == "untested"


def test_strain_rises_for_a_distorted_pose():
    mol = _ethanol()
    conf = mol.GetConformer()
    pos = conf.GetAtomPosition(0)
    conf.SetAtomPosition(0, (pos.x + 0.6, pos.y, pos.z))
    strain, reason = pv.internal_strain(mol)
    assert reason is None
    assert strain > 1.0


def test_read_ligand_refuses_missing_and_unsupported(tmp_path):
    with pytest.raises(pv.PoseUnreadableError):
        pv.read_ligand(tmp_path / "nope.sdf")
    bad = tmp_path / "x.xyz"
    bad.write_text("nonsense", encoding="utf-8")
    with pytest.raises(pv.PoseUnreadableError):
        pv.read_ligand(bad)


def test_read_ligand_refuses_mol_without_conformer():
    flat = Chem.MolFromSmiles("CCO")
    with pytest.raises(pv.PoseUnreadableError):
        pv.read_ligand(flat)


def test_picker_checks_real_pose_inputs_without_running_posebusters(tmp_path, monkeypatch):
    from rfproteina.disease import screen_selection

    monkeypatch.setattr(pv, "run_posebusters", lambda *_: pytest.fail("The picker ran PoseBusters"))
    pose = _ethanol()
    pose.SetProp("_Name", "ethanol")
    content = Chem.MolToMolBlock(pose) + "$$$$\n"
    drawing = Chem.MolFromSmiles("CCO")
    drawing.SetProp("_Name", "drawing")
    path = tmp_path / "ligand_candidates.sdf"
    path.write_text(content)

    def state(candidate=None):
        choice, = screen_selection.plan(tmp_path, "ligand", example=candidate, available={"posebusters": True})
        return choice.applicable, choice.recommended

    assert state() == (True, True)
    path.write_text(content + Chem.MolToMolBlock(drawing) + "$$$$\n")
    assert state() == (True, False)
    assert state("ethanol") == (True, True)
    assert state("drawing") == (False, False)
    path.write_text(content + content)
    assert state() == (False, False)


def test_protein_clashes_counts_overlap_and_zero_when_far(tmp_path):
    mol = _ethanol()
    conf = mol.GetConformer()
    first = conf.GetAtomPosition(0)
    protein = _write_pdb(
        tmp_path / "prot.pdb",
        [("CA", "ALA", "A", 1, first.x, first.y, first.z, "C")],
    )
    count, closest, reason = pv.protein_clashes(mol, protein)
    assert reason is None
    assert count >= 1
    assert closest < 0.1

    far = _write_pdb(tmp_path / "far.pdb", [("CA", "ALA", "A", 1, 999.0, 999.0, 999.0, "C")])
    count_far, _, reason_far = pv.protein_clashes(mol, far)
    assert reason_far is None
    assert count_far == 0


def test_validate_pose_with_protein_fills_clash_fields(tmp_path):
    mol = _ethanol()
    protein = _write_pdb(tmp_path / "p.pdb", [("CA", "ALA", "A", 1, 100.0, 100.0, 100.0, "C")])
    result = pv.validate_pose(mol, protein_path=protein)
    assert result.posebusters_mode == "dock"
    assert result.protein_clashes == 0
    assert pv.verdicts(result)["protein_clashes"]["state"] == "clear"


def test_failures_map_to_hit_in_verdicts():
    built = pv.PoseValidation(
        source="x",
        posebusters_mode="mol",
        posebusters_checks={"bond_lengths": False},
        posebusters_failures=1,
        internal_strain_kcal_mol=5.0,
        internal_strain_reason=None,
        protein_clashes=None,
        min_protein_contact_a=None,
        clash_reason="no_protein_supplied",
    )
    assert pv.verdicts(built)["posebusters_failures"]["state"] == "hit"


def test_every_threshold_names_a_source():
    for rule in pv.REFERENCE_THRESHOLDS.values():
        assert rule["source"]


@pytest.mark.parametrize(("rows", "state", "count"), [
    ([], "untested", None),
    ([{"bond_lengths": True}], "untested", None),
    ([{"bond_lengths": False}], "hit", 1),
    ([dict.fromkeys(pv.POSEBUSTERS_MOL_CHECKS, True)] * 2, "untested", None),
    ([{**dict.fromkeys(pv.POSEBUSTERS_MOL_CHECKS, True), "internal_energy": np.nan}], "untested", None),
    ([{**dict.fromkeys(pv.POSEBUSTERS_MOL_CHECKS, True), "bond_lengths": "False"}], "untested", None),
    ([{**dict.fromkeys(pv.POSEBUSTERS_MOL_CHECKS, True), "mol_pred_loaded": False}], "untested", None),
])
def test_only_attributable_boolean_evidence_can_clear(monkeypatch, rows, state, count):
    from posebusters import PoseBusters

    monkeypatch.setattr(PoseBusters, "bust", lambda *args, **kwargs: pd.DataFrame(rows))
    result = pv.validate_pose(_ethanol())
    assert result.posebusters_failures == count
    assert result.posebusters_reason
    assert pv.verdicts(result)["posebusters_failures"]["state"] == state


def test_optional_missing_provider_and_empty_input_stay_unmeasured(monkeypatch, tmp_path):
    import sys

    monkeypatch.setitem(sys.modules, "posebusters", None)
    result = pv.validate_pose(_ethanol(), require_posebusters=False)
    assert result.posebusters_failures is None and result.posebusters_reason == "posebusters_unavailable"
    assert pv.verdicts(result)["posebusters_failures"]["state"] == "untested"
    assert result.internal_strain_kcal_mol is not None
    with pytest.raises(pv.PoseBustersUnavailableError):
        pv.validate_pose(_ethanol())
    empty = tmp_path / "empty.sdf"
    empty.touch()
    with pytest.raises(pv.PoseUnreadableError):
        pv.validate_pose(empty)


def test_rejects_ambiguous_or_non_3d_ligand_geometry(tmp_path):
    mol = _ethanol()
    path = tmp_path / "two.sdf"
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)
        writer.write(mol)
    with pytest.raises(pv.PoseUnreadableError, match="multiple_records"):
        pv.validate_pose(path)
    mol.AddConformer(Chem.Conformer(mol.GetConformer()), assignId=True)
    with pytest.raises(pv.PoseUnreadableError, match="multiple_conformers"):
        pv.validate_pose(mol)
    drawing = Chem.MolFromSmiles("CCO")
    AllChem.Compute2DCoords(drawing)
    with pytest.raises(pv.PoseUnreadableError, match="coordinates_2d"):
        pv.validate_pose(drawing)
    mol = _ethanol()
    mol.GetConformer().SetAtomPosition(0, [float("nan"), 0., 0.])
    with pytest.raises(pv.PoseUnreadableError, match="coordinates_nonfinite"):
        pv.validate_pose(mol)


@pytest.mark.parametrize("compressed", [False, True])
def test_dock_cif_preserves_the_pdb_checks_and_excludes_water_from_native_counts(tmp_path, compressed):
    from biotite.structure.io import load_structure, pdbx
    from posebusters import PoseBusters

    mol = _ethanol()
    pdb = _write_pdb(tmp_path / "reference.pdb", [
        ("CA", "ALA", "A", 1, 60., 65., 70., "C"), ("O", "HOH", "W", 2, 0., 0., 0., "O"),
        ("C", "LIG", "L", 3, 0., 0., 0., "C"),
    ])
    # Preserve HETATM for the non-protein records in both representations.
    pdb.write_text("".join(line.replace("ATOM  ", "HETATM") if "HOH" in line or "LIG" in line else line
                           for line in pdb.read_text().splitlines(keepends=True)))
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, load_structure(str(pdb), model=1))
    path = tmp_path / "reference.cif"
    cif.write(path)
    if compressed:
        encoded = gzip.compress(path.read_bytes())
        path = tmp_path / "reference.cif.gz"
        path.write_bytes(encoded)
    original = PoseBusters(config="dock").bust([mol], None, str(pdb)).iloc[0].to_dict()
    result = pv.validate_pose(mol, path)
    assert result.posebusters_reason is None and result.posebusters_checks == original
    assert result.protein_clashes == 0
    assert result.min_protein_contact_a > 80


def test_unreadable_receptor_records_two_unmeasured_readings(tmp_path):
    result = pv.validate_pose(_ethanol(), tmp_path / "gone.pdb")
    verdict = pv.verdicts(result)
    assert verdict["posebusters_failures"]["state"] == verdict["protein_clashes"]["state"] == "untested"
    assert result.posebusters_reason == result.clash_reason


@pytest.mark.parametrize(("status", "energy", "reason"), [
    (1, 1.0, "mmff_minimization_unconverged"), (0, np.inf, "mmff_energy_nonfinite"),
])
def test_failed_mmff_has_no_strain(monkeypatch, status, energy, reason):
    from unittest.mock import Mock

    mol = _ethanol()
    field = Mock()
    field.CalcEnergy.side_effect = [1.0, energy]
    field.Minimize.return_value = status
    monkeypatch.setattr(AllChem, "MMFFGetMoleculeForceField", lambda *args: field)
    assert pv.internal_strain(mol) == (None, reason)
