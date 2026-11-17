"""Tests for rfproteina.disease.pose_validation (CPU-only, fast)."""

from __future__ import annotations

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
        pv._read_ligand(tmp_path / "nope.sdf")
    bad = tmp_path / "x.xyz"
    bad.write_text("nonsense", encoding="utf-8")
    with pytest.raises(pv.PoseUnreadableError):
        pv._read_ligand(bad)


def test_read_ligand_refuses_mol_without_conformer():
    flat = Chem.MolFromSmiles("CCO")
    with pytest.raises(pv.PoseUnreadableError):
        pv._read_ligand(flat)


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
