"""Tests for rfproteina.disease.interactions (CPU-only, fast)."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from rfproteina.disease import interactions as ix


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


def _mol_at(smiles, centroid):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    heavy = [conf.GetAtomPosition(a.GetIdx()) for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    cx = np.mean([p.x for p in heavy]); cy = np.mean([p.y for p in heavy]); cz = np.mean([p.z for p in heavy])
    shift = (centroid[0] - cx, centroid[1] - cy, centroid[2] - cz)
    for a in mol.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        conf.SetAtomPosition(a.GetIdx(), (p.x + shift[0], p.y + shift[1], p.z + shift[2]))
    return mol


def _protein(tmp_path):
    ring = [
        ("CG", "PHE", "A", 1, 1.4, 0.0, 0.0, "C"),
        ("CD1", "PHE", "A", 1, 0.7, 1.2, 0.0, "C"),
        ("CD2", "PHE", "A", 1, -0.7, 1.2, 0.0, "C"),
        ("CE1", "PHE", "A", 1, -1.4, 0.0, 0.0, "C"),
        ("CE2", "PHE", "A", 1, -0.7, -1.2, 0.0, "C"),
        ("CZ", "PHE", "A", 1, 0.7, -1.2, 0.0, "C"),
        ("OD1", "ASP", "A", 2, 20.0, 0.0, 0.0, "O"),
        ("OD2", "ASP", "A", 2, 21.0, 0.0, 0.0, "O"),
        ("OG", "SER", "A", 3, 40.0, 0.0, 0.0, "O"),
        ("O", "GLY", "A", 4, 60.0, 0.0, 0.0, "O"),
    ]
    return _write_pdb(tmp_path / "prot.pdb", ring)


def test_pi_stacking_and_hydrophobic_when_benzene_sits_over_phe(tmp_path):
    protein = _protein(tmp_path)
    mol = _mol_at("c1ccccc1", (0.0, 0.0, 4.0))
    fp = ix.native_fingerprint(protein, mol)
    assert fp.counts["pi_stacking"] > 0
    assert fp.counts["hydrophobic"] > 0


def test_everything_zero_when_ligand_is_far(tmp_path):
    protein = _protein(tmp_path)
    mol = _mol_at("c1ccccc1", (0.0, 0.0, 100.0))
    fp = ix.native_fingerprint(protein, mol)
    assert sum(fp.counts.values()) == 0


def test_hbond_proxy_counts_oxygen_pair(tmp_path):
    protein = _protein(tmp_path)
    mol = _mol_at("CCO", (40.0, 0.0, 2.0))
    fp = ix.native_fingerprint(protein, mol)
    assert fp.counts["hbond"] > 0


def test_salt_bridge_between_cation_and_aspartate(tmp_path):
    protein = _protein(tmp_path)
    mol = _mol_at("C[NH3+]", (20.5, 0.0, 2.0))
    fp = ix.native_fingerprint(protein, mol)
    assert fp.counts["salt_bridge"] > 0


def test_halogen_contact(tmp_path):
    protein = _protein(tmp_path)
    mol = _mol_at("Clc1ccccc1", (60.0, 0.0, 2.0))
    fp = ix.native_fingerprint(protein, mol)
    assert fp.counts["halogen"] > 0


def test_native_fingerprint_refuses_mol_without_conformer(tmp_path):
    protein = _protein(tmp_path)
    flat = Chem.MolFromSmiles("CCO")
    with pytest.raises(ix.StructureUnreadableError):
        ix.native_fingerprint(protein, flat)


def test_native_verdicts_are_all_no_cut(tmp_path):
    protein = _protein(tmp_path)
    mol = _mol_at("c1ccccc1", (0.0, 0.0, 4.0))
    fp = ix.native_fingerprint(protein, mol)
    verdicts = ix.native_verdicts(fp)
    assert {v["state"] for v in verdicts.values()} == {"no_cut"}


def test_fnat_identical_is_one_and_partial_is_fractional(tmp_path):
    reference = _write_pdb(
        tmp_path / "ref.pdb",
        [
            ("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
            ("CA", "ALA", "A", 2, 0.0, 0.0, 10.0, "C"),
            ("CA", "ALA", "B", 1, 3.0, 0.0, 0.0, "C"),
            ("CA", "ALA", "B", 2, 3.0, 0.0, 10.0, "C"),
        ],
    )
    same = ix.compare_to_reference(reference, reference, "A", "B")
    assert same.fnat == 1.0
    assert same.fnonnat == 0.0

    model = _write_pdb(
        tmp_path / "model.pdb",
        [
            ("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
            ("CA", "ALA", "A", 2, 0.0, 0.0, 10.0, "C"),
            ("CA", "ALA", "B", 1, 3.0, 0.0, 0.0, "C"),
            ("CA", "ALA", "B", 2, 3.0, 0.0, 100.0, "C"),
        ],
    )
    partial = ix.compare_to_reference(model, reference, "A", "B")
    assert partial.fnat == 0.5


def test_comparison_verdicts_gate_only_when_bar_supplied(tmp_path):
    reference = _write_pdb(
        tmp_path / "r.pdb",
        [
            ("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
            ("CA", "ALA", "B", 1, 3.0, 0.0, 0.0, "C"),
        ],
    )
    comp = ix.compare_to_reference(reference, reference, "A", "B")
    assert ix.comparison_verdicts(comp)["fnat"]["state"] == "no_cut"
    assert ix.comparison_verdicts(comp, fnat_min=0.5)["fnat"]["state"] == "clear"
    assert ix.comparison_verdicts(comp, fnat_min=1.5)["fnat"]["state"] == "hit"


def test_prolif_and_dockq_refuse_loudly_when_absent(tmp_path):
    protein = _protein(tmp_path)
    with pytest.raises(ix.ProLIFUnavailableError):
        ix.prolif_fingerprint(protein, protein)
    with pytest.raises(ix.DockQUnavailableError):
        ix.dockq(protein, protein)
