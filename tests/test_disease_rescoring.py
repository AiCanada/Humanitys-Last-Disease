"""Tests for the MM-GBSA rescoring + FEP scaffold. CPU-only, OpenMM mocked."""

from __future__ import annotations

import numpy as np
import pytest
from biotite.structure import Atom, array
from biotite.structure.io.pdb import PDBFile

from rfproteina.disease import rescoring


def _two_chain_structure(path):
    atoms = []
    for i, chain in enumerate("AB"):
        for j in range(3):
            a = Atom(
                [float(i * 10 + j), 0.0, 0.0],
                chain_id=chain,
                res_id=j + 1,
                res_name="ALA",
                atom_name="CA",
                element="C",
            )
            atoms.append(a)
    structure = array(atoms)
    pdb = PDBFile()
    pdb.set_structure(structure)
    pdb.write(str(path))
    return path


def test_rescore_complex_subtracts_energies(tmp_path):
    structure = _two_chain_structure(tmp_path / "complex.pdb")

    # Energy keyed by which file (complex/receptor/ligand) so the subtraction is checked.
    energies = {"complex": -100.0, "receptor": -60.0, "ligand": -30.0}

    def fake_energy(pdb_path, implicit_solvent, forcefield, minimize):
        return energies[pdb_path.stem]

    result = rescoring.rescore_complex(
        structure, receptor_chains=["A"], ligand_chains=["B"], energy_fn=fake_energy
    )
    assert result.e_complex == -100.0
    assert result.dg_bind == pytest.approx(-100.0 - (-60.0) - (-30.0))
    assert result.dg_bind == pytest.approx(-10.0)
    assert result.units == "kcal/mol"


def test_rescore_rejects_overlapping_chains(tmp_path):
    structure = _two_chain_structure(tmp_path / "complex.pdb")
    with pytest.raises(ValueError, match="overlap"):
        rescoring.rescore_complex(structure, ["A"], ["A"], energy_fn=lambda *a: 0.0)


def test_rescore_rejects_empty_chains(tmp_path):
    structure = _two_chain_structure(tmp_path / "complex.pdb")
    with pytest.raises(ValueError, match="non-empty"):
        rescoring.rescore_complex(structure, [], ["B"], energy_fn=lambda *a: 0.0)


def test_rescore_missing_structure():
    with pytest.raises(FileNotFoundError):
        rescoring.rescore_complex("/no/such.pdb", ["A"], ["B"], energy_fn=lambda *a: 0.0)


def test_rescore_unavailable_openmm(tmp_path, monkeypatch):
    structure = _two_chain_structure(tmp_path / "complex.pdb")
    monkeypatch.setattr(rescoring, "openmm_available", lambda: False)
    with pytest.raises(rescoring.OpenMMUnavailableError, match="not installed"):
        rescoring.rescore_complex(structure, ["A"], ["B"])


def test_write_subset_selects_chains(tmp_path):
    structure = _two_chain_structure(tmp_path / "complex.pdb")
    out = rescoring._write_subset(structure, ["A"], tmp_path / "recA.pdb")
    atoms = PDBFile.read(str(out)).get_structure(model=1)
    assert set(np.unique(atoms.chain_id)) == {"A"}


def test_write_subset_errors_on_absent_chain(tmp_path):
    structure = _two_chain_structure(tmp_path / "complex.pdb")
    with pytest.raises(ValueError, match="no atoms for chains"):
        rescoring._write_subset(structure, ["Z"], tmp_path / "z.pdb")


def test_verdicts_dg_bind_bar():
    result = rescoring.MMGBSAResult(
        e_complex=-100, e_receptor=-60, e_ligand=-30, dg_bind=-10.0, units="kcal/mol",
        implicit_solvent="GBn2", forcefield="amber14-all.xml", minimized=True, notes=(),
    )
    v = rescoring.verdicts(result)
    # dg_bind -10 <= -5 favourable bar => clear (not a liability)
    assert v["dg_bind"]["state"] == "clear"
    assert "pending calibration" in v["dg_bind"]["threshold_source"]


def test_verdicts_dg_bind_weak_is_hit():
    result = rescoring.MMGBSAResult(
        e_complex=-100, e_receptor=-60, e_ligand=-40, dg_bind=0.0, units="kcal/mol",
        implicit_solvent="GBn2", forcefield="amber14-all.xml", minimized=True, notes=(),
    )
    # 0.0 is not <= -5 favourable bar => flagged as not favourable enough
    assert rescoring.verdicts(result)["dg_bind"]["state"] == "hit"


def test_plan_rbfe_is_scaffold_only():
    plan = rescoring.plan_rbfe("ligA", "ligB", backend="openfe")
    assert plan["status"] == "scaffold_only"
    assert plan["backend"] == "openfe"
    assert "backend_importable" in plan
    assert plan["edge"] == {"ligand_a": "ligA", "ligand_b": "ligB"}


def test_plan_rbfe_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown FEP backend"):
        rescoring.plan_rbfe("a", "b", backend="rosetta")


def test_fep_backends_available_reports_three():
    backends = rescoring.fep_backends_available()
    assert set(backends) == {"openfe", "pmx", "alchemlyb"}
    assert all(isinstance(v, bool) for v in backends.values())
