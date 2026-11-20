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

    result = rescoring.rescore_complex(structure, receptor_chains=["A"], ligand_chains=["B"], energy_fn=fake_energy)
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
        e_complex=-100,
        e_receptor=-60,
        e_ligand=-30,
        dg_bind=-10.0,
        units="kcal/mol",
        implicit_solvent="GBn2",
        forcefield="amber14-all.xml",
        minimized=True,
        notes=(),
    )
    v = rescoring.verdicts(result)
    # dg_bind -10 <= -5 favourable bar => clear (not a liability)
    assert v["dg_bind"]["state"] == "clear"
    assert "pending calibration" in v["dg_bind"]["threshold_source"]


def test_verdicts_dg_bind_weak_is_hit():
    result = rescoring.MMGBSAResult(
        e_complex=-100,
        e_receptor=-60,
        e_ligand=-40,
        dg_bind=0.0,
        units="kcal/mol",
        implicit_solvent="GBn2",
        forcefield="amber14-all.xml",
        minimized=True,
        notes=(),
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


# --------------------------------------------- the real OpenMM back-end
#
# Everything above injects ``energy_fn`` or stubs ``openmm_available``, which is
# how this module came to ship a call OpenMM has never accepted: the mocked path
# never reached ``_openmm_implicit_energy``. These run it for real, and are
# skipped rather than failed where OpenMM is not installed.

needs_openmm = pytest.mark.skipif(not rescoring.openmm_available(), reason="OpenMM is not installed")

_DIPEPTIDE = (
    (1, "N", 1, -1.204, 0.000, 0.000, "N"),
    (2, "CA", 1, 0.246, 0.000, 0.000, "C"),
    (3, "C", 1, 0.800, 1.420, 0.000, "C"),
    (4, "O", 1, 0.100, 2.420, 0.000, "O"),
    (5, "CB", 1, 0.760, -0.760, 1.210, "C"),
    (6, "N", 2, 2.130, 1.520, 0.000, "N"),
    (7, "CA", 2, 2.800, 2.810, 0.000, "C"),
    (8, "C", 2, 4.310, 2.620, 0.000, "C"),
    (9, "O", 2, 4.800, 1.490, 0.000, "O"),
    (10, "CB", 2, 2.380, 3.640, 1.210, "C"),
)


def _peptide_pdb(path, chains=("A",), offset=0.0):
    """An all-atom ALA-ALA per chain, in fixed-width PDB columns.

    All-atom on purpose: the CA-only helper above cannot be parameterised by a
    force field, so it can only ever exercise the mocked path. PDBFixer adds the
    hydrogens and the terminal atoms.
    """
    lines = []
    serial = 0
    for index, chain in enumerate(chains):
        shift = index * 25.0 + offset
        for _, name, seq, x, y, z, element in _DIPEPTIDE:
            serial += 1
            lines.append(
                f"ATOM  {serial:5d} {name:^4s} ALA {chain}{seq:4d}    "
                f"{x + shift:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
            )
    path.write_text("\n".join(lines) + "\nTER\nEND\n", encoding="utf-8")
    return path


@needs_openmm
def test_the_real_openmm_backend_returns_an_energy(tmp_path):
    """The regression that would have caught it.

    ``ForceField.createSystem`` has never taken ``implicitSolvent=``; that
    keyword belongs to ``AmberPrmtopFile.createSystem``. OpenMM 8 refuses the
    call rather than ignoring the argument, so every real rescore raised
    ValueError while the mocked tests stayed green.
    """
    pdb = _peptide_pdb(tmp_path / "ala2.pdb")
    energy = rescoring._openmm_implicit_energy(pdb, "GBn2", rescoring.DEFAULT_FORCEFIELD, False)
    assert isinstance(energy, float)
    assert energy == pytest.approx(energy)  # finite: NaN would fail its own equality
    assert -1e5 < energy < 1e5


@needs_openmm
def test_every_named_implicit_solvent_is_one_openmm_accepts(tmp_path):
    """The names are not decoration: each must name a shipped implicit/*.xml."""
    pdb = _peptide_pdb(tmp_path / "ala2.pdb")
    for model in ("GBn2", "GBn", "OBC1", "OBC2", "HCT"):
        energy = rescoring._openmm_implicit_energy(pdb, model, rescoring.DEFAULT_FORCEFIELD, False)
        assert isinstance(energy, float)


@needs_openmm
def test_an_unknown_implicit_solvent_is_refused_before_openmm_is_asked(tmp_path):
    pdb = _peptide_pdb(tmp_path / "ala2.pdb")
    with pytest.raises(ValueError, match="unknown implicit solvent"):
        rescoring._openmm_implicit_energy(pdb, "NOPE", rescoring.DEFAULT_FORCEFIELD, False)


@needs_openmm
@pytest.mark.slow
def test_rescore_complex_runs_end_to_end_without_an_injected_energy_fn(tmp_path):
    """The acceptance: three real subsystem energies and a dG from them."""
    pdb = _peptide_pdb(tmp_path / "pair.pdb", chains=("A", "B"))
    result = rescoring.rescore_complex(pdb, receptor_chains=["A"], ligand_chains=["B"], minimize=False)
    record = result.as_record()
    assert record["units"] == "kcal/mol"
    assert record["minimized"] is False
    assert record["dg_bind"] == pytest.approx(record["e_complex"] - record["e_receptor"] - record["e_ligand"], abs=1e-6)
