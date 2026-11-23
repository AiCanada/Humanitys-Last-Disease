"""Tests for PoseBusters ligand validation metric."""

import numpy as np
import pytest
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.ml.utils.testing import cached_parse

from rfproteina.metrics.posebusters_metrics import validate_single_ligand


def _has_posebusters() -> bool:
    """Check if PoseBusters is installed."""
    try:
        from posebusters import PoseBusters  # noqa: F401

        return True
    except ImportError:
        return False


skip_if_no_posebusters = pytest.mark.skipif(not _has_posebusters(), reason="PoseBusters not installed")


def _extract_sa8_ligand():
    """Extract SA8 ligand from 6wtf structure."""
    atom_array = cached_parse("6wtf")["atom_array"]
    ensure_annotations(atom_array, "is_ligand")
    ligand_atoms = atom_array[atom_array.is_ligand]
    sa8_mask = (ligand_atoms.chain_id == "E") & (ligand_atoms.res_id == 803) & (ligand_atoms.res_name == "SA8")
    return ligand_atoms[sa8_mask]


@skip_if_no_posebusters
@pytest.mark.parametrize(
    "transform,expected_pass",
    [
        ("identity", True),
        ("noise", False),
        ("mirror", True),
    ],
)
def test_validate_sa8(transform: str, expected_pass: bool):
    """SA8 ligand: identity and mirror pass PoseBusters; large noise fails."""
    sa8 = _extract_sa8_ligand().copy()

    if transform == "noise":
        rng = np.random.default_rng(42)
        sa8.coord = sa8.coord + rng.normal(scale=2.0, size=sa8.coord.shape).astype(sa8.coord.dtype)
    elif transform == "mirror":
        sa8.coord = sa8.coord.copy()
        sa8.coord[:, 0] = -sa8.coord[:, 0]

    result = validate_single_ligand(sa8)
    assert result["passes_validation"] is expected_pass
    assert result["n_unresolved_hydrogens"] == np.isnan(sa8.coord).all(axis=1).sum()


def _ethanol_atoms():
    from biotite.interface.rdkit import from_mol
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    assert AllChem.MMFFOptimizeMolecule(mol) == 0
    atoms = from_mol(mol, conformer_id=0)
    atoms.chain_id[:] = "L"
    atoms.res_name[:] = "ETL"
    atoms.res_id[:] = 1
    atoms.hetero[:] = True
    atoms.set_annotation("is_ligand", np.ones(len(atoms), dtype=bool))
    return atoms


def test_whole_pose_and_missing_measurements_from_the_normal_metric_entrypoint():
    from rfproteina.metrics.posebusters_metrics import PoseBustersMetric

    atoms = _ethanol_atoms()
    other = atoms.copy()
    other.ins_code[:] = "A"
    other.coord[0, 0] = np.nan
    metric = PoseBustersMetric(output_level="per_ligand", timeout_seconds=None)
    result = metric.compute(atoms + other)
    assert result["n_ligands"] == 2 and result["n_measured"] == result["n_passed"] == 1
    assert result["pass_rate"] == 0.5 and result["measured_pass_rate"] == 1
    assert result["ligand_0_id"] != result["ligand_1_id"]
    assert result["ligand_1_passes_validation"] is None and result["ligand_1_blocker"]


def test_unobserved_and_excluded_checks_do_not_pass(monkeypatch):
    import pandas as pd
    from posebusters import PoseBusters

    from rfproteina.metrics.posebusters_metrics import validate_ligands
    from rfproteina.utils.pose_checks import POSEBUSTERS_MOL_CHECKS

    atoms = _ethanol_atoms()
    flags = dict.fromkeys(POSEBUSTERS_MOL_CHECKS, True)
    monkeypatch.setattr(PoseBusters, "bust", lambda *a, **k: pd.DataFrame([flags]))
    flags["all_atoms_connected"] = False
    assert validate_ligands(atoms, timeout_seconds=None)[0]["passes_validation"] is True
    assert validate_ligands(atoms, excluded_checks=[], timeout_seconds=None)[0]["passes_validation"] is False
    flags["all_atoms_connected"] = True
    flags["bond_lengths"] = np.nan
    result = validate_single_ligand(atoms, timeout_seconds=None)
    assert result["passes_validation"] is None and result["blocker"] == "posebusters_checks_incomplete"
    flags["internal_energy"] = False
    assert validate_single_ligand(atoms, timeout_seconds=None)["passes_validation"] is False


def test_timeout_and_missing_checks_cannot_raise_reported_pass_rates(monkeypatch):
    from rfproteina.metrics import posebusters_metrics as pm

    def timed_out(*args):
        raise TimeoutError

    monkeypatch.setattr(pm, "pose_checks", timed_out)
    timed = validate_single_ligand(_ethanol_atoms(), timeout_seconds=None)
    assert timed["timed_out"] and timed["passes_validation"] is None
    records = [{"passes_validation": True, "pb_bonds": True},
               {"passes_validation": False, "pb_energy": False}, timed]
    table = pm.aggregate_ligand_results(records)
    assert table["n_measured"] == 2 and table["n_unmeasured"] == 1
    assert table["pass_rate"] == table["pb_bonds_pass_rate"] == 1 / 3
    assert table["pb_energy_pass_rate"] == 0 and table["pb_energy_n_measured"] == 1
    assert np.isnan(pm.aggregate_ligand_results([timed])["pass_rate"])
