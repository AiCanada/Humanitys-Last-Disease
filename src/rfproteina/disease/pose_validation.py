"""Ligand-pose validation for a finished generation, by PoseBusters and RDKit.

A docked or co-folded ligand can clear an affinity panel and still sit in a
pose no chemist would accept: bonds the wrong length, a ring bent flat, atoms
driven into each other or into the protein. Those are physical-validity
questions, not affinity questions, and the safety panel in
:mod:`rfproteina.disease.toxicology` never asks them. This module asks them.

Three readings, each from a named tool and each labelled for what it is:

* **PoseBusters** (Buttenschoen, Morris and Deane, *Chem. Sci.* 2024;
  tested here with version 0.6.5). The ``mol``
  configuration tests the ligand alone -- loadable, sanitisable, connected,
  no radicals, bond lengths and angles, internal steric clash, ring flatness,
  double-bond flatness and internal energy. The ``dock`` configuration adds the
  protein and tests the minimum distance to it and the volume overlap with it.
  The paper defines PB-valid using all tests in its suite. Here a clear requires
  a boolean pass for every required check in the selected mode.

* **Internal strain**, from RDKit's MMFF94 force field: the pose's energy minus
  the energy of a converged local minimum for the same atoms, in kcal/mol.
  This project records the measured difference descriptively.

* **Protein clashes**, native: heavy-atom pairs of ligand against protein whose
  separation is shorter than the sum of their Bondi (1964) van der Waals radii
  minus a tolerance. A count and the minimum contact distance, reported beside
  a *suggested* bar, never a published one.

Everything here is CPU-only and makes no network call. A ligand this cannot
read, or a PoseBusters that is not installed, is refused loudly rather than
guessed around: a wrong clear on a pose is worse than no reading.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rfproteina.utils.pose_checks import (
    POSEBUSTERS_DOCK_CHECKS,
    POSEBUSTERS_MOL_CHECKS,
    PoseBustersResultError,
    PoseBustersUnavailableError,
    pose_checks,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from biotite.structure import AtomArray
    from rdkit.Chem import Mol

METHOD = "posebusters_rdkit_v2"

#: Bondi (1964) van der Waals radii in angstrom, for the clash proxy only.
#: Elements outside this table fall back to :data:`DEFAULT_VDW_A`.
BONDI_VDW_A: dict[str, float] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
}
DEFAULT_VDW_A = 1.70
#: Allowed overlap before a heavy-atom pair is counted as a clash, in angstrom.
#: A suggested tolerance, not a published one.
CLASH_TOLERANCE_A = 0.4

#: Thresholds, each with its provenance. A bar with no source is a bar nobody
#: can check, so none is left empty.
REFERENCE_THRESHOLDS: dict[str, dict[str, object]] = {
    "posebusters_failures": {
        "max": 0,
        "source": (
            "Buttenschoen et al., Chem. Sci. 2024 (doi:10.1039/D3SC04185A) define PB-valid "
            "using all tests in their suite. This program applies that rule to the selected mode."
        ),
    },
    "protein_clashes": {
        "max": 0,
        "source": (
            "suggested default pending calibration: zero heavy-atom pairs overlapping beyond "
            f"the {CLASH_TOLERANCE_A} A tolerance. Bondi 1964 radii; no published pass mark."
        ),
    },
    "internal_strain_kcal_mol": {
        "max": None,
        "source": (
            "Descriptive MMFF94 energy difference. This project has no calibrated cutoff for this measurement."
        ),
    },
}


@dataclass(frozen=True)
class PoseValidation:
    """One ligand pose, validated. Energies are kcal/mol; distances are angstrom."""

    source: str
    posebusters_mode: str
    posebusters_checks: dict[str, bool | None]
    posebusters_failures: int | None
    internal_strain_kcal_mol: float | None
    internal_strain_reason: str | None
    protein_clashes: int | None
    min_protein_contact_a: float | None
    clash_reason: str | None
    method: str = METHOD
    notes: tuple[str, ...] = field(default_factory=tuple)
    posebusters_reason: str | None = None

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["posebusters_checks"] = dict(self.posebusters_checks)
        record["notes"] = list(self.notes)
        record["reference_thresholds"] = REFERENCE_THRESHOLDS
        return record


class PoseUnreadableError(Exception):
    """The ligand could not be read, or the structure does not parse. Never guessed around."""


def read_ligand(source: str | Path | Mol) -> Mol:
    """An RDKit mol with 3D coordinates, from an SDF/MOL/PDB path or an existing mol."""
    try:
        from rdkit import Chem
    except ImportError as error:  # pragma: no cover - rdkit is pinned in pyproject
        raise PoseUnreadableError(f"rdkit_unavailable:{error}") from error
    if isinstance(source, Chem.Mol):
        mol = source
    elif isinstance(source, str | Path):
        path = Path(source)
        if not path.is_file():
            raise PoseUnreadableError(f"ligand_absent:{path.name}")
        suffix = path.suffix.lower()
        try:
            if suffix in {".sdf", ".mol"}:
                with path.open("rb") as stream, Chem.ForwardSDMolSupplier(stream, removeHs=False) as supplier:
                    mol = next(supplier, None)
                    if next(supplier, ...) is not ...:
                        raise PoseUnreadableError("ligand_multiple_records")
            elif suffix in {".pdb", ".ent"}:
                mol = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=True)
            elif suffix == ".mol2":
                with path.open("rb") as stream:
                    if sum(line.strip() == b"@<TRIPOS>MOLECULE" for line in stream) > 1:
                        raise PoseUnreadableError("ligand_multiple_records")
                mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=True)
            else:
                raise PoseUnreadableError(f"unsupported_ligand_format:{suffix or 'none'}")
        except (OSError, RuntimeError, ValueError) as error:
            raise PoseUnreadableError("ligand_unparsable") from error
    else:
        raise PoseUnreadableError("unsupported_ligand_object")
    if mol is None or not mol.GetNumAtoms():
        raise PoseUnreadableError("ligand_unparsable")
    if mol.GetNumConformers() == 0:
        raise PoseUnreadableError("ligand_without_conformer")
    if mol.GetNumConformers() != 1:
        raise PoseUnreadableError("ligand_multiple_conformers")
    conformer = mol.GetConformer()
    if not conformer.Is3D():
        raise PoseUnreadableError("ligand_coordinates_2d")
    if not np.isfinite(conformer.GetPositions()).all():
        raise PoseUnreadableError("ligand_coordinates_nonfinite")
    return mol


def run_posebusters(mol: Mol, protein_path: str | Path | None = None) -> tuple[str, dict[str, bool | None]]:
    """PoseBusters ``mol`` (ligand only) or ``dock`` (with a protein) checks.

    Returns the mode run and a dict of check name to pass/fail. Raises
    :class:`PoseBustersUnavailableError` if PoseBusters is not installed; the
    caller decides whether that is fatal, so the absence is never hidden behind
    an invented all-clear.
    """
    receptor = _posebusters_receptor(Path(protein_path)) if protein_path is not None else None
    return pose_checks(mol, receptor)


def read_nominated_pose(structure: str, problem: str) -> Mol:
    """Read a single nominated SDF pose with its supplied coordinates and atom identities."""
    from rdkit import Chem

    if problem:
        raise PoseUnreadableError(problem)
    mol = Chem.MolFromMolBlock(structure, removeHs=False, strictParsing=True)
    if mol is None:
        raise PoseUnreadableError("ligand_unparsable")
    if any(a.GetAtomicNum() == 0 or a.HasQuery() for a in mol.GetAtoms()) or any(b.HasQuery() for b in mol.GetBonds()):
        raise PoseUnreadableError("ligand_contains_query_or_dummy")
    return read_ligand(mol)


def _failure_count(mode: str, checks: dict[str, bool | None], reason: str | None) -> tuple[int | None, str | None]:
    if mode == "unavailable":
        return None, "posebusters_unavailable"
    if reason and reason != "posebusters_checks_incomplete":
        return None, reason
    required = POSEBUSTERS_DOCK_CHECKS if mode == "dock" else POSEBUSTERS_MOL_CHECKS
    if checks.get("mol_pred_loaded") is False or checks.get("mol_cond_loaded") is False:
        return None, "posebusters_input_not_loaded"
    failures = sum(value is False for value in checks.values())
    if any(name not in checks for name in required) or any(type(value) is not bool for value in checks.values()):
        return failures or None, "posebusters_checks_incomplete"
    return failures, None


def internal_strain(mol: Mol) -> tuple[float | None, str | None]:
    """MMFF94 strain of the pose: its energy minus a relaxed copy's, in kcal/mol.

    Returns ``(None, reason)`` when MMFF has no parameters for the molecule,
    rather than substituting a different force field silently.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as error:  # pragma: no cover - rdkit is pinned
        return None, f"rdkit_unavailable:{type(error).__name__}"
    try:
        probe = Chem.Mol(mol)
        props = AllChem.MMFFGetMoleculeProperties(probe)
        if props is None:
            return None, "mmff_no_parameters"
        pose_field = AllChem.MMFFGetMoleculeForceField(probe, props)
        if pose_field is None:
            return None, "mmff_no_forcefield"
        pose_energy = float(pose_field.CalcEnergy())
        if not np.isfinite(pose_energy):
            return None, "mmff_energy_nonfinite"
        if pose_field.Minimize(maxIts=1000) != 0:
            return None, "mmff_minimization_unconverged"
        relaxed_energy = float(pose_field.CalcEnergy())
    except Exception as error:  # any force-field failure is the same answer
        return None, f"mmff_failed:{type(error).__name__}"
    strain = pose_energy - relaxed_energy
    return (strain, None) if np.isfinite(strain) else (None, "mmff_energy_nonfinite")


def _heavy_coords_and_radii(mol: Mol) -> tuple[np.ndarray, np.ndarray]:
    conformer = mol.GetConformer()
    coords: list[list[float]] = []
    radii: list[float] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        position = conformer.GetAtomPosition(atom.GetIdx())
        coords.append([position.x, position.y, position.z])
        radii.append(BONDI_VDW_A.get(atom.GetSymbol().upper(), DEFAULT_VDW_A))
    return np.asarray(coords, dtype=float), np.asarray(radii, dtype=float)


def _receptor_atoms(path: Path) -> AtomArray:
    from biotite.structure.io import pdbx
    from biotite.structure.io.pdb import PDBFile

    compressed = path.suffix.lower() in {".gz", ".gzip", ".zst"}
    suffix = (path.with_suffix("") if compressed else path).suffix.lower()
    if suffix not in {".pdb", ".ent", ".cif", ".mmcif", ".pdbx"}:
        raise PoseBustersResultError("protein_format_unsupported")
    try:
        if compressed:
            from atomworks.io.utils.compression import open_compressed

            handle = open_compressed(path, "rt", encoding="utf-8")
        else:
            handle = path.open(encoding="utf-8")
        with handle as stream:
            if suffix in {".pdb", ".ent"}:
                atoms = PDBFile.read(stream).get_structure(model=1)
            else:
                atoms = pdbx.get_structure(pdbx.CIFFile.read(stream), model=1)
    except Exception as error:
        raise PoseBustersResultError(f"protein_unreadable:{type(error).__name__}") from error
    if not np.isfinite(atoms.coord).all():
        raise PoseBustersResultError("protein_coordinates_nonfinite")
    return atoms


def _protein_heavy_mask(atoms: AtomArray) -> np.ndarray:
    from biotite.structure import filter_amino_acids

    return filter_amino_acids(atoms) & ~np.isin(np.char.upper(atoms.element), ("H", "D", "T"))


def _posebusters_receptor(path: Path) -> Mol:
    from biotite.interface.rdkit import to_mol
    from biotite.structure import BondList

    atoms = _receptor_atoms(path)
    if not _protein_heavy_mask(atoms).any():
        raise PoseBustersResultError("protein_empty_after_filtering")
    # The dock-mode receptor tests use atom elements, residue annotations and
    # coordinates. Transfer these without inferring a receptor bond graph or
    # rounding mmCIF coordinates through a fixed-width PDB file.
    if atoms.bonds is None:
        atoms.bonds = BondList(len(atoms))
    try:
        return to_mol(atoms)
    except Exception as error:
        raise PoseBustersResultError(f"protein_conversion_failed:{type(error).__name__}") from error


def protein_clashes(mol: Mol, protein_path: str | Path, tolerance_a: float = CLASH_TOLERANCE_A) -> tuple[
    int | None, float | None, str | None
]:
    """Heavy-atom clashes of the ligand against the protein, plus the closest contact.

    A clash is a pair closer than ``vdw_i + vdw_j - tolerance_a``. Returns
    ``(count, min_distance, None)`` or ``(None, None, reason)``.
    """
    from scipy.spatial import cKDTree

    if not np.isfinite(tolerance_a) or tolerance_a < 0:
        raise ValueError("Clash tolerance must be finite and nonnegative.")
    try:
        atoms = _receptor_atoms(Path(protein_path))
    except PoseBustersResultError as error:
        return None, None, str(error)
    atoms = atoms[_protein_heavy_mask(atoms)]
    if atoms.array_length() == 0:
        return None, None, "protein_empty_after_filtering"
    protein_coords = np.asarray(atoms.coord, dtype=float)
    protein_radii = np.array(
        [BONDI_VDW_A.get(str(element).upper(), DEFAULT_VDW_A) for element in atoms.element],
        dtype=float,
    )
    ligand_coords, ligand_radii = _heavy_coords_and_radii(mol)
    if ligand_coords.size == 0:
        return None, None, "ligand_has_no_heavy_atoms"
    if not np.isfinite(ligand_coords).all():
        return None, None, "ligand_coordinates_nonfinite"
    tree = cKDTree(protein_coords)
    nearest, _ = tree.query(ligand_coords)
    largest_radius = float(protein_radii.max())
    clash_count = 0
    for point, radius in zip(ligand_coords, ligand_radii, strict=True):
        neighbors = tree.query_ball_point(point, max(0.0, radius + largest_radius - tolerance_a))
        if neighbors:
            distances = np.linalg.norm(point - protein_coords[neighbors], axis=1)
            clash_count += int(np.sum(distances < radius + protein_radii[neighbors] - tolerance_a))
    return clash_count, float(nearest.min()), None


def validate_pose(
    ligand: str | Path | Mol,
    protein_path: str | Path | None = None,
    require_posebusters: bool = True,
) -> PoseValidation:
    """Validate one ligand pose, optionally against a protein.

    ``require_posebusters=True`` raises if PoseBusters is absent. Set it False
    to record its absence as ``posebusters_mode="unavailable"`` and still return
    the native strain and clash readings.
    """
    mol = read_ligand(ligand)
    mode = "dock" if protein_path is not None else "mol"
    checks: dict[str, bool | None] = {}
    reason = None
    try:
        mode, checks = run_posebusters(mol, protein_path)
    except PoseBustersUnavailableError:
        if require_posebusters:
            raise
        mode, reason = "unavailable", "posebusters_unavailable"
    except PoseBustersResultError as error:
        reason = str(error)
    except TimeoutError:
        reason = "posebusters_timed_out"
    failures, reason = _failure_count(mode, checks, reason)
    strain, strain_reason = internal_strain(mol)
    clashes: int | None = None
    min_contact: float | None = None
    clash_reason: str | None = "no_protein_supplied"
    if protein_path is not None:
        clashes, min_contact, clash_reason = protein_clashes(mol, protein_path)
    source = ligand if isinstance(ligand, str | Path) else "<rdkit.Mol>"
    return PoseValidation(
        source=str(source),
        posebusters_mode=mode,
        posebusters_checks=checks,
        posebusters_failures=failures,
        internal_strain_kcal_mol=strain,
        internal_strain_reason=strain_reason,
        protein_clashes=clashes,
        min_protein_contact_a=min_contact,
        clash_reason=clash_reason,
        posebusters_reason=reason,
        notes=(
            "any failed PoseBusters check is this module's definition of a hit",
            "MMFF94 uses the supplied atoms and hydrogens. Strain requires converged local minimization.",
            "protein clashes use Bondi 1964 radii and a suggested tolerance, not a Rosetta clash score",
        ),
    )


def verdicts(validated: PoseValidation) -> dict[str, dict[str, object]]:
    """Each threshold applied, with its source carried along. ``None`` values stay untested."""
    record = asdict(validated)
    record["posebusters_failures"], reason = _failure_count(
        validated.posebusters_mode, validated.posebusters_checks, validated.posebusters_reason,
    )
    out: dict[str, dict[str, object]] = {}
    for name, rule in REFERENCE_THRESHOLDS.items():
        value = record.get(name)
        maximum = rule.get("max")
        if value is None or not np.isfinite(value):
            state = "untested"
        elif maximum is None:
            state = "no_cut"
        else:
            state = "clear" if value <= maximum else "hit"
        out[name] = {
            "value": value,
            "state": state,
            "threshold": maximum,
            "direction": "max" if maximum is not None else "none",
            "threshold_source": rule["source"],
        }
        if name == "posebusters_failures" and reason:
            out[name]["reason"] = reason
    return out
