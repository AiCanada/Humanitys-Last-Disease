"""Ligand-pose validation for a finished generation, by PoseBusters and RDKit.

A docked or co-folded ligand can clear an affinity panel and still sit in a
pose no chemist would accept: bonds the wrong length, a ring bent flat, atoms
driven into each other or into the protein. Those are physical-validity
questions, not affinity questions, and the safety panel in
:mod:`rfproteina.disease.toxicology` never asks them. This module asks them.

Three readings, each from a named tool and each labelled for what it is:

* **PoseBusters** (Buttenschoen, Morris and Deane, *Chem. Sci.* 2024;
  ``posebusters`` is pinned in ``pyproject.toml`` optional extras). The ``mol``
  configuration tests the ligand alone -- loadable, sanitisable, connected,
  no radicals, bond lengths and angles, internal steric clash, ring flatness,
  double-bond flatness and internal energy. The ``dock`` configuration adds the
  protein and tests the minimum distance to it and the volume overlap with it.
  Each check passes or fails by PoseBusters' own definition; this module's only
  convention is that *any* failed check is reported as a hit.

* **Internal strain**, from RDKit's MMFF94 force field: the pose's energy minus
  the energy of a relaxed copy of the same connectivity, in kcal/mol. PoseCheck
  (Harris et al., 2023) reports this distribution but publishes no pass mark, so
  it is reported here and **not gated** -- the field carries ``no_cut``.

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

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rdkit.Chem import Mol

METHOD = "posebusters_rdkit_v1"

#: The PoseBusters ``mol`` checks, in report order. Read off PoseBusters 0.x
#: output on 2026-11-16; if the tool renames a column this list is the contract
#: that will surface the change rather than silently drop a check.
POSEBUSTERS_MOL_CHECKS = (
    "mol_pred_loaded",
    "sanitization",
    "inchi_convertible",
    "all_atoms_connected",
    "no_radicals",
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "aromatic_ring_flatness",
    "non-aromatic_ring_non-flatness",
    "double_bond_flatness",
    "internal_energy",
)

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
            "program convention: any failed PoseBusters check is a hit. PoseBusters itself "
            "(Buttenschoen et al., Chem. Sci. 2024) defines each check; it publishes no overall pass count."
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
            "No cut adopted. PoseCheck (Harris et al. 2023) reports MMFF strain but publishes no "
            "pass mark, so this is reported, not gated."
        ),
    },
}


@dataclass(frozen=True)
class PoseValidation:
    """One ligand pose, validated. Energies are kcal/mol; distances are angstrom."""

    source: str
    posebusters_mode: str
    posebusters_checks: dict[str, bool | None]
    posebusters_failures: int
    internal_strain_kcal_mol: float | None
    internal_strain_reason: str | None
    protein_clashes: int | None
    min_protein_contact_a: float | None
    clash_reason: str | None
    method: str = METHOD
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["posebusters_checks"] = dict(self.posebusters_checks)
        record["notes"] = list(self.notes)
        record["reference_thresholds"] = REFERENCE_THRESHOLDS
        return record


class PoseUnreadableError(Exception):
    """The ligand could not be read, or the structure does not parse. Never guessed around."""


class PoseBustersUnavailableError(Exception):
    """PoseBusters is not installed. The screen refuses rather than approximating it."""


def _read_ligand(source: str | Path | Mol) -> Mol:
    """An RDKit mol with 3D coordinates, from an SDF/MOL/PDB path or an existing mol."""
    try:
        from rdkit import Chem
    except ImportError as error:  # pragma: no cover - rdkit is pinned in pyproject
        raise PoseUnreadableError(f"rdkit_unavailable:{error}") from error
    if not isinstance(source, str | Path):
        mol = source
        if mol is None or mol.GetNumConformers() == 0:
            raise PoseUnreadableError("mol_without_conformer")
        return mol
    path = Path(source)
    if not path.exists():
        raise PoseUnreadableError(f"ligand_absent:{path.name}")
    suffix = path.suffix.lower()
    if suffix in {".sdf", ".mol"}:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
        mol = next((item for item in supplier if item is not None), None)
    elif suffix in {".pdb", ".ent"}:
        mol = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=True)
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=True)
    else:
        raise PoseUnreadableError(f"unsupported_ligand_format:{suffix or 'none'}")
    if mol is None:
        raise PoseUnreadableError("ligand_unparsable")
    if mol.GetNumConformers() == 0:
        raise PoseUnreadableError("ligand_without_conformer")
    return mol


def run_posebusters(mol: Mol, protein_path: str | Path | None = None) -> tuple[str, dict[str, bool | None]]:
    """PoseBusters ``mol`` (ligand only) or ``dock`` (with a protein) checks.

    Returns the mode run and a dict of check name to pass/fail. Raises
    :class:`PoseBustersUnavailableError` if PoseBusters is not installed; the
    caller decides whether that is fatal, so the absence is never hidden behind
    an invented all-clear.
    """
    try:
        from posebusters import PoseBusters
    except ImportError as error:
        raise PoseBustersUnavailableError(f"posebusters_unavailable:{error}") from error
    mode = "dock" if protein_path is not None else "mol"
    buster = PoseBusters(config=mode)
    protein = str(protein_path) if protein_path is not None else None
    frame = buster.bust([mol], None, protein)
    row = frame.to_dict(orient="records")[0] if len(frame) else {}
    checks: dict[str, bool | None] = {}
    for name, value in row.items():
        if value is None:
            checks[str(name)] = None
        else:
            try:
                checks[str(name)] = bool(value)
            except (TypeError, ValueError):
                checks[str(name)] = None
    return mode, checks


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
        relaxed = Chem.Mol(mol)
        relaxed_props = AllChem.MMFFGetMoleculeProperties(relaxed)
        relaxed_field = AllChem.MMFFGetMoleculeForceField(relaxed, relaxed_props)
        relaxed_field.Minimize(maxIts=1000)
        relaxed_energy = float(relaxed_field.CalcEnergy())
    except Exception as error:  # any force-field failure is the same answer
        return None, f"mmff_failed:{type(error).__name__}"
    return round(pose_energy - relaxed_energy, 3), None


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


def protein_clashes(mol: Mol, protein_path: str | Path, tolerance_a: float = CLASH_TOLERANCE_A) -> tuple[
    int | None, float | None, str | None
]:
    """Heavy-atom clashes of the ligand against the protein, plus the closest contact.

    A clash is a pair closer than ``vdw_i + vdw_j - tolerance_a``. Returns
    ``(count, min_distance, None)`` or ``(None, None, reason)``.
    """
    try:
        from biotite.structure.io import load_structure
    except ImportError as error:  # pragma: no cover - biotite is a hard dependency
        return None, None, f"biotite_unavailable:{type(error).__name__}"
    try:
        atoms = load_structure(str(protein_path), model=1)
    except Exception as error:
        return None, None, f"protein_unreadable:{type(error).__name__}"
    keep = (atoms.element != "H") & (atoms.res_name != "HOH")
    atoms = atoms[keep]
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
    distances = np.linalg.norm(ligand_coords[:, None, :] - protein_coords[None, :, :], axis=-1)
    allowed = ligand_radii[:, None] + protein_radii[None, :] - float(tolerance_a)
    clash_count = int(np.sum(distances < allowed))
    return clash_count, round(float(distances.min()), 3), None


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
    mol = _read_ligand(ligand)
    try:
        mode, checks = run_posebusters(mol, protein_path)
    except PoseBustersUnavailableError:
        if require_posebusters:
            raise
        mode, checks = "unavailable", {}
    failures = sum(1 for value in checks.values() if value is False)
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
        notes=(
            "any failed PoseBusters check is this module's definition of a hit",
            "internal strain is reported, not gated: PoseCheck publishes no pass mark",
            "protein clashes use Bondi 1964 radii and a suggested tolerance, not a Rosetta clash score",
        ),
    )


def verdicts(validated: PoseValidation) -> dict[str, dict[str, object]]:
    """Each threshold applied, with its source carried along. ``None`` values stay untested."""
    record = asdict(validated)
    out: dict[str, dict[str, object]] = {}
    for name, rule in REFERENCE_THRESHOLDS.items():
        value = record.get(name)
        maximum = rule.get("max")
        if value is None:
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
    return out
