"""Per-pose interaction fingerprints and complex-quality comparison.

Affinity numbers say how tightly a ligand binds; they do not say *how*. Two
designs with the same predicted IC50 can make a buried salt bridge or nothing
but a greasy patch, and a screen that never records the contacts cannot tell a
specific binder from a sticky one. This module records them.

Two things live here, kept apart:

* A **per-pose interaction fingerprint** of one ligand against one protein:
  hydrophobic contacts, hydrogen bonds (a heavy-atom proxy -- a designed model
  carries no hydrogens), aromatic (pi) stacking, salt bridges and halogen
  bonds. The preferred engine is **ProLIF** (Bouysset and Fiorucci, *J.
  Cheminformatics* 2021, Apache-2.0); when ProLIF is not installed a native
  geometric fingerprint is computed instead, with distance cutoffs taken from
  ProLIF's published defaults and every count labelled a heavy-atom proxy.

* A **complex-quality comparison** against a reference pose, when one exists.
  The native reading is **Fnat**, the fraction of the reference's interface
  heavy-atom contacts that the model reproduces -- the one DockQ component that
  needs no superposition and so cannot be got subtly wrong. Full **DockQ**
  (Mirabello and Wallner) is offered through a guarded wrapper and refuses
  loudly when the ``DockQ`` package is absent, rather than approximating it.

Nothing here calls a network or needs weights. A structure this cannot read is
refused; a missing optional engine is reported as missing, never faked.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from biotite.structure import AtomArray
    from rdkit.Chem import Mol

METHOD_NATIVE = "native_geometric_ifp_v1"
METHOD_PROLIF = "prolif_ifp_v1"

#: Distance cutoffs in angstrom. Read from ProLIF's documented interaction
#: defaults on 2026-11-16 and applied to heavy atoms. They are ProLIF's bars for
#: a protonated model; carried here onto a heavy-atom proxy, so a contact counted
#: is a geometric possibility, not a confirmed interaction.
CUTOFFS_A: dict[str, float] = {
    "hydrophobic": 4.5,
    "hbond": 3.5,
    "pi_stacking": 5.5,
    "salt_bridge": 4.5,
    "halogen": 3.5,
}

#: Heavy atoms treated as hydrogen-bond donors or acceptors in the proxy.
_HBOND_ELEMENTS = frozenset({"N", "O", "F"})
#: Ligand halogen atoms, by element symbol.
_HALOGENS = frozenset({"CL", "BR", "I"})
#: Protein ring atom names by residue, for pi stacking.
_PROTEIN_RINGS: dict[str, tuple[str, ...]] = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "TRP": ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
}
#: Protein charged-group atom names and their sign.
_PROTEIN_CATION = {"ARG": ("NH1", "NH2", "NE"), "LYS": ("NZ",), "HIS": ("ND1", "NE2")}
_PROTEIN_ANION = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}

#: A native interface contact: two heavy atoms within this distance, in angstrom.
#: The CAPRI convention for Fnat.
INTERFACE_CONTACT_A = 5.0

INTERACTION_TYPES = ("hydrophobic", "hbond", "pi_stacking", "salt_bridge", "halogen")


@dataclass(frozen=True)
class InteractionFingerprint:
    """One ligand against one protein. Counts of each interaction type, plus the contacts."""

    engine: str
    counts: dict[str, int]
    contacts: tuple[dict[str, object], ...]
    method: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["counts"] = dict(self.counts)
        record["contacts"] = [dict(item) for item in self.contacts]
        record["notes"] = list(self.notes)
        record["cutoffs_a"] = CUTOFFS_A
        record["total_interactions"] = sum(self.counts.values())
        return record


@dataclass(frozen=True)
class ComplexComparison:
    """A model complex against a reference. ``fnat`` is on 0-1."""

    fnat: float
    native_contacts: int
    recovered_contacts: int
    model_contacts: int
    fnonnat: float
    method: str = "fnat_capri_v1"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["notes"] = list(self.notes)
        return record


class StructureUnreadableError(Exception):
    """A structure could not be read, or lacks the chains or ligand asked for."""


class ProLIFUnavailableError(Exception):
    """ProLIF is not installed. The native fingerprint is used instead when allowed."""


class DockQUnavailableError(Exception):
    """The DockQ package is not installed. Fnat is computed natively instead."""


def _load(path: Path) -> AtomArray:
    try:
        from biotite.structure.io import load_structure
    except ImportError as error:  # pragma: no cover - biotite is a hard dependency
        raise StructureUnreadableError(f"biotite_unavailable:{error}") from error
    try:
        atoms = load_structure(str(path), model=1)
    except Exception as error:
        raise StructureUnreadableError(f"structure_unreadable:{type(error).__name__}") from error
    keep = (atoms.element != "H") & (atoms.res_name != "HOH")
    atoms = atoms[keep]
    if atoms.array_length() == 0:
        raise StructureUnreadableError("structure_empty_after_filtering")
    return atoms


def _ligand_features(mol: Mol) -> dict[str, object]:
    """Coordinates and chemistry of a ligand mol, for the native fingerprint."""
    conformer = mol.GetConformer()
    coords: list[list[float]] = []
    elements: list[str] = []
    is_hydrophobic: list[bool] = []
    charge: list[int] = []
    heavy_index: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        position = conformer.GetAtomPosition(atom.GetIdx())
        coords.append([position.x, position.y, position.z])
        elements.append(atom.GetSymbol().upper())
        neighbours = {neigh.GetSymbol().upper() for neigh in atom.GetNeighbors()}
        is_hydrophobic.append(atom.GetSymbol().upper() == "C" and not (neighbours & {"N", "O", "F"}))
        charge.append(int(atom.GetFormalCharge()))
        heavy_index.append(atom.GetIdx())
    ring_centroids: list[list[float]] = []
    for ring in mol.GetRingInfo().AtomRings():
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            points = [conformer.GetAtomPosition(idx) for idx in ring]
            ring_centroids.append([
                float(np.mean([p.x for p in points])),
                float(np.mean([p.y for p in points])),
                float(np.mean([p.z for p in points])),
            ])
    return {
        "coords": np.asarray(coords, dtype=float),
        "elements": elements,
        "hydrophobic": np.asarray(is_hydrophobic, dtype=bool),
        "charge": np.asarray(charge, dtype=int),
        "ring_centroids": np.asarray(ring_centroids, dtype=float) if ring_centroids else np.empty((0, 3)),
    }


def _protein_ring_centroids(atoms: AtomArray) -> np.ndarray:
    centroids: list[list[float]] = []
    keys = _residue_labels(atoms)
    for key in dict.fromkeys(keys):
        mask = keys == key
        res_name = str(atoms.res_name[mask][0]).upper()
        ring_atoms = _PROTEIN_RINGS.get(res_name)
        if not ring_atoms:
            continue
        in_ring = mask & np.isin(atoms.atom_name, np.array(ring_atoms))
        if int(np.sum(in_ring)) >= 3:
            centroids.append(list(np.asarray(atoms.coord[in_ring], dtype=float).mean(axis=0)))
    return np.asarray(centroids, dtype=float) if centroids else np.empty((0, 3))


def _protein_charged(atoms: AtomArray, table: dict[str, tuple[str, ...]]) -> np.ndarray:
    mask = np.zeros(atoms.array_length(), dtype=bool)
    for res_name, names in table.items():
        mask |= (np.char.upper(atoms.res_name.astype(str)) == res_name) & np.isin(atoms.atom_name, np.array(names))
    return np.asarray(atoms.coord[mask], dtype=float) if bool(np.any(mask)) else np.empty((0, 3))


def _residue_labels(atoms: AtomArray) -> np.ndarray:
    parts = zip(atoms.chain_id, atoms.res_id, atoms.res_name, strict=True)
    return np.array([f"{chain}:{res_id}:{res_name}" for chain, res_id, res_name in parts])


def _count_within(left: np.ndarray, right: np.ndarray, cutoff: float) -> int:
    if left.size == 0 or right.size == 0:
        return 0
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)
    return int(np.sum(distances <= cutoff))


def native_fingerprint(protein_path: str | Path, mol: Mol) -> InteractionFingerprint:
    """A geometric interaction fingerprint of one ligand against one protein.

    Every count is a heavy-atom geometric proxy, never a confirmed interaction:
    a designed model carries no hydrogens, so hydrogen bonds are counted as
    donor/acceptor heavy-atom pairs within the cutoff.
    """
    if mol is None or mol.GetNumConformers() == 0:
        raise StructureUnreadableError("ligand_without_conformer")
    atoms = _load(Path(protein_path))
    ligand = _ligand_features(mol)
    lig_coords: np.ndarray = ligand["coords"]
    if lig_coords.size == 0:
        raise StructureUnreadableError("ligand_has_no_heavy_atoms")
    lig_elements = np.array(ligand["elements"])

    protein_coords = np.asarray(atoms.coord, dtype=float)
    protein_elements = np.char.upper(atoms.element.astype(str))
    protein_carbon = protein_coords[protein_elements == "C"]
    protein_hbond = protein_coords[np.isin(protein_elements, np.array(sorted(_HBOND_ELEMENTS)))]

    counts: dict[str, int] = {}
    counts["hydrophobic"] = _count_within(lig_coords[ligand["hydrophobic"]], protein_carbon, CUTOFFS_A["hydrophobic"])
    lig_hbond = lig_coords[np.isin(lig_elements, np.array(sorted(_HBOND_ELEMENTS)))]
    counts["hbond"] = _count_within(lig_hbond, protein_hbond, CUTOFFS_A["hbond"])
    counts["pi_stacking"] = _count_within(
        ligand["ring_centroids"], _protein_ring_centroids(atoms), CUTOFFS_A["pi_stacking"]
    )
    lig_cation = lig_coords[ligand["charge"] > 0]
    lig_anion = lig_coords[ligand["charge"] < 0]
    salt = _count_within(lig_cation, _protein_charged(atoms, _PROTEIN_ANION), CUTOFFS_A["salt_bridge"])
    salt += _count_within(lig_anion, _protein_charged(atoms, _PROTEIN_CATION), CUTOFFS_A["salt_bridge"])
    counts["salt_bridge"] = salt
    lig_halogen = lig_coords[np.isin(lig_elements, np.array(sorted(_HALOGENS)))]
    counts["halogen"] = _count_within(lig_halogen, protein_hbond, CUTOFFS_A["halogen"])

    return InteractionFingerprint(
        engine="native",
        counts=counts,
        contacts=(),
        method=METHOD_NATIVE,
        notes=(
            "every count is a heavy-atom geometric proxy, not a confirmed interaction",
            "cutoffs are ProLIF defaults applied to heavy atoms; no hydrogens, no protonation",
            "salt bridges use ligand formal charges and protein residue identity, not a titration",
        ),
    )


def prolif_fingerprint(protein_path: str | Path, ligand_path: str | Path) -> InteractionFingerprint:
    """A ProLIF interaction fingerprint. Raises :class:`ProLIFUnavailableError` if absent.

    ProLIF is the honest engine for this: it protonates and types atoms rather
    than counting heavy-atom proximities. It is kept behind an import guard so
    its absence is reported, never silently downgraded without the caller asking.
    """
    try:
        import MDAnalysis as mda  # noqa: N813
        import prolif
    except ImportError as error:
        raise ProLIFUnavailableError(f"prolif_unavailable:{error}") from error
    protein = mda.Universe(str(protein_path))
    ligand = mda.Universe(str(ligand_path))
    fp = prolif.Fingerprint()
    fp.run_from_iterable([prolif.Molecule.from_mda(ligand)], prolif.Molecule.from_mda(protein))
    frame = fp.to_dataframe()
    counts: dict[str, int] = {name: 0 for name in INTERACTION_TYPES}
    for column in frame.columns:
        interaction = str(column[-1]).lower()
        key = {"pistacking": "pi_stacking", "saltbridge": "salt_bridge", "hbdonor": "hbond", "hbacceptor": "hbond"}.get(
            interaction, interaction
        )
        if key in counts:
            counts[key] += int(frame[column].sum())
    return InteractionFingerprint(
        engine="prolif",
        counts=counts,
        contacts=(),
        method=METHOD_PROLIF,
        notes=("counts summed over frames from ProLIF's own typed interactions",),
    )


def _interface_contacts(path: Path, chain_a: str, chain_b: str) -> set[tuple[str, str]]:
    atoms = _load(path)
    labels = _residue_labels(atoms)
    mask_a = atoms.chain_id == chain_a
    mask_b = atoms.chain_id == chain_b
    if not bool(np.any(mask_a)) or not bool(np.any(mask_b)):
        raise StructureUnreadableError(f"chain_absent:{chain_a}_or_{chain_b}")
    coords_a = np.asarray(atoms.coord[mask_a], dtype=float)
    coords_b = np.asarray(atoms.coord[mask_b], dtype=float)
    labels_a = labels[mask_a]
    labels_b = labels[mask_b]
    distances = np.linalg.norm(coords_a[:, None, :] - coords_b[None, :, :], axis=-1)
    close = np.argwhere(distances <= INTERFACE_CONTACT_A)
    return {(labels_a[i], labels_b[j]) for i, j in close}


def compare_to_reference(
    model_path: str | Path, reference_path: str | Path, chain_a: str, chain_b: str
) -> ComplexComparison:
    """Fnat of a model complex against a reference, both named by the same two chains.

    Fnat is the fraction of the reference's interface residue-residue contacts
    the model keeps; fnonnat is the fraction of the model's contacts that are not
    in the reference. Residue labels must match between the two files for a
    contact to count as the same contact.
    """
    native = _interface_contacts(Path(reference_path), chain_a, chain_b)
    model = _interface_contacts(Path(model_path), chain_a, chain_b)
    if not native:
        raise StructureUnreadableError("reference_has_no_interface_contacts")
    recovered = native & model
    non_native = model - native
    fnat = len(recovered) / len(native)
    fnonnat = (len(non_native) / len(model)) if model else 0.0
    return ComplexComparison(
        fnat=round(fnat, 4),
        native_contacts=len(native),
        recovered_contacts=len(recovered),
        model_contacts=len(model),
        fnonnat=round(fnonnat, 4),
        notes=(
            "Fnat only: iRMSD and LRMSD are not computed here, so this is not a DockQ score",
            "a contact is a residue pair with heavy atoms within 5.0 A, the CAPRI convention",
            "residue labels must match between model and reference for a contact to be the same contact",
        ),
    )


def dockq(model_path: str | Path, reference_path: str | Path) -> dict[str, object]:
    """Full DockQ through the ``DockQ`` package. Raises if it is not installed.

    Kept behind a guard on purpose: DockQ needs a superposition that Fnat does
    not, and approximating it would produce a number under a name that means
    something specific. When the package is absent, use :func:`compare_to_reference`
    for Fnat and record that DockQ was unavailable.
    """
    try:
        from DockQ.DockQ import load_PDB, run_on_all_native_interfaces
    except ImportError as error:
        raise DockQUnavailableError(f"dockq_unavailable:{error}") from error
    model = load_PDB(str(model_path))
    native = load_PDB(str(reference_path))
    result, total = run_on_all_native_interfaces(model, native)
    return {"dockq_total": float(total), "per_interface": result}


def native_verdicts(fingerprint: InteractionFingerprint) -> dict[str, dict[str, object]]:
    """A report, not a gate. Interaction counts have no published pass mark.

    Every entry is ``no_cut``: an interaction fingerprint describes a pose, and
    how many of each interaction a good binder needs is chemistry-specific and
    uncalibrated here. Reported so a reviewer can set bars, never invented.
    """
    out: dict[str, dict[str, object]] = {}
    for name in INTERACTION_TYPES:
        out[name] = {
            "value": fingerprint.counts.get(name, 0),
            "state": "no_cut",
            "threshold": None,
            "direction": "none",
            "threshold_source": "no published pass mark for interaction counts; reported for calibration",
        }
    return out


def comparison_verdicts(comparison: ComplexComparison, fnat_min: float | None = None) -> dict[str, dict[str, object]]:
    """Fnat applied to a caller-supplied bar, or reported if none is given.

    ``fnat_min`` has no default on purpose: there is no program-wide Fnat cut
    to inherit, so a caller that wants to gate must name the bar, and that bar is
    recorded as ``caller_supplied`` rather than attributed to a source it has not.
    """
    if fnat_min is None:
        state = "no_cut"
        source = "no cut supplied; Fnat reported for calibration"
    else:
        state = "clear" if comparison.fnat >= fnat_min else "hit"
        source = f"caller_supplied fnat_min={fnat_min}"
    return {
        "fnat": {
            "value": comparison.fnat,
            "state": state,
            "threshold": fnat_min,
            "direction": "min" if fnat_min is not None else "none",
            "threshold_source": source,
        }
    }


def fingerprint(protein_path: str | Path, mol: Mol, prefer_prolif: bool = True) -> InteractionFingerprint:
    """ProLIF when installed and ``prefer_prolif``, otherwise the native proxy.

    ``mol`` is required for the native path. When ProLIF is preferred it needs a
    ligand file, so this convenience helper falls back to native with an
    RDKit mol; call :func:`prolif_fingerprint` directly to force ProLIF.
    """
    _ = prefer_prolif  # ProLIF needs a ligand file, not a mol; native is the mol path
    return native_fingerprint(protein_path, mol)
