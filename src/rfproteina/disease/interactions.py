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
  the project configuration and every count labelled a heavy-atom proxy.

* A **complex-quality comparison** against a reference pose, when one exists.
  The native reading is **Fnat**, the fraction of the reference's interface
  heavy-atom contacts that the model reproduces -- the one DockQ component that
  needs no superposition and so cannot be got subtly wrong. Full **DockQ**
  (Mirabello and Wallner, MIT) is offered through a guarded wrapper and refuses
  loudly when no DockQ executable is found, rather than approximating it.

  DockQ runs as a **subprocess against its own virtual environment**, the way
  ``mhc_tables`` runs mhcflurry, and for a hard reason: every released DockQ
  pins ``numpy<2.0`` while this project runs numpy 2.x, so installing it here
  would downgrade numpy underneath torch. Going through its ``--json``
  interface also buys the automatic chain mapping v2 added, which calling
  ``run_on_all_native_interfaces`` in-process does not do.

Nothing here calls a network or needs weights. A structure this cannot read is
refused; a missing optional engine is reported as missing, never faked.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from biotite.structure import AtomArray
    from rdkit.Chem import Mol

METHOD_NATIVE = "native_geometric_ifp_v2"
METHOD_PROLIF = "prolif_ifp_v2"
_PROLIF_TYPES = {
    "Hydrophobic": "hydrophobic", "PiStacking": "pi_stacking",
    "HBDonor": "hbond", "HBAcceptor": "hbond",
    "Anionic": "salt_bridge", "Cationic": "salt_bridge",
    "XBAcceptor": "halogen", "XBDonor": "halogen",
}
METHOD_DOCKQ = "dockq_v2_cli"

DOCKQ_SOURCE = "DockQ v2 (MIT), https://github.com/bjornwallner/DockQ"
DOCKQ_PAPER = (
    "Mirabello and Wallner, 'DockQ v2: improved automatic quality measure for protein multimers, "
    "nucleic acids, and small molecules', Bioinformatics 40(10) (2024), doi:10.1093/bioinformatics/btae586"
)
#: Names the ``DockQ`` executable. DockQ pins ``numpy<2.0`` and this project
#: runs numpy 2.x, so it is installed into its own virtual environment and run
#: as a subprocess -- the same isolation ``mhc_tables`` uses for mhcflurry.
#: Installing it into the project environment would downgrade numpy under torch.
DOCKQ_COMMAND_ENV = "DOCKQ_EXECUTABLE"
#: Where ``scripts/install_dockq.py`` puts that environment, under ``DATA_PATH``.
DOCKQ_SUBDIR = Path("dockq")
DOCKQ_REQUIREMENT = "DockQ>=2.1.3,<3"
#: The chain-mapping search is combinatorial in the number of chains, so a run
#: is bounded rather than left to hang on a large multimer.
DOCKQ_TIMEOUT_S = 900

#: The authors' own quality bands, as the CLI banner prints them: a published
#: classification, not a bar chosen here. Highest floor first.
DOCKQ_BANDS: tuple[tuple[float, str], ...] = (
    (0.80, "high"),
    (0.49, "medium"),
    (0.23, "acceptable"),
    (0.0, "incorrect"),
)
#: Keys DockQ's ``--json`` output must carry. Checked so a release that renames
#: one is refused loudly instead of reported as a missing score.
_DOCKQ_KEYS = ("best_dockq", "GlobalDockQ", "best_result", "best_mapping_str")

#: Project distance cutoffs in angstrom for geometric proximity counts.
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
        if self.engine == "native":
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
    fnonnat: float | None
    method: str = "fnat_capri_v2"
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
    """No DockQ executable could be found. Fnat is computed natively instead."""


class DockQFailedError(Exception):
    """A DockQ executable was found and run, and did not produce a score.

    Separate from :class:`DockQUnavailableError` because the two call for
    different actions: one means install it, the other means the inputs are
    not a pair DockQ can compare.
    """


def _load(path: Path) -> AtomArray:
    from rfproteina.disease.pose_validation import PoseBustersResultError, _receptor_atoms

    try:
        atoms = _receptor_atoms(path)
    except PoseBustersResultError as error:
        raise StructureUnreadableError(str(error)) from error
    keep = ~np.isin(atoms.element, ("H", "D", "T")) & ~np.isin(atoms.res_name, ("HOH", "DOD", "WAT"))
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
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        position = conformer.GetAtomPosition(atom.GetIdx())
        coords.append([position.x, position.y, position.z])
        elements.append(atom.GetSymbol().upper())
        neighbours = {neigh.GetSymbol().upper() for neigh in atom.GetNeighbors()}
        is_hydrophobic.append(atom.GetSymbol().upper() == "C" and not (neighbours & {"N", "O", "F"}))
        charge.append(int(atom.GetFormalCharge()))
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
    parts = zip(atoms.chain_id, atoms.res_id, atoms.ins_code, strict=True)
    return np.array([f"{chain}:{res_id}:{insertion}" for chain, res_id, insertion in parts])


def _count_within(left: np.ndarray, right: np.ndarray, cutoff: float) -> int:
    from scipy.spatial import cKDTree

    if left.size == 0 or right.size == 0:
        return 0
    return int(cKDTree(left).count_neighbors(cKDTree(right), cutoff))


def native_fingerprint(protein_path: str | Path, mol: Mol) -> InteractionFingerprint:
    """A geometric interaction fingerprint of one ligand against one protein.

    Every count is a heavy-atom geometric proxy, never a confirmed interaction:
    a designed model carries no hydrogens, so hydrogen bonds are counted as
    donor/acceptor heavy-atom pairs within the cutoff.
    """
    from rfproteina.disease.pose_validation import (
        PoseBustersResultError,
        PoseUnreadableError,
        _protein_heavy_mask,
        _receptor_atoms,
        read_ligand,
    )

    try:
        mol = read_ligand(mol)
        receptor = _receptor_atoms(Path(protein_path))
    except (PoseUnreadableError, PoseBustersResultError) as error:
        raise StructureUnreadableError(str(error)) from error
    atoms = receptor[_protein_heavy_mask(receptor)]
    if not len(atoms):
        raise StructureUnreadableError("protein_empty_after_filtering")
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
            "distance cutoffs define the project's heavy-atom proximity counts",
            "salt bridges use ligand formal charges and protein residue identity, not a titration",
        ),
    )


def prolif_fingerprint(protein_path: str | Path, ligand_path: str | Path | Mol) -> InteractionFingerprint:
    """Measure residue-pair interactions on one supplied pose and protonation state."""
    from rdkit import Chem
    try:
        import MDAnalysis as mda  # noqa: N813
        import prolif
    except ImportError as error:
        raise ProLIFUnavailableError(f"prolif_unavailable:{error}") from error
    def read_one(path: str | Path) -> prolif.Molecule:
        universe = mda.Universe(str(path))
        if len(universe.trajectory) != 1 or not np.isfinite(universe.atoms.positions).all():
            raise StructureUnreadableError("prolif_requires_one_finite_pose")
        return prolif.Molecule.from_mda(universe)

    protein = read_one(protein_path)
    if isinstance(ligand_path, Chem.Mol):
        from rfproteina.disease.pose_validation import read_ligand

        ligand = prolif.Molecule.from_rdkit(read_ligand(ligand_path))
    else:
        ligand = read_one(ligand_path)
    fp = prolif.Fingerprint(list(_PROLIF_TYPES))
    fp.run_from_iterable([ligand], protein, n_jobs=1, progress=False)
    frame = fp.to_dataframe()
    if len(frame) != 1:
        raise StructureUnreadableError("prolif_missing_pose_result")
    counts: dict[str, int] = {name: 0 for name in INTERACTION_TYPES}
    for column in frame.columns:
        flag = frame[column].iloc[0]
        if not isinstance(flag, bool | np.bool_):
            raise StructureUnreadableError("prolif_invalid_interaction_flag")
        counts[_PROLIF_TYPES[column[-1]]] += int(flag)
    return InteractionFingerprint(
        engine="prolif",
        counts=counts,
        contacts=(),
        method=METHOD_PROLIF,
        notes=("typed interaction flags summed across residue pairs for one supplied pose",),
    )


def _interface_contacts(path: Path, chain_a: str, chain_b: str) -> set[tuple[str, str]]:
    from scipy.spatial import cKDTree

    if not chain_a or not chain_b or chain_a == chain_b:
        raise StructureUnreadableError("two_distinct_chains_required")
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
    tree = cKDTree(coords_b)
    contacts = set()
    for label, point in zip(labels_a, coords_a, strict=True):
        contacts.update((label, labels_b[j]) for j in tree.query_ball_point(point, INTERFACE_CONTACT_A))
    return contacts


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
    fnonnat = (len(non_native) / len(model)) if model else None
    return ComplexComparison(
        fnat=fnat,
        native_contacts=len(native),
        recovered_contacts=len(recovered),
        model_contacts=len(model),
        fnonnat=fnonnat,
        notes=(
            "Fnat only: iRMSD and LRMSD are not computed here, so this is not a DockQ score",
            "a contact is a residue pair with heavy atoms within 5.0 A, the CAPRI convention",
            "residues correspond by chain, residue number and insertion code in the two structures",
        ),
    )


def dockq_install_root() -> Path | None:
    """``DATA_PATH/dockq``, or ``None`` where ``DATA_PATH`` is unset."""
    root = os.environ.get("DATA_PATH", "").strip()
    return Path(root) / DOCKQ_SUBDIR if root else None


def _executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def dockq_command() -> str | None:
    """The ``DockQ`` executable to run, or ``None`` when there is none.

    :data:`DOCKQ_COMMAND_ENV` wins, then the virtual environment under
    :func:`dockq_install_root`, then ``PATH``. A value that is not an
    executable file is ignored rather than handed to :mod:`subprocess`, so a
    stale variable reads as "not installed" instead of as a crash -- the rule
    :func:`rfproteina.disease.mhc_tables.predictor_command` already follows.
    """
    named = os.environ.get(DOCKQ_COMMAND_ENV, "").strip()
    if named:
        if _executable(Path(named)):
            return named
        found = shutil.which(named)
        if found:
            return found
    root = dockq_install_root()
    if root is not None:
        for relative in (Path("venv") / "bin" / "DockQ", Path("venv") / "Scripts" / "DockQ.exe"):
            candidate = root / relative
            if _executable(candidate):
                return str(candidate)
    return shutil.which("DockQ")


def dockq_available() -> tuple[bool, str | None]:
    """Whether a DockQ executable can be found, and where to get one if not."""
    command = dockq_command()
    if command is None:
        return False, "dockq_unavailable:no_executable_on_path_or_under_data_path"
    return True, None


def dockq_version(command: str | None = None) -> str | None:
    """DockQ's version, or ``None``.

    DockQ has no ``--version`` flag -- it exits 2 on one -- so the version is
    read from the metadata of the interpreter beside the executable. Best
    effort: a missing version is reported as unknown, never guessed.
    """
    command = command or dockq_command()
    if command is None:
        return None
    python = Path(command).parent / "python"
    if not _executable(python):
        python = Path(command).parent / "python.exe"
        if not _executable(python):
            return None
    try:
        done = subprocess.run(
            [str(python), "-c", "import importlib.metadata as m; print(m.version('DockQ'))"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = done.stdout.strip()
    return value or None


def dockq_quality(score: float) -> str:
    """The authors' own band for a DockQ score in ``[0, 1]``."""
    if not _valid_dockq_number(score, 1):
        raise ValueError("DockQ scores must be finite numbers in [0, 1]")
    for floor, name in DOCKQ_BANDS:
        if score >= floor:
            return name
    return "incorrect"


def _valid_dockq_number(value: object, maximum: int) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and 0 <= value <= maximum


def dockq(
    model_path: str | Path,
    reference_path: str | Path,
    *,
    mapping: str | None = None,
    no_align: bool = False,
    small_molecule: bool = False,
    n_cpu: int | None = None,
    allowed_mismatches: int | None = None,
    timeout: float = DOCKQ_TIMEOUT_S,
) -> dict[str, object]:
    """Full DockQ against a reference complex, through the DockQ executable.

    Kept behind a guard on purpose: DockQ needs a superposition that Fnat does
    not, and approximating it would produce a number under a name that means
    something specific. When no executable is found, use
    :func:`compare_to_reference` for Fnat and record that DockQ was
    unavailable.

    Run as a subprocess rather than imported. Two reasons, both measured:
    DockQ pins ``numpy<2.0`` and this project runs numpy 2.x, so importing it
    would mean downgrading numpy underneath torch; and the ``--json`` interface
    performs the automatic chain mapping that v2 added, which calling
    ``run_on_all_native_interfaces`` directly does not -- that path scores
    whatever chains happen to share an identifier.

    ``mapping`` is DockQ's own ``MODELCHAINS:NATIVECHAINS`` form, and
    ``':BC'`` limits the comparison to one native interface. ``no_align``
    matches residues by numbering instead of by sequence alignment, which is
    what a design and its own refold want.

    Returns a record whose ``global_dockq`` is the 0-to-1 figure the CLI prints
    as "Total DockQ". ``dockq_sum`` adds the interfaces up and can exceed
    one in a multimer. It measured 3.0 for a
    three-chain identity comparison -- so the two are named apart rather than
    both called a total.
    """
    command = dockq_command()
    if command is None:
        raise DockQUnavailableError("dockq_unavailable:no_executable_on_path_or_under_data_path")
    model = Path(model_path)
    native = Path(reference_path)
    for label, path in (("model", model), ("reference", native)):
        if not path.is_file():
            raise StructureUnreadableError(f"dockq_{label}_absent:{path.name}")

    with tempfile.TemporaryDirectory(prefix="rfd4_dockq_") as scratch:
        out = Path(scratch) / "dockq.json"
        argv = [command, str(model), str(native), "--json", str(out)]
        if mapping:
            argv += ["--mapping", mapping]
        if no_align:
            argv.append("--no_align")
        if small_molecule:
            argv.append("--small_molecule")
        if n_cpu is not None:
            argv += ["--n_cpu", str(int(n_cpu))]
        if allowed_mismatches is not None:
            argv += ["--allowed_mismatches", str(int(allowed_mismatches))]
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as error:
            raise DockQFailedError(f"dockq_timeout:{timeout:g}s") from error
        except OSError as error:
            raise DockQUnavailableError(f"dockq_unavailable:{error}") from error

        # The exit code is not enough. Given a single-chain input DockQ prints
        # "Need at least two chains in the two inputs", writes no JSON, and
        # still exits 0, so a wrapper that trusted the status would report a
        # silent success with no score. The file is what is checked.
        if not out.is_file():
            detail = (done.stdout or done.stderr or "").strip().splitlines()
            reason = detail[-1] if detail else f"exit_{done.returncode}_and_no_output"
            raise DockQFailedError(f"dockq_no_result:{reason}")
        if done.returncode:
            raise DockQFailedError(f"dockq_process_failed:exit_{done.returncode}")
        try:
            document = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise DockQFailedError(f"dockq_result_unreadable:{type(error).__name__}") from error

    if not isinstance(document, dict):
        raise DockQFailedError("dockq_result_not_an_object")
    missing = [key for key in _DOCKQ_KEYS if key not in document]
    if missing:
        raise DockQFailedError(f"dockq_result_keys_absent:{','.join(missing)}")
    interfaces = document["best_result"]
    if not isinstance(interfaces, dict) or not interfaces:
        raise DockQFailedError("dockq_no_interface_scored")

    global_score = document["GlobalDockQ"]
    score_sum = document["best_dockq"]
    if not _valid_dockq_number(global_score, 1) or not _valid_dockq_number(score_sum, len(interfaces)):
        raise DockQFailedError("dockq_score_outside_range")
    if not all(isinstance(values, dict) and values for values in interfaces.values()):
        raise DockQFailedError("dockq_interface_result_empty")
    return {
        "method": METHOD_DOCKQ,
        "source": DOCKQ_SOURCE,
        "paper": DOCKQ_PAPER,
        "command": command,
        "version": dockq_version(command) or "unknown",
        "model": str(model),
        "reference": str(native),
        "global_dockq": global_score,
        "quality": dockq_quality(global_score),
        "dockq_sum": score_sum,
        "interface_count": len(interfaces),
        # DockQ's own per-interface fields, passed through: these are the
        # authoritative numbers, and rewriting them would only add a way to
        # get one wrong. Keys are interface names such as "BC".
        "interfaces": interfaces,
        "mapping": document["best_mapping_str"],
        "chain_map": document.get("best_mapping", {}),
        "notes": [
            "global_dockq is the 0-to-1 score the CLI prints as Total DockQ",
            "dockq_sum adds the per-interface scores and can exceed 1 for a multimer",
            "bands are the authors' own: >=0.80 high, >=0.49 medium, >=0.23 acceptable, below that incorrect",
            "a DockQ score is always against a reference; a design with no native complex has none",
        ],
    }


def dockq_verdicts(record: dict[str, object], dockq_min: float | None = None) -> dict[str, dict[str, object]]:
    """DockQ applied to a caller-supplied bar, with the published band reported.

    The band is reported whatever happens, because unlike the interaction
    counts it is the authors' own published classification rather than a bar
    invented here. The pass or fail is still only a caller's: ``dockq_min`` has
    no default, so a gate names its bar and that bar is recorded as
    ``caller_supplied``.
    """
    score = record.get("global_dockq")
    if dockq_min is not None and not _valid_dockq_number(dockq_min, 1):
        raise ValueError("dockq_min must be a finite number in [0, 1]")
    if dockq_min is None:
        state = "no_cut"
        source = "no cut supplied; DockQ reported with its published band"
    else:
        state = "clear" if _valid_dockq_number(score, 1) and score >= dockq_min else "hit"
        source = f"caller_supplied dockq_min={dockq_min}"
    if not _valid_dockq_number(score, 1):
        score, state = None, "untested"
    return {
        "dockq": {
            "value": score,
            "band": dockq_quality(score) if score is not None else None,
            "band_source": DOCKQ_PAPER,
            "state": state,
            "threshold": dockq_min,
            "direction": "min" if dockq_min is not None else "none",
            "threshold_source": source,
        }
    }


def native_verdicts(fingerprint: InteractionFingerprint) -> dict[str, dict[str, object]]:
    """A report, not a gate. Interaction counts have no published pass mark.

    Every entry is ``no_cut``: an interaction fingerprint describes a pose, and
    how many of each interaction a good binder needs is chemistry-specific and
    uncalibrated here. Reported so a reviewer can set bars, never invented.
    """
    out: dict[str, dict[str, object]] = {}
    for name in INTERACTION_TYPES:
        out[name] = {
            "value": fingerprint.counts.get(name),
            "state": "no_cut" if name in fingerprint.counts else "untested",
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
    """Prefer ProLIF when requested and installed. The result names the engine used."""
    if prefer_prolif:
        try:
            return prolif_fingerprint(protein_path, mol)
        except ProLIFUnavailableError:
            pass
    return native_fingerprint(protein_path, mol)
