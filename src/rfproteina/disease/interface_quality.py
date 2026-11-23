"""Interface quality terms for a finished design, computed without PyRosetta.

The pass criteria in ``configs/metrics/pass_criteria`` judge an interface on
refolding RMSD, ipTM and ipAE. None of those says how much surface the binder
actually buries, how many polar contacts it makes, how many buried polar atoms
it leaves with no partner, or how well the two surfaces fit each other. Those
are the terms the published binder filter stacks carry and this repository does
not, so they are added here.

Nothing here duplicates ``design_metrics.ClashMetric`` or
``interaction_metrics.InterchainInteractionMetric``: those count clashes and
contacts, and these measure buried area, polar satisfaction, interface
composition and shape complementarity.

Honest naming, because the thresholds people quote come from a different
engine:

* ``interface_polar_contacts`` is a heavy-atom proxy. A designed CIF carries no
  hydrogens, so a real hydrogen-bond test cannot run on it. This counts N and O
  atom pairs across the interface within :data:`POLAR_CONTACT_A`. It is not a
  Rosetta hydrogen-bond count and will not equal one.
* ``buried_unsatisfied_polar_atoms`` is a chemistry-typed proxy (see
  :func:`_buried_unsatisfied`): a buried polar atom with no *complementary*
  (donor-for-acceptor) heavy-atom partner from another residue within
  :data:`HBOND_MAX_A`, filtered by a coarse heavy-atom angle where an antecedent
  atom is known. It is a better proxy than an element-only distance test but it
  is still not a Rosetta unsatisfied-hydrogen-bond count: without hydrogens the
  true donor-H-acceptor angle cannot be measured.
* Shape complementarity uses the Lawrence & Colman normal/distance score on
  probe-accessible van der Waals contact patches. The dot surface omits
  re-entrant patches and the median includes peripheral dots.

The BindCraft defaults in REFERENCE_THRESHOLDS describe its Rosetta methods.
The historical set of 23 generated designs has no outcome labels with which
to calibrate a classification threshold for these proxies.

This module does not call a predictor, a service, or a network.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from biotite.structure import AtomArray

#: Heavy-atom separation at which two polar atoms are counted as a contact, in
#: angstrom. The usual donor-acceptor heavy-atom distance cut.
POLAR_CONTACT_A = 3.5
#: Any heavy-atom pair closer than this counts toward ``interface_atom_contacts``.
CONTACT_A = 5.0
#: Per-atom solvent-accessible surface at or below which an atom is called
#: buried, in square angstrom.
BURIED_SASA_A2 = 1.0
#: Per-residue buried surface above which a residue is called an interface
#: residue, in square angstrom.
INTERFACE_RESIDUE_DSASA_A2 = 1.0
#: Per-atom buried surface above which a polar atom is treated as part of the
#: interface (buried on complexation) for the unsatisfied count, in A^2.
INTERFACE_ATOM_DSASA_A2 = 0.1
#: The elements treated as polar for the contact and satisfaction proxies.
POLAR_ELEMENTS = frozenset({"N", "O"})

#: Hydrogen-bond heavy-atom distance window, in angstrom. A donor-acceptor pair
#: closer than :data:`HBOND_MIN_A` is a clash, not a bond; beyond
#: :data:`HBOND_MAX_A` is too far. Cuts from the usual N/O...N/O hydrogen-bond
#: heavy-atom range (Baker & Hubbard 1984; McDonald & Thornton 1994).
HBOND_MIN_A = 2.4
HBOND_MAX_A = 3.5
#: Minimum acceptor-antecedent / donor-antecedent heavy-atom angle, in degrees,
#: for a pair to count as satisfying. A coarse geometric filter in place of the
#: donor-H-acceptor angle, which needs hydrogens this file does not carry.
HBOND_MIN_ANGLE_DEG = 90.0

#: van der Waals radii (angstrom) for the Sc dot surface. Bondi 1964 values for
#: the common elements, 1.70 for anything else. The choice of radius set, like
#: the probe below, shifts absolute Sc; both are recorded in ``sc_method``.
_VDW_RADII = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
              "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98}
_VDW_DEFAULT = 1.70
#: Sc probe radius. Lawrence & Colman use 1.7 A. Kept the same here.
SC_PROBE_A = 1.7
#: Sc Gaussian weight w in exp(-w * d^2), in A^-2. Lawrence & Colman use 0.5.
SC_WEIGHT = 0.5
#: Dots generated per atom sphere for the Sc surface. Higher is smoother and
#: slower; 100 is enough for a stable median on a protein interface.
SC_DOT_DENSITY = 100
#: An atom gets surface dots only if it is within this of the other molecule, in
#: angstrom. Only interface dots enter Sc, so dots on the far side are a waste.
SC_NEIGHBOUR_A = 8.0
#: The fewest interface dots on a side for Sc to be reported. Below this the
#: median is noise, so Sc is returned as None with a reason.
SC_MIN_DOTS = 10

METHOD = "biotite_sasa_kdtree_v3"
SC_METHOD = (
    "Lawrence-Colman normal/distance score on contact patches, partner-only burial "
    f"(probe={SC_PROBE_A}A, w={SC_WEIGHT}, density={SC_DOT_DENSITY}, Bondi radii)"
)

#: Reference values with their provenance. ``source`` is never empty: a bar
#: with no source is a bar nobody can check.
REFERENCE_THRESHOLDS: dict[str, dict[str, object]] = {
    "interface_residues_binder": {
        "min": 7,
        "source": (
            "BindCraft's interface-residue floor of 7 is a project default, not a published value. "
            "BindCraft counts contacting residues. The local count uses per-residue surface burial."
        ),
    },
    "interface_polar_contacts": {
        "min": 3,
        "source": (
            "BindCraft's floor of 3 applies to Rosetta hydrogen bonds. DEFINITION DIFFERS. The local "
            "measurement counts cross-interface N/O pairs within 3.5 A."
        ),
    },
    "buried_unsatisfied_polar_atoms": {
        "max": 4,
        "source": (
            "BindCraft uses a Rosetta buried-unsatisfied hydrogen-bond filter. The local heavy-atom proxy "
            "uses different burial and satisfaction rules. The Worker A calibration sample contained 23 "
            "unlabeled designs."
        ),
    },
    "shape_complementarity": {
        "min": None,
        "source": (
            "Lawrence & Colman 1993. SCALE DIFFERS because this proxy uses contact patches. The "
            "historical median of 0.166 used the earlier burial calculation."
        ),
    },
    "dsasa_total_a2": {
        "min": None,
        "source": (
            "Area in square angstroms. BindCraft reports Rosetta delta SASA and a separate interface SASA "
            "percentage. This Biotite measurement has no calibrated cutoff."
        ),
    },
}


@dataclass(frozen=True)
class InterfaceQuality:
    """One interface, measured. Areas are square angstrom; counts are atoms or residues."""

    binder_chain: str
    target_chains: tuple[str, ...]
    dsasa_total_a2: float
    dsasa_binder_a2: float
    dsasa_target_a2: float
    interface_residues_binder: int
    interface_residues_target: int
    interface_polar_contacts: int
    interface_atom_contacts: int
    buried_unsatisfied_polar_atoms: int
    buried_unsatisfied_donors: int
    buried_unsatisfied_acceptors: int
    shape_complementarity: float | None
    shape_complementarity_reason: str | None
    shape_complementarity_binder_median: float | None
    shape_complementarity_target_median: float | None
    interface_dots_binder: int
    interface_dots_target: int
    method: str = METHOD
    sc_method: str = SC_METHOD
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_record(self) -> dict[str, object]:
        """A plain dict for ``screen.json``, with the thresholds that apply and their sources."""
        record = asdict(self)
        record["target_chains"] = list(self.target_chains)
        record["notes"] = list(self.notes)
        record["reference_thresholds"] = REFERENCE_THRESHOLDS
        return record


class InterfaceUnreadableError(Exception):
    """The structure could not be read, or does not hold the chains asked for."""


def _structure(path: Path, cif_text: str | None = None) -> AtomArray:
    """First model of a CIF or PDB file, hydrogens and waters dropped."""
    try:
        from biotite.structure.io import load_structure
    except ImportError as error:  # pragma: no cover - biotite is a hard dependency of the package
        raise InterfaceUnreadableError(f"biotite_unavailable:{error}") from error
    try:
        from biotite.structure.io import pdbx

        if cif_text is not None:
            atoms = pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(cif_text)), model=1)
        elif path.suffix.lower() in {".gz", ".gzip", ".zst"}:
            from atomworks.io.utils.compression import open_compressed
            from biotite.structure.io.pdb import PDBFile

            fmt = path.with_suffix("").suffix.lower()
            with open_compressed(path, "rt", encoding="utf-8") as stream:
                if fmt == ".pdb":
                    atoms = PDBFile.read(stream).get_structure(model=1)
                elif fmt in {".cif", ".mmcif", ".pdbx"}:
                    atoms = pdbx.get_structure(pdbx.CIFFile.read(stream), model=1)
                else:
                    raise ValueError(f"unsupported compressed structure format {fmt}")
        else:
            atoms = load_structure(str(path), model=1)
    except Exception as error:  # every parse failure is the same answer: refused
        raise InterfaceUnreadableError(f"structure_unreadable:{type(error).__name__}") from error
    keep = (atoms.element != "H") & (atoms.res_name != "HOH")
    atoms = atoms[keep]
    if atoms.array_length() == 0:
        raise InterfaceUnreadableError("structure_empty_after_filtering")
    return atoms


def _sasa(atoms: AtomArray) -> np.ndarray:
    """Require a measured solvent-accessible area for each selected atom."""
    from biotite.structure import sasa

    values = np.asarray(sasa(atoms, vdw_radii="Single"), dtype=float)
    if not np.isfinite(values).all():
        raise InterfaceUnreadableError("sasa_nonfinite")
    return values


def _residue_keys(atoms: AtomArray) -> np.ndarray:
    parts = zip(atoms.chain_id, atoms.res_id, atoms.ins_code, strict=True)
    return np.array([f"{chain}:{res_id}:{ins}" for chain, res_id, ins in parts])


def _radii(elements: np.ndarray) -> np.ndarray:
    """van der Waals radius per atom, from :data:`_VDW_RADII`."""
    return np.array([_VDW_RADII.get(str(e).upper(), _VDW_DEFAULT) for e in elements], dtype=float)


def measure_interface(
    path: Path, binder_chain: str, target_chains: Sequence[str] | None = None, *, cif_text: str | None = None
) -> InterfaceQuality:
    """Measure one binder chain against every other chain, or against the ones named.

    Raises :class:`InterfaceUnreadableError` rather than returning a partial
    answer: a wrong clear on an interface term is worse than no term.
    """
    atoms = _structure(Path(path), cif_text)
    chains = list(dict.fromkeys(str(value) for value in atoms.chain_id))
    if binder_chain not in chains:
        raise InterfaceUnreadableError(f"binder_chain_absent:{binder_chain}")
    if target_chains is None:
        targets = tuple(name for name in chains if name != binder_chain)
    else:
        targets = tuple(dict.fromkeys(str(name) for name in target_chains))
        if binder_chain in targets:
            raise InterfaceUnreadableError("binder_also_target")
        missing = [name for name in targets if name not in chains]
        if missing:
            raise InterfaceUnreadableError(f"target_chain_absent:{','.join(missing)}")
    if not targets:
        raise InterfaceUnreadableError("no_target_chain")

    binder_mask = atoms.chain_id == binder_chain
    target_mask = np.isin(atoms.chain_id, np.array(targets))
    pair = atoms[binder_mask | target_mask]
    if not np.isfinite(pair.coord).all():
        raise InterfaceUnreadableError("coordinates_nonfinite")
    pair_binder = pair.chain_id == binder_chain

    complex_sasa = _sasa(pair)
    binder_alone = _sasa(pair[pair_binder])
    target_alone = _sasa(pair[~pair_binder])

    delta = np.zeros(pair.array_length(), dtype=float)
    delta[pair_binder] = binder_alone - complex_sasa[pair_binder]
    delta[~pair_binder] = target_alone - complex_sasa[~pair_binder]
    delta = np.clip(delta, 0.0, None)

    keys = _residue_keys(pair)
    residue_delta: dict[str, float] = {}
    residue_side: dict[str, bool] = {}
    for key, value, is_binder in zip(keys, delta, pair_binder, strict=True):
        residue_delta[key] = residue_delta.get(key, 0.0) + float(value)
        residue_side[key] = bool(is_binder)
    binder_residues = sum(
        1 for key, value in residue_delta.items() if residue_side[key] and value > INTERFACE_RESIDUE_DSASA_A2
    )
    target_residues = sum(
        1 for key, value in residue_delta.items() if not residue_side[key] and value > INTERFACE_RESIDUE_DSASA_A2
    )

    coords = pair.coord.astype(float)
    binder_coords = coords[pair_binder]
    target_coords = coords[~pair_binder]
    if binder_coords.size == 0 or target_coords.size == 0:
        raise InterfaceUnreadableError("empty_side")

    polar_contacts, atom_contacts = _contacts(pair, pair_binder)
    unsatisfied, unsat_donors, unsat_acceptors = _buried_unsatisfied(pair, complex_sasa, delta, keys)
    sc = _shape_complementarity(coords, _radii(pair.element), pair_binder)

    return InterfaceQuality(
        binder_chain=binder_chain,
        target_chains=targets,
        dsasa_total_a2=round(float(delta.sum()), 3),
        dsasa_binder_a2=round(float(delta[pair_binder].sum()), 3),
        dsasa_target_a2=round(float(delta[~pair_binder].sum()), 3),
        interface_residues_binder=binder_residues,
        interface_residues_target=target_residues,
        interface_polar_contacts=polar_contacts,
        interface_atom_contacts=atom_contacts,
        buried_unsatisfied_polar_atoms=unsatisfied,
        buried_unsatisfied_donors=unsat_donors,
        buried_unsatisfied_acceptors=unsat_acceptors,
        shape_complementarity=sc.value,
        shape_complementarity_reason=sc.reason,
        shape_complementarity_binder_median=sc.binder_median,
        shape_complementarity_target_median=sc.target_median,
        interface_dots_binder=sc.binder_dots,
        interface_dots_target=sc.target_dots,
        notes=(
            "polar contacts are a heavy-atom proxy, not hydrogen-bond counts",
            "buried-unsatisfied uses donor/acceptor typing and a coarse heavy-atom angle, not hydrogens",
            "shape_complementarity uses probe-accessible van der Waals contact patches",
            "hydrogens and waters were dropped before measuring",
        ),
    )


def _contacts(pair: AtomArray, pair_binder: np.ndarray) -> tuple[int, int]:
    """Cross-interface polar and all-atom contact counts, via a KD-tree neighbour list.

    Replaces the dense ``n_binder x n_target`` distance matrix the first pass
    built: that allocated one float per atom pair and scaled as the product of
    the two chain sizes. ``scipy.spatial.cKDTree`` returns only the pairs within
    the cut, so memory and time scale with the number of contacts, not with the
    product of the two chains.
    """
    from scipy.spatial import cKDTree

    coords = pair.coord.astype(float)
    binder_coords = coords[pair_binder]
    target_coords = coords[~pair_binder]
    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)

    # All-atom contacts: every binder-target pair within CONTACT_A.
    atom_pairs = binder_tree.sparse_distance_matrix(target_tree, CONTACT_A, output_type="coo_matrix")
    atom_contacts = int(atom_pairs.nnz)

    # Polar contacts: restrict the tree query to N/O atoms on each side.
    binder_polar_idx = np.flatnonzero(np.isin(pair.element[pair_binder], np.array(sorted(POLAR_ELEMENTS))))
    target_polar_idx = np.flatnonzero(np.isin(pair.element[~pair_binder], np.array(sorted(POLAR_ELEMENTS))))
    if binder_polar_idx.size == 0 or target_polar_idx.size == 0:
        return 0, atom_contacts
    binder_polar_tree = cKDTree(binder_coords[binder_polar_idx])
    target_polar_tree = cKDTree(target_coords[target_polar_idx])
    polar_pairs = binder_polar_tree.query_ball_tree(target_polar_tree, POLAR_CONTACT_A)
    polar_contacts = int(sum(len(hits) for hits in polar_pairs))
    return polar_contacts, atom_contacts


# --- buried-unsatisfied polar atoms, donor/acceptor typed --------------------

#: Polar sidechain atoms by residue, tagged ``d`` (donor), ``a`` (acceptor) or
#: ``b`` (both). Backbone N is a donor and backbone O an acceptor for every
#: residue. Charged donors are treated as donors; carboxylates as acceptors.
_SIDECHAIN_POLAR: dict[tuple[str, str], str] = {
    ("ASN", "ND2"): "d", ("ASN", "OD1"): "a",
    ("GLN", "NE2"): "d", ("GLN", "OE1"): "a",
    ("ASP", "OD1"): "a", ("ASP", "OD2"): "a",
    ("GLU", "OE1"): "a", ("GLU", "OE2"): "a",
    ("SER", "OG"): "b", ("THR", "OG1"): "b", ("TYR", "OH"): "b",
    ("HIS", "ND1"): "b", ("HIS", "NE2"): "b",
    ("LYS", "NZ"): "d", ("ARG", "NE"): "d", ("ARG", "NH1"): "d", ("ARG", "NH2"): "d",
    ("TRP", "NE1"): "d",
}
#: The heavy atom bonded to a polar atom, for the coarse angle. Backbone N->CA,
#: backbone O->C. A polar atom with no antecedent here is tested by distance only.
_ANTECEDENT: dict[tuple[str, str], str] = {
    ("ASN", "ND2"): "CG", ("ASN", "OD1"): "CG",
    ("GLN", "NE2"): "CD", ("GLN", "OE1"): "CD",
    ("ASP", "OD1"): "CG", ("ASP", "OD2"): "CG",
    ("GLU", "OE1"): "CD", ("GLU", "OE2"): "CD",
    ("SER", "OG"): "CB", ("THR", "OG1"): "CB", ("TYR", "OH"): "CZ",
    ("HIS", "ND1"): "CG", ("HIS", "NE2"): "CD2",
    ("LYS", "NZ"): "CE", ("ARG", "NE"): "CD", ("ARG", "NH1"): "CZ", ("ARG", "NH2"): "CZ",
    ("TRP", "NE1"): "CD1",
}


def _polar_type(res_name: str, atom_name: str, element: str) -> str | None:
    """``d``/``a``/``b`` for a polar heavy atom, or None when it is not polar."""
    name = atom_name.strip().upper()
    res = res_name.strip().upper()
    if name == "N":
        return "d"
    if name in {"O", "OXT"}:
        return "a"
    found = _SIDECHAIN_POLAR.get((res, name))
    if found is not None:
        return found
    # Unmapped N/O (modified residue, ligand, nucleic): treat as both so it can
    # satisfy and be satisfied, which is the lenient, non-overcounting choice.
    if str(element).upper() in POLAR_ELEMENTS:
        return "b"
    return None


def _buried_unsatisfied(
    pair: AtomArray, complex_sasa: np.ndarray, delta: np.ndarray, keys: np.ndarray
) -> tuple[int, int, int]:
    """Buried polar atoms with no complementary heavy-atom partner, donor/acceptor typed.

    An improvement on the element-only proxy: a buried donor is satisfied only by
    an acceptor (and vice versa), from a *different* residue, within
    :data:`HBOND_MIN_A`..:data:`HBOND_MAX_A`, and -- where an antecedent heavy
    atom is known for either side -- with a donor/acceptor-antecedent angle of at
    least :data:`HBOND_MIN_ANGLE_DEG`. ``both`` atoms (hydroxyls, histidine,
    unmapped N/O) can be satisfied by either.

    This still is not a Rosetta unsatisfied-hydrogen-bond count: without
    hydrogens the donor-H-acceptor angle itself cannot be measured, so the angle
    here is the heavy-atom D...A-antecedent angle, a coarse stand-in. Returns
    ``(total, donors, acceptors)``.
    """
    from scipy.spatial import cKDTree

    coords = pair.coord.astype(float)
    names = [str(n) for n in pair.atom_name]
    resn = [str(r) for r in pair.res_name]
    elems = [str(e) for e in pair.element]

    types: list[str | None] = [_polar_type(resn[i], names[i], elems[i]) for i in range(pair.array_length())]
    polar_idx = np.array([i for i, t in enumerate(types) if t is not None], dtype=int)
    if polar_idx.size == 0:
        return 0, 0, 0
    # Interface buried-unsatisfied, not whole-complex: a polar atom counts only
    # when it is buried in the complex AND it buried surface on complexation
    # (delta > INTERFACE_ATOM_DSASA_A2). That excludes the protein cores, whose
    # buried polar atoms are a property of the fold, not of this interface, and
    # which is what BindCraft's InterfaceUnsatHbonds counts as well.
    buried_mask = complex_sasa <= BURIED_SASA_A2
    candidates = np.array(
        [i for i in polar_idx if buried_mask[i] and delta[i] > INTERFACE_ATOM_DSASA_A2], dtype=int
    )
    if candidates.size == 0:
        return 0, 0, 0

    # Antecedent coordinate per polar atom, by (residue-key, mapped antecedent name).
    atom_by_resatom: dict[tuple[str, str], int] = {}
    for i in range(pair.array_length()):
        atom_by_resatom[(keys[i], names[i].strip().upper())] = i

    def antecedent(i: int) -> np.ndarray | None:
        nm = names[i].strip().upper()
        res = resn[i].strip().upper()
        if nm == "N":
            anc = "CA"
        elif nm in {"O", "OXT"}:
            anc = "C"
        else:
            anc = _ANTECEDENT.get((res, nm))
        if anc is None:
            return None
        j = atom_by_resatom.get((keys[i], anc))
        return coords[j] if j is not None else None

    polar_coords = coords[polar_idx]
    tree = cKDTree(polar_coords)
    unsat_d = 0
    unsat_a = 0
    for i in candidates:
        want_partner = {"d": {"a", "b"}, "a": {"d", "b"}, "b": {"a", "d", "b"}}[types[i]]  # type: ignore[index]
        neigh = tree.query_ball_point(coords[i], HBOND_MAX_A)
        satisfied = False
        anc_i = antecedent(i)
        for local in neigh:
            j = int(polar_idx[local])
            if j == i or keys[j] == keys[i]:
                continue
            if types[j] not in want_partner:
                continue
            d = float(np.linalg.norm(coords[j] - coords[i]))
            if d < HBOND_MIN_A or d > HBOND_MAX_A:
                continue
            if not _angle_ok(coords[i], coords[j], anc_i, antecedent(j)):
                continue
            satisfied = True
            break
        if not satisfied:
            if types[i] == "a":
                unsat_a += 1
            else:
                unsat_d += 1
    return unsat_d + unsat_a, unsat_d, unsat_a


def _angle_ok(a: np.ndarray, b: np.ndarray, anc_a: np.ndarray | None, anc_b: np.ndarray | None) -> bool:
    """Coarse heavy-atom angle filter. True when no antecedent is known for either side."""
    for centre, other, anc in ((a, b, anc_a), (b, a, anc_b)):
        if anc is None:
            continue
        v1 = other - centre
        v2 = anc - centre
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 == 0.0 or n2 == 0.0:
            continue
        cos = float(np.dot(v1, v2) / (n1 * n2))
        cos = max(-1.0, min(1.0, cos))
        angle = float(np.degrees(np.arccos(cos)))
        if angle < HBOND_MIN_ANGLE_DEG:
            return False
    return True


# --- Lawrence-Colman shape complementarity -----------------------------------


@dataclass(frozen=True)
class _ScResult:
    value: float | None
    reason: str | None
    binder_median: float | None
    target_median: float | None
    binder_dots: int
    target_dots: int


def _unit_sphere(density: int) -> np.ndarray:
    """``density`` roughly-even points on the unit sphere, by the Fibonacci spiral."""
    i = np.arange(density, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / density)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def _surface_dots(
    coords: np.ndarray,
    vdw: np.ndarray,
    gen_idx: np.ndarray,
    context_coords: np.ndarray,
    context_vdw: np.ndarray,
    unit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Connolly contact-surface dots (points on the vdW surface, outward normals).

    A dot is placed on the van der Waals sphere of every atom in ``gen_idx``. It
    is kept when a probe sphere resting tangent to it -- centre at
    ``point + probe * normal`` -- overlaps no atom in the context set; that is
    the contact portion of the molecular surface (Connolly 1983), the part the
    solvent probe can actually touch. Re-entrant (toroidal and concave) patches
    are not generated. ``context`` is the atom set whose presence decides
    accessibility: the molecule alone for its isolated surface, the whole complex
    for burial. Points lie on the vdW surface, so the two sides of a packed
    interface sit close together, which is what makes the Gaussian in
    :func:`_side_sc` meaningful.
    """
    from scipy.spatial import cKDTree

    ctx_reach = context_vdw + SC_PROBE_A
    ctx_tree = cKDTree(context_coords)
    ctx_rmax = float(ctx_reach.max()) if ctx_reach.size else 0.0
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for i in gen_idx:
        centre = coords[i]
        surface = centre + vdw[i] * unit
        probe_centres = centre + (vdw[i] + SC_PROBE_A) * unit
        near = ctx_tree.query_ball_point(centre, vdw[i] + 2.0 * SC_PROBE_A + ctx_rmax)
        near = [j for j in near if not np.allclose(context_coords[j], centre)]
        if near:
            near_coords = context_coords[near]
            near_reach = ctx_reach[near]
            d2 = np.sum((probe_centres[:, None, :] - near_coords[None, :, :]) ** 2, axis=2)
            blocked = np.any(d2 < (near_reach[None, :] ** 2), axis=1)
            keep = ~blocked
        else:
            keep = np.ones(surface.shape[0], dtype=bool)
        if np.any(keep):
            points.append(surface[keep])
            normals.append(unit[keep])
    if not points:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.concatenate(points), np.concatenate(normals)


def _side_sc(
    iface_points: np.ndarray,
    iface_normals: np.ndarray,
    other_points: np.ndarray,
    other_normals: np.ndarray,
) -> float | None:
    """Median S over one side's interface dots against the other side's full surface."""
    from scipy.spatial import cKDTree

    if iface_points.shape[0] < SC_MIN_DOTS or other_points.shape[0] == 0:
        return None
    tree = cKDTree(other_points)
    dist, idx = tree.query(iface_points, k=1)
    nearest_normals = other_normals[idx]
    # Opposing outward normals point toward each other, so n . (-n') = +1 at a
    # perfect fit. Lawrence & Colman 1993, eq. for S(A).
    dot = np.sum(iface_normals * (-nearest_normals), axis=1)
    s = dot * np.exp(-SC_WEIGHT * np.asarray(dist) ** 2)
    return float(np.median(s))


def _shape_complementarity(coords: np.ndarray, radii: np.ndarray, pair_binder: np.ndarray) -> _ScResult:
    """Score opposing contact normals, weighted by nearest-neighbour distance.

    Dots lie on van der Waals spheres where the probe can touch them. Each
    side contributes its median over dots buried by the other molecule.
    The score includes peripheral dots and omits re-entrant patches.
    """
    binder_idx = np.flatnonzero(pair_binder)
    target_idx = np.flatnonzero(~pair_binder)
    if binder_idx.size == 0 or target_idx.size == 0:
        return _ScResult(None, "empty_side", None, None, 0, 0)

    from scipy.spatial import cKDTree

    binder_coords = coords[binder_idx]
    target_coords = coords[target_idx]
    # Only atoms near the other molecule can bear interface dots.
    b_tree = cKDTree(binder_coords)
    t_tree = cKDTree(target_coords)
    b_near_local = b_tree.query_ball_point(target_coords, SC_NEIGHBOUR_A)
    t_near_local = t_tree.query_ball_point(binder_coords, SC_NEIGHBOUR_A)
    b_near = sorted({binder_idx[j] for hits in b_near_local for j in hits})
    t_near = sorted({target_idx[j] for hits in t_near_local for j in hits})
    if not b_near or not t_near:
        return _ScResult(None, "no_interface_atoms", None, None, 0, 0)
    b_near_arr = np.array(b_near, dtype=int)
    t_near_arr = np.array(t_near, dtype=int)

    unit = _unit_sphere(SC_DOT_DENSITY)
    binder_radii = radii[binder_idx]
    target_radii = radii[target_idx]

    # Isolated surfaces (burial by own molecule only).
    b_iso_pts, b_iso_norm = _surface_dots(
        coords, radii, b_near_arr, binder_coords, binder_radii, unit
    )
    t_iso_pts, t_iso_norm = _surface_dots(
        coords, radii, t_near_arr, target_coords, target_radii, unit
    )
    if b_iso_pts.shape[0] == 0 or t_iso_pts.shape[0] == 0:
        return _ScResult(None, "no_surface_dots", None, None, 0, 0)

    # Each probe already cleared its own molecule. Testing its tangent atom
    # again lets roundoff classify exposed dots as buried.
    b_iface_pts, b_iface_norm = _buried_subset(b_iso_pts, b_iso_norm, target_coords, target_radii)
    t_iface_pts, t_iface_norm = _buried_subset(t_iso_pts, t_iso_norm, binder_coords, binder_radii)

    binder_dots = int(b_iface_pts.shape[0])
    target_dots = int(t_iface_pts.shape[0])
    if binder_dots < SC_MIN_DOTS or target_dots < SC_MIN_DOTS:
        return _ScResult(None, f"interface_too_small_for_sc:{binder_dots},{target_dots}",
                         None, None, binder_dots, target_dots)

    sc_b = _side_sc(b_iface_pts, b_iface_norm, t_iso_pts, t_iso_norm)
    sc_t = _side_sc(t_iface_pts, t_iface_norm, b_iso_pts, b_iso_norm)
    if sc_b is None or sc_t is None:
        return _ScResult(None, "sc_side_empty", sc_b, sc_t, binder_dots, target_dots)
    value = round((sc_b + sc_t) / 2.0, 4)
    return _ScResult(value, None, round(sc_b, 4), round(sc_t, 4), binder_dots, target_dots)


def _buried_subset(
    points: np.ndarray, normals: np.ndarray, partner_coords: np.ndarray, partner_vdw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The isolated contact-surface dots the full complex buries -- the interface patch.

    A dot survived the isolated-surface test, so its probe was clear of its own
    molecule. It is an interface dot when that same probe (centre at
    ``point + probe * normal``) now overlaps an atom of the complex -- i.e. the
    partner is what desolvates it.
    """
    from scipy.spatial import cKDTree

    reach = partner_vdw + SC_PROBE_A
    rmax = float(reach.max()) if reach.size else 0.0
    probe_centres = points + SC_PROBE_A * normals
    tree = cKDTree(partner_coords)
    near = tree.query_ball_point(probe_centres, rmax)
    buried = np.zeros(points.shape[0], dtype=bool)
    for k, hits in enumerate(near):
        if not hits:
            continue
        hits_arr = np.array(hits)
        d2 = np.sum((partner_coords[hits_arr] - probe_centres[k]) ** 2, axis=1)
        if np.any(d2 < reach[hits_arr] ** 2):
            buried[k] = True
    return points[buried], normals[buried]


def verdicts(measured: InterfaceQuality) -> dict[str, dict[str, object]]:
    """Report measured values and the reference thresholds on their original scale."""
    record = asdict(measured)
    out: dict[str, dict[str, object]] = {}
    for name, rule in REFERENCE_THRESHOLDS.items():
        value = record.get(name)
        reference = {key: rule[key] for key in ("min", "max") if rule.get(key) is not None}
        out[name] = {
            "value": value,
            "state": "untested" if value is None else "no_cut",
            "threshold": None,
            "direction": "none",
            "reference_threshold": reference or None,
            "threshold_source": rule["source"],
        }
    return out
