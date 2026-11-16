"""Comparing a retrieved model against the structure this run produced.

What goes under each overlay on the results page. Kept to ``_atom_site`` rows
and numpy so the panel stays as light as the rest of it -- the viewers and the
FASTA need no torch, no atomworks and no biotite, and this must not be the
thing that drags them in.

Two numbers are reported and neither means much without the other: the RMSD,
and **how many residues it was taken over**. An RMSD of 1.2 A over twelve of a
hundred and nine residues is not the claim an RMSD of 1.2 A over all of them
is, and a single figure cannot tell them apart.

That is also why correspondence is established from the sequences rather than
from residue numbers. A model from the AlphaFold database is numbered from 1
across the whole UniProt sequence; an experimental structure of the same protein
may begin at residue 20, stop early, and skip a disordered loop in the middle.
Pairing those by ``label_seq_id`` silently compares residue 30 of one against a
different residue 30 of the other, and reports a large RMSD as though the
structures disagreed when it is the numbering that does.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rfproteina.disease.design_spec import _atom_site_rows

#: Three-letter to one-letter, for the correspondence sequence only. A residue
#: outside this is kept as ``X``: it still has to occupy a position, because
#: dropping it would shift every pairing after it.
_ONE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}
#: Below this many paired residues an RMSD is not worth reporting: three points
#: can be superposed onto any other three almost exactly, so the number would
#: say more about the arithmetic than the structures.
MIN_PAIRED = 8


@dataclass(frozen=True)
class Residues:
    """One chain's alpha carbons, in file order."""

    chain: str
    sequence: str
    coords: np.ndarray

    def __len__(self) -> int:
        return len(self.sequence)


@dataclass(frozen=True)
class Comparison:
    """What one overlay's caption says, or why there is nothing to say."""

    rmsd: float | None = None
    paired: int = 0
    left_residues: int = 0
    right_residues: int = 0
    left_chain: str = ""
    right_chain: str = ""
    blocker: str = ""
    #: Rotation and translation putting the RIGHT structure into the left's
    #: frame, or None when nothing was superposed. Without this an overlay is
    #: two structures in unrelated frames.
    transform: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def measured(self) -> bool:
        return self.rmsd is not None and not self.blocker

    @property
    def coverage(self) -> float | None:
        """Paired residues as a share of the smaller chain, or None."""
        smaller = min(self.left_residues, self.right_residues)
        return self.paired / smaller if smaller else None

    def sentence(self) -> str:
        """The detail line under an overlay. Never a bare RMSD."""
        if not self.measured:
            return _BLOCKER_TEXT.get(self.blocker, self.blocker or "Not compared.")
        share = self.coverage
        over = (
            f"{self.paired} of {min(self.left_residues, self.right_residues)} residues"
            if share is not None
            else f"{self.paired} residues"
        )
        caveat = ""
        if share is not None and share < 0.8:
            # Said rather than left to the reader's arithmetic: a partial
            # pairing is the case where a small RMSD is least informative.
            caveat = " Most of the chain was not paired, so this compares the part that was."
        return (
            f"C-alpha RMSD {self.rmsd:.2f} A over {over}"
            f" (chains {self.left_chain} and {self.right_chain})."
            f"{caveat}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "rmsd": None if self.rmsd is None else round(self.rmsd, 3),
            "paired": self.paired,
            "left_residues": self.left_residues,
            "right_residues": self.right_residues,
            "left_chain": self.left_chain,
            "right_chain": self.right_chain,
            "blocker": self.blocker,
            "sentence": self.sentence(),
        }


_BLOCKER_TEXT = {
    "no_left_chain": "The run's own structure had no chain to compare.",
    "no_right_chain": "The retrieved model had no chain to compare.",
    "too_few_paired": (
        f"Fewer than {MIN_PAIRED} residues could be paired by sequence, so no RMSD was taken: "
        "a handful of points superpose onto almost anything."
    ),
    "no_common_sequence": (
        "No run of residues was shared by the two sequences, so there is nothing to "
        "superpose. A retrieved model of a different protein looks like this."
    ),
}


def looks_like_pdb(text: str) -> bool:
    """Whether this is PDB text rather than mmCIF.

    Both are offered for every model in the AlphaFold database -- ``cifUrl``
    and ``pdbUrl`` -- and a person retrieving one by hand will take whichever
    their browser opened. Sniffed on the records rather than on a file
    extension, because pasted text has no extension at all.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("data_", "loop_", "_atom_site.")):
            return False
        if stripped.startswith(("ATOM  ", "HETATM", "MODEL ", "HEADER", "TITLE ")):
            return True
    return False


def _pdb_alpha_carbons(text: str, chain: str) -> Residues | None:
    """One chain's CA coordinates from PDB ATOM records, by column.

    PDB is a fixed-column format and splitting on whitespace is the classic way
    to misread it: a four-character atom name runs into the residue name, and a
    coordinate of -100.000 runs into the one before it. So the fields are taken
    at their documented columns.
    """
    wanted = chain.strip()
    sequence: list[str] = []
    coords: list[tuple[float, float, float]] = []
    seen: set[str] = set()
    for line in text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        if line[12:16].strip() != "CA":
            continue
        if line[21:22].strip() != wanted:
            continue
        altloc = line[16:17].strip()
        if altloc not in {"", "A"}:
            # A second position for one alpha carbon would pair twice and
            # weight that residue double.
            continue
        key = line[22:27]
        if key in seen:
            continue
        try:
            point = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        seen.add(key)
        sequence.append(_ONE_LETTER.get(line[17:20].strip().upper(), "X"))
        coords.append(point)
    if not coords:
        return None
    return Residues(wanted, "".join(sequence), np.asarray(coords, dtype=float))


def alpha_carbons(text: str, chain: str) -> Residues | None:
    """One chain's CA coordinates and sequence, or None when it has none.

    ``label_asym_id`` first and ``auth_asym_id`` as a fallback, because a model
    from the AlphaFold database labels its single chain ``A`` in both while a
    generated design and an experimental deposit do not always agree.

    Altlocs beyond the first for a residue are skipped: two positions for one
    alpha carbon would pair twice and weight that residue double.
    """
    wanted = chain.strip()
    if not wanted:
        return None
    if looks_like_pdb(text):
        return _pdb_alpha_carbons(text, wanted)
    sequence: list[str] = []
    coords: list[tuple[float, float, float]] = []
    seen: set[str] = set()
    for row in _atom_site_rows(text):
        if (row.get("label_atom_id") or row.get("auth_atom_id") or "").strip().strip('"') != "CA":
            continue
        label = (row.get("label_asym_id") or "").strip()
        auth = (row.get("auth_asym_id") or "").strip()
        if wanted not in {label, auth}:
            continue
        key = f"{label}|{auth}|{(row.get('label_seq_id') or '').strip()}|{(row.get('auth_seq_id') or '').strip()}"
        if key in seen:
            continue
        try:
            point = (
                float(row.get("Cartn_x", "")),
                float(row.get("Cartn_y", "")),
                float(row.get("Cartn_z", "")),
            )
        except (TypeError, ValueError):
            continue
        seen.add(key)
        name = (row.get("label_comp_id") or row.get("auth_comp_id") or "").strip().upper()
        sequence.append(_ONE_LETTER.get(name, "X"))
        coords.append(point)
    if not coords:
        return None
    return Residues(wanted, "".join(sequence), np.asarray(coords, dtype=float))


def longest_shared_run(left: str, right: str) -> tuple[int, int, int]:
    """Start in each sequence and the length of their longest common substring.

    A substring rather than a subsequence, and deliberately: a subsequence
    match may interleave residues from either side of a gap, which pairs
    positions that are not the same position and produces an RMSD over a
    correspondence no alignment would accept. A contiguous run is a claim that
    can be checked by eye.
    """
    if not left or not right:
        return 0, 0, 0
    previous = [0] * (len(right) + 1)
    best = best_left = best_right = 0
    for i, letter in enumerate(left, start=1):
        current = [0] * (len(right) + 1)
        for j, other in enumerate(right, start=1):
            if letter != other or letter == "X":
                continue
            current[j] = previous[j - 1] + 1
            if current[j] > best:
                best, best_left, best_right = current[j], i - current[j], j - current[j]
        previous = current
    return best_left, best_right, best


def kabsch_transform(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rotation and translation putting ``mobile`` onto ``target``.

    Returned as well as the RMSD because an overlay that is not superposed is
    not an overlay. The viewer draws each structure at the coordinates in its
    file, and a model from the AlphaFold database sits in whatever frame the
    predictor produced -- nothing relates it to an experimental deposit's
    frame. Drawn untransformed the two appear as separate objects in one box,
    which looks like a comparison and is not one.

    Apply as ``point @ rotation.T + translation``.
    """
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[0] == 0:
        raise ValueError("kabsch_transform needs two equal, non-empty (n, 3) arrays")
    mobile_centre = mobile.mean(axis=0)
    target_centre = target.mean(axis=0)
    left = mobile - mobile_centre
    right = target - target_centre
    u, _singular, vt = np.linalg.svd(left.T @ right)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, sign if sign != 0 else 1.0])
    rotation = vt.T @ correction @ u.T
    return rotation, target_centre - mobile_centre @ rotation.T


def apply_transform(text: str, rotation: np.ndarray, translation: np.ndarray) -> str:
    """``text`` with every atom moved, so a viewer draws it in the other frame.

    Every atom, not only the alpha carbons the fit was taken over: a structure
    half-moved is not a structure. Rewritten in place rather than re-serialised,
    because everything else in the file -- the entities, the chain labels, the
    confidence columns a reader may want -- is the model's and not ours to
    reformat.
    """
    if looks_like_pdb(text):
        return _apply_pdb(text, rotation, translation)
    return _apply_cif(text, rotation, translation)


def _moved(x: float, y: float, z: float, rotation: np.ndarray, translation: np.ndarray) -> tuple[float, float, float]:
    point = np.asarray([x, y, z], dtype=float) @ rotation.T + translation
    return float(point[0]), float(point[1]), float(point[2])


def _apply_pdb(text: str, rotation: np.ndarray, translation: np.ndarray) -> str:
    out: list[str] = []
    for line in text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            out.append(line)
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            out.append(line)
            continue
        new_x, new_y, new_z = _moved(x, y, z, rotation, translation)
        out.append(f"{line[:30]}{new_x:8.3f}{new_y:8.3f}{new_z:8.3f}{line[54:]}")
    return "\n".join(out) + "\n"


def _apply_cif(text: str, rotation: np.ndarray, translation: np.ndarray) -> str:
    """Rewrite the three coordinate columns of each ``_atom_site`` loop row.

    Whitespace-delimited and column-counted the same way
    :func:`~rfproteina.disease.design_spec._atom_site_rows` reads it, so a row
    wrapped across lines is handled by the same rule that parses it. A row this
    cannot line up is passed through untouched rather than half-rewritten.
    """
    lines = text.splitlines()
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.strip() != "loop_":
            out.append(line)
            index += 1
            continue
        out.append(line)
        index += 1
        tags: list[str] = []
        while index < len(lines) and lines[index].lstrip().startswith("_"):
            tags.append(lines[index].strip().split()[0])
            out.append(lines[index])
            index += 1
        columns = [tag.removeprefix("_atom_site.") for tag in tags]
        if not tags or not tags[0].startswith("_atom_site."):
            continue
        try:
            slots = (columns.index("Cartn_x"), columns.index("Cartn_y"), columns.index("Cartn_z"))
        except ValueError:
            continue
        while index < len(lines):
            row = lines[index]
            stripped = row.strip()
            if not stripped or stripped.startswith(("_", "#", "loop_", "data_")):
                break
            parts = stripped.split()
            if len(parts) != len(columns):
                out.append(row)
                index += 1
                continue
            try:
                x, y, z = (float(parts[slot]) for slot in slots)
            except ValueError:
                out.append(row)
                index += 1
                continue
            new = _moved(x, y, z, rotation, translation)
            for slot, value in zip(slots, new, strict=True):
                parts[slot] = f"{value:.3f}"
            out.append(" ".join(parts))
            index += 1
    return "\n".join(out) + "\n"


def kabsch_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    """RMSD after optimal superposition, by Kabsch with a reflection guard.

    The guard is the part that is easy to leave out and impossible to notice:
    without correcting the determinant, the optimal rotation may be improper --
    a reflection -- which superposes a structure onto the mirror image of its
    partner and returns an RMSD lower than any rotation can achieve. A chain
    and its mirror are not the same chain, and a protein's handedness is not a
    detail, so a reflected fit flatters the comparison.
    """
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[0] == 0:
        raise ValueError("kabsch_rmsd needs two equal, non-empty (n, 3) arrays")
    left = mobile - mobile.mean(axis=0)
    right = target - target.mean(axis=0)
    covariance = left.T @ right
    u, _singular, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, sign if sign != 0 else 1.0])
    rotation = vt.T @ correction @ u.T
    moved = left @ rotation.T
    return float(np.sqrt(np.mean(np.sum((moved - right) ** 2, axis=1))))


def compare_chains(
    left_text: str,
    right_text: str,
    *,
    left_chain: str,
    right_chain: str,
) -> Comparison:
    """Superpose two chains on the residues their sequences actually share."""
    left = alpha_carbons(left_text, left_chain)
    if left is None:
        return Comparison(left_chain=left_chain, right_chain=right_chain, blocker="no_left_chain")
    right = alpha_carbons(right_text, right_chain)
    if right is None:
        return Comparison(left_chain=left_chain, right_chain=right_chain, blocker="no_right_chain")

    start_left, start_right, run = longest_shared_run(left.sequence, right.sequence)
    common = Comparison(
        paired=run,
        left_residues=len(left),
        right_residues=len(right),
        left_chain=left.chain,
        right_chain=right.chain,
    )
    if run == 0:
        return Comparison(**{**common.__dict__, "blocker": "no_common_sequence"})
    if run < MIN_PAIRED:
        return Comparison(**{**common.__dict__, "blocker": "too_few_paired"})
    taken = left.coords[start_left : start_left + run]
    against = right.coords[start_right : start_right + run]
    rotation, translation = kabsch_transform(against, taken)
    return Comparison(
        **{
            **common.__dict__,
            "rmsd": kabsch_rmsd(against, taken),
            "transform": (rotation, translation),
        }
    )


def first_chain(text: str) -> str:
    """The first polymer chain label in a model, for a file that names one chain.

    A model retrieved from the AlphaFold database holds exactly one chain, and
    its label is not worth asking the caller for.
    """
    labels = chain_labels(text)
    return labels[0] if labels else ""


def chain_labels(text: str) -> tuple[str, ...]:
    """Every chain label in file order, without repeats."""
    found: list[str] = []
    if looks_like_pdb(text):
        for line in text.splitlines():
            if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 22:
                continue
            label = line[21:22].strip()
            if label and label not in found:
                found.append(label)
        return tuple(found)
    for row in _atom_site_rows(text):
        label = (row.get("label_asym_id") or row.get("auth_asym_id") or "").strip()
        if label and label not in found:
            found.append(label)
    return tuple(found)


def best_match(left_text: str, left_chain: str, right_text: str, candidates: Sequence[str]) -> Comparison:
    """The candidate chain that superposes best onto ``left_chain``.

    A retrieved complex may name its chains in an order nobody promised, so the
    pairing is measured rather than assumed. Ties go to the first, and a
    candidate that cannot be compared at all is skipped rather than returned as
    a comparison with no number in it.
    """
    best: Comparison | None = None
    for candidate in candidates:
        found = compare_chains(
            left_text, right_text, left_chain=left_chain, right_chain=candidate
        )
        if not found.measured:
            continue
        if best is None or (found.rmsd is not None and best.rmsd is not None and found.rmsd < best.rmsd):
            best = found
    if best is not None:
        return best
    return Comparison(left_chain=left_chain, blocker="no_common_sequence")


# --------------------------------------------------------- whole complexes


@dataclass(frozen=True)
class ComplexComparison:
    """A retrieved complex against the one this run produced.

    One number cannot answer this. A model can fold both chains well and dock
    them wrongly, and an overall RMSD taken after fitting everything at once
    hides exactly that: the fit splits the error between the two chains and
    reports something middling for a prediction that put the binder on the
    wrong face.

    So three things are reported. ``overall`` is the fit over both chains
    together. ``per_chain`` is each chain fitted on its own, which says whether
    the folds are right. ``binder_on_target`` is the one that answers the
    interface question: superpose on the TARGET alone, then measure the binder
    where it lands, without re-fitting it. That is what the refold criteria
    call an aligned binder RMSD, and a model that mislocates the binder shows
    it there and nowhere else.

    ``transform`` is the one the overlay is drawn with, and it is the target
    fit rather than the whole-complex fit, so the picture shows what
    ``binder_on_target`` reports instead of a compromise that lines up
    neither chain.
    """

    overall: Comparison | None = None
    per_chain: tuple[tuple[str, Comparison], ...] = ()
    binder_on_target: float | None = None
    binder_chain: str = ""
    target_chain: str = ""
    paired: int = 0
    #: How many of this run's chains were matched, and how many it has. A
    #: comparison that covered one chain of two is not a complex comparison,
    #: and the sentence has to say so rather than calling it one.
    chains_matched: int = 0
    chains_total: int = 0
    blocker: str = ""
    transform: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def measured(self) -> bool:
        return self.overall is not None and self.overall.measured and not self.blocker

    def sentence(self) -> str:
        if not self.measured or self.overall is None:
            return _COMPLEX_BLOCKERS.get(self.blocker, self.blocker or "Not compared.")
        whole = self.chains_total and self.chains_matched >= self.chains_total
        if whole:
            parts = [
                f"Whole complex C-alpha RMSD {self.overall.rmsd:.2f} A over "
                f"{self.paired} residues."
            ]
        else:
            # Said plainly. A model holding one chain of a two-chain complex
            # produces a perfectly good number for that chain, and calling it
            # a whole-complex RMSD would be the page reporting a complex
            # result from a comparison that never saw the complex.
            parts = [
                f"Only {self.chains_matched} of {self.chains_total} chains matched, so this is "
                f"not a comparison of the complex. C-alpha RMSD {self.overall.rmsd:.2f} A over "
                f"{self.paired} residues of the chains that did."
            ]
        for chain, found in self.per_chain:
            if found.measured and found.rmsd is not None:
                parts.append(f"Chain {chain} alone {found.rmsd:.2f} A over {found.paired}.")
        if not whole:
            parts.append(
                "Nothing here says whether the complex forms: that needs a model holding "
                "every chain."
            )
        if self.binder_on_target is not None:
            parts.append(
                f"Binder {self.binder_chain} sits {self.binder_on_target:.2f} A off once the "
                f"target {self.target_chain} is superposed, which is the number that says "
                "whether the complex was predicted and not merely the folds."
            )
        return " ".join(parts)

    def as_dict(self) -> dict[str, object]:
        return {
            "overall": None if self.overall is None else self.overall.as_dict(),
            "per_chain": {chain: found.as_dict() for chain, found in self.per_chain},
            "binder_on_target": None
            if self.binder_on_target is None
            else round(self.binder_on_target, 3),
            "binder_chain": self.binder_chain,
            "target_chain": self.target_chain,
            "paired": self.paired,
            "chains_matched": self.chains_matched,
            "chains_total": self.chains_total,
            "blocker": self.blocker,
            "sentence": self.sentence(),
        }


_COMPLEX_BLOCKERS = {
    "no_chains": "This run's structure names no chains to compare.",
    "no_model_chains": "The retrieved model has no chains to compare.",
    "nothing_paired": (
        "No chain of the retrieved model matched any chain of this complex by sequence. "
        "A model of a different complex looks like this."
    ),
}


def pair_chains(
    own_text: str,
    own_chains: Sequence[str],
    model_text: str,
    model_chains: Sequence[str],
) -> list[tuple[str, str, Comparison]]:
    """Each of this run's chains matched to the model chain that fits it best.

    Measured rather than assumed, and each model chain is used at most once: a
    retrieved complex names its chains in whatever order it likes, and a
    homodimer would otherwise pair both of ours to the same one of theirs and
    report the second as a perfect fit to a chain already spoken for.
    """
    taken: set[str] = set()
    pairs: list[tuple[str, str, Comparison]] = []
    for mine in own_chains:
        best: tuple[str, Comparison] | None = None
        for theirs in model_chains:
            if theirs in taken:
                continue
            found = compare_chains(own_text, model_text, left_chain=mine, right_chain=theirs)
            if not found.measured or found.rmsd is None:
                continue
            if best is None or (best[1].rmsd is not None and found.rmsd < best[1].rmsd):
                best = (theirs, found)
        if best is not None:
            taken.add(best[0])
            pairs.append((mine, best[0], best[1]))
    return pairs


def _matched_coords(
    own_text: str, model_text: str, pairs: Sequence[tuple[str, str]]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Our alpha carbons and theirs, in correspondence, over the named pairs."""
    ours: list[np.ndarray] = []
    theirs: list[np.ndarray] = []
    for mine, matched in pairs:
        left = alpha_carbons(own_text, mine)
        right = alpha_carbons(model_text, matched)
        if left is None or right is None:
            continue
        start_left, start_right, run = longest_shared_run(left.sequence, right.sequence)
        if run < MIN_PAIRED:
            continue
        ours.append(left.coords[start_left : start_left + run])
        theirs.append(right.coords[start_right : start_right + run])
    if not ours:
        return None
    return np.concatenate(ours), np.concatenate(theirs)


def compare_complex(
    own_text: str,
    model_text: str,
    *,
    own_chains: Sequence[str],
    model_chains: Sequence[str],
    binder_chain: str = "",
) -> ComplexComparison:
    """A whole predicted complex against the one this run produced."""
    if not own_chains:
        return ComplexComparison(blocker="no_chains")
    if not model_chains:
        return ComplexComparison(blocker="no_model_chains")
    pairs = pair_chains(own_text, own_chains, model_text, model_chains)
    if not pairs:
        return ComplexComparison(blocker="nothing_paired")

    flat = [(mine, matched) for mine, matched, _found in pairs]
    coords = _matched_coords(own_text, model_text, flat)
    if coords is None:
        return ComplexComparison(blocker="nothing_paired")
    stacked_ours, stacked_theirs = coords
    rotation, translation = kabsch_transform(stacked_theirs, stacked_ours)
    overall = Comparison(
        rmsd=kabsch_rmsd(stacked_theirs, stacked_ours),
        paired=len(stacked_ours),
        left_residues=len(stacked_ours),
        right_residues=len(stacked_theirs),
        left_chain=",".join(mine for mine, _m in flat),
        right_chain=",".join(matched for _mine, matched in flat),
        transform=(rotation, translation),
    )

    binder_off: float | None = None
    target_named = ""
    # What the overlay is drawn with. The whole-complex fit is the right
    # frame for the overall number and the wrong one for the picture: it is a
    # compromise across both chains, so a model that mislocates the binder is
    # drawn with NEITHER chain on ours -- measured on a real AlphaFold 3 fold,
    # the target landed 24 A out and the binder 11 A out, and the reader saw
    # two complexes that overlap nowhere. Superposing on the target instead
    # puts the target on ours and leaves the binder standing exactly as far
    # off as the sentence underneath says it is.
    drawn = (rotation, translation)
    chosen = binder_chain if binder_chain else ""
    if chosen and len(flat) > 1:
        targets = [pair for pair in flat if pair[0] != chosen]
        binders = [pair for pair in flat if pair[0] == chosen]
        if targets and binders:
            target_named = targets[0][0]
            fitted = _matched_coords(own_text, model_text, targets)
            if fitted is not None:
                rot, shift = kabsch_transform(fitted[1], fitted[0])
                binder_off = _deviation_after(own_text, model_text, binders[0], rot, shift)
                drawn = (rot, shift)
    return ComplexComparison(
        overall=overall,
        per_chain=tuple((mine, found) for mine, _matched, found in pairs),
        binder_on_target=binder_off,
        binder_chain=chosen,
        target_chain=target_named,
        paired=len(stacked_ours),
        chains_matched=len(pairs),
        chains_total=len(own_chains),
        transform=drawn,
    )


def _deviation_after(
    own_text: str,
    model_text: str,
    pair: tuple[str, str],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> float | None:
    """How far the model's chain sits from ours under a transform already fixed.

    Not re-fitted. Fitting the binder again would measure its fold a second
    time and say nothing about where the model placed it, which is the whole
    question this number exists to answer.
    """
    coords = _matched_coords(own_text, model_text, [pair])
    if coords is None:
        return None
    ours, theirs = coords
    moved = theirs @ rotation.T + translation
    return float(np.sqrt(np.mean(np.sum((moved - ours) ** 2, axis=1))))
