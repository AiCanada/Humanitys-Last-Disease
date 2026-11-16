"""Chemical-liability report for one designed chain.

The motifs are the ones used to rank forced-degradation risk: Asn-Gly and
Asn-Ser (deamidation), Asp-Gly (isomerization), and the N-glycosylation sequon
Asn-X-Ser/Thr where X is not Pro. Methionine and tryptophan are counted.
When the CIF names disulfide partners, a cysteine with no partner is free.
When it names no bonds, free cysteine stays untested and no count is stored.
A partner that is not a positive integer is ignored. The measured line and
the closed page name only NG, NS, DG, NXS, and NXT.

The isoelectric point and the net charge at pH 7.4 use the Bjellqvist pKa set
(Bjellqvist et al., Electrophoresis 15:529-539, 1994), including the
residue-specific terminal values used by ExPASy Compute pI/Mw. A chain whose
charge never crosses zero inside pH 0 to 14 has no isoelectric point to report
there, so the record names a blocker instead of storing one. No pass band
applies. A motif does not change the screen call, and the sequence is not stored.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

#: The published set named on the stored record.
PKA_SET = "bjellqvist_1994"
PH74 = 7.4

# Side-chain and terminal pKa values from Bjellqvist et al. 1994, as used by
# ExPASy Compute pI/Mw and Biopython's IsoelectricPoint. A residue absent from
# a terminal table keeps the default terminus.
_POSITIVE = {"K": 10.0, "R": 12.0, "H": 5.98}
_NEGATIVE = {"D": 4.05, "E": 4.45, "C": 9.0, "Y": 10.0}
_NTERM_DEFAULT = 7.50
_CTERM_DEFAULT = 3.55
_NTERM_SPECIFIC = {"A": 7.59, "M": 7.00, "S": 6.93, "P": 8.36, "T": 6.82, "V": 7.44, "E": 7.70}
_CTERM_SPECIFIC = {"D": 4.55, "E": 4.75}
_STANDARD = frozenset("ACDEFGHIKLMNPQRSTVWY")
_DEAMIDATION = frozenset({"NG", "NS"})
_ISOMERIZATION = frozenset({"DG"})
#: Motif names the measured line and the closed page may print.
MOTIF_NAMES = frozenset({"NG", "NS", "DG", "NXS", "NXT"})
#: Width the pI bracket is narrowed to before the midpoint is taken. The record
#: stores four decimals, so the bisection has to resolve well inside the 5e-5
#: that the fourth decimal rounds on. At 1e-4 it stopped at a bracket of
#: 5.34e-5 and the midpoint could sit 2.67e-5 from the true crossing, which put
#: the fourth stored decimal one unit out on about one chain in eight; 1e-6
#: still missed one in six hundred. This leaves a residual near 4e-10, for
#: sixteen more charge evaluations on a chain.
_PI_SPAN = 1e-9
#: Why a record carries no isoelectric point. The bisection reports where the
#: charge crosses zero, and a chain that keeps one sign from pH 0 to pH 14 has
#: no crossing inside the bracket: 101 arginines with no acidic side chain are
#: still net positive at pH 14, where the 100 the comment used to name are not
#: -- their charge there is -0.0099 and their pI 13.9956. Halving such a
#: bracket walks to the bound it started from, so the old code stored a flat
#: 14.0000 under the same key, with the same pKa-set name beside it, as if it
#: had been solved for.
PI_BLOCKER = "charge_does_not_cross_zero_in_ph_0_to_14"


def chemical_record(sequence: str, *, partners: Collection[object] | None) -> dict[str, object] | None:
    """Motif counts, free cysteine, pI, and net charge at pH 7.4.

    ``partners`` is None when the CIF does not name bonds. An empty collection
    means bonds were named and no cysteine on this chain has a disulfide
    partner. A partner that is not a positive integer, including a boolean, is
    ignored. A non-standard residue returns None. A chain whose charge never
    crosses zero between pH 0 and pH 14 carries ``isoelectric_point_blocker``
    and no ``isoelectric_point``; everything else on the record still stands.
    The sequence is not copied onto the record.
    """
    if not _standard(sequence):
        return None
    deamidation = _pair_sites(sequence, _DEAMIDATION)
    isomerization = _pair_sites(sequence, _ISOMERIZATION)
    glycosylation = _sequons(sequence)
    methionine = _residues(sequence, "M")
    tryptophan = _residues(sequence, "W")
    charge = net_charge(sequence, PH74)
    point = _isoelectric_block(sequence)
    record: dict[str, object] = {
        "deamidation_count": len(deamidation),
        "deamidation_sites": deamidation,
        "isomerization_count": len(isomerization),
        "isomerization_sites": isomerization,
        "glycosylation_count": len(glycosylation),
        "glycosylation_sites": glycosylation,
        "methionine_count": len(methionine),
        "methionine_residues": methionine,
        "tryptophan_count": len(tryptophan),
        "tryptophan_residues": tryptophan,
        **point,
        "net_charge_ph74": _round4(charge),
        "pka_set": PKA_SET,
    }
    record.update(_free_cysteine(sequence, partners))
    return record


def chemical_phrase(record: Mapping[str, object]) -> str | None:
    """The measured-line clause. A count of zero stays off the line.

    A motif name outside ``MOTIF_NAMES`` is omitted. The count stays.
    """
    parts: list[str] = []
    for label, prefix in (
        ("deamidation", "deamidation"),
        ("isomerization", "isomerization"),
        ("glycosylation", "glycosylation"),
    ):
        piece = _site_piece(label, record, prefix)
        if piece is not None:
            parts.append(piece)
    free = _free_piece(record)
    if free is not None:
        parts.append(free)
    for label, prefix in (("methionine", "methionine"), ("tryptophan", "tryptophan")):
        piece = _residue_piece(label, record, prefix)
        if piece is not None:
            parts.append(piece)
    if not parts:
        return None
    return "; ".join(parts)


def net_charge(sequence: str, ph: float) -> float:
    """Net charge at ``ph`` from the Bjellqvist 1994 pKa set."""
    if not _standard(sequence):
        raise ValueError("The sequence is not a standard protein chain.")
    n_pka = _NTERM_SPECIFIC.get(sequence[0], _NTERM_DEFAULT)
    c_pka = _CTERM_SPECIFIC.get(sequence[-1], _CTERM_DEFAULT)
    positive = 1.0 / (10 ** (ph - n_pka) + 1.0)
    negative = 1.0 / (10 ** (c_pka - ph) + 1.0)
    for letter in sequence:
        side = _POSITIVE.get(letter)
        if side is not None:
            positive += 1.0 / (10 ** (ph - side) + 1.0)
            continue
        side = _NEGATIVE.get(letter)
        if side is not None:
            negative += 1.0 / (10 ** (side - ph) + 1.0)
    return positive - negative


def isoelectric_point(sequence: str) -> float:
    """The pH where the Bjellqvist net charge crosses zero.

    The bracket is pH 0 to 14. A chain still positive at pH 14, or still
    negative at pH 0, has no crossing inside it, and halving that bracket
    returns the bound rather than a solved pH, so this refuses instead.
    """
    if not _standard(sequence):
        raise ValueError("The sequence is not a standard protein chain.")
    low = 0.0
    high = 14.0
    if net_charge(sequence, low) < 0.0 or net_charge(sequence, high) > 0.0:
        raise ValueError(PI_BLOCKER)
    while high - low > _PI_SPAN:
        mid = (low + high) / 2.0
        if net_charge(sequence, mid) > 0.0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _isoelectric_block(sequence: str) -> dict[str, object]:
    """The stored pI, or the named blocker when the charge has no crossing.

    The sequence is already known to be standard here, so the only refusal
    left is the bracket one.
    """
    try:
        return {"isoelectric_point": _round4(isoelectric_point(sequence))}
    except ValueError:
        return {"isoelectric_point_blocker": PI_BLOCKER}


def _standard(sequence: str) -> bool:
    return bool(sequence) and set(sequence) <= _STANDARD


def _round4(number: float) -> float:
    return float(f"{number:.4f}")


def _pair_sites(sequence: str, motifs: frozenset[str]) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    for index in range(len(sequence) - 1):
        pair = sequence[index : index + 2]
        if pair in motifs:
            found.append({"residue": index + 1, "motif": pair})
    return found


def _sequons(sequence: str) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    for index in range(len(sequence) - 2):
        if sequence[index] != "N" or sequence[index + 1] == "P":
            continue
        third = sequence[index + 2]
        if third == "S":
            found.append({"residue": index + 1, "motif": "NXS"})
        elif third == "T":
            found.append({"residue": index + 1, "motif": "NXT"})
    return found


def _residues(sequence: str, letter: str) -> list[int]:
    return [index + 1 for index, item in enumerate(sequence) if item == letter]


def _free_cysteine(sequence: str, partners: Collection[object] | None) -> dict[str, object]:
    if partners is None:
        return {"free_cysteine": "untested"}
    paired = {item for item in partners if isinstance(item, int) and not isinstance(item, bool) and item >= 1}
    residues = [index + 1 for index, item in enumerate(sequence) if item == "C" and index + 1 not in paired]
    return {
        "free_cysteine": "reported",
        "free_cysteine_count": len(residues),
        "free_cysteine_residues": residues,
    }


def _number_text(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if isinstance(item, int) and not isinstance(item, bool)]


def _count(record: Mapping[str, object], key: str) -> int | None:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _site_piece(label: str, record: Mapping[str, object], prefix: str) -> str | None:
    count = _count(record, f"{prefix}_count")
    if count is None:
        return None
    sites = record.get(f"{prefix}_sites")
    details: list[str] = []
    if isinstance(sites, list):
        for site in sites:
            if not isinstance(site, dict):
                continue
            residue = site.get("residue")
            motif = site.get("motif")
            if isinstance(residue, bool) or not isinstance(residue, int):
                continue
            if not isinstance(motif, str) or motif not in MOTIF_NAMES:
                continue
            details.append(f"residue {residue} motif {motif}")
    if details:
        return f"{label} {count}, " + ", ".join(details)
    return f"{label} {count}"


def _residue_piece(label: str, record: Mapping[str, object], prefix: str) -> str | None:
    count = _count(record, f"{prefix}_count")
    if count is None:
        return None
    numbers = _number_text(record.get(f"{prefix}_residues"))
    if not numbers:
        return f"{label} {count}"
    word = "residue" if len(numbers) == 1 else "residues"
    return f"{label} {count}, {word} {', '.join(numbers)}"


def _free_piece(record: Mapping[str, object]) -> str | None:
    if record.get("free_cysteine") != "reported":
        return None
    count = _count(record, "free_cysteine_count")
    if count is None:
        return None
    numbers = _number_text(record.get("free_cysteine_residues"))
    if not numbers:
        return f"free cysteine {count}"
    word = "residue" if len(numbers) == 1 else "residues"
    return f"free cysteine {count}, {word} {', '.join(numbers)}"
