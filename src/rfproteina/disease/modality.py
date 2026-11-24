"""Drug-type reports for one finished design.

A protein or peptide screen lists furin cleavage sites (Arg-X-Lys/Arg-Arg),
Asp-Pro bonds, N-terminal glutamine, nonstandard letters by position, the
longest hydrophobic and same-residue runs, the Ikai aliphatic index, the
Shannon entropy of the residue composition with the commonest residue's share,
the average molecular weight, the Pace 1995 extinction at 280 nm, the N-end
rule half-life, the residue and atomic composition, Kyte-Doolittle spans, and
the AGGRESCAN 2007 normalized sum.
A nucleic screen lists GC fraction, CpG sites, and the longest G run. A ligand
screen lists a formula weight when the structure names one non-polymer
component. A missing weight is calculated from the chemical formula when every
element is known, and left out when that formula does not parse. Nothing here
is an axis, and nothing here changes the call. The sequence is not stored.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

#: Motif names the measured line and the closed page may print.
TRIAGE_MOTIFS = frozenset({"DP", "RXKR", "RXRR"})
#: Kyte-Doolittle residues with a positive hydropathy. Same scale as GRAVY.
HYDROPHOBIC = frozenset("ACFILMV")

_STANDARD = frozenset("ACDEFGHIKLMNPQRSTVWY")
_NUCLEOTIDE = frozenset("ACGTU")
_WATER = frozenset({"HOH", "WAT", "DOD", "H2O"})
_KIND_ALIAS = {
    "small_molecule": "ligand",
    "non_polymer": "ligand",
    "dna": "nucleic",
    "rna": "nucleic",
    "nucleic_acid": "nucleic",
    "polypeptide": "protein",
}
_POLYPEPTIDE = frozenset({"protein", "peptide"})
_WEIGHT_SOURCE = "iupac_conventional"
# Free amino acids. The chain mass subtracts one water for each peptide bond.
_RESIDUE_FORMULA = {
    "A": "C3 H7 N O2",
    "R": "C6 H14 N4 O2",
    "N": "C4 H8 N2 O3",
    "D": "C4 H7 N O4",
    "C": "C3 H7 N O2 S",
    "Q": "C5 H10 N2 O3",
    "E": "C5 H9 N O4",
    "G": "C2 H5 N O2",
    "H": "C6 H9 N3 O2",
    "I": "C6 H13 N O2",
    "L": "C6 H13 N O2",
    "K": "C6 H14 N2 O2",
    "M": "C5 H11 N O2 S",
    "F": "C9 H11 N O2",
    "P": "C5 H9 N O2",
    "S": "C3 H7 N O3",
    "T": "C4 H9 N O3",
    "W": "C11 H12 N2 O2",
    "Y": "C9 H11 N O3",
    "V": "C5 H11 N O2",
}
# Ikai, J. Biochem. 1980, 88:1895. Relative side-chain volumes of Val and of Ile/Leu.
_ALIPHATIC_VAL = Decimal("2.9")
_ALIPHATIC_ILE_LEU = Decimal("3.9")
# Pace et al., Protein Sci. 1995, 4:2411. Edelhoch 1967; Gill and von Hippel 1989.
# Molar extinction at 280 nm in water. Cystine is floor of the cysteine count over two.
_EXT_TYR = 1490
_EXT_TRP = 5500
_EXT_CYSTINE = 125
_EXTINCTION_SOURCE = "pace_1995"
_ATOM_ORDER = ("C", "H", "N", "O", "S")
# Kyte and Doolittle, J. Mol. Biol. 1982, 157:105. Same scale as GRAVY.
# A window of 19 residues whose mean hydropathy is greater than 1.6 is membrane-spanning.
_KD_WINDOW = 19
_KD_CUT = Decimal("1.6")
_TRANSMEMBRANE_METHOD = "kyte_doolittle_1982"
# Sánchez de Groot et al., BMC Struct. Biol. 2005, 5:18, Table 1. Relative
# aggregation propensities used by AGGRESCAN. Conchillo-Solé et al., BMC
# Bioinformatics 2007, 8:65. Window 5, 7, 9, or 11 by length. The N-terminal
# virtual residue is the mean of Lys and Arg, and the C-terminal virtual
# residue is the mean of Asp and Glu. The hot-spot threshold is the value
# stated in the 2007 help file.
_AGGRESCAN_METHOD = "aggrescan_2007"
_AGGREGATION_SCALE = "sanchez_de_groot_2005"
_HST = Decimal("-0.02")
_HOTSPOT_LENGTH = 5
_A3V: dict[str, Decimal] = {
    "I": Decimal("1.822"),
    "F": Decimal("1.754"),
    "V": Decimal("1.594"),
    "L": Decimal("1.380"),
    "Y": Decimal("1.159"),
    "W": Decimal("1.037"),
    "M": Decimal("0.910"),
    "C": Decimal("0.604"),
    "A": Decimal("-0.036"),
    "T": Decimal("-0.159"),
    "S": Decimal("-0.294"),
    "P": Decimal("-0.334"),
    "G": Decimal("-0.535"),
    "K": Decimal("-0.931"),
    "H": Decimal("-1.033"),
    "Q": Decimal("-1.231"),
    "R": Decimal("-1.240"),
    "N": Decimal("-1.302"),
    "E": Decimal("-1.412"),
    "D": Decimal("-1.836"),
}
_VIRTUAL_BASIC = (_A3V["K"] + _A3V["R"]) / 2
_VIRTUAL_ACIDIC = (_A3V["D"] + _A3V["E"]) / 2
_KD: dict[str, str] = {
    "A": "1.8",
    "R": "-4.5",
    "N": "-3.5",
    "D": "-3.5",
    "C": "2.5",
    "Q": "-3.5",
    "E": "-3.5",
    "G": "-0.4",
    "H": "-3.2",
    "I": "4.5",
    "L": "3.8",
    "K": "-3.9",
    "M": "1.9",
    "F": "2.8",
    "P": "-1.6",
    "S": "-0.8",
    "T": "-0.7",
    "W": "-0.9",
    "Y": "-1.3",
    "V": "4.2",
}
# N-end rule words from the ExPASy ProtParam table (Gasteiger et al., Proteomics
# Protocols Handbook 2005, pp. 571-607). Mammalian, yeast, then E. coli.
# Bachmair, Finley and Varshavsky, Science 1986, 234:179; Gonda et al., J. Biol.
# Chem. 1989, 264:16700; Tobias et al., Science 1991, 254:1374. Proline has no
# published E. coli estimate, so that field is omitted rather than stored as "?".
_HALF_LIFE_SOURCE = "n_end_rule"
_HALF_LIFE: dict[str, tuple[str, str, str | None]] = {
    "A": ("4.4 hour", ">20 hour", ">10 hour"),
    "R": ("1 hour", "2 min", "2 min"),
    "N": ("1.4 hour", "3 min", ">10 hour"),
    "D": ("1.1 hour", "3 min", ">10 hour"),
    "C": ("1.2 hour", ">20 hour", ">10 hour"),
    "Q": ("0.8 hour", "10 min", ">10 hour"),
    "E": ("1 hour", "30 min", ">10 hour"),
    "G": ("30 hour", ">20 hour", ">10 hour"),
    "H": ("3.5 hour", "10 min", ">10 hour"),
    "I": ("20 hour", "30 min", ">10 hour"),
    "L": ("5.5 hour", "3 min", "2 min"),
    "K": ("1.3 hour", "3 min", "2 min"),
    "M": ("30 hour", ">20 hour", ">10 hour"),
    "F": ("1.1 hour", "3 min", "2 min"),
    "P": (">20 hour", ">20 hour", None),
    "S": ("1.9 hour", ">20 hour", ">10 hour"),
    "T": ("7.2 hour", ">20 hour", ">10 hour"),
    "W": ("2.8 hour", "3 min", "2 min"),
    "Y": ("2.8 hour", "10 min", "2 min"),
    "V": ("100 hour", ">20 hour", ">10 hour"),
}
#: Published half-life phrases the closed page may print.
HALF_LIFE_PHRASES = frozenset(phrase for row in _HALF_LIFE.values() for phrase in row if phrase is not None)
# Conventional single values from the IUPAC standard atomic weights. C, H, N,
# O, P, S and most of the table are the 2005 values (Wieser, Pure Appl. Chem.
# 2006, 78, 2051); about twenty entries carry a revision later than 2005, among
# them Pt 195.084, Na 22.98976928, Mn 54.938045, Co 58.933195 and Au
# 196.966569. Naming 2005 alone was wrong by up to 0.006 Da on those elements,
# which a three-decimal weight shows. The uncertainty digit is omitted. A whole
# number is the mass number of the longest-lived isotope. These single values
# reproduce a deposited formula weight to the stored digit except where the sum
# lands exactly on a half, which rounds up. Cn through Og are the current names
# of the 2005 temporary symbols.
#
# This table is deliberately NOT the current IUPAC one, and seven entries are
# where that shows: Zn 65.409, Ge 72.64, Se 78.96, Mo 95.94, Cd 112.411,
# Hg 200.59 and Yb 173.04, against 65.38, 72.630, 78.971, 95.95, 112.414,
# 200.592 and 173.045 in IUPAC's 2024 table. The weights here are the vintage
# the Chemical Component Dictionary itself used, and the deposited files this
# module reads carry that arithmetic: a deposited ZINC ION states 65.409. The
# point of the table is to reproduce a deposited formula weight, so that a
# computed one can be compared with it, and on the components checked every one
# of eleven reproduces exactly as the table stands while five stop doing so
# with the 2024 values. Updating these seven to the current table would
# therefore make this module disagree with the data it is checking. Leave them,
# and read ``_WEIGHT_SOURCE`` as naming this vintage rather than today's.
# tests/test_disease_atomic_weights.py holds both halves of that statement.
_ATOMIC_WEIGHT: dict[str, str] = {
    "H": "1.00794",
    "He": "4.002602",
    "Li": "6.941",
    "Be": "9.012182",
    "B": "10.811",
    "C": "12.0107",
    "N": "14.0067",
    "O": "15.9994",
    "F": "18.9984032",
    "Ne": "20.1797",
    "Na": "22.98976928",
    "Mg": "24.3050",
    "Al": "26.9815386",
    "Si": "28.0855",
    "P": "30.973762",
    "S": "32.065",
    "Cl": "35.453",
    "Ar": "39.948",
    "K": "39.0983",
    "Ca": "40.078",
    "Sc": "44.955912",
    "Ti": "47.867",
    "V": "50.9415",
    "Cr": "51.9961",
    "Mn": "54.938045",
    "Fe": "55.845",
    "Co": "58.933195",
    "Ni": "58.6934",
    "Cu": "63.546",
    "Zn": "65.409",
    "Ga": "69.723",
    "Ge": "72.64",
    "As": "74.92160",
    "Se": "78.96",
    "Br": "79.904",
    "Kr": "83.798",
    "Rb": "85.4678",
    "Sr": "87.62",
    "Y": "88.90585",
    "Zr": "91.224",
    "Nb": "92.90638",
    "Mo": "95.94",
    "Tc": "98",
    "Ru": "101.07",
    "Rh": "102.90550",
    "Pd": "106.42",
    "Ag": "107.8682",
    "Cd": "112.411",
    "In": "114.818",
    "Sn": "118.710",
    "Sb": "121.760",
    "Te": "127.60",
    "I": "126.90447",
    "Xe": "131.293",
    "Cs": "132.9054519",
    "Ba": "137.327",
    "La": "138.90547",
    "Ce": "140.116",
    "Pr": "140.90765",
    "Nd": "144.242",
    "Pm": "145",
    "Sm": "150.36",
    "Eu": "151.964",
    "Gd": "157.25",
    "Tb": "158.92535",
    "Dy": "162.500",
    "Ho": "164.93032",
    "Er": "167.259",
    "Tm": "168.93421",
    "Yb": "173.04",
    "Lu": "174.967",
    "Hf": "178.49",
    "Ta": "180.94788",
    "W": "183.84",
    "Re": "186.207",
    "Os": "190.23",
    "Ir": "192.217",
    "Pt": "195.084",
    "Au": "196.966569",
    "Hg": "200.59",
    "Tl": "204.3833",
    "Pb": "207.2",
    "Bi": "208.98040",
    "Po": "209",
    "At": "210",
    "Rn": "222",
    "Fr": "223",
    "Ra": "226",
    "Ac": "227",
    "Th": "232.03806",
    "Pa": "231.03588",
    "U": "238.02891",
    "Np": "237",
    "Pu": "244",
    "Am": "243",
    "Cm": "247",
    "Bk": "247",
    "Cf": "251",
    "Es": "252",
    "Fm": "257",
    "Md": "258",
    "No": "259",
    "Lr": "262",
    "Rf": "267",
    "Db": "268",
    "Sg": "271",
    "Bh": "272",
    "Hs": "270",
    "Mt": "276",
    "Ds": "281",
    "Rg": "280",
    "Cn": "285",
    "Nh": "284",
    "Fl": "289",
    "Mc": "288",
    "Lv": "293",
    "Og": "294",
    "Uub": "285",
    "Uut": "284",
    "Uuq": "289",
    "Uup": "288",
    "Uuh": "293",
    "Uuo": "294",
}


def modality_record(
    kind: str,
    sequence: str | None = None,
    *,
    components: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, object] | None:
    """The report for this binder kind. An unrecognized kind returns None.

    ``components`` are ``_chem_comp`` rows. A ligand report uses them and leaves
    a polymer sequence unread. Protein, peptide, and nucleic reports use the
    designed chain and leave the component table unread. Whitespace in a chain
    is ignored. A nonstandard protein or peptide letter is counted and the rest
    of that chain is still scanned. An unresolved chain reports only the
    contribution of its known residues to the whole-chain aliphatic index.
    The molecular weight, the extinction, and the atomic composition are left
    out when any letter has no residue mass. The half-life follows the
    N-terminal residue and is left out when that residue is nonstandard.
    A Kyte-Doolittle span is left out when any letter has no hydropathy.
    A ligand with one component and no positive
    formula weight uses the chemical formula when every element is known.
    Otherwise that report is left out. The formula string is not stored.
    """
    canonical = _kind(kind)
    if canonical in _POLYPEPTIDE:
        return _polypeptide(canonical, sequence or "")
    if canonical == "nucleic":
        return _oligo(sequence or "")
    if canonical == "ligand":
        return _ligand(components or ())
    return None


def modality_phrase(record: Mapping[str, object]) -> str | None:
    """The measured-line clause. A count of zero stays off the line.

    A motif name outside ``TRIAGE_MOTIFS`` is omitted. The count stays.
    A residue number below 1 is omitted. The count stays. GC fraction, run
    lengths, the aliphatic index, molecular weight, formula weight, extinction,
    half-life, composition, atomic counts, entropy, and transmembrane spans stay
    off this line.
    """
    kind = record.get("binder_kind")
    if kind in _POLYPEPTIDE:
        return _polypeptide_phrase(record)
    if kind == "nucleic":
        return _cpg_piece(record)
    return None


def component_label(raw: object) -> str | None:
    """A component id the page may print. A sequence-shaped id is omitted."""
    if not isinstance(raw, str):
        return None
    text = _unquote(raw.strip())
    if text in {"", ".", "?"} or len(text) > 10 or any(character.isspace() for character in text):
        return None
    letters = text.upper()
    if len(letters) >= 9 and set(letters) <= _STANDARD:
        return None
    return text


def _kind(kind: str) -> str:
    key = kind.strip().casefold().replace("-", "_").replace(" ", "_")
    return _KIND_ALIAS.get(key, key)


def _complexity(chain: str) -> dict[str, object]:
    """Shannon entropy over the residues, and the share the commonest one takes.

    The longest run already here catches ``AAAAAAAA``. It does not catch
    ``AVAVAVAVAV`` or a chain that is half alanine with the alanines spread
    out, which are low-complexity in the way that matters for expression and
    for a design that has collapsed toward one residue. Entropy counts every
    position rather than the longest unbroken one.

    The unit is bits, so the scale is fixed and readable: a chain drawing
    evenly on all twenty residues approaches 4.32, one residue alone is 0.
    """
    counts: dict[str, int] = {}
    for letter in chain:
        counts[letter] = counts.get(letter, 0) + 1
    total = len(chain)
    # Summed as p*log2(1/p) rather than as -p*log2(p). The two are the same
    # number, but the negated form returns -0.0 for a chain of one residue, and
    # JSON writes that sign out: a reader met an entropy of "-0.0" on a
    # quantity that has no negative side.
    entropy = sum((count / total) * math.log2(total / count) for count in counts.values())
    # Sorted by count then letter, so one of two residues tied at the top is
    # not chosen by whichever happened to be seen first.
    commonest = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return {
        "sequence_entropy_bits": float(f"{entropy:.4f}"),
        "most_common_residue": commonest[0],
        "most_common_fraction": float(f"{commonest[1] / total:.4f}"),
        "distinct_residues": len(counts),
    }


def _polypeptide(kind: str, sequence: str) -> dict[str, object] | None:
    chain = _letters(sequence)
    if not chain:
        return None
    furin = _furin_sites(chain)
    cleavage = _acid_sites(chain)
    unusual = [index + 1 for index, letter in enumerate(chain) if letter not in _STANDARD]
    run, residue, start = _homopolymer(chain)
    hydrophobic, hydrophobic_start = _hydrophobic_run(chain)
    record: dict[str, object] = {
        "binder_kind": kind,
        "length": len(chain),
        "furin_count": len(furin),
        "furin_sites": furin,
        "acid_cleavage_count": len(cleavage),
        "acid_cleavage_sites": cleavage,
        "n_terminal_glutamine": chain[0] == "Q",
        "nonstandard_count": len(unusual),
        "nonstandard_residues": unusual,
        "longest_homopolymer": run,
        "homopolymer_residue": residue,
        "homopolymer_start": start,
        "longest_hydrophobic_run": hydrophobic,
        "hydrophobic_run_start": hydrophobic_start,
        **_complexity(chain),
    }
    if unusual:
        if len(unusual) < len(chain):
            record["aliphatic_index_lower_bound"] = _aliphatic_index(chain)
            record["aliphatic_index_basis"] = "known_residues_lower_bound"
    else:
        record["aliphatic_index"] = _aliphatic_index(chain)
    mass = _molecular_weight(chain)
    if mass is not None:
        record["molecular_weight"] = mass
        record["molecular_weight_source"] = _WEIGHT_SOURCE
    extinction = _extinction(chain, mass)
    if extinction is not None:
        record.update(extinction)
    half_life = _half_life(chain)
    if half_life is not None:
        record.update(half_life)
    composition = _residue_composition(chain)
    if composition is not None:
        record["residue_composition"] = composition
    atoms = _atom_counts(chain)
    if atoms is not None:
        record["atom_counts"] = atoms
    membrane = _transmembrane(chain)
    if membrane is not None:
        record.update(membrane)
    aggregation = _aggrescan(chain)
    if aggregation is not None:
        record["aggrescan"] = aggregation
    else:
        record["aggrescan_blocker"] = "unsupported_residue" if unusual else "sequence_shorter_than_window"
    return record


def _oligo(sequence: str) -> dict[str, object] | None:
    chain = _letters(sequence)
    if not chain or not set(chain) <= _NUCLEOTIDE:
        return None
    positions = [index + 1 for index in range(len(chain) - 1) if chain[index : index + 2] == "CG"]
    g_run, g_start = _letter_run(chain, "G")
    gc = (chain.count("G") + chain.count("C")) / len(chain)
    return {
        "binder_kind": "nucleic",
        "length": len(chain),
        "gc_fraction": float(f"{gc:.4f}"),
        "cpg_count": len(positions),
        "cpg_positions": positions,
        "longest_g_run": g_run,
        "g_run_start": g_start,
    }


def _ligand(components: Sequence[Mapping[str, str]]) -> dict[str, object] | None:
    named = [row for row in components if _is_ligand_component(row)]
    if not named:
        return None
    if len(named) > 1:
        return {"binder_kind": "ligand", "composition": "ambiguous", "component_count": len(named)}
    row = named[0]
    weight = _formula_weight(row)
    calculated = False
    if weight is None:
        weight = _mass_from_formula(_cell(row, "_chem_comp.formula"))
        calculated = True
    if weight is None:
        return None
    record: dict[str, object] = {"binder_kind": "ligand", "formula_weight": weight}
    if calculated:
        record["formula_weight_source"] = _WEIGHT_SOURCE
    component_id = component_label(row.get("_chem_comp.id", ""))
    if component_id is not None:
        record["component_id"] = component_id
    return record


def _is_ligand_component(row: Mapping[str, str]) -> bool:
    kind = _cell(row, "_chem_comp.type").casefold()
    if kind != "non-polymer":
        return False
    name = _cell(row, "_chem_comp.id").upper()
    return name not in _WATER


def identified_component_weight(components: Sequence[Mapping[str, str]]) -> dict:
    """Return a formula weight for one named non-water component, with its source and any disagreement."""
    rows = [row for row in components if _is_ligand_component(row)]
    if len(rows) != 1:
        return {}
    row = rows[0]
    record = _ligand(rows)
    if not record or "component_id" not in record:
        return {}
    derived = "formula_weight_source" in record
    result = {
        "component_id": record["component_id"],
        "formula_weight_Da": record["formula_weight"],
        "formula_weight_source": _WEIGHT_SOURCE if derived else "_chem_comp.formula_weight",
    }
    reported = _cell(row, "_chem_comp.formula_weight")
    formula = _cell(row, "_chem_comp.formula")
    if reported in {"", ".", "?"} or formula in {"", ".", "?"}:
        return result
    if derived:
        result["input_warning"] = "invalid_supplied_formula_weight"
        return result
    mass = _mass_from_formula(formula)
    if mass is None:
        result["input_warning"] = "formula_not_interpretable"
        return result
    # The formula calculation reports three decimal places. Compare at the less precise scale.
    stated = Decimal(reported)
    quantum = Decimal(1).scaleb(max(-3, stated.as_tuple().exponent))
    try:
        agrees = stated.quantize(quantum, ROUND_HALF_UP) == Decimal(str(mass)).quantize(quantum, ROUND_HALF_UP)
    except InvalidOperation:
        agrees = False
    if not agrees:
        result.update(input_warning="formula_weights_disagree", calculated_formula_weight_Da=mass)
    return result


def _formula_weight(row: Mapping[str, str]) -> float | None:
    text = _cell(row, "_chem_comp.formula_weight")
    if text in {"", ".", "?"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    rounded = float(f"{number:.4f}")
    return rounded if rounded > 0 else None


def _mass_from_formula(formula: str) -> float | None:
    """Formula weight from the ``_ATOMIC_WEIGHT`` table, or None.

    That table is the Chemical Component Dictionary's own vintage of the IUPAC
    conventional weights, which is mostly 2005 and not only 2005: see the
    comment above it for the entries that differ and why none of them moves.

    Each whitespace-separated token is one element and an optional count.
    The element symbol is read with its first letter upper and the rest lower,
    so ``ZN`` is zinc. A trailing whole number is a charge and is not mass.
    Parentheses, a dot hydrate, a signed charge, an unknown element, a count
    that is not a base-ten numeral, or a zero count leave the whole formula
    unread. No partial sum is returned. The sum is rounded to three decimal
    places.
    """
    total = _formula_total(formula)
    if total is None:
        return None
    return _stored_mass(total, Decimal("0.001"))


def _formula_total(formula: str) -> Decimal | None:
    """Unrounded mass of one formula. An unreadable formula returns None."""
    if formula in {"", ".", "?"} or any(mark in formula for mark in ".()?[]+-,/*="):
        return None
    parts = formula.split()
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    if not parts:
        return None
    total = Decimal(0)
    for part in parts:
        counted = _element_count(part)
        if counted is None:
            return None
        symbol, count = counted
        total += Decimal(_ATOMIC_WEIGHT[symbol]) * count
    if total <= 0:
        return None
    return total


def _stored_mass(total: Decimal, quantum: Decimal) -> float | None:
    try:
        rounded = total.quantize(quantum, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        # A formula with an absurd count -- C followed by twenty-five digits --
        # sums past the decimal context's precision, and quantize then raises
        # rather than returning a number. The exception is an ArithmeticError,
        # which the screen's own handler does not catch, so it travelled out of
        # the report and ended the whole screen for that run. No weight is a
        # complete answer here; no screen is not.
        return None
    number = float(f"{rounded:.4f}")
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _aliphatic_index(chain: str) -> float:
    """Ikai 1980 aliphatic index. Mole percent of Ala, Val, Ile, and Leu."""
    length = Decimal(len(chain))
    alanine = Decimal(chain.count("A")) / length * 100
    valine = Decimal(chain.count("V")) / length * 100
    branched = Decimal(chain.count("I") + chain.count("L")) / length * 100
    index = alanine + _ALIPHATIC_VAL * valine + _ALIPHATIC_ILE_LEU * branched
    rounded = index.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    return float(f"{rounded:.4f}")


def _molecular_weight(chain: str) -> float | None:
    """Average mass. One water is removed for each peptide bond.

    A letter with no residue formula leaves the mass out. No partial sum is
    returned.
    """
    if any(letter not in _RESIDUE_FORMULA for letter in chain):
        return None
    water = _formula_total("H2 O")
    if water is None:
        return None
    total = Decimal(0)
    for letter in chain:
        piece = _formula_total(_RESIDUE_FORMULA[letter])
        if piece is None:
            return None
        total += piece
    total -= water * (len(chain) - 1)
    return _stored_mass(total, Decimal("0.0001"))


def _extinction(chain: str, mass: float | None) -> dict[str, object] | None:
    """Pace 1995 extinction at 280 nm. Absorbance is stored only when it is above zero.

    Cystine is one disulfide under the assumption that every pair of cysteines
    is paired. An odd cysteine adds nothing. A nonstandard letter leaves the
    whole report out. The absorbance uses the already rounded molecular weight.
    """
    if any(letter not in _STANDARD for letter in chain):
        return None
    reduced = chain.count("Y") * _EXT_TYR + chain.count("W") * _EXT_TRP
    paired = reduced + (chain.count("C") // 2) * _EXT_CYSTINE
    record: dict[str, object] = {
        "extinction_280_reduced": reduced,
        "extinction_280_cystine": paired,
        "extinction_source": _EXTINCTION_SOURCE,
    }
    if mass is None or mass <= 0:
        return record
    if reduced > 0:
        absorbance = _absorbance(reduced, mass)
        if absorbance is not None:
            record["absorbance_280_reduced"] = absorbance
    if paired > 0:
        absorbance = _absorbance(paired, mass)
        if absorbance is not None:
            record["absorbance_280_cystine"] = absorbance
    return record


def _absorbance(extinction: int, mass: float) -> float | None:
    """Extinction divided by the stored molecular weight, at four decimal places."""
    try:
        value = (Decimal(extinction) / Decimal(f"{mass:.4f}")).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    except (InvalidOperation, ZeroDivisionError):
        return None
    number = float(f"{value:.4f}")
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _half_life(chain: str) -> dict[str, str] | None:
    """N-end rule words for the N-terminal residue. Proline omits E. coli.

    A nonstandard N-terminus leaves the estimate out. The rest of the chain
    does not enter it.
    """
    row = _HALF_LIFE.get(chain[0])
    if row is None:
        return None
    mammalian, yeast, ecoli = row
    record = {
        "half_life_source": _HALF_LIFE_SOURCE,
        "half_life_mammalian": mammalian,
        "half_life_yeast": yeast,
    }
    if ecoli is not None:
        record["half_life_ecoli"] = ecoli
    return record


def _residue_composition(chain: str) -> dict[str, int] | None:
    """Counts of the standard letters that occur. An empty count is left out."""
    found: dict[str, int] = {}
    for letter in chain:
        if letter in _STANDARD:
            found[letter] = found.get(letter, 0) + 1
    if not found:
        return None
    return {letter: found[letter] for letter in "ACDEFGHIKLMNPQRSTVWY" if letter in found}


def _atom_counts(chain: str) -> dict[str, int] | None:
    """C, H, N, O, and S. One water is removed for each peptide bond.

    A letter with no residue formula leaves the counts out. A count that would
    go negative leaves them out. No partial sum is returned. A zero is omitted.
    """
    if any(letter not in _RESIDUE_FORMULA for letter in chain):
        return None
    totals = dict.fromkeys(_ATOM_ORDER, 0)
    for letter in chain:
        counts = _residue_atoms(_RESIDUE_FORMULA[letter])
        if counts is None:
            return None
        for element, count in counts.items():
            if element not in totals:
                return None
            totals[element] += count
    bonds = len(chain) - 1
    totals["H"] -= 2 * bonds
    totals["O"] -= bonds
    if any(count < 0 for count in totals.values()):
        return None
    return {element: totals[element] for element in _ATOM_ORDER if totals[element] > 0}


def _residue_atoms(formula: str) -> dict[str, int] | None:
    """Element counts in one free-residue formula."""
    counts: dict[str, int] = {}
    for part in formula.split():
        counted = _element_count(part)
        if counted is None:
            return None
        symbol, count = counted
        counts[symbol] = counts.get(symbol, 0) + count
    return counts or None


def _aggrescan(chain: str) -> dict[str, object] | None:
    """AGGRESCAN 2007 profile. A short chain or a nonstandard letter is left out.

    Off-centre residues copy the first or last window centre. A hot spot is
    five or more residues whose window average is strictly above -0.02, and
    proline breaks the run. The normalized sum is not a pass band.
    """
    if any(letter not in _A3V for letter in chain):
        return None
    window = _aggrescan_window(len(chain))
    if len(chain) < window:
        return None
    profile = _a4v_profile(chain, window)
    if len(profile) != len(chain):
        return None
    total = sum(profile, start=Decimal(0))
    composition = sum((_A3V[letter] for letter in chain), start=Decimal(0))
    length = Decimal(len(chain))
    normalized = _four_places(total / length * Decimal(100))
    sequence_sum = _four_places(total)
    average = _four_places(composition / length)
    threshold = _four_places(_HST)
    if normalized is None or sequence_sum is None or average is None or threshold is None:
        return None
    spans = _aggregation_hotspots(chain, profile)
    return {
        "method": _AGGRESCAN_METHOD,
        "scale_source": _AGGREGATION_SCALE,
        "window": window,
        "hot_spot_threshold": threshold,
        "na4vss": normalized,
        "a4vss": sequence_sum,
        "a3vsa": average,
        "hotspot_count": len(spans),
        "hotspots": spans,
    }


def _aggrescan_window(length: int) -> int:
    """5 at or below 75 residues, 7 at or below 175, 9 at or below 300, else 11."""
    if length <= 75:
        return 5
    if length <= 175:
        return 7
    if length <= 300:
        return 9
    return 11


def _a4v_profile(chain: str, window: int) -> list[Decimal]:
    """Window averages, with one virtual residue on each side of the chain."""
    radius = (window - 1) // 2
    values = [_VIRTUAL_BASIC, *(_A3V[letter] for letter in chain), _VIRTUAL_ACIDIC]
    width = Decimal(window)
    profile = [Decimal(0)] * len(chain)
    first: int | None = None
    last = 0
    for index in range(len(chain)):
        center = index + 1
        left = center - radius
        right = center + radius
        if left < 0 or right >= len(values):
            continue
        profile[index] = sum(values[left : right + 1], start=Decimal(0)) / width
        if first is None:
            first = index
        last = index
    if first is None:
        return []
    for index in range(first):
        profile[index] = profile[first]
    for index in range(last + 1, len(chain)):
        profile[index] = profile[last]
    return profile


def _aggregation_hotspots(chain: str, profile: Sequence[Decimal]) -> list[dict[str, int]]:
    """Runs of at least five residues above the threshold and free of proline."""
    found: list[dict[str, int]] = []
    start: int | None = None
    for index, letter in enumerate(chain):
        if profile[index] > _HST and letter != "P":
            if start is None:
                start = index
            continue
        if start is not None and index - start >= _HOTSPOT_LENGTH:
            found.append({"residue_start": start + 1, "residue_end": index})
        start = None
    if start is not None and len(chain) - start >= _HOTSPOT_LENGTH:
        found.append({"residue_start": start + 1, "residue_end": len(chain)})
    return found


def _four_places(value: Decimal) -> float | None:
    """Four decimal places. A negative score stays, because this scale crosses zero."""
    try:
        rounded = value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None
    number = float(f"{rounded:.4f}")
    if not math.isfinite(number):
        return None
    return number


def membrane_axis(sequence: str | None) -> dict[str, object]:
    """The ``membrane`` axis, measured from the span scan already computed here.

    ``PEPTIDE_ONLY_AXES`` has carried ``membrane`` since the classes were
    defined, and the peptide ``sequence`` bar requires it, but nothing
    measured it: the number was computed for the drug-type report and then
    thrown away, so a peptide binder could not reach ``screened_clear`` on its
    own bar however hydrophilic it was.

    A span is a 19-residue Kyte-Doolittle window whose mean hydropathy is
    strictly above 1.6, merged with its overlapping neighbours. One is a hit.
    None, over a chain long enough to hold a window, is a clear. A chain
    shorter than the window is refused rather than cleared -- it cannot hold a
    span by this criterion, which is not the same as having none -- and so is
    one carrying a letter with no hydropathy value.
    """
    chain = _letters(sequence or "")
    if not chain:
        return _membrane_state("untested", None, "sequence_absent")
    if any(letter not in _KD for letter in chain):
        return _membrane_state("untested", None, "hydropathy_unknown")
    if len(chain) < _KD_WINDOW:
        return _membrane_state("untested", None, "chain_shorter_than_window")
    spans = _membrane_spans(chain)
    record = _membrane_state("hit" if spans else "clear", len(spans), None)
    record["transmembrane_spans"] = spans
    record["window_residues"] = _KD_WINDOW
    record["method"] = _TRANSMEMBRANE_METHOD
    return record


def _membrane_state(state: str, measurement: int | None, blocker: str | None) -> dict[str, object]:
    return {"state": state, "measurement": measurement, "blocker": blocker}


def _transmembrane(chain: str) -> dict[str, object] | None:
    """Kyte-Doolittle spans. A chain with no hydropathy for a letter is left out.

    A chain shorter than 19 residues stores a count of zero. A window mean equal
    to 1.6 is not a span. Overlapping windows that clear the cut merge.
    """
    if any(letter not in _KD for letter in chain):
        return None
    spans = _membrane_spans(chain) if len(chain) >= _KD_WINDOW else []
    return {
        "transmembrane_count": len(spans),
        "transmembrane_spans": spans,
        "transmembrane_method": _TRANSMEMBRANE_METHOD,
    }


def _membrane_spans(chain: str) -> list[dict[str, object]]:
    """Merged windows whose mean hydropathy is strictly greater than 1.6."""
    values = [Decimal(_KD[letter]) for letter in chain]
    window = _KD_WINDOW
    total = sum(values[:window], start=Decimal(0))
    passing: list[tuple[int, int, Decimal]] = []
    if total / window > _KD_CUT:
        passing.append((1, window, total / window))
    for start in range(1, len(values) - window + 1):
        total += values[start + window - 1] - values[start - 1]
        if total / window > _KD_CUT:
            passing.append((start + 1, start + window, total / window))
    return _merge_windows(passing)


def _merge_windows(windows: list[tuple[int, int, Decimal]]) -> list[dict[str, object]]:
    if not windows:
        return []
    start, end, peak = windows[0]
    merged: list[dict[str, object]] = []
    for next_start, next_end, mean in windows[1:]:
        if next_start <= end:
            end = max(end, next_end)
            if mean > peak:
                peak = mean
            continue
        merged.append(_span_record(start, end, peak))
        start, end, peak = next_start, next_end, mean
    merged.append(_span_record(start, end, peak))
    return merged


def _span_record(start: int, end: int, peak: Decimal) -> dict[str, object]:
    rounded = peak.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    return {
        "residue_start": start,
        "residue_end": end,
        "hydropathy": float(f"{rounded:.4f}"),
    }


def _element_count(token: str) -> tuple[str, int] | None:
    """Element symbol and count from one formula token.

    The count is tested with ``str.isdecimal``, which is true of exactly the
    characters ``int`` reads as a base-ten numeral. ``str.isdigit`` is also
    true of superscripts and subscripts, which ``int`` refuses: ``C²`` passed
    the test and then raised ``ValueError`` out of a function documented to
    return None for a formula it cannot read. Refusing it is also the right
    reading -- a superscript after an element symbol is a charge, not a count,
    so the formula is one this has only partly understood and no weight is
    published for it.
    """
    split_at = 0
    while split_at < len(token) and token[split_at].isalpha():
        split_at += 1
    letters = token[:split_at]
    count_text = token[split_at:]
    if not letters or len(letters) > 3 or (count_text and not count_text.isdecimal()):
        return None
    symbol = letters[0].upper() + letters[1:].lower()
    if symbol not in _ATOMIC_WEIGHT:
        return None
    if not count_text:
        return symbol, 1
    count = int(count_text)
    if count < 1:
        return None
    return symbol, count


def _letters(sequence: str) -> str:
    return "".join(sequence.split()).upper()


def _cell(row: Mapping[str, str], key: str) -> str:
    raw = row.get(key, "")
    if not isinstance(raw, str):
        return ""
    return _unquote(raw.strip())


def _unquote(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text


def _furin_sites(sequence: str) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    for index in range(len(sequence) - 3):
        if sequence[index] != "R" or sequence[index + 3] != "R":
            continue
        third = sequence[index + 2]
        if third == "K":
            found.append({"residue": index + 1, "motif": "RXKR"})
        elif third == "R":
            found.append({"residue": index + 1, "motif": "RXRR"})
    return found


def _acid_sites(sequence: str) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    for index in range(len(sequence) - 1):
        if sequence[index : index + 2] == "DP":
            found.append({"residue": index + 1, "motif": "DP"})
    return found


def _homopolymer(sequence: str) -> tuple[int, str | None, int | None]:
    best = 0
    residue: str | None = None
    start: int | None = None
    index = 0
    while index < len(sequence):
        end = index + 1
        while end < len(sequence) and sequence[end] == sequence[index]:
            end += 1
        length = end - index
        if length > best:
            best = length
            residue = sequence[index]
            start = index + 1
        index = end
    return best, residue, start


def _hydrophobic_run(sequence: str) -> tuple[int, int | None]:
    return _letter_run(sequence, HYDROPHOBIC)


def _letter_run(sequence: str, allowed: str | frozenset[str]) -> tuple[int, int | None]:
    best = 0
    start: int | None = None
    run = 0
    run_start = 0
    for index, letter in enumerate(sequence):
        if letter in allowed:
            if run == 0:
                run_start = index
            run += 1
            if run > best:
                best = run
                start = run_start + 1
            continue
        run = 0
    return best, start


def _polypeptide_phrase(record: Mapping[str, object]) -> str | None:
    parts: list[str] = []
    furin = _site_piece("furin", record, "furin")
    if furin is not None:
        parts.append(furin)
    cleavage = _site_piece("acid cleavage", record, "acid_cleavage")
    if cleavage is not None:
        parts.append(cleavage)
    if record.get("n_terminal_glutamine") is True:
        parts.append("N-terminal glutamine")
    unusual = _position_piece("nonstandard", record, "nonstandard")
    if unusual is not None:
        parts.append(unusual)
    if not parts:
        return None
    return "; ".join(parts)


def _cpg_piece(record: Mapping[str, object]) -> str | None:
    count = _count(record, "cpg_count")
    if count is None:
        return None
    numbers = _number_text(record.get("cpg_positions"))
    if not numbers:
        return f"CpG {count}"
    word = "residue" if len(numbers) == 1 else "residues"
    return f"CpG {count}, {word} {', '.join(numbers)}"


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
            if isinstance(residue, bool) or not isinstance(residue, int) or residue < 1:
                continue
            if not isinstance(motif, str) or motif not in TRIAGE_MOTIFS:
                continue
            details.append(f"residue {residue} motif {motif}")
    if details:
        return f"{label} {count}, " + ", ".join(details)
    return f"{label} {count}"


def _position_piece(label: str, record: Mapping[str, object], prefix: str) -> str | None:
    count = _count(record, f"{prefix}_count")
    if count is None:
        return None
    numbers = _number_text(record.get(f"{prefix}_residues"))
    if not numbers:
        return f"{label} {count}"
    word = "residue" if len(numbers) == 1 else "residues"
    return f"{label} {count}, {word} {', '.join(numbers)}"


def _count(record: Mapping[str, object], key: str) -> int | None:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _number_text(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if isinstance(item, int) and not isinstance(item, bool) and item >= 1]
