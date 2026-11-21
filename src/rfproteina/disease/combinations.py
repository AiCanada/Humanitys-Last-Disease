"""Toxic combinations of one disease protein and one binder.

A combination is the disease protein, the binder kind, and the liability axes
that apply to that kind. Each axis is ``hit``, ``clear``, or ``untested``.
Untested stays untested. A hit is recorded on that axis alone. ``screened_clear``
requires every axis in a named bar to be clear, and it leaves axes outside that
bar untested.

Binder kinds select different axes:

* ``ligand`` — panel proteins, plus plasma strata. A reading at or below
  10_000 nM (10 uM) is a hit on that protein. A higher reading is clear on that
  protein. No reading is untested.
* ``protein`` — sequence liabilities, plus plasma strata.
* ``peptide`` — the protein sequence liabilities, a caller-supplied membrane
  result, and plasma strata. A polypeptide of 20 or fewer residues is a peptide.
* ``nucleic`` — caller-supplied hybridization and innate-immune results.
* ``other`` — no axes. The combination stays untested.

Plasma axes are abundance, hub, and fold. ``partner_id`` names the protein when
the caller has it. This module does not score sequences, list motifs, or add
atoms.

The disease protein being on the safety panel is its own outcome. Inference
asks for ``requested_clear`` on the chosen kind, which sets every applicable
axis to clear and records that request. A coarse ``high_affinity_clean`` or
``high_affinity_toxic`` filename still loads, and it does not set an axis.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rfproteina.disease.tokens import BAD_CLASS, GOOD_CLASS

HIT = "hit"
CLEAR = "clear"
UNTESTED = "untested"
STATE_VALUE = {CLEAR: 1.0, UNTESTED: 0.0, HIT: -1.0}

#: At or below this concentration, a ligand reading is a hit on that one axis.
#: This is the panel-wide fallback for an axis :data:`PANEL_NM` does not name.
#: It is a program convention, not a published threshold.
HIT_NM = 10_000.0

#: A polypeptide at or below this length is a peptide. Matches AtomWorks.
PEPTIDE_MAX_RESIDUES = 20

LIGAND_AXES = (
    "herg",
    "nav1_5",
    "cyp3a4",
    "cyp2d6",
    "cyp2c9",
    "cyp1a2",
    "cyp2c19",
    "pgp",
    "bcrp",
    "bsep",
)

#: The cut in nM each panel axis is judged against, one entry per axis.
#:
#: One number for ten different proteins -- a cardiac ion channel, a sodium
#: channel, five CYPs and three transporters -- was a convention with no
#: authority named for any axis, and a stored record could not say what it had
#: been judged against. Every entry is still 10 uM, so no stored verdict moves;
#: what is new is that the cut is now per axis and carries its provenance in
#: :data:`PANEL_NM_SOURCE`, which is what an owner needs before changing one.
#:
#: NOT A PUBLISHED THRESHOLD. 10 uM is this program's screening convention for
#: every axis here. No entry below is taken from a published rule: hERG is
#: normally judged as a margin over free Cmax rather than at a fixed
#: concentration, and BSEP inhibition in DILI work is commonly discussed nearer
#: 25-50 uM. Changing a number here is the owner's call and needs a source
#: recorded in :data:`PANEL_NM_SOURCE` beside it.
PANEL_NM: dict[str, float] = {axis: HIT_NM for axis in LIGAND_AXES}

#: Where each axis's cut in :data:`PANEL_NM` came from. ``program_convention_10uM``
#: means exactly that: a convention this program applies, with no publication
#: behind it. It is stored on every panel record so a reader is never left to
#: assume the cut was sourced.
PANEL_NM_SOURCE: dict[str, str] = {axis: "program_convention_10uM" for axis in LIGAND_AXES}

#: The provenance an axis outside :data:`PANEL_NM_SOURCE` falls back to.
PANEL_NM_SOURCE_DEFAULT = "program_convention_10uM"


def panel_cut(axis: str) -> float:
    """The nM cut this panel axis is judged against. A reading at or below it is a hit."""
    return PANEL_NM.get(axis, HIT_NM)


def panel_cut_source(axis: str) -> str:
    """Where this axis's cut came from. Never empty, so a record always says."""
    return PANEL_NM_SOURCE.get(axis, PANEL_NM_SOURCE_DEFAULT)


PROTEIN_AXES = (
    "immunogenicity",
    "proteome_offtarget",
    "mhc_i",
    "mhc_ii",
    "ada",
    "paralog",
)
PEPTIDE_ONLY_AXES = ("membrane",)
NUCLEIC_AXES = ("off_target_hybridization", "innate_immune")
PLASMA_AXES = ("plasma_abundance", "plasma_hub", "plasma_fold")

AXES = LIGAND_AXES + PROTEIN_AXES + PEPTIDE_ONLY_AXES + NUCLEIC_AXES + PLASMA_AXES
CLASS_CONDITION_DIM = 2 * len(AXES)

LIGAND_APPLICABLE = LIGAND_AXES + PLASMA_AXES
PROTEIN_APPLICABLE = PROTEIN_AXES + PLASMA_AXES
PEPTIDE_AXES = PROTEIN_AXES + PEPTIDE_ONLY_AXES
PEPTIDE_APPLICABLE = PEPTIDE_AXES + PLASMA_AXES

AXES_BY_KIND = {
    "ligand": LIGAND_APPLICABLE,
    "protein": PROTEIN_APPLICABLE,
    "peptide": PEPTIDE_APPLICABLE,
    "nucleic": NUCLEIC_AXES,
    "other": (),
}

#: Coverage is cardiac plus the two efflux transporters. Strict adds the five
#: drug-drug-interaction cytochromes. Neither bar speaks for axes left out of it.
BARS: dict[str, dict[str, tuple[str, ...]]] = {
    "ligand": {
        "coverage": ("herg", "pgp", "bcrp"),
        "strict": ("herg", "cyp1a2", "cyp2c9", "cyp2c19", "cyp2d6", "cyp3a4", "pgp", "bcrp"),
        "plasma": PLASMA_AXES,
        "all": LIGAND_APPLICABLE,
    },
    "protein": {
        "sequence": PROTEIN_AXES,
        "plasma": PLASMA_AXES,
        "all": PROTEIN_APPLICABLE,
    },
    "peptide": {
        "sequence": PEPTIDE_AXES,
        "plasma": PLASMA_AXES,
        "all": PEPTIDE_APPLICABLE,
    },
    "nucleic": {
        "oligo": NUCLEIC_AXES,
        "all": NUCLEIC_AXES,
    },
    "other": {},
}

_CALLS = frozenset({"requested_clear", "screened_clear", "axis_hit", "untested", "target_is_panel", "coarse"})
_KINDS = frozenset(AXES_BY_KIND)
_KIND_FROM_ALIAS = {
    "small_molecule": "ligand",
    "non_polymer": "ligand",
    "dna": "nucleic",
    "rna": "nucleic",
    "nucleic_acid": "nucleic",
    "polypeptide": "protein",
}

# Accessions, gene symbols, and entry names of the safety panel. Used to keep a
# panel protein from being labeled clear when it is the intended target.
_PANEL_TARGET_KEYS = frozenset(
    {
        "q12809",
        "kcnh2",
        "kcnh2_human",
        "herg",
        "q14524",
        "scn5a",
        "scn5a_human",
        "nav1_5",
        "p04798",
        "cyp1a1",
        "cp1a1_human",
        "p05177",
        "cyp1a2",
        "cp1a2_human",
        "q16678",
        "cyp1b1",
        "cp1b1_human",
        "p11509",
        "cyp2a6",
        "cp2a6_human",
        "p20813",
        "cyp2b6",
        "cp2b6_human",
        "p10632",
        "cyp2c8",
        "cp2c8_human",
        "p11712",
        "cyp2c9",
        "cp2c9_human",
        "p33261",
        "cyp2c19",
        "cp2cj_human",
        "p10635",
        "cyp2d6",
        "cp2d6_human",
        "p05181",
        "cyp2e1",
        "cp2e1_human",
        "p08684",
        "cyp3a4",
        "cp3a4_human",
        "p20815",
        "cyp3a5",
        "cp3a5_human",
        "p08183",
        "abcb1",
        "abcb1_human",
        "mdr1",
        "mdr1_human",
        "pgp",
        "p_gp",
        "o95342",
        "abcb11",
        "abcbb_human",
        "bsep",
        "q9unq0",
        "abcg2",
        "abcg2_human",
        "bcrp",
    }
)

_AXIS_ALIASES: tuple[tuple[str, str], ...] = (
    ("off_target_hybridization", "off_target_hybridization"),
    ("plasma_abundance", "plasma_abundance"),
    ("proteome_offtarget", "proteome_offtarget"),
    ("immunogenicity", "immunogenicity"),
    ("innate_immune", "innate_immune"),
    ("plasma_fold", "plasma_fold"),
    ("plasma_hub", "plasma_hub"),
    ("cyp2c19", "cyp2c19"),
    ("cyp3a4", "cyp3a4"),
    ("cyp2d6", "cyp2d6"),
    ("cyp2c9", "cyp2c9"),
    ("cyp1a2", "cyp1a2"),
    ("nav1_5", "nav1_5"),
    ("membrane", "membrane"),
    ("paralog", "paralog"),
    ("mhc_ii", "mhc_ii"),
    ("mhc_i", "mhc_i"),
    ("bcrp", "bcrp"),
    ("bsep", "bsep"),
    ("herg", "herg"),
    ("p_gp", "pgp"),
    ("ada", "ada"),
    ("pgp", "pgp"),
)
_KIND_ALIASES: tuple[tuple[str, str], ...] = (
    ("small_molecule", "ligand"),
    ("non_polymer", "ligand"),
    ("nucleic_acid", "nucleic"),
    ("nucleic", "nucleic"),
    ("peptide", "peptide"),
    ("protein", "protein"),
    ("ligand", "ligand"),
    ("other", "other"),
    ("dna", "nucleic"),
    ("rna", "nucleic"),
)
_STATE_TOKENS = {
    "clean": CLEAR,
    "clear": CLEAR,
    "toxic": HIT,
    "hit": HIT,
}
# Every structure encoding AtomWorks accepts (``CIF_LIKE_EXTENSIONS``), plus the table formats a
# class directory may also hold. Sorted longest first, so a compressed name loses both the
# compression and the format suffix and the final filename token stays visible to the state scan.
_STRUCTURE_SUFFIXES = tuple(
    base + compression for base in (".cif", ".pdb", ".bcif", ".mmjson") for compression in ("", ".gz", ".gzip", ".zst")
)
_TABLE_SUFFIXES = (".tsv", ".parquet", ".txt")
_SUFFIXES = tuple(sorted(_STRUCTURE_SUFFIXES + _TABLE_SUFFIXES, key=len, reverse=True))


@dataclass(frozen=True)
class Combination:
    """One disease-protein plus binder combination, judged per applicable axis."""

    binder_kind: str
    call: str
    bar: str
    target_id: str
    target_is_panel: bool
    partner_id: str
    evidence_source: str
    states: tuple[tuple[str, str], ...]
    hits: tuple[str, ...]
    untested: tuple[str, ...]
    coarse_label: str = ""

    def __post_init__(self) -> None:
        if self.call not in _CALLS:
            raise ValueError(f"Unknown combination call {self.call!r}.")
        if self.call == "coarse":
            if self.coarse_label not in {GOOD_CLASS, BAD_CLASS}:
                raise ValueError(f"A coarse combination needs {GOOD_CLASS!r} or {BAD_CLASS!r}.")
            return
        if self.binder_kind not in _KINDS:
            raise ValueError(f"Unknown binder kind {self.binder_kind!r}.")
        expected_axes = AXES_BY_KIND[self.binder_kind]
        axes = tuple(axis for axis, _state in self.states)
        if axes != expected_axes:
            raise ValueError(f"{self.binder_kind} states must list {expected_axes}.")
        known = dict(self.states)
        if self.hits != tuple(axis for axis in expected_axes if known[axis] == HIT):
            raise ValueError("hits do not match the axis states.")
        if self.untested != tuple(axis for axis in expected_axes if known[axis] == UNTESTED):
            raise ValueError("untested axes do not match the axis states.")
        _check_call(self, known)


@dataclass(frozen=True)
class ParsedClassName:
    """What a class filename declares. A coarse name sets no axis."""

    label: str
    coarse: bool
    binder_kind: str | None
    axis: str | None
    state: str | None


def rollup_axis(
    example_ids: Iterable[str],
    by_example: Mapping[str, Mapping[str, object]],
    empty: Mapping[str, object],
) -> dict[str, object]:
    """One axis for a whole run, from its per-design records.

    A run passes or fails as a set of designs: one design hitting makes the
    axis a hit and is named in ``failed``, every design clear makes it clear,
    and anything else stays untested carrying the first undecided design's
    own blocker rather than a new one invented here. The measurement on a hit
    is the worst of the hits, so a rollup cannot read milder than the designs
    it summarizes, and the rest of the record is that same design's, so a count
    and its own denominator never come from two different designs.

    This is the rule the proteome axis has always applied by hand. It lives
    here because three more axes now need the same one, and a safety rule
    written twice is a safety rule that drifts.
    """
    names = list(example_ids)
    if not names:
        record = dict(empty)
        record["failed"] = []
        return record
    states = {name: _rolled_state(by_example.get(name)) for name in names}
    failed = [name for name in names if states[name] == HIT]
    if failed:
        numbers = {name: _rolled_number(by_example.get(name)) for name in failed}
        measured = [name for name in failed if numbers[name] is not None]
        # The worst hit's own record, not the first hit's with another design's
        # number written over it. Every other field here is a fact about one
        # design -- which entry it matched, at what identity, against which
        # file -- so pairing them with a measurement from a different design
        # describes a design that does not exist.
        worst = max(measured, key=lambda name: numbers[name]) if measured else failed[0]
        record = dict(by_example.get(worst) or empty)
        record["state"] = HIT
        record["blocker"] = None
        record["measurement"] = numbers[worst]
        record["failed"] = failed
        return record
    if all(states[name] == CLEAR for name in names):
        record = dict(by_example.get(names[0]) or empty)
        record["state"] = CLEAR
        record["blocker"] = None
        record["measurement"] = 0
        record["failed"] = []
        return record
    undecided = next((name for name in names if states[name] not in {HIT, CLEAR}), names[0])
    record = dict(by_example.get(undecided) or empty)
    record["state"] = UNTESTED
    record["failed"] = []
    return record


def _rolled_state(record: Mapping[str, object] | None) -> str:
    state = (record or {}).get("state")
    return state if isinstance(state, str) else UNTESTED


def _rolled_number(record: Mapping[str, object] | None) -> int | float | None:
    value = (record or {}).get("measurement")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value


def target_is_panel_protein(identifier: str) -> bool:
    """Whether ``identifier`` is a safety-panel protein.

    An empty identifier is not on the panel. Matching it would mark every
    unnamed target as a liability.
    """
    key = identifier.strip().casefold().replace("-", "_").replace(" ", "_").replace(".", "_")
    if not key:
        return False
    return key in _PANEL_TARGET_KEYS


def binder_kind_from_partner(polymer: str, n_residues: int | None = None) -> str:
    """Map a partner polymer to a binder kind. Length splits protein from peptide."""
    if n_residues is not None and n_residues < 1:
        raise ValueError(f"n_residues must be >= 1 when given, got {n_residues}.")
    key = polymer.strip().casefold().replace("-", "_").replace(" ", "_")
    if key in {"glycan", "branched", "other", "unknown"}:
        return "other"
    kind = _require_kind(polymer)
    if kind == "protein" and n_residues is not None and n_residues <= PEPTIDE_MAX_RESIDUES:
        return "peptide"
    return kind


def ligand_states(
    measurements: Mapping[str, float | None],
    *,
    inconclusive: Iterable[str] = (),
) -> dict[str, str]:
    """Turn nM readings into per-protein states. Absent and inconclusive stay untested.

    Each axis is judged against its own cut in :data:`PANEL_NM`, so a panel
    whose entries differ cannot be judged by one number read from elsewhere.
    """
    if isinstance(inconclusive, str):
        raise ValueError("inconclusive is a collection of axis names, not one string.")
    skipped = set(inconclusive)
    states: dict[str, str] = {}
    for axis, value in measurements.items():
        if axis not in LIGAND_AXES:
            raise ValueError(f"{axis} is not a ligand panel axis.")
        if axis in skipped or value is None:
            states[axis] = UNTESTED
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{axis} measurement must be a non-negative nM value or absent.")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"{axis} measurement must be a non-negative nM value or absent.")
        states[axis] = HIT if numeric <= panel_cut(axis) else CLEAR
    return states


def judge_combination(
    binder_kind: str,
    evidence: Mapping[str, str],
    *,
    bar: str | None = None,
    target_id: str = "",
    target_is_panel: bool = False,
    partner_id: str = "",
) -> Combination:
    """Judge one combination. Missing axes are untested. Hits are not merged."""
    kind = _require_kind(binder_kind)
    _identifier("target_id", target_id)
    _identifier("partner_id", partner_id)
    panel = target_is_panel or target_is_panel_protein(target_id)
    if kind == "other":
        if evidence:
            raise ValueError("A binder of kind other has no liability axes. Leave the combination untested.")
        call = "target_is_panel" if panel else "untested"
        return _make(
            kind,
            call,
            "none",
            target_id,
            panel,
            partner_id,
            "measured",
            (),
        )

    _reject_foreign_axes(kind, evidence)
    applicable = AXES_BY_KIND[kind]
    states = tuple((axis, _require_state(evidence.get(axis, UNTESTED), axis)) for axis in applicable)
    hits = tuple(axis for axis, state in states if state == HIT)
    if hits:
        used_bar = "all" if bar is None else bar
        _bar_axes(kind, used_bar)
        return _make(kind, "axis_hit", used_bar, target_id, panel, partner_id, "measured", states)
    if panel:
        used_bar = "none" if bar is None else bar
        if bar is not None:
            _bar_axes(kind, used_bar)
        return _make(kind, "target_is_panel", used_bar, target_id, True, partner_id, "measured", states)

    used_bar = "all" if bar is None else bar
    bar_axes = _bar_axes(kind, used_bar)
    known = dict(states)
    call = "screened_clear" if all(known[axis] == CLEAR for axis in bar_axes) else "untested"
    return _make(kind, call, used_bar, target_id, False, partner_id, "measured", states)


def requested_clear(
    binder_kind: str,
    *,
    target_id: str = "",
    target_is_panel: bool = False,
) -> Combination:
    """The inference request: every applicable axis clear, recorded as a request."""
    kind = _require_kind(binder_kind)
    _identifier("target_id", target_id)
    panel = target_is_panel or target_is_panel_protein(target_id)
    if kind == "other":
        raise ValueError("A binder of kind other has no clear request. Its combination stays untested.")
    if panel:
        name = target_id or "the given target"
        raise ValueError(
            f"The disease protein {name} is on the safety panel. "
            "Inference will not mark that combination requested-clear."
        )
    states = tuple((axis, CLEAR) for axis in AXES_BY_KIND[kind])
    return _make(kind, "requested_clear", "all", target_id, False, "", "requested", states)


def combination_condition(n_tokens: int, combination: Combination) -> np.ndarray:
    """Broadcast state then applicability onto ``n_tokens`` real tokens.

    Column ``i`` is the state of axis ``i`` (clear 1, untested 0, hit -1).
    Column ``len(AXES) + i`` is 1 when that axis applies to the binder kind.
    An untested applicable axis stays 0 on the state side and 1 on the mask,
    which keeps it distinct from an axis that does not apply.
    """
    if n_tokens < 1:
        raise ValueError(f"The combination marks real tokens, so n_tokens must be >= 1, got {n_tokens}.")
    if combination.call == "coarse":
        raise ValueError(
            f"{combination.coarse_label!r} does not say which binder or which axis. "
            "A coarse name cannot be broadcast onto tokens."
        )
    state = np.zeros(len(AXES), dtype=np.float32)
    mask = np.zeros(len(AXES), dtype=np.float32)
    known = dict(combination.states)
    for index, axis in enumerate(AXES):
        if axis not in known:
            continue
        mask[index] = 1.0
        state[index] = STATE_VALUE[known[axis]]
    row = np.concatenate([state, mask])
    return np.tile(row, (n_tokens, 1))


def _cif_list(items: tuple[str, ...]) -> str:
    """One mmCIF token. An empty list is the null placeholder."""
    if not items:
        return "."
    return ",".join(items)


def combination_record(combination: Combination) -> str:
    """mmCIF items for the combination, to sit inside an existing data block.

    The record carries no ``data_`` header. A second header would make the file a
    multi-block CIF, which the structure loader rejects outright, so these items
    join the source block as one more category. It contains no atom sites.
    """
    lines = [
        "#",
        "_rfproteina_disease.combination target_plus_binder",
        f"_rfproteina_disease.binder_kind {combination.binder_kind}",
        f"_rfproteina_disease.call {combination.call}",
        f"_rfproteina_disease.bar {combination.bar}",
        f"_rfproteina_disease.evidence {combination.evidence_source}",
        f"_rfproteina_disease.target_id {combination.target_id or '.'}",
        f"_rfproteina_disease.target_is_panel {'true' if combination.target_is_panel else 'false'}",
        f"_rfproteina_disease.partner_id {combination.partner_id or '.'}",
        f"_rfproteina_disease.condition_dim {CLASS_CONDITION_DIM}",
        f"_rfproteina_disease.hits {_cif_list(combination.hits)}",
        f"_rfproteina_disease.untested {_cif_list(combination.untested)}",
    ]
    if combination.coarse_label:
        lines.append(f"_rfproteina_disease.coarse_label {combination.coarse_label}")
    for axis, state in combination.states:
        lines.append(f"_rfproteina_disease.{axis} {state}")
    lines.append("#")
    return "\n".join(lines) + "\n"


def coarse_combination(label: str) -> Combination:
    """A filename that only says clean or toxic. No axis is set."""
    if label not in {GOOD_CLASS, BAD_CLASS}:
        raise ValueError(f"Coarse label must be {GOOD_CLASS!r} or {BAD_CLASS!r}.")
    return Combination(
        binder_kind="unspecified",
        call="coarse",
        bar="none",
        target_id="",
        target_is_panel=False,
        partner_id="",
        evidence_source="coarse",
        states=(),
        hits=(),
        untested=(),
        coarse_label=label,
    )


def parse_class_filename(name: str) -> ParsedClassName | None:
    """Read a class filename.

    Returns ``None`` when the name is not a class file. A shared axis without a
    binder kind raises ``ValueError``, because protein and peptide would
    otherwise be filed as the same combination.
    """
    stem = _stem(name)
    axes = _bounded_matches(stem, _AXIS_ALIASES)
    if len(axes) > 1:
        raise ValueError(f"{Path(name).name} names more than one axis ({', '.join(axes)}). One file records one axis.")
    kinds = _bounded_matches(stem, _KIND_ALIASES)
    if len(kinds) > 1:
        raise ValueError(f"{Path(name).name} names more than one binder kind ({', '.join(kinds)}).")
    if axes:
        return _parsed_axis(Path(name).name, stem, axes[0], kinds)
    if GOOD_CLASS in stem and BAD_CLASS in stem:
        raise ValueError(f"{Path(name).name} contains both {GOOD_CLASS} and {BAD_CLASS}.")
    if GOOD_CLASS in stem:
        return ParsedClassName(GOOD_CLASS, True, None, None, None)
    if BAD_CLASS in stem:
        return ParsedClassName(BAD_CLASS, True, None, None, None)
    return None


def combination_from_parsed(parsed: ParsedClassName) -> Combination:
    """Build the combination a filename is allowed to claim."""
    if parsed.coarse:
        return coarse_combination(parsed.label)
    if parsed.binder_kind is None or parsed.axis is None or parsed.state is None:
        raise ValueError("A measured combination needs a binder kind, one axis, and a state.")
    return judge_combination(parsed.binder_kind, {parsed.axis: parsed.state})


def _make(
    kind: str,
    call: str,
    bar: str,
    target_id: str,
    panel: bool,
    partner_id: str,
    evidence_source: str,
    states: tuple[tuple[str, str], ...],
) -> Combination:
    hits = tuple(axis for axis, state in states if state == HIT)
    untested = tuple(axis for axis, state in states if state == UNTESTED)
    return Combination(
        binder_kind=kind,
        call=call,
        bar=bar,
        target_id=target_id,
        target_is_panel=panel,
        partner_id=partner_id,
        evidence_source=evidence_source,
        states=states,
        hits=hits,
        untested=untested,
    )


def _check_call(combination: Combination, known: dict[str, str]) -> None:
    hits = combination.hits
    if hits and combination.call != "axis_hit":
        raise ValueError("A hit makes the call axis_hit.")
    if combination.call == "axis_hit" and not hits:
        raise ValueError("axis_hit requires a hit axis.")
    if combination.call == "requested_clear":
        if combination.evidence_source != "requested" or combination.target_is_panel or combination.bar != "all":
            raise ValueError("requested_clear is an inference request on every applicable axis.")
        if any(state != CLEAR for state in known.values()):
            raise ValueError("requested_clear sets every applicable axis clear.")
        return
    if combination.evidence_source != "measured":
        raise ValueError("A measured combination records evidence_source measured.")
    if combination.call == "screened_clear":
        if combination.target_is_panel:
            raise ValueError("A safety-panel target is not screened_clear.")
        for axis in _bar_axes(combination.binder_kind, combination.bar):
            if known[axis] != CLEAR:
                raise ValueError(f"screened_clear requires {axis} to be clear.")
        return
    if combination.call == "target_is_panel" and not combination.target_is_panel:
        raise ValueError("target_is_panel requires the target flag.")
    if combination.call == "untested" and hits:
        raise ValueError("untested combination cannot contain a hit.")


def _require_kind(binder_kind: str) -> str:
    kind = binder_kind.strip().casefold().replace("-", "_").replace(" ", "_")
    kind = _KIND_FROM_ALIAS.get(kind, kind)
    if kind not in _KINDS:
        known = ", ".join(sorted(_KINDS))
        raise ValueError(f"Unknown binder kind {binder_kind!r}. Expected one of {known}.")
    return kind


def _require_state(value: str, axis: str) -> str:
    if value not in {HIT, CLEAR, UNTESTED}:
        raise ValueError(f"{axis} state must be hit, clear, or untested, got {value!r}.")
    return value


def _reject_foreign_axes(kind: str, evidence: Mapping[str, str]) -> None:
    applicable = AXES_BY_KIND[kind]
    for axis in evidence:
        if axis not in AXES:
            raise ValueError(f"Unknown axis {axis!r}.")
        if axis not in applicable:
            owners = ", ".join(axis_kinds(axis))
            raise ValueError(f"{axis} is not a {kind} axis. It belongs to {owners}.")


def axis_kinds(axis: str) -> tuple[str, ...]:
    """Binder kinds that carry ``axis``."""
    return tuple(kind for kind, axes in AXES_BY_KIND.items() if axis in axes)


def bar_axes(binder_kind: str, bar: str) -> tuple[str, ...]:
    """The axes one bar requires, or a ValueError naming the bars that exist.

    Public because a screen needs this list twice: once to refuse an unknown
    bar before it measures anything, and once to record what the verdict had
    to clear. Reading it from ``BARS`` both times is what keeps the recorded
    requirement from drifting away from the applied one.
    """
    return _bar_axes(_require_kind(binder_kind), bar)


def _bar_axes(kind: str, bar: str) -> tuple[str, ...]:
    bars = BARS[kind]
    if bar not in bars:
        known = ", ".join(bars) if bars else "none"
        raise ValueError(f"{kind} has no {bar!r} bar. Known bars: {known}.")
    return bars[bar]


def _identifier(field: str, value: str) -> None:
    if value and any(character.isspace() for character in value):
        raise ValueError(f"{field} is one identifier.")


def _stem(name: str) -> str:
    base = Path(name).name.casefold().replace("-", "_")
    for suffix in _SUFFIXES:
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return Path(base).stem


def _bounded_matches(stem: str, aliases: tuple[tuple[str, str], ...]) -> list[str]:
    padded = f"_{stem}_"
    found: list[str] = []
    for alias, canonical in aliases:
        needle = f"_{alias}_"
        if needle not in padded:
            continue
        if canonical not in found:
            found.append(canonical)
        padded = padded.replace(needle, "_")
    return found


def _parsed_axis(filename: str, stem: str, axis: str, kinds: list[str]) -> ParsedClassName:
    tokens = [token for token in stem.split("_") if token in _STATE_TOKENS]
    if len(tokens) != 1:
        raise ValueError(f"{filename} names {axis} but not exactly one of clean, clear, toxic, or hit.")
    state = _STATE_TOKENS[tokens[0]]
    if kinds:
        kind = kinds[0]
        if axis not in AXES_BY_KIND[kind]:
            raise ValueError(f"{filename} puts {axis} on a {kind} binder, and that axis does not apply.")
    else:
        owners = axis_kinds(axis)
        if len(owners) != 1:
            joined = " and ".join(owners)
            raise ValueError(
                f"{filename} names {axis}, which is shared by {joined} binders. Put the binder kind in the file name."
            )
        kind = owners[0]
    side = "toxic" if state == HIT else "clean"
    label = f"high_affinity_{kind}_{axis}_{side}"
    return ParsedClassName(label, False, kind, axis, state)
