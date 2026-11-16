"""Pathway, mechanism, and optional target for one confirmed disease.

A disease can involve more than one pathway, protein, and mechanism. The
choices here name which of those the structure search should prefer. They do
not decide which protein starts the disease, and they do not say that a protein
involved in progression is the protein that initiates it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rfproteina.disease.structures import StructureOption

MECHANISMS: tuple[tuple[str, str], ...] = (
    ("any", "Any therapeutic mechanism"),
    ("disease_associated", "Protein associated with the disease"),
    ("pathway", "Protein in the selected pathway"),
    ("interaction", "Protein that interacts with a disease-associated protein"),
)
#: Shown on the focus page and again on the choose-structures page.
DISTINCTION = (
    "A disease can involve multiple pathways, proteins, and mechanisms. "
    "Targeting a protein involved in disease progression is not necessarily equivalent "
    "to targeting a protein responsible for initiating the disease. "
    "This page does not decide which protein starts the disease. "
    "An association with the disease is not proof that the protein initiates it, "
    "and a protein in a pathway is not the same role as a protein that interacts "
    "with a disease-associated protein."
)
NAMED_ROLE = "Named target"
DISEASE_ROLE = "Disease-associated"
PATHWAY_ROLE = "Pathway member"
PARTNER_ROLE = "Interaction partner"
_PRIMARY = {
    "disease_associated": DISEASE_ROLE,
    "pathway": PATHWAY_ROLE,
    "interaction": PARTNER_ROLE,
}
_OTHER_ROLES = (DISEASE_ROLE, PATHWAY_ROLE, PARTNER_ROLE)


@dataclass(frozen=True)
class BiologicalFocus:
    """What the structure search should prefer after the disease is confirmed.

    The accession and symbol tuples are evidence from the lookup. They are empty
    when that lookup was not run. ``target`` is the text the user typed or picked.
    """

    pathway_id: str = "any"
    pathway_name: str = ""
    mechanism: str = "any"
    target: str = ""
    target_accession: str = ""
    target_symbol: str = ""
    disease_accessions: tuple[str, ...] = ()
    disease_symbols: tuple[str, ...] = ()
    pathway_accessions: tuple[str, ...] = ()
    interaction_symbols: tuple[str, ...] = ()
    #: Live reads that errored rather than returning nothing, named by the keys
    #: in ``trace.FETCH_KEYS``. An empty tuple of accessions says the lookup
    #: found nothing *or* that it failed, and the evidence chain was recording
    #: both as "consulted". The caller that caught the error is the only place
    #: that knows which, so it says so here and the trace reads it back.
    fetch_failures: tuple[str, ...] = ()


def focus_from_fields(
    fields: dict[str, list[str]],
    pathways: Sequence[tuple[str, str]] = (),
) -> BiologicalFocus:
    """Read the focus page. A typed target replaces the menu. An unknown id is Any."""
    allowed = dict(pathways)
    raw_pathway = _first(fields, "pathway") or "any"
    if raw_pathway in allowed:
        pathway_id = raw_pathway
        pathway_name = allowed[raw_pathway]
    else:
        pathway_id = "any"
        pathway_name = ""
    mechanism = _first(fields, "mechanism") or "any"
    if mechanism not in {item[0] for item in MECHANISMS}:
        mechanism = "any"
    typed = _first(fields, "target_name")
    menu = _first(fields, "target_menu")
    target = typed or menu
    return BiologicalFocus(pathway_id=pathway_id, pathway_name=pathway_name, mechanism=mechanism, target=target)


def focus_is_active(focus: BiologicalFocus) -> bool:
    """True when the user named a pathway, a mechanism, or a target."""
    if focus.target.strip():
        return True
    if focus.mechanism != "any":
        return True
    return focus.pathway_id not in {"", "any"}


def focus_summary(focus: BiologicalFocus) -> str:
    """The chosen pathway, mechanism, and target, then why those roles differ."""
    if focus.pathway_name:
        pathway = focus.pathway_name
    elif focus.pathway_id not in {"", "any"}:
        pathway = focus.pathway_id
    else:
        pathway = "Any pathway"
    mechanism = dict(MECHANISMS).get(focus.mechanism, "Any therapeutic mechanism")
    target = target_text(focus) or "none"
    return f"Pathway: {pathway}. Mechanism: {mechanism}. Target: {target}. {DISTINCTION}"


def target_text(focus: BiologicalFocus) -> str:
    """The named target the way the menu offered it, not the way the form encodes it.

    The menu posts ``SYMBOL|ACCESSION`` because the page needs both halves.
    Printed straight back, that pipe was the only form of the target a reader
    was ever shown -- on the structures page and in the terminal line.

    The fallback does not split the raw text into a symbol and an accession.
    Which half is which is ``query_parts``'s judgement, made from the shape of
    each token, and a focus whose halves were never resolved has no judgement
    to report; naming them by position would invent one.
    """
    symbol = focus.target_symbol.strip()
    accession = focus.target_accession.strip()
    if symbol and accession:
        return f"{symbol} ({accession})"
    named = symbol or accession
    if named:
        return named
    return " ".join(part.strip() for part in focus.target.split("|") if part.strip())


def search_additions(focus: BiologicalFocus) -> tuple[list[str], list[str]]:
    """UniProt accessions and gene symbols the structure search should add.

    The named target comes first so a long pathway list cannot push it past the
    search cap. Interaction partners are added only for that mechanism. The
    disease-associated proteins are already the search the caller runs.
    """
    accessions: list[str] = []
    symbols: list[str] = []
    if focus.target_accession:
        accessions.append(focus.target_accession)
    if focus.target_symbol:
        symbols.append(focus.target_symbol)
    accessions.extend(focus.pathway_accessions)
    if focus.mechanism == "interaction":
        symbols.extend(focus.interaction_symbols)
    return accessions, symbols


def row_roles(row: StructureOption, focus: BiologicalFocus) -> tuple[str, ...]:
    """Roles this row actually matches. Empty when nothing in the focus names it."""
    found: list[str] = []
    if _named(row, focus):
        found.append(NAMED_ROLE)
    key = row.accession.strip().casefold()
    if key and key in _keys(focus.disease_accessions, focus.disease_symbols):
        found.append(DISEASE_ROLE)
    if key and key in _keys(focus.pathway_accessions):
        found.append(PATHWAY_ROLE)
    if key and key in _keys(focus.interaction_symbols):
        found.append(PARTNER_ROLE)
    return tuple(found)


def row_role(row: StructureOption, focus: BiologicalFocus) -> str:
    """The role that places this row. Named target outranks the chosen mechanism."""
    roles = set(row_roles(row, focus))
    if NAMED_ROLE in roles:
        return NAMED_ROLE
    primary = _preferred_role(focus)
    if primary and primary in roles:
        return primary
    for role in _OTHER_ROLES:
        if role in roles:
            return role
    return ""


def rank_score(row: StructureOption, focus: BiologicalFocus) -> int:
    """Higher is the row the chosen mechanism asked to see first.

    400 is the named target. 300 is the chosen mechanism's own set. Choosing a
    pathway and leaving the mechanism on Any prefers that pathway the same way.
    The other sets follow: disease association, then pathway membership, then an
    interaction partner. A row with none of those is 0.
    """
    roles = set(row_roles(row, focus))
    if NAMED_ROLE in roles:
        return 400
    primary = _preferred_role(focus)
    if primary and primary in roles:
        return 300
    best = 0
    for index, role in enumerate(_OTHER_ROLES):
        if role == primary or role not in roles:
            continue
        best = max(best, 200 - index)
    return best


def _preferred_role(focus: BiologicalFocus) -> str | None:
    """The role the user asked to see first.

    An explicit mechanism wins. A pathway with the mechanism left on Any is
    still a choice, so that pathway's proteins come before the disease list.
    """
    chosen = _PRIMARY.get(focus.mechanism)
    if chosen:
        return chosen
    if focus.pathway_id not in {"", "any"}:
        return PATHWAY_ROLE
    return None


def display_rank(
    rows: Sequence[StructureOption],
    focus: BiologicalFocus | None,
) -> list[tuple[int, StructureOption]]:
    """Listed numbers in rank order. The numbers themselves stay the original ones.

    An inactive focus keeps the listed order. Checkbox values are those numbers,
    so a later ``--pick`` still names the same row.
    """
    numbered = list(enumerate(rows, start=1))
    if focus is None or not focus_is_active(focus):
        return numbered
    return sorted(numbered, key=lambda item: (-rank_score(item[1], focus), item[0]))


def _named(row: StructureOption, focus: BiologicalFocus) -> bool:
    """Whether this row is the optional target the user named."""
    raw = focus.target.strip().casefold()
    if not raw:
        return False
    keys = {row.accession.strip().casefold(), row.entry_id.strip().casefold()}
    keys.discard("")
    if focus.target_accession.casefold() in keys or focus.target_symbol.casefold() in keys:
        return True
    if raw in keys:
        return True
    for part in raw.split("|"):
        token = part.strip()
        if token and token in keys:
            return True
    return False


def _keys(*groups: Sequence[str]) -> set[str]:
    found: set[str] = set()
    for group in groups:
        for item in group:
            text = item.strip().casefold()
            if text:
                found.add(text)
    return found


def _first(fields: dict[str, list[str]], name: str) -> str:
    values = fields.get(name) or []
    return values[0].strip() if values else ""
