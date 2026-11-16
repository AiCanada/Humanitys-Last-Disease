"""Confirm a disease name before any structure is fetched."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

NOT_FOUND_MESSAGE = "Disease not found. Input the scientific or medical name of the disease."


class DiseaseNotFoundError(Exception):
    """The query matched nothing, or the user declined the match."""

    def __init__(self) -> None:
        super().__init__(NOT_FOUND_MESSAGE)


@dataclass(frozen=True)
class Disease:
    """A medical disease name and its ontology identifier."""

    name: str
    disease_id: str


def name_matches(query: str, disease: Disease) -> bool:
    """Whether this name is a substring match for the query, either way round.

    The test :py:func:`match_diseases` applies once an exact name has not
    won. It is named here because the confirm menu needs the same question
    answered the same way, and a second copy of it would drift.
    """
    text = query.strip().casefold()
    if not text:
        return False
    name = disease.name.casefold()
    return text in name or name in text


def related_names(query: str, catalog: Sequence[Disease]) -> list[Disease]:
    """Catalog entries whose name matches the query, in the catalog's order.

    Unlike :py:func:`match_diseases` this does not stop at an exact hit. An
    exact name is one of its own relatives, so a query that lands exactly
    still offers the siblings that share the name -- which is the whole
    point of the confirm menu.
    """
    return [disease for disease in catalog if name_matches(query, disease)]


def match_diseases(query: str, catalog: list[Disease], *, ranked: bool = False) -> list[Disease]:
    """Exact names win, then substring matches.

    ``ranked`` declares that the catalog is already this query's answer, as an
    ontology search returns. A synonym or an abbreviation is a substring of no
    resolved label, so filtering such a catalog locally would discard the very
    hits the search exists to produce, and those hits are kept in their given
    order instead. A local catalog is filtered, so an unknown name is still not
    found, and an empty catalog is never found.
    """
    text = query.strip().casefold()
    if not text:
        return []
    exact = [disease for disease in catalog if disease.name.casefold() == text]
    if exact:
        return exact
    contains = related_names(query, catalog)
    if contains:
        return contains
    return list(catalog) if ranked else []


def confirm_disease(
    query: str,
    catalog: list[Disease],
    *,
    confirm_one: Callable[[Disease], bool],
    choose_one: Callable[[list[Disease]], int],
    ranked: bool = False,
) -> Disease:
    """Ask the caller to confirm one hit, or to pick among similar names.

    ``confirm_one`` receives the single hit and returns true to accept it.
    ``choose_one`` receives the similar-name list and returns the chosen index.
    ``ranked`` is passed to :py:func:`match_diseases`.
    """
    hits = match_diseases(query, catalog, ranked=ranked)
    if not hits:
        raise DiseaseNotFoundError
    if len(hits) == 1:
        if not confirm_one(hits[0]):
            raise DiseaseNotFoundError
        return hits[0]
    index = choose_one(hits)
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(hits):
        raise DiseaseNotFoundError
    return hits[index]
