"""The confirm page's Matching names menu, and what it is allowed to offer.

An ontology search answers a disease query with its semantic neighbourhood, not
with name matches. A real Open Targets search for "Huntington disease" returns
fifteen entries, and six of them are neighbours rather than matches: familial
Alzheimer disease, Alzheimer disease 3, mild Canavan disease, choreatic disease,
and two gene-named neurodegenerative diseases. The menu listed all fifteen under
a heading that reads "Matching names".

It now lists only the names that match, by the same substring test
``match_diseases`` applies once an exact name has not won. One predicate,
``related_names``, used in both places, so the menu and the matcher cannot come
to disagree about what "matching" means.

The index mapping is the part worth guarding: the posted choice is resolved
against the list the page rendered. If the filter were applied to one and not
the other, a posted index would select a disease the user never saw.
"""

import re

from rfproteina.disease.lookup import Disease, match_diseases, name_matches, related_names
from rfproteina.disease.ui import OptionPanel, render_confirm

#: The shape a real ontology search returns: matches first, neighbours after.
_CATALOG = [
    Disease("Huntington disease", "MONDO_0007739"),
    Disease("Huntington disease and related disorders", "MONDO_0000167"),
    Disease("juvenile Huntington disease", "MONDO_0016621"),
    Disease("age of onset of Huntington disease", "OBA_2001024"),
    Disease("Huntington disease-like 3", "MONDO_0011487"),
    Disease("Huntington disease-like 2", "MONDO_0011671"),
    Disease("Huntington disease-like 1", "MONDO_0011299"),
    Disease("Huntington disease-like syndrome", "MONDO_0015548"),
    Disease("Huntington disease-like syndrome due to C9ORF72 expansions", "MONDO_0018425"),
    # Neighbours, not matches. None of these contains the query.
    Disease("VPS13A-related neurodegenerative disease", "MONDO_0008695"),
    Disease("mild Canavan disease", "MONDO_0017831"),
    Disease("XK-related neurodegenerative disease", "MONDO_0018945"),
    Disease("choreatic disease", "MONDO_0001595"),
    Disease("familial Alzheimer disease", "MONDO_0100087"),
    Disease("Alzheimer disease 3", "MONDO_0011913"),
]
_QUERY = "Huntington disease"
_NAMED = _CATALOG[0]
_NEIGHBOURS = ("Canavan", "Alzheimer", "choreatic", "VPS13A", "XK-related")


def _menu(page: str) -> list[str]:
    """The option labels of the rendered menu, in order.

    By regex, not by splitting on ">": the options are newline-joined, so a
    split picks up the separators as entries and every count comes out as
    options plus separators.
    """
    block = page.split('<select name="choice">', 1)[1].split("</select>", 1)[0]
    return re.findall(r"<option[^>]*>([^<]+)</option>", block)


def test_the_menu_offers_the_matching_names_and_not_the_neighbours():
    page = render_confirm(_NAMED, _CATALOG, _QUERY)
    labels = _menu(page)
    assert len(labels) == 9, labels
    assert labels[0].startswith("Huntington disease (")
    for neighbour in _NEIGHBOURS:
        assert neighbour not in page, f"{neighbour} is a neighbour, not a matching name"


def test_the_filter_is_the_matcher_s_own_substring_test():
    """One predicate, so the menu and the matcher agree on what matching means."""
    assert related_names(_QUERY, _CATALOG) == [item for item in _CATALOG if name_matches(_QUERY, item)]
    assert len(related_names(_QUERY, _CATALOG)) == 9
    # match_diseases stops at the exact hit; the menu deliberately does not,
    # because an exact name is one of its own relatives.
    assert match_diseases(_QUERY, _CATALOG, ranked=True) == [_NAMED]


def test_a_posted_choice_resolves_against_the_list_that_was_rendered():
    """The guard that matters: an unfiltered resolve would return an unseen disease."""
    seen: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        seen.append(page)
        return {"choice": ["3"]}

    chosen = OptionPanel(exchange).confirm_disease(_NAMED, _CATALOG, _QUERY)
    labels = _menu(seen[0])
    assert labels[2].startswith("juvenile Huntington disease")
    assert chosen == Disease("juvenile Huntington disease", "MONDO_0016621")
    # The third entry of the unfiltered catalog is the same here, so pin the
    # case that would actually diverge: the last offered option.
    last = str(len(labels))
    assert OptionPanel(lambda _p: {"choice": [last]}).confirm_disease(_NAMED, _CATALOG, _QUERY) == _CATALOG[8]
    # One past the filtered list declines rather than reaching a neighbour.
    beyond = str(len(labels) + 1)
    assert OptionPanel(lambda _p: {"choice": [beyond]}).confirm_disease(_NAMED, _CATALOG, _QUERY) is None


def test_the_typed_query_decides_the_menu_not_the_matched_name():
    """A broader query offers more, because the user asked more broadly."""
    broad = render_confirm(_NAMED, _CATALOG, "disease")
    assert len(_menu(broad)) == len(_CATALOG), "every catalog name contains 'disease'"
    narrow = render_confirm(_NAMED, _CATALOG, "Huntington disease-like")
    labels = _menu(narrow)
    assert labels[0].startswith("Huntington disease (")
    assert all("Huntington disease-like" in label or label.startswith("Huntington disease (") for label in labels)


def test_no_query_falls_back_to_the_confirmed_name():
    """A caller with no query still gets siblings, not the whole catalog."""
    page = render_confirm(_NAMED, _CATALOG)
    assert len(_menu(page)) == 9
    for neighbour in _NEIGHBOURS:
        assert neighbour not in page


def test_the_tick_box_still_wins_over_the_menu():
    panel = OptionPanel(lambda _p: {"accept": ["yes"], "choice": ["3"]})
    assert panel.confirm_disease(_NAMED, _CATALOG, _QUERY) == _NAMED


def test_a_duplicate_id_is_listed_once():
    duplicate = Disease("Huntington disease restated", "MONDO_0007739")
    page = render_confirm(_NAMED, [duplicate, *_CATALOG], _QUERY)
    assert "restated" not in page
    assert page.count("MONDO_0007739") == 2  # the heading, and option one
