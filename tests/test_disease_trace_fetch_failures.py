"""A live read that errored is not a live read that was consulted.

``_service_notes`` took one boolean -- was this run live -- and wrote
"consulted" against Open Targets and Reactome. ``cli._biological_focus`` catches
``OSError`` and ``ValueError`` from both reads, logs a warning and carries on
with empty lists, which is the right thing for a search to do: a disease whose
target list cannot be fetched should still let the user pick a structure. It is
the wrong thing for the evidence chain, which then stated that a service had
been consulted on a run where the fetch raised and nothing came back.

Empty lists cannot be the signal, because a successful read of a disease with no
associated target returns the same empty list. The caller that caught the error
is the only place that knows which happened, so it records the outcome on the
focus and this is what reads it back.

There are three outcomes, not two. The Reactome read takes the accessions the
Open Targets read returns, so a failed target list leaves the pathway list
unmade rather than failed: ``not_performed``, the status for a call the run
never sent, and neither a consult nor Reactome's fault.

The guards matter as much as the fix: a live run that worked must still say
consulted, or the chain has lost the ability to record positive evidence.
"""

from rfproteina.disease.focus import BiologicalFocus
from rfproteina.disease.trace import (
    FETCH_FAILED,
    NOT_PERFORMED,
    OPEN_TARGETS_FOCUS,
    REACTOME_PATHWAY_LIST,
    search_trace,
)

_TARGETS = ("Open Targets", "disease-associated targets")
_PAGE = ("Reactome", "pathway list on the biological target page")


def _status(focus: BiologicalFocus | None, service: tuple[str, str], *, live: bool = True) -> str:
    """The status one service note carries, found by name and role."""
    trace = search_trace(live_structures=live, live_disease=live, focus=focus, collected=None)
    for note in trace.services:
        if (note.name, note.role) == service:
            return note.status
    raise AssertionError(f"no service note for {service}")


def test_a_failed_target_fetch_is_not_recorded_as_consulted() -> None:
    """The reported defect, on the main live path."""
    focus = BiologicalFocus(fetch_failures=(OPEN_TARGETS_FOCUS,))
    assert _status(focus, _TARGETS) == FETCH_FAILED


def test_a_failed_pathway_list_is_not_recorded_as_consulted() -> None:
    """The same for the other read the caller catches."""
    focus = BiologicalFocus(fetch_failures=(REACTOME_PATHWAY_LIST,))
    assert _status(focus, _PAGE) == FETCH_FAILED


def test_a_live_read_that_worked_still_says_consulted() -> None:
    """The guard: the fix withholds a claim, it does not stop making them."""
    focus = BiologicalFocus()
    assert _status(focus, _TARGETS) == "consulted"
    assert _status(focus, _PAGE) == "consulted"


def test_a_failure_on_one_read_does_not_withdraw_the_other() -> None:
    """Each service answers for its own call and no more."""
    focus = BiologicalFocus(fetch_failures=(REACTOME_PATHWAY_LIST,))
    assert _status(focus, _TARGETS) == "consulted"


def test_a_failed_target_list_leaves_the_pathway_list_not_performed() -> None:
    """A read the run never made is not a read that was consulted.

    The Reactome read takes the accessions the Open Targets read returns, so
    ``cli._biological_focus`` makes it in that read's ``else`` branch: when the
    target list raises, Reactome is never called. This asserted "consulted"
    until the call was traced -- the same unearned claim the rest of this file
    exists to withdraw, left standing on the one read the restructure had just
    moved. ``FETCH_FAILED`` is wrong too: it would blame Reactome for a call it
    never received.
    """
    focus = BiologicalFocus(fetch_failures=(OPEN_TARGETS_FOCUS,))
    assert _status(focus, _PAGE) == NOT_PERFORMED
    # Withheld on the read that was never made, still reported on the one that failed.
    assert _status(focus, _TARGETS) == FETCH_FAILED


def test_a_run_that_was_not_live_is_still_not_performed() -> None:
    """A failure cannot be reported for a read the run never made."""
    focus = BiologicalFocus(fetch_failures=(OPEN_TARGETS_FOCUS, REACTOME_PATHWAY_LIST))
    assert _status(focus, _TARGETS, live=False) == NOT_PERFORMED
    assert _status(focus, _PAGE, live=False) == NOT_PERFORMED


def test_an_unknown_failure_name_is_ignored() -> None:
    """A name the trace does not know cannot silently blank a status.

    ``fetch_failures`` is read off a dataclass a caller fills in, so a typo
    there must not turn a consult into something unreadable.
    """
    focus = BiologicalFocus(fetch_failures=("open_targets", "reactome", ""))
    assert _status(focus, _TARGETS) == "consulted"
    assert _status(focus, _PAGE) == "consulted"


def test_no_focus_at_all_is_unchanged() -> None:
    """The page was never shown, so there is nothing to read a failure off."""
    assert _status(None, _TARGETS) == "consulted"
    assert _status(None, _PAGE) == NOT_PERFORMED


_STRING = ("STRING", "partners for the interaction mechanism")
_PATHWAY = ("Reactome", "participants of the selected pathway")


def test_the_string_note_needs_a_seed_to_have_been_consulted() -> None:
    """``partner_symbols`` loops over its seeds, so with none it sends nothing.

    The seeds are the disease accessions and symbols, which are empty exactly
    when the Open Targets read failed. Taking the chosen mechanism as proof of a
    consult recorded STRING against a request that was never sent -- the same
    unearned claim, on the third service in this function.
    """
    failed = BiologicalFocus(mechanism="interaction", fetch_failures=(OPEN_TARGETS_FOCUS,))
    assert _status(failed, _STRING) == NOT_PERFORMED
    # A successful lookup that found no target is the same situation.
    empty = BiologicalFocus(mechanism="interaction")
    assert _status(empty, _STRING) == NOT_PERFORMED


def test_a_seeded_interaction_run_still_says_consulted() -> None:
    """The guard: the fix withholds an unearned claim, not an earned one."""
    seeded = BiologicalFocus(mechanism="interaction", disease_accessions=("P01234",))
    assert _status(seeded, _STRING) == "consulted"
    by_symbol = BiologicalFocus(mechanism="interaction", disease_symbols=("ABC1",))
    assert _status(by_symbol, _STRING) == "consulted"


def test_another_mechanism_does_not_consult_string() -> None:
    """Unchanged: the call is only made for the interaction mechanism."""
    other = BiologicalFocus(mechanism="any", disease_accessions=("P01234",))
    assert _status(other, _STRING) == NOT_PERFORMED


def test_the_selected_pathway_note_needs_a_reactome_id() -> None:
    """``pathway_uniprot_members`` returns without a request for any other id."""
    chosen = BiologicalFocus(pathway_id="R-HSA-109582")
    assert _status(chosen, _PATHWAY) == "consulted"
    for identifier in ("", "any", "GO:0006915", "109582"):
        held = BiologicalFocus(pathway_id=identifier)
        assert _status(held, _PATHWAY) == NOT_PERFORMED, identifier
