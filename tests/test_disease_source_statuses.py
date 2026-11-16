"""A database's status says what happened, and the four cases are different.

``research`` reports one status per database per subject, and the evidence chain
copies it verbatim into ``research_traces/*.json``. Three of the four were
already honest: ``record`` found something, ``no record`` was asked and holds
nothing, ``unavailable`` could not be read.

``PeptideAtlas`` is keyed by UniProt accession, and ``research_subjects``
deliberately produces gene-symbol subjects as well, so for every one of those
the lookup returned ``no record`` -- an answer to a question that was never put.
It is the same unearned claim the trace work removed from the service notes, and
worse here, because ``cli`` remembers a definite answer so the claim stood for
the rest of the run folder and re-asking could not correct it.
"""

from rfproteina.disease.research import Subject, research_subjects
from rfproteina.disease.structures import StructureOption


def _row(group: str, accession: str) -> StructureOption:
    return StructureOption(
        group=group,
        accession=accession,
        entry_id="1ABC",
        chain="A",
        title=f"{group} {accession}",
        predicted=False,
        source="fixture",
    )


def _refuse(url: str) -> object:
    raise AssertionError(f"PeptideAtlas was asked after all: {url}")


def test_a_gene_subject_is_not_asked_rather_than_answered() -> None:
    """The reported defect. Nothing was read, so nothing may be reported."""
    from rfproteina.disease.research import _peptideatlas

    hit = _peptideatlas(Subject("gene", "IL5", "interleukin 5"), _refuse)
    assert hit.status == "not asked"
    assert "UniProt accession" in hit.detail
    # And it is not counted as evidence, which is what every consumer tests for.
    assert hit.status != "record"


def test_an_accession_subject_is_still_asked() -> None:
    """The guard: the fix withholds a claim, it does not stop making the call."""
    from rfproteina.disease.research import _peptideatlas

    asked: list[str] = []

    def read(url: str) -> object:
        asked.append(url)
        return {"PeptideAtlas": "proteins", "results": [{"biosequence_accession": "P05113"}]}

    hit = _peptideatlas(Subject("protein", "P05113", "interleukin 5"), read)
    assert asked, "an accession subject must still reach the service"
    assert hit.status == "record"


def test_an_asked_accession_that_holds_nothing_is_still_no_record() -> None:
    """``no record`` keeps its meaning: asked, and holds nothing."""
    from rfproteina.disease.research import _peptideatlas

    hit = _peptideatlas(Subject("protein", "P05113", "interleukin 5"), lambda _url: {"nothing": True})
    assert hit.status == "no record"


def test_gene_subjects_are_a_first_class_case_and_not_an_edge() -> None:
    """Why this mattered: the subject list is built to hold them.

    A gene row is kept exactly when its identifier is NOT an accession, so every
    gene subject reached the accession-keyed lookup.
    """
    subjects = research_subjects([_row("protein", "P05113"), _row("gene", "IL5")])
    kinds = {subject.kind for subject in subjects}
    assert kinds == {"protein", "gene"}
    gene = next(subject for subject in subjects if subject.kind == "gene")
    assert gene.identifier == "IL5"


def test_the_four_statuses_stay_distinct() -> None:
    """Each word says a different thing, and only one of them is evidence."""
    from rfproteina.disease.research import _hit_down, _hit_none, _hit_not_asked, _hit_record

    statuses = {
        _hit_record("found it").status,
        _hit_none().status,
        _hit_down("the host refused").status,
        _hit_not_asked("keyed by accession").status,
    }
    assert statuses == {"record", "no record", "unavailable", "not asked"}
    assert _hit_record("found it").status == "record"
    for absent in (_hit_none(), _hit_down("x"), _hit_not_asked("x")):
        assert absent.status != "record", "only a real record counts as evidence"
