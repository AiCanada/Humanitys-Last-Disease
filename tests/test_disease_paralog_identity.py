"""The paralog axis measuring identity, where UniRef50 could only shrug.

A binder raised against one protein can bind a close relative, and that is what
this axis is for. Its only source was UniRef50 membership, and the module's own
docstring already warned that this misses the cases that matter: a UniRef50
cluster is built around a representative rather than gathering a protein's
relatives. Measured against the live services, the warning is understated --

* KRAS comes back a singleton, with ``paralogs=[]``, while Compara puts HRAS at
  85.6 per cent identity to it and NRAS at 84.1;
* ERBB2 comes back a singleton while EGFR is 50.7 per cent identical.

Those are the two cross-reactivity risks a reviewer would name first, and the
axis reported nothing for either. So a singleton is now handed to Ensembl
Compara, asked over REST for one gene rather than downloaded as 11.23 GB.

Two things this must not do, and each has tests below.

**It must not judge on a count.** Compara holds every paralog in its gene
trees at any identity: ERBB2 has 53 whose median identity is 17.4 per cent.
A count against a cut would make ERBB2 a liability on the strength of fifty
proteins that share no surface with it -- a false alarm, which is the opposite
failure from the suppression this fixes and no better.

**It must not invent a threshold.** No published cross-reactivity identity cut
exists, so the lookup happens only once a caller names one -- and until then
the axis answers exactly as it did before, having touched no network. That
keeps the cost honest too: two requests per target that cannot be judged would
be the difference between a screen and a rate limit on a real campaign.
"""

from __future__ import annotations

from typing import Any

import pytest

from rfproteina.disease.research import LookupMissError
from rfproteina.disease.target_liability import (
    TargetLimits,
    _compara_paralogs,
    _paralog_record,
)

_KRAS = "P01116"


def _uniref(accession: str = _KRAS) -> dict[str, Any]:
    return {"results": [{"id": f"UniRef50_{accession}"}]}


def _members(accession: str = _KRAS) -> dict[str, Any]:
    """A cluster holding only the target: the singleton that reaches Compara."""
    return {"results": [{"primaryAccession": accession, "genes": [{"geneName": {"value": "KRAS"}}]}]}


def _symbol(symbol: str = "KRAS") -> dict[str, Any]:
    return {"results": [{"genes": [{"geneName": {"value": symbol}}]}]}


def _homology(pairs: list[tuple[str, float, float]]) -> dict[str, Any]:
    """Compara's shape: each homology carries identity for BOTH sequences."""
    return {
        "data": [
            {
                "homologies": [
                    {
                        "type": "other_paralog",
                        "source": {"id": "ENSG_QUERY", "perc_id": source},
                        "target": {"id": named, "perc_id": target},
                    }
                    for named, source, target in pairs
                ]
            }
        ]
    }


def reader(*, symbol: object = None, homology: object = None, members: object = None):
    """A ``read_json`` that answers by URL, so each lookup can be steered.

    Values may be an exception instance, which is raised: a service being
    unreachable is a case this axis has to get right and not a case to skip.
    """

    def read(url: str) -> object:
        if "uniref/search" in url:
            answer: object = _uniref()
        elif "rest.ensembl.org" in url:
            answer = _homology([]) if homology is None else homology
        elif "gene_primary" in url:
            answer = _symbol() if symbol is None else symbol
        else:
            answer = _members() if members is None else members
        if isinstance(answer, BaseException):
            raise answer
        return answer

    return read


# ------------------------------------------------- the gap this exists to close


def test_a_uniref_singleton_with_a_close_paralog_is_a_hit() -> None:
    """The KRAS case, which reported nothing at all before.

    UniRef50 says singleton; Compara says HRAS at 85.6 per cent. Over a cut the
    caller named, that is the liability the axis is for.
    """
    record = _paralog_record(
        _KRAS,
        reader(homology=_homology([("ENSG00000174775", 85.0, 85.6)])),
        50.0,
    )
    assert record["state"] == "hit"
    assert record["measurement"] == pytest.approx(85.6)
    assert record["closest_paralog"] == "ENSG00000174775"
    assert record["threshold"] == 50.0
    assert record.get("blocker") is None


def test_the_same_singleton_clears_against_a_higher_cut() -> None:
    """The measurement is the same; only the caller's cut moved."""
    record = _paralog_record(_KRAS, reader(homology=_homology([("ENSG00000174775", 85.0, 85.6)])), 90.0)
    assert record["state"] == "clear"
    assert record["measurement"] == pytest.approx(85.6)


def test_a_gene_compara_examined_and_found_nothing_in_is_a_clear() -> None:
    """The first clear this axis has ever been able to give.

    An empty homology list under a present gene tree is Compara having looked,
    which is a measurement. It is not the same as Compara having no entry, and
    the test below keeps those apart.
    """
    record = _paralog_record(_KRAS, reader(homology=_homology([])), 50.0)
    assert record["state"] == "clear"
    assert record["compara_examined"] == 0
    assert record.get("blocker") is None


# ------------------------------------------------------- not inventing a cut


def test_with_no_threshold_the_network_is_not_touched_at_all() -> None:
    """No cut, no lookup -- and that is the design, not an optimisation.

    Without a threshold this axis cannot judge, so the two extra requests would
    buy a ``threshold_not_set`` that ``cluster_singleton_not_evidence`` already
    says for free. The sibling axes report an unjudged measurement because
    their lookup happens either way; this one is two more requests per target,
    which on a campaign of any size is the difference between a screen and a
    rate limit.

    It also means no existing caller's behaviour changed when this was added,
    which is why the suite's strict "unplanned lookup" readers still pass --
    they are not being humoured, the calls genuinely are not made.
    """
    asked: list[str] = []

    def read(url: str) -> object:
        asked.append(url)
        if "uniref/search" in url:
            return _uniref()
        if "rest.ensembl.org" in url or "gene_primary" in url:
            raise AssertionError(f"asked with no threshold: {url}")
        return _members()

    record = _paralog_record(_KRAS, read, None)
    assert record["state"] == "untested"
    assert record["blocker"] == "cluster_singleton_not_evidence"
    assert not any("ensembl" in url for url in asked)
    assert not any("gene_primary" in url for url in asked)
    assert "compara_examined" not in record and "compara_blocker" not in record


def test_naming_a_cut_is_what_turns_the_measurement_on() -> None:
    """The same target, the same services, and now an answer."""
    reachable = reader(homology=_homology([("ENSG00000174775", 85.0, 85.6)]))
    assert _paralog_record(_KRAS, reachable, None)["measurement"] == 0
    judged = _paralog_record(_KRAS, reachable, 50.0)
    assert judged["state"] == "hit"
    assert judged["measurement"] == pytest.approx(85.6)


# ------------------------------------------------- the count is not the measure


def test_many_distant_paralogs_do_not_make_a_hit() -> None:
    """ERBB2's real shape: 53 paralogs, median identity 17.4 per cent.

    Judged on how many there are, this is the loudest signal in the screen.
    Judged on identity -- the question actually being asked -- it is a clear,
    and the fifty-odd distant kinases say nothing about cross-reactivity.
    """
    distant = [(f"ENSG{index:08d}", 16.0 + index % 5, 17.0 + index % 5) for index in range(53)]
    record = _paralog_record(_KRAS, reader(homology=_homology(distant)), 50.0)
    assert record["state"] == "clear", "a count of 53 must not become a liability"
    assert record["compara_examined"] == 53
    assert record["measurement"] < 50.0


def test_one_close_paralog_among_many_distant_ones_is_still_found() -> None:
    """The other direction: the loudest is what is reported, not the typical."""
    mixed = [(f"ENSG{index:08d}", 16.0, 17.0) for index in range(52)]
    mixed.append(("ENSG00000146648", 48.9, 50.7))
    record = _paralog_record(_KRAS, reader(homology=_homology(mixed)), 50.0)
    assert record["state"] == "hit"
    assert record["closest_paralog"] == "ENSG00000146648"


# ----------------------------------------------------- identity is symmetric


def test_a_pair_measures_the_same_whichever_protein_was_asked_about() -> None:
    """Compara's ``perc_id`` is identity over the length of its own sequence.

    So a pair has two numbers -- ERBB2 and EGFR are 50.7 per cent identical
    over EGFR and 48.9 over ERBB2 -- and reading only the target's made the
    verdict depend on which protein the run happened to name. Across a 50 per
    cent cut that made ERBB2 a hit and EGFR a clear for the same relationship,
    and against the live service it suppressed EGFR's hit outright: 48.9
    directional became 51.9 symmetric once the pair with ERBB4 was read whole.
    """
    examined, scored = _compara_paralogs(reader(homology=_homology([("ENSG_EGFR", 48.9243, 50.7438)])), "ERBB2")
    assert examined == 1
    assert scored[0][0] == pytest.approx(50.7438), "the larger share of the pair"

    flipped, scored_back = _compara_paralogs(reader(homology=_homology([("ENSG_ERBB2", 50.7438, 48.9243)])), "EGFR")
    assert flipped == 1
    assert scored_back[0][0] == pytest.approx(50.7438), "the same number from the other side"


def test_a_homology_missing_one_direction_still_uses_the_other() -> None:
    """Half a pair is a measurement; none of it is not."""
    payload = _homology([("ENSG_X", 40.0, 60.0)])
    del payload["data"][0]["homologies"][0]["source"]["perc_id"]
    _examined, scored = _compara_paralogs(reader(homology=payload), "X")
    assert scored[0][0] == pytest.approx(60.0)


# ------------------------------------- what must survive Compara being no help


def test_a_positive_uniref_cluster_is_never_downgraded_by_compara() -> None:
    """Cluster membership is positive evidence of a family.

    Compara is consulted only where UniRef50 found nothing, so a hit from
    membership cannot be softened by a Compara answer -- including by Compara
    being unreachable, which is the shape that would be easiest to get wrong.
    """
    members = {
        "results": [
            {"primaryAccession": _KRAS, "genes": [{"geneName": {"value": "KRAS"}}]},
            {"primaryAccession": "P01112", "genes": [{"geneName": {"value": "HRAS"}}]},
        ]
    }
    record = _paralog_record(
        _KRAS,
        reader(members=members, homology=OSError("ensembl is down")),
        50.0,
    )
    assert record["state"] == "hit"
    assert record["measurement"] == 1, "one other member of the cluster"
    assert "HRAS" in record["paralogs"]
    assert "compara_examined" not in record, "Compara is not asked when membership answered"


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (OSError("unreachable"), "lookup_unavailable"),
        (ValueError("bad json"), "lookup_unavailable"),
        (LookupMissError("no entry"), "gene_not_in_compara"),
    ],
)
def test_compara_failing_leaves_the_old_answer_exactly_as_it_was(failure: Exception, expected: str) -> None:
    """Falling back loses nothing: the axis had no clear at all before this.

    The blocker that was there is kept, and the Compara trouble is recorded
    beside it rather than replacing it -- a service being unreachable is not a
    measurement, and must not read as one.
    """
    record = _paralog_record(_KRAS, reader(homology=failure), 50.0)
    assert record["state"] == "untested"
    assert record["blocker"] == "cluster_singleton_not_evidence"
    assert record["compara_blocker"] == expected
    assert "threshold" not in record


def test_a_gene_with_no_symbol_cannot_be_asked_and_says_so() -> None:
    """Compara's endpoint is keyed by symbol, so no symbol is no lookup."""
    record = _paralog_record(_KRAS, reader(symbol={"results": []}), 50.0)
    assert record["blocker"] == "cluster_singleton_not_evidence"
    assert record["compara_blocker"] == "gene_symbol_not_found"


def test_a_target_that_is_not_an_accession_never_reaches_the_network() -> None:
    """A gene symbol is not an accession, and is reported rather than guessed."""
    asked: list[str] = []

    def read(url: str) -> object:
        asked.append(url)
        return {}

    record = _paralog_record("gene_9GVN_1", read, 50.0)
    assert record["blocker"] == "target_not_accession"
    assert not asked


# ------------------------------------------------------------- the whole record


def test_the_threshold_travels_with_the_limits_object() -> None:
    """Named on ``TargetLimits`` like every other cut the caller supplies."""
    assert TargetLimits().paralog_identity_max is None
    assert TargetLimits(paralog_identity_max=50.0).paralog_identity_max == 50.0


def test_the_record_names_the_method_it_measured_with() -> None:
    """Two sources now answer this axis, and a reader of the record has to be
    able to tell which one did."""
    record = _paralog_record(_KRAS, reader(homology=_homology([("ENSG_X", 60.0, 61.0)])), 50.0)
    assert record["method"] == "ensembl_compara_paralogues_identity"
    assert record["cluster"] == f"UniRef50_{_KRAS}"
