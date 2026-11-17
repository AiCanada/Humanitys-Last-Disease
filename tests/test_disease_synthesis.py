"""Whether a construct can be ordered: GC windows, repeats, hairpins, runs.

These are the vendor-facing terms ``oligo.py`` does not carry. Nothing here is
a published specification, so one test exists to make sure every bar keeps
saying so.
"""

from __future__ import annotations

import pytest

from rfproteina.disease.synthesis import (
    SYNTHESIS_BARS,
    SequenceUnreadableError,
    screen_sequence,
    verdicts,
)

_BALANCED = "ATGCATGCTAGCTAGCATCGATCGTACGATCGATCGTAGCTAGCTAGCATCGATCGTACGTAGCTAGCATCGATCG"


def test_a_balanced_construct_reports_its_gc_and_window_range() -> None:
    report = screen_sequence(_BALANCED)
    assert 0.4 < report.gc_fraction < 0.6
    assert report.min_window_gc is not None
    assert report.max_window_gc >= report.min_window_gc
    assert report.alphabet == "dna"


def test_a_short_construct_reports_no_window_rather_than_a_fake_one() -> None:
    report = screen_sequence("ATGCATGCAT")
    assert report.min_window_gc is None
    assert verdicts(report)["min_window_gc"]["state"] == "untested"


def test_a_homopolymer_run_is_found_with_the_base_that_made_it() -> None:
    report = screen_sequence("ATGC" + "G" * 14 + "ATGC")
    assert report.longest_homopolymer == 14
    assert report.homopolymer_base == "G"
    assert verdicts(report)["longest_homopolymer"]["state"] == "hit"


def test_a_direct_repeat_is_found_and_reported_back() -> None:
    unit = "ATGCATCGATCGTTACGATC"
    report = screen_sequence(unit + "AAGGTT" + unit)
    assert report.longest_repeat >= len(unit)
    assert report.repeat_sequence is not None


def test_a_hairpin_stem_is_found_where_it_sits() -> None:
    stem = "GGCCTTAA"
    complement = "TTAAGGCC"
    report = screen_sequence("AT" + stem + "GCGCG" + complement + "AT")
    assert report.longest_hairpin_stem >= 8
    # Two stems open in this sequence, at base 1 and base 3. The first found is
    # the one reported, and this says so rather than asserting the one the
    # author had in mind.
    assert report.hairpin_at == 1
    # A purine-only stretch has no reverse-complement of itself anywhere, so it
    # is the honest negative. An ATGC repeat is NOT: that pattern is
    # self-complementary and the scan reports a stem in it, correctly.
    assert screen_sequence("AAAGAAAGAAAGAAAGAAAGAAAG").longest_hairpin_stem == 0


def test_an_rna_sequence_is_accepted_and_a_protein_is_not() -> None:
    assert screen_sequence("AUGCAUGCAUGC").alphabet == "rna"
    with pytest.raises(SequenceUnreadableError, match="not_a_nucleic"):
        screen_sequence("MKTIIALSYIF")
    with pytest.raises(SequenceUnreadableError, match="empty_sequence"):
        screen_sequence("")


def test_a_forbidden_site_is_reported_at_every_position_it_appears() -> None:
    report = screen_sequence("GGTCTCAAAAAAGGTCTC", forbidden={"BsaI": "GGTCTC"})
    site = report.forbidden_sites[0]
    assert site["positions"] == [1, 13]
    assert verdicts(report)["forbidden_sites"]["state"] == "hit"


def test_every_bar_still_admits_it_is_only_a_suggestion() -> None:
    for name, rule in SYNTHESIS_BARS.items():
        assert "suggested default pending calibration" in str(rule["source"]), name
