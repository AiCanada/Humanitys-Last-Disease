"""OASis identity, AbLang-2 antibody-likeness, and the line between them.

Both backends are real and installed, but both are slow to load: the OASis
peptide set is a 40 MB gzip and AbLang-2 loads torch weights. Neither is
loaded here. The arithmetic and the labelling are what these tests pin.
"""

from __future__ import annotations

import pytest

from rfproteina.disease import humanness
from rfproteina.disease.humanness import (
    ABLANG2_METHOD,
    OASIS_SUBJECT_THRESHOLD_PERCENT,
    HumannessUnavailableError,
    ablang2_naturalness,
    humanness_options,
    humanness_report,
    oasis_identity,
    peptide_windows,
)


class _FakeDb:
    """Stands in for the packaged OASis peptide set."""

    def __init__(self, peptides: set[str]) -> None:
        self.peptides = peptides

    def contains(self, peptide: str) -> bool:
        return peptide in self.peptides


def test_windows_are_nine_long_and_drop_non_standard_residues() -> None:
    assert peptide_windows("ACDEFGHIK") == ["ACDEFGHIK"]
    assert peptide_windows("ACDEFGHI") == []
    assert peptide_windows("XCDEFGHIKL") == ["CDEFGHIKL"]


def test_the_identity_is_matched_over_scored_and_carries_its_threshold(monkeypatch) -> None:
    sequence = "ACDEFGHIKLM"  # three 9-mers
    windows = peptide_windows(sequence)
    monkeypatch.setattr(humanness, "_oasis_db", lambda: _FakeDb({windows[0], windows[2]}))
    record = oasis_identity({"H": sequence})
    chain = record["chains"][0]
    assert chain["peptides_scored"] == 3
    assert chain["peptides_matched"] == 2
    assert chain["identity"] == pytest.approx(2 / 3, abs=1e-5)
    assert record["subject_threshold_percent"] == OASIS_SUBJECT_THRESHOLD_PERCENT
    assert record["pass_cut"] is None


def test_a_domain_with_nothing_to_score_is_blocked_not_scored_zero(monkeypatch) -> None:
    monkeypatch.setattr(humanness, "_oasis_db", lambda: _FakeDb(set()))
    chain = oasis_identity({"H": "ACDEF"})["chains"][0]
    assert chain["identity"] is None
    assert chain["blocker"] == "no_scorable_peptides"


def test_an_identity_of_zero_is_reported_when_peptides_were_actually_looked_up(monkeypatch) -> None:
    monkeypatch.setattr(humanness, "_oasis_db", lambda: _FakeDb(set()))
    chain = oasis_identity({"H": "ACDEFGHIK"})["chains"][0]
    assert chain["identity"] == 0.0
    assert chain["peptides_scored"] == 1


def test_ablang2_is_reported_as_antibody_likeness_and_never_as_humanness(monkeypatch) -> None:
    monkeypatch.setattr(humanness, "_ablang2_model", lambda device="cpu", ncpu=1: lambda seqs, mode: [-0.5])
    record = ablang2_naturalness([("design", "EVQ", "DIQ")])
    assert record["measures"] == "antibody_likeness"
    assert "humanness" not in str(record["scale"]).lower()
    assert "88 per cent human" in record["not_humanness_reason"]
    assert record["chains"][0]["score"] == -0.5
    assert record["chains"][0]["input"] == "paired"
    assert record["pass_cut"] is None


def test_a_single_chain_input_is_labelled_so_scores_are_not_compared_across_shapes(monkeypatch) -> None:
    monkeypatch.setattr(humanness, "_ablang2_model", lambda device="cpu", ncpu=1: lambda seqs, mode: [-1.0])
    record = ablang2_naturalness([("design", "EVQ", "")])
    assert record["chains"][0]["input"] == "single_chain"


def test_no_pairs_is_an_error_rather_than_an_empty_record() -> None:
    with pytest.raises(ValueError, match="no_pairs"):
        ablang2_naturalness([])


def test_a_missing_backend_blocks_its_own_section_and_does_not_fall_back(monkeypatch) -> None:
    def _raise() -> None:
        raise HumannessUnavailableError("promb_import_failed:test")

    monkeypatch.setattr(humanness, "_oasis_db", _raise)
    record = humanness_report({"H": "ACDEFGHIK"})
    assert record["humanness"]["blocker"] == "promb_import_failed:test"
    assert record["antibody_likeness"]["blocker"] == "no_paired_chains_supplied"
    assert record["antibody_likeness"]["method"] == ABLANG2_METHOD


def test_every_option_states_a_licence_and_how_it_was_verified() -> None:
    options = humanness_options()
    assert options
    for option in options:
        assert option.licence.strip()
        assert option.licence_verified.strip()
    igfold = [option for option in options if option.name == "IgFold"]
    assert igfold and igfold[0].installed_here is False
    assert "academic" in igfold[0].licence
