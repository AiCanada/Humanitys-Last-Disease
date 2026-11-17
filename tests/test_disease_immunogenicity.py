"""The class II interface, its documented unavailable path, and the converter.

Nothing academic-only is installed or called. The integration test writes a
table and hands it to ``liability.py``, which is the only way to show the
converter actually satisfies the contract rather than looking like it does.
"""

from __future__ import annotations

import pytest

from rfproteina.disease import immunogenicity as imm
from rfproteina.disease.immunogenicity import (
    CLASS_II_OPTIONS,
    DR_PANEL,
    ClassIIUnavailableError,
    class_ii_status,
    convert_class_ii_rows,
    predict_class_ii,
    register_backend,
    windows,
    write_class_ii_table,
)
from rfproteina.disease.liability import measure_sequence_liability

_SEQUENCE = "MKTIIALSYIFCLVFADYKDDDDKGSAAAPTSSSTKKTQLQLEHLLLDLQMILNGINNYKNPKLTRMLTFKFYMPKKATELKHLQCLEEEL"


@pytest.fixture(autouse=True)
def _no_backends():
    """Each test starts with no backend registered, and leaves none behind."""
    imm._BACKENDS.clear()
    yield
    imm._BACKENDS.clear()


def test_class_ii_is_unavailable_and_says_why() -> None:
    status = class_ii_status()
    assert status["available"] is False
    assert status["reason"] == "no_permissively_licensed_class_ii_predictor"
    assert status["panel"] == list(DR_PANEL)
    assert "consensus" in str(status["scale_required_for_ada"]).lower()


def test_every_option_quotes_its_licence_and_the_date_it_was_read() -> None:
    assert CLASS_II_OPTIONS
    for option in CLASS_II_OPTIONS:
        assert option.licence.strip()
        assert option.licence_quote.strip()
        assert option.licence_read_on.strip()
        assert option.commercial_use.strip()


def test_the_pypi_licence_contradiction_is_recorded_not_smoothed_over() -> None:
    nuggets = [option for option in CLASS_II_OPTIONS if "MHCnuggets" in option.name]
    assert nuggets
    assert "Apache" in nuggets[0].note
    assert "non-commercial" in nuggets[0].licence.lower()


def test_predicting_without_a_backend_raises_rather_than_returning_an_empty_table() -> None:
    with pytest.raises(ClassIIUnavailableError, match="no_permissively_licensed_class_ii_predictor"):
        predict_class_ii(_SEQUENCE)


def test_a_registered_backend_is_handed_fifteen_mers() -> None:
    seen: dict[str, object] = {}

    def _backend(peptides, alleles):
        seen["peptides"] = list(peptides)
        seen["alleles"] = list(alleles)
        return "peptide,allele,percentile_rank\n"

    register_backend("fake", _backend)
    predict_class_ii(_SEQUENCE)
    assert class_ii_status()["available"] is True
    assert all(len(peptide) == 15 for peptide in seen["peptides"])
    assert seen["alleles"] == list(DR_PANEL)


def test_windows_refuse_a_nonsense_width() -> None:
    with pytest.raises(ValueError, match="width_must_be_positive"):
        windows(_SEQUENCE, 0)


def test_a_non_consensus_method_is_warned_about_before_the_axis_refuses_it() -> None:
    rows = [{"peptide": "A" * 15, "allele": allele, "percentile_rank": 2.0} for allele in DR_PANEL]
    _table, warnings = convert_class_ii_rows(rows, method="netmhciipan_4.3_el")
    assert any("rank_scale_unknown" in warning for warning in warnings)
    _table, clean = convert_class_ii_rows(rows, method="IEDB recommended (consensus)")
    assert clean == []


def test_a_partial_panel_is_warned_about() -> None:
    rows = [{"peptide": "A" * 15, "allele": DR_PANEL[0], "percentile_rank": 2.0}]
    _table, warnings = convert_class_ii_rows(rows, method="consensus")
    assert any("allele_panel_incomplete" in warning for warning in warnings)


def test_an_affinity_is_only_called_nanomolar_when_one_was_given() -> None:
    rows = [{"peptide": "A" * 15, "allele": DR_PANEL[0], "percentile_rank": 2.0}]
    table, _warnings = convert_class_ii_rows(rows, method="consensus")
    header = table.splitlines()[0].split("\t")
    first = dict(zip(header, table.splitlines()[1].split("\t"), strict=True))
    assert first["ic50_nm"] == ""
    assert first["unit"] == ""
    with_affinity = [{**rows[0], "ic50_nm": 120.0}]
    table, _warnings = convert_class_ii_rows(with_affinity, method="consensus")
    second = dict(zip(header, table.splitlines()[1].split("\t"), strict=True))
    assert second["ic50_nm"] == "120.0"
    assert second["unit"] == "nM"


def test_a_row_without_a_peptide_or_an_allele_is_refused() -> None:
    with pytest.raises(ValueError, match="peptide_and_allele_required"):
        convert_class_ii_rows([{"peptide": "", "allele": "DRB1*03:01"}], method="consensus")
    with pytest.raises(ValueError, match="no_rows"):
        convert_class_ii_rows([], method="consensus")


def test_the_written_table_reaches_a_verdict_on_the_ada_axis(tmp_path) -> None:
    """The point of the converter: liability.py reads it and the axis moves.

    A median consensus rank of 2.0 over the whole DR panel is a promiscuous
    binder under Paul et al.'s rule, so the axis should be a hit. The peptide
    is taken from the sample sequence, because the axis checks that the table
    belongs to this design.
    """
    peptide = _SEQUENCE[20:35]
    rows = [{"peptide": peptide, "allele": allele, "percentile_rank": 2.0} for allele in DR_PANEL]
    written = write_class_ii_table(rows, tmp_path, method="IEDB recommended (consensus)")
    assert written["warnings"] == []
    axes, _ = measure_sequence_liability([("design1", _SEQUENCE)], directory=tmp_path, binder_kind="peptide")
    assert axes["ada"]["state"] == "hit"
    assert axes["ada"]["method"] == "mhc_ii_dr_breadth_paul_2015"


def test_a_non_consensus_table_leaves_the_ada_axis_untested(tmp_path) -> None:
    peptide = _SEQUENCE[20:35]
    rows = [{"peptide": peptide, "allele": allele, "percentile_rank": 2.0} for allele in DR_PANEL]
    write_class_ii_table(rows, tmp_path, method="netmhciipan_4.3_el")
    axes, _ = measure_sequence_liability([("design1", _SEQUENCE)], directory=tmp_path, binder_kind="peptide")
    assert axes["ada"]["state"] == "untested"
    assert axes["ada"]["blocker"] == "rank_scale_unknown"
