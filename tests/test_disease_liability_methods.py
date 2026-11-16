"""The published methods behind the sequence bar, checked against their sources.

Every constant here is pinned to a primary source rather than to the value the
module happens to hold, because a wrong constant in one of these methods is
silent: it looks exactly like a measurement and the evidence chain repeats it.

- Calis et al., PLoS Comput Biol 2013;9:e1003266. Table 2 gives the position
  Kullback-Leibler divergences (positions 1, 2 and 9 are anchors, masked), and
  the IEDB reference implementation ``predict_immunogenicity.py`` v1.1
  (downloads.iedb.org/tools/immunogenicity) carries the 20 log-enrichment
  values as ``immunoscale``.
- Paul et al., J Immunol Methods 422:28-34 (2015): "select peptides based on
  predicted median binding percentiles for a set of seven DRB1 and DRB3/4/5
  alleles ... This corresponded to an IEDB consensus percentile rank of 20.0,
  which could be used as a universal prediction threshold." The seven are the
  panel IEDB's TepiTool offers as the 7-allele method.
- IEDB "Selecting thresholds (cut-offs) for MHC class I and II binding
  predictions": class I percentile rank <= 1 or IC50 <= 500 nM, class II
  percentile rank <= 10 or IC50 <= 1000 nM.
"""

from pathlib import Path

import pytest

from rfproteina.disease.liability import (
    _CALIS_SCALE,
    _CALIS_WEIGHT,
    _DR_MEDIAN_MAX,
    _DR_PANEL,
    _MHC_I_IC50,
    _MHC_I_RANK,
    _MHC_II_IC50,
    _MHC_II_RANK,
    _calis_score,
    measure_sequence_liability,
)
from tests.helpers.disease import UBIQUITIN

_UBIQUITIN = UBIQUITIN
#: The IEDB reference tool's ``immunoscale``, typed from its source.
_IEDB_SCALE = {
    "A": 0.127,
    "C": -0.175,
    "D": 0.072,
    "E": 0.325,
    "F": 0.380,
    "G": 0.110,
    "H": 0.105,
    "I": 0.432,
    "K": -0.700,
    "L": -0.036,
    "M": -0.570,
    "N": -0.021,
    "P": -0.036,
    "Q": -0.376,
    "R": 0.168,
    "S": -0.537,
    "T": 0.126,
    "V": 0.134,
    "W": 0.719,
    "Y": -0.012,
}
#: Calis 2013 Table 2, by position. Positions 1, 2 and 9 are NA (anchors).
_PAPER_DIVERGENCE = {3: 0.10, 4: 0.31, 5: 0.30, 6: 0.29, 7: 0.26, 8: 0.18}
#: The 7-allele method as IEDB's TepiTool lists it.
_PAUL_PANEL = (
    "DRB1*03:01",
    "DRB1*07:01",
    "DRB1*15:01",
    "DRB3*01:01",
    "DRB3*02:02",
    "DRB4*01:01",
    "DRB5*01:01",
)


def _published_calis(peptide: str) -> float:
    """S = sum over non-anchor p of w_p * E(A(L,p)), written from the paper."""
    return sum(weight * _IEDB_SCALE[peptide[position - 1]] for position, weight in _PAPER_DIVERGENCE.items())


def _windows(sequence: str, width: int = 9) -> list[str]:
    return [sequence[start : start + width] for start in range(len(sequence) - width + 1)]


def test_the_calis_scale_is_the_published_one() -> None:
    """All twenty log-enrichment values, against the IEDB reference tool."""
    assert _CALIS_SCALE == _IEDB_SCALE


def test_the_calis_weights_are_the_paper_divergences() -> None:
    """Table 2's divergences at 3-8, and zero where the paper masks an anchor."""
    assert len(_CALIS_WEIGHT) == 9
    for position, divergence in _PAPER_DIVERGENCE.items():
        assert _CALIS_WEIGHT[position - 1] == divergence
    for anchor in (1, 2, 9):
        assert _CALIS_WEIGHT[anchor - 1] == 0.0


def test_calis_agrees_with_the_published_formula_on_every_window() -> None:
    """An independent implementation of the paper's equation, window by window."""
    for sequence in (_UBIQUITIN, "LLVVIIFFAAMMLLVVIIFFWWAAVVLLIIGG", "HCYHCYWKACDHHCYYTCGHMHCYEHCYRKHCY"):
        for window in _windows(sequence):
            assert _calis_score(window) == pytest.approx(_published_calis(window), abs=1e-12)


def test_the_documented_binding_cutoffs_are_the_iedb_ones() -> None:
    assert (_MHC_I_RANK, _MHC_I_IC50) == (1.0, 500.0)
    assert (_MHC_II_RANK, _MHC_II_IC50) == (10.0, 1000.0)


def test_the_ada_panel_and_threshold_are_paul_2015() -> None:
    assert _DR_PANEL == _PAUL_PANEL
    assert _DR_MEDIAN_MAX == 20.0


@pytest.mark.parametrize(
    ("rank", "state"),
    [("1.0", "hit"), ("1.0001", "clear")],
)
def test_a_class_i_rank_at_the_cut_binds(tmp_path: Path, rank: str, state: str) -> None:
    """Equality at the cut is binding: IEDB's recommendation is rank <= 1."""
    (tmp_path / "mhc_i.tsv").write_text(f"peptide\trank\nACDEFGHIK\t{rank}\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability(
        [("binder_0", "ACDEFGHIKLMNPQRSTVWY")], directory=tmp_path, binder_kind="peptide"
    )
    assert axes["mhc_i"]["state"] == state


def test_an_unreadable_rank_cannot_restrict_immunogenicity(tmp_path: Path) -> None:
    """A table ``mhc_i`` refuses must not quietly restrict this axis instead.

    A German or French Excel export writes a rank of 0.5 as ``0,5``. Those
    cells do not parse, so the windows they belong to left the presented set
    while every integral non-binder stayed: the axis cleared at -0.0806 on the
    one benign ligand left, with the four binding windows -- the highest of
    them Calis-positive at +0.2825 -- dropped, and ``mhc_i`` reporting
    ``cell_unparsed`` on the very same file.
    """
    windows = _windows(_UBIQUITIN)
    hidden = {"MQIFVKTLT", "TLTGKTITL", "TGKTITLEV", "GKTITLEVE"}
    lines = ["peptide\trank"]
    for window in windows:
        if window in hidden:
            lines.append(f"{window}\t0,5")
        elif window == "QIFVKTLTG":
            lines.append(f"{window}\t0.5")
        else:
            lines.append(f"{window}\t5.0")
    (tmp_path / "mhc_i.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([("binder_0", _UBIQUITIN)], directory=tmp_path)
    record = axes["immunogenicity"]
    assert record["state"] == "untested"
    assert record["blocker"] == "cell_unparsed"
    assert record["unparsed_columns"] == ["rank"]
    assert record["measurement"] is None
    assert record["presentation"] == "restricted"
    assert axes["mhc_i"]["blocker"] == "cell_unparsed"


def test_the_same_table_with_readable_ranks_is_a_hit(tmp_path: Path) -> None:
    """The binders the comma hid are Calis-positive, which is the point."""
    windows = _windows(_UBIQUITIN)
    presented = {"MQIFVKTLT", "TLTGKTITL", "TGKTITLEV", "GKTITLEVE", "QIFVKTLTG"}
    lines = ["peptide\trank"]
    for window in windows:
        lines.append(f"{window}\t0.5" if window in presented else f"{window}\t5.0")
    (tmp_path / "mhc_i.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([("binder_0", _UBIQUITIN)], directory=tmp_path)
    record = axes["immunogenicity"]
    assert record["state"] == "hit"
    assert record["measurement"] == 0.2825
    assert record["positive_windows"] == 4
