"""Units, scales and widths a predictor table does not state.

Every test here is about a number whose meaning the file leaves open. The rule
the module holds is that a verdict must never say more than was measured, with
one asymmetry: a binder that was found is positive evidence however little else
was read, while an absence of binders is only as wide as the cells that could
be read. So an unreadable unit refuses a clear and leaves a hit standing.

Private helpers are imported inside the test that needs them, so this file
still collects against a tree that does not have them yet and each test fails
on its own merits.
"""

from __future__ import annotations

from pathlib import Path

from rfproteina.disease.liability import measure_sequence_liability
from tests.helpers.disease import UBIQUITIN

#: Human ubiquitin, P0CG47 residues 1-76. 68 distinct 9-mer windows and 65
#: distinct 12-mer windows.
_UBIQUITIN = UBIQUITIN


def test_a_bare_ic50_column_states_no_unit_and_is_not_read_as_nanomolar(tmp_path: Path) -> None:
    """``ic50`` names the analyte, not the unit; the cut is 500 nM.

    A table written in micromolar under a bare ``ic50`` header is a
    thousandfold out: 2.0 is 2 uM, a non-binder two orders above the cutoff,
    and it was published as ``hit`` at 2.0. The design side errs toward hit
    there, but the same cell on the reference side counts as a binder and
    inflates the bar, which clears designs.
    """
    peptide = "MQIFVKTLT"
    table = tmp_path / "mhc_i.tsv"
    table.write_text(f"allele\tpeptide\tic50\nHLA-A*02:01\t{peptide}\t2.0\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([("b", peptide)], directory=tmp_path, binder_kind="peptide")
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "ic50_not_nm"
    assert record["not_nm_columns"] == ["ic50"]
    # Nothing was read, so no number is published as the measurement either.
    assert record["measurement"] is None

    # A unit cell saying nanomolar is what makes the same number a verdict.
    table.write_text(
        f"allele\tpeptide\tic50\tunits\nHLA-A*02:01\t{peptide}\t2.0\tnM\n",
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability([("b", peptide)], directory=tmp_path, binder_kind="peptide")
    assert axes["mhc_i"]["state"] == "hit"
    assert axes["mhc_i"]["measurement"] == 2.0
    assert "not_nm_columns" not in axes["mhc_i"]

    # And a column whose own name states the unit needs no unit cell at all.
    for column in ("ic50_nm", "affinity_nm", "aff_nm"):
        table.write_text(
            f"allele\tpeptide\t{column}\nHLA-A*02:01\t{peptide}\t2.0\n",
            encoding="utf-8",
        )
        axes, _by_example = measure_sequence_liability([("b", peptide)], directory=tmp_path, binder_kind="peptide")
        assert axes["mhc_i"]["state"] == "hit", column
        assert axes["mhc_i"]["measurement"] == 2.0, column


def test_a_micromolar_unit_cell_beats_a_column_name_that_says_nanomolar(tmp_path: Path) -> None:
    """``ic50_nm`` beside a ``uM`` unit cell is a table contradicting itself."""
    peptide = "MQIFVKTLT"
    (tmp_path / "mhc_i.tsv").write_text(
        f"allele\tpeptide\tic50_nm\tunit\nHLA-A*02:01\t{peptide}\t2.0\tuM\n",
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability([("b", peptide)], directory=tmp_path, binder_kind="peptide")
    assert axes["mhc_i"]["state"] == "untested"
    assert axes["mhc_i"]["blocker"] == "ic50_not_nm"
    assert axes["mhc_i"]["not_nm_columns"] == ["ic50_nm"]


def test_an_unstated_affinity_unit_refuses_a_clear_and_leaves_a_hit_standing(tmp_path: Path) -> None:
    """The asymmetry, on one table.

    A rank of 0.4 is at the IEDB class I cut and is a hit whatever the affinity
    column beside it holds -- a liability measured cannot be deleted by a
    sibling cell that could not be read. A rank of 50 is above the cut, so the
    verdict there is an absence claim, and an affinity column nobody can read
    cannot support one.
    """
    peptide = "MQIFVKTLT"
    table = tmp_path / "mhc_i.tsv"
    table.write_text(
        f"allele\tpeptide\tpercentile_rank\tic50\nHLA-A*02:01\t{peptide}\t0.4\t2000\n",
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability([("b", peptide)], directory=tmp_path, binder_kind="peptide")
    record = axes["mhc_i"]
    assert record["state"] == "hit"
    assert record["measurement"] == 0.4
    assert record["blocker"] is None
    # The hit stands, and the record still says a column was left unread.
    assert record["not_nm_columns"] == ["ic50"]

    table.write_text(
        f"allele\tpeptide\tpercentile_rank\tic50\nHLA-A*02:01\t{peptide}\t50\t2000\n",
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability([("b", peptide)], directory=tmp_path, binder_kind="peptide")
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "ic50_not_nm"
    assert record["not_nm_columns"] == ["ic50"]
    # 50 is still the rank the table holds, and it is still reported.
    assert record["measurement"] == 50.0


def test_a_reference_affinity_of_unstated_unit_is_not_a_bar(tmp_path: Path) -> None:
    """On the reference side an unread unit moves the verdict toward clear.

    Read as nanomolar, every 2.0 in a micromolar albumin export counts as a 2 nM
    binder, and the bar it sets clears designs that are hits against the real
    one. Left unread it understates the bar. Neither is a bar.
    """
    windows = [_UBIQUITIN[index : index + 9] for index in range(68)]
    rows = ["allele\tpeptide\tpercentile_rank"]
    rows += [f"HLA-A*02:01\t{window}\t{0.4 if index < 4 else 50.0}" for index, window in enumerate(windows)]
    (tmp_path / "mhc_i.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    reference = tmp_path / "mhc_i_reference.tsv"
    reference.write_text(
        "protein\tpeptide\tlength\tallele\tic50\n"
        "P02768\tDAHKSEVAH\t609\tHLA-A*02:01\t2.0\n"
        "P02768\tAHKSEVAHR\t609\tHLA-A*02:01\t3.0\n",
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability(
        [("binder_0", _UBIQUITIN)], directory=tmp_path, binder_kind="protein"
    )
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "reference_ic50_not_nm"
    assert record["not_nm_columns"] == ["ic50"]
    assert "reference_density" not in record

    # The same two rows under a header that states the unit do set a bar.
    reference.write_text(
        "protein\tpeptide\tlength\tallele\tic50_nm\n"
        "P02768\tDAHKSEVAH\t609\tHLA-A*02:01\t2.0\n"
        "P02768\tAHKSEVAHR\t609\tHLA-A*02:01\t3.0\n",
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability(
        [("binder_0", _UBIQUITIN)], directory=tmp_path, binder_kind="protein"
    )
    assert axes["mhc_i"]["blocker"] is None
    assert axes["mhc_i"]["state"] == "hit"


def test_a_table_whose_only_numbers_are_bare_affinities_names_the_unit(tmp_path: Path) -> None:
    """``score_absent`` sent the reader to add a column that was already there."""
    peptide = "MQIFVKTLT"
    (tmp_path / "mhc_i.tsv").write_text(
        f"allele\tpeptide\tba_affinity\nHLA-A*02:01\t{peptide}\t2.0\n",
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability([("b", peptide)], directory=tmp_path, binder_kind="peptide")
    assert axes["mhc_i"]["state"] == "untested"
    assert axes["mhc_i"]["blocker"] == "ic50_not_nm"
    assert axes["mhc_i"]["not_nm_columns"] == ["ba_affinity"]


def test_an_unstated_affinity_unit_cannot_clear_the_immunogenicity_axis(tmp_path: Path) -> None:
    """The presented set is only as wide as the cells that could be read."""
    presented = "ENVKAKIQD"
    (tmp_path / "mhc_i.tsv").write_text(
        f"allele\tpeptide\tpercentile_rank\tic50\nHLA-A*02:01\t{presented}\t0.4\t2000\n",
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability([("binder_0", presented)], directory=tmp_path)
    record = axes["immunogenicity"]
    assert record["state"] == "untested"
    assert record["blocker"] == "ic50_not_nm"
    assert record["not_nm_columns"] == ["ic50"]


def test_a_rank_that_could_be_a_fraction_is_recorded_on_the_reference(tmp_path: Path) -> None:
    """0.005 is 0.5 per cent in one export and 0.005 per cent here.

    Nothing published tells a fraction of one from a very small percentage, so
    no floor is invented and no cell is rescaled. But a rank above 1 cannot be
    a fraction at all, so a table whose every rank is at or below 1 is the
    ambiguous case, and the reference side is where the ambiguity clears
    designs: the percentage reading calls more rows binding and sets a higher
    bar. The record says so beside the verdict.
    """
    from rfproteina.disease.liability import _rank_scale_ambiguous

    assert _rank_scale_ambiguous([{"percentile_rank": "0.005"}]) is True
    assert _rank_scale_ambiguous([{"percentile_rank": "1.0"}]) is True
    # One rank above 1 settles it: a fraction cannot exceed one.
    assert _rank_scale_ambiguous([{"percentile_rank": "0.005"}, {"percentile_rank": "1.01"}]) is False
    assert _rank_scale_ambiguous([{"percentile_rank": ""}]) is False

    windows = [_UBIQUITIN[index : index + 9] for index in range(68)]
    rows = ["allele\tpeptide\tpercentile_rank"]
    rows += [f"HLA-A*02:01\t{window}\t{0.4 if index < 4 else 50.0}" for index, window in enumerate(windows)]
    (tmp_path / "mhc_i.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    reference = tmp_path / "mhc_i_reference.tsv"
    reference.write_text(
        "protein\tpeptide\tlength\tallele\tpercentile_rank\n"
        "P02768\tDAHKSEVAH\t609\tHLA-A*02:01\t0.3\n"
        "P02768\tAHKSEVAHR\t609\tHLA-A*02:01\t0.4\n",
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability(
        [("binder_0", _UBIQUITIN)], directory=tmp_path, binder_kind="protein"
    )
    record = axes["mhc_i"]
    assert record["blocker"] is None
    assert record["reference_rank_scale_ambiguous"] is True

    # One non-binding row in the reference settles the scale, and then the
    # verdict carries no such note.
    reference.write_text(
        "protein\tpeptide\tlength\tallele\tpercentile_rank\n"
        "P02768\tDAHKSEVAH\t609\tHLA-A*02:01\t0.3\n"
        "P02768\tAHKSEVAHR\t609\tHLA-A*02:01\t50.0\n",
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability(
        [("binder_0", _UBIQUITIN)], directory=tmp_path, binder_kind="protein"
    )
    assert axes["mhc_i"]["blocker"] is None
    assert "reference_rank_scale_ambiguous" not in axes["mhc_i"]


def test_a_width_no_cutoff_is_defined_for_is_named_as_such(tmp_path: Path) -> None:
    """Width 12 falls in the hole between the two published ranges.

    IEDB calibrates class I at 8 to 11 and class II at 13 to 25, which is what
    ``_SCORED_WIDTHS`` holds, so a 12-mer carries no binding call this module
    can make. A design and a reference both scored only at 12 left both width
    sets empty, so ``reference_width_mismatch`` passed trivially and
    ``_scan_covers`` could never be true: the reader was told
    ``scan_incomplete`` and sent to re-export a complete file, or
    ``sequence_absent`` about a chain that was right there.
    """
    from rfproteina.disease.liability import _SCORED_WIDTHS

    assert 12 not in _SCORED_WIDTHS
    assert {8, 9, 10, 11}.issubset(_SCORED_WIDTHS)
    assert set(range(13, 26)).issubset(_SCORED_WIDTHS)

    windows = [_UBIQUITIN[index : index + 12] for index in range(65)]
    (tmp_path / "mhc_i_reference.tsv").write_text(
        "protein\tpeptide\tlength\tallele\tpercentile_rank\nP02768\tDAHKSEVAHRFK\t609\tHLA-A*02:01\t0.3\n",
        encoding="utf-8",
    )

    # A complete 12-mer scan of the chain, holding one binder.
    rows = ["allele\tpeptide\tpercentile_rank"]
    rows += [f"HLA-A*02:01\t{window}\t{0.4 if index == 3 else 50.0}" for index, window in enumerate(windows)]
    (tmp_path / "mhc_i.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability(
        [("binder_0", _UBIQUITIN)], directory=tmp_path, binder_kind="protein"
    )
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "width_without_cutoff"
    assert record["measurement"] == 0.4

    # And the same complete scan with no binder in it, which used to be
    # refused as an incomplete scan.
    rows = ["allele\tpeptide\tpercentile_rank"]
    rows += [f"HLA-A*02:01\t{window}\t50.0" for window in windows]
    (tmp_path / "mhc_i.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability(
        [("binder_0", _UBIQUITIN)], directory=tmp_path, binder_kind="protein"
    )
    assert axes["mhc_i"]["state"] == "untested"
    assert axes["mhc_i"]["blocker"] == "width_without_cutoff"

    # A 9-mer scan of the same chain is a width a cutoff is defined for, so the
    # blocker is not reached and the reference is read.
    nine = [_UBIQUITIN[index : index + 9] for index in range(68)]
    rows = ["allele\tpeptide\tpercentile_rank"]
    rows += [f"HLA-A*02:01\t{window}\t{0.4 if index == 3 else 50.0}" for index, window in enumerate(nine)]
    (tmp_path / "mhc_i.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability(
        [("binder_0", _UBIQUITIN)], directory=tmp_path, binder_kind="protein"
    )
    assert axes["mhc_i"]["blocker"] != "width_without_cutoff"


def test_a_reference_protein_too_short_for_the_width_is_not_an_unknown_length(tmp_path: Path) -> None:
    """Two different faults shared one blocker.

    ``reference_length_unknown`` says the length could not be read. A tolerated
    protein whose length *was* read and is shorter than every width the table
    was scored at is a different thing, and the remedy is different too, so it
    has its own blocker and names the proteins.
    """
    fifteen = [_UBIQUITIN[index : index + 15] for index in range(62)]
    rows = ["allele\tpeptide\tpercentile_rank"]
    rows += [f"HLA-A*02:01\t{window}\t{0.4 if index == 3 else 50.0}" for index, window in enumerate(fifteen)]
    (tmp_path / "mhc_i.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    reference = tmp_path / "mhc_i_reference.tsv"
    # The 15-mer rows belong to a protein that may not set the bar and are
    # dropped; the one tolerated protein left states a length of 14, which
    # holds no 15-mer window at all.
    reference.write_text(
        "protein\tpeptide\tlength\tallele\tpercentile_rank\n"
        "Q9Y6K9\tVSSDAAAAAAAAAAA\t\tHLA-A*02:01\t0.3\n"
        "P0CG47\t\t14\tHLA-A*02:01\t\n",
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability(
        [("binder_0", _UBIQUITIN)], directory=tmp_path, binder_kind="protein"
    )
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "reference_too_short_for_width"
    assert record["reference_proteins"] == ["P0CG47"]
    assert record["stated_length"] == {"P0CG47": 14}
    assert record["reference_widths"] == [15]

    # When only some candidates drop, the ones that did are named beside the
    # bar the others set, so a protein that carried no binder is not confused
    # with one that was never in the running.
    reference.write_text(
        "protein\tpeptide\tlength\tallele\tpercentile_rank\n"
        f"P02768\t{fifteen[0]}\t609\tHLA-A*02:01\t0.3\n"
        "P0CG47\t\t14\tHLA-A*02:01\t\n",
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability(
        [("binder_0", _UBIQUITIN)], directory=tmp_path, binder_kind="protein"
    )
    record = axes["mhc_i"]
    assert record["blocker"] is None
    assert record["reference_proteins"] == ["P02768"]
    assert record["reference_proteins_too_short"] == ["P0CG47"]


def test_the_calis_record_names_the_mask_it_applied_and_the_restricting_allele(tmp_path: Path) -> None:
    """Calis et al. 2013 make the masked anchors per molecule.

    doi:10.1371/journal.pcbi.1003266, Results: the positions excluded are "P1,
    P2 and P9 for most HLA molecules, but e.g. P2, P5 and P9 for HLA-B*0801".
    One vector is applied to every presented 9-mer here, because the paper's
    position importances are published for the default masking only -- P1, P2
    and P9 read "NA (anchor)" in its Table 2 -- so no weight exists for scoring
    P1 under any other mask. The allele was read off every row and discarded;
    now the record says which mask was used and what restricted the ligands.
    """
    from rfproteina.disease.liability import _CALIS_MASK, _calis_score

    presented = "ENVKAKIQD"
    assert _calis_score(presented) < 0
    (tmp_path / "mhc_i.tsv").write_text(
        f"allele\tpeptide\tpercentile_rank\nHLA-A*02:01\t{presented}\t0.4\n",
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability([("binder_0", presented)], directory=tmp_path)
    record = axes["immunogenicity"]
    assert record["state"] == "clear"
    assert record["calis_mask"] == [1, 2, 9]
    assert list(_CALIS_MASK) == [1, 2, 9]
    # The mask is the paper's default, not this allele's, and says so.
    assert record["calis_mask_allele_specific"] is False
    assert record["restricting_alleles"] == ["A0201"]


def test_a_ligand_restricted_by_the_paper_s_exception_allele_cannot_clear(tmp_path: Path) -> None:
    """HLA-B*08:01 is the one molecule the paper names as masked elsewhere.

    Its anchors are P2, P5 and P9, so the default mask scores P5 (weight 0.30)
    and masks P1, which the paper's own position table publishes no weight for.
    A score computed with the wrong mask cannot support a clear, and no anchor
    set is invented to replace it. A positive ligand is still positive evidence,
    so a hit is not suppressed.
    """
    from rfproteina.disease.liability import _calis_score

    presented = "ENVKAKIQD"
    table = tmp_path / "mhc_i.tsv"
    table.write_text(
        f"allele\tpeptide\tpercentile_rank\nHLA-B*08:01\t{presented}\t0.4\n",
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability([("binder_0", presented)], directory=tmp_path)
    record = axes["immunogenicity"]
    assert record["state"] == "untested"
    assert record["blocker"] == "calis_mask_unpublished_for_allele"
    assert record["mask_unpublished_alleles"] == ["B0801"]
    assert record["restricting_alleles"] == ["B0801"]
    # The number the default mask gave is still reported, as information.
    assert record["measurement"] == round(_calis_score(presented), 5)

    # A positive ligand restricted by the same allele is still a hit.
    positive = "AAWWWWWWA"
    assert _calis_score(positive) > 0
    table.write_text(
        f"allele\tpeptide\tpercentile_rank\nHLA-B*08:01\t{positive}\t0.4\n",
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability([("binder_0", positive)], directory=tmp_path)
    record = axes["immunogenicity"]
    assert record["state"] == "hit"
    assert record["positive_windows"] == 1
    assert record["restricting_alleles"] == ["B0801"]


def test_the_exception_allele_is_matched_however_many_fields_it_names(tmp_path: Path) -> None:
    """A three-field IMGT/HLA name is the same molecule, and used to escape.

    ``_allele_key`` keeps every field, so ``HLA-B*08:01`` normalises to
    ``B0801`` and ``HLA-B*08:01:01`` to ``B080101``. An exact match read those
    as different molecules and published a ``clear`` on a score the module
    states it cannot compute -- the one place this gap errs toward an absence
    claim rather than toward withholding one. IEDB writes the two-field form and
    NetMHCpan the compact one, so a three-field name arrives when a user pastes
    it from an HLA typing report.

    The protein is the first two fields; a further field is a sub-type of the
    same molecule.
    """
    presented = "ENVKAKIQD"
    for spelling in (
        "HLA-B*08:01:01",
        "HLA-B*08:01:01:02",
        "B*08:01:01",
        "hla-b*08:01:01",
    ):
        folder = tmp_path / spelling.replace("*", "s").replace(":", "_")
        folder.mkdir()
        (folder / "mhc_i.tsv").write_text(
            f"allele\tpeptide\tpercentile_rank\n{spelling}\t{presented}\t0.4\n",
            encoding="utf-8",
        )
        axes, _by_example = measure_sequence_liability([("binder_0", presented)], directory=folder)
        record = axes["immunogenicity"]
        assert record["state"] == "untested", spelling
        assert record["blocker"] == "calis_mask_unpublished_for_allele", spelling
        assert record["mask_unpublished_alleles"] == ["B0801"], spelling


def test_a_different_molecule_at_the_same_locus_is_not_the_exception() -> None:
    """Matching broadly costs a clear and never a hit, but not that broadly.

    ``B*08:11`` and ``B*44:02`` are other molecules, and the paper names no
    anchor exception for them, so the default mask is the published treatment
    and a verdict stands.
    """
    from rfproteina.disease.liability import _calis_exceptions

    assert _calis_exceptions(["B0801"]) == {"B0801"}
    assert _calis_exceptions(["B080101"]) == {"B0801"}
    assert _calis_exceptions(["B0811"]) == set()
    assert _calis_exceptions(["B4402"]) == set()
    assert _calis_exceptions(["A0201"]) == set()
    assert _calis_exceptions(["B0801X"]) == set(), "a non-digit tail is not a sub-type"
    assert _calis_exceptions([]) == set()
    # One exception reported once, however many sub-types presented it.
    assert _calis_exceptions(["B0801", "B080101", "B08010102"]) == {"B0801"}


def test_a_three_field_panel_allele_counts_toward_the_panel(tmp_path: Path) -> None:
    """The DR panel skipped a sub-type spelling and then blamed the user for it.

    ``_allele_key`` keeps every field, so ``DRB1*03:01:02`` normalises to
    ``DRB1030102`` and an exact comparison against the panel's ``DRB10301``
    missed it. The row was skipped, the panel read incomplete, and
    ``missing_alleles`` named the very alleles that had just been scored -- the
    same loop the ``rank_absent`` branch exists to avoid, and unresolvable,
    because re-scoring produces the same spelling again.
    """
    from rfproteina.disease.liability import _DR_PANEL, _allele_key, _allele_molecule

    wanted = {_allele_key(name) for name in _DR_PANEL}
    for spelling in ("DRB1*03:01", "DRB1*03:01:02", "HLA-DRB1*03:01:02:01", "DRB1_0301"):
        key = _allele_key(spelling)
        assert _allele_molecule(key, wanted) == "DRB10301", spelling

    # A different molecule at the same locus is still not a panel member.
    for other in ("DRB1*03:02", "DRB1*13:01"):
        assert _allele_molecule(_allele_key(other), wanted) is None, other


def test_two_sub_types_of_one_molecule_keep_the_better_rank(tmp_path: Path) -> None:
    """They are one protein, so the panel sees one allele and its best rank."""
    from rfproteina.disease.liability import _DR_PANEL, _allele_key, _allele_molecule

    wanted = {_allele_key(name) for name in _DR_PANEL}
    first = _allele_molecule(_allele_key("DRB1*03:01:01"), wanted)
    second = _allele_molecule(_allele_key("DRB1*03:01:02"), wanted)
    assert first == second == "DRB10301"


def test_the_molecule_match_is_shared_with_the_calis_exception() -> None:
    """One comparison, because the two had the same gap in opposite directions.

    The Calis exception missed the molecule and published a clear on a score the
    module cannot compute; the panel missed it and refused a table it had.
    """
    from rfproteina.disease.liability import _CALIS_EXCEPTION_ALLELES, _allele_molecule

    assert _allele_molecule("B080101", _CALIS_EXCEPTION_ALLELES) == "B0801"
    assert _allele_molecule("B0811", _CALIS_EXCEPTION_ALLELES) is None
