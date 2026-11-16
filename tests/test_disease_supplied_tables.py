"""The three readers that turn a table supplied beside a run into a record.

``plasma_fold``, the paralog co-evolution note and the solubility report each
read a file the user puts in the output folder. Nothing here reaches the
network: the fold and table tests pass no reader at all, and the two that need
one answer by URL fragment and fail the test on an unplanned lookup, because an
unplanned lookup returning nothing is how an axis quietly becomes untested.

Each test asks the same question of a different reader: can a table that does
not cover this target, or holds nothing, or cannot be read without guessing,
reach a clear or a published number?
"""

from pathlib import Path

from rfproteina.disease.coevolution import read_paralog_coevolution
from rfproteina.disease.solubility import read_solubility
from rfproteina.disease.target_liability import (
    FOLD_FILE_LIMIT,
    FOLD_ROW_LIMIT,
    TargetLimits,
    measure_target_liability,
)
from tests.helpers.disease import url_reader

_IL5 = "P05113"
_IL4 = "P05112"
_CLUSTER = "UniRef50_P05113"


def _answers(**extra: object) -> dict[str, object]:
    answers: dict[str, object] = {
        "/uniref/search": {"results": [{"id": _CLUSTER}]},
        "/uniprotkb/search": {"results": [{"primaryAccession": _IL5, "genes": [{"geneName": {"value": "IL5"}}]}]},
        "proteinatlas.org": [
            {
                "Gene": "IL5",
                "Uniprot": [_IL5],
                "Blood concentration - Conc. blood IM [pg/L]": 1.0e9,
                "Blood concentration - Conc. blood MS [pg/L]": None,
            }
        ],
        "string-db.org": [{"preferredName_A": "IL5", "preferredName_B": "IL5RA"}],
    }
    answers.update(extra)
    return answers


def _fold(directory: Path, text: str, *, name: str = "plasma_fold.tsv", target: str = _IL5) -> dict[str, object]:
    """One fold axis, measured from a supplied table with no reader at all."""
    (directory / name).write_text(text, encoding="utf-8")
    axes = measure_target_liability(target, directory=directory, limits=TargetLimits(fold_max=2.0))
    return axes["plasma_fold"]


def test_a_hub_degree_of_zero_is_not_a_measured_clear(tmp_path: Path) -> None:
    """An empty partner list is also what an identifier STRING cannot map returns.

    A protein with no high-confidence physical partner and a stale accession
    give the same empty response, and a response with no row carries no
    ``preferredName_A`` to check the mapping against. Clearing on it would say
    "not a hub" on the strength of a lookup that may never have found the
    target.
    """
    answers = _answers(**{"string-db.org": []})
    axes = measure_target_liability(
        _IL5,
        directory=tmp_path,
        read_json=url_reader(answers),
        limits=TargetLimits(hub_max=5),
    )
    record = axes["plasma_hub"]
    assert record["state"] == "untested"
    assert record["blocker"] == "partners_not_listed"
    assert record["measurement"] == 0
    assert "threshold" not in record


def test_a_partner_list_naming_two_proteins_is_not_this_targets_degree(tmp_path: Path) -> None:
    """One identifier was asked about, so rows for two proteins are not one degree."""
    answers = _answers(
        **{
            "string-db.org": [
                {"preferredName_A": "IL5", "preferredName_B": "IL5RA"},
                {"preferredName_A": "IL4", "preferredName_B": "IL4R"},
            ]
        }
    )
    axes = measure_target_liability(
        _IL5,
        directory=tmp_path,
        read_json=url_reader(answers),
        limits=TargetLimits(hub_max=5),
    )
    record = axes["plasma_hub"]
    assert record["state"] == "untested"
    assert record["blocker"] == "partners_not_listed"


def test_a_labelled_fold_table_judges_only_this_targets_row(tmp_path: Path) -> None:
    """A row naming another protein is another protein's measurement.

    The axis takes the largest magnitude in the table, so one campaign-wide
    table published its biggest swing against every target in it.
    """
    record = _fold(
        tmp_path,
        "accession\tfold\tsource\nP05112\t9.0\tSun 2023 UKB-PPP\nP05113\t1.1\tSun 2023 UKB-PPP\n",
    )
    assert record["state"] == "clear"
    assert record["measurement"] == 1.1


def test_a_labelled_fold_table_that_never_names_this_target(tmp_path: Path) -> None:
    """Present, readable, sourced, and about someone else: not a measurement here."""
    record = _fold(tmp_path, "accession\tfold\tsource\nP05112\t9.0\tSun 2023 UKB-PPP\n")
    assert record["state"] == "untested"
    assert record["blocker"] == "fold_target_not_in_table"
    assert record["measurement"] is None


def test_a_gene_labelled_fold_table_matches_the_symbol_it_was_asked_about(tmp_path: Path) -> None:
    """A label the caller's own identifier matches still measures the axis."""
    record = _fold(tmp_path, "gene\tfold\tsource\nIL5\t3.4\tSun 2023 UKB-PPP\n", target="IL5")
    assert record["state"] == "hit"
    assert record["measurement"] == 3.4


def test_an_unlabelled_fold_table_is_still_this_runs_table(tmp_path: Path) -> None:
    """The documented two-column table has no label and keeps working."""
    record = _fold(tmp_path, "fold\tsource\n3.4\tSun 2023 UKB-PPP\n")
    assert record["state"] == "hit"
    assert record["measurement"] == 3.4


def test_a_repeated_fold_column_is_not_judged(tmp_path: Path) -> None:
    """csv keeps the last of two columns with one name, silently."""
    record = _fold(tmp_path, "fold\tfold\tsource\n1.1\t9.0\tSun 2023\n")
    assert record["state"] == "untested"
    assert record["blocker"] == "fold_unreadable"


def test_a_fold_row_with_a_surplus_field_is_not_judged(tmp_path: Path) -> None:
    """A stray separator reads every later cell under its neighbour's name."""
    record = _fold(tmp_path, "fold,source\n3.4,Sun 2023,plasma\n", name="plasma_fold.csv")
    assert record["state"] == "untested"
    assert record["blocker"] == "fold_unreadable"


def test_an_unterminated_quote_in_a_fold_table_is_not_a_clear(tmp_path: Path) -> None:
    """The swallowed row held the larger swing, and nothing said it was gone."""
    record = _fold(tmp_path, 'fold,source\n1.1,"Sun 2023\n9.0,Other study\n', name="plasma_fold.csv")
    assert record["state"] == "untested"
    assert record["blocker"] == "fold_unreadable"


def test_a_fold_header_holding_no_column_this_reads_says_so(tmp_path: Path) -> None:
    """A semicolon export is a delimiter problem, not a missing column."""
    record = _fold(tmp_path, "fold;source\n3.4;Sun 2023\n", name="plasma_fold.csv")
    assert record["state"] == "untested"
    assert record["blocker"] == "fold_unreadable"


def test_a_row_filling_two_fold_columns_keeps_the_larger_swing(tmp_path: Path) -> None:
    """A log2 of 3 is about eightfold, and it was dropped for the 1.1 beside it."""
    record = _fold(tmp_path, "fold\tlog2_fold_change\tsource\n1.1\t3\tSun 2023 UKB-PPP\n")
    assert record["state"] == "hit"
    assert record["measurement"] == 3.0


def test_a_sentinel_source_names_no_study(tmp_path: Path) -> None:
    """``NA`` is how an export writes an empty cell; read as text it was a study."""
    record = _fold(tmp_path, "fold\tsource\n3.4\tNA\n")
    assert record["state"] == "untested"
    assert record["blocker"] == "source_not_named"
    assert record["measurement"] == 3.4
    assert "fold_source" not in record


def test_an_empty_fold_table_is_not_a_clear(tmp_path: Path) -> None:
    """Present with no rows reports nothing, as it already did."""
    record = _fold(tmp_path, "fold\tsource\n")
    assert record["state"] == "untested"
    assert record["blocker"] == "fold_not_reported"


def test_two_disagreeing_score_columns_are_not_reported(tmp_path: Path) -> None:
    """The column the number came from is what says which scale it is on.

    0.4 is a poor CamSol score and a low AGGRESCAN one. A row holding both a
    solubility and an aggregation number carries one method, so nothing can
    say which of them it names.
    """
    (tmp_path / "solubility.tsv").write_text(
        "solubility\taggregation\tmethod\tsource\n0.1\t0.9\tCamSol\tSormanni 2015\n",
        encoding="utf-8",
    )
    record = read_solubility(tmp_path)
    assert record["solubility_blocker"] == "cell_unparsed"
    assert record["unparsed_columns"] == ["aggregation", "solubility"]
    assert "scores" not in record


def test_the_same_number_in_two_aliases_is_one_measurement(tmp_path: Path) -> None:
    """Two spellings of one column are not a disagreement."""
    (tmp_path / "solubility.tsv").write_text(
        "score\tvalue\tmethod\tsource\n0.42\t0.42\tCamSol 2.1\tSormanni 2015\n",
        encoding="utf-8",
    )
    record = read_solubility(tmp_path)
    assert record["rows"] == 1
    assert record["scores"] == [{"score": 0.42, "method": "CamSol 2.1", "source": "Sormanni 2015"}]


def test_an_empty_solubility_table_reports_no_score(tmp_path: Path) -> None:
    """A header with no row is not a number, and it is not a state either."""
    (tmp_path / "solubility.tsv").write_text("score\tmethod\tsource\n", encoding="utf-8")
    record = read_solubility(tmp_path)
    assert record == {"solubility_blocker": "solubility_not_reported"}
    assert "state" not in record


def test_a_solubility_score_without_its_method_is_not_stored(tmp_path: Path) -> None:
    """A bare number carries no scale, so it is dropped and the drop is said."""
    (tmp_path / "solubility.csv").write_text("score,source\n0.42,Sormanni 2015\n", encoding="utf-8")
    record = read_solubility(tmp_path)
    assert record["solubility_blocker"] == "solubility_not_reported"
    assert record["incomplete_rows"] == 1


def test_an_empty_coevolution_table_is_not_a_note(tmp_path: Path) -> None:
    """Present with no rows, and carrying no key the paralog axis could read as a state."""
    (tmp_path / "paralog_coevolution.tsv").write_text("same_shape\tcoevolution\ttoxicity\n", encoding="utf-8")
    record = read_paralog_coevolution(tmp_path)
    assert record == {"coevolution_blocker": "coevolution_not_reported"}


def test_a_non_toxic_coevolution_note_does_not_clear_the_paralog_axis(tmp_path: Path) -> None:
    """The note sits beside the UniRef50 result and never becomes it.

    "Probably non-toxic in people" is the most favourable of the four transfer
    notes, and the paralog axis has no measured clear: a singleton cluster is
    not evidence of no family, and only a curated judgement clears it.
    """
    (tmp_path / "paralog_coevolution.tsv").write_text(
        "species\tsame_shape\tcoevolution\ttoxicity\nkillifish\tyes\thigh\tnon-toxic\n",
        encoding="utf-8",
    )
    axes = measure_target_liability(_IL5, directory=tmp_path, read_json=url_reader(_answers()))
    record = axes["paralog"]
    assert record["coevolution_signal"] == "probably_non_toxic"
    assert record["state"] == "untested"
    assert record["blocker"] == "cluster_singleton_not_evidence"
    assert record["coevolution_species"] == ["killifish"]


def test_an_unsourced_row_louder_than_the_cut_denies_the_clear(tmp_path: Path) -> None:
    """A clear speaks for the whole table, so an unsourced 8-fold answers it.

    The axis refuses to be measured from a row that names no study, and it was
    right to drop the log2 of 3. Publishing ``clear`` on the 1.1 beside it was
    the mistake: the table's own larger number says this target swings past the
    cut, and a clear covers rows it never read. The number is reported and the
    verdict withheld.
    """
    record = _fold(tmp_path, "fold\tsource\tlog2_fold_change\n1.1\tSun 2023\t\n\t\t3\n")
    assert record["state"] == "untested"
    assert record["blocker"] == "source_not_named"
    assert record["measurement"] == 3.0
    assert "fold_source" not in record


def test_a_hit_is_not_withheld_for_an_unsourced_row(tmp_path: Path) -> None:
    """A hit rests on its own sourced row; a louder unsourced row only agrees.

    The guard above withholds a clear, never a hit. A hit says this target
    moves, the sourced row says it, and an unsourced row saying it moves even
    further is not a reason to stop saying it.
    """
    record = _fold(tmp_path, "fold\tsource\tlog2_fold_change\n5.0\tSun 2023\t\n\t\t3\n")
    assert record["state"] == "hit"
    assert record["measurement"] == 5.0
    assert record["fold_source"] == "Sun 2023"


def test_an_unsourced_row_inside_the_cut_leaves_the_clear_alone(tmp_path: Path) -> None:
    """An unsourced row that also clears contradicts nothing, so it is ignored.

    A log2 of 0.5 is about 1.4-fold, inside the twofold cut. The clear already
    covers it, so withholding the verdict here would refuse a table that agrees
    with itself.
    """
    record = _fold(tmp_path, "fold\tsource\tlog2_fold_change\n1.1\tSun 2023\t\n\t\t0.5\n")
    assert record["state"] == "clear"
    assert record["measurement"] == 1.1
    assert record["fold_source"] == "Sun 2023"
    assert record["threshold"] == 2.0


def test_a_fold_table_past_the_byte_limit_is_refused_not_read(tmp_path: Path) -> None:
    """The fold reader read the whole file with no bound; its siblings do not.

    A fold table is a handful of rows per target. A file past a mebibyte is a
    wrong file -- a proteome, a metrics dump, an export that never terminated --
    and reading it to find that out is the one supplied table that could take
    the screen down rather than refuse it.

    The file holds exactly ``FOLD_ROW_LIMIT`` rows, so the row bound below
    cannot be what refuses it and the byte bound is the only thing under test.
    Written as 6000 short rows the same assertion passed with the byte check
    deleted, because the row bound caught that file and named the same blocker.
    """
    padding = "x" * 4200
    rows = "".join(f"1.1\tSun 2023 {padding}\n" for _ in range(FOLD_ROW_LIMIT))
    table = f"fold\tsource\n{rows}"
    assert len(table.encode("utf-8")) > FOLD_FILE_LIMIT
    record = _fold(tmp_path, table)
    assert record["state"] == "untested"
    assert record["blocker"] == "fold_too_large"
    assert record["measurement"] is None


def test_a_fold_table_past_the_row_limit_is_refused(tmp_path: Path) -> None:
    """Small enough to read, and still not a fold table."""
    rows = "".join(f"{1.0 + index / 1000:.3f}\tSun 2023\n" for index in range(FOLD_ROW_LIMIT + 1))
    record = _fold(tmp_path, f"fold\tsource\n{rows}")
    assert record["state"] == "untested"
    assert record["blocker"] == "fold_too_large"


def test_a_fold_table_at_the_row_limit_is_still_read(tmp_path: Path) -> None:
    """The bound is a bound, not a margin: the last row inside it is measured."""
    rows = "".join("1.1\tSun 2023\n" for _ in range(FOLD_ROW_LIMIT))
    record = _fold(tmp_path, f"fold\tsource\n{rows}")
    assert record["state"] == "clear"
    assert record["measurement"] == 1.1


def test_the_row_bound_counts_this_targets_rows_not_the_campaigns(tmp_path: Path) -> None:
    """A campaign of more than ``FOLD_ROW_LIMIT`` proteins is an ordinary campaign.

    ``_FOLD_TARGET_COLUMNS`` exists so that one table can label the rows of a
    whole campaign, and the bound is there because a fold table is a handful of
    rows *per target*. Counted over the whole file before the rows were narrowed,
    it called a 301-protein campaign table a wrong file: this target's own
    sourced 3.4-fold row -- one row, a hit -- came back ``fold_too_large``, and
    so did every other target's, for no reason but that they were listed
    together.
    """
    others = "".join(f"P{index:05d}\t1.05\tSun 2023 UKB-PPP\n" for index in range(FOLD_ROW_LIMIT + 44))
    record = _fold(tmp_path, f"accession\tfold\tsource\n{others}{_IL5}\t3.4\tSun 2023 UKB-PPP\n")
    assert record["state"] == "hit"
    assert record["measurement"] == 3.4
    assert record["fold_source"] == "Sun 2023 UKB-PPP"

    # And the bound still bites on what is left after narrowing: a labelled file
    # holding more than a handful of rows for this one target is not a fold
    # table either.
    mine = tmp_path / "mine"
    mine.mkdir()
    rows = "".join(f"{_IL5}\t1.1\tSun 2023\n" for _ in range(FOLD_ROW_LIMIT + 1))
    record = _fold(mine, f"accession\tfold\tsource\n{rows}")
    assert record["state"] == "untested"
    assert record["blocker"] == "fold_too_large"


def test_a_long_study_citation_is_cut_rather_than_stored_whole(tmp_path: Path) -> None:
    """A study is a citation, not a document.

    ``fold_source`` is stored in screen.json and returned over the MCP view, so
    an export that pasted a whole abstract into the source column would carry it
    into both. Cut to the same length the solubility reader cuts its text to.
    """
    long_source = "Sun " + "et al. " * 60
    record = _fold(tmp_path, f"fold\tsource\n9.0\t{long_source}\n")
    assert record["state"] == "hit"
    assert record["measurement"] == 9.0
    source = record["fold_source"]
    assert isinstance(source, str)
    assert len(source) == 120
    assert source.endswith("...")
    assert source.startswith("Sun et al.")


def test_an_ordinary_study_citation_is_kept_exactly(tmp_path: Path) -> None:
    """The guard: the cut must not touch a citation of normal length."""
    record = _fold(tmp_path, "fold\tsource\n9.0\tSun et al., Nature 622:329-338, 2023\n")
    assert record["fold_source"] == "Sun et al., Nature 622:329-338, 2023"
