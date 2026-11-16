"""The per-design metrics table, which is the one input that moves the verdict.

Every other table this screen reads is supplied by hand and reports rather than
judges. ``inference_metrics/per_example_metrics.csv`` is where ``ppi.pass``,
``pre_oracle_pass``, the chain of interest and the per-design numbers come from,
and it was the least guarded reader in the module: a repeated column, a
misaligned row or an unterminated quote all returned an empty list, which reads
downstream exactly like a run nobody scored.

Three questions get named answers here. Why the table yielded no rows
(``metrics_blocker``). Which row measured a given design, when it was not
exactly one (``row_blocker``). And whether the table agrees with itself about
the chain of interest (``chain_blocker``).
"""

from pathlib import Path

import pytest

from rfproteina.disease.screen import designed_chain, screen_generation
from tests.helpers.disease import closing_page

_CIF = "data_binder\n#\n"


def _run(tmp_path: Path, name: str, metrics: str | None, *, designs: tuple[str, ...] = ("binder_0",)) -> dict:
    """One finished run, its designs, and the metrics table as given."""
    output = tmp_path / name
    for design in designs:
        folder = output / "inference_samples" / design
        folder.mkdir(parents=True)
        (folder / f"{design}_generated.cif").write_text(_CIF, encoding="utf-8")
    if metrics is not None:
        table = output / "inference_metrics" / "per_example_metrics.csv"
        table.parent.mkdir(parents=True, exist_ok=True)
        table.write_text(metrics, encoding="utf-8")
    return screen_generation(output, binder_kind="protein", target_id="P05113")


def _sample(record: dict, design: str) -> dict:
    return next(item for item in record["samples"] if item["example_id"].endswith(design))


def test_a_clean_table_is_read_and_carries_no_blocker(tmp_path: Path) -> None:
    record = _run(tmp_path, "good", "example_id,chain_of_interest,ppi.pass\nbinder_0,A_1,1\n")
    assert record["ppi_pass"] is True
    assert "metrics_blocker" not in record
    assert "chain_blocker" not in record
    assert "row_blocker" not in record["samples"][0]


def test_a_run_with_no_table_says_which_silence_it_is(tmp_path: Path) -> None:
    record = _run(tmp_path, "none", None)
    assert record["metrics_blocker"] == "metrics_not_configured"
    assert record["metrics_file"] is None
    # One run-level reason, not the same sentence once per design.
    assert "row_blocker" not in record["samples"][0]


def test_a_header_with_no_rows_under_it_is_not_a_missing_table(tmp_path: Path) -> None:
    record = _run(tmp_path, "empty", "example_id,ppi.pass\n")
    assert record["metrics_blocker"] == "metrics_not_reported"
    assert record["metrics_file"] == "inference_metrics/per_example_metrics.csv"


def test_a_repeated_column_is_refused_and_its_hidden_false_does_not_clear(tmp_path: Path) -> None:
    """``csv`` keeps the last copy of a repeated column, so the first was invisible."""
    record = _run(tmp_path, "repeated", "example_id,ppi.pass,ppi.pass\nbinder_0,0,1\n")
    assert record["metrics_blocker"] == "metrics_columns_repeated"
    assert record["ppi_pass"] == "absent", "a refused table measures nothing"


def test_a_column_this_screen_never_reads_may_repeat(tmp_path: Path) -> None:
    text = "example_id,note,note,ppi.pass\nbinder_0,a,b,1\n"
    record = _run(tmp_path, "harmless", text)
    assert "metrics_blocker" not in record
    assert record["ppi_pass"] is True


@pytest.mark.parametrize(
    ("label", "text", "blocker"),
    [
        ("a stray separator", "example_id,ppi.pass\nbinder_0,1,stray\n", "metrics_row_too_long"),
        ("a row cut short", "example_id,chain_of_interest,ppi.pass\nbinder_0,A_1\n", "metrics_row_too_short"),
        (
            "an unterminated quote",
            'example_id,chain_of_interest,ppi.pass\nbinder_0,"A_1,0\nbinder_1,A_1,1\n',
            "metrics_row_too_short",
        ),
    ],
)
def test_a_misaligned_table_is_refused_whole(tmp_path: Path, label: str, text: str, blocker: str) -> None:
    """Each of these reads a flag under another column's name."""
    record = _run(tmp_path, label.replace(" ", "_"), text)
    assert record["metrics_blocker"] == blocker
    assert record["ppi_pass"] == "absent"
    assert record["pre_oracle_pass"] == "absent"


def test_the_closed_page_explains_a_refused_table(tmp_path: Path) -> None:
    page = closing_page(_run(tmp_path, "repeated", "example_id,ppi.pass,ppi.pass\nbinder_0,0,1\n"))
    assert "The metrics table names a column this screen reads more than once." in page
    assert "Keep one copy of each column and screen again." in page


def test_the_closed_page_explains_a_missing_table(tmp_path: Path) -> None:
    page = closing_page(_run(tmp_path, "none", None))
    assert "No per-design metrics table was found in the output folder." in page


def test_two_rows_that_agree_are_still_that_designs_row(tmp_path: Path) -> None:
    """Concatenated job tables repeat rows. An exact duplicate loses no verdict."""
    text = "example_id,chain_of_interest,ca_rmsd_to_input.rmsd\nbinder_0,A_1,1.5\nbinder_0,A_1,1.5\n"
    record = _run(tmp_path, "duplicate", text)
    assert "row_blocker" not in record["samples"][0]
    assert record["samples"][0]["ca_rmsd"] == 1.5


def test_two_rows_that_disagree_are_refused_rather_than_resolved_to_the_first(tmp_path: Path) -> None:
    text = "example_id,chain_of_interest,ca_rmsd_to_input.rmsd\nbinder_0,A_1,1.5\nbinder_0,A_1,9.9\n"
    record = _run(tmp_path, "disagree", text)
    assert record["samples"][0]["row_blocker"] == "metrics_row_ambiguous"
    assert record["samples"][0]["ca_rmsd"] is None, "neither row measured the file on disk"


def test_a_design_no_row_names_says_so(tmp_path: Path) -> None:
    record = _run(tmp_path, "unnamed", "example_id,chain_of_interest\nbinder_7,A_1\n")
    assert record["samples"][0]["row_blocker"] == "metrics_row_absent"


def test_the_closed_page_counts_the_designs_a_row_blocker_applies_to(tmp_path: Path) -> None:
    text = "example_id,chain_of_interest,ca_rmsd_to_input.rmsd\nbinder_0,A_1,1.5\nbinder_0,A_1,9.9\n"
    page = closing_page(_run(tmp_path, "disagree", text))
    assert "1 design(s): Two metrics rows name this design and disagree about it." in page


def test_a_table_naming_two_chains_leaves_an_unnamed_design_unmeasured(tmp_path: Path) -> None:
    """The fallback across the table stands in only while the table is of one mind."""
    text = "example_id,chain_of_interest\nbinder_0,A_1\nbinder_1,B_1\n"
    record = _run(tmp_path, "chains", text, designs=("binder_0", "binder_1", "binder_2"))
    third = _sample(record, "binder_2")
    assert third["interface_blocker"] == "chain_disagreement"
    assert third["sequence_blocker"] == "chain_disagreement"
    assert third["contact_ms"] is None
    # A design whose own row names its chain is measured as before.
    assert _sample(record, "binder_0")["sequence_blocker"] == "sequence_absent"


def test_the_disagreement_is_recorded_on_the_run_and_shown(tmp_path: Path) -> None:
    """The top-level blocks repeat the first design, which may be unaffected."""
    text = "example_id,chain_of_interest\nbinder_0,A_1\nbinder_1,B_1\n"
    record = _run(tmp_path, "chains", text, designs=("binder_0", "binder_1"))
    assert record["chain_blocker"] == "chain_disagreement"
    assert "The metrics table names more than one chain of interest." in closing_page(record)


def test_nothing_refolding_a_design_is_handed_a_contested_chain(tmp_path: Path) -> None:
    text = "example_id,chain_of_interest\nbinder_0,A_1\nbinder_1,B_1\n"
    _run(tmp_path, "chains", text, designs=("binder_0", "binder_1"))
    assert designed_chain(tmp_path / "chains") == ""
    _run(tmp_path, "settled", "example_id,chain_of_interest\nbinder_0,A_1\n")
    assert designed_chain(tmp_path / "settled") == "A_1"
