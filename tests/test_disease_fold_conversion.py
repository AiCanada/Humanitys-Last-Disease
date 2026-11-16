"""Turning the downloaded HPA blood table into a fold table the screen reads.

None of the four downloaded screening tables drops into the screen as it
stands, and this is the one that can be converted. The other three cannot be
converted at all: ``mhc_i``, ``mhc_ii`` and ``ada`` must be *scanned from this
run's own designed chain* -- ``_table_belongs`` requires the table's peptides to
be windows of the design -- so IEDB's experimental epitope exports are a
different artifact entirely, not a differently-named version of the same thing.

Three things the reader needs that the download does not have, and each is a
silent failure on its own:

* ``logFC`` normalises to ``logfc``, which is in neither ``_FOLD_COLUMNS`` nor
  anything else the reader looks for. Dropped in unchanged, the file is
  recognised as a table and matched to the target, and then reports no fold --
  untested, which reads exactly like a target that genuinely does not move.
* no study column anywhere in the file, and the reader refuses a fold without
  provenance on purpose.
* 1.4 MB on disk, over the reader's 1 MB file bound, so it is refused as
  ``fold_too_large`` before a row is parsed. Note which bound that is: the
  256-row cap applies only *after* the rows are narrowed to one target, so it
  governs one protein's rows and never the file's.

And one thing it must NOT do: rescale. HPA's column is already log2 and
``_linear_fold`` raises 2 to it, so converting here as well would compare a
squared fold change against a linear cut -- the same class of error as reading
a micromolar column as nanomolar.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from rfproteina.disease.tables import plasma_fold_from_hpa
from rfproteina.disease.target_liability import FOLD_FILE_LIMIT, FOLD_ROW_LIMIT, _fold_record

HEADER = ("Gene", "ENSG ID", "Binder", "Disease", "Class", "Control", "logFC", "p-value adjusted")


def hpa_table(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    """A file shaped like the real blood_pea_disease_de.tsv, with (gene, disease, logFC)."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER)
        for gene, disease, log2 in rows:
            writer.writerow((gene, "ENSG0", "seq.1", disease, "Cancer", "All other", log2, "1E-3"))
    return path


def test_a_narrowed_conversion_measures_through_the_real_reader(tmp_path: Path) -> None:
    """The whole point: a downloaded table becoming a screen verdict."""
    table = hpa_table(
        tmp_path / "hpa.tsv",
        [("ERBB2", "Breast cancer", "0.280"), ("A1BG", "Breast cancer", "-0.042")],
    )
    out = tmp_path / "run"
    made = plasma_fold_from_hpa(table, out, disease="Breast cancer", genes=["ERBB2"], release="v26.0")
    assert made.usable, made.as_dict()

    record = _fold_record("ERBB2", out, 2.0)
    assert record["state"] == "clear"
    assert record["measurement"] == pytest.approx(0.280)
    assert "Human Protein Atlas" in str(record["fold_source"])


def test_the_log2_value_is_carried_across_untouched(tmp_path: Path) -> None:
    """Rescaling here would square the fold change the reader then compares.

    ``_linear_fold`` turns log2 1.0 into a twofold change. A converter that
    "helpfully" raised 2 to it first would hand over 2.0 and the reader would
    read a fourfold swing, clearing or failing on a number nobody measured.
    """
    table = hpa_table(tmp_path / "hpa.tsv", [("TP53", "Breast cancer", "1.0")])
    out = tmp_path / "run"
    plasma_fold_from_hpa(table, out, disease="Breast cancer", genes=["TP53"])
    written = (out / "plasma_fold.tsv").read_text(encoding="utf-8").splitlines()
    assert written[0].split("\t") == ["gene", "log2_fold_change", "source"]
    assert written[1].split("\t")[1] == "1.0", "the log2 value must be copied, not converted"
    # And the reader's own arithmetic turns that 1.0 into a twofold change.
    record = _fold_record("TP53", out, 1.5)
    assert record["state"] == "hit", "2**1.0 = 2.0 is over a 1.5 cut"


def test_every_row_carries_the_provenance_the_reader_demands(tmp_path: Path) -> None:
    """The download names no study, and a fold with none is refused."""
    table = hpa_table(tmp_path / "hpa.tsv", [("ERBB2", "Breast cancer", "0.28")])
    out = tmp_path / "run"
    made = plasma_fold_from_hpa(table, out, disease="Breast cancer", genes=["ERBB2"], release="v26.0")
    body = made.path.read_text(encoding="utf-8")
    assert "v26.0" in body and "disease=Breast cancer" in body
    assert _fold_record("ERBB2", out, 2.0)["fold_source"]


def test_a_whole_proteome_conversion_is_written_but_named_unusable(tmp_path: Path) -> None:
    """Written and refused is the trap: the file exists and measures nothing.

    The reader weighs the *file* before it parses anything, so a conversion of
    every gene produces a real file that `_fold_record` then rejects as
    `fold_too_large`. Saying so at conversion time is the difference between
    "narrow it to your targets" and a blocker on a run that looked equipped.
    """
    rows = [(f"GENE{index}", "Breast cancer", "0.1") for index in range(20_000)]
    table = hpa_table(tmp_path / "hpa.tsv", rows)
    out = tmp_path / "run"
    made = plasma_fold_from_hpa(table, out, disease="Breast cancer")
    assert made.written, "the file is written"
    assert made.path is not None and made.path.stat().st_size > FOLD_FILE_LIMIT
    assert not made.usable, "but the reader will not take it"
    assert made.blocker is not None and made.blocker.startswith("bytes_over_reader_limit")
    assert _fold_record("GENE1", out, 2.0)["blocker"] == "fold_too_large"


def test_more_genes_than_the_row_cap_is_usable_because_the_reader_narrows_first(
    tmp_path: Path,
) -> None:
    """The row cap bounds ONE target's rows, not the file's.

    `_fold_record` filters to the target before it counts, because the table
    carries a gene column. So a labelled table of 1,000 proteins -- four times
    the row cap, but small on disk -- is an ordinary campaign table and the
    reader takes it.

    Counting the whole file against that cap instead is the mistake this pins:
    it would report every multi-target table as unusable and send the user off
    narrowing a file that already worked, while the axis it was meant to
    protect went unmeasured.
    """
    rows = [(f"GENE{index}", "Breast cancer", "0.1") for index in range(FOLD_ROW_LIMIT * 4)]
    table = hpa_table(tmp_path / "hpa.tsv", rows)
    out = tmp_path / "run"
    made = plasma_fold_from_hpa(table, out, disease="Breast cancer")
    assert made.rows == FOLD_ROW_LIMIT * 4 > FOLD_ROW_LIMIT
    assert made.path is not None and made.path.stat().st_size <= FOLD_FILE_LIMIT
    assert made.usable, f"the reader narrows before counting: {made.as_dict()}"
    assert _fold_record("GENE7", out, 2.0)["state"] == "clear"
    assert _fold_record("GENE7", out, 2.0).get("blocker") is None


def test_one_gene_repeated_past_the_row_cap_is_named_unusable(tmp_path: Path) -> None:
    """The bound that does apply: rows surviving the narrowing, for one target.

    Small on disk and a single protein, so only the per-target cap can catch
    it -- and it is the cap the reader actually applies after filtering.
    """
    rows = [("ERBB2", "Breast cancer", "0.1") for _ in range(FOLD_ROW_LIMIT + 5)]
    table = hpa_table(tmp_path / "hpa.tsv", rows)
    out = tmp_path / "run"
    made = plasma_fold_from_hpa(table, out, disease="Breast cancer")
    assert made.path is not None and made.path.stat().st_size <= FOLD_FILE_LIMIT
    assert not made.usable
    assert made.blocker is not None and made.blocker.startswith("rows_for_one_gene_over_reader_limit")
    assert _fold_record("ERBB2", out, 2.0)["blocker"] == "fold_too_large"


def test_a_disease_that_is_not_there_writes_nothing_and_names_what_is(tmp_path: Path) -> None:
    """An empty table would read as "no fold reported", which is indistinguishable
    from a target that genuinely does not move."""
    table = hpa_table(
        tmp_path / "hpa.tsv",
        [("ERBB2", "Breast cancer", "0.28"), ("TP53", "Acute COVID", "0.4")],
    )
    out = tmp_path / "run"
    made = plasma_fold_from_hpa(table, out, disease="Asthma")
    assert not made.written and not made.usable
    assert made.blocker == "disease_not_in_table"
    assert set(made.available) == {"Breast cancer", "Acute COVID"}
    assert not (out / "plasma_fold.tsv").exists()


@pytest.mark.parametrize("named", ["breast cancer", "BREAST CANCER", "  Breast cancer  "])
def test_the_disease_name_is_matched_without_regard_to_case_or_padding(tmp_path: Path, named: str) -> None:
    table = hpa_table(tmp_path / "hpa.tsv", [("ERBB2", "Breast cancer", "0.28")])
    made = plasma_fold_from_hpa(table, tmp_path / "run", disease=named, genes=["erbb2"])
    assert made.usable, made.as_dict()
    assert made.rows == 1


def test_a_row_with_no_fold_value_is_dropped_rather_than_written_empty(tmp_path: Path) -> None:
    """An empty cell written out is one the reader cannot parse, and one such
    cell refuses the whole table.

    Defensive, and said as such: the v26.0 blood table has no blank and no
    non-numeric logFC cell in any of its 676,059 rows, which was checked rather
    than assumed. An earlier version of this docstring said HPA leaves the
    column blank where it reported nothing. It does not, and a guard is not
    improved by inventing the input it guards against -- the reason to keep it
    is that an empty cell costs the whole table, not that this release has one.
    """
    table = hpa_table(
        tmp_path / "hpa.tsv",
        [("ERBB2", "Breast cancer", ""), ("TP53", "Breast cancer", "0.5"), ("EGFR", "Breast cancer", "n/a")],
    )
    out = tmp_path / "run"
    made = plasma_fold_from_hpa(table, out, disease="Breast cancer")
    assert made.rows == 1 and made.genes == 1
    assert _fold_record("TP53", out, 2.0)["state"] == "clear"


def test_a_file_without_the_columns_names_the_one_it_lacks(tmp_path: Path) -> None:
    path = tmp_path / "wrong.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("Gene", "Disease"))
        writer.writerow(("ERBB2", "Breast cancer"))
    made = plasma_fold_from_hpa(path, tmp_path / "run", disease="Breast cancer")
    assert made.blocker == "column_absent:logFC"


def test_a_missing_file_is_a_named_blocker(tmp_path: Path) -> None:
    made = plasma_fold_from_hpa(tmp_path / "absent.tsv", tmp_path / "run", disease="Breast cancer")
    assert made.blocker == "table_not_found"


def test_naming_genes_that_are_all_blank_is_refused_rather_than_read_as_all(tmp_path: Path) -> None:
    """`genes=[]` asking for everything would be the opposite of what it says."""
    table = hpa_table(tmp_path / "hpa.tsv", [("ERBB2", "Breast cancer", "0.28")])
    made = plasma_fold_from_hpa(table, tmp_path / "run", disease="Breast cancer", genes=["", "  "])
    assert made.blocker == "genes_named_but_empty"
    assert not made.written
