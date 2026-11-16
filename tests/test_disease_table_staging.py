"""Joining the downloads to the screen.

Downloads land in one directory under the names their servers give them; the
screen reads tables out of each run's own output directory under its own stems.
Nothing joined the two, so a machine holding every download screened exactly as
a bare one did -- and the presence check said so in the most misleading way
available, by reporting a fully stocked directory as entirely empty.

Three failures here are silent by construction, and each has a test below:

* a presence check looking for a name nothing writes,
* a table the reader accepts and no target matches, because the file is keyed
  by gene symbol and a real run asks by accession,
* a download that is a different artifact from the one its axis reads, sitting
  in the registry beside a stem that invites it to be renamed.
"""

from __future__ import annotations

import csv
import gzip
import io
import zipfile
from pathlib import Path

import pytest

from rfproteina.disease.tables import (
    SOURCES,
    coverage,
    downloaded_name,
    downloads,
    fetch,
    gene_accessions,
    plasma_fold_from_hpa,
    sample_line,
    stage_for_run,
    url_leaf,
)
from rfproteina.disease.target_liability import TargetLimits, measure_target_liability

HPA_HEADER = ("Gene", "ENSG ID", "Binder", "Disease", "Class", "Control", "logFC", "p-value adjusted")


def hpa(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(HPA_HEADER)
        for gene, disease, log2 in rows:
            writer.writerow((gene, "ENSG0", "seq.1", disease, "Cancer", "All other", log2, "1E-3"))
    return path


def proteome(path: Path, pairs: list[tuple[str, str]]) -> Path:
    """A reference-proteome FASTA, gzipped the way UniProt ships it."""
    body = "".join(
        f">sp|{accession}|{gene}_HUMAN Some protein OS=Homo sapiens OX=9606 GN={gene} PE=1 SV=1\nMKV\n"
        for gene, accession in pairs
    )
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(body)
    return path


def stock(directory: Path, *, disease: str = "Breast cancer", with_proteome: bool = True) -> Path:
    """A download directory holding what the registry says it holds."""
    directory.mkdir(parents=True, exist_ok=True)
    hpa(
        directory / "blood_pea_disease_de.tsv",
        [("ERBB2", disease, "0.28"), ("TP53", disease, "1.5"), ("MYC", "Asthma", "0.9")],
    )
    (directory / "mhc_ligand_full.tsv").write_text("x\n", encoding="utf-8")
    (directory / "tcell_full_v3.tsv").write_text("x\n", encoding="utf-8")
    (directory / "aggregated_ada_values.csv").write_text("x\n", encoding="utf-8")
    if with_proteome:
        proteome(
            directory / "UP000005640_9606.fasta.gz",
            [("ERBB2", "P04626"), ("TP53", "P04637"), ("MYC", "P01106")],
        )
    return directory


# --------------------------------------------------------- the presence check


def test_a_stocked_directory_is_not_reported_as_empty(tmp_path: Path) -> None:
    """The headline bug: every download present, every one reported missing.

    ``fetch`` writes the name the server gives; the old check looked for
    ``<stem>.tsv``, which nothing writes. So the report was not merely wrong,
    it was wrong in the direction that costs the most -- it told an operator
    holding 11 GB of downloads to go and fetch them again, and the obvious way
    to satisfy it, renaming each file to its stem, is the one action that
    breaks an axis rather than feeding it.
    """
    directory = stock(tmp_path / "downloads")
    found = downloads(directory)
    assert found.complete, found.as_dict()
    assert {source.name for source, _ in found.present} == {source.name for source in SOURCES}
    for source, path in found.present:
        assert path.name == source.file


def test_an_empty_directory_reports_every_source_absent(tmp_path: Path) -> None:
    found = downloads(tmp_path / "nothing")
    assert not found.complete
    assert len(found.absent) == len(SOURCES)
    assert not found.present


@pytest.mark.parametrize("source", SOURCES, ids=lambda source: source.file)
def test_the_name_fetch_writes_is_the_name_the_check_looks_for(
    tmp_path: Path, source, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the drift that caused the bug, for every source at once.

    The write and the presence check derived the file name separately, and
    separately is how they came to disagree. This serves each source's URL
    through a stub and asserts the file that lands is the one ``downloads``
    goes looking for -- so a source added with a wrong ``file`` fails here
    rather than in a report that reads "missing" forever.
    """
    leaf = url_leaf(source)
    payload = b"gene\tvalue\n"
    if leaf.endswith(".zip"):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as bundle:
            bundle.writestr(source.file, payload.decode())
        payload = buffer.getvalue()

    class Response(io.BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def info(self) -> dict[str, str]:
            return {}

    monkeypatch.setattr("rfproteina.disease.tables.declared_size", lambda *_args, **_kwargs: len(payload))
    destination = tmp_path / "downloads"
    fetch(source, destination, opener=lambda *_args, **_kwargs: Response(payload))

    assert (destination / downloaded_name(source)).is_file(), f"{source.name} lands somewhere other than Source.file"
    assert downloaded_name(source) == source.file


# ------------------------------------------------------------------- staging


def test_a_staged_run_answers_a_lookup_by_accession(tmp_path: Path) -> None:
    """The failure that would have made this whole axis a no-op.

    HPA keys its rows by gene symbol. The screen resolves a target to a UniProt
    accession whenever a condition spec names one, which on a real run it does,
    and ``_names_target`` compares the two literally -- by design, so that no
    row belonging to another protein is ever read. A symbol-keyed table against
    an accession target therefore reports ``fold_target_not_in_table``: the
    protein is in the table, under its other name, and the axis says it is not.
    """
    directory = stock(tmp_path / "downloads")
    run = tmp_path / "run"
    made = stage_for_run(run, disease="Breast cancer", source_directory=directory)
    assert made.equipped, made.as_dict()
    assert made.fold is not None and made.fold.answers_accessions

    limits = TargetLimits(fold_max=2.0)
    by_symbol = measure_target_liability("ERBB2", directory=run, limits=limits)["plasma_fold"]
    by_accession = measure_target_liability("P04626", directory=run, limits=limits)["plasma_fold"]
    assert by_accession["measurement"] == by_symbol["measurement"] == pytest.approx(0.28)
    assert by_accession["state"] == "clear"
    assert by_accession.get("blocker") is None


def test_a_protein_absent_from_the_table_still_says_so(tmp_path: Path) -> None:
    """The accession column must not turn a miss into a match."""
    run = tmp_path / "run"
    stage_for_run(run, disease="Breast cancer", source_directory=stock(tmp_path / "downloads"))
    record = measure_target_liability("P11169", directory=run, limits=TargetLimits(fold_max=2.0))["plasma_fold"]
    assert record["state"] == "untested"
    assert record["blocker"] == "fold_target_not_in_table"


def test_narrowing_accepts_the_accession_a_run_actually_holds(tmp_path: Path) -> None:
    """``targets`` takes either identifier, and that is not a convenience.

    Downstream of generation the caller holds an accession, because that is
    what the screen resolved the target to. HPA labels its rows by symbol. A
    narrowing knob that took only symbols would be unusable from the one place
    that needs it, and narrowing is not optional: unnarrowed, the conversion
    writes every gene for the disease and is refused over the 1 MB bound.
    """
    directory = stock(tmp_path / "downloads")
    run = tmp_path / "run"
    made = stage_for_run(run, disease="Breast cancer", source_directory=directory, targets=["P04626"])
    assert made.equipped, made.as_dict()
    assert made.fold is not None and made.fold.genes == 1
    rows = (run / "plasma_fold.tsv").read_text(encoding="utf-8").splitlines()[1:]
    assert all(row.split("\t")[0] == "ERBB2" for row in rows), rows
    assert measure_target_liability("P04626", directory=run, limits=TargetLimits(fold_max=2.0))["plasma_fold"][
        "measurement"
    ] == pytest.approx(0.28)


def test_an_unlisted_identifier_misses_by_name_and_blames_the_right_filter(
    tmp_path: Path,
) -> None:
    """Not guessed at, and the blocker names which filter emptied the result.

    This used to report ``disease_not_in_table`` here, which is the one thing
    that was NOT wrong: "Breast cancer" is in the fixture, and the gene filter
    is what matched nothing. It sent an operator to check a disease name that
    was fine. A run whose target resolves to no accession narrows by something
    like ``gene_9GVN_1`` and lands here, so this is the ordinary case and not
    an exotic one.
    """
    run = tmp_path / "run"
    made = stage_for_run(
        run,
        disease="Breast cancer",
        source_directory=stock(tmp_path / "downloads"),
        targets=["Q00000"],
    )
    assert not made.equipped
    assert dict(made.shortfalls)["plasma_fold"] == "conversion_unusable:targets_not_in_table"
    # And it names genes this disease does report, which is the way forward.
    assert made.fold is not None
    assert set(made.fold.available_genes) == {"ERBB2", "TP53"}
    assert made.fold.available == (), "the disease matched, so no disease list belongs here"


def test_the_two_empty_results_are_told_apart(tmp_path: Path) -> None:
    """A disease that is absent and a target that is absent are different fixes.

    Reported as one blocker they were indistinguishable, and the wrong half of
    the pair is the one an operator would act on.
    """
    directory = stock(tmp_path / "downloads")

    absent_disease = stage_for_run(tmp_path / "a", disease="Psoriasis", source_directory=directory, targets=["ERBB2"])
    assert dict(absent_disease.shortfalls)["plasma_fold"] == ("conversion_unusable:disease_not_in_table")
    assert absent_disease.fold is not None
    assert "Breast cancer" in absent_disease.fold.available
    assert absent_disease.fold.available_genes == ()

    absent_target = stage_for_run(
        tmp_path / "b", disease="Breast cancer", source_directory=directory, targets=["NOPE1"]
    )
    assert dict(absent_target.shortfalls)["plasma_fold"] == ("conversion_unusable:targets_not_in_table")
    assert absent_target.fold is not None
    assert absent_target.fold.available_genes
    assert absent_target.fold.available == ()


def test_naming_no_target_is_not_naming_every_target(tmp_path: Path) -> None:
    """``targets=[" "]`` asking for the whole proteome would be the opposite of
    what it says, and the whole proteome is refused by the reader anyway."""
    run = tmp_path / "run"
    made = stage_for_run(
        run,
        disease="Breast cancer",
        source_directory=stock(tmp_path / "downloads"),
        targets=["", "  "],
    )
    assert not made.equipped
    assert dict(made.shortfalls)["plasma_fold"] == "targets_named_but_empty"
    assert not (run / "plasma_fold.tsv").exists()


def test_the_three_downloads_that_feed_nothing_are_named_not_placed(tmp_path: Path) -> None:
    """Accounted for, never staged.

    An operator who fetched 11 GB is owed the reason in a report rather than in
    an axis that quietly reads untested.
    """
    run = tmp_path / "run"
    made = stage_for_run(run, disease="Breast cancer", source_directory=stock(tmp_path / "downloads"))
    reasons = dict(made.unstaged)
    for stem, leaf in (
        ("mhc_i", "mhc_ligand_full.tsv"),
        ("immunogenicity", "tcell_full_v3.tsv"),
        ("ada", "aggregated_ada_values.csv"),
    ):
        assert reasons[stem] == f"no_conversion_from:{leaf}"
        assert not (run / f"{stem}.tsv").exists(), f"{stem} was placed; it must not be"
        assert not (run / f"{stem}.csv").exists(), f"{stem} was placed; it must not be"
    # Named, but not a shortfall: no download was ever going to close these.
    assert not any(stem in dict(made.shortfalls) for stem in ("mhc_i", "immunogenicity", "ada"))


def test_a_missing_proteome_is_a_shortfall_and_not_a_failure(tmp_path: Path) -> None:
    """Without the mapping the table is still written and still read by symbol.

    So it is reported as falling short rather than as broken -- and the run is
    NOT called equipped, because the lookup a real run makes will miss.
    """
    directory = stock(tmp_path / "downloads", with_proteome=False)
    run = tmp_path / "run"
    made = stage_for_run(run, disease="Breast cancer", source_directory=directory)

    assert not made.equipped
    assert ("accession lookup", "proteome_absent:UP000005640_9606.fasta.gz") in made.shortfalls
    assert made.fold is not None and made.fold.usable and not made.fold.answers_accessions
    assert ("plasma_fold", run / "plasma_fold.tsv") in made.staged

    limits = TargetLimits(fold_max=2.0)
    assert (
        measure_target_liability("ERBB2", directory=run, limits=limits)["plasma_fold"]["state"] == "clear"
    ), "the symbol lookup still works"
    assert (
        measure_target_liability("P04626", directory=run, limits=limits)["plasma_fold"]["blocker"]
        == "fold_target_not_in_table"
    ), "and the accession lookup is the thing that is missing"


def test_a_disease_not_in_the_table_stages_nothing_for_that_axis(tmp_path: Path) -> None:
    """``Asthma`` is in the fixture on purpose; ``Psoriasis`` is not.

    An empty fold table would read as "no fold reported", which is the one
    outcome indistinguishable from a target that genuinely does not move, so
    nothing is written and the axis falls short by name.
    """
    run = tmp_path / "run"
    made = stage_for_run(run, disease="Psoriasis", source_directory=stock(tmp_path / "downloads"))
    assert not made.equipped
    assert dict(made.shortfalls)["plasma_fold"] == "conversion_unusable:disease_not_in_table"
    assert not (run / "plasma_fold.tsv").exists()


def test_coverage_counts_only_the_stems_a_download_can_produce(tmp_path: Path) -> None:
    """An axis no download feeds is not a gap a download could close.

    Listing it as missing sent an operator to fetch a file that would not have
    helped -- and for ``ada``, one that displaces a real verdict if dropped in.
    """
    run = tmp_path / "run"
    stage_for_run(run, disease="Breast cancer", source_directory=stock(tmp_path / "downloads"))
    found = coverage(run)
    assert found.complete, found.as_dict()
    assert [stem for stem, _ in found.present] == ["plasma_fold"]
    # The set of axes, not the list of queries: one axis may be served by more
    # than one endpoint, and `paralog` now is -- UniRef50 membership, then
    # Ensembl Compara for the singletons that membership cannot speak to.
    assert {query.axis for query in found.online} == {"paralog", "plasma_abundance", "plasma_hub"}
    assert sum(query.axis == "paralog" for query in found.online) == 2


def test_coverage_does_not_call_a_run_offline(tmp_path: Path) -> None:
    """Every table present is still three REST lookups away from offline."""
    run = tmp_path / "run"
    stage_for_run(run, disease="Breast cancer", source_directory=stock(tmp_path / "downloads"))
    assert coverage(run).online, "a complete run that reports no online axes is the lie"


# ------------------------------------------------------------- the registry


def test_every_source_records_the_name_it_lands_under() -> None:
    """``file`` is not derivable and must not be left to a default.

    ``mhc_ligand_full_tsv.zip`` unpacks to ``mhc_ligand_full.tsv``; no rule
    turns one into the other, which is why this is recorded and not computed.
    """
    for source in SOURCES:
        assert source.file, f"{source.name} records no on-disk name"
        assert source.file == Path(source.file).name, f"{source.name} records a path, not a name"


def test_a_source_that_feeds_nothing_says_why_where_a_stem_invites_it() -> None:
    """A stem reads as "rename the download to this".

    Where that is wrong the row has to say so, because the registry is what an
    operator acts on.
    """
    for source in SOURCES:
        if source.handoff == "none" and source.stem is not None:
            assert source.hazard, (
                f"{source.name} advertises stem {source.stem!r} and no path to it, "
                "and says nothing about what dropping it in does"
            )


def test_the_handoff_values_are_the_three_the_readers_expect() -> None:
    assert {source.handoff for source in SOURCES} <= {"convert", "operator", "none"}


def test_the_proteome_mapping_reads_both_identifiers(tmp_path: Path) -> None:
    path = proteome(tmp_path / "p.fasta.gz", [("ERBB2", "P04626"), ("TP53", "P04637")])
    assert gene_accessions(path) == {"ERBB2": "P04626", "TP53": "P04637"}


def test_a_symbol_appearing_twice_keeps_the_first_accession(tmp_path: Path) -> None:
    """In a reference proteome the first is the canonical entry."""
    path = proteome(tmp_path / "p.fasta.gz", [("ERBB2", "P04626"), ("ERBB2", "X99999")])
    assert gene_accessions(path)["ERBB2"] == "P04626"


def test_a_header_without_a_gene_name_is_skipped_not_guessed(tmp_path: Path) -> None:
    path = tmp_path / "p.fasta.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(">sp|P04626|ERBB2_HUMAN No gene name here OS=Homo sapiens OX=9606 PE=1 SV=1\nMKV\n")
    assert gene_accessions(path) == {}


# ------------------------------------------------- what an empty result reports


def test_every_disease_is_carried_not_a_sorted_dozen(tmp_path: Path) -> None:
    """The truncation was applied to the data, and the names are sorted.

    So a table holding 72 diseases reported the twelve beginning A to C, and an
    operator whose disease began with S could not tell from that whether it was
    among the other sixty. The record now carries all of them; shortening is
    the printer's job.
    """
    rows = [(f"GENE{index}", f"Disease {index:03d}", "0.1") for index in range(40)]
    table = hpa(tmp_path / "hpa.tsv", rows)
    made = plasma_fold_from_hpa(table, tmp_path / "run", disease="Not A Disease")
    assert made.blocker == "disease_not_in_table"
    assert len(made.available) == 40, "every disease the file holds"
    assert "Disease 039" in made.available, "including the ones past a sorted dozen"


def test_a_shortened_line_says_how_many_it_left_out(tmp_path: Path) -> None:
    """Never a silent truncation: the count is what makes the line honest."""
    assert sample_line(["b", "a", "c"]) == "a, b, c"
    assert sample_line([f"d{index:02d}" for index in range(20)], limit=3) == ("d00, d01, d02 (+17 more)")
    assert sample_line([]) == ""
    assert sample_line(["only"], limit=1) == "only"
