"""The screen against the databases it actually downloads.

Every other test in this area builds a two-row fixture. A fixture proves the
plumbing and cannot prove the contract, because the contract is with files this
repository does not contain: 676,059 rows of Human Protein Atlas blood data, a
20,000-sequence reference proteome, and two IEDB exports of 10.26 GB and
1.29 GB. The things that break are the things a fixture gets to decide -- a
column spelling, a header that is two rows deep, how many rows one protein has
in one disease -- so they are asserted here against the real thing.

Skipped, not failed, when the downloads are absent: most machines running this
suite will not have 11 GB of reference data, and
``scripts/fetch_screening_tables.py`` is how it arrives.

Two rules shape what is pinned here.

**Compute the expectation, do not hardcode it.** A test asserting IL-5 moves
0.578 in psoriasis fails on the next HPA release and says nothing about the
code. So where a value comes from the data, these tests read it out of the raw
file themselves and assert the pipeline agrees. What is hardcoded is only what
is stable by definition: a UniProt accession for a gene, an ``O``/``P``/``Q``
accession shape, a column name the converter indexes by.

**Pin the prose that cannot be checked any other way.** ``Source.hazard``
claims the IEDB export has a two-row header and no column the reader knows.
That is a factual claim about a 10.26 GB file nobody will re-check by hand, and
it is the claim an operator acts on when deciding not to rename the file. It is
asserted here, cheaply, by reading two lines.
"""

from __future__ import annotations

import csv
import itertools
import re
from collections import Counter
from pathlib import Path

import pytest

from rfproteina.disease.liability import (
    _ALLELE_COLUMNS,
    _PEPTIDE_COLUMNS,
    _RANK_COLUMNS,
    _SCORE_COLUMNS,
    _load_proteome,
)
from rfproteina.disease.tables import (
    SOURCES,
    by_name,
    default_directory,
    downloads,
    gene_accessions,
    plasma_fold_from_hpa,
    stage_for_run,
)
from rfproteina.disease.target_liability import (
    FOLD_FILE_LIMIT,
    FOLD_ROW_LIMIT,
    TargetLimits,
    _linear_fold,
    measure_target_liability,
    uniprot_accession,
)

pytestmark = pytest.mark.slow

_TABLES = default_directory()
_HPA = _TABLES / "blood_pea_disease_de.tsv"
_PROTEOME = _TABLES / "UP000005640_9606.fasta.gz"
_IEDB_MHC = _TABLES / "mhc_ligand_full.tsv"
_IEDB_TCELL = _TABLES / "tcell_full_v3.tsv"
_IDC = _TABLES / "aggregated_ada_values.csv"


def _needs(path: Path) -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        not path.is_file(),
        reason=f"{path.name} is not downloaded; run scripts/fetch_screening_tables.py",
    )


needs_hpa = _needs(_HPA)
needs_proteome = _needs(_PROTEOME)
needs_iedb_mhc = _needs(_IEDB_MHC)
needs_iedb_tcell = _needs(_IEDB_TCELL)
needs_idc = _needs(_IDC)

#: UniProt accessions for genes whose mapping is not going to change. Used to
#: check the FASTA parse, not to check UniProt.
_KNOWN = {"ERBB2": "P04626", "TP53": "P04637", "EGFR": "P00533", "IL5": "P05113"}
_ACCESSION_SHAPE = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}$")


# ----------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def hpa_rows() -> list[dict[str, str]]:
    """The real blood table, read once for the whole session (~4s, 63 MB)."""
    with _HPA.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


@pytest.fixture(scope="session")
def accessions() -> dict[str, str]:
    return gene_accessions(_PROTEOME)


def _two_row_header(path: Path) -> tuple[list[str], list[str]]:
    """The first two lines only. Cheap even on a 10.26 GB file."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(itertools.islice(csv.reader(handle, delimiter="\t"), 2))
    return rows[0], rows[1]


# ------------------------------------------------- the columns the code indexes


@needs_hpa
def test_the_three_columns_the_converter_names_are_in_the_real_file() -> None:
    """The highest-value pin in this file.

    The converter indexes HPA by ``Gene``, ``Disease`` and ``logFC``. Rename any
    one of them upstream and the conversion reports ``column_absent`` -- which
    is at least loud. The quieter half is that these spellings are the only
    reason the axis works at all, and nothing else in the suite reads them from
    the real file.
    """
    from rfproteina.disease.tables import _HPA_DISEASE, _HPA_GENE, _HPA_LOG2

    with _HPA.open("r", encoding="utf-8-sig", newline="") as handle:
        names = csv.DictReader(handle, delimiter="\t").fieldnames or []
    for column in (_HPA_GENE, _HPA_DISEASE, _HPA_LOG2):
        assert column in names, f"{column!r} is gone from HPA; the fold axis is keyed on it"


@needs_hpa
def test_the_real_file_is_wider_than_the_fixtures_and_that_is_fine() -> None:
    """Indexed by name, so the extra columns are not a problem -- but the
    fixtures elsewhere are an 8-column subset of a 13-column file, and this
    says so rather than leaving the discrepancy to be discovered."""
    with _HPA.open("r", encoding="utf-8-sig", newline="") as handle:
        names = csv.DictReader(handle, delimiter="\t").fieldnames or []
    assert len(names) > 8
    assert "Mean importance" in names, "a column no fixture has, and none needs"


# --------------------------------------------------- the real disease vocabulary


@needs_hpa
def test_the_disease_list_is_reported_whole_and_not_a_sorted_dozen(
    hpa_rows: list[dict[str, str]], tmp_path: Path
) -> None:
    """At the scale that made the truncation a bug.

    The sample was twelve names and the names are sorted, so this table's 72
    diseases reported the ones beginning A to C. An operator whose disease began
    with S could not tell from that whether it was among the other sixty.
    """
    present = {(row.get("Disease") or "").strip() for row in hpa_rows}
    present.discard("")
    assert len(present) > 50, "a small vocabulary would not exercise the truncation"

    made = plasma_fold_from_hpa(_HPA, tmp_path / "run", disease="Not A Real Disease")
    assert made.blocker == "disease_not_in_table"
    assert set(made.available) == present, "every disease, not a sample"
    last = sorted(present)[-1]
    assert last in made.available, f"{last} is past a sorted dozen and must still be reported"


@needs_hpa
def test_a_disease_absent_from_this_release_is_named_as_absent(tmp_path: Path) -> None:
    """Asthma is the case that prompted this test, and it is a true absence.

    IL-5 is an asthma target, so pointing the screen at asthma is the obvious
    thing to do with it, and the blood atlas does not report that disease. The
    axis has to say so rather than return nothing and read as "does not move".
    """
    made = plasma_fold_from_hpa(_HPA, tmp_path / "run", disease="Asthma", genes=["IL5"])
    assert made.blocker == "disease_not_in_table"
    assert not made.written
    assert "asthma" not in {name.casefold() for name in made.available}


# ------------------------------------------- the reader's choice among real rows


@needs_hpa
@needs_proteome
@pytest.mark.parametrize("gene", ["IL5", "ERBB2", "TP53"])
def test_the_verdict_uses_the_loudest_sourced_row_of_the_real_table(
    gene: str, hpa_rows: list[dict[str, str]], accessions: dict[str, str], tmp_path: Path
) -> None:
    """Computed from the raw file, not hardcoded, so a new release cannot
    silently turn this into a test of nothing.

    HPA reports one row per assay binder, so a protein has several folds in one
    disease -- up to ten in this release. The conversion keeps them all and the
    reader judges the largest linear magnitude among them, which is the reading
    that cannot suppress a hit one binder saw and another missed. This finds a
    disease where that actually matters, works out the answer from the raw
    rows, and asserts the pipeline agrees.
    """
    folds: dict[str, list[float]] = {}
    for row in hpa_rows:
        if (row.get("Gene") or "").strip().upper() != gene:
            continue
        disease = (row.get("Disease") or "").strip()
        value = (row.get("logFC") or "").strip()
        if not disease or not value:
            continue
        try:
            folds.setdefault(disease, []).append(float(value))
        except ValueError:
            continue
    multi = {name: values for name, values in folds.items() if len(values) > 1}
    if not multi:
        pytest.skip(f"{gene} has no disease with several rows in this release")

    disease = sorted(multi, key=lambda name: len(multi[name]), reverse=True)[0]
    values = multi[disease]
    expected = max(values, key=lambda value: _linear_fold("log2_fold_change", value))

    run = tmp_path / "run"
    made = plasma_fold_from_hpa(_HPA, run, disease=disease, genes=[gene], release="v26.0", accessions=accessions)
    assert made.usable, made.as_dict()
    assert made.rows == len(values), f"all {len(values)} binder rows are kept, not collapsed"

    record = measure_target_liability(_KNOWN[gene], directory=run, limits=TargetLimits(fold_max=2.0))["plasma_fold"]
    assert record["measurement"] == pytest.approx(expected), (
        f"{gene} in {disease}: the reader took {record['measurement']} from {values}, " f"and the loudest is {expected}"
    )


@needs_hpa
def test_no_single_protein_exceeds_the_per_target_row_cap(
    hpa_rows: list[dict[str, str]],
) -> None:
    """The bound that applies after narrowing, checked against the real spread.

    If a real release ever did put more than ``FOLD_ROW_LIMIT`` rows on one
    protein in one disease, the axis would report ``fold_too_large`` for that
    target and the converter would say so at conversion time. Neither is wrong,
    but it would be a surprise worth having a test fail about.
    """
    counts = Counter(
        ((row.get("Gene") or "").strip(), (row.get("Disease") or "").strip())
        for row in hpa_rows
        if (row.get("Gene") or "").strip() and (row.get("Disease") or "").strip()
    )
    worst_pair, worst = counts.most_common(1)[0]
    assert worst <= FOLD_ROW_LIMIT, f"{worst_pair} has {worst} rows, over the per-target cap of {FOLD_ROW_LIMIT}"


# ------------------------------------------------ the trap a fixture approximates


@needs_hpa
def test_an_unnarrowed_conversion_of_a_real_disease_is_written_and_refused(
    hpa_rows: list[dict[str, str]], tmp_path: Path
) -> None:
    """The reason ``targets`` is not optional, at real scale.

    Every gene HPA reports for one disease is past the reader's 1 MB file bound,
    so the conversion writes a file the screen then refuses. That is the worst
    shape available -- present on disk, measuring nothing -- and the fixtures
    only approximate it by generating 20,000 synthetic rows.
    """
    counts = Counter((row.get("Disease") or "").strip() for row in hpa_rows if (row.get("Disease") or "").strip())
    disease, _rows = counts.most_common(1)[0]

    made = plasma_fold_from_hpa(_HPA, tmp_path / "run", disease=disease)
    assert made.written, "the file is written"
    assert made.path is not None and made.path.stat().st_size > FOLD_FILE_LIMIT
    assert not made.usable
    assert made.blocker is not None and made.blocker.startswith("bytes_over_reader_limit")

    record = measure_target_liability("P04626", directory=tmp_path / "run", limits=TargetLimits(fold_max=2.0))[
        "plasma_fold"
    ]
    assert record["blocker"] == "fold_too_large"


# ---------------------------------------------------------------- the proteome


@needs_proteome
def test_the_downloaded_proteome_is_read_and_clears_the_floor() -> None:
    """Gzipped, protein, and big enough to mean something.

    The floor exists because every 9-mer the file lacks is an off-target a
    design would be cleared against, so a proteome that parses but is too small
    is refused. The real one has to clear it, or the axis is unusable with the
    file this program fetches for it.
    """
    blocker, kmers = _load_proteome(_PROTEOME)
    assert blocker is None, blocker
    assert kmers.size > 1_000_000


@needs_proteome
def test_known_symbols_map_to_their_accessions(accessions: dict[str, str]) -> None:
    """Hardcoded on purpose: these pairs are stable UniProt facts, so a
    mismatch is a parsing bug and not a release difference."""
    for gene, accession in _KNOWN.items():
        assert accessions.get(gene) == accession


@needs_proteome
def test_every_mapped_value_is_shaped_like_an_accession(accessions: dict[str, str]) -> None:
    """Catches a parse that drifts onto the wrong bar-separated field.

    The header is ``>sp|P04626|ERBB2_HUMAN``, and taking the first or third
    field instead of the second yields ``sp`` or ``ERBB2_HUMAN`` for every row
    -- a mapping that is wrong everywhere and still looks populated. Also
    checked through ``uniprot_accession``, which is what the screen resolves a
    target with, so the two agree on what an accession is.
    """
    assert len(accessions) > 15_000
    for gene, accession in accessions.items():
        assert _ACCESSION_SHAPE.match(accession), f"{gene} -> {accession!r} is not an accession"
        assert uniprot_accession(accession) == accession, f"the screen rejects {accession!r}"


@needs_proteome
def test_the_mapping_is_injective_enough_to_be_a_mapping(accessions: dict[str, str]) -> None:
    """A symbol keeps its first accession, and the first is the canonical one.

    If the parse were collapsing headers, many symbols would share one
    accession. A handful legitimately do in a reference proteome; most of the
    file sharing one would be a bug that this catches and a spot check does not.
    """
    reused = Counter(accessions.values())
    worst, count = reused.most_common(1)[0]
    assert count < 10, f"{worst} is the accession for {count} symbols, which is a collapsed parse"


# ------------------------------- pinning the hazard prose against the real files


@needs_iedb_mhc
def test_the_iedb_ligand_export_is_shaped_as_the_hazard_says() -> None:
    """The claim an operator acts on when deciding not to rename this file.

    ``Source.hazard`` says: a two-row header, 112 columns, and none of them a
    peptide, allele or rank column by the names the reader uses. Nobody will
    re-check that by hand on a 10.26 GB file, so it is checked here from two
    lines. If it ever stops being true, the registry is giving advice about a
    file that no longer exists.
    """
    group, sub = _two_row_header(_IEDB_MHC)
    assert len(sub) > 100, "a two-row header of about 112 columns"
    assert len(group) == len(sub)
    assert (
        "Epitope" in group and "MHC Restriction" in group
    ), "the group row names the fields the reader would need, under other names"

    flat = {name.strip().lower().replace(" ", "_") for name in (*group, *sub)}
    flat |= {f"{g} {s}".strip().lower().replace(" ", "_") for g, s in zip(group, sub, strict=True)}
    for label, columns in (
        ("peptide", _PEPTIDE_COLUMNS),
        ("allele", _ALLELE_COLUMNS),
        ("rank", _RANK_COLUMNS),
        ("score", _SCORE_COLUMNS),
    ):
        overlap = set(columns) & flat
        assert not overlap, (
            f"the export now has a {label} column the reader understands ({overlap}); "
            "the hazard text and the 'no conversion' handoff both need revisiting"
        )

    source = by_name("IEDB MHC ligand assays")
    assert source is not None and source.handoff == "none"
    assert source.hazard, "a stem with no path to it has to say what dropping it in does"


@needs_iedb_tcell
def test_the_iedb_tcell_export_is_shaped_as_the_hazard_says() -> None:
    """Same claim, same cost, for the file whose drop-in displaces the Calis
    computation rather than the DR-breadth one."""
    group, sub = _two_row_header(_IEDB_TCELL)
    assert len(sub) > 150, "about 161 columns"
    flat = {name.strip().lower().replace(" ", "_") for name in (*group, *sub)}
    assert not set(_SCORE_COLUMNS) & flat, "a score column here would be read as a supplied score"

    source = by_name("IEDB T cell assays")
    assert source is not None and source.handoff == "none" and source.hazard


@needs_idc
def test_the_idc_export_has_no_column_the_score_reader_accepts() -> None:
    """The hazard that matters most, because it suppresses rather than fabricates.

    Renamed to ``ada.csv`` this file reports ``score_absent`` -- and a supplied
    table takes precedence, so the DR-breadth measurement ``mhc_ii.tsv`` would
    have produced is displaced and never run. The axis reads untested where it
    could have been a hit. If a future release ever did add a ``score`` column,
    the failure mode changes from suppression to a fabricated number, and this
    test is the warning.
    """
    with _IDC.open("r", encoding="utf-8-sig", newline="") as handle:
        names = next(csv.reader(handle))
    assert len(names) > 50, "57 columns in this release"
    flat = {name.strip().lower().replace(" ", "_") for name in names}
    assert not set(_SCORE_COLUMNS) & flat, (
        "the IDC export now has a score column; dropped in at the ada stem it would "
        "report a clinical rate as this design's own score"
    )

    source = by_name("IDC anti-drug antibody incidence")
    assert source is not None and source.handoff == "none" and source.hazard


# ------------------------------------------------- the registry against the disk


def test_every_registry_source_is_on_disk_under_the_name_it_records() -> None:
    """The bug that started this: a stocked directory reported as empty.

    Skipped rather than failed when nothing is downloaded, because an empty
    machine is not a wrong registry.
    """
    found = downloads(_TABLES)
    if not found.present:
        pytest.skip("no screening tables are downloaded")
    for source, path in found.present:
        assert path.name == source.file
        assert path.stat().st_size > 0


@pytest.mark.parametrize("source", [item for item in SOURCES if not item.url.endswith(".zip")], ids=lambda s: s.file)
def test_a_declared_size_matches_the_file_that_arrived(source) -> None:
    """``size`` is bytes over the wire, which for an unzipped source is also
    the bytes on disk. Pinned only for those: a zipped source's declared size
    is the archive, and the HPA zip is 7.34 MB against 63 MB unpacked."""
    path = _TABLES / source.file
    if not path.is_file() or source.size is None:
        pytest.skip(f"{source.file} is not downloaded, or declares no size")
    assert path.stat().st_size == source.size


# ------------------------------------------------------- the whole path, for real


@needs_hpa
@needs_proteome
def test_staging_a_run_from_the_real_downloads_measures_both_axes(tmp_path: Path) -> None:
    """End to end on the real files: download, stage, screen, verdict.

    Narrowed by accession, which is what a run downstream of generation holds,
    through a disease the table actually reports. Both axes a download can serve
    have to come back measured -- that is the whole of what this wiring is for,
    and every other test here checks one joint of it.
    """
    run = tmp_path / "run"
    made = stage_for_run(run, disease="Psoriasis", source_directory=_TABLES, targets=["P05113"], release="v26.0")
    assert made.equipped, made.as_dict()
    assert [stem for stem, _path in made.staged] == ["plasma_fold"]
    assert made.fold is not None and made.fold.answers_accessions
    assert made.proteome is not None, "the off-target axis needs the FASTA it downloaded"

    found = measure_target_liability("P05113", directory=run, limits=TargetLimits(fold_max=2.0))["plasma_fold"]
    assert found["state"] in {"hit", "clear"}, found
    assert found.get("blocker") is None
    assert "Human Protein Atlas" in str(found["fold_source"])
    assert "Psoriasis" in str(found["fold_source"]), "the provenance names the disease asked for"

    blocker, kmers = _load_proteome(made.proteome)
    assert blocker is None and kmers.size > 1_000_000
