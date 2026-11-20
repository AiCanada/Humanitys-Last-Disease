"""The screening table sources, and the 10 GB limit on fetching them.

The limit is the whole point of this module, and it has to hold in the case
where the obvious implementation does not. Every source here sits beside a
whole-database file with a near-identical name and a size between 11 GB and
1.39 TB, so "download the database" reaches the wrong one by default. A guard
that only read ``Content-Length`` would stop four of those five and wave
through the one source that sends no ``Content-Length`` at all -- IEDB, which
is also the largest thing this actually fetches.

So the guard is tested twice over: a declared size past the limit is refused
before any body is read, and an *undeclared* body that grows past the limit is
abandoned while arriving. The second is the one that matters, and it is the one
a header-only check silently fails.

No test here reaches the network. ``fetch`` takes its opener, so every case is
driven by a fake server whose behaviour -- including lying about its size -- is
written out in the test.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from rfproteina.disease.tables import (
    BINDINGDB_PANEL_ACCESSIONS,
    BINDINGDB_PANEL_SUBSET_FILE,
    GTOPDB_FILES,
    GTOPDB_OFFTARGET_COLUMNS,
    GTOPDB_PANEL_GENES,
    PANEL_SUBSET_COLUMNS,
    QUERIES,
    REFUSED,
    SIZE_LIMIT,
    SOURCES,
    Source,
    TooLargeError,
    bindingdb_panel_subset,
    by_name,
    coverage,
    declared_size,
    fetch,
    gtopdb_offtarget_subset,
    gtopdb_panel_subset,
    gtopdb_release,
    merge_panel_subsets,
    missing_gtopdb_files,
    readable,
)


class _Response(io.BytesIO):
    """One fake HTTP response, with headers and a context manager."""

    def __init__(self, body: bytes = b"", headers: dict[str, str] | None = None) -> None:
        super().__init__(body)
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def server(body: bytes, *, declare: int | None = None, head: bool = True):
    """An opener that serves ``body``, declaring ``declare`` bytes if asked.

    ``declare`` is deliberately separate from ``len(body)`` so a test can make
    the server lie, which is the case the running count exists for.
    """

    def opener(request, timeout=None):  # noqa: ARG001 - signature mirrors urlopen
        headers = {} if declare is None else {"Content-Length": str(declare)}
        if getattr(request, "method", None) == "HEAD" or request.get_method() == "HEAD":
            return _Response(b"", headers if head else {})
        return _Response(body, headers)

    return opener


def a_source(**changes) -> Source:
    base = {
        "name": "test source",
        "url": "https://example.invalid/table.tsv",
        "stem": "mhc_i",
        "axis": "mhc_i",
        "size": None,
        "licence": "CC BY 4.0",
    }
    return Source(**{**base, **changes})


class TestTheLimit:
    def test_a_declared_size_over_the_limit_is_refused_before_anything_is_read(self, tmp_path: Path) -> None:
        """The cheap check: no body is fetched at all."""
        body = b"x" * 10
        with pytest.raises(TooLargeError, match="Nothing was downloaded"):
            fetch(a_source(), tmp_path, limit=100, opener=server(body, declare=101))
        assert list(tmp_path.iterdir()) == []

    def test_a_body_that_outgrows_the_limit_is_abandoned_mid_download(self, tmp_path: Path) -> None:
        """The check that matters, and the one a header-only guard fails.

        The server declares nothing -- which is exactly what IEDB does for
        every one of its exports -- and then sends more than the limit.
        """
        with pytest.raises(TooLargeError, match="while downloading"):
            fetch(a_source(), tmp_path, limit=64, opener=server(b"y" * 4096, declare=None))
        assert list(tmp_path.iterdir()) == [], "a refused download left a file behind"

    def test_a_server_that_understates_its_size_is_still_caught(self, tmp_path: Path) -> None:
        """A declared size is a claim. The bytes are what count."""
        with pytest.raises(TooLargeError, match="while downloading"):
            fetch(a_source(), tmp_path, limit=64, opener=server(b"z" * 4096, declare=1))
        assert list(tmp_path.iterdir()) == []

    def test_the_error_says_the_server_declared_nothing_when_it_did(self, tmp_path: Path) -> None:
        """So the reader can tell an overrun from a lie without re-running it."""
        with pytest.raises(TooLargeError, match="declared no size"):
            fetch(a_source(), tmp_path, limit=64, opener=server(b"y" * 4096, declare=None))

    def test_a_download_inside_the_limit_lands_whole(self, tmp_path: Path) -> None:
        body = b"allele\tpeptide\nA*02:01\tSIINFEKL\n"
        fetch(a_source(), tmp_path, limit=SIZE_LIMIT, opener=server(body, declare=len(body)))
        written = sorted(path.name for path in tmp_path.iterdir())
        assert written == ["table.tsv"]
        assert (tmp_path / "table.tsv").read_bytes() == body

    def test_no_part_file_survives_a_refusal_or_a_success(self, tmp_path: Path) -> None:
        """A half file named like a whole one is a table nobody knows is partial."""
        with pytest.raises(TooLargeError):
            fetch(a_source(), tmp_path, limit=8, opener=server(b"q" * 999, declare=None))
        assert not any(path.name.endswith(".part") for path in tmp_path.iterdir())
        fetch(a_source(), tmp_path, limit=SIZE_LIMIT, opener=server(b"ok", declare=2))
        assert not any(path.name.endswith(".part") for path in tmp_path.iterdir())

    def test_the_shipped_limit_is_ten_gigabytes(self) -> None:
        assert SIZE_LIMIT == 10 * 1024**3
        assert readable(SIZE_LIMIT) == "10.00 GB"


class TestDeclaredSize:
    def test_a_missing_content_length_reads_as_unknown_not_as_zero(self) -> None:
        """Unknown must not be mistaken for small, which is the IEDB case."""
        assert declared_size("https://example.invalid/x", opener=server(b"", declare=None)) is None

    def test_a_declared_size_is_returned(self) -> None:
        assert declared_size("https://example.invalid/x", opener=server(b"", declare=4096)) == 4096

    def test_an_unreachable_host_is_unknown_rather_than_an_error(self) -> None:
        def broken(request, timeout=None):  # noqa: ARG001
            raise OSError("no route to host")

        assert declared_size("https://example.invalid/x", opener=broken) is None


class TestArchives:
    def test_a_zip_is_unpacked_and_the_archive_removed(self, tmp_path: Path) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as bundle:
            bundle.writestr("mhc_ligand_full.tsv", "allele\tpeptide\n")
        body = buffer.getvalue()
        fetch(
            a_source(url="https://example.invalid/pack.zip"),
            tmp_path,
            limit=SIZE_LIMIT,
            opener=server(body, declare=len(body)),
        )
        written = sorted(path.name for path in tmp_path.iterdir())
        assert written == ["mhc_ligand_full.tsv"]

    def test_a_zip_member_escaping_the_directory_is_not_written(self, tmp_path: Path) -> None:
        """An archive naming ../ writes a file nobody asked for, outside the run."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as bundle:
            bundle.writestr("../escaped.tsv", "no\n")
            bundle.writestr("kept.tsv", "yes\n")
        body = buffer.getvalue()
        fetch(
            a_source(url="https://example.invalid/pack.zip"),
            tmp_path,
            limit=SIZE_LIMIT,
            opener=server(body, declare=len(body)),
        )
        assert sorted(path.name for path in tmp_path.iterdir()) == ["kept.tsv"]
        assert not (tmp_path.parent / "escaped.tsv").exists()


class TestTheRegistry:
    def test_every_source_has_a_verified_https_url(self) -> None:
        for source in SOURCES:
            assert source.url.startswith("https://"), source.name

    def test_every_known_size_is_under_the_limit(self) -> None:
        """A source listed here must be one this is willing to fetch."""
        for source in SOURCES:
            if source.size is not None:
                assert source.size < SIZE_LIMIT, f"{source.name} is {readable(source.size)}"

    def test_every_refusal_is_genuinely_over_the_limit(self) -> None:
        """An entry that is not actually too large would be a refusal with no reason."""
        for refused in REFUSED:
            assert refused.size > SIZE_LIMIT, f"{refused.name} is only {readable(refused.size)}"

    def test_every_refusal_names_a_substitute_that_exists(self) -> None:
        """Telling someone not to take a file is only useful with the one to take.

        Each refusal points either at a download in this registry or at a live
        query that answers the same question without the bulk file -- STRING's
        133.76 GB link dump is replaced by its interaction-partners endpoint,
        not by a smaller download. Prose is allowed only for the two cases where
        the substitute is a file at the same host that this does not fetch, and
        it has to be recognisable as prose so it cannot be a stale name.
        """
        known = {source.name for source in SOURCES} | {query.name for query in QUERIES}
        for refused in REFUSED:
            assert refused.instead, refused.name
            if not refused.instead.startswith("the "):
                assert refused.instead in known, (
                    f"{refused.name} points at {refused.instead!r}, which is neither a listed "
                    "source nor a listed query"
                )

    def test_the_refusals_cover_the_five_traps_that_prompted_the_limit(self) -> None:
        """Pinned, because a removed entry silently re-opens the trap it named."""
        assert len(REFUSED) == 5
        biggest = max(refused.size for refused in REFUSED)
        assert biggest > 1_000 * 1024**3, "the 1.39 TB raw spectra deposit is the one to never follow"

    def test_a_source_with_no_reader_says_so(self) -> None:
        """A table nothing opens must not look like one the screen will read.

        Asserted on ``handoff`` rather than on the wording of the note. The
        prose is the explanation and it gets rewritten; the field is the claim,
        and pinning the claim to a spelling meant that improving a note broke
        this test while a source that genuinely lied about having a reader
        would have passed it.
        """
        for source in SOURCES:
            if source.stem is None:
                assert source.handoff != "convert", f"{source.name} has no stem to convert into, so it cannot be staged"
                assert "not a table" in source.note.lower(), source.name

    def test_the_stems_are_ones_the_screen_actually_reads(self) -> None:
        """Bound to the reader, so a renamed stem here cannot drift from it."""
        from rfproteina.disease.coevolution import COEVOLUTION_TABLE
        from rfproteina.disease.solubility import SOLUBILITY_TABLE
        from rfproteina.disease.toxicology import TOXICOLOGY_TABLE

        known = {
            "mhc_i",
            "mhc_ii",
            "ada",
            "immunogenicity",
            "plasma_fold",
            COEVOLUTION_TABLE,
            SOLUBILITY_TABLE,
            TOXICOLOGY_TABLE,
        }
        for source in SOURCES:
            if source.stem is not None:
                assert source.stem in known, f"{source.name} writes {source.stem}, which no axis reads"

    def test_by_name_finds_a_source_and_refuses_a_typo(self) -> None:
        assert by_name("IDC anti-drug antibody incidence") is not None
        assert by_name("IDC anti drug antibody incidence") is None


def test_readable_is_comparable_against_the_limit_at_a_glance() -> None:
    assert readable(62_813_591) == "59.90 MB"
    assert readable(133_758_389_005) == "124.57 GB"
    # The biggest refusal reads as TB, not as four digits of GB.
    assert readable(1_393_178_404_881) == "1.27 TB"


class TestWhatIsLocalAndWhatGoesOnline:
    """The question a half-stocked run directory raises.

    Downloading the human subset does not make a run offline: three axes are
    served by REST lookups and no download replaces them. A report that only
    listed missing files would imply the opposite, and the operator would find
    out from a network error on a machine they believed was stocked.
    """

    def test_an_empty_directory_names_every_table_it_lacks(self, tmp_path: Path) -> None:
        found = coverage(tmp_path)
        assert not found.complete
        assert {stem for stem, _name in found.missing} == {"plasma_fold"}
        assert found.present == ()

    def test_an_axis_no_download_feeds_is_not_listed_as_a_missing_download(self) -> None:
        """Rewritten, and deliberately narrower than it was.

        This used to assert that an empty directory lacks ``mhc_i``,
        ``immunogenicity`` and ``ada`` as well, which read as "fetch these
        three and the run is stocked". None of the three is true: every one of
        those downloads is a different artifact from the table its axis reads
        (an epitope census or a clinical ADA table, where the axis wants a scan
        of this design's own chain), so no download closes the gap and the
        report was sending an operator after files that would not have helped.

        For ``ada`` and ``immunogenicity`` it was worse than unhelpful. Both
        axes prefer a supplied table over their own computation, so a file
        dropped in at that stem displaces the verdict ``mhc_ii``/``mhc_i``
        would have produced -- and since neither download carries a column
        these readers recognise, the axis reports untested where it could have
        been measured. The registry records that as :attr:`Source.hazard`;
        this pins that the report never asks for it.
        """
        listed = {stem for stem, _name in coverage(Path("/nonexistent")).missing}
        for stem in ("mhc_i", "immunogenicity", "ada"):
            assert stem not in listed, (
                f"{stem} is listed as a missing download; no download produces it, "
                "and the file that looks like it suppresses a real verdict"
            )

    def test_a_table_on_disk_is_found_under_either_suffix(self, tmp_path: Path) -> None:
        (tmp_path / "plasma_fold.csv").write_text("x\n", encoding="utf-8")
        found = coverage(tmp_path)
        assert {stem for stem, _path in found.present} == {"plasma_fold"}
        assert found.complete

    def test_a_fully_stocked_run_is_complete_but_still_goes_online(self, tmp_path: Path) -> None:
        """The point of the report: complete does not mean offline."""
        (tmp_path / "plasma_fold.tsv").write_text("x\n", encoding="utf-8")
        found = coverage(tmp_path)
        assert found.complete
        assert found.online, "a complete run still has live lookups, and must say so"
        assert {query.axis for query in found.online} == {"paralog", "plasma_abundance", "plasma_hub"}

    def test_a_missing_table_names_the_source_to_get_it_from(self, tmp_path: Path) -> None:
        names = dict(coverage(tmp_path).missing)
        assert "Human Protein Atlas" in names["plasma_fold"]

    def test_the_record_is_json_shaped(self, tmp_path: Path) -> None:
        import json

        json.dumps(coverage(tmp_path).as_dict())


class TestTheQueryRegistry:
    def test_every_query_endpoint_is_one_the_screen_actually_calls(self) -> None:
        """Bound to the caller, so a moved endpoint cannot leave this stale."""
        from pathlib import Path as _Path

        from rfproteina.disease import target_liability

        source = _Path(target_liability.__file__).read_text(encoding="utf-8")
        for query in QUERIES:
            for endpoint in query.endpoint.split(", "):
                assert endpoint in source, f"{query.name}: {endpoint} is not called by target_liability"

    def test_every_query_names_how_it_is_narrowed(self) -> None:
        """An unnarrowed query against STRING or UniProt is the bulk download again."""
        for query in QUERIES:
            assert query.narrowed, query.name
            assert "9606" in query.narrowed or "one accession" in query.narrowed or "one identifier" in query.narrowed

    def test_the_oversized_sources_are_each_answered_by_a_query_or_a_human_file(self) -> None:
        """No refusal is a dead end: each names something that works instead."""
        covered = " ".join(query.note + query.name for query in QUERIES)
        for refused in REFUSED:
            answered = refused.instead in {source.name for source in SOURCES} or refused.name.split()[0] in covered
            assert answered or refused.instead, f"{refused.name} refuses with no way forward"


# --------------------------------------------------------------------- GtoPdb
#
# Judged on three things: that it reads GtoPdb's own shape, that it writes the
# shared panel-subset format so its rows reach a reader at all, and that what
# it writes cannot be mistaken for the screen's toxicology table. The earlier
# long-form reference table satisfied only the first, and was kept apart from
# the panel by its filename alone.

_GTOPDB_VERSION_LINE = '"# GtoPdb Version: 2026.3.1 - published: 2026-10-12"'
_GTOPDB_COLUMNS = (
    "Target Gene Symbol",
    "Target Species",
    "Ligand",
    "Ligand ID",
    "Type",
    "Original Affinity Units",
    "Original Affinity Median nm",
    "Original Affinity Relation",
    "PubMed ID",
)
#: Aspirin, so the recomputed InChIKey is a real one.
_GTOPDB_SMILES = "CC(=O)Oc1ccccc1C(=O)O"


def _gtopdb_row(**overrides: str) -> dict[str, str]:
    row = {
        "Target Gene Symbol": "KCNH2",
        "Target Species": "Human",
        "Ligand": "elacridar",
        "Ligand ID": "13578",
        "Type": "Inhibitor",
        "Original Affinity Units": "IC50",
        "Original Affinity Median nm": "40.0",
        "Original Affinity Relation": "=",
        "PubMed ID": "12636153",
    }
    row.update(overrides)
    return row


def _gtopdb_csv(path: Path, rows: list[dict[str, str]], *, version_comment: bool = True) -> Path:
    import csv as _csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        if version_comment:
            handle.write(_GTOPDB_VERSION_LINE + "\n")
        writer = _csv.DictWriter(handle, fieldnames=list(_GTOPDB_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _gtopdb_ligands(path: Path, mapping: dict[str, str] | None = None) -> Path:
    """GtoPdb's ligands.csv: the structures interactions.csv does not carry."""
    import csv as _csv

    rows = mapping if mapping is not None else {"13578": _GTOPDB_SMILES}
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(_GTOPDB_VERSION_LINE + "\n")
        writer = _csv.DictWriter(handle, fieldnames=["Ligand ID", "Name", "SMILES", "InChIKey"], lineterminator="\n")
        writer.writeheader()
        for key, smiles in rows.items():
            writer.writerow({"Ligand ID": key, "Name": f"ligand{key}", "SMILES": smiles, "InChIKey": "PUBLISHED-KEY-N"})
    return path


def _subset_rows(path: Path) -> list[dict[str, str]]:
    import csv as _csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(_csv.DictReader(handle, delimiter="\t"))


def _gtopdb_pair(tmp_path, rows, **kwargs):
    return (
        _gtopdb_csv(tmp_path / "interactions.csv", rows, **kwargs),
        _gtopdb_ligands(tmp_path / "ligands.csv"),
    )


def test_gtopdb_panel_genes_cover_every_ligand_axis():
    from rfproteina.disease.combinations import LIGAND_AXES

    assert set(GTOPDB_PANEL_GENES) == set(LIGAND_AXES)
    assert GTOPDB_PANEL_GENES["nav1_5"] == "SCN5A"
    assert GTOPDB_PANEL_GENES["bsep"] == "ABCB11"


def test_gtopdb_is_registered_and_closes_no_gap_by_itself():
    """Its path to an axis runs through a per-ligand lookup, as BindingDB's does."""
    from rfproteina.disease.toxicology import TOXICOLOGY_TABLE

    source = next(item for item in SOURCES if "Guide to PHARMACOLOGY" in item.name)
    assert source.handoff == "none"
    assert source.stem == TOXICOLOGY_TABLE, "the stem names the tempting destination"
    assert "ODbL" in source.licence
    assert source.hazard


def test_gtopdb_subset_writes_the_shared_format(tmp_path):
    """The point of the reshape: one format, so one reader serves both sources."""
    interactions, ligands = _gtopdb_pair(tmp_path, [_gtopdb_row()])
    result = gtopdb_panel_subset(interactions, ligands, tmp_path / "out")
    assert result.blocker is None
    assert result.rows == 1
    assert result.ligands == 1
    assert result.axes == ("herg",)
    assert result.path.name == "gtopdb_panel_subset.tsv"
    header = result.path.read_text(encoding="utf-8").splitlines()[0].split("\t")
    assert header == list(PANEL_SUBSET_COLUMNS)
    (row,) = _subset_rows(result.path)
    assert row["panel_axis"] == "herg"
    assert row["accession"] == "Q12809", "the UniProt accession, so it joins with BindingDB's rows"
    assert row["affinity_nanomolar"] == "40.0"
    assert row["relation"] == "="
    assert row["assay"] == "IC50"
    assert row["chains"] == "1"
    assert row["smiles"] == _GTOPDB_SMILES
    assert row["inchikey"] and row["inchikey"] != "PUBLISHED-KEY-N", "recomputed, not taken on trust"
    assert row["inchikey_upstream"] == "PUBLISHED-KEY-N", "and theirs kept beside it"
    assert "GtoPdb" in row["curation"] and "ODbL" in row["curation"]
    assert "12636153" in row["source"]


def test_gtopdb_subset_needs_the_structures(tmp_path):
    """interactions.csv has a ligand id and no SMILES, so both files are required."""
    interactions = _gtopdb_csv(tmp_path / "interactions.csv", [_gtopdb_row()])
    assert gtopdb_panel_subset(interactions, tmp_path / "absent.csv", tmp_path / "o").blocker == "ligands_not_found"
    assert (
        gtopdb_panel_subset(tmp_path / "absent.csv", tmp_path / "absent.csv", tmp_path / "o").blocker
        == "interactions_not_found"
    )


def test_gtopdb_subset_counts_a_ligand_with_no_structure(tmp_path):
    """11,590 of GtoPdb's 14,029 ligands have a SMILES. The rest cannot be fingerprinted."""
    interactions = _gtopdb_csv(tmp_path / "interactions.csv", [_gtopdb_row(**{"Ligand ID": "99999"})])
    ligands = _gtopdb_ligands(tmp_path / "ligands.csv")
    result = gtopdb_panel_subset(interactions, ligands, tmp_path / "out")
    assert not result.written
    assert result.skipped.get("ligand_without_structure") == 1


def test_gtopdb_subset_skips_the_version_comment_and_survives_its_absence(tmp_path):
    with_comment, ligands = _gtopdb_pair(tmp_path, [_gtopdb_row()])
    assert gtopdb_panel_subset(with_comment, ligands, tmp_path / "a").rows == 1
    bare = _gtopdb_csv(tmp_path / "bare.csv", [_gtopdb_row()], version_comment=False)
    assert gtopdb_panel_subset(bare, ligands, tmp_path / "b").rows == 1


def test_gtopdb_subset_narrows_to_one_species(tmp_path):
    interactions, ligands = _gtopdb_pair(
        tmp_path,
        [_gtopdb_row(), _gtopdb_row(**{"Target Species": "Rat", "Original Affinity Median nm": "1.0"})],
    )
    result = gtopdb_panel_subset(interactions, ligands, tmp_path / "out")
    assert result.rows == 1, "a rat hERG IC50 is a measurement of a different protein"
    assert result.skipped.get("other_species") == 1


def test_gtopdb_subset_keeps_only_binding_and_inhibition_constants(tmp_path):
    rows = [
        _gtopdb_row(**{"Original Affinity Units": unit, "Original Affinity Median nm": "10.0"})
        for unit in ("IC50", "Ki", "Kd", "EC50", "KB")
    ]
    interactions, ligands = _gtopdb_pair(tmp_path, rows)
    result = gtopdb_panel_subset(interactions, ligands, tmp_path / "out")
    assert result.rows == 3, "EC50 and KB are a different quantity from the panel cut"
    assert result.skipped.get("assay_not_an_affinity") == 2


def test_gtopdb_subset_drops_an_approximate_relation_but_keeps_a_bound(tmp_path):
    rows = [
        _gtopdb_row(**{"Original Affinity Relation": relation, "Original Affinity Median nm": "50.0"})
        for relation in ("=", "", "<", ">", "~")
    ]
    interactions, ligands = _gtopdb_pair(tmp_path, rows)
    result = gtopdb_panel_subset(interactions, ligands, tmp_path / "out")
    assert result.rows == 4
    assert result.skipped.get("approximate_relation") == 1
    assert [row["relation"] for row in _subset_rows(result.path)] == ["=", "=", "<", ">"]


def test_gtopdb_subset_refuses_a_value_that_is_not_a_concentration(tmp_path):
    rows = [_gtopdb_row(**{"Original Affinity Median nm": value}) for value in ("40.0", "", "not-a-number", "-1.0")]
    interactions, ligands = _gtopdb_pair(tmp_path, rows)
    result = gtopdb_panel_subset(interactions, ligands, tmp_path / "out")
    assert result.rows == 1
    assert result.skipped.get("no_nanomolar_value") == 1
    assert result.skipped.get("value_unparsed") == 1
    assert result.skipped.get("value_not_a_concentration") == 1


def test_gtopdb_subset_names_the_axes_it_found_nothing_for(tmp_path):
    interactions, ligands = _gtopdb_pair(tmp_path, [_gtopdb_row()])
    result = gtopdb_panel_subset(interactions, ligands, tmp_path / "out")
    assert result.axes == ("herg",)
    assert "bsep" in result.missing, "an axis with nothing in it is named, not left to subtraction"


def test_gtopdb_subset_writes_nothing_when_nothing_matches(tmp_path):
    interactions, ligands = _gtopdb_pair(tmp_path, [_gtopdb_row(**{"Target Gene Symbol": "EGFR"})])
    result = gtopdb_panel_subset(interactions, ligands, tmp_path / "out")
    assert not result.written
    assert result.blocker == "panel_not_in_table"


def test_gtopdb_subset_refuses_bad_arguments(tmp_path):
    interactions, ligands = _gtopdb_pair(tmp_path, [_gtopdb_row()])
    assert gtopdb_panel_subset(interactions, ligands, tmp_path / "o", axes=["nope"]).blocker == "axes_not_in_panel:nope"
    assert gtopdb_panel_subset(interactions, ligands, tmp_path / "o", axes=[]).blocker == "axes_named_but_empty"
    assert gtopdb_panel_subset(interactions, ligands, tmp_path / "o", species=" ").blocker == "species_not_named"


def test_a_gtopdb_subset_cannot_be_misread_as_the_screens_panel_table(tmp_path):
    """Guarded by the columns, not the filename, which the old form was not."""
    from rfproteina.disease.toxicology import measure_toxicology

    interactions, ligands = _gtopdb_pair(tmp_path, [_gtopdb_row()])
    result = gtopdb_panel_subset(interactions, ligands, tmp_path / "out")
    run = tmp_path / "run"
    run.mkdir()
    (run / "toxicology.tsv").write_text(result.path.read_text(encoding="utf-8"), encoding="utf-8")
    measured = measure_toxicology(run)
    assert all(record.get("measurement") is None for record in measured.values())


# ------------------------------------------------------- merging the two sources


def test_merge_joins_two_sources_and_keeps_their_licences_apart(tmp_path):
    """Volume and curation answer different questions; there is no need to choose."""
    interactions, ligands = _gtopdb_pair(tmp_path, [_gtopdb_row()])
    gto = gtopdb_panel_subset(interactions, ligands, tmp_path / "gto")
    other = tmp_path / "other" / "bindingdb_panel_subset.tsv"
    other.parent.mkdir(parents=True)
    rows = _subset_rows(gto.path)
    with other.open("w", encoding="utf-8", newline="") as handle:
        import csv as _csv

        writer = _csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(PANEL_SUBSET_COLUMNS)
        borrowed = dict(rows[0]) | {"curation": "ChEMBL", "affinity_nanomolar": "900.0"}
        writer.writerow([borrowed[column] for column in PANEL_SUBSET_COLUMNS])

    merged = merge_panel_subsets([gto.path, other], tmp_path / "merged")
    assert merged.blocker is None
    assert merged.path.name == "panel_subset.tsv"
    assert merged.rows == 2
    assert merged.ligands == 1, "one compound, two measurements"
    assert merged.curation == {"IUPHAR/BPS GtoPdb (ODbL)": 1, "ChEMBL": 1}


def test_merge_writes_a_repeated_row_once(tmp_path):
    interactions, ligands = _gtopdb_pair(tmp_path, [_gtopdb_row()])
    gto = gtopdb_panel_subset(interactions, ligands, tmp_path / "gto")
    merged = merge_panel_subsets([gto.path, gto.path], tmp_path / "merged")
    assert merged.rows == 1, "identical across all eleven columns is one measurement recorded twice"
    assert merged.skipped.get("duplicate_row") == 1


def test_merge_refuses_a_missing_or_foreign_file(tmp_path):
    assert merge_panel_subsets([], tmp_path / "m").blocker == "no_sources_named"
    assert "subset_not_found" in str(merge_panel_subsets([tmp_path / "nope.tsv"], tmp_path / "m").blocker)
    foreign = tmp_path / "foreign.tsv"
    foreign.write_text("a\tb\n1\t2\n", encoding="utf-8")
    assert "column_absent" in str(merge_panel_subsets([foreign], tmp_path / "m").blocker)


def test_a_merged_subset_is_read_by_both_ligand_safety_routes(tmp_path):
    """The whole reason for one format: GtoPdb's rows reach a reader."""
    from rfproteina.disease import ligand_safety as ls

    interactions, ligands = _gtopdb_pair(tmp_path, [_gtopdb_row()])
    gto = gtopdb_panel_subset(interactions, ligands, tmp_path / "gto")
    merged = merge_panel_subsets([gto.path], tmp_path / "merged")
    ls.reset_bindingdb_index()
    try:
        panel = ls.measured_nm_from_bindingdb(_GTOPDB_SMILES, merged.path)
        assert panel.axes["herg"]["value_nm"] == 40.0
        assert any("GtoPdb" in note for note in panel.notes)
        across = ls.nearest_measured_neighbours(_GTOPDB_SMILES, merged.path, min_similarity=0.0)
        assert across.neighbours["herg"], "a GtoPdb row is reachable as a neighbour"
        assert across.neighbours["herg"][0]["similarity"] == 1.0
    finally:
        ls.reset_bindingdb_index()


# ------------------------------------------------------------------- BindingDB
#
# Three properties of the real file decide whether a reader is right, and all
# three are exercised here against a synthetic file shaped like it: the UniProt
# accession is repeated per target chain and is often not in chain 1, the
# relation is written inside the value cell, and the licence is a per-row
# marking that a derived table has to keep.

_BDB_BASE = (
    "BindingDB Reactant_set_id",
    "Ligand SMILES",
    "Ligand InChI",
    "Ligand InChI Key",
    "Target Name",
    "Target Source Organism According to Curator or DataSource",
    "Ki (nM)",
    "IC50 (nM)",
    "Kd (nM)",
    "EC50 (nM)",
    "Curation/DataSource",
    "Article DOI",
)
#: Four chain blocks is enough to prove the scan is not pinned to chain 1.
_BDB_CHAINS = tuple(
    f"UniProt ({flavour}) Primary ID of Target Chain {n}" for n in (1, 2, 3) for flavour in ("SwissProt", "TrEMBL")
)
_BDB_HEADER = _BDB_BASE + _BDB_CHAINS
_HERG = "Q12809"


def _bdb_row(**overrides: str) -> dict[str, str]:
    row = dict.fromkeys(_BDB_HEADER, "")
    row.update(
        {
            "BindingDB Reactant_set_id": "1",
            "Ligand SMILES": "CC(=O)Oc1ccccc1C(=O)O",
            "Ligand InChI Key": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
            "Target Name": "Potassium voltage-gated channel subfamily H member 2",
            "Target Source Organism According to Curator or DataSource": "Homo sapiens",
            "IC50 (nM)": "250",
            "Curation/DataSource": "Curated from the literature by BindingDB",
            "Article DOI": "10.1000/example",
            "UniProt (SwissProt) Primary ID of Target Chain 1": _HERG,
        }
    )
    row.update(overrides)
    return row


def _bdb_tsv(path: Path, rows: list[dict[str, str]]) -> Path:
    import csv as _csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = _csv.DictWriter(
            handle, fieldnames=list(_BDB_HEADER), delimiter="\t", lineterminator="\n", quoting=_csv.QUOTE_NONE
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _subset_rows(path: Path) -> list[dict[str, str]]:
    import csv as _csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(_csv.DictReader(handle, delimiter="\t"))


def test_bindingdb_is_registered_and_closes_no_gap_by_itself():
    """Its path to an axis runs through a per-ligand lookup, so a download alone closes nothing."""
    source = next(item for item in SOURCES if item.name.startswith("BindingDB"))
    assert source.handoff == "none", "coverage() counts a convert source as a gap a download could close"
    assert source.member == "BindingDB_All.tsv"
    assert source.size == 594_284_548, "Content-Length, measured, not the download page's claim"
    assert "CC BY 3.0" in source.licence and "CC BY-SA 3.0" in source.licence
    assert source.hazard


def test_bindingdb_accessions_are_one_definition_shared_with_ligand_safety():
    from rfproteina.disease import ligand_safety

    assert BINDINGDB_PANEL_ACCESSIONS is ligand_safety.BINDINGDB_PANEL_ACCESSIONS
    from rfproteina.disease.combinations import LIGAND_AXES

    assert set(BINDINGDB_PANEL_ACCESSIONS) == set(LIGAND_AXES)


def test_bindingdb_subset_reads_a_panel_row(tmp_path):
    table = _bdb_tsv(tmp_path / "bdb.tsv", [_bdb_row()])
    result = bindingdb_panel_subset(table, tmp_path / "out")
    assert result.blocker is None
    assert result.rows == 1
    assert result.ligands == 1
    assert result.axes == ("herg",)
    assert result.scanned == 1
    (row,) = _subset_rows(result.path)
    assert row["panel_axis"] == "herg"
    assert row["accession"] == _HERG
    assert row["affinity_nanomolar"] == "250"
    assert row["relation"] == "="
    assert row["assay"] == "IC50"
    assert row["chains"] == "1"
    assert row["curation"] == "Curated from the literature by BindingDB"
    assert "BindingDB" in row["source"] and "10.1000/example" in row["source"]


def test_bindingdb_subset_finds_an_accession_outside_chain_one(tmp_path):
    """The trap. A quarter of a sampled 23,736 real rows keep theirs outside chain 1."""
    row = _bdb_row(**{"UniProt (SwissProt) Primary ID of Target Chain 1": ""})
    row["UniProt (SwissProt) Primary ID of Target Chain 3"] = _HERG
    result = bindingdb_panel_subset(_bdb_tsv(tmp_path / "bdb.tsv", [row]), tmp_path / "out")
    assert result.rows == 1, "an accession in chain 3 must be found, not skipped"
    assert _subset_rows(result.path)[0]["panel_axis"] == "herg"


def test_bindingdb_subset_finds_an_accession_in_a_trembl_column(tmp_path):
    row = _bdb_row(**{"UniProt (SwissProt) Primary ID of Target Chain 1": ""})
    row["UniProt (TrEMBL) Primary ID of Target Chain 2"] = _HERG
    assert bindingdb_panel_subset(_bdb_tsv(tmp_path / "bdb.tsv", [row]), tmp_path / "out").rows == 1


def test_bindingdb_subset_splits_the_relation_out_of_the_value(tmp_path):
    """BindingDB writes >10000 and even '> 10000'. float() raises on both."""
    rows = [
        _bdb_row(**{"IC50 (nM)": cell}) for cell in ("250", ">10000", "> 10000", "<0.5", ">=40", "not-a-number", "")
    ]
    result = bindingdb_panel_subset(_bdb_tsv(tmp_path / "bdb.tsv", rows), tmp_path / "out")
    written = _subset_rows(result.path)
    assert [(r["relation"], r["affinity_nanomolar"]) for r in written] == [
        ("=", "250"),
        (">", "10000"),
        (">", "10000"),
        ("<", "0.5"),
        (">=", "40"),
    ]
    assert result.skipped.get("value_unparsed") == 1


def test_bindingdb_subset_narrows_to_one_organism(tmp_path):
    rows = [
        _bdb_row(),
        _bdb_row(**{"Target Source Organism According to Curator or DataSource": "Rattus norvegicus"}),
    ]
    result = bindingdb_panel_subset(_bdb_tsv(tmp_path / "bdb.tsv", rows), tmp_path / "out")
    assert result.rows == 1
    assert result.skipped.get("other_organism") == 1


def test_bindingdb_subset_excludes_ec50(tmp_path):
    rows = [_bdb_row(**{"IC50 (nM)": "", "EC50 (nM)": "100"})]
    result = bindingdb_panel_subset(_bdb_tsv(tmp_path / "bdb.tsv", rows), tmp_path / "out")
    assert not result.written, "a half-maximal effect is not the quantity the panel cut is written against"


def test_bindingdb_subset_keeps_every_affinity_column_on_one_row(tmp_path):
    rows = [_bdb_row(**{"Ki (nM)": "10", "IC50 (nM)": "20", "Kd (nM)": "30"})]
    result = bindingdb_panel_subset(_bdb_tsv(tmp_path / "bdb.tsv", rows), tmp_path / "out")
    assert sorted(r["assay"] for r in _subset_rows(result.path)) == ["IC50", "Kd", "Ki"]
    assert result.ligands == 1, "three measurements of one compound is still one compound"


def test_bindingdb_subset_counts_the_chains_and_keeps_the_licence_marking(tmp_path):
    row = _bdb_row(**{"Curation/DataSource": "ChEMBL"})
    row["UniProt (SwissProt) Primary ID of Target Chain 2"] = "P00001"
    result = bindingdb_panel_subset(_bdb_tsv(tmp_path / "bdb.tsv", [row]), tmp_path / "out")
    (written,) = _subset_rows(result.path)
    assert written["chains"] == "2", "an affinity for a complex is not an affinity for one chain of it"
    assert written["curation"] == "ChEMBL"
    assert result.curation == {"ChEMBL": 1}, "the per-row licence marking has to survive into the summary"


def test_bindingdb_subset_reads_the_member_inside_a_zip(tmp_path):
    """567 MB on the wire unpacks to about 2 GB. Reading in place keeps it off the disk."""
    import zipfile as _zipfile

    table = _bdb_tsv(tmp_path / "bdb.tsv", [_bdb_row()])
    archive = tmp_path / "BindingDB_All_tsv.zip"
    with _zipfile.ZipFile(archive, "w", _zipfile.ZIP_DEFLATED) as zf:
        zf.write(table, arcname="BindingDB_All.tsv")
    result = bindingdb_panel_subset(archive, tmp_path / "out")
    assert result.rows == 1
    assert _subset_rows(result.path)[0]["accession"] == _HERG


def test_bindingdb_subset_writes_nothing_when_no_panel_target_appears(tmp_path):
    row = _bdb_row(**{"UniProt (SwissProt) Primary ID of Target Chain 1": "P99999"})
    result = bindingdb_panel_subset(_bdb_tsv(tmp_path / "bdb.tsv", [row]), tmp_path / "out")
    assert not result.written
    assert result.blocker == "panel_not_in_table"
    assert result.scanned == 1


def test_bindingdb_subset_refuses_bad_arguments_and_missing_columns(tmp_path):
    table = _bdb_tsv(tmp_path / "bdb.tsv", [_bdb_row()])
    assert bindingdb_panel_subset(tmp_path / "nope.tsv", tmp_path / "o").blocker == "table_not_found"
    assert bindingdb_panel_subset(table, tmp_path / "o", axes=["nope"]).blocker == "axes_not_in_panel:nope"
    assert bindingdb_panel_subset(table, tmp_path / "o", axes=[]).blocker == "axes_named_but_empty"
    assert bindingdb_panel_subset(table, tmp_path / "o", organism=" ").blocker == "organism_not_named"
    bare = tmp_path / "bare.tsv"
    bare.write_text("Ligand InChI Key\tCuration/DataSource\n", encoding="utf-8")
    assert bindingdb_panel_subset(bare, tmp_path / "o").blocker.startswith("column_absent:")


def test_bindingdb_subset_cannot_be_misread_as_the_screens_panel_table(tmp_path):
    """The guard, by construction rather than by filename.

    The GtoPdb reference set reads perfectly as a panel table and is kept apart
    only by what it is called. This one cannot be: no column it writes is one of
    the toxicology reader's axis, value or unit columns, so a copy renamed to
    toxicology.tsv is refused instead of being believed.
    """
    from rfproteina.disease.toxicology import measure_toxicology

    result = bindingdb_panel_subset(_bdb_tsv(tmp_path / "bdb.tsv", [_bdb_row()]), tmp_path / "out")
    assert result.path.name == "bindingdb_panel_subset.tsv"
    run = tmp_path / "run"
    run.mkdir()
    (run / "toxicology.tsv").write_text(result.path.read_text(encoding="utf-8"), encoding="utf-8")
    measured = measure_toxicology(run)
    assert all(record.get("measurement") is None for record in measured.values())
    # Parsed, and silent about the panel. A valid TSV whose columns the reader
    # does not recognise reports toxicology_not_reported -- a cleaner refusal
    # than the GtoPdb reference set's toxicology_unreadable, and the same
    # outcome where it counts: no axis gets a number.
    assert {record.get("blocker") for record in measured.values()} == {"toxicology_not_reported"}


# ------------------------------------------- GtoPdb, the secondary-pharmacology reduction
#
# The panel reduction narrows the same download to ten tox targets and keeps 69
# rows. That narrowing was the mistake: across every target the same filters
# keep 9,365 rows over 1,477 targets and 4,891 compounds, spanning ten family
# classes. These tests hold the wider reduction to its columns, its joins, and
# the one cell whose spelling decides whether a withdrawal is visible at all.

_GTOPDB_TARGETS_COLUMNS = ("Type", "Family id", "Family name", "Target id", "Target name", "HGNC symbol")


def _gtopdb_targets(path: Path, rows: list[tuple[str, str, str, str]] | None = None) -> Path:
    """targets_and_families.csv: the family each target id belongs to."""
    import csv as _csv

    entries = rows if rows is not None else [("vgic", "68", "Voltage-gated potassium channels", "572", "KCNH2")]
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(_GTOPDB_VERSION_LINE + "\n")
        writer = _csv.DictWriter(handle, fieldnames=list(_GTOPDB_TARGETS_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for kind, family_id, family, target_id, symbol in entries:
            writer.writerow(
                {
                    "Type": kind,
                    "Family id": family_id,
                    "Family name": family,
                    "Target id": target_id,
                    "Target name": symbol,
                    "HGNC symbol": symbol,
                }
            )
    return path


def _gtopdb_interactions_wide(path: Path, rows: list[dict[str, str]]) -> Path:
    """interactions.csv with the columns the wider reduction reads."""
    import csv as _csv

    columns = (
        *_GTOPDB_COLUMNS,
        "Target",
        "Target ID",
        "Selectivity",
        "Primary Target",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(_GTOPDB_VERSION_LINE + "\n")
        writer = _csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _wide_row(**overrides: str) -> dict[str, str]:
    row = _gtopdb_row()
    row.update(
        {
            "Target": "K<SUB>v</SUB>11.1",
            "Target ID": "572",
            "Selectivity": "Not Determined",
            "Primary Target": "false",
        }
    )
    row.update(overrides)
    return row


def _gtopdb_ligands_status(path: Path, approved: str = "", withdrawn: str = "") -> Path:
    import csv as _csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(_GTOPDB_VERSION_LINE + "\n")
        writer = _csv.DictWriter(
            handle,
            fieldnames=["Ligand ID", "Name", "SMILES", "InChIKey", "Approved", "Withdrawn"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "Ligand ID": "13578",
                "Name": "terfenadine-like",
                "SMILES": _GTOPDB_SMILES,
                "InChIKey": "PUBLISHED-KEY-N",
                "Approved": approved,
                "Withdrawn": withdrawn,
            }
        )
    return path


def _wide_trio(tmp_path, rows, *, approved="", withdrawn=""):
    return (
        _gtopdb_interactions_wide(tmp_path / "interactions.csv", rows),
        _gtopdb_ligands_status(tmp_path / "ligands.csv", approved, withdrawn),
        _gtopdb_targets(tmp_path / "targets.csv"),
    )


def test_offtarget_subset_writes_its_own_wider_columns(tmp_path):
    interactions, ligands, targets = _wide_trio(tmp_path, [_wide_row()])
    result = gtopdb_offtarget_subset(interactions, ligands, targets, tmp_path / "out")
    assert result.blocker is None
    assert result.rows == 1
    assert result.ligands == 1
    assert result.targets == 1
    header = result.path.read_text(encoding="utf-8").splitlines()[0].split("\t")
    assert header == list(GTOPDB_OFFTARGET_COLUMNS)
    (row,) = _subset_rows(result.path)
    assert row["target_name"] == "K<SUB>v</SUB>11.1"
    assert row["gene_symbol"] == "KCNH2", "joined from targets_and_families.csv"
    assert row["family_class"] == "vgic"
    assert row["family_name"] == "Voltage-gated potassium channels"
    assert row["affinity_nanomolar"] == "40.0"
    assert row["action"] == "Inhibitor", "what it does, which a bare affinity throws away"
    assert row["primary_target"] == "false"
    assert row["selectivity"] == "Not Determined"


def test_offtarget_subset_keeps_targets_the_tox_panel_has_never_heard_of(tmp_path):
    """The whole point. The panel reduction would drop every one of these."""
    rows = [
        _wide_row(**{"Target Gene Symbol": "HTR2B", "Target ID": "6", "Target": "5-HT2B"}),
        _wide_row(**{"Target Gene Symbol": "DRD2", "Target ID": "215", "Target": "D2"}),
    ]
    interactions, ligands, _targets = _wide_trio(tmp_path, rows)
    targets = _gtopdb_targets(
        tmp_path / "targets.csv",
        [
            ("gpcr", "1", "5-Hydroxytryptamine receptors", "6", "HTR2B"),
            ("gpcr", "20", "Dopamine receptors", "215", "DRD2"),
        ],
    )
    result = gtopdb_offtarget_subset(interactions, ligands, targets, tmp_path / "out")
    assert result.rows == 2
    assert result.families == {"gpcr": 2}
    assert {row["gene_symbol"] for row in _subset_rows(result.path)} == {"HTR2B", "DRD2"}


def test_offtarget_subset_reads_the_withdrawn_cell_as_gtopdb_writes_it(tmp_path):
    """``yes``, not ``true``. Tested the obvious way, every withdrawal vanishes."""
    interactions, ligands, targets = _wide_trio(tmp_path, [_wide_row()], approved="yes", withdrawn="yes")
    result = gtopdb_offtarget_subset(interactions, ligands, targets, tmp_path / "out")
    assert result.withdrawn_rows == 1
    assert result.approved_rows == 1
    (row,) = _subset_rows(result.path)
    assert row["withdrawn"] == "yes"
    assert row["approved"] == "yes"

    # And the failure this guards: "true" is not how GtoPdb spells it.
    _interactions, wrong, _targets = _wide_trio(tmp_path, [_wide_row()], approved="true", withdrawn="true")
    quiet = gtopdb_offtarget_subset(interactions, wrong, targets, tmp_path / "out2")
    assert quiet.withdrawn_rows == 0, "the cell holds 'yes'; anything else is not a withdrawal"


def test_offtarget_subset_refuses_what_it_cannot_read(tmp_path):
    interactions, ligands, targets = _wide_trio(tmp_path, [_wide_row()])
    absent = tmp_path / "nope.csv"
    assert gtopdb_offtarget_subset(absent, ligands, targets, tmp_path / "o").blocker == "interactions_not_found"
    assert gtopdb_offtarget_subset(interactions, absent, targets, tmp_path / "o").blocker == "ligands_not_found"
    assert gtopdb_offtarget_subset(interactions, ligands, absent, tmp_path / "o").blocker == "targets_not_found"
    assert (
        gtopdb_offtarget_subset(interactions, ligands, targets, tmp_path / "o", species=" ").blocker
        == "species_not_named"
    )


def test_offtarget_subset_counts_a_ligand_with_no_structure(tmp_path):
    interactions, ligands, targets = _wide_trio(tmp_path, [_wide_row(**{"Ligand ID": "99999"})])
    result = gtopdb_offtarget_subset(interactions, ligands, targets, tmp_path / "out")
    assert not result.written
    assert result.blocker == "no_usable_interactions"
    assert result.skipped.get("ligand_without_structure") == 1


def test_an_offtarget_subset_cannot_be_misread_as_the_panel_table(tmp_path):
    from rfproteina.disease.toxicology import measure_toxicology

    interactions, ligands, targets = _wide_trio(tmp_path, [_wide_row()])
    result = gtopdb_offtarget_subset(interactions, ligands, targets, tmp_path / "out")
    run = tmp_path / "run"
    run.mkdir()
    (run / "toxicology.tsv").write_text(result.path.read_text(encoding="utf-8"), encoding="utf-8")
    assert all(record.get("measurement") is None for record in measure_toxicology(run).values())


# ------------------------------------- the GtoPdb release, as the three files it is


def test_every_gtopdb_file_the_reductions_need_is_registered():
    """The defect this closes: two of the three were reachable only by hand.

    ``gtopdb_panel_subset`` takes interactions and ligands, and
    ``gtopdb_offtarget_subset`` takes a third. Only ``interactions.csv`` was in
    the registry, so ``scripts/fetch_screening_tables.py`` could not fetch the
    other two and neither reduction could run on a machine stocked by it --
    the functions shipped, and nothing in the repository could feed them.
    """
    assert set(GTOPDB_FILES) <= {source.file for source in SOURCES}


def test_the_gtopdb_companions_are_odbl_and_feed_no_axis_on_their_own():
    for name in ("ligands.csv", "targets_and_families.csv"):
        source = next(item for item in SOURCES if item.file == name)
        assert "ODbL" in source.licence, f"{name} must carry the share-alike obligation"
        assert source.handoff == "none"
        assert source.stem is None, "a stem reads as 'rename the download to this', and these are inputs"
        assert source.size, "the size is pinned so a new quarterly release is noticed"


def test_interactions_stays_the_first_guide_to_pharmacology_source():
    """Pins an ordering another test in this file depends on.

    ``test_gtopdb_is_registered_and_closes_no_gap_by_itself`` takes the *first*
    source whose name mentions the Guide to PHARMACOLOGY and asserts a stem and
    a hazard. Three sources match that name now and only ``interactions.csv``
    carries either, so the order is load-bearing rather than cosmetic.
    """
    guide = [item for item in SOURCES if "Guide to PHARMACOLOGY" in item.name]
    assert len(guide) == 3
    assert guide[0].file == "interactions.csv"


def test_the_release_lookup_answers_only_once_every_file_is_there(tmp_path):
    """Partial is the same as absent: two files out of three run no reduction."""
    assert gtopdb_release(tmp_path) is None
    assert missing_gtopdb_files(tmp_path) == GTOPDB_FILES

    _gtopdb_pair(tmp_path, [_gtopdb_row()])
    assert gtopdb_release(tmp_path) is None
    assert missing_gtopdb_files(tmp_path) == ("targets_and_families.csv",)

    _gtopdb_targets(tmp_path / "targets_and_families.csv")
    release = gtopdb_release(tmp_path)
    assert release is not None
    assert missing_gtopdb_files(tmp_path) == ()
    assert release.as_arguments() == (
        tmp_path / "interactions.csv",
        tmp_path / "ligands.csv",
        tmp_path / "targets_and_families.csv",
    )


def test_the_release_lookup_takes_the_files_in_the_order_the_reduction_wants(tmp_path):
    """``as_arguments`` is positional, and the wrong order is a silent misread:
    every one of the three parses as CSV, so nothing would raise."""
    _gtopdb_interactions_wide(tmp_path / "interactions.csv", [_wide_row()])
    _gtopdb_ligands_status(tmp_path / "ligands.csv")
    _gtopdb_targets(tmp_path / "targets_and_families.csv")
    release = gtopdb_release(tmp_path)
    result = gtopdb_offtarget_subset(*release.as_arguments(), tmp_path / "out")
    assert result.blocker is None, "the three files reached the reduction in the order it takes them"
    assert result.rows == 1


def test_an_empty_gtopdb_file_counts_as_absent(tmp_path):
    """What a transfer that failed without raising leaves behind. Handed one, a
    reduction reports a parse error rather than the plain truth."""
    _gtopdb_pair(tmp_path, [_gtopdb_row()])
    (tmp_path / "targets_and_families.csv").write_text("", encoding="utf-8")
    assert missing_gtopdb_files(tmp_path) == ("targets_and_families.csv",)
    assert gtopdb_release(tmp_path) is None


def test_the_public_bindingdb_subset_name_is_the_name_that_gets_written(tmp_path):
    """Spelled once, for the reason ``downloaded_name`` records: a name derived
    separately by the writer and by the reader is one they can come to disagree
    about. The merge step looks the subset up by this constant."""
    result = bindingdb_panel_subset(_bdb_tsv(tmp_path / "bdb.tsv", [_bdb_row()]), tmp_path / "out")
    assert result.blocker is None
    assert result.path.name == BINDINGDB_PANEL_SUBSET_FILE


def test_a_subset_written_before_the_column_was_renamed_is_refused_not_coerced(tmp_path):
    """The state found on the installed machine, pinned.

    A ``bindingdb_panel_subset.tsv`` written by an older build carries
    ``inchikey_bindingdb`` where the contract now says ``inchikey_upstream``.
    The merge has to refuse it and name the column: the alternative is a
    silently short merged table, or a rewrite of a derived file whose other
    drift nobody has checked.
    """
    import csv as _csv

    stale_columns = ["inchikey_bindingdb" if name == "inchikey_upstream" else name for name in PANEL_SUBSET_COLUMNS]
    stale = tmp_path / BINDINGDB_PANEL_SUBSET_FILE
    with stale.open("w", encoding="utf-8", newline="") as handle:
        writer = _csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(stale_columns)
        writer.writerow(["KEY-N", "KEY-N", "CCO", "herg", "Q12809", "40.0", "=", "IC50", "1", "x", "y"])

    interactions, ligands = _gtopdb_pair(tmp_path, [_gtopdb_row()])
    fresh = gtopdb_panel_subset(interactions, ligands, tmp_path / "out")
    assert fresh.blocker is None

    merged = merge_panel_subsets([stale, fresh.path], tmp_path / "merged")
    assert merged.blocker == "column_absent:inchikey_upstream"
    assert not (tmp_path / "merged").exists(), "a refused merge writes nothing"
