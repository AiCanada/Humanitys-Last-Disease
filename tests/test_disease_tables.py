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
    QUERIES,
    REFUSED,
    SIZE_LIMIT,
    SOURCES,
    Source,
    TooLargeError,
    by_name,
    coverage,
    declared_size,
    fetch,
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
                assert source.handoff != "convert", (
                    f"{source.name} has no stem to convert into, so it cannot be staged"
                )
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
