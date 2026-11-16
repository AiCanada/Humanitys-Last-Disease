"""Filing a downloaded AlphaFold Server archive against the designs it folded.

The archive's shape is pinned here from real server bundles:
``fold_<job>_model_<N>.cif``, ``fold_<job>_summary_confidences_<N>.json`` and
``fold_<job>_full_data_<N>.json`` for N in 0..4, plus ``fold_<job>_job_request``
and a lowercase ``terms_of_use.md``. The job name is lowercased inside those
filenames.

The defect these tests exist for is not a crash. The server's summary has ten
keys and ``chain_ids`` is not one of them, and the code that consumed it read a
missing ``chain_ids`` as an empty list. ``extract_confidence_metrics`` then
iterates ``combinations(range(0))`` and emits no chain-pair metric, and
``chain_of_interest not in []`` returns before every per-chain and interface
value. So a two-chain binder refold produced ``ptm`` and ``iptm`` and then
silently produced no ``ptm_coi``, no ``iptm_coi``, no ``min_ipae_coi`` and no
``min_ipae_A_B`` -- the four numbers a binder refold is run to get. Nothing
raised, so the only symptom was empty columns.

Each test below names what it would miss if the fix were removed.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from rfproteina.af3server import bundle as bundle_module
from rfproteina.af3server.bundle import (
    CONFIDENCE_LIMIT,
    MEMBER_LIMIT,
    FoldResult,
    import_archive,
    read_archive,
)
from rfproteina.metrics.oracles.imported_refolds import IMPORT_MARK, PathOracle

#: An _atom_site-only CIF, which is what the server actually returns: its model
#: files carry no _entry, no _entity and no _entity_poly at all.
SERVER_CIF = (
    "data_fold_binder_0_model_0\n"
    "#\n"
    "loop_\n"
    "_atom_site.group_PDB\n"
    "_atom_site.id\n"
    "_atom_site.type_symbol\n"
    "_atom_site.label_atom_id\n"
    "_atom_site.label_comp_id\n"
    "_atom_site.label_asym_id\n"
    "_atom_site.label_entity_id\n"
    "_atom_site.B_iso_or_equiv\n"
    "ATOM 1 N N ILE A ? 76.63\n"
    "ATOM 2 C CA ILE A ? 78.12\n"
    "#\n"
)


@pytest.fixture
def oracle() -> PathOracle:
    """The production path shim, not a hand-rolled copy of it.

    There was a ``PathOracle`` here restating the four canonical path members.
    ``PathOracle`` is the same four, is what ``rfd4-af3server retrieve`` itself
    passes, and is kept in step with ``OracleMetric`` by a test that calls the
    real class's unbound methods against one. Testing against a private copy
    meant these tests could agree with themselves while the command disagreed
    with the oracle -- and it is exactly where this module's two worst bugs hid.
    """
    return PathOracle(oracle_name="af3")


def summary(*, chains: int = 2, rank: float = 0.66, chain_ids: list[str] | None = None) -> dict:
    """A server summary: the real ten keys, in the real types.

    ``has_clash`` and ``num_recycles`` are floats, not a bool and an int, which
    is what the real files hold despite the documentation calling the first a
    boolean. ``chain_ids`` is absent unless a test asks for it, because the
    server never sends it.
    """
    body: dict[str, object] = {
        "chain_iptm": [0.51] * chains,
        "chain_pair_iptm": [[0.68 if i == j else 0.51 for j in range(chains)] for i in range(chains)],
        "chain_pair_pae_min": [[0.76 if i == j else 2.21 for j in range(chains)] for i in range(chains)],
        "chain_ptm": [0.68] * chains,
        "fraction_disordered": 0.25,
        "has_clash": 0.0,
        "iptm": 0.51 if chains > 1 else None,
        "num_recycles": 10.0,
        "ptm": 0.61,
        "ranking_score": rank,
    }
    if chain_ids is not None:
        body["chain_ids"] = chain_ids
    return body


def full_data(chains: tuple[str, ...] = ("A", "B"), per_chain: int = 3) -> dict:
    """A server full-data JSON: the real six keys, with a usable chain order."""
    tokens = [chain for chain in chains for _ in range(per_chain)]
    size = len(tokens)
    return {
        "atom_chain_ids": tokens,
        "atom_plddts": [76.63] * size,
        "contact_probs": [[0.0] * size for _ in range(size)],
        "pae": [[1.0] * size for _ in range(size)],
        "token_chain_ids": tokens,
        "token_res_ids": list(range(1, size + 1)),
    }


def archive(
    path: Path,
    job: str = "binder_0",
    *,
    models: int = 5,
    with_full: bool = True,
    chains: tuple[str, ...] = ("A", "B"),
    extra: dict[str, bytes] | None = None,
) -> Path:
    """A bundle shaped like a real download, best-ranked model first."""
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(f"fold_{job}_job_request.json", json.dumps([{"name": job, "modelSeeds": ["1437240881"]}]))
        bundle.writestr("terms_of_use.md", "non-commercial use only\n")
        for index in range(models):
            bundle.writestr(f"fold_{job}_model_{index}.cif", SERVER_CIF)
            bundle.writestr(
                f"fold_{job}_summary_confidences_{index}.json",
                json.dumps(summary(chains=len(chains), rank=0.66 - 0.01 * index)),
            )
            if with_full:
                bundle.writestr(f"fold_{job}_full_data_{index}.json", json.dumps(full_data(chains)))
        for name, payload in (extra or {}).items():
            bundle.writestr(name, payload)
    return path


#: A two-chain design whose binder is strand ``A`` and target strand ``E``.
#: The strand names matter: the server keeps none of them and labels chains by
#: position, so this design's ``E`` comes back as chain ``B``.
DESIGN_CIF = (
    "data_design\n"
    "#\n"
    "loop_\n"
    "_entity.id \n"
    "_entity.type \n"
    "0 polymer\n"
    "1 polymer\n"
    "#\n"
    "loop_\n"
    "_entity_poly.entity_id \n"
    "_entity_poly.type \n"
    "_entity_poly.pdbx_seq_one_letter_code_can \n"
    "_entity_poly.pdbx_strand_id \n"
    "0 polypeptide(L) SILDSLLEKYQFPEDLTLYIKSVFPATD A\n"
    "1 polypeptide(L) SPITEYLASLSTYNDQSITFALEDESYEI E\n"
    "#\n"
)


def design(run: Path, example_id: str = "binder_0", *, text: str = DESIGN_CIF) -> Path:
    """One design where a run actually puts one.

    Under ``inference_samples/``, not directly under the run. That is the whole
    point of the fixture: a refold belongs beside its design, and the earlier
    version of this helper put the design directly under the run so that the
    import's own wrong base matched it. Both were wrong together and the tests
    passed.
    """
    directory = run / "inference_samples" / example_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{example_id}_generated.cif").write_text(text, encoding="utf-8")
    return directory


def base(run: Path) -> Path:
    """Where a design's own files live, which is what the oracle's paths join to."""
    return run / "inference_samples"


class TestReadArchive:
    def test_every_model_is_found_and_ranked_best_first(self, tmp_path: Path) -> None:
        """Five models per job, ordered by the server's own ranking_score."""
        read = read_archive(archive(tmp_path / "fold.zip"))
        assert not isinstance(read, str), read
        assert len(read.results) == 5
        models = read.jobs()["binder_0"]
        assert [item.model for item in models] == [0, 1, 2, 3, 4]
        assert [item.ranking_score for item in models] == sorted((item.ranking_score for item in models), reverse=True)

    def test_the_best_model_is_the_highest_score_not_the_lowest_index(self, tmp_path: Path) -> None:
        """Index 0 is best in every real bundle, but the score is what means best.

        Built here with the order inverted, so a reader that trusted the index
        would pick the worst of the five.
        """
        path = tmp_path / "fold.zip"
        with zipfile.ZipFile(path, "w") as bundle:
            for index, rank in enumerate((0.40, 0.95)):
                bundle.writestr(f"fold_j_model_{index}.cif", SERVER_CIF)
                bundle.writestr(f"fold_j_summary_confidences_{index}.json", json.dumps(summary(rank=rank)))
        read = read_archive(path)
        assert not isinstance(read, str)
        assert read.jobs()["j"][0].model == 1

    def test_the_jobs_own_extra_files_are_not_reported_as_problems(self, tmp_path: Path) -> None:
        """terms_of_use.md and the job request are expected members."""
        read = read_archive(archive(tmp_path / "fold.zip"))
        assert not isinstance(read, str)
        assert read.skipped == []

    def test_a_model_without_its_summary_is_refused_not_filed(self, tmp_path: Path) -> None:
        """A structure with no confidence numbers would read as refolded and measure nothing."""
        path = tmp_path / "fold.zip"
        with zipfile.ZipFile(path, "w") as bundle:
            bundle.writestr("fold_j_model_0.cif", SERVER_CIF)
        read = read_archive(path)
        assert not isinstance(read, str)
        assert read.results == []
        assert read.skipped == [("fold_j_model_0.cif", "summary_confidences_absent")]

    def test_a_member_that_is_not_json_is_named(self, tmp_path: Path) -> None:
        path = tmp_path / "fold.zip"
        with zipfile.ZipFile(path, "w") as bundle:
            bundle.writestr("fold_j_model_0.cif", SERVER_CIF)
            bundle.writestr("fold_j_summary_confidences_0.json", "{not json")
        read = read_archive(path)
        assert not isinstance(read, str)
        assert ("fold_j_summary_confidences_0.json", "member_not_json") in read.skipped

    @pytest.mark.parametrize("member", ["../escape.cif", "/abs/fold_j_model_0.cif", "a/../../fold_j_model_0.cif"])
    def test_a_member_naming_a_path_outside_the_archive_is_refused(self, tmp_path: Path, member: str) -> None:
        """Nothing here extracts to an archive-supplied path, and it still says so."""
        path = tmp_path / "fold.zip"
        with zipfile.ZipFile(path, "w") as bundle:
            bundle.writestr(member, SERVER_CIF)
        read = read_archive(path)
        assert not isinstance(read, str)
        assert read.results == []
        assert read.skipped and read.skipped[0][1] == "member_path_unsafe"

    def test_an_archive_that_is_not_a_zip_is_a_named_blocker(self, tmp_path: Path) -> None:
        path = tmp_path / "fold.zip"
        path.write_text("not a zip", encoding="utf-8")
        assert read_archive(path) == "archive_unreadable"

    def test_a_missing_archive_is_a_named_blocker(self, tmp_path: Path) -> None:
        assert read_archive(tmp_path / "absent.zip") == "archive_not_found"

    def test_an_archive_with_too_many_members_is_refused_whole(self, tmp_path: Path, monkeypatch) -> None:
        """The member bound is on the count, before anything is read.

        The real bound is 8192. It is lowered here rather than writing 8193
        members, because the thing under test is the comparison, not the loop.
        """
        monkeypatch.setattr(bundle_module, "MEMBER_LIMIT", 3)
        path = tmp_path / "fold.zip"
        with zipfile.ZipFile(path, "w") as bundle:
            for index in range(4):
                bundle.writestr(f"pad_{index}.txt", "x")
        assert read_archive(path) == "archive_too_many_members"

    def test_an_archive_that_expands_past_the_bound_is_refused_whole(self, tmp_path: Path, monkeypatch) -> None:
        """Refused on the declared expanded size, before any member is decompressed."""
        monkeypatch.setattr(bundle_module, "EXPANDED_LIMIT", 16)
        path = tmp_path / "fold.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("fold_j_model_0.cif", SERVER_CIF)
        assert read_archive(path) == "archive_too_large"

    def test_an_oversized_confidence_member_is_skipped_by_name(self, tmp_path: Path, monkeypatch) -> None:
        """A declared size past the bound is refused without being decompressed.

        Both bounds are lowered: the per-member one is what is being tested, and
        the archive-wide one would otherwise refuse the whole file first.
        """
        monkeypatch.setattr(bundle_module, "CONFIDENCE_LIMIT", 8)
        path = tmp_path / "fold.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("fold_j_model_0.cif", SERVER_CIF)
            bundle.writestr("fold_j_summary_confidences_0.json", json.dumps(summary()))
        read = read_archive(path)
        assert not isinstance(read, str)
        assert ("fold_j_summary_confidences_0.json", "member_too_large") in read.skipped
        # And the structure beside it is then not filed either, for the same
        # reason a model with no summary is not: there would be no numbers.
        assert read.results == []

    def test_the_real_bounds_are_large_enough_for_a_real_bundle(self) -> None:
        """The lowered bounds above must not be mistaken for the shipped ones."""
        assert MEMBER_LIMIT >= 100 * 16, "a 100-job upload returns about 16 files per job"
        assert CONFIDENCE_LIMIT >= 1024**3, "a 5,000-token PAE matrix is tens of millions of numbers"


class TestChainOrder:
    def test_the_chain_order_is_derived_when_the_server_omits_chain_ids(self, tmp_path: Path) -> None:
        """The defect. A server summary has no chain_ids; the order comes from token_chain_ids."""
        read = read_archive(archive(tmp_path / "fold.zip", chains=("A", "B")))
        assert not isinstance(read, str)
        best = read.jobs()["binder_0"][0]
        assert "chain_ids" not in best.summary, "the fixture stopped resembling a real server summary"
        assert best.chain_order() == ["A", "B"]

    def test_a_declared_chain_ids_still_wins(self, tmp_path: Path) -> None:
        """A local AF3 summary that does name its order is not overridden."""
        result = FoldResult("j", 0, b"", summary(chain_ids=["X", "Y"]), full_data(("A", "B")))
        assert result.chain_order() == ["X", "Y"]

    def test_the_order_is_first_appearance_not_sorted(self, tmp_path: Path) -> None:
        """It indexes the chain_pair matrices, so the file's order is the only right one."""
        result = FoldResult("j", 0, b"", summary(), full_data(("B", "A")))
        assert result.chain_order() == ["B", "A"]

    def test_no_full_data_leaves_no_chain_order_rather_than_a_guess(self) -> None:
        """A wrong order is worse than none: it attributes a number to another chain."""
        assert FoldResult("j", 0, b"", summary(), None).chain_order() == []

    def test_the_interface_metrics_exist_because_the_order_does(self, tmp_path: Path) -> None:
        """What the empty order silently removed, named one key at a time.

        With ``chains_in_order`` empty every one of these is absent while
        ``overall_ptm`` and ``overall_iptm`` stay present -- which is exactly
        what made the defect hard to see. A refold looked like it had worked.
        """
        read = read_archive(archive(tmp_path / "fold.zip", chains=("A", "B")))
        assert not isinstance(read, str)
        metrics = read.jobs()["binder_0"][0].metrics("A")
        for key in ("ptm_coi", "iptm_coi", "min_ipae_coi", "min_ipae_A_B", "iptm_A_B"):
            assert key in metrics, f"{key} is missing, which is the whole defect"
        assert metrics["ptm_coi"] == pytest.approx(68.0)
        assert metrics["min_ipae_coi"] == pytest.approx(2.21)
        assert metrics["overall_ptm"] == pytest.approx(61.0)

    def test_the_metrics_the_server_cannot_supply_are_absent_rather_than_zero(self, tmp_path: Path) -> None:
        """A real gap, pinned so nobody later reads a missing bound as a failed one.

        The server's summary has exactly ten keys. Three of the flat metrics the
        pipeline knows about are derived from keys it does not send:

        * ``plddt_coi`` needs ``chain_plddt``
        * ``pae_coi`` needs ``chain_pae``
        * ``ipae_coi`` needs ``chain_pair_pae`` (it sends only ``chain_pair_pae_min``)

        So they are absent from a server refold, and a pass criterion bounding
        one of them reports an unmet bound rather than a failed one. Per-atom
        pLDDT is still there -- it is in the model CIF's B-factor column, which
        is where the pipeline reads it from on the local path too.
        """
        read = read_archive(archive(tmp_path / "fold.zip", chains=("A", "B")))
        assert not isinstance(read, str)
        best = read.jobs()["binder_0"][0]
        metrics = best.metrics("A")
        for key, needs in (("plddt_coi", "chain_plddt"), ("pae_coi", "chain_pae"), ("ipae_coi", "chain_pair_pae")):
            assert needs not in best.summary, f"the fixture now sends {needs}; the real server does not"
            assert key not in metrics, f"{key} cannot be derived from a server summary"
        assert b"B_iso_or_equiv" in best.structure, "per-atom pLDDT must still arrive in the CIF"

    def test_without_a_chain_order_only_the_global_metrics_survive(self) -> None:
        """Pins the old behaviour as the thing being fixed, so it cannot return quietly."""
        metrics = FoldResult("j", 0, b"", summary(), None).metrics("A")
        assert "overall_ptm" in metrics and "overall_iptm" in metrics
        for key in ("ptm_coi", "iptm_coi", "min_ipae_coi", "min_ipae_A_B"):
            assert key not in metrics

    def test_a_null_iptm_from_a_single_chain_fold_does_not_break_the_read(self) -> None:
        """The writer substitutes null for NaN, and iptm is NaN for one chain."""
        result = FoldResult("j", 0, b"", summary(chains=1), full_data(("A",)))
        metrics = result.metrics("A")
        assert "overall_iptm" not in metrics, "a null must be skipped, not read as 0.0"
        assert metrics["overall_ptm"] == pytest.approx(61.0)
        assert metrics["ptm_coi"] == pytest.approx(68.0)


class TestWhereTheRefoldLands:
    """The refold goes beside its design, which is not a child of the run.

    A run nests designs under ``inference_samples/``, and one holding several
    batches nests them a level further. Handing the run itself to
    ``place_refold`` wrote the pair into ``<run>/<id>/`` -- a fresh directory no
    design lives in and no oracle looks in -- and reported the design as filed.
    ``already_refolded`` stayed false where it matters, so every refolding
    metric was silently absent while the import claimed success.
    """

    def test_the_pair_lands_beside_the_design_not_under_the_run(self, tmp_path: Path, oracle) -> None:
        out = tmp_path / "run"
        design(out)
        import_archive(oracle, out, archive(tmp_path / "fold.zip"))
        assert oracle.confidence_path(base(out), "binder_0").is_file()
        # The directory the old behaviour would have invented is not created.
        assert not (out / "binder_0").exists()

    def test_the_oracle_finds_it_where_it_actually_looks(self, tmp_path: Path, oracle) -> None:
        """The whole contract: `already_refolded` at the design's own base."""
        out = tmp_path / "run"
        design(out)
        import_archive(oracle, out, archive(tmp_path / "fold.zip"))
        assert oracle.already_refolded(base(out), "binder_0")
        assert not oracle.already_refolded(out, "binder_0"), "filed at the run root, which nothing reads"

    def test_a_design_nested_one_level_deeper_still_lands_right(self, tmp_path: Path, oracle) -> None:
        """A run of several batches nests designs under their batch."""
        out = tmp_path / "run"
        directory = out / "inference_samples" / "run_a" / "binder_7"
        directory.mkdir(parents=True)
        (directory / "binder_7_generated.cif").write_text(DESIGN_CIF, encoding="utf-8")
        report = import_archive(oracle, out, archive(tmp_path / "fold.zip", job="binder_7"))
        assert len(report.placed) == 1, report
        assert oracle.already_refolded(out / "inference_samples" / "run_a", "binder_7")

    def test_a_design_whose_id_cannot_be_joined_back_is_named(self, tmp_path: Path, oracle) -> None:
        """A repeated binder token flattens to a path id with no base to join.

        ``_sample_ids`` calls a design at ``inference_samples/run_a/binder_0/``
        ``run_a_binder_0`` when the token repeats across batches, and no base
        joins that id back onto that folder. Misfiling it would put the refold
        in a directory holding no design, where ``already_refolded`` would then
        read the same wrong place and agree with the import that it worked.
        """
        out = tmp_path / "run"
        for batch in ("run_a", "run_b"):
            directory = out / "inference_samples" / batch / "binder_0"
            directory.mkdir(parents=True)
            (directory / "binder_0_generated.cif").write_text(DESIGN_CIF, encoding="utf-8")
        report = import_archive(oracle, out, archive(tmp_path / "fold.zip", job="run_a_binder_0"))
        assert report.placed == []
        assert [item.blocker for item in report.refused] == ["design_base_not_resolved"]


class TestChainOfInterestIsTranslated:
    """The server keeps none of the design's chain names.

    It labels chains by position in ``sequences``, in AlphaFold 3's reverse
    spreadsheet order. A design whose target is strand ``E`` comes back as chain
    ``B``, so taking the interface metrics under ``E`` finds nothing -- and finds
    it silently, because a chain order naming no chain of interest just returns
    before every per-chain value.
    """

    def test_the_designs_own_strand_is_translated_to_the_servers_label(self, tmp_path: Path, oracle) -> None:
        out = tmp_path / "run"
        design(out)
        import_archive(oracle, out, archive(tmp_path / "fold.zip"), chain_of_interest="E")
        body = json.loads(oracle.confidence_path(base(out), "binder_0").read_text(encoding="utf-8"))
        # Strand E is the second entity, so the server called it B -- and the
        # per-chain metrics are taken on B's values, not dropped.
        assert body["chain_of_interest"] == "B"
        assert body["ptm_coi"] == pytest.approx(68.0)
        assert body["min_ipae_coi"] == pytest.approx(2.21)

    def test_the_first_strand_translates_to_itself(self, tmp_path: Path, oracle) -> None:
        """Strand A is the first entity, so it is chain A either way."""
        out = tmp_path / "run"
        design(out)
        import_archive(oracle, out, archive(tmp_path / "fold.zip"), chain_of_interest="A")
        body = json.loads(oracle.confidence_path(base(out), "binder_0").read_text(encoding="utf-8"))
        assert body["chain_of_interest"] == "A"
        assert body["ptm_coi"] == pytest.approx(68.0)

    def test_an_untranslatable_name_is_passed_through_rather_than_dropped(self, tmp_path: Path, oracle) -> None:
        """A caller who already knows the server's label is no worse off."""
        out = tmp_path / "run"
        design(out, text="data_x\n#\n")  # no entity table, so no request can be rebuilt
        import_archive(oracle, out, archive(tmp_path / "fold.zip"), chain_of_interest="B")
        body = json.loads(oracle.confidence_path(base(out), "binder_0").read_text(encoding="utf-8"))
        assert body["chain_of_interest"] == "B"


class TestImportArchive:
    def test_a_filed_fold_makes_the_oracle_call_the_design_refolded(self, tmp_path: Path, oracle) -> None:
        """The contract the whole round trip rests on."""
        out = tmp_path / "run"
        design(out)
        report = import_archive(oracle, out, archive(tmp_path / "fold.zip"), chain_of_interest="A")
        assert report.refused == [] and report.unmatched == []
        assert len(report.placed) == 1
        assert oracle.already_refolded(base(out), "binder_0")

    def test_only_the_best_model_is_filed_by_default(self, tmp_path: Path, oracle) -> None:
        """Five models under one design id would overwrite each other."""
        out = tmp_path / "run"
        design(out)
        report = import_archive(oracle, out, archive(tmp_path / "fold.zip"), chain_of_interest="A")
        assert len(report.placed) == 1
        body = json.loads(oracle.confidence_path(base(out), "binder_0").read_text(encoding="utf-8"))
        assert body["af3server_model"] == 0
        assert body["af3server_ranking_score"] == pytest.approx(0.66)

    def test_every_model_can_be_filed_under_its_own_save_id(self, tmp_path: Path, oracle) -> None:
        out = tmp_path / "run"
        design(out)
        report = import_archive(oracle, out, archive(tmp_path / "fold.zip"), chain_of_interest="A", every_model=True)
        assert len(report.placed) == 5
        for index in range(5):
            assert oracle.already_refolded(base(out), "binder_0", f"model_{index}")

    def test_the_filed_confidence_holds_the_flat_metric_keys(self, tmp_path: Path, oracle) -> None:
        """Not the raw server summary: the pipeline reads this file as its metrics."""
        out = tmp_path / "run"
        design(out)
        import_archive(oracle, out, archive(tmp_path / "fold.zip"), chain_of_interest="A")
        body = json.loads(oracle.confidence_path(base(out), "binder_0").read_text(encoding="utf-8"))
        assert body["ptm_coi"] == pytest.approx(68.0)
        assert body["min_ipae_coi"] == pytest.approx(2.21)
        assert body["chain_of_interest"] == "A"
        assert body[IMPORT_MARK] is True
        assert body["oracle_name"] == "af3"

    def test_an_absent_chain_order_is_recorded_rather_than_left_as_empty_columns(self, tmp_path: Path, oracle) -> None:
        """A reader must be able to tell why the interface numbers are missing."""
        out = tmp_path / "run"
        design(out)
        import_archive(
            oracle,
            out,
            archive(tmp_path / "fold.zip", with_full=False),
            chain_of_interest="A",
        )
        body = json.loads(oracle.confidence_path(base(out), "binder_0").read_text(encoding="utf-8"))
        assert body["af3server_chain_order"] == "not_recorded"
        assert "ptm_coi" not in body

    def test_a_lowercased_server_job_name_still_matches_its_design(self, tmp_path: Path, oracle) -> None:
        """The server lowercases the name, so M0664_2dhn comes back as fold_m0664_2dhn_*."""
        out = tmp_path / "run"
        design(out, "M0664_2dhn")
        report = import_archive(oracle, out, archive(tmp_path / "fold.zip", job="m0664_2dhn"), chain_of_interest="A")
        assert len(report.placed) == 1 and report.unmatched == []
        assert oracle.already_refolded(base(out), "M0664_2dhn")

    def test_a_job_matching_no_design_is_named_not_skipped(self, tmp_path: Path, oracle) -> None:
        """An import that silently did nothing is the failure this most likely has."""
        out = tmp_path / "run"
        design(out, "binder_0")
        report = import_archive(oracle, out, archive(tmp_path / "fold.zip", job="someone_elses_job"))
        assert report.placed == []
        assert report.unmatched == ["fold_someone_elses_job"]

    def test_two_designs_the_server_would_spell_alike_are_refused_not_guessed(self, tmp_path: Path, oracle) -> None:
        """binder-0 and binder_0 both flatten to binder_0. Filing either is a coin toss."""
        out = tmp_path / "run"
        design(out, "binder-0")
        design(out, "binder_0")
        report = import_archive(oracle, out, archive(tmp_path / "fold.zip", job="binder_0"))
        assert report.placed == []
        assert [item.blocker for item in report.refused] == ["design_id_ambiguous"]

    def test_a_design_id_that_could_escape_the_workspace_is_refused(self, tmp_path: Path, oracle) -> None:
        """``--designs`` is not validated on the way in, and the id reaches a path.

        :py:func:`_place_one` interpolates it into its two scratch filenames, so
        an id carrying a separator resolves them outside the
        ``TemporaryDirectory`` that owns them -- written, and then left behind
        when it is cleaned. ``_normalised`` strips exactly the characters that
        would do it, so ``../../evil`` matches a job the server called ``evil``
        and the name match does not turn it away.

        What stops it is the base lookup: a base is read by the design's exact
        id, and :py:func:`design_bases` only ever holds ids that name a real
        directory. Pinned because the protection sits one gate earlier than the
        path it protects, and a refactor that moved or dropped that gate would
        leave no other sign that the scratch write had become reachable.
        """
        out = tmp_path / "run"
        design(out, "evil")
        report = import_archive(
            oracle,
            out,
            archive(tmp_path / "fold.zip", job="evil"),
            example_ids=["../../evil"],
            chain_of_interest="A",
        )
        assert report.placed == []
        assert [item.blocker for item in report.refused] == ["design_base_not_resolved"]
        # And nothing was filed beside the real design under its own name either.
        assert not oracle.already_refolded(out, "evil")

    def test_only_the_named_designs_are_considered(self, tmp_path: Path, oracle) -> None:
        out = tmp_path / "run"
        design(out, "binder_0")
        design(out, "binder_1")
        report = import_archive(oracle, out, archive(tmp_path / "fold.zip", job="binder_1"), example_ids=["binder_0"])
        assert report.placed == [] and report.unmatched == ["fold_binder_1"]

    def test_a_bad_archive_is_reported_rather_than_raised(self, tmp_path: Path, oracle) -> None:
        out = tmp_path / "run"
        design(out)
        report = import_archive(oracle, out, tmp_path / "absent.zip")
        assert report.placed == []
        assert report.unmatched == ["absent.zip: archive_not_found"]

    def test_the_structure_keeps_the_servers_own_bytes(self, tmp_path: Path, oracle) -> None:
        """The filed CIF is the server's file, not a re-encoding of it."""
        out = tmp_path / "run"
        design(out)
        import_archive(oracle, out, archive(tmp_path / "fold.zip"), chain_of_interest="A")
        placed = oracle.refolded_cif_path(base(out), "binder_0")
        assert placed.read_text(encoding="utf-8") == SERVER_CIF
