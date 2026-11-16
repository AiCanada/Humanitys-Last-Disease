"""Exporting a design for an external folder, and the round trip back.

`refold_export` writes the request; `imported_refolds` files the answer. The
pair exists for folders driven by sequence rather than by a local checkpoint,
including a web service with no API where the submission is a person pasting
text and the return is a download.

The test that earns its keep here is the last one: export, fold elsewhere,
import, and the configured oracle calls the design refolded. If the names do
not line up across that gap, nothing else in this file matters.
"""

import json
from pathlib import Path

import pytest

from rfproteina.disease.refold_export import (
    REQUEST_CHAIN_LIMIT,
    REQUEST_RESIDUE_LIMIT,
    design_bases,
    design_fasta,
    refold_status,
    write_refold_requests,
)
from rfproteina.utils.common import design_output_dir
from tests.helpers.disease import TARGET_CHAIN, UBIQUITIN

_BINDER = UBIQUITIN


def _complex_cif(chains: dict[str, str]) -> str:
    """A generated structure holding one entity_poly row per chain."""
    rows = "\n".join(f"{index + 1} {chain} {seq}" for index, (chain, seq) in enumerate(chains.items()))
    return (
        "data_binder\n#\nloop_\n_entity_poly.entity_id\n"
        "_entity_poly.pdbx_strand_id\n_entity_poly.pdbx_seq_one_letter_code_can\n"
        f"{rows}\n#\n"
    )


def _run(tmp_path: Path, designs: dict[str, dict[str, str]], *, chain: str | None = "A") -> Path:
    out = tmp_path / "run"
    for name, chains in designs.items():
        folder = out / "inference_samples" / name
        folder.mkdir(parents=True)
        (folder / f"{name}_generated.cif").write_text(_complex_cif(chains), encoding="utf-8")
    table = out / "inference_metrics" / "per_example_metrics.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(f"{name},{chain or ''}\n" for name in designs)
    table.write_text("example_id,chain_of_interest\n" + rows, encoding="utf-8")
    return out


class TestOneRequest:
    def test_every_chain_is_in_the_request(self) -> None:
        """A binder folded alone cannot answer whether the complex forms."""
        made = design_fasta(
            _complex_cif({"A": _BINDER, "B": TARGET_CHAIN}),
            example_id="binder_0",
            chain_of_interest="A",
        )
        body, chains, residues = made
        assert chains == ("A", "B")
        assert residues == len(_BINDER) + len(TARGET_CHAIN)
        assert body.count(">") == 2

    def test_the_design_chain_is_marked_and_the_rest_are_context(self) -> None:
        body, _chains, _residues = design_fasta(
            _complex_cif({"A": _BINDER, "B": TARGET_CHAIN}),
            example_id="binder_0",
            chain_of_interest="A",
        )
        assert ">binder_0|chain_A|design" in body
        assert ">binder_0|chain_B|context" in body

    def test_with_no_chain_named_nothing_is_claimed_as_the_design(self) -> None:
        body, _chains, _residues = design_fasta(
            _complex_cif({"A": _BINDER}), example_id="binder_0", chain_of_interest=None
        )
        assert "|design" not in body
        assert ">binder_0|chain_A|context" in body

    def test_the_body_wraps_at_sixty(self) -> None:
        body, _chains, _residues = design_fasta(
            _complex_cif({"A": _BINDER}), example_id="binder_0", chain_of_interest="A"
        )
        lines = [line for line in body.splitlines() if not line.startswith(">")]
        assert max(len(line) for line in lines) == 60
        assert "".join(lines) == _BINDER

    @pytest.mark.parametrize(
        ("label", "chains", "blocker"),
        [
            ("nothing to fold", {}, "no_polymer_sequence"),
            ("a blank sequence", {"A": "   "}, "no_polymer_sequence"),
        ],
    )
    def test_a_structure_that_yields_no_request_says_why(self, label: str, chains: dict, blocker: str) -> None:
        assert design_fasta(_complex_cif(chains), example_id="b", chain_of_interest=None) == blocker

    def test_a_request_past_the_residue_limit_is_refused_before_it_is_sent(self) -> None:
        """A service bounds what it accepts; better to know here than from a rejection."""
        big = "A" * (REQUEST_RESIDUE_LIMIT + 1)
        assert design_fasta(_complex_cif({"A": big}), example_id="b", chain_of_interest="A") == "over_residue_limit"

    def test_a_file_with_more_chains_than_one_complex_is_refused(self) -> None:
        many = {chr(ord("A") + index): "AAAA" for index in range(REQUEST_CHAIN_LIMIT + 1)}
        assert design_fasta(_complex_cif(many), example_id="b", chain_of_interest="A") == "too_many_chains"


class TestWriteRequests:
    def test_one_request_per_design_named_for_the_design(self, tmp_path: Path) -> None:
        out = _run(tmp_path, {"binder_0": {"A": _BINDER, "B": TARGET_CHAIN}, "binder_1": {"A": _BINDER}})
        report = write_refold_requests(out, tmp_path / "requests")
        assert [item.path.name for item in report.written] == ["binder_0.fasta", "binder_1.fasta"]
        assert report.refused == []

    def test_the_chain_of_interest_comes_from_the_metrics_table(self, tmp_path: Path) -> None:
        out = _run(tmp_path, {"binder_0": {"A": _BINDER, "B": TARGET_CHAIN}}, chain="A")
        report = write_refold_requests(out, tmp_path / "requests")
        assert report.written[0].chain_of_interest == "A"
        assert "|chain_A|design" in report.written[0].path.read_text(encoding="utf-8")

    def test_a_run_naming_no_chain_still_exports(self, tmp_path: Path) -> None:
        out = _run(tmp_path, {"binder_0": {"A": _BINDER, "B": TARGET_CHAIN}}, chain=None)
        report = write_refold_requests(out, tmp_path / "requests")
        assert report.written[0].chain_of_interest is None
        assert report.written[0].chains == ("A", "B"), "every chain is in the request either way"

    def test_a_manifest_records_what_was_asked_for(self, tmp_path: Path) -> None:
        """The return leg is manual, so there has to be something to check arrivals against."""
        out = _run(tmp_path, {"binder_0": {"A": _BINDER, "B": TARGET_CHAIN}})
        write_refold_requests(out, tmp_path / "requests")
        manifest = json.loads((tmp_path / "requests" / "refold_requests.json").read_text(encoding="utf-8"))
        assert manifest["residue_limit"] == REQUEST_RESIDUE_LIMIT
        assert manifest["requests"] == [
            {
                "example_id": "binder_0",
                "file": "binder_0.fasta",
                "chains": ["A", "B"],
                "residues": len(_BINDER) + len(TARGET_CHAIN),
                "chain_of_interest": "A",
            }
        ]

    def test_the_report_names_every_refusal(self, tmp_path: Path) -> None:
        out = _run(tmp_path, {"binder_0": {"A": _BINDER}, "binder_1": {"A": "  "}})
        report = write_refold_requests(out, tmp_path / "requests")
        assert report.as_dict() == {
            "requests": 1,
            "residues": len(_BINDER),
            "refused": [{"example_id": "binder_1", "blocker": "no_polymer_sequence"}],
        }

    def test_a_run_with_no_designs_writes_nothing(self, tmp_path: Path) -> None:
        out = tmp_path / "empty"
        out.mkdir()
        report = write_refold_requests(out, tmp_path / "requests")
        assert report.written == []
        assert not (tmp_path / "requests" / "refold_requests.json").exists()


class TestTheChainInstanceId:
    """A real run records ``A_1``, not ``A``.

    The metrics table writes a chain instance id because the screen matches it
    against a parsed structure's ``chain_iid``. ``entity_poly`` names the
    strand ``A``. Nothing caught this until the export ran on a real run, where
    every design came out with its chain unmarked and the map recorded no
    chain of interest at all.
    """

    def test_an_instance_id_still_marks_the_design(self) -> None:
        body, _chains, _residues = design_fasta(
            _complex_cif({"A": _BINDER, "E": TARGET_CHAIN}), example_id="binder_0", chain_of_interest="A_1"
        )
        assert ">binder_0|chain_A|design" in body
        assert ">binder_0|chain_E|context" in body

    def test_a_bare_strand_id_still_works(self) -> None:
        body, _chains, _residues = design_fasta(
            _complex_cif({"A": _BINDER}), example_id="binder_0", chain_of_interest="A"
        )
        assert "|design" in body

    def test_a_chain_that_is_not_there_marks_nothing(self) -> None:
        """Better unmarked than marking the wrong chain as the design."""
        body, _chains, _residues = design_fasta(
            _complex_cif({"A": _BINDER, "E": TARGET_CHAIN}), example_id="binder_0", chain_of_interest="Z_1"
        )
        assert "|design" not in body

    def test_only_the_trailing_number_is_dropped(self) -> None:
        """`E_1` must resolve to `E`, never to `A` because it was listed first."""
        body, _chains, _residues = design_fasta(
            _complex_cif({"A": _BINDER, "E": TARGET_CHAIN}), example_id="binder_0", chain_of_interest="E_1"
        )
        assert ">binder_0|chain_E|design" in body
        assert ">binder_0|chain_A|context" in body

    def test_the_manifest_records_the_strand_not_the_instance(self, tmp_path: Path) -> None:
        out = _run(tmp_path, {"binder_0": {"A": _BINDER, "E": TARGET_CHAIN}}, chain="A_1")
        report = write_refold_requests(out, tmp_path / "requests")
        assert report.written[0].chain_of_interest == "A", "the strand, as entity_poly spells it"


class TestRoundTripStatus:
    """Which designs are still outstanding.

    The middle leg is manual and the AlphaFold Server caps jobs per day, so a
    batch comes back over several sittings and the question "what is left" gets
    asked repeatedly.
    """

    class _Oracle:
        """Refolded iff a marker file is there, which is the real predicate's shape.

        ``out_dir`` is the base handed to it per design, so this resolves to
        ``inference_samples/<id>/done.json`` for a real run.
        """

        def already_refolded(self, out_dir: Path, example_id: str, save_id: str | None = None) -> bool:
            return (out_dir / example_id / "done.json").exists()

    def _back(self, out: Path, *example_ids: str) -> None:
        """Mark a design as returned, where a run actually keeps it.

        Under ``inference_samples/<id>/`` and not ``<id>/``, because
        ``design_output_dir`` requires a design's folder to be a direct child
        of the base it is given and the base for a real run is
        ``inference_samples``. A fixture that wrote it at the run root passed
        against a status check that looked in the same wrong place.
        """
        for example_id in example_ids:
            folder = out / "inference_samples" / example_id
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "done.json").write_text("{}", encoding="utf-8")

    def test_nothing_back_yet_leaves_everything_outstanding(self, tmp_path: Path) -> None:
        out = _run(tmp_path, {"binder_0": {"A": _BINDER}, "binder_1": {"A": _BINDER}})
        write_refold_requests(out, tmp_path / "requests")
        status = refold_status(self._Oracle(), out, tmp_path / "requests")
        assert status.outstanding == ("binder_0", "binder_1")
        assert status.refolded == ()
        assert not status.complete

    def test_a_partly_returned_batch_names_only_what_is_left(self, tmp_path: Path) -> None:
        out = _run(tmp_path, {"binder_0": {"A": _BINDER}, "binder_1": {"A": _BINDER}})
        write_refold_requests(out, tmp_path / "requests")
        self._back(out, "binder_0")
        status = refold_status(self._Oracle(), out, tmp_path / "requests")
        assert status.refolded == ("binder_0",)
        assert status.outstanding == ("binder_1",)
        assert status.as_dict() == {"requested": 2, "refolded": 1, "outstanding": ["binder_1"]}

    def test_a_finished_batch_says_so(self, tmp_path: Path) -> None:
        out = _run(tmp_path, {"binder_0": {"A": _BINDER}})
        write_refold_requests(out, tmp_path / "requests")
        self._back(out, "binder_0")
        assert refold_status(self._Oracle(), out, tmp_path / "requests").complete

    def test_a_fold_no_one_asked_for_is_reported_not_counted(self, tmp_path: Path) -> None:
        """A design generated after the export was written, and folded anyway.

        Not an error, and not counted against the batch either -- but said, so
        that a count here cannot quietly disagree with the directory.
        """
        out = _run(tmp_path, {"binder_0": {"A": _BINDER}})
        write_refold_requests(out, tmp_path / "requests")
        # Arrives after the manifest was written, so the manifest does not name it.
        later = out / "inference_samples" / "binder_9"
        later.mkdir(parents=True)
        (later / "binder_9_generated.cif").write_text(_complex_cif({"A": TARGET_CHAIN}), encoding="utf-8")
        self._back(out, "binder_0", "binder_9")
        status = refold_status(self._Oracle(), out, tmp_path / "requests")
        assert status.requested == ("binder_0",)
        assert status.unrequested == ("binder_9",)
        assert status.as_dict()["unrequested"] == ["binder_9"]

    def test_a_design_whose_refold_has_nowhere_to_go_is_named_not_misfiled(self, tmp_path: Path) -> None:
        """Two batches repeating the binder token, which flattens the id.

        ``inference_samples/run_a/binder_0/`` is exported as ``run_a_binder_0``,
        because the bare token is not unique across batches. The parent of its
        folder is ``inference_samples/run_a``, so pairing the two put the refold
        in ``inference_samples/run_a/run_a_binder_0/`` -- a directory holding no
        design. ``already_refolded`` read that same wrong place, so a misfiled
        refold reported itself as back and the real design's numbers were absent
        from the screen with nothing raised anywhere.

        There is no base that joins a flattened id onto a nested folder, so the
        honest answer is to say which designs cannot be filed.
        """
        out = tmp_path / "run"
        for batch in ("run_a", "run_b"):
            folder = out / "inference_samples" / batch / "binder_0"
            folder.mkdir(parents=True)
            (folder / "binder_0_generated.cif").write_text(_complex_cif({"A": _BINDER}), encoding="utf-8")
        write_refold_requests(out, tmp_path / "requests")
        # A marker in the one place the old pairing would have filed it: beside
        # no design. It must not be read as the design having come back.
        for batch in ("run_a", "run_b"):
            misfiled = out / "inference_samples" / batch / f"{batch}_binder_0"
            misfiled.mkdir(parents=True)
            (misfiled / "done.json").write_text("{}", encoding="utf-8")

        status = refold_status(self._Oracle(), out, tmp_path / "requests")
        assert status.requested == ("run_a_binder_0", "run_b_binder_0")
        assert status.unplaceable == ("run_a_binder_0", "run_b_binder_0")
        # Outstanding, not refolded: a refold filed where no design is does not
        # count as that design having come back.
        assert status.outstanding == ("run_a_binder_0", "run_b_binder_0")
        assert status.refolded == ()
        assert not status.complete
        assert status.as_dict()["unplaceable"] == ["run_a_binder_0", "run_b_binder_0"]

    def test_the_ordinary_one_batch_layout_is_placeable(self, tmp_path: Path) -> None:
        """The guard above must not refuse the layout every real run uses."""
        out = _run(tmp_path, {"binder_0": {"A": _BINDER}, "binder_1": {"A": _BINDER}})
        assert sorted(design_bases(out)) == ["binder_0", "binder_1"]
        for example_id, base in design_bases(out).items():
            assert design_output_dir(base, example_id).is_dir()

    @pytest.mark.parametrize(
        ("label", "text", "blocker"),
        [
            ("absent", None, "no_manifest"),
            ("not json", "{nope", "manifest_unreadable"),
            ("a list", "[1, 2]", "manifest_not_a_manifest"),
            ("missing its requests", '{"residue_limit": 5000}', "manifest_not_a_manifest"),
        ],
    )
    def test_a_manifest_that_cannot_be_read_says_why(
        self, tmp_path: Path, label: str, text: str | None, blocker: str
    ) -> None:
        requests = tmp_path / "requests"
        requests.mkdir()
        if text is not None:
            (requests / "refold_requests.json").write_text(text, encoding="utf-8")
        status = refold_status(self._Oracle(), tmp_path / "run", requests)
        assert status.as_dict() == {"blocker": blocker}
        assert not status.complete

    def test_an_oracle_that_raises_counts_as_not_back(self, tmp_path: Path) -> None:
        """One unusable id must not take the whole status report down."""

        class _Raises:
            def already_refolded(self, out_dir: Path, example_id: str, save_id: str | None = None) -> bool:
                raise ValueError("not a directory name")

        out = _run(tmp_path, {"binder_0": {"A": _BINDER}})
        write_refold_requests(out, tmp_path / "requests")
        status = refold_status(_Raises(), out, tmp_path / "requests")
        assert status.outstanding == ("binder_0",)


#: Exactly the keys DeepMind's converter allows on a fold job. Taken from
#: ``_validate_keys`` in ``alphafold3/common/folding_input.py``, which raises
#: on any other field -- so an extra one of ours invalidates the whole upload.
_JOB_KEYS = {"name", "modelSeeds", "sequences", "dialect", "version"}
#: Likewise for a protein entity.
_PROTEIN_KEYS = {"sequence", "glycans", "modifications", "count", "maxTemplateDate", "useStructureTemplate"}

_DNA = "ACGTACGTACGTACGTACGT"
_RNA = "ACGUACGUACGUACGUACGU"


# The AlphaFold Server JSON tests that were here are gone with the exporter
# they covered; `rfproteina.af3server.request` is the one implementation of
# that dialect now and `tests/test_af3server_request.py` tests it. What they
# asserted was carried across, including the two decisions that changed:
# `polypeptide(D)` is refused rather than written as a `proteinChain` the
# server would fold as the mirror-image molecule, and an untyped polymer is
# refused rather than inferred from its alphabet.


def test_the_round_trip_closes(tmp_path: Path) -> None:
    """Export, fold elsewhere, import, and the oracle calls the design refolded.

    The middle step is whatever the operator does -- a local folder, a cluster,
    a web form. Here it is a file appearing under the exported name with a
    tool's own suffix on it, which is what comes back in practice.
    """
    from rfproteina.metrics.oracles.imported_refolds import PathOracle, import_refolds

    # PathOracle rather than a fourth hand-rolled copy of the canonical paths:
    # it is what the import contract is written against, and a drift test keeps
    # it in step with OracleMetric's own.
    oracle = PathOracle(oracle_name="of3")

    out = _run(tmp_path, {"binder_0": {"A": _BINDER, "B": TARGET_CHAIN}})
    requests = tmp_path / "requests"
    export = write_refold_requests(out, requests)
    assert [item.path.name for item in export.written] == ["binder_0.fasta"]

    # The operator folds it somewhere and saves the result under that name.
    returned = tmp_path / "downloads"
    returned.mkdir()
    (returned / "binder_0_model_1.cif").write_text("data_refold\n#\nloop_\n_atom_site.group_PDB\nATOM\n#\n", "utf-8")

    # The base is the design's own, not the run. A run nests its designs under
    # inference_samples/, so handing the run to import_refolds writes the pair
    # into <run>/binder_0/ -- a directory holding no design and one no oracle
    # reads. The check afterwards then looked in that same wrong place, so this
    # test agreed with itself while neither end matched a real run. That is the
    # question design_bases answers, and it is the same mistake that made
    # af3server's own import file refolds where nothing would find them.
    base = design_bases(out)["binder_0"]
    assert base == out / "inference_samples"

    assert not oracle.already_refolded(base, "binder_0")
    report = import_refolds(oracle, base, returned, example_ids=["binder_0"])

    assert [item.example_id for item in report.placed] == ["binder_0"]
    assert report.unmatched == []
    assert oracle.already_refolded(base, "binder_0"), "the exported name carried through to the import"
    # Beside the design, and the directory the old mistake invented is absent.
    assert oracle.confidence_path(base, "binder_0").parent == out / "inference_samples" / "binder_0"
    assert not (out / "binder_0").exists()
