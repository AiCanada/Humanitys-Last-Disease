"""The ``--refold`` mode: export a finished run, file what comes back.

These drive ``main`` through ``build_parser`` rather than calling the export
functions, because the thing being tested is that the mode is reachable and
that the two directions are wired to the same request directory. The functions
themselves are covered in ``test_disease_refold_export.py``.
"""

import json
from pathlib import Path

import pytest

from rfproteina.disease.cli import build_parser, main
from rfproteina.disease.refold_export import design_bases
from rfproteina.metrics.oracles.imported_refolds import PathOracle
from tests.helpers.disease import REFOLDED_CIF, TARGET_CHAIN, UBIQUITIN

_BINDER = UBIQUITIN


def _run_dir(tmp_path: Path, chains: dict[str, str], *, name: str = "binder_0", chain: str | None = "A") -> Path:
    out = tmp_path / "run"
    folder = out / "inference_samples" / name
    folder.mkdir(parents=True)
    rows = "\n".join(f"{i + 1} {c} {s}" for i, (c, s) in enumerate(chains.items()))
    (folder / f"{name}_generated.cif").write_text(
        "data_binder\n#\nloop_\n_entity_poly.entity_id\n_entity_poly.pdbx_strand_id\n"
        f"_entity_poly.pdbx_seq_one_letter_code_can\n{rows}\n#\n",
        encoding="utf-8",
    )
    table = out / "inference_metrics" / "per_example_metrics.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(f"example_id,chain_of_interest\n{name},{chain or ''}\n", encoding="utf-8")
    return out


def _spoken(capsys) -> dict:
    """The JSON the mode prints, which is its whole interface to a script.

    The run-log line is printed before it, and the record itself is indented
    over many lines, so it is the last JSON object in the output rather than
    the last line of it.
    """
    out = capsys.readouterr().out
    marker = "\n{\n"
    start = out.rindex(marker) + 1 if marker in out else out.index("{")
    return json.loads(out[start:])


class TestTheModeIsReachable:
    def test_refold_is_a_mode_of_its_own(self) -> None:
        parsed = build_parser().parse_args(["--refold", "somewhere"])
        assert parsed.refold == Path("somewhere")

    def test_it_cannot_be_combined_with_the_other_modes(self) -> None:
        """A mutually exclusive group, as --screen and --ui already are."""
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--refold", "a", "--screen", "b"])

    def test_the_oracle_name_defaults_but_can_be_set(self) -> None:
        assert build_parser().parse_args(["--refold", "a"]).refold_oracle == "af3"
        assert build_parser().parse_args(["--refold", "a", "--refold-oracle", "of3"]).refold_oracle == "of3"


class TestExport:
    def test_it_writes_a_fasta_and_a_manifest(self, tmp_path: Path, capsys) -> None:
        """A FASTA and the record of what was asked for, and nothing else.

        No upload file: the hosted server's dialect belongs to
        ``rfd4-af3server prepare``, which has the allowlists and the quota.
        """
        out = _run_dir(tmp_path, {"A": _BINDER, "B": TARGET_CHAIN})
        assert main(["--refold", str(out)]) == 0
        requests = out / "refold_requests"
        assert (requests / "binder_0.fasta").is_file()
        assert (requests / "refold_requests.json").is_file()
        assert sorted(path.name for path in requests.iterdir()) == [
            "binder_0.fasta",
            "refold_requests.json",
        ]

    def test_it_reports_what_is_outstanding(self, tmp_path: Path, capsys) -> None:
        """Right after an export that is everything, which is the point of saying it."""
        out = _run_dir(tmp_path, {"A": _BINDER})
        main(["--refold", str(out)])
        record = _spoken(capsys)
        assert record["status"]["outstanding"] == ["binder_0"]
        assert record["fasta"]["requests"] == 1
        assert "server_jobs" not in record, "the server dialect is af3server's, not this mode's"

    def test_the_request_directory_can_be_chosen(self, tmp_path: Path, capsys) -> None:
        out = _run_dir(tmp_path, {"A": _BINDER})
        elsewhere = tmp_path / "elsewhere"
        main(["--refold", str(out), "--refold-dir", str(elsewhere)])
        assert (elsewhere / "binder_0.fasta").is_file()
        assert not (out / "refold_requests").exists()

    def test_a_directory_that_is_not_there_is_refused(self, tmp_path: Path) -> None:
        assert main(["--refold", str(tmp_path / "absent")]) == 2


class TestImport:
    def test_a_returned_fold_is_filed_and_clears_the_outstanding_list(self, tmp_path: Path, capsys) -> None:
        """The whole round trip, driven the way an operator would drive it."""
        out = _run_dir(tmp_path, {"A": _BINDER})
        main(["--refold", str(out)])
        capsys.readouterr()

        # The operator folds it elsewhere; the download unpacks to a folder.
        downloads = tmp_path / "downloads" / "fold_binder_0"
        downloads.mkdir(parents=True)
        (downloads / "fold_binder_0_model_0.cif").write_text(REFOLDED_CIF, encoding="utf-8")
        (downloads / "fold_binder_0_summary_confidences_0.json").write_text(
            json.dumps({"ptm_coi": 88.0}), encoding="utf-8"
        )

        assert main(["--refold", str(out), "--refold-import", str(tmp_path / "downloads")]) == 0
        record = _spoken(capsys)
        assert record["imported"]["imported"] == 1
        assert record["status"]["outstanding"] == [], "the design is no longer waiting"
        assert record["status"]["refolded"] == 1
        # Asked at the design's own base, which is where the refolding metrics
        # look -- not at the run root, where nothing is kept.
        assert PathOracle().already_refolded(design_bases(out)["binder_0"], "binder_0")

    def test_the_import_directory_must_exist(self, tmp_path: Path) -> None:
        out = _run_dir(tmp_path, {"A": _BINDER})
        assert main(["--refold", str(out), "--refold-import", str(tmp_path / "absent")]) == 2

    def test_importing_writes_no_requests(self, tmp_path: Path, capsys) -> None:
        """An import is not an export; it must not overwrite the manifest."""
        out = _run_dir(tmp_path, {"A": _BINDER})
        downloads = tmp_path / "downloads"
        downloads.mkdir()
        main(["--refold", str(out), "--refold-import", str(downloads)])
        record = _spoken(capsys)
        assert "fasta" not in record
        assert record["status"] == {"blocker": "no_manifest"}, "nothing was ever exported"
