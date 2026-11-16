"""What a verdict rests on, and what moved when it changed.

Three gaps in the evidence a screen keeps.

A supplied predictor table is one scan and a run holds many designs. `mhc_i`,
`mhc_ii`, `ada` and a restricted `immunogenicity` find the design the table's
peptides are windows of -- the best-matching one, which is the right test for a
table inherited from an earlier design in the same folder -- and then recorded
the verdict for the whole run with nothing saying whose chain it came from. A
run of eight could read `mhc_ii: clear` on a scan of one.

Two inputs carried a digest, the proteome and the off-target reference, and the
reason given there held for all of them: a swapped table is otherwise invisible
between two runs.

And `replaced` said that the verdict had changed without saying what changed,
which is the part a reader can act on.
"""

import json
from pathlib import Path

from tests.helpers.disease import (
    TARGET_CHAIN,
    UBIQUITIN,
    binder_cif,
    closing_page,
    no_developer_files,  # noqa: F401
    screen_protein_run,
)

_ONE = UBIQUITIN
#: The Paul 2015 DR panel, which the ADA axis requires in full.
_DR = (
    "HLA-DRB1*03:01",
    "HLA-DRB1*07:01",
    "HLA-DRB1*15:01",
    "HLA-DRB3*01:01",
    "HLA-DRB3*02:02",
    "HLA-DRB4*01:01",
    "HLA-DRB5*01:01",
)


def _class_ii_scan(chain: str) -> str:
    """A full DR-panel consensus scan of one chain: every 15-mer, every allele, weak."""
    rows = ["peptide\tallele\t%Rank_EL\tmethod"]
    for start in range(len(chain) - 14):
        window = chain[start : start + 15]
        for allele in _DR:
            rows.append(f"{window}\t{allele}\t60.0\tIEDB consensus")
    return "\n".join(rows) + "\n"


def _build(root: Path, name: str, designs: dict[str, str], *, files: dict[str, str] | None = None) -> Path:
    output = root / name
    for design, sequence in designs.items():
        folder = output / "inference_samples" / design
        folder.mkdir(parents=True)
        (folder / f"{design}_generated.cif").write_text(binder_cif(sequence), encoding="utf-8")
    table = output / "inference_metrics" / "per_example_metrics.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(f"{design},A\n" for design in designs)
    table.write_text("example_id,chain_of_interest\n" + rows, encoding="utf-8")
    for filename, text in (files or {}).items():
        (output / filename).write_text(text, encoding="utf-8")
    return output


class TestTableCoverage:
    def test_a_single_design_run_is_fully_covered(self, tmp_path: Path) -> None:
        output = _build(tmp_path, "one", {"binder_0": _ONE}, files={"mhc_ii.tsv": _class_ii_scan(_ONE)})
        ada = screen_protein_run(output)["sequence_liability"]["ada"]
        assert ada["state"] == "clear"
        assert ada["covered_examples"] == ["binder_0"]
        assert "uncovered_examples" not in ada

    def test_a_scan_of_one_design_names_the_designs_it_never_saw(self, tmp_path: Path) -> None:
        output = _build(
            tmp_path,
            "two",
            {"binder_0": _ONE, "binder_1": TARGET_CHAIN},
            files={"mhc_ii.tsv": _class_ii_scan(_ONE)},
        )
        ada = screen_protein_run(output)["sequence_liability"]["ada"]
        assert ada["covered_examples"] == ["binder_0"]
        assert ada["uncovered_examples"] == ["binder_1"]

    def test_the_state_is_recorded_for_the_run_and_the_scope_beside_it(self, tmp_path: Path) -> None:
        """The clear covers a design no predictor scored. That is now on the record.

        This test pins the scope, not a narrowing: whether such a clear should
        stand is a decision about what the verdict means, and tightening it here
        would make the protein sequence bar unreachable for any multi-design run.
        """
        output = _build(
            tmp_path,
            "scope",
            {"binder_0": _ONE, "binder_1": TARGET_CHAIN},
            files={"mhc_ii.tsv": _class_ii_scan(_ONE)},
        )
        ada = screen_protein_run(output)["sequence_liability"]["ada"]
        assert ada["state"] == "clear"
        assert "binder_1" in ada["uncovered_examples"]

    def test_the_class_ii_axis_reports_coverage_too(self, tmp_path: Path) -> None:
        output = _build(
            tmp_path,
            "two",
            {"binder_0": _ONE, "binder_1": TARGET_CHAIN},
            files={"mhc_ii.tsv": _class_ii_scan(_ONE)},
        )
        assert screen_protein_run(output)["sequence_liability"]["mhc_ii"]["uncovered_examples"] == ["binder_1"]

    def test_the_closed_page_names_the_uncovered_designs(self, tmp_path: Path) -> None:
        output = _build(
            tmp_path,
            "two",
            {"binder_0": _ONE, "binder_1": TARGET_CHAIN},
            files={"mhc_ii.tsv": _class_ii_scan(_ONE)},
        )
        assert "Designs this table does not cover: binder_1." in closing_page(screen_protein_run(output))

    def test_a_refused_table_states_no_coverage(self, tmp_path: Path) -> None:
        """There is no claim to put a scope on."""
        output = _build(
            tmp_path,
            "refused",
            {"binder_0": _ONE, "binder_1": TARGET_CHAIN},
            files={"mhc_ii.tsv": _class_ii_scan(TARGET_CHAIN)[:200]},
        )
        ada = screen_protein_run(output)["sequence_liability"]["ada"]
        assert ada["state"] == "untested"
        assert "covered_examples" not in ada


class TestSuppliedInputs:
    def test_every_supplied_file_is_recorded_with_its_digest(self, tmp_path: Path) -> None:
        output = _build(tmp_path, "run", {"binder_0": _ONE}, files={"mhc_ii.tsv": _class_ii_scan(_ONE)})
        inputs = screen_protein_run(output)["inputs"]
        entry = next(item for item in inputs if item["file"] == "mhc_ii.tsv")
        assert entry["bytes"] > 0
        assert len(entry["digest"]) == 16

    def test_curated_evidence_is_an_input_too(self, tmp_path: Path) -> None:
        """It is the most editable input of all."""
        output = _build(
            tmp_path,
            "curated",
            {"binder_0": _ONE},
            files={"axis_evidence.json": json.dumps({"paralog": "clear"})},
        )
        assert "axis_evidence.json" in {item["file"] for item in screen_protein_run(output)["inputs"]}

    def test_the_screen_is_not_one_of_its_own_inputs(self, tmp_path: Path) -> None:
        output = _build(tmp_path, "run", {"binder_0": _ONE})
        screen_protein_run(output)
        assert "screen.json" not in {item["file"] for item in screen_protein_run(output)["inputs"]}

    def test_a_run_with_nothing_supplied_records_an_empty_list(self, tmp_path: Path) -> None:
        output = _build(tmp_path, "bare", {"binder_0": _ONE})
        assert screen_protein_run(output)["inputs"] == []


class TestChangedInputs:
    def test_screening_twice_unchanged_records_no_replacement(self, tmp_path: Path) -> None:
        output = _build(tmp_path, "same", {"binder_0": _ONE})
        assert "replaced" not in screen_protein_run(output)
        assert "replaced" not in screen_protein_run(output)

    def test_a_file_that_appeared_is_named(self, tmp_path: Path) -> None:
        output = _build(tmp_path, "added", {"binder_0": _ONE})
        first = screen_protein_run(output)
        (output / "axis_evidence.json").write_text(json.dumps({"paralog": "clear"}), encoding="utf-8")
        second = screen_protein_run(output)
        assert second["replaced"]["call"] == first["call"]
        assert second["replaced"]["changed_inputs"] == [{"file": "axis_evidence.json", "change": "added"}]

    def test_an_edited_file_is_named_as_edited(self, tmp_path: Path) -> None:
        output = _build(
            tmp_path,
            "edited",
            {"binder_0": _ONE},
            files={"axis_evidence.json": json.dumps({"paralog": "clear"})},
        )
        screen_protein_run(output)
        (output / "axis_evidence.json").write_text(json.dumps({"paralog": "hit"}), encoding="utf-8")
        assert screen_protein_run(output)["replaced"]["changed_inputs"] == [
            {"file": "axis_evidence.json", "change": "edited"}
        ]

    def test_a_removed_file_is_named_as_removed(self, tmp_path: Path) -> None:
        output = _build(
            tmp_path,
            "removed",
            {"binder_0": _ONE},
            files={"axis_evidence.json": json.dumps({"paralog": "clear"})},
        )
        screen_protein_run(output)
        (output / "axis_evidence.json").unlink()
        assert screen_protein_run(output)["replaced"]["changed_inputs"] == [
            {"file": "axis_evidence.json", "change": "removed"}
        ]

    def test_the_prior_inputs_are_not_written_into_the_replacement(self, tmp_path: Path) -> None:
        """They are read to be compared, not stored a second time."""
        output = _build(tmp_path, "once", {"binder_0": _ONE})
        screen_protein_run(output)
        (output / "axis_evidence.json").write_text(json.dumps({"paralog": "clear"}), encoding="utf-8")
        assert "inputs" not in screen_protein_run(output)["replaced"]
