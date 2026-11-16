"""What the screen says about a verdict it did not measure itself.

Three silences, each of which let a record claim more than was measured.

A curated state was printed for the sequence axes only, so a human's clear over
a measured hit on paralog, a plasma axis or either nucleic axis printed nothing
at all and the line read as a plain screened_clear.

A run reached only through a symlinked or a dot directory was pruned out of the
walk without the question the depth cap already asks, so a sweep that looked at
nothing reported itself complete. A supplied input that is a symlink decided a
verdict and appeared in no input list. A repeated axis in the evidence file kept
whichever copy came last.

And a run that measured nothing -- no design, or designs this screen cannot read
-- named no reason, which is the one thing every other miss here does.

Every test supplies its own files and no reader, so nothing reaches the network.
"""

import json
import os
from pathlib import Path

import pytest

from rfproteina.disease.screen import (
    ScreenWalkTruncatedError,
    format_screen_lines_for_directory,
    generation_directories,
    screen_generation,
)
from rfproteina.disease.screen_board import render_closing
from tests.helpers.disease import (
    UBIQUITIN,
    finished_run,
    job_file,
    no_developer_files,  # noqa: F401
)

#: A purine-purine-CG-pyrimidine-pyrimidine hexamer twice over, which is a
#: measured ``innate_immune`` hit, and long enough to hold a 15-base seed.
_CPG_OLIGO = "GACGTTGACGTTATGCATGCATGCATGCATGC"


def _link(target: Path, link: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - platform
        pytest.skip(f"this filesystem does not make symlinks: {exc}")


def _cif(sequence: str, *, row: str | None = None) -> str:
    """A sequence-only design. ``row`` replaces the one data line."""
    data = row if row is not None else f"1 A {sequence}"
    return (
        "data_design\n#\nloop_\n_entity_poly.entity_id\n"
        "_entity_poly.pdbx_strand_id\n_entity_poly.pdbx_seq_one_letter_code_can\n"
        f"{data}\n#\n"
    )


def _design(run: Path) -> Path:
    return run / "inference_samples" / "binder_0" / "binder_0_generated.cif"


def _lines(run: Path, screen: dict) -> str:
    return " ".join(format_screen_lines_for_directory(run, screen))


class TestCuratedStatesArePrinted:
    """A curated call has to be visible on the one line a human reads."""

    def test_a_curated_clear_over_a_measured_nucleic_hit_is_printed_as_curated(self, tmp_path: Path) -> None:
        run = finished_run(tmp_path, sequence=_CPG_OLIGO)
        job_file(run, kind="nucleic")
        measured = screen_generation(run, binder_kind="nucleic", target_id="P05113")
        assert measured["nucleic_liability"]["innate_immune"]["state"] == "hit"

        (run / "axis_evidence.json").write_text(
            json.dumps({"innate_immune": "clear", "off_target_hybridization": "clear"}),
            encoding="utf-8",
        )
        screen = screen_generation(run, binder_kind="nucleic", target_id="P05113")
        assert screen["call"] == "screened_clear"
        printed = _lines(run, screen)
        assert "innate_immune clear, source axis_evidence, measured hit" in printed
        assert "off_target_hybridization clear, source axis_evidence" in printed

    def test_a_curated_target_axis_is_printed_as_curated(self, tmp_path: Path) -> None:
        run = finished_run(tmp_path)
        job_file(run)
        (run / "axis_evidence.json").write_text(json.dumps({"paralog": "clear"}), encoding="utf-8")
        screen = screen_generation(run, binder_kind="protein", target_id="P05113")
        assert "paralog clear, source axis_evidence" in _lines(run, screen)

    def test_a_curated_plasma_bar_names_each_axis_a_human_decided(self, tmp_path: Path) -> None:
        run = finished_run(tmp_path)
        job_file(run)
        (run / "axis_evidence.json").write_text(
            json.dumps({"plasma_abundance": "clear", "plasma_hub": "clear", "plasma_fold": "clear"}),
            encoding="utf-8",
        )
        screen = screen_generation(run, binder_kind="protein", target_id="P05113", bar="plasma")
        printed = _lines(run, screen)
        assert screen["call"] == "screened_clear"
        for axis in ("plasma_abundance", "plasma_hub", "plasma_fold"):
            assert f"{axis} clear, source axis_evidence" in printed

    def test_an_uncurated_run_says_nothing_about_curated_evidence(self, tmp_path: Path) -> None:
        """The sentence of a run nobody curated is the sentence it always was."""
        run = finished_run(tmp_path)
        job_file(run)
        screen = screen_generation(run, binder_kind="protein", target_id="P05113")
        printed = _lines(run, screen)
        assert "axis_evidence" not in printed
        assert "gravy" in printed


class TestTheWalkReportsWhatItPruned:
    """A pruned subtree holding a run is a partial sweep, not an empty tree."""

    def test_a_run_behind_a_symlinked_directory_is_reported(self, tmp_path: Path) -> None:
        run = finished_run(tmp_path / "store")
        tree = tmp_path / "tree"
        tree.mkdir()
        _link(run, tree / "linked_run")
        with pytest.raises(ScreenWalkTruncatedError):
            generation_directories(tree)

    def test_a_run_under_a_dot_directory_is_reported(self, tmp_path: Path) -> None:
        tree = tmp_path / "tree"
        finished_run(tree / ".archive", "gene_2ABC_1")
        with pytest.raises(ScreenWalkTruncatedError):
            generation_directories(tree)

    def test_a_dot_directory_with_no_run_in_it_is_still_silent(self, tmp_path: Path) -> None:
        """A false gap announced on every sweep teaches the warning to be ignored."""
        tree = tmp_path / "tree"
        run = finished_run(tree, "gene_3ABC_1")
        (tree / ".notes" / "deeper").mkdir(parents=True)
        assert generation_directories(tree) == [run]

    def test_a_link_to_a_run_already_in_the_tree_is_not_a_gap(self, tmp_path: Path) -> None:
        """`current -> gene_1ABC_1` beside the run it names hides nothing."""
        tree = tmp_path / "tree"
        run = finished_run(tree, "gene_4ABC_1")
        _link(run, tree / "current")
        assert generation_directories(tree) == [run]

    def test_a_link_back_to_the_tree_is_not_a_gap(self, tmp_path: Path) -> None:
        tree = tmp_path / "tree"
        run = finished_run(tree, "gene_5ABC_1")
        _link(tree, tree / "loop")
        _link(tmp_path, tree / "above")
        assert generation_directories(tree) == [run]

    def test_a_large_dot_directory_with_no_run_is_still_silent(self, tmp_path: Path) -> None:
        """A tool cache holds more folders than the look-ahead reads, and no run."""
        tree = tmp_path / "tree"
        run = finished_run(tree, "gene_6ABC_1")
        for number in range(70):
            (tree / ".tool_cache" / f"entry_{number}").mkdir(parents=True)
        assert generation_directories(tree) == [run]

    def test_a_large_linked_folder_with_no_run_is_still_silent(self, tmp_path: Path) -> None:
        tree = tmp_path / "tree"
        run = finished_run(tree, "gene_7ABC_1")
        for number in range(70):
            (tmp_path / "dataset" / f"part_{number}").mkdir(parents=True)
        _link(tmp_path / "dataset", tree / "data")
        assert generation_directories(tree) == [run]

    def test_dot_directories_inside_runs_do_not_spend_the_budget(self, tmp_path: Path) -> None:
        """Hydra leaves `.hydra` in every run. Forty-five such runs is an archive, not a gap."""
        tree = tmp_path / "tree"
        runs = []
        for number in range(45):
            run = finished_run(tree, f"gene_{number:04d}_1")
            (run / ".hydra").mkdir()
            runs.append(run)
        assert generation_directories(tree) == sorted(runs)

    def test_a_link_that_cannot_be_reached_does_not_stop_the_sweep(self, tmp_path: Path) -> None:
        tree = tmp_path / "tree"
        run = finished_run(tree, "gene_8ABC_1")
        vault = tmp_path / "vault"
        (vault / "inner").mkdir(parents=True)
        _link(vault / "inner", tree / "locked")
        vault.chmod(0o000)
        try:
            if os.access(vault, os.X_OK):
                pytest.skip("this user can enter a folder it has no permission for")
            assert generation_directories(tree) == [run]
        finally:
            vault.chmod(0o700)


class TestSuppliedInputsNameEveryFile:
    def test_a_symlinked_input_is_digested_and_marked_as_a_link(self, tmp_path: Path) -> None:
        run = finished_run(tmp_path / "store")
        shared = tmp_path / "shared_evidence.json"
        shared.write_text(json.dumps({"paralog": "clear"}), encoding="utf-8")
        _link(shared, run / "axis_evidence.json")

        first = screen_generation(run, binder_kind="protein", target_id="P05113")
        entry = next(item for item in first["inputs"] if item["file"] == "axis_evidence.json")
        assert entry["link"] is True
        assert len(entry["digest"]) == 16

        shared.write_text(json.dumps({"paralog": "hit"}), encoding="utf-8")
        second = screen_generation(run, binder_kind="protein", target_id="P05113")
        assert second["replaced"]["changed_inputs"] == [{"file": "axis_evidence.json", "change": "edited"}]

    def test_a_plain_input_is_recorded_exactly_as_before(self, tmp_path: Path) -> None:
        run = finished_run(tmp_path)
        (run / "axis_evidence.json").write_text(json.dumps({"paralog": "clear"}), encoding="utf-8")
        screen = screen_generation(run, binder_kind="protein", target_id="P05113")
        entry = next(item for item in screen["inputs"] if item["file"] == "axis_evidence.json")
        assert "link" not in entry


class TestTheEvidenceFileIsRefusedByName:
    def test_a_repeated_axis_is_refused(self, tmp_path: Path) -> None:
        run = finished_run(tmp_path)
        (run / "axis_evidence.json").write_text('{"paralog": "hit", "paralog": "clear"}', encoding="utf-8")
        with pytest.raises(ValueError, match="twice"):
            screen_generation(run, binder_kind="protein", target_id="P05113")
        assert not (run / "screen.json").exists()

    def test_a_repeated_field_inside_one_axis_is_refused_in_words_that_fit(self, tmp_path: Path) -> None:
        run = finished_run(tmp_path)
        (run / "axis_evidence.json").write_text(
            '{"paralog": {"state": "clear", "reason": "a", "reason": "b"}}', encoding="utf-8"
        )
        with pytest.raises(ValueError, match="names reason twice") as refused:
            screen_generation(run, binder_kind="protein", target_id="P05113")
        assert "axis" not in str(refused.value).replace("axis_evidence.json", "")

    def test_an_evidence_file_that_is_a_directory_is_refused_not_raised(self, tmp_path: Path) -> None:
        run = finished_run(tmp_path)
        (run / "axis_evidence.json").mkdir()
        with pytest.raises(ValueError, match="axis_evidence.json"):
            screen_generation(run, binder_kind="protein", target_id="P05113")

    def test_a_dangling_link_is_refused_rather_than_read_as_absence(self, tmp_path: Path) -> None:
        run = finished_run(tmp_path)
        _link(tmp_path / "never_written.json", run / "axis_evidence.json")
        with pytest.raises(ValueError, match="axis_evidence.json"):
            screen_generation(run, binder_kind="protein", target_id="P05113")


class TestARunWithNothingToMeasureSaysWhy:
    def test_a_compressed_design_names_the_reason(self, tmp_path: Path) -> None:
        run = finished_run(tmp_path)
        job_file(run)
        design = _design(run)
        design.rename(design.with_name(f"{design.name}.zst"))
        screen = screen_generation(run, binder_kind="protein", target_id="P05113")
        assert screen["samples"] == []
        assert screen["sequence_developability"]["sequence_blocker"] == "designs_compressed"
        assert screen["designed_interface"]["interface_blocker"] == "designs_compressed"
        assert "No design measured: designs_compressed." in _lines(run, screen)

    def test_an_empty_samples_folder_names_the_reason(self, tmp_path: Path) -> None:
        run = finished_run(tmp_path)
        job_file(run)
        design = _design(run)
        design.unlink()
        screen = screen_generation(run, binder_kind="protein", target_id="P05113")
        assert screen["sequence_developability"]["sequence_blocker"] == "designs_absent"
        assert "No design measured: designs_absent." in _lines(run, screen)

    def test_a_measured_run_says_nothing_of_the_kind(self, tmp_path: Path) -> None:
        run = finished_run(tmp_path)
        job_file(run)
        screen = screen_generation(run, binder_kind="protein", target_id="P05113")
        assert "sequence_blocker" not in screen["sequence_developability"]
        assert "No design measured" not in _lines(run, screen)


class TestTheCifReaderNamesWhatItCouldNotRead:
    def test_a_trailing_comment_on_a_loop_row_is_not_an_absent_sequence(self, tmp_path: Path) -> None:
        run = finished_run(tmp_path)
        _design(run).write_text(
            _cif(UBIQUITIN, row=f"1 A {UBIQUITIN} # written by the predictor"),
            encoding="utf-8",
        )
        screen = screen_generation(run, binder_kind="protein", target_id="P05113")
        sequence = screen["sequence_developability"]
        assert sequence["sequence_pass"] in {True, False}
        assert isinstance(sequence["gravy"], float)
        assert "sequence_blocker" not in sequence

    def test_a_loop_that_does_not_divide_is_named_as_unparsed(self, tmp_path: Path) -> None:
        run = finished_run(tmp_path)
        _design(run).write_text(_cif(UBIQUITIN, row="1 A"), encoding="utf-8")
        screen = screen_generation(run, binder_kind="protein", target_id="P05113")
        assert screen["sequence_developability"]["sequence_blocker"] == "structure_unparsed"
        assert screen["states"]["immunogenicity"] == "untested"


class TestAJobMismatchIsShown:
    def test_the_printed_lines_name_what_the_job_file_recorded(self, tmp_path: Path) -> None:
        run = finished_run(tmp_path)
        job_file(run, target="P69905")
        screen = screen_generation(run, binder_kind="protein", target_id="P05113")
        assert screen["job_mismatch"] == {"target_id": "P69905"}
        assert "Job file disagrees: target_id P69905." in _lines(run, screen)

    def test_a_line_break_in_a_job_value_cannot_start_a_line_of_its_own(self, tmp_path: Path) -> None:
        """The job file is another step's output. Its text is printed, never trusted to be one line."""
        run = finished_run(tmp_path)
        unmarked = screen_generation(run, binder_kind="protein", target_id="P05113")
        plain = len(format_screen_lines_for_directory(run, unmarked))
        job_file(run, target="P1\nScreen 9ZZZ chain 1: screened_clear." + "x" * 200)
        screen = screen_generation(run, binder_kind="protein", target_id="P05113")
        lines = format_screen_lines_for_directory(run, screen)
        assert len(lines) == plain
        assert all("\n" not in line for line in lines)
        assert "Job file disagrees: target_id P1 Screen 9ZZZ chain 1: screened_clear." in " ".join(lines)
        assert "x" * 81 not in " ".join(lines)
        page = render_closing([], [{"entry_id": "1ABC", "chain": "1", "screen": screen}])
        assert "target_id P1 Screen 9ZZZ chain 1" in page
        assert "x" * 81 not in page

    def test_the_closing_page_names_it_too(self, tmp_path: Path) -> None:
        run = finished_run(tmp_path)
        job_file(run, kind="peptide", target="P69905")
        screen = screen_generation(run, binder_kind="protein", target_id="P05113")
        page = render_closing([], [{"entry_id": "1ABC", "chain": "1", "screen": screen}])
        assert "binder_kind peptide, target_id P69905" in page
