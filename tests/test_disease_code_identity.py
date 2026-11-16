"""Which code produced a verdict.

A verdict is reproducible against four things: its inputs, the cuts it was
taken against, the bar it had to clear, and the code that applied them. The
screen recorded the first three. It did not record the fourth, so two screens
agreeing on every stored field could still disagree about the designs with
nothing on disk saying why.

The gap became visible once `replaced` learned to name which supplied files
moved: a re-screen that changes the verdict while reporting no changed inputs
is pointing at the code, and the record could not say which code.

The same helper the research traces use, now shared rather than written twice,
because two copies are how the two records come to disagree about one checkout.
"""

import json
from pathlib import Path

import pytest

from rfproteina.disease.provenance import NOT_RECORDED, code_version, head_commit
from rfproteina.disease.screen import _INPUT_LIMIT
from tests.helpers.disease import (
    UBIQUITIN,
    no_developer_files,  # noqa: F401
    screen_protein_run,
)

_ONE = UBIQUITIN


def _build(root: Path, name: str, *, extra: int = 0, files: dict[str, str] | None = None) -> Path:
    output = root / name
    folder = output / "inference_samples" / "binder_0"
    folder.mkdir(parents=True)
    (folder / "binder_0_generated.cif").write_text(
        "data_binder\n#\nloop_\n_entity_poly.entity_id\n_entity_poly.pdbx_strand_id\n"
        f"_entity_poly.pdbx_seq_one_letter_code_can\n1 A {_ONE}\n#\n",
        encoding="utf-8",
    )
    table = output / "inference_metrics" / "per_example_metrics.csv"
    table.parent.mkdir(parents=True)
    table.write_text("example_id,chain_of_interest\nbinder_0,A\n", encoding="utf-8")
    for filename, text in (files or {}).items():
        (output / filename).write_text(text, encoding="utf-8")
    for index in range(extra):
        (output / f"note_{index:03d}.txt").write_text(str(index), encoding="utf-8")
    return output


class TestCodeOnTheScreen:
    def test_a_screen_names_the_code_that_produced_it(self, tmp_path: Path) -> None:
        code = screen_protein_run(_build(tmp_path, "plain"))["code"]
        assert len(code["commit"]) in (40, 64) or code["commit"] == NOT_RECORDED
        assert code["tree_state"] == "not_checked"

    def test_it_is_the_same_answer_the_traces_record(self, tmp_path: Path) -> None:
        """Written twice is how the two records come to disagree about one checkout."""
        assert screen_protein_run(_build(tmp_path, "same"))["code"] == code_version()

    def test_the_tree_state_admits_it_was_not_checked(self, tmp_path: Path) -> None:
        """A commit recorded as though it described the working tree would be worse."""
        assert screen_protein_run(_build(tmp_path, "honest"))["code"]["tree_state"] == "not_checked"

    def test_a_verdict_that_changed_with_no_input_moved_still_carries_the_code(self, tmp_path: Path) -> None:
        output = _build(tmp_path, "moved", files={"axis_evidence.json": json.dumps({"paralog": "clear"})})
        first = screen_protein_run(output)
        (output / "axis_evidence.json").write_text(json.dumps({"paralog": "hit"}), encoding="utf-8")
        second = screen_protein_run(output)
        assert second["replaced"]["changed_inputs"][0]["file"] == "axis_evidence.json"
        assert second["code"]["commit"] == first["code"]["commit"]


class TestHeadCommit:
    """Read out of .git without running git, so a screen starts no subprocess."""

    def test_a_ref_is_followed(self, tmp_path: Path) -> None:
        git = tmp_path / "git"
        (git / "refs" / "heads").mkdir(parents=True)
        (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (git / "refs" / "heads" / "main").write_text("a" * 40 + "\n", encoding="utf-8")
        assert head_commit(git) == "a" * 40

    def test_a_detached_head_answers(self, tmp_path: Path) -> None:
        git = tmp_path / "git"
        git.mkdir()
        (git / "HEAD").write_text("b" * 40 + "\n", encoding="utf-8")
        assert head_commit(git) == "b" * 40

    def test_a_packed_ref_answers(self, tmp_path: Path) -> None:
        git = tmp_path / "git"
        git.mkdir()
        (git / "HEAD").write_text("ref: refs/heads/packed\n", encoding="utf-8")
        (git / "packed-refs").write_text("# comment\n" + "c" * 40 + " refs/heads/packed\n", encoding="utf-8")
        assert head_commit(git) == "c" * 40

    def test_a_sha256_repository_answers(self, tmp_path: Path) -> None:
        git = tmp_path / "git"
        git.mkdir()
        (git / "HEAD").write_text("d" * 64 + "\n", encoding="utf-8")
        assert head_commit(git) == "d" * 64

    @pytest.mark.parametrize("head", ["ref: refs/heads/missing\n", "ref:\n", "not a commit\n"])
    def test_what_cannot_be_resolved_answers_nothing(self, tmp_path: Path, head: str) -> None:
        git = tmp_path / "git"
        git.mkdir()
        (git / "HEAD").write_text(head, encoding="utf-8")
        assert head_commit(git) == ""

    def test_a_folder_with_no_git_answers_nothing(self, tmp_path: Path) -> None:
        assert head_commit(tmp_path / "absent") == ""


class TestInputListBound:
    def test_a_folder_at_the_limit_is_listed_in_full(self, tmp_path: Path) -> None:
        inputs = screen_protein_run(_build(tmp_path, "under", extra=_INPUT_LIMIT - 1))["inputs"]
        assert len(inputs) == _INPUT_LIMIT - 1
        assert all(item["file"] != "inputs_truncated" for item in inputs)

    def test_past_it_the_list_is_withheld_whole_rather_than_cut(self, tmp_path: Path) -> None:
        """A list truncated in silence reads as a complete one."""
        output = _build(tmp_path, "over", extra=_INPUT_LIMIT + 5)
        assert screen_protein_run(output)["inputs"] == [{"file": "inputs_truncated", "files": _INPUT_LIMIT + 5}]

    def test_nothing_false_is_claimed_about_what_moved(self, tmp_path: Path) -> None:
        """Saying everything moved is the alternative to saying nothing."""
        output = _build(tmp_path, "over", extra=_INPUT_LIMIT + 5)
        first = screen_protein_run(output)
        (output / "note_000.txt").write_text("changed", encoding="utf-8")
        (output / "axis_evidence.json").write_text(json.dumps({"paralog": "clear"}), encoding="utf-8")
        second = screen_protein_run(output)
        assert "changed_inputs" not in second["replaced"]
        assert second["replaced"]["call"] == first["call"]
