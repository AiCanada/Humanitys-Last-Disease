"""What a verdict was taken against, and what a curated call rests on.

Three gaps, all of the same kind: the screen made a judgement and did not keep
enough of it to be checked afterwards.

`cuts` already records every number a verdict was taken against, because
editing one of those constants later changes the meaning of every stored
screen. The bar is the same hazard and only its *name* was stored, so adding an
axis to `BARS["protein"]["sequence"]` made every stored `screened_clear`
silently start claiming more than it was ever checked for.

`BARS` declares thirteen named strategies and `judge_combination` has always
taken a `bar` argument, but nothing passed one: nine of the thirteen could not
be selected at all.

And some axes have no measured clear -- paralog's singleton cluster, a CpG-free
oligo -- so a curated clear is the only way they ever clear. `axis_evidence.json`
accepted a bare state and recorded no reason for it.
"""

import json
from pathlib import Path

import pytest

from rfproteina.disease.combinations import BARS
from rfproteina.disease.screen import screen_generation
from tests.helpers.disease import (
    UBIQUITIN,
    binder_cif,
    closing_page,
    no_developer_files,  # noqa: F401
)

_UBIQUITIN = UBIQUITIN


def _run(
    tmp_path: Path,
    name: str,
    *,
    kind: str = "protein",
    bar: str | None = None,
    evidence: object = None,
    sequence: str = _UBIQUITIN,
) -> dict:
    output = tmp_path / name
    folder = output / "inference_samples" / "binder_0"
    folder.mkdir(parents=True)
    (folder / "binder_0_generated.cif").write_text(binder_cif(sequence), encoding="utf-8")
    table = output / "inference_metrics" / "per_example_metrics.csv"
    table.parent.mkdir(parents=True)
    table.write_text("example_id,chain_of_interest\nbinder_0,A\n", encoding="utf-8")
    if evidence is not None:
        (output / "axis_evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    return screen_generation(output, binder_kind=kind, target_id="P05113", bar=bar)


class TestBarAxes:
    def test_a_screen_records_the_axes_its_bar_required(self, tmp_path: Path) -> None:
        record = _run(tmp_path, "plain")
        assert record["bar"] == "sequence"
        assert record["bar_axes"] == list(BARS["protein"]["sequence"])

    def test_the_list_is_the_one_that_was_applied(self, tmp_path: Path) -> None:
        """Read from BARS at the moment of the verdict, not written out by hand."""
        record = _run(tmp_path, "applied")
        assert record["bar_axes"] == [
            "immunogenicity",
            "proteome_offtarget",
            "mhc_i",
            "mhc_ii",
            "ada",
            "paralog",
        ]

    def test_a_chosen_bar_records_its_own_axes(self, tmp_path: Path) -> None:
        record = _run(tmp_path, "plasma", bar="plasma")
        assert record["bar"] == "plasma"
        assert record["bar_axes"] == ["plasma_abundance", "plasma_hub", "plasma_fold"]

    def test_a_binder_with_no_bar_records_no_requirement(self, tmp_path: Path) -> None:
        record = _run(tmp_path, "other", kind="other")
        assert record["bar_axes"] == "absent"


class TestChoosingABar:
    def test_a_ligand_can_be_screened_strictly(self, tmp_path: Path) -> None:
        """Eight axes rather than the three of the coverage default."""
        record = _run(tmp_path, "strict", kind="ligand", bar="strict")
        assert record["bar"] == "strict"
        assert record["bar_axes"] == list(BARS["ligand"]["strict"])
        assert len(record["bar_axes"]) == 8

    def test_the_default_is_still_the_kinds_own_bar(self, tmp_path: Path) -> None:
        assert _run(tmp_path, "ligand", kind="ligand")["bar"] == "coverage"
        assert _run(tmp_path, "nucleic", kind="nucleic", sequence="ATGCATGCATGCATGC")["bar"] == "oligo"

    def test_an_unknown_bar_is_refused_and_names_the_ones_that_exist(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no 'sequenc' bar"):
            _run(tmp_path, "typo", bar="sequenc")

    def test_another_kinds_bar_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no 'coverage' bar"):
            _run(tmp_path, "foreign", bar="coverage")

    def test_a_bad_bar_is_refused_before_anything_is_measured(self, tmp_path: Path) -> None:
        """A misspelling should not cost a proteome scan first."""
        output = tmp_path / "early"
        output.mkdir()
        with pytest.raises(ValueError, match="no 'nonsense' bar"):
            screen_generation(output, binder_kind="protein", target_id="P05113", bar="nonsense")
        assert not (output / "screen.json").exists()


class TestCuratedBasis:
    def test_the_bare_state_form_still_works(self, tmp_path: Path) -> None:
        record = _run(tmp_path, "bare", evidence={"paralog": "clear"})
        paralog = record["target_liability"]["paralog"]
        assert paralog["state"] == "clear"
        assert paralog["source"] == "axis_evidence"

    def test_a_curated_call_with_no_basis_says_so(self, tmp_path: Path) -> None:
        """A curated call with nothing behind it is a fact, not an absence to pass over."""
        record = _run(tmp_path, "bare", evidence={"paralog": "clear"})
        assert record["target_liability"]["paralog"]["curated"] == "basis_not_stated"

    def test_a_stated_reason_and_source_are_kept(self, tmp_path: Path) -> None:
        record = _run(
            tmp_path,
            "stated",
            evidence={
                "paralog": {
                    "state": "clear",
                    "reason": "No reviewed human paralog shares the binding face.",
                    "source": "UniProt 2026_03 review",
                }
            },
        )
        curated = record["target_liability"]["paralog"]["curated"]
        assert curated == {
            "reason": "No reviewed human paralog shares the binding face.",
            "source": "UniProt 2026_03 review",
        }
        assert record["target_liability"]["paralog"]["state"] == "clear"

    def test_the_closed_page_shows_the_basis(self, tmp_path: Path) -> None:
        record = _run(
            tmp_path,
            "stated",
            evidence={"paralog": {"state": "clear", "reason": "Reviewed by hand.", "source": "ticket RFD-412"}},
        )
        page = closing_page(record)
        assert "A curated call is recorded for this check." in page
        assert "The stated reason is Reviewed by hand." in page
        assert "The stated source is ticket RFD-412" in page

    def test_the_closed_page_says_when_there_is_no_basis(self, tmp_path: Path) -> None:
        page = closing_page(_run(tmp_path, "bare", evidence={"paralog": "clear"}))
        assert "No reason or source was recorded with it." in page

    @pytest.mark.parametrize(
        ("label", "payload", "message"),
        [
            ("a misspelled field", {"state": "clear", "resaon": "typo"}, "no 'resaon' field"),
            ("a state that is not one", {"state": "probably"}, "must be hit or clear"),
            ("no state at all", {"reason": "because"}, "must name a state"),
            ("a blank reason", {"state": "clear", "reason": "   "}, "reason must be text"),
            ("a reason past the bound", {"state": "clear", "reason": "x" * 400}, "longer than 300"),
            ("a reason that is not text", {"state": "clear", "source": 7}, "source must be text"),
        ],
    )
    def test_a_basis_that_cannot_be_trusted_is_refused(
        self, tmp_path: Path, label: str, payload: dict, message: str
    ) -> None:
        """A misspelled reason that is silently dropped leaves the record it was meant to prevent."""
        with pytest.raises(ValueError, match=message):
            _run(tmp_path, label.replace(" ", "_"), evidence={"paralog": payload})
