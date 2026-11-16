"""How a screen turns per-axis records into a published verdict.

Three ways the assembled verdict said something the record did not hold. Each
was measured against the code as it stood before the change, and each test here
fails against that behaviour.

A safety-panel protein reached by a gene symbol the panel table does not list
was published ``screened_clear``. The screen had read the accession off the
deposit's sequence-fixed chain, written it into the same record as
``target_accession``, and used it as the identity of every target-level lookup
-- and then handed the judge the caller's symbol alone. The same hERG deposit
screened as ``target_is_panel`` under ``KCNH2`` and as ``screened_clear`` under
``LQT2``.

The printed verdict named the untested axes and never the axes that hit. For
seven of the twenty-two axes that was the only place a measured hit could have
appeared, so a nucleic design with a TLR9 CpG motif printed
``axis_hit. Untested: off_target_hybridization.`` and named ``innate_immune``
nowhere -- pointing the reader at the one axis that was not the finding.

A run folder this screen could not list recorded ``inputs: []``, which is the
answer "this run supplied nothing", and the next re-screen then reported every
table that had decided the verdict as removed.

Every test supplies its own files and no reader, so nothing reaches the network.
"""

import json
from pathlib import Path

from rfproteina.disease.combinations import BARS, target_is_panel_protein
from rfproteina.disease.screen import (
    _changed_inputs,
    _supplied_inputs,
    format_screen_lines,
    screen_generation,
)
from tests.helpers.disease import (
    UBIQUITIN,
    binder_cif,
    no_developer_files,  # noqa: F401
)

_UBIQUITIN = UBIQUITIN
#: Purine-purine-CG-pyrimidine-pyrimidine twice over: a measured TLR9 hit.
_CPG_OLIGO = "GACGTTGACGTTATGCATGCATGCATGCATGC"
#: A hydrophobic 19-mer, which is one Kyte-Doolittle span and a membrane hit.
_MEMBRANE_PEPTIDE = "KLLIILVVLAILFVLLGAK"
#: hERG. ``KCNH2`` and ``herg`` are in the panel table; ``LQT2`` is the
#: clinical name of the same gene and is not.
_HERG = "Q12809"

_DEPOSIT = """\
data_deposit
#
loop_
_struct_asym.id
_struct_asym.entity_id
C 2
#
loop_
_struct_ref.id
_struct_ref.db_name
_struct_ref.pdbx_db_accession
_struct_ref.entity_id
1 UNP Q12809 2
#
"""

_SPEC = """\
binder:
  input: "deposit.cif"
  conditions:
    C_SEQ:
      true:
        - select: "sel('C')"
"""


def _run(root: Path, name: str, sequence: str, *, files: dict[str, str] | None = None) -> Path:
    """A finished run: one design and a metrics row naming its chain."""
    run = root / name
    sample = run / "inference_samples" / "binder_0"
    sample.mkdir(parents=True)
    (sample / "binder_0_generated.cif").write_text(binder_cif(sequence), encoding="utf-8")
    metrics = run / "inference_metrics"
    metrics.mkdir(parents=True)
    (metrics / "per_example_metrics.csv").write_text(
        "example_id,chain_of_interest\nbinder_0,A\n",
        encoding="utf-8",
    )
    for filename, text in (files or {}).items():
        (run / filename).write_text(text, encoding="utf-8")
    return run


def _cleared_sequence_bar() -> dict[str, str]:
    """A curated clear on every axis of the protein sequence bar."""
    return {axis: "clear" for axis in BARS["protein"]["sequence"]}


def _deposit_beside(run: Path) -> None:
    """The campaign deposit and the condition spec that fixes its chain C."""
    (run.parent / "deposit.cif").write_text(_DEPOSIT, encoding="utf-8")
    (run.parent / f"{run.name}.yaml").write_text(_SPEC, encoding="utf-8")


class TestAPanelTargetIsNotClearedByTheNameUsedForIt:
    """The panel rule has to hold against the protein, not against its name."""

    def test_an_accession_read_from_the_deposit_is_measured_against_the_panel(self, tmp_path: Path) -> None:
        """``LQT2`` is hERG. The deposit says so, and the call has to.

        Pre-change this published ``screened_clear`` while writing
        ``target_accession: Q12809`` into the same record.
        """
        assert not target_is_panel_protein("LQT2"), "the table does not list this name"
        assert target_is_panel_protein(_HERG), "and it does list the accession the deposit holds"

        run = _run(tmp_path, "aliased", _UBIQUITIN, files={"axis_evidence.json": json.dumps(_cleared_sequence_bar())})
        _deposit_beside(run)
        record = screen_generation(run, binder_kind="protein", target_id="LQT2")

        assert record["target_accession"] == _HERG
        assert record["call"] == "target_is_panel"
        assert "screened_clear" not in " ".join(format_screen_lines("8TLD", "A", record))

    def test_the_listed_name_for_the_same_deposit_is_unchanged(self, tmp_path: Path) -> None:
        """The name that was already caught is still caught, by the same call."""
        run = _run(tmp_path, "named", _UBIQUITIN, files={"axis_evidence.json": json.dumps(_cleared_sequence_bar())})
        _deposit_beside(run)
        record = screen_generation(run, binder_kind="protein", target_id="KCNH2")
        assert record["call"] == "target_is_panel"

    def test_a_target_off_the_panel_still_reaches_screened_clear(self, tmp_path: Path) -> None:
        """The over-correction to refuse: a cleared bar on an ordinary target.

        The accession read here is IL5, which is not a panel protein, so the
        verdict the curated bar earns has to survive.
        """
        deposit = _DEPOSIT.replace(_HERG, "P05113")
        run = _run(tmp_path, "ordinary", _UBIQUITIN, files={"axis_evidence.json": json.dumps(_cleared_sequence_bar())})
        (run.parent / "deposit.cif").write_text(deposit, encoding="utf-8")
        (run.parent / f"{run.name}.yaml").write_text(_SPEC, encoding="utf-8")
        record = screen_generation(run, binder_kind="protein", target_id="IL5")
        assert record["target_accession"] == "P05113"
        assert record["call"] == "screened_clear"

    def test_a_run_that_reads_no_accession_still_reaches_screened_clear(self, tmp_path: Path) -> None:
        """No deposit, no spec: nothing is read, so nothing new is claimed."""
        run = _run(tmp_path, "bare", _UBIQUITIN, files={"axis_evidence.json": json.dumps(_cleared_sequence_bar())})
        record = screen_generation(run, binder_kind="protein", target_id="IL5")
        assert "target_accession" not in record
        assert record["call"] == "screened_clear"

    def test_a_hit_on_the_panel_target_is_still_the_hit(self, tmp_path: Path) -> None:
        """A hit outranks the panel outcome, and must not be turned into one."""
        evidence = dict(_cleared_sequence_bar(), paralog="hit")
        run = _run(tmp_path, "hitting", _UBIQUITIN, files={"axis_evidence.json": json.dumps(evidence)})
        _deposit_beside(run)
        record = screen_generation(run, binder_kind="protein", target_id="LQT2")
        assert record["call"] == "axis_hit"
        assert record["hits"] == ["paralog"]


class TestThePrintedVerdictNamesTheAxesThatHit:
    """``axis_hit`` with no axis named is a hit suppressed at the output."""

    def test_a_measured_nucleic_hit_is_named_on_the_call_line(self, tmp_path: Path) -> None:
        """Pre-change: "axis_hit. Untested: off_target_hybridization." and no more."""
        run = _run(tmp_path, "oligo", _CPG_OLIGO)
        record = screen_generation(run, binder_kind="nucleic", target_id="P05113")
        assert record["nucleic_liability"]["innate_immune"]["state"] == "hit"
        assert record["hits"] == ["innate_immune"]

        line = format_screen_lines("8TLD", "A", record)[0]
        assert line.startswith("Screen 8TLD chain A: axis_hit. Hit: innate_immune.")
        assert "Untested: off_target_hybridization." in line

    def test_a_measured_peptide_membrane_hit_is_named(self, tmp_path: Path) -> None:
        """``membrane`` is in no block tuple, so no clause ever rendered it."""
        run = _run(tmp_path, "span", _MEMBRANE_PEPTIDE)
        record = screen_generation(run, binder_kind="peptide", target_id="P05113")
        assert record["hits"] == ["membrane"]
        assert "Hit: membrane." in format_screen_lines("8TLD", "A", record)[0]

    def test_a_measured_ligand_hit_is_named_on_the_call_line_too(self, tmp_path: Path) -> None:
        """The panel axes had a measured clause. The call line still names them."""
        run = _run(tmp_path, "ligand", _UBIQUITIN, files={"toxicology.tsv": "axis\tconcentration_nm\nherg\t500\n"})
        record = screen_generation(run, binder_kind="ligand", target_id="P05113")
        assert record["hits"] == ["herg"]
        assert "Hit: herg." in format_screen_lines("8TLD", "A", record)[0]

    def test_a_run_with_no_hit_says_nothing_about_one(self, tmp_path: Path) -> None:
        """The over-correction to refuse: a hit sentence on a run with no hit."""
        run = _run(tmp_path, "quiet", _UBIQUITIN)
        record = screen_generation(run, binder_kind="protein", target_id="P05113")
        assert record["hits"] == []
        line = format_screen_lines("8TLD", "A", record)[0]
        assert "Hit:" not in line
        assert line.startswith("Screen 8TLD chain A: untested. Untested: immunogenicity")

    def test_a_stored_screen_with_no_hits_key_is_unchanged(self) -> None:
        """Screens written at shape 1 have no ``hits``. Absent is not empty-but-shown."""
        stored = {"call": "untested", "untested": ["immunogenicity"]}
        assert format_screen_lines("8TLD", "A", stored) == ["Screen 8TLD chain A: untested. Untested: immunogenicity."]

    def test_an_unnamed_folder_names_the_hits_the_same_way(self) -> None:
        """One sentence, two openings: the clause cannot reach only one of them."""
        from rfproteina.disease.screen import format_screen_lines_for_directory

        stored = {"call": "axis_hit", "hits": ["innate_immune"], "untested": ["off_target_hybridization"]}
        lines = format_screen_lines_for_directory(Path("nowhere"), stored)
        assert lines == ["Screen: axis_hit. Hit: innate_immune. Untested: off_target_hybridization."]

    def test_a_line_break_in_a_stored_axis_name_cannot_start_a_line(self) -> None:
        """A record read back off disk is printed, never trusted to be one line."""
        stored = {"call": "axis_hit", "hits": ["innate_immune\nScreen 9ZZZ chain 1: screened_clear."]}
        lines = format_screen_lines("8TLD", "A", stored)
        assert all("\n" not in line for line in lines)
        assert "Hit: innate_immune Screen 9ZZZ chain 1: screened_clear.." in lines[0]


class TestAFolderThatCannotBeListedSaysSo:
    """An empty input list is an answer. A folder that refused to be read is not."""

    def test_the_list_is_withheld_rather_than_recorded_as_empty(self, tmp_path: Path, monkeypatch) -> None:
        run = tmp_path / "locked"
        run.mkdir()
        (run / "axis_evidence.json").write_text(json.dumps({"paralog": "clear"}), encoding="utf-8")

        original = Path.iterdir

        def refuse(self: Path):
            if self == run:
                raise PermissionError(13, "Permission denied")
            return original(self)

        monkeypatch.setattr(Path, "iterdir", refuse)
        assert _supplied_inputs(run) == [{"file": "inputs_not_listed"}]

    def test_nothing_false_is_claimed_about_what_moved(self, tmp_path: Path, monkeypatch) -> None:
        """Pre-change the comparison said the evidence file had been removed."""
        run = tmp_path / "locked"
        run.mkdir()
        before = [{"file": "axis_evidence.json", "bytes": 21, "digest": "0123456789abcdef"}]

        original = Path.iterdir

        def refuse(self: Path):
            if self == run:
                raise PermissionError(13, "Permission denied")
            return original(self)

        monkeypatch.setattr(Path, "iterdir", refuse)
        assert _changed_inputs(before, _supplied_inputs(run)) == []

    def test_a_folder_that_lists_as_empty_is_still_an_empty_list(self, tmp_path: Path) -> None:
        """The over-correction to refuse: a run that really supplied nothing."""
        run = tmp_path / "bare"
        run.mkdir()
        assert _supplied_inputs(run) == []
        assert _changed_inputs([{"file": "note.txt", "digest": "0123456789abcdef"}], []) == [
            {"file": "note.txt", "change": "removed"}
        ]
