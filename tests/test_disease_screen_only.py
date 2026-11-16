"""Re-screening a finished run, which nothing could do before.

The tables the screen reads are produced after a run finishes: the user scores
the designed chain with a local predictor and drops the result in. Until this
path existed the screen ran once, in the seconds after the generate subprocess
returned, so anything supplied afterwards was never read by anything.

Every test here supplies its own reader or none at all, so nothing reaches the
network.
"""

import json
import os
from pathlib import Path

import pytest

from rfproteina.disease.cli import main as cure_main
from rfproteina.disease.research import LookupMissError
from rfproteina.disease.screen import (
    base_name,
    generation_directories,
    job_file_for_directory,
    read_screen,
    screen_lines_for_directory,
)
from rfproteina.disease.ui import wants_option_ui
from tests.helpers.disease import (
    finished_run,
    job_file,
    no_developer_proteome,  # noqa: F401
)

_PRESENTED = "MQIFVKTLT"


def _screen_of(run: Path) -> dict:
    return json.loads((run / "screen.json").read_text(encoding="utf-8"))


def test_a_finished_run_is_rescreened_from_tables_supplied_afterwards(tmp_path, capsys):
    """The acceptance test: the whole reason this path exists.

    A class I table scored after the run is what restricts the immunogenicity
    scan. Before this there was no way to make the screen read it.
    """
    run = finished_run(tmp_path)
    job_file(run)

    assert cure_main(["--screen", str(run)]) == 0
    first = _screen_of(run)
    assert first["sequence_liability"]["immunogenicity"]["presentation"] == "not_configured"

    # The predictor is run, and its output is dropped in beside the designs.
    (run / "mhc_i.tsv").write_text(
        f"allele\tpeptide\tpercentile_rank\nHLA-A*02:01\t{_PRESENTED}\t0.4\n",
        encoding="utf-8",
    )
    before = sorted(path.name for path in (run / "inference_samples").rglob("*"))
    assert cure_main(["--screen", str(run)]) == 0
    second = _screen_of(run)

    immunogenicity = second["sequence_liability"]["immunogenicity"]
    assert immunogenicity["presentation"] == "restricted", "the table was read this time"
    assert immunogenicity["presented_ligands"] == 1
    assert immunogenicity["state"] != first["sequence_liability"]["immunogenicity"]["state"]
    # Nothing was generated: the designs are untouched.
    assert sorted(path.name for path in (run / "inference_samples").rglob("*")) == before


def test_screening_prints_the_same_sentences_the_stored_screen_renders(tmp_path, capsys):
    """One renderer, so a screen written now reads back identically."""
    run = finished_run(tmp_path)
    job_file(run)
    assert cure_main(["--screen", str(run)]) == 0
    printed = [line for line in capsys.readouterr().out.splitlines() if line.startswith("Screen ")]
    assert printed == [line for line in screen_lines_for_directory(run) if line.startswith("Screen ")]


def test_screening_twice_writes_the_same_bytes(tmp_path):
    """Without a lookup there is nothing non-deterministic, so it is a no-op."""
    run = finished_run(tmp_path)
    job_file(run)
    assert cure_main(["--screen", str(run)]) == 0
    first = (run / "screen.json").read_bytes()
    assert cure_main(["--screen", str(run)]) == 0
    assert (run / "screen.json").read_bytes() == first


def test_an_unreadable_screen_is_not_replaced(tmp_path, capsys):
    """A screen that cannot be read is evidence; overwriting it destroys that."""
    run = finished_run(tmp_path)
    job_file(run)
    planted = b"\xff\xfe not a screen"
    (run / "screen.json").write_bytes(planted)
    assert cure_main(["--screen", str(run)]) == 1
    assert (run / "screen.json").read_bytes() == planted
    assert "was not replaced" in capsys.readouterr().out


def test_a_failed_write_leaves_the_previous_screen_in_place(tmp_path, monkeypatch):
    """The atomic write, asserted rather than assumed."""
    run = finished_run(tmp_path)
    job_file(run)
    assert cure_main(["--screen", str(run)]) == 0
    first = (run / "screen.json").read_bytes()

    def refuse(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("rfproteina.disease.publish.os.replace", refuse)
    assert cure_main(["--screen", str(run)]) == 1
    assert (run / "screen.json").read_bytes() == first, "the old verdict survived"
    assert not list(run.glob("*.tmp")), "no temporary file was left behind"


def test_a_directory_with_no_designs_is_refused(tmp_path, capsys):
    """Nothing to measure is a usage error, not an empty verdict."""
    empty = tmp_path / "nothing"
    empty.mkdir()
    assert cure_main(["--screen", str(empty)]) == 0
    assert "No finished generation" in capsys.readouterr().out
    assert not (empty / "screen.json").exists()


def test_a_directory_that_names_no_kind_is_refused(tmp_path, capsys):
    """The kind decides which axes exist, so guessing it writes a false verdict."""
    run = finished_run(tmp_path)
    assert cure_main(["--screen", str(run)]) == 1
    assert "nothing names the binder kind" in capsys.readouterr().out
    assert not (run / "screen.json").exists()


def test_the_binder_kind_can_be_supplied_for_a_hand_made_directory(tmp_path):
    run = finished_run(tmp_path)
    assert cure_main(["--screen", str(run), "--binder-kind", "protein"]) == 0
    assert "sequence_liability" in _screen_of(run)


def test_generation_json_outranks_a_disagreeing_job_file(tmp_path):
    """The archive wrote it from the structure that drove the original screen."""
    run = finished_run(tmp_path)
    job_file(run, kind="peptide", target="P00000")
    (run / "generation.json").write_text(
        json.dumps({"binder_kind": "protein", "target_id": "P05113", "entry_id": "1ABC", "chain": "1"}),
        encoding="utf-8",
    )
    assert cure_main(["--screen", str(run)]) == 0
    record = _screen_of(run)
    assert record["bar"] == "sequence", "the protein bar, from generation.json"
    assert record["job_mismatch"] == {"binder_kind": "peptide", "target_id": "P00000"}


def test_a_run_that_names_no_target_still_screens_the_design_axes(tmp_path, capsys):
    run = finished_run(tmp_path)
    job_file(run, target="")
    assert cure_main(["--screen", str(run)]) == 0
    assert "names no target" in capsys.readouterr().out
    assert "sequence_liability" in _screen_of(run)


@pytest.mark.parametrize("inside", [True, False])
def test_the_job_file_is_found_beside_a_live_run_and_inside_an_archived_one(tmp_path, inside):
    """Looking only beside meant this never fired for anything archived."""
    run = finished_run(tmp_path)
    job_file(run, kind="peptide", inside=inside)
    assert job_file_for_directory(run) is not None
    assert cure_main(["--screen", str(run), "--binder-kind", "protein"]) == 0
    assert _screen_of(run)["job_mismatch"] == {"binder_kind": "peptide"}


def test_an_archive_collision_suffix_still_finds_its_job_file(tmp_path):
    """save_generation appends -2 and the copied files keep the original stem."""
    run = finished_run(tmp_path, "gene_1ABC_1-2")
    assert base_name(run) == "gene_1ABC_1"
    (run / "gene_1ABC_1.job.json").write_text(
        json.dumps({"binder_kind": "peptide", "target_id": "P05113"}),
        encoding="utf-8",
    )
    assert job_file_for_directory(run) == run / "gene_1ABC_1.job.json"
    assert cure_main(["--screen", str(run), "--binder-kind", "protein"]) == 0
    assert _screen_of(run)["job_mismatch"] == {"binder_kind": "peptide"}


def test_a_directory_with_no_job_file_is_not_a_disagreement(tmp_path):
    run = finished_run(tmp_path)
    assert job_file_for_directory(run) is None
    assert cure_main(["--screen", str(run), "--binder-kind", "protein"]) == 0
    assert "job_mismatch" not in _screen_of(run)


def test_a_tree_is_walked_and_every_run_beneath_is_screened(tmp_path, capsys):
    root = tmp_path / "generations" / "asthma"
    for target, name in (("IL5", "gene_1ABC_1"), ("IL5", "gene_2DEF_1"), ("IL33", "gene_3GHI_1")):
        run = finished_run(root / target, name)
        job_file(run)
    assert cure_main(["--screen", str(tmp_path / "generations")]) == 0
    written = sorted(path.parent.name for path in (tmp_path / "generations").rglob("screen.json"))
    assert written == ["gene_1ABC_1", "gene_2DEF_1", "gene_3GHI_1"]
    out = capsys.readouterr().out
    assert "asthma/IL5/gene_1ABC_1" in out, "each block carries its relative label"
    assert "asthma/IL33/gene_3GHI_1" in out


def test_a_run_given_directly_is_the_only_one_screened(tmp_path):
    """The own-first rule the stored screens already follow."""
    outer = finished_run(tmp_path, "outer")
    job_file(outer)
    inner = finished_run(outer, "inner")
    job_file(inner)
    assert generation_directories(outer) == [outer]
    assert cure_main(["--screen", str(outer)]) == 0
    assert (outer / "screen.json").is_file()
    assert not (inner / "screen.json").exists()


def test_the_walk_does_not_descend_into_the_designs(tmp_path):
    root = tmp_path / "tree"
    run = finished_run(root)
    job_file(run)
    decoy = run / "inference_samples" / "decoy"
    (decoy / "inference_metrics").mkdir(parents=True)
    assert cure_main(["--screen", str(root)]) == 0
    assert not (decoy / "screen.json").exists()


def test_a_truncated_search_says_so_and_does_not_report_success(tmp_path, monkeypatch, capsys):
    """Stopping early must never read as having found everything.

    Silence here meant a run under the named directory went unscreened while
    the output said none existed and the status said success.
    """
    monkeypatch.setattr("rfproteina.disease.screen.SCREEN_RUN_LIMIT", 1)
    root = tmp_path / "tree"
    for name in ("gene_1ABC_1", "gene_2DEF_1"):
        job_file(finished_run(root, name))
    assert cure_main(["--screen", str(root)]) == 1, "a cut-short search is not a success"
    out = capsys.readouterr().out
    assert "may be unscreened" in out
    assert len(list(root.rglob("screen.json"))) == 1, "what was found is still screened"


def test_a_run_hidden_behind_the_visit_cap_is_not_reported_as_absent(tmp_path, monkeypatch, capsys):
    """The visit cap had no warning at all, unlike the run cap."""
    monkeypatch.setattr("rfproteina.disease.screen._SCREEN_WALK_CAP", 3)
    root = tmp_path / "tree"
    for index in range(6):
        (root / f"empty{index}").mkdir(parents=True)
    job_file(finished_run(root, "gene_1ABC_1"))
    assert cure_main(["--screen", str(root)]) == 1
    out = capsys.readouterr().out
    assert "The search stopped after" in out
    assert "No finished generation was found" not in out or "may be unscreened" in out


def test_the_panel_and_a_screen_are_not_asked_for_together(tmp_path):
    """argparse refuses it, because the panel would otherwise drop the flag.

    Every spelling has to be refused, not just the bare token: the panel
    rebuilds the command from its own form, which has no field for this flag,
    so a spelling that slipped past the gate would open the panel and then lose
    the flag entirely.
    """
    for spelled in (["--screen", "somewhere"], ["--screen=somewhere"], ["--ui", "--screen=x"]):
        assert wants_option_ui(spelled) is False, spelled
    with pytest.raises(SystemExit):
        cure_main(["--ui", "--screen", str(tmp_path)])
    with pytest.raises(SystemExit):
        cure_main(["--ui", f"--screen={tmp_path}"])


def test_two_archived_runs_do_not_share_one_follow_up_root(tmp_path, capsys):
    """The collision the archive's own -N suffix creates.

    ``base_name`` strips that suffix, so deriving the follow-up root from it
    sent ``gene_1ABC_1`` and ``gene_1ABC_1-2`` to one directory. Both have one
    failed ``binder_0``, so the second overwrote the first's spec and the
    surviving plan re-folded the other run's design.
    """
    archive = tmp_path / "generations" / "asthma" / "IL5"
    for name in ("gene_1ABC_1", "gene_1ABC_1-2"):
        run = finished_run(archive, name, sequence="PQS" * 8)
        job_file(run)
        (run / f"{base_name(run)}.yaml").write_text('binder:\n  contig: "60-110"\n', encoding="utf-8")
    out = tmp_path / "out"
    roots: list[Path] = []

    def record(_run, _screen, *, fresh_root):
        roots.append(fresh_root)
        return []

    import rfproteina.disease.cli as cli

    original = cli._record_follow_up
    cli._record_follow_up = record
    try:
        assert cure_main(["--screen", str(tmp_path / "generations"), "--follow-up", "--out-dir", str(out)]) == 0
    finally:
        cli._record_follow_up = original

    assert len(roots) == 2, roots
    assert len(set(roots)) == 2, f"both runs were sent to one directory: {roots}"
    assert roots[0] == out / "follow_up" / "asthma" / "IL5" / "gene_1ABC_1"
    assert roots[1] == out / "follow_up" / "asthma" / "IL5" / "gene_1ABC_1-2"


def test_a_tree_asks_each_target_once(tmp_path):
    """An archive holds several runs per target; the lookups are not repeated."""
    root = tmp_path / "generations" / "asthma" / "IL5"
    for name in ("gene_1ABC_1", "gene_2DEF_1", "gene_3GHI_1"):
        job_file(finished_run(root, name))
    asked: list[str] = []

    def read_json(url: str) -> object:
        asked.append(url)
        raise LookupMissError(url)

    assert (
        cure_main(
            ["--screen", str(tmp_path / "generations"), "--lookup-targets"],
            read_json=read_json,
        )
        == 0
    )
    assert asked, "the reader was used"
    assert len(asked) == len(set(asked)), f"a URL was asked twice: {asked}"


def test_a_definite_miss_is_remembered_and_a_failure_is_not(tmp_path):
    """Remembering one unavailable moment would report the whole tree unavailable."""
    root = tmp_path / "tree"
    for name in ("gene_1ABC_1", "gene_2DEF_1"):
        job_file(finished_run(root, name))
    asked: list[str] = []

    def read_json(url: str) -> object:
        asked.append(url)
        if "uniref" in url:
            raise LookupMissError(url)
        raise ValueError("service unavailable")

    assert cure_main(["--screen", str(root), "--lookup-targets"], read_json=read_json) == 0
    uniref = [url for url in asked if "uniref" in url]
    failing = [url for url in asked if "uniref" not in url]
    assert len(uniref) == len(set(uniref)), "a miss is a definite answer and is remembered"
    assert len(failing) > len(set(failing)), "a failure is retried for the next run"


def test_read_screen_tells_absent_from_unreadable(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    assert read_screen(run) is None
    (run / "screen.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not a readable screen"):
        read_screen(run)
    # The panel's readout treats both as nothing to show, unchanged.
    assert screen_lines_for_directory(run) == []


def test_the_atomic_write_leaves_no_temporary_file(tmp_path):
    from rfproteina.disease.publish import write_text_atomically

    target = tmp_path / "out.json"
    write_text_atomically(target, "first\n")
    assert target.read_text(encoding="utf-8") == "first\n"
    write_text_atomically(target, "second\n")
    assert target.read_text(encoding="utf-8") == "second\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["out.json"]
    assert f"out.json.{os.getpid()}.tmp" not in [path.name for path in tmp_path.iterdir()]


def test_two_designs_sharing_a_binder_token_get_distinct_ids(tmp_path):
    """One id for two molecules let a proteome hit be overwritten by a clear.

    A run pooling designs under ``run_a/binder_0`` and ``run_b/binder_0`` named
    both ``binder_0``, so one axis record stood for two chains: ``failed`` came
    back empty and ``axis_hit`` became ``screened_clear``. Adding a design
    removed a hit.
    """
    from rfproteina.disease.screen import sample_cifs

    run = finished_run(tmp_path)
    second = run / "inference_samples" / "run_b" / "binder_0"
    second.mkdir(parents=True)
    (second / "binder_0_generated.cif").write_text(
        "data_design\n#\nloop_\n_entity_poly.entity_id\n"
        "_entity_poly.pdbx_strand_id\n_entity_poly.pdbx_seq_one_letter_code_can\n"
        "1 A WWWWWWWWWWWWWWWWWWWW\n#\n",
        encoding="utf-8",
    )
    found = sample_cifs(run)
    assert len(found) == 2, "two CIFs must not collapse into one sample"
    assert len(set(found.values())) == 2


def test_a_dotted_example_id_cannot_leave_its_directory(tmp_path):
    """``..`` matched the safe-id pattern and is used as a path component."""
    from rfproteina.disease.screen import _SAFE_EXAMPLE_ID

    assert _SAFE_EXAMPLE_ID.fullmatch("binder_0")
    assert _SAFE_EXAMPLE_ID.fullmatch("sample.1-a")
    assert not _SAFE_EXAMPLE_ID.fullmatch("..")
    assert not _SAFE_EXAMPLE_ID.fullmatch(".")
    assert not _SAFE_EXAMPLE_ID.fullmatch(".hidden")


def test_a_curated_clear_is_printed_as_curated(tmp_path):
    """A declared state printed nothing, so a hidden hit read as a clean screen.

    The record kept ``measured_state``; the only output a human reads did not.
    """
    from rfproteina.disease.screen import format_screen_lines

    screen = {
        "call": "screened_clear",
        "bar": "sequence",
        "untested": [],
        "hits": [],
        "samples": [],
        "sequence_liability": {
            "immunogenicity": {
                "state": "clear",
                "measurement": 0.39963,
                "blocker": None,
                "source": "axis_evidence",
                "measured_state": "hit",
            }
        },
    }
    lines = " ".join(format_screen_lines("e", "A", screen))
    assert "immunogenicity" in lines, "the axis must still be named"
    assert "axis_evidence" in lines, "a declared state must not read as measured"
    assert "hit" in lines, "the measurement it overrode must survive"


def test_a_stray_root_screen_does_not_hide_the_runs_beneath(tmp_path):
    """The own-first shortcut needs the folder to BE a run.

    A campaign root holding a copied screen.json returned that one file as the
    only report and hid every run under it, hit included.
    """
    from rfproteina.disease.screen import screen_reports_for_directory

    root = tmp_path / "campaign"
    root.mkdir()
    run = finished_run(root, "gene_1ABC_1")
    (run / "screen.json").write_text(
        json.dumps({"call": "axis_hit", "untested": [], "hits": ["mhc_i"], "samples": []}),
        encoding="utf-8",
    )
    # The root is not a run: no inference_samples and no metrics of its own.
    (root / "screen.json").write_text(
        json.dumps({"call": "untested", "untested": [], "hits": [], "samples": []}),
        encoding="utf-8",
    )
    reports = screen_reports_for_directory(root)
    labels = [label for label, _lines in reports]
    assert "gene_1ABC_1" in labels, f"the run beneath must still be reported, got {labels}"


def test_a_truncated_report_page_says_so(tmp_path):
    """Silence at the cap read as "this is all of them"."""
    from rfproteina.disease.screen import screen_reports_for_directory

    root = tmp_path / "many"
    root.mkdir()
    for index in range(4):
        run = finished_run(root, f"gene_{index}ABC_1")
        (run / "screen.json").write_text(
            json.dumps({"call": "untested", "untested": [], "hits": [], "samples": []}),
            encoding="utf-8",
        )
    reports = screen_reports_for_directory(root, limit=2)
    joined = " ".join(line for _label, lines in reports for line in lines)
    assert "more screens than the 2 shown" in joined, f"a truncated page must say so, got {joined}"
    # Prefixed, because the notice rides on the last report's lines and is a
    # fact about the listing rather than about that run.
    assert "Listing:" in joined
    # The notice rides on the last report, so the labels still name only runs.
    assert all(label for label, _lines in reports)


def test_collapsed_duplicate_screens_are_announced(tmp_path):
    """The archive stores a run twice and the copies read alike, so one is
    collapsed -- which is deliberate. What must not happen is collapsing it in
    silence, so the page reads as the whole set."""
    from rfproteina.disease.screen import screen_reports_for_directory

    root = tmp_path / "pair"
    root.mkdir()
    # The archive's collision suffix is stripped for naming, so these two render
    # the same sentences. Differently named folders would not collapse at all.
    for name in ("gene_8TLD_2", "gene_8TLD_2-2"):
        run = finished_run(root, name)
        (run / "screen.json").write_text(
            json.dumps({"call": "untested", "untested": [], "hits": [], "samples": []}),
            encoding="utf-8",
        )
    reports = screen_reports_for_directory(root)
    labels = {label for label, _lines in reports if label}
    assert len(labels) == 1, f"identical readouts collapse, got {labels}"
    joined = " ".join(line for _label, lines in reports for line in lines)
    assert "collapsed" in joined, f"the collapse must be stated, got {joined}"


def test_a_complete_page_does_not_claim_screens_were_hidden(tmp_path):
    """A screen-less folder visited after the last screen is not a hidden screen.

    The cap used to be tested at the top of the walk, before knowing whether the
    next directory held anything, so one notes folder -- or the archive's own
    ``generations/`` level -- turned a complete listing into a false claim.
    """
    from rfproteina.disease.screen import screen_reports_for_directory

    root = tmp_path / "exact"
    root.mkdir()
    for index in range(2):
        run = finished_run(root, f"gene_{index}ABC_1")
        (run / "screen.json").write_text(
            json.dumps({"call": "untested", "untested": [], "hits": [], "samples": []}),
            encoding="utf-8",
        )
    # Sorts after both runs and holds no screen.
    (root / "zz_notes").mkdir()
    reports = screen_reports_for_directory(root, limit=2)
    joined = " ".join(line for _label, lines in reports for line in lines)
    assert "more screens" not in joined, f"nothing was hidden, got {joined}"
    assert len([label for label, _lines in reports if label]) == 2


def test_a_suffixed_run_folder_is_named_by_shape_not_position(tmp_path):
    """``gene_8TLD_2_inverse`` is 8TLD chain 2, inverse-folded.

    Taking the last two underscore parts read it as entry "2", chain "inverse":
    every measurement in the line was right and both names were wrong, on the
    one archived run a reader would go looking for by name.
    """
    from rfproteina.disease.screen import _names_for_directory

    cases = {
        "gene_8TLD_2": ("8TLD", "2"),
        "gene_8TLD_2-2": ("8TLD", "2"),
        "gene_8TLD_2_inverse": ("8TLD", "2"),
        "gene_9BSB_3": ("9BSB", "3"),
        "protein_8K4Q_1": ("8K4Q", "1"),
        # Not the canonical shape and no entry-like part: not named, rather than
        # named wrongly. _unnamed_lines reports the measurements without a claim.
        "one_two_three_four": ("", ""),
    }
    for name, want in cases.items():
        assert _names_for_directory(tmp_path / name) == want, name


def test_generation_json_still_wins_over_the_folder_name(tmp_path):
    """The folder name is the fallback, not the source of truth."""
    from rfproteina.disease.screen import _names_for_directory

    run = tmp_path / "gene_8TLD_2_inverse"
    run.mkdir()
    (run / "generation.json").write_text(json.dumps({"entry_id": "9XYZ", "chain": "B"}), encoding="utf-8")
    assert _names_for_directory(run) == ("9XYZ", "B")


def test_a_listing_that_gave_up_before_any_screen_still_says_so(tmp_path, monkeypatch):
    """Silence is worst here: the walk stopped without finding one screen.

    The notice rode on the last report's lines, so with no reports at all there
    was nothing to ride on and it was dropped -- the page then said nothing
    whatever about having given up looking, which reads as "there is nothing
    here".

    The walk cap and the report limit are two different facts, so they are two
    different sentences: fewer reports than the limit can still be an
    incomplete listing.
    """
    from rfproteina.disease import screen as screen_module
    from rfproteina.disease.screen import screen_reports_for_directory

    root = tmp_path / "wide"
    root.mkdir()
    for index in range(6):
        (root / f"folder_{index}").mkdir()
    monkeypatch.setattr(screen_module, "_SCREEN_WALK_CAP", 2)
    reports = screen_reports_for_directory(root)
    joined = " ".join(line for _label, lines in reports for line in lines)
    assert "Listing:" in joined, f"giving up must be stated, got {reports}"
    assert "stopped before every folder was read" in joined
    assert "more screens than" not in joined, "nothing was over the report limit"
