"""The Conformational Dynamics Analyst: what moves, measured over an ensemble.

No model, no checkpoint, no GPU and no network. The generator is injected --
the same way ``offtarget`` and ``stability`` are already testable -- and the
ensembles here are built by hand, so every number is produced by this module's
own arithmetic rather than by whatever happens to be installed.

The geometry tests plant a known answer and check it comes back: an ensemble
whose last five residues move and whose core does not has to read as five
mobile residues, not as a uniformly wobbly protein.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from rfproteina.disease import confrover_runner, results_actions, ui
from rfproteina.disease import dynamics as dyn
from rfproteina.disease.screening_tools import CLEAR, HIT, UNTESTED
from tests.helpers.disease import no_developer_proteome  # noqa: F401


def _pdb(coords: np.ndarray, chain: str = "A") -> str:
    """CA-only PDB text, by column, from an ``(n, 3)`` array.

    Written by column because PDB is a fixed-column format and
    ``results_compare`` reads it as one.
    """
    lines = []
    for index, (x, y, z) in enumerate(coords, start=1):
        lines.append(
            "ATOM  "
            + f"{index:5d}"
            + " "
            + " CA "
            + " "
            + "ALA"
            + " "
            + chain
            + f"{index:4d}"
            + "    "
            + f"{x:8.3f}{y:8.3f}{z:8.3f}"
            + f"{1.00:6.2f}{0.00:6.2f}"
            + " " * 10
            + " C"
        )
    return "\n".join(lines) + "\nEND\n"


_RESIDUES = 20
#: An alpha-helical trace rather than a straight line, and not for realism. A
#: collinear chain is degenerate for a Kabsch superposition -- rotation about
#: the chain axis is unconstrained, so the decomposition may return any of them
#: -- and a rigid ensemble of one then showed a residual fluctuation of 4e-4 A
#: that read as a bug in the measurement rather than in the fixture.
_TURN = np.arange(_RESIDUES) * 1.745
_CHAIN = np.stack([np.arange(_RESIDUES) * 1.5, 2.3 * np.cos(_TURN), 2.3 * np.sin(_TURN)], axis=1).astype(float)
#: How many residues at the end of the chain the fixtures below move.
_TAIL = 5


def _wobbly(frames: int = 12, *, tail: float = 1.5, core: float = 0.05, seed: int = 0) -> list[str]:
    """Conformers whose last ``_TAIL`` residues move and whose core does not."""
    rng = np.random.default_rng(seed)
    made = []
    for _ in range(frames):
        moved = _CHAIN.copy()
        moved[-_TAIL:] += rng.normal(0, tail, size=(_TAIL, 3))
        moved[:-_TAIL] += rng.normal(0, core, size=(_RESIDUES - _TAIL, 3))
        made.append(_pdb(moved))
    return made


def _rigid(frames: int = 12) -> list[str]:
    """The same chain every time, translated and rotated but never deformed."""
    made = []
    for index in range(frames):
        angle = index * 0.3
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        made.append(_pdb(_CHAIN @ rotation + np.array([index * 5.0, 0.0, 0.0])))
    return made


def _ensemble(texts: list[str], **kwargs) -> dyn.Ensemble:
    frames, blocker = dyn.frames_from_structures(texts, chain="A")
    assert not blocker, blocker
    return dyn.Ensemble(frames=frames, **kwargs)


# --- the vocabulary ----------------------------------------------------------


def test_the_three_task_modes_are_a_closed_set_and_each_states_its_purpose():
    assert set(dyn.TASK_MODES) == {dyn.FORWARD, dyn.IID, dyn.INTERPOLATE}
    assert set(dyn.TASK_PURPOSE) == set(dyn.TASK_MODES)
    for mode in dyn.TASK_MODES:
        assert dyn.TASK_PURPOSE[mode].endswith(".")


def test_every_sentence_reads_as_one():
    for name, said in dyn._BLOCKER_TEXT.items():
        assert said.endswith("."), name
        assert name not in said, f"{name} explains itself with its own slug"


def test_an_unknown_blocker_renders_as_its_spaced_name():
    assert dyn.sentence("some_future_name") == "some future name"
    assert dyn.sentence("") == ""


def test_the_caveats_say_what_the_generator_is_not():
    """Both ride on every record, so a number cannot be quoted without them."""
    assert "out of distribution" in dyn.OUT_OF_DISTRIBUTION
    assert "not measurements" in dyn.OUT_OF_DISTRIBUTION
    assert "100 ns" in dyn.TIMESCALE_CAVEAT


def test_the_runner_and_the_screen_agree_on_the_payload():
    """Two lists of the same four pieces would drift the moment either moved."""
    assert set(dyn.REQUIRED_PAYLOAD) == set(confrover_runner.PAYLOAD)
    assert dyn.REQUIRED_PAYLOAD == confrover_runner.PAYLOAD


# --- the geometry ------------------------------------------------------------


def test_the_moving_tail_is_the_part_reported_as_mobile():
    """The planted answer: five residues move, and five come back.

    A fluctuation taken without superposing first would read every residue as
    mobile on an ensemble that merely drifted, which is what ``_rigid`` below
    checks from the other side.
    """
    measured, blocker = dyn.measure(_ensemble(_wobbly()))
    assert not blocker
    assert measured is not None
    mobile = measured.mobile_residues(1.0)
    assert set(mobile) == set(range(_RESIDUES - _TAIL + 1, _RESIDUES + 1))
    assert len(mobile) == _TAIL


#: What a round trip through a PDB file costs. The format writes coordinates to
#: three decimal places, so every axis is quantised to a thousandth of an
#: angstrom and a rigid ensemble cannot superpose better than that. Asserting
#: against float64 instead asked for better than the format can carry and read
#: as a measurement bug; the real superposition bug this catches was angstroms
#: wide, hundreds of times above this bound.
_PDB_PRECISION_A = 2e-3


def test_a_chain_that_only_moved_through_space_is_not_mobile():
    """Rigid-body motion is not fluctuation, so it is taken out first.

    This is the test that catches a wrong superposition. Written with
    ``coords @ rotation`` instead of ``coords @ rotation.T`` -- the convention
    ``kabsch_transform`` documents -- every frame was rotated by the transpose
    and this ensemble read as one whose every residue moves. The wobbly
    fixtures could not see it: a planted 1.5 A signal swamps the error.
    """
    measured, blocker = dyn.measure(_ensemble(_rigid()))
    assert not blocker
    assert measured is not None
    assert measured.rmsf_max_a is not None
    assert measured.rmsf_max_a < _PDB_PRECISION_A, measured.rmsf_max_a
    assert measured.drift_a is not None and measured.drift_a < _PDB_PRECISION_A


def test_the_radius_of_gyration_separates_moving_loops_from_coming_apart():
    """Fluctuation rises for both; only one expands."""
    loops, _ = dyn.measure(_ensemble(_wobbly(tail=1.5)))
    apart = []
    rng = np.random.default_rng(1)
    for _ in range(12):
        apart.append(_pdb(_CHAIN * (1.0 + rng.uniform(0.2, 0.5))))
    blown, _ = dyn.measure(_ensemble(apart))
    assert loops is not None and blown is not None
    assert blown.radius_of_gyration_spread_a > loops.radius_of_gyration_spread_a


def test_the_spread_is_zero_for_an_ensemble_of_one_shape():
    measured, _ = dyn.measure(_ensemble(_rigid()))
    assert measured is not None
    assert measured.spread_a is not None and measured.spread_a < _PDB_PRECISION_A


# --- the refusals ------------------------------------------------------------


def test_too_few_conformers_is_arithmetic_rather_than_a_measurement():
    """Two frames give every residue the same distance from their mean."""
    frames, _ = dyn.frames_from_structures(_wobbly(frames=2), chain="A")
    measured, blocker = dyn.measure(dyn.Ensemble(frames=frames))
    assert measured is None
    assert blocker == "ensemble_too_small"
    assert dyn.MIN_FRAMES >= 5


def test_too_many_conformers_is_refused_rather_than_read_in_a_panel():
    frames, _ = dyn.frames_from_structures(_wobbly(frames=6), chain="A")
    over = dyn.Ensemble(frames=frames * (dyn.MAX_FRAMES // 6 + 1))
    measured, blocker = dyn.measure(over)
    assert measured is None
    assert blocker == "ensemble_too_large"


def test_an_ensemble_that_never_ran_is_not_a_design_that_does_not_move():
    """The one answer it must not give."""
    blocked = dyn.Ensemble(blocker="msa_not_cached")
    assert not blocked.ran
    measured, blocker = dyn.measure(blocked)
    assert measured is None
    assert blocker == "msa_not_cached"


def test_conformers_of_different_lengths_are_not_an_ensemble():
    short = _pdb(_CHAIN[:12])
    frames, blocker = dyn.frames_from_structures([*_wobbly(frames=5), short], chain="A")
    assert frames == ()
    assert blocker == "frames_disagree_on_length"


def test_a_chain_too_short_to_superpose_is_refused():
    frames, blocker = dyn.frames_from_structures([_pdb(_CHAIN[:4])] * 6, chain="A")
    assert blocker == "chain_too_short"
    assert dyn.MIN_RESIDUES >= 8


def test_a_complex_is_refused_before_anything_is_generated():
    """The model takes one chain, so a complex is not an input it has.

    Refused at the door rather than by analysing one chain of it: on most of
    this program's outputs the first chain is the target, not the design.
    """
    two = _pdb(_CHAIN, "A").replace("END\n", "") + _pdb(_CHAIN + 25.0, "B")
    record = dyn.analyse(two)
    assert record["blocker"] == "design_chain_not_single"
    assert record["measured"] is None
    # Naming the chain makes it analysable again.
    named = dyn.analyse(two, chain="B", root=Path("/nonexistent"))
    assert named["blocker"] == "weights_not_installed"


def test_no_design_and_an_unknown_task_are_each_refused_by_name():
    assert dyn.analyse("")["blocker"] == "design_not_supplied"
    assert dyn.analyse(_pdb(_CHAIN), task_mode="wander")["blocker"] == "task_mode_unknown"
    assert dyn.analyse("not a structure")["blocker"] == "design_unreadable"


def test_a_degenerate_ensemble_does_not_end_the_run(monkeypatch):
    """A view must not be able to end a finished run."""

    def explode(frames):
        raise MemoryError("not enough room")

    monkeypatch.setattr(dyn, "rmsf", explode)
    measured, blocker = dyn.measure(_ensemble(_wobbly()))
    assert measured is None
    assert blocker.startswith("generator_failed")


# --- the install, checked rather than attempted ------------------------------


def test_each_missing_piece_of_the_install_is_named(tmp_path: Path):
    """Every one of these the generator would fetch or build on first call."""
    assert dyn.payload_blocker(tmp_path / "absent") == "weights_not_installed"
    root = tmp_path / "confrover"
    for name, expected in (
        ("weights", "cutlass_not_installed"),
        ("cutlass", "openfold_params_not_installed"),
        ("openfold_params", "msa_not_cached"),
    ):
        made = root / name
        made.mkdir(parents=True)
        (made / "something").write_text("x", encoding="utf-8")
        assert dyn.payload_blocker(root) == expected
    (root / "msa").mkdir()
    (root / "msa" / "cached.a3m").write_text("x", encoding="utf-8")
    assert dyn.payload_blocker(root) is None


def test_an_empty_directory_does_not_count_as_installed(tmp_path: Path):
    """An interrupted install leaves the directory it was writing into.

    Read with a bare ``exists()`` that looked done, so the screen waved a half
    install through and the failure surfaced deeper, inside the generator,
    looking like a model problem. The installer creating ``weights/`` before it
    had anything to put in it was enough to do it.
    """
    root = tmp_path / "confrover"
    (root / "weights").mkdir(parents=True)
    assert dyn.payload_blocker(root) == "weights_not_installed"
    assert not dyn._installed(root / "weights")
    assert not confrover_runner.installed(root / "weights")
    (root / "weights" / "confrover_base_20m_v1_0.pt").write_text("x", encoding="utf-8")
    assert dyn._installed(root / "weights")
    assert confrover_runner.installed(root / "weights")


def test_the_msa_decline_says_why_the_screen_will_not_do_it():
    """The one decline that exists to stop a sequence leaving the machine."""
    said = dyn.sentence("msa_not_cached")
    assert "public MSA server" in said
    assert "off this machine" in said


def test_the_cutlass_decline_says_the_screen_will_not_download():
    said = dyn.sentence("cutlass_not_installed")
    assert "clone it from the internet" in said
    assert "installer" in said


def test_the_screen_declines_rather_than_falling_back_to_this_interpreter(monkeypatch):
    """This project's interpreter cannot import the generator, and must not be tried.

    Falling back would turn a missing install into a ModuleNotFoundError deep
    inside somebody else's source rather than the named decline.
    """
    monkeypatch.delenv(dyn.PYTHON_ENV, raising=False)
    monkeypatch.setattr(dyn, "data_root", lambda: None)
    assert dyn.python_executable() is None


def test_a_named_interpreter_has_to_exist(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(dyn.PYTHON_ENV, str(tmp_path / "nope"))
    assert dyn.python_executable() is None
    made = tmp_path / "python"
    made.write_text("", encoding="utf-8")
    monkeypatch.setenv(dyn.PYTHON_ENV, str(made))
    assert dyn.python_executable() == str(made)


def test_the_vendored_tree_and_viewer_are_where_the_screen_looks():
    """Both are vendored, so these resolve on any checkout rather than this one."""
    code = dyn.code_root()
    assert code is not None and (code / "rbase" / "__init__.py").is_file()
    viewer = dyn.viewer_root()
    assert viewer is not None and (viewer / "visualize_ensemble.py").is_file()
    assert dyn.runner_script().is_file()


# --- the record --------------------------------------------------------------


def test_a_measurement_is_reported_and_not_judged_without_a_cut():
    """Measuring and clearing are different acts, and this module will only do one."""
    record = dyn.ensemble_record(_ensemble(_wobbly()))
    assert record["state"] == UNTESTED
    assert record["blocker"] == "rmsf_threshold_not_set"
    assert record["measured"] is not None
    assert record["thresholds"]["rmsf_max_a"] is None


def test_a_cut_turns_the_same_measurement_into_a_verdict():
    ensemble = _ensemble(_wobbly())
    hit = dyn.ensemble_record(
        ensemble,
        limits=dyn.DynamicsLimits(rmsf_max_a=1.0, mobile_fraction_max=0.1, drift_max_a=5.0),
    )
    assert hit["state"] == HIT
    assert hit["mobile_residues"] == list(range(_RESIDUES - _TAIL + 1, _RESIDUES + 1))
    clear = dyn.ensemble_record(
        _ensemble(_rigid()),
        limits=dyn.DynamicsLimits(rmsf_max_a=1.0, mobile_fraction_max=0.1, drift_max_a=1.0),
    )
    assert clear["state"] == CLEAR
    assert clear["blocker"] is None


def test_clearing_the_fluctuation_alone_does_not_clear_the_screen():
    """Half the question answered is not a clear."""
    record = dyn.ensemble_record(
        _ensemble(_rigid()),
        limits=dyn.DynamicsLimits(rmsf_max_a=1.0, mobile_fraction_max=0.5),
    )
    assert record["state"] == UNTESTED
    assert record["blocker"] == "drift_threshold_not_set"


def test_a_fluctuation_cut_without_a_share_decides_nothing():
    """It used to clear, while listing the mobile residues underneath.

    A per-residue cut names which residues move. Whether that matters depends
    on how much of the chain they are, so on its own it cannot decide -- and a
    CLEAR contradicted by the rows beside it is worse than an undecided.
    """
    record = dyn.ensemble_record(_ensemble(_wobbly()), limits=dyn.DynamicsLimits(rmsf_max_a=1.0))
    assert record["state"] == UNTESTED
    assert record["blocker"] == "mobile_fraction_not_set"
    assert record["mobile_residues"], "the residues are still named"


def test_every_record_carries_both_caveats_and_the_code_that_produced_it():
    for ensemble in (_ensemble(_wobbly()), dyn.Ensemble(blocker="msa_not_cached")):
        record = dyn.ensemble_record(ensemble)
        assert record["out_of_distribution"] == dyn.OUT_OF_DISTRIBUTION
        assert record["timescale"] == dyn.TIMESCALE_CAVEAT
        assert record["single_chain_only"] is True
        assert set(record["code"]) == {"commit", "tree_state"}


def test_every_record_survives_a_json_round_trip():
    """A numpy float leaking into a record is the likeliest first bug here."""
    record = dyn.ensemble_record(_ensemble(_wobbly()), limits=dyn.DynamicsLimits(rmsf_max_a=1.0, drift_max_a=2.0))
    assert json.loads(json.dumps(record, allow_nan=False)) == record


def test_the_readout_states_the_generator_and_both_caveats():
    record = dyn.ensemble_record(_ensemble(_wobbly()))
    text = "\n".join(dyn.readout_lines(record))
    assert dyn.GENERATOR in text
    assert dyn.OUT_OF_DISTRIBUTION in text
    assert dyn.TIMESCALE_CAVEAT in text
    assert "fluctuation:" in text
    assert "drift from the design:" in text


def test_the_summary_counts_the_conformers_it_was_over():
    record = dyn.ensemble_record(_ensemble(_wobbly(frames=9)))
    said = dyn.summary_line(record)
    assert said.startswith(dyn.ANALYST_LABEL)
    assert "over 9 conformers" in said


# --- the tab -----------------------------------------------------------------


def test_the_tab_sits_last_of_the_four():
    """Generation, then screening, then this: it reads a finished design."""
    page = ui._results_tools("")
    assert ui.RESULTS_TABS[-1] == ui.DYNAMICS_TAB
    assert f'id="results-tab-dynamics" value="{ui.DYNAMICS_TAB}"' in page
    assert dyn.ANALYST_LABEL in page
    assert 'class="results-panel results-panel-dynamics"' in page


def test_the_tab_opens_when_it_was_the_one_submitted():
    page = ui._results_tools("", active=ui.DYNAMICS_TAB)
    assert 'id="results-tab-dynamics" value="dynamics" checked' in page
    assert 'id="results-tab-binder" value="binder">' in page


def test_every_field_the_buttons_read_is_one_the_tab_posts():
    page = ui._results_tools("")
    for name in results_actions.DYNAMICS_FIELDS:
        assert f'name="{name}"' in page, name
    for action in (results_actions.DYNAMICS_RUN, results_actions.DYNAMICS_DRAW):
        assert f'value="{action}"' in page
        assert action in results_actions.ACTIONS


def test_the_single_chain_limit_is_on_the_tab_before_the_buttons():
    """The thing a reader is most likely to want is the thing it cannot do."""
    from html import escape

    page = ui._results_tools("")
    assert escape(ui.DYNAMICS_SINGLE_CHAIN, quote=True) in page
    assert "design-plus-target complex is not an input it has" in page


def test_all_three_task_modes_are_offered_with_their_purposes():
    page = ui._results_tools("")
    for mode in dyn.TASK_MODES:
        assert f'value="{mode}"' in page
        assert dyn.TASK_PURPOSE[mode].split(":")[0][:30] in page


def test_the_readout_draws_the_numbers_and_keeps_the_caveats():
    record = {"dynamics": dyn.ensemble_record(_ensemble(_wobbly()), limits=dyn.DynamicsLimits(rmsf_max_a=1.0))}
    page = ui._results_tools("", record=record)
    assert "Fluctuation, mean" in page
    assert "Drift from the design" in page
    assert "Radius of gyration" in page
    assert "Mobile residues:" in page
    from html import escape

    assert escape(dyn.OUT_OF_DISTRIBUTION, quote=True) in page


def test_nothing_is_drawn_before_a_run():
    page = ui._results_tools("")
    assert "Fluctuation, mean" not in page


def test_the_run_button_declines_when_the_catalog_holds_no_design():
    done = results_actions.perform(results_actions.DYNAMICS_RUN, [{"sample": "d1"}], lambda token: None)
    assert done.record is not None
    assert done.record["dynamics"]["blocker"] == "design_not_supplied"


def test_the_draw_button_says_to_run_first_rather_than_drawing_nothing():
    """An empty page would read as a design that does not move."""
    catalog = [{"sample": "d1", "complex_token": "tok", "binder_chains": ["A"]}]
    done = results_actions.perform(results_actions.DYNAMICS_DRAW, catalog, lambda token: _pdb(_CHAIN).encode("utf-8"))
    assert done.record is not None
    assert done.record["dynamics"]["blocker"] == "ensemble_not_supplied"
    assert dyn.OUT_OF_DISTRIBUTION in done.sentence


# --- drawing -----------------------------------------------------------------


def test_the_viewer_draws_a_real_ensemble_into_a_page(tmp_path: Path):
    """The vendored viewer, run for real on conformers written to disk.

    Not stubbed. This is the half the generator's absence does not block: the
    viewer needs numpy and nothing else, so a page can be drawn on a machine
    where nothing is installed, from an ensemble somebody else produced.
    """
    source = tmp_path / "ensemble"
    source.mkdir()
    for index, text in enumerate(_wobbly(frames=8)):
        (source / f"frame_{index:03d}.pdb").write_text(text, encoding="utf-8")
    drawn = dyn.draw_ensemble(source, tmp_path / "page", chain="A", title="design_1")
    assert drawn["blocker"] is None, drawn.get("reason") or drawn.get("detail")
    page = drawn["page"]
    assert isinstance(page, str) and Path(page).is_file()
    assert Path(page).suffix == ".html"
    assert Path(page).stat().st_size > 1000
    assert drawn["conformers_drawn"] == 8
    assert drawn["movie"] is None


def test_drawing_nothing_is_refused_rather_than_producing_an_empty_page(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert dyn.draw_ensemble(empty, tmp_path / "out")["blocker"] == dyn.NO_ENSEMBLE
    assert dyn.draw_ensemble(tmp_path / "absent", tmp_path / "out")["blocker"] == dyn.NO_ENSEMBLE


def test_a_missing_mp4_writer_loses_the_movie_and_not_the_page(tmp_path: Path, monkeypatch):
    """The page is worth drawing either way, so only the movie is dropped.

    Asked before the run rather than discovered at the last step: the viewer
    imports its mp4 writer inside the render, so without this check the
    request would spend the whole page render and then fail.
    """
    monkeypatch.setattr(dyn, "_movie_writer_available", lambda: False)
    source = tmp_path / "ensemble"
    source.mkdir()
    for index, text in enumerate(_wobbly(frames=6)):
        (source / f"frame_{index:03d}.pdb").write_text(text, encoding="utf-8")
    drawn = dyn.draw_ensemble(source, tmp_path / "page", chain="A", movie=True)
    assert drawn["blocker"] == "movie_tool_not_installed"
    assert drawn["movie"] is None
    assert isinstance(drawn["page"], str) and Path(drawn["page"]).is_file()


def test_the_viewer_runs_under_this_project_s_interpreter_not_the_generator_s():
    """It needs numpy, which this has.

    Tying it to the generator's venv would have made a viewer that works wait
    for a generator that is not there.
    """
    import sys

    assert dyn.viewer_python() == sys.executable
    script = dyn.viewer_script()
    assert script is not None and script.is_file()


def test_the_run_keeps_the_ensemble_where_the_draw_can_find_it(monkeypatch, tmp_path: Path):
    """Two button presses, one job.

    An ensemble that vanished with the process left the second with nothing,
    which reads as a design that does not move.
    """
    monkeypatch.setattr(dyn, "payload_blocker", lambda root=None: None)
    monkeypatch.setattr(dyn, "ensembles_root", lambda: tmp_path / "ensembles")
    seen: dict[str, object] = {}

    def produce(text, chain, mode, **kwargs):
        seen["work_dir"] = kwargs.get("work_dir")
        return _ensemble(_wobbly(), task_mode=mode, design=chain)

    monkeypatch.setattr(dyn, "generate_ensemble", produce)
    catalog = [{"sample": "design_1", "complex_token": "tok", "binder_chains": ["A"]}]
    done = results_actions.perform(
        results_actions.DYNAMICS_RUN,
        catalog,
        lambda token: _pdb(_CHAIN).encode("utf-8"),
        [f"{results_actions.DYNAMICS_CHAIN}=A"],
    )
    assert done.record is not None
    assert seen["work_dir"] == tmp_path / "ensembles" / "design_1"
    assert str(done.record["dynamics"]["ensemble_dir"]).endswith("design_1/ensemble")


def test_the_draw_button_reads_the_directory_the_run_left(tmp_path: Path):
    """The glue between the two buttons, through the record the page holds."""
    source = tmp_path / "design_1" / "ensemble"
    source.mkdir(parents=True)
    for index, text in enumerate(_wobbly(frames=6)):
        (source / f"frame_{index:03d}.pdb").write_text(text, encoding="utf-8")
    catalog = [{"sample": "design_1", "complex_token": "tok", "binder_chains": ["A"]}]
    done = results_actions.perform(
        results_actions.DYNAMICS_DRAW,
        catalog,
        lambda token: _pdb(_CHAIN).encode("utf-8"),
        [f"{results_actions.DYNAMICS_CHAIN}=A"],
        held={"dynamics": {"ensemble_dir": str(source), "design_sample": "design_1"}},
    )
    assert done.record is not None
    drawn = done.record["dynamics"]["drawn"]
    assert drawn["blocker"] is None, drawn.get("reason")
    assert Path(drawn["page"]).is_file()
    assert "Drew 6 conformers" in done.sentence
    assert dyn.OUT_OF_DISTRIBUTION in done.sentence
    assert "Page:" in ui._results_tools("", record=done.record)


def test_the_movie_tick_is_offered_and_explains_its_cost():
    page = ui._results_tools("")
    assert f'name="{results_actions.DYNAMICS_MOVIE}"' in page
    assert "minutes to an hour" in page


def test_the_run_button_reaches_the_screen_through_the_catalog(monkeypatch):
    """The glue: a submission in the shape the browser posts.

    The generator is replaced, because running it needs a checkpoint, a CUDA
    clone and an alignment this test has none of -- which is the whole reason
    the screen takes it as an argument.
    """
    monkeypatch.setattr(dyn, "payload_blocker", lambda root=None: None)
    monkeypatch.setattr(
        dyn,
        "generate_ensemble",
        lambda text, chain, mode, **kwargs: _ensemble(_wobbly(), task_mode=mode, design=chain, replicates=1),
    )
    catalog = [{"sample": "design_1", "complex_token": "tok", "binder_chains": ["A"]}]
    done = results_actions.perform(
        results_actions.DYNAMICS_RUN,
        catalog,
        lambda token: _pdb(_CHAIN).encode("utf-8"),
        [
            f"{results_actions.DYNAMICS_CHAIN}=A",
            f"{results_actions.DYNAMICS_TASK}=iid",
            f"{results_actions.DYNAMICS_RMSF_MAX}=1.0",
            f"{results_actions.DYNAMICS_MOBILE_MAX}=0.1",
            f"{results_actions.DYNAMICS_DRIFT_MAX}=5.0",
        ],
    )
    assert done.record is not None
    record = done.record["dynamics"]
    assert record["task_mode"] == "iid"
    assert record["state"] == HIT
    assert record["design_sample"] == "design_1"
    assert done.payload_name == "conformational_dynamics.txt"
    assert dyn.ANALYST_LABEL in done.sentence


def test_a_typed_cut_that_is_not_a_number_is_no_cut(monkeypatch):
    """A blank or a word leaves the measurement unjudged rather than crashing."""
    assert results_actions._number("") is None
    assert results_actions._number("two") is None
    assert results_actions._number("nan") is None
    assert results_actions._number("inf") is None
    assert results_actions._number(" 2.5 ") == 2.5


# --- the headless route ------------------------------------------------------


def test_the_mode_refuses_a_path_that_is_not_a_file(tmp_path: Path):
    from rfproteina.disease.cli import main

    assert main(["--dynamics", str(tmp_path / "absent.pdb")]) == 2


def test_the_mode_refuses_a_complex_without_a_named_chain(tmp_path: Path, capsys):
    from rfproteina.disease.cli import main

    path = tmp_path / "complex.pdb"
    path.write_text(_pdb(_CHAIN, "A").replace("END\n", "") + _pdb(_CHAIN + 25.0, "B"), encoding="utf-8")
    assert main(["--dynamics", str(path)]) == 2
    said = capsys.readouterr()
    assert "more than one chain" in said.out + said.err
    assert "--dynamics-chain" in said.out + said.err


def test_the_mode_refuses_a_chain_the_file_does_not_hold(tmp_path: Path, capsys):
    from rfproteina.disease.cli import main

    path = tmp_path / "one.pdb"
    path.write_text(_pdb(_CHAIN, "A"), encoding="utf-8")
    assert main(["--dynamics", str(path), "--dynamics-chain", "Z"]) == 2
    said = capsys.readouterr()
    assert "no chain Z" in said.out + said.err


def test_the_mode_cannot_be_asked_for_beside_another(tmp_path: Path):
    from rfproteina.disease.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--dynamics", "x.pdb", "--screen", "."])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--dynamics", "x.pdb", "--human-protocol", "y.fasta"])


# --- the runner --------------------------------------------------------------


def test_the_runner_imports_nothing_from_this_package():
    """It runs under another interpreter, so an import would tie it to this one."""
    source = dyn.runner_script().read_text(encoding="utf-8")
    body = "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))
    # The docstring names the package; an import statement is what matters.
    assert "import rfproteina" not in body
    assert "from rfproteina" not in body


def test_the_runner_names_the_checkpoint_the_task_needs(tmp_path: Path):
    """Interpolation has its own weights, and the base model refuses that task."""
    root = tmp_path / "confrover"
    weights = root / "weights"
    weights.mkdir(parents=True)
    assert confrover_runner.checkpoint_for(root, "forward") is None
    (weights / "confrover_base_20m_v1_0.pt").write_text("x", encoding="utf-8")
    (weights / "confrover_interp_20m_v1_0.pt").write_text("x", encoding="utf-8")
    base = confrover_runner.checkpoint_for(root, "forward")
    interp = confrover_runner.checkpoint_for(root, "interp")
    assert base is not None and base.name == "confrover_base_20m_v1_0.pt"
    assert interp is not None and interp.name == "confrover_interp_20m_v1_0.pt"
    assert confrover_runner.checkpoint_for(root, "iid") == base


def test_the_runner_reports_the_install_without_running_anything(tmp_path: Path, capsys):
    """``--check-only`` is what the screen asks before it asks for an ensemble."""
    code = dyn.code_root()
    assert code is not None
    assert confrover_runner.main(["--code", str(code), "--weights", str(tmp_path / "nothing"), "--check-only"]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith(confrover_runner.RESULT_PREFIX)
    found = json.loads(printed[len(confrover_runner.RESULT_PREFIX) :])
    assert found["missing"] == "weights_not_installed"
    assert found["base_checkpoint"] is None
    assert found["payload"] == dict.fromkeys(confrover_runner.PAYLOAD, False)


def test_the_runner_refuses_rather_than_fetching_what_is_missing(tmp_path: Path, capsys):
    """The whole reason this wrapper exists."""
    code = dyn.code_root()
    assert code is not None
    assert (
        confrover_runner.main(["--code", str(code), "--weights", str(tmp_path / "nothing"), "--task", "forward"]) == 1
    )
    printed = capsys.readouterr().out
    found = json.loads(printed.split(confrover_runner.RESULT_PREFIX)[-1])
    assert found["blocker"] == "weights_not_installed"


def test_a_result_line_is_marked_rather_than_assumed_to_be_the_last():
    """The generator logs freely, so the caller looks for the marker."""
    said = dyn._result_line("loading weights\n" + confrover_runner.RESULT_PREFIX + '{"blocker": "msa_not_cached"}\n')
    assert said == {"blocker": "msa_not_cached"}
    assert dyn._result_line("nothing useful here") is None
    assert dyn._result_line(confrover_runner.RESULT_PREFIX + "not json") is None


def test_a_detail_carries_no_path(tmp_path: Path):
    """It is stored on a record a reader can save."""
    said = dyn._short(f"failed reading {tmp_path / 'secret.pt'} after 3 tries")
    assert "secret.pt" not in said
    assert str(tmp_path) not in said
    assert "failed reading" in said
    assert len(dyn._short("x" * 500)) <= 120
