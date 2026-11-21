"""The run-level roll-up and the gate's treatment of a design it cannot read.

Both were found by the screening review of 2026-11-21 and neither had a test.
``rollup_axis`` is reached today only through ``screen_generation``, which does
UniProt, Ensembl and STRING lookups for its target, so the suites that cover it
block on the network and nothing pinned its behaviour. These two functions are
reachable directly, so they are tested directly: no network, no torch, no
checkpoints.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from rfproteina.disease import screen
from rfproteina.disease.combinations import rollup_axis


def test_a_rolled_up_hit_carries_the_worst_designs_own_record() -> None:
    """The number and the evidence beside it must come from one design.

    The roll-up took the first hit's record and wrote the worst hit's
    measurement over it, so a novelty roll-up read 0.91 next to the *other*
    design's matched entry, identity and target -- describing a design that does
    not exist. ``offtarget.verdict`` stores the same quantity twice, so there it
    produced one record contradicting itself.
    """
    by_example = {
        "d1": {
            "state": "hit",
            "measurement": 0.62,
            "best_target": "1abc",
            "matched_entry": "d1",
            "fident": 0.55,
        },
        "d2": {
            "state": "hit",
            "measurement": 0.91,
            "best_target": "9xyz",
            "matched_entry": "d2",
            "fident": 0.98,
        },
    }
    record = rollup_axis(["d1", "d2"], by_example, {})

    assert record["state"] == "hit"
    assert record["measurement"] == 0.91, "the worst of the hits, as the docstring promises"
    assert record["best_target"] == "9xyz", "and the evidence is that same design's"
    assert record["fident"] == 0.98
    assert record["matched_entry"] == "d2"
    assert record["failed"] == ["d1", "d2"]


def test_a_tie_keeps_the_first_design_as_it_always_did() -> None:
    """Equal measurements must not reorder a stored record."""
    by_example = {
        "d1": {"state": "hit", "measurement": 0.5, "best_target": "first"},
        "d2": {"state": "hit", "measurement": 0.5, "best_target": "second"},
    }
    assert rollup_axis(["d1", "d2"], by_example, {})["best_target"] == "first"


def test_a_hit_with_no_measurable_number_still_rolls_up() -> None:
    """Biosecurity's hit carries no number, and that must stay a hit."""
    by_example = {"d1": {"state": "hit", "measurement": None, "flag": "x"}}
    record = rollup_axis(["d1"], by_example, {})
    assert record["state"] == "hit"
    assert record["measurement"] is None
    assert record["failed"] == ["d1"]


def test_a_design_the_screen_cannot_decode_is_found() -> None:
    """``_generated_cifs`` lists what it can read and drops the rest in silence.

    Those paths reached no axis, so every roll-up ran over the designs that
    happened to be readable and a run whose *hitting* design was written in a
    spelling the screen cannot decode published the other design's clear as the
    run's -- ``axis_hit`` to ``screened_clear``, with no blocker anywhere.
    """
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder)
        for design, suffix in (("binder_0", "cif"), ("binder_1", "bcif")):
            sub = out / "inference_samples" / design
            sub.mkdir(parents=True)
            (sub / f"{design}_generated.{suffix}").write_bytes(b"x")

        readable = sorted((out / "inference_samples").rglob("*_generated.cif"))
        refused = screen._refused_designs(out, readable)
        assert [path.name for path in refused] == ["binder_1_generated.bcif"]


def test_a_second_spelling_of_a_design_already_read_is_not_refused() -> None:
    """One molecule written twice is not a design nothing looked at.

    ``_generated_cifs`` settles that by stem, so refusing it here would report a
    design as unscreened when it had in fact been screened.
    """
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder)
        sub = out / "inference_samples" / "binder_0"
        sub.mkdir(parents=True)
        (sub / "binder_0_generated.cif").write_bytes(b"x")
        (sub / "binder_0_generated.bcif").write_bytes(b"x")

        readable = sorted(sub.glob("*_generated.cif"))
        assert screen._refused_designs(out, readable) == []


def test_the_caller_guards_the_append_on_having_read_something() -> None:
    """With nothing readable, ``_refused_designs`` still names the files.

    That is why the append is guarded on ``paths``: the ``if not samples``
    branch already reports ``designs_compressed`` for a run where nothing could
    be read, and a phantom chain there would move the protein path's blockers.
    This pins that the guard is load-bearing rather than decorative.
    """
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder)
        sub = out / "inference_samples" / "binder_0"
        sub.mkdir(parents=True)
        (sub / "binder_0_generated.bcif").write_bytes(b"x")

        assert [path.name for path in screen._refused_designs(out, [])] == ["binder_0_generated.bcif"]
        source = Path(screen.__file__).read_text(encoding="utf-8")
        assert "if paths:" in source, "the guard the adversary required must still be there"
