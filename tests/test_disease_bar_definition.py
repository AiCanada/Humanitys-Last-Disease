"""What a bar requires, pinned so it cannot drift quietly.

``judge_combination`` publishes ``screened_clear`` when every axis of the bar it
was given reads ``clear``, and it reads those axes out of ``BARS``. Two facts
about that table are load-bearing and were true only by care:

1. Every axis a bar names must be an axis of its own kind. ``judge_combination``
   looks each one up in a dict built from ``AXES_BY_KIND``, so a bar naming an
   axis its kind does not have raises ``KeyError`` -- after the screen has done
   all of its measuring, in the one call that decides the verdict.

2. The ``all`` bar must be exactly every applicable axis. It is the DEFAULT:
   ``judge_combination`` and ``requested_clear`` both fall back to it when no
   bar is named. An ``all`` that had drifted to a subset would publish
   ``screened_clear`` while requiring less than every axis -- a false clear in
   the one place a reader is least likely to look for one, because the bar would
   still be called "all".

Neither is checked anywhere else. A typo in the table is a one-character change
and both facts are invisible in review.
"""

import pytest

from rfproteina.disease.combinations import AXES_BY_KIND, BARS, bar_axes, judge_combination


def test_every_bar_names_only_axes_its_own_kind_has() -> None:
    """A bar naming a foreign axis is a KeyError in the verdict call."""
    for kind, bars in BARS.items():
        applicable = set(AXES_BY_KIND.get(kind, ()))
        for bar, axes in bars.items():
            stray = sorted(set(axes) - applicable)
            assert not stray, f"{kind} bar {bar!r} names {stray}, which are not axes of {kind}"


def test_no_bar_names_an_axis_twice() -> None:
    """A repeat is harmless to the verdict and a sign the table was edited blind."""
    for kind, bars in BARS.items():
        for bar, axes in bars.items():
            assert len(axes) == len(set(axes)), f"{kind} bar {bar!r} repeats an axis"


def test_the_default_bar_requires_every_applicable_axis() -> None:
    """``all`` is what an unnamed bar falls back to, so it may not be a subset."""
    for kind, applicable in AXES_BY_KIND.items():
        if not applicable:
            continue
        assert "all" in BARS[kind], f"{kind} has no all bar, which is the default"
        assert set(BARS[kind]["all"]) == set(applicable), (
            f"{kind}'s all bar is not every applicable axis; "
            f"an unnamed bar would clear on {sorted(set(applicable) - set(BARS[kind]['all']))} untested"
        )


def test_a_narrower_bar_is_a_subset_of_all() -> None:
    """Every other bar is a deliberate subset, never a different set."""
    for kind, bars in BARS.items():
        if "all" not in bars:
            continue
        everything = set(bars["all"])
        for bar, axes in bars.items():
            assert set(axes) <= everything, f"{kind} bar {bar!r} names axes outside all"


def test_an_unknown_bar_is_refused_by_name() -> None:
    """The error names the bars that exist, so a typo is self-correcting."""
    with pytest.raises(ValueError, match="Known bars"):
        bar_axes("protein", "no-such-bar")


def test_a_missing_axis_is_untested_and_cannot_clear() -> None:
    """The guard the whole table rests on: silence is not a clear."""
    applicable = AXES_BY_KIND["protein"]
    cleared = dict.fromkeys(applicable, "clear")
    assert judge_combination("protein", cleared).call == "screened_clear"
    # Drop one axis of the bar and the same evidence must no longer clear.
    for axis in bar_axes("protein", "all"):
        short = {name: state for name, state in cleared.items() if name != axis}
        assert judge_combination("protein", short).call == "untested", f"{axis} missing still cleared"


def test_a_hit_outranks_both_the_panel_and_the_bar() -> None:
    """A hit is never suppressed, whatever else the combination says."""
    applicable = AXES_BY_KIND["protein"]
    evidence = dict.fromkeys(applicable, "clear")
    evidence[applicable[0]] = "hit"
    assert judge_combination("protein", evidence).call == "axis_hit"
    # Including on a panel target, where a clear is refused outright.
    panel = judge_combination("protein", evidence, target_is_panel=True)
    assert panel.call == "axis_hit", "the panel refusal swallowed a measured hit"


def test_a_panel_target_never_reads_as_screened_clear() -> None:
    """Every axis clear on a safety-panel protein is still not a clear."""
    evidence = dict.fromkeys(AXES_BY_KIND["protein"], "clear")
    assert judge_combination("protein", evidence, target_is_panel=True).call == "target_is_panel"
