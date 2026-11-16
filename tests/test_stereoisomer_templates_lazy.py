"""``STEREOISOMER_TEMPLATES`` is lazy, and must be indistinguishable from the dict it replaced.

It was an eagerly-built dict comprehension, which meant 1,080 permutations' worth
of ``Rotation.align_vectors`` -- LAPACK alignments -- ran at **module import
time**. ``metrics.utils.metrics_utils`` imports this module, so every spawned
metrics worker ran them while starting up, and a full-suite run saw one die
there:

    chirals.py:54, in _enumerate_stereoisomer_templates
    numpy.linalg.LinAlgError: 0-dimensional array given.
                              Array must be at least two-dimensional

surfacing in the parent as ``BrokenProcessPool``. The arrays are ``(n, 3)`` and
never 0-dimensional, so that message does not describe its own inputs, and the
fragility behind it is not understood -- eight spawned workers importing the
module directly did not reproduce it. The change is narrow on purpose: the call
no longer runs at import, which is the one place it was seen to fail.

It is NOT a performance change and must not be sold as one. Enumerating every
geometry costs 0.36s, against 23s for importing ``atomworks.enums``, which this
module does regardless.

So the whole risk is behavioural: a mapping that is subtly not the dict it
replaced. These tests are that comparison, computed against the eager
enumeration rather than against recorded numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfproteina.utils.chirals import (
    IDEAL_POLYHEDRA,
    STEREOISOMER_TEMPLATES,
    _enumerate_stereoisomer_templates,
)


@pytest.fixture(scope="module")
def eager() -> dict[tuple[str, int], list[np.ndarray]]:
    """The dict comprehension this replaced, built the old way."""
    return {
        (geom, n): _enumerate_stereoisomer_templates(full[:n])
        for geom, full in IDEAL_POLYHEDRA.items()
        for n in range(3, len(full) + 1)
    }


def test_the_keys_are_the_same_set(eager: dict[tuple[str, int], list[np.ndarray]]) -> None:
    assert set(STEREOISOMER_TEMPLATES) == set(eager)
    assert len(STEREOISOMER_TEMPLATES) == len(eager)


def test_every_value_is_the_same_geometry(
    eager: dict[tuple[str, int], list[np.ndarray]],
) -> None:
    """One representative per proper-rotation orbit, and the same ones.

    Compared elementwise rather than by count: a mapping returning the right
    NUMBER of representatives from the wrong orbits would pass a length check
    and silently change what a chirality call matches against.
    """
    for key, want in eager.items():
        got = STEREOISOMER_TEMPLATES[key]
        assert len(got) == len(want), key
        for index, (one, two) in enumerate(zip(got, want, strict=True)):
            assert np.allclose(one, two), (key, index)


def test_membership_is_answered_without_enumerating(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard in ``chiral_centers`` asks whether a geometry is supported.

    It must not pay for an enumeration to answer, and this is the reason the
    keys are listed up front rather than derived from a populated cache.
    """
    import rfproteina.utils.chirals as chirals

    def refuse(*_args: object, **_kwargs: object) -> list[np.ndarray]:
        raise AssertionError("membership must not enumerate")

    monkeypatch.setattr(chirals, "_enumerate_stereoisomer_templates", refuse)
    fresh = chirals._StereoisomerTemplates(IDEAL_POLYHEDRA)
    assert ("tetrahedral", 4) in fresh
    assert ("octahedral", 6) in fresh
    assert ("bogus", 9) not in fresh
    assert len(fresh) == len(STEREOISOMER_TEMPLATES)
    assert sorted(fresh) == sorted(STEREOISOMER_TEMPLATES)


def test_a_value_is_computed_once_and_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chirality feature asks per atom, so recomputing per access would turn
    0.36s of startup into an inner-loop cost."""
    import rfproteina.utils.chirals as chirals

    calls: list[int] = []
    real = chirals._enumerate_stereoisomer_templates

    def counted(template: np.ndarray, *args: object, **kwargs: object) -> list[np.ndarray]:
        calls.append(1)
        return real(template, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(chirals, "_enumerate_stereoisomer_templates", counted)
    fresh = chirals._StereoisomerTemplates(IDEAL_POLYHEDRA)
    first = fresh[("tetrahedral", 4)]
    again = fresh[("tetrahedral", 4)]
    assert len(calls) == 1
    assert first is again, "the cached list itself is handed back"


def test_only_the_geometry_asked_for_is_enumerated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per key, not all-on-first-touch: a run meeting only tetrahedral carbon
    never enumerates the octahedral orbits."""
    import rfproteina.utils.chirals as chirals

    asked: list[int] = []
    real = chirals._enumerate_stereoisomer_templates

    def counted(template: np.ndarray, *args: object, **kwargs: object) -> list[np.ndarray]:
        asked.append(len(template))
        return real(template, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(chirals, "_enumerate_stereoisomer_templates", counted)
    fresh = chirals._StereoisomerTemplates(IDEAL_POLYHEDRA)
    fresh[("tetrahedral", 4)]
    assert asked == [4], "one enumeration, for the one key asked"


def test_an_unknown_key_behaves_as_a_dict_would() -> None:
    """``.get`` and ``[]`` are both used by callers, so both are pinned."""
    assert STEREOISOMER_TEMPLATES.get(("bogus", 9)) is None
    with pytest.raises(KeyError):
        STEREOISOMER_TEMPLATES[("bogus", 9)]


def test_a_supported_geometry_outside_the_key_range_is_absent() -> None:
    """Keys start at 3 neighbours, and a 2-neighbour centre is not a centre."""
    assert ("tetrahedral", 2) not in STEREOISOMER_TEMPLATES
    assert ("tetrahedral", 3) in STEREOISOMER_TEMPLATES
    widest = max(len(full) for full in IDEAL_POLYHEDRA.values())
    assert ("octahedral", widest + 1) not in STEREOISOMER_TEMPLATES


def test_the_import_itself_enumerates_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The point of the change, asserted on a fresh instance.

    Constructing the mapping must not touch LAPACK, because construction is
    what happened at import in every spawned worker.
    """
    import rfproteina.utils.chirals as chirals

    def refuse(*_args: object, **_kwargs: object) -> list[np.ndarray]:
        raise AssertionError("construction must not enumerate")

    monkeypatch.setattr(chirals, "_enumerate_stereoisomer_templates", refuse)
    chirals._StereoisomerTemplates(IDEAL_POLYHEDRA)
