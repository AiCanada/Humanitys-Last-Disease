"""The shape of a stored research trace, pinned beside the version that describes it.

``TRACE_VERSION`` exists for the reason ``SCREEN_VERSION`` does: a reader can
tell which record it is holding rather than inferring it from which keys happen
to be present. Nothing pinned it, and the screening block then fell eight
verdict fields and three per-design reasons behind the record it copies from --
with the number still reading 1.

That matters more here than it does for the screen. The keys this block was
missing are blockers, and a blocker is absent when it does not apply, so a
trace with no ``screening.sample.interface_blocker`` either came from a writer
that never knew the key or from one that did and had nothing to report. Only
the version tells those apart, and ``artifact_digest`` cannot: a narrow writer
beside a current screen produces a perfectly matching digest with the fields
missing.

So this file does not rely on memory. The lists are pinned here as literals
beside the number. If you are here because one of these failed, that is the
test working: add the key to the list in ``trace.py``, add it to the matching
literal below, and raise ``TRACE_VERSION``.
"""

import json
from pathlib import Path

from rfproteina.disease import mcp_layers
from rfproteina.disease.trace import (
    _SAMPLE_FIELDS,
    _SCREEN_FIELDS,
    TRACE_VERSION,
    _screening,
    restamp_screening,
)

#: Raised whenever either list below changes. See the module docstring.
_EXPECTED_VERSION = 2

#: The verdict fields copied into ``screening``.
_SCREEN = (
    "bar",
    "bar_axes",
    "call",
    "code",
    "hits",
    "ppi_blocker",
    "ppi_pass",
    "pre_oracle_pass",
    "requested_class",
    "screen_version",
    "states",
    "untested",
)
#: One design's measurements, and the reasons a measurement is missing.
_SAMPLE = (
    "ca_rmsd",
    "clash_fraction",
    "contact_ms",
    "example_id",
    "gravy",
    "instability_index",
    "interface_blocker",
    "row_blocker",
    "sequence_blocker",
    "sequence_identity",
    "sequence_pass",
)


def test_the_version_matches_the_shape_pinned_here() -> None:
    """Raise TRACE_VERSION in the same edit that changes either list below."""
    assert TRACE_VERSION == _EXPECTED_VERSION


def test_the_copied_verdict_fields_are_the_ones_pinned_here() -> None:
    assert tuple(sorted(_SCREEN_FIELDS)) == _SCREEN


def test_the_copied_sample_fields_are_the_ones_pinned_here() -> None:
    assert tuple(sorted(_SAMPLE_FIELDS)) == _SAMPLE


def test_the_trace_and_the_server_read_one_sample_the_same_way() -> None:
    """Two lists over one record, and the stored one had fallen behind the view.

    The server's list is the same record read for a client. Keeping them equal
    is what stops a reason reaching an MCP reader while the stored artifact
    beside the designs says nothing.
    """
    assert tuple(sorted(_SAMPLE_FIELDS)) == tuple(sorted(mcp_layers._SAMPLE_FIELDS))


def test_the_verdict_fields_are_all_fields_the_screen_writes() -> None:
    """A field copied here that no screen writes would be stored as not_recorded forever."""
    from rfproteina.disease.screen import SCREEN_OPTIONAL_KEYS, SCREEN_REQUIRED_KEYS

    unknown = set(_SCREEN_FIELDS) - set(SCREEN_REQUIRED_KEYS) - set(SCREEN_OPTIONAL_KEYS)
    assert not unknown, f"the trace copies {sorted(unknown)}, which screen.py does not write"


def test_the_bulky_block_is_not_copied_into_every_design() -> None:
    """``refold`` is four fifths of a verdict, and this file is written once per design.

    ``mcp_layers._SCREEN_OMITTED`` holds it back from a tool that walks a tree
    for exactly that reason; a trace is written beside every design and copied
    into the archive with them.
    """
    assert "refold" not in _SCREEN_FIELDS


def _run(folder: Path) -> Path:
    run = folder / "gene_1ABC_1"
    (run / "research_traces").mkdir(parents=True)
    return run


def _screen_record() -> dict:
    return {
        "call": "axis_hit",
        "bar": "sequence",
        "bar_axes": ["mhc_i", "ada"],
        "states": {"mhc_i": "hit", "ada": "clear"},
        "hits": ["mhc_i"],
        "untested": [],
        "screen_version": 3,
        "code": {"commit": "0" * 40, "tree_state": "clean"},
        "requested_class": "requested_clear",
        "ppi_pass": "absent",
        "ppi_blocker": "oracle_not_configured",
        "pre_oracle_pass": True,
        "samples": [
            {
                "example_id": "binder_0",
                "contact_ms": None,
                "clash_fraction": None,
                "gravy": None,
                "instability_index": None,
                "sequence_pass": "absent",
                "sequence_identity": None,
                "ca_rmsd": None,
                "row_blocker": "metrics_row_ambiguous",
                "sequence_blocker": "designs_compressed",
                "interface_blocker": "designs_compressed",
            }
        ],
    }


def test_the_copied_block_carries_the_verdict_and_the_reasons() -> None:
    """The stored block held a false pass flag beside two nulls and named no cause."""
    screening = _screening(_screen_record(), "binder_0", {})
    assert screening["bar_axes"] == ["mhc_i", "ada"], "a bar with no axes claims whatever that bar names today"
    assert screening["states"] == {"mhc_i": "hit", "ada": "clear"}, "a verdict with no axes behind it"
    assert screening["screen_version"] == 3
    assert screening["code"]["tree_state"] == "clean"
    sample = screening["sample"]
    assert sample["row_blocker"] == "metrics_row_ambiguous"
    assert sample["sequence_blocker"] == "designs_compressed"
    assert sample["interface_blocker"] == "designs_compressed"


def test_a_mapping_is_copied_as_a_mapping() -> None:
    """``_plain`` answers not_recorded for a dict, which stored those words where the axes were."""
    screening = _screening(_screen_record(), "binder_0", {})
    for name in ("states", "code"):
        assert isinstance(screening[name], dict), f"{name} was stored as the text not_recorded"


def test_the_copied_block_holds_nothing_the_lists_do_not_name() -> None:
    """The shape pin itself: a key added to the writer and not to the list above fails here."""
    screening = _screening(_screen_record(), "binder_0", {})
    declared = set(_SCREEN) | {"status", "method", "artifact", "artifact_digest", "note", "sample"}
    assert set(screening) == declared, "raise TRACE_VERSION and pin the new key above"
    # The sample block adds its own status and the two pass flags it reads from
    # this design's metrics row rather than from the screen.
    assert set(screening["sample"]) <= set(_SAMPLE) | {"status", "pre_oracle_pass", "ppi_pass"}


def _trace_file(run: Path, version: int) -> Path:
    """A stored trace of the shape ``_stored_trace`` accepts."""
    path = run / "research_traces" / "binder_0.json"
    record = {
        "version": version,
        "kind": "binder",
        "example_id": "binder_0",
        "screening": {"status": "not_recorded", "artifact": "screen.json"},
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_a_restamped_trace_says_which_shape_it_now_is(tmp_path: Path) -> None:
    """A rewritten block under the old number is the version lying the worst way.

    ``restamp_screening`` rebuilds only the screening block, so a reader keying
    on the number would skip fields that are there, and could not tell a trace
    it refreshed from one it never touched.
    """
    run = _run(tmp_path)
    path = _trace_file(run, version=1)
    assert restamp_screening(run, _screen_record()) == [path]
    refreshed = json.loads(path.read_text(encoding="utf-8"))
    assert refreshed["version"] == TRACE_VERSION
    assert refreshed["screening"]["states"] == {"mhc_i": "hit", "ada": "clear"}
    assert refreshed["screening"]["sample"]["sequence_blocker"] == "designs_compressed"


def test_a_trace_that_cannot_be_read_keeps_its_own_number(tmp_path: Path) -> None:
    """One it never touched must not claim the current shape."""
    run = _run(tmp_path)
    path = run / "research_traces" / "binder_0.json"
    path.write_text(json.dumps({"version": 1, "kind": "binder"}), encoding="utf-8")
    assert restamp_screening(run, _screen_record()) == []
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1
