"""What an MCP client is told about a screen.

A verdict must not say more than was measured, and a state a person declared in
``axis_evidence.json`` has to read as declared wherever the verdict is shown.
These tests write ``screen.json`` themselves and do not run a screen, so they
hold whatever the screening code does. Each structured result is checked
against the output schema its own tool declares.
"""

import json
import os
from pathlib import Path

import jsonschema
import pytest

from tests.helpers.disease import UBIQUITIN
from tests.helpers.disease import load_mcp_server as _server

_ROOT = Path(__file__).resolve().parents[1]
_ROOTS = "RFD4_MCP_ROOTS"
_CURATED = {
    "screen_version": 3,
    "code": {"commit": "0123456789", "tree_state": "not_checked"},
    "call": "screened_clear",
    "bar": "sequence",
    "states": {"ada": "clear", "mhc_i": "clear"},
    "untested": [],
    "hits": [],
    "sequence_liability": {
        "ada": {
            "state": "clear",
            "measurement": 1.0,
            "blocker": None,
            "measured_state": "hit",
            "measured_blocker": None,
            "source": "axis_evidence",
            "curated": "basis_not_stated",
        },
        "mhc_i": {"state": "clear", "measurement": 0.1, "blocker": None, "source": "measured"},
    },
}
_PLAIN = {"call": "untested", "bar": "sequence", "states": {}, "untested": [], "hits": []}


def _screen_in(directory: Path, payload) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    (directory / "screen.json").write_text(text, encoding="utf-8")
    return directory


def _validated(tool: str, result):
    """The result as a client receives it, checked against the tool's own schema."""
    declared = {str(item["name"]): item for item in _server()._TOOLS}[tool]["outputSchema"]
    wire = json.loads(json.dumps(result, allow_nan=False))
    errors = [
        f"{list(error.absolute_path)}: {error.message}"
        for error in jsonschema.Draft202012Validator(declared).iter_errors(wire)
    ]
    assert not errors, errors
    return wire


def _quiet(monkeypatch, effect=None):
    """Stand in for the program, so a screen_directory call screens nothing."""
    from rfproteina.disease import mcp_api

    calls = []

    def called(argv, on_line=None):
        calls.append(list(argv))
        if effect is not None:
            effect()
        return 0, []

    monkeypatch.setattr(mcp_api, "_call_main", called)
    return calls


def test_a_curated_clear_is_not_reported_as_a_measured_one(tmp_path):
    """A clear a person declared over a measured hit used to arrive as a measured clear."""
    from rfproteina.disease.mcp_api import read_verdict

    _screen_in(tmp_path / "gene_1ABC_1", _CURATED)
    result = _validated("read_verdict", read_verdict({"directory": str(tmp_path)}))
    entry = result["runs"][0]
    assert "error" not in entry
    verdict = entry["verdict"]
    ada = verdict["sequence_liability"]["ada"]
    assert ada["state"] == "clear"
    assert ada["source"] == "axis_evidence"
    assert ada["measured_state"] == "hit"
    assert ada["curated"] == "basis_not_stated"
    assert "measured_blocker" in ada
    measured = verdict["sequence_liability"]["mhc_i"]
    assert measured["source"] == "measured"
    assert "measured_state" not in measured
    assert "curated" not in measured
    assert verdict["curated_axes"] == ["ada"], "the declared axis is named beside the call"


def test_what_a_person_recorded_behind_a_curated_state_is_returned(tmp_path):
    from rfproteina.disease.mcp_api import read_verdict

    record = json.loads(json.dumps(_CURATED))
    basis = {"reason": "assay 12 found no response", "source": "notebook, page 4"}
    record["sequence_liability"]["ada"]["curated"] = basis
    _screen_in(tmp_path / "gene_1ABC_1", record)
    result = _validated("read_verdict", read_verdict({"directory": str(tmp_path)}))
    assert result["runs"][0]["verdict"]["sequence_liability"]["ada"]["curated"] == basis


def test_a_screen_that_records_no_source_does_not_claim_nothing_was_curated(tmp_path):
    """An older record cannot say where a state came from, so the list is absent, not empty."""
    from rfproteina.disease.mcp_api import read_verdict

    record = dict(_PLAIN, sequence_liability={"ada": {"state": "untested", "blocker": "no_pass_cut"}})
    _screen_in(tmp_path / "gene_1ABC_1", record)
    verdict = read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]
    assert "curated_axes" not in verdict
    measured = dict(_PLAIN, sequence_liability={"ada": {"state": "clear", "source": "measured"}})
    _screen_in(tmp_path / "gene_1ABC_1", measured)
    assert read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]["curated_axes"] == []


def test_every_axis_group_reaches_the_verdict(tmp_path):
    """A nucleic or a ligand run used to arrive with a call and no axis record at all."""
    from rfproteina.disease import mcp_api
    from rfproteina.disease.mcp_api import read_verdict

    record = {
        "call": "screened_clear",
        "bar": "all",
        "states": {"mhc_i": "clear", "paralog": "clear", "innate_immune": "clear", "herg": "clear"},
        "untested": [],
        "hits": [],
        "sequence_liability": {"mhc_i": {"state": "clear", "measurement": 0.1, "source": "measured"}},
        "target_liability": {"paralog": {"state": "clear", "measurement": 0, "source": "measured"}},
        "nucleic_liability": {
            "innate_immune": {
                "state": "clear",
                "measurement": 4,
                "blocker": None,
                "measured_state": "hit",
                "measured_blocker": None,
                "source": "axis_evidence",
                "curated": "basis_not_stated",
            }
        },
        "toxicology": {"herg": {"state": "clear", "measurement": 25000.0, "source": "measured"}},
        "chemical_liability": {"deamidation_count": 2, "isoelectric_point": 6.1},
    }
    _screen_in(tmp_path / "gene_1ABC_1", record)
    result = _validated("read_verdict", read_verdict({"directory": str(tmp_path)}))
    verdict = result["runs"][0]["verdict"]
    assert verdict["sequence_liability"]["mhc_i"]["measurement"] == 0.1
    assert verdict["target_liability"]["paralog"]["source"] == "measured"
    assert verdict["nucleic_liability"]["innate_immune"]["measured_state"] == "hit"
    assert verdict["nucleic_liability"]["innate_immune"]["source"] == "axis_evidence"
    assert verdict["toxicology"]["herg"]["measurement"] == 25000.0
    assert verdict["curated_axes"] == ["innate_immune"]
    # Motif counts and charge for the first design. It has no state, so it is
    # not an axis GROUP -- the thing this test is about -- and must not be
    # summarised as one or counted among the curated axes. It is now returned
    # as a plain report field, the way the other no-call reports are, so the
    # assertion pins the property rather than the absence: the earlier
    # `"chemical_liability" not in verdict` conflated "not a group of axes"
    # with "not returned at all", and only the first is what this test guards.
    assert "chemical_liability" not in mcp_api._AXIS_GROUPS
    assert verdict["chemical_liability"] == {"deamidation_count": 2, "isoelectric_point": 6.1}
    assert "chemical_liability" not in verdict["curated_axes"]


def test_the_schema_declares_every_group_and_field_the_verdict_returns():
    from rfproteina.disease import mcp_api

    server = _server()
    declared = server._VERDICT_SCHEMA["properties"]
    groups = {name for name, shape in declared.items() if shape is server._AXIS_SCHEMA}
    assert groups == set(mcp_api._AXIS_GROUPS)
    assert set(mcp_api._VERDICT_FIELDS) <= set(declared)
    assert {"replaced", "curated_axes", "malformed", "record"} <= set(declared)
    assert set(server._AXIS_SCHEMA["additionalProperties"]["properties"]) == set(mcp_api._AXIS_FIELDS)
    assert {"source", "curated", "measured_state", "measured_blocker"} <= set(mcp_api._AXIS_FIELDS)
    by_name = {str(item["name"]): item for item in server._TOOLS}
    entry = by_name["screen_directory"]["outputSchema"]["properties"]["runs"]["items"]["properties"]
    assert {"written", "changed", "error"} <= set(entry)
    for tool in ("read_results", "backend_status"):
        assert "unreadable" in by_name[tool]["outputSchema"]["properties"]
    for tool in ("read_sequence_measures", "retrieve_structures", "retrieve_databases"):
        assert "unreadable" not in by_name[tool]["outputSchema"]["properties"]


def test_one_unrenderable_screen_does_not_lose_the_others(tmp_path, monkeypatch):
    """One file that parses and cannot be shown used to end the whole call."""
    import rfproteina.disease.screen as screen
    from rfproteina.disease.mcp_api import call_tool

    _screen_in(tmp_path / "gene_1ABC_1", _CURATED)
    _screen_in(tmp_path / "gene_3BAD_1", {"call": "screened_clear", "sequence_developability": {"sequence_pass": True}})
    deep = "[" * 3000 + "]" * 3000
    _screen_in(tmp_path / "gene_4DEEP_1", '{"call": "untested", "ppi_pass": ' + deep + "}")
    _screen_in(tmp_path / "gene_5DEEPER_1", "[" * 200_000 + "]" * 200_000)
    rendered = screen.format_screen_lines_for_directory

    def render(directory, payload):
        if directory.name == "gene_3BAD_1":
            raise KeyError("gravy")
        return rendered(directory, payload)

    monkeypatch.setattr(screen, "format_screen_lines_for_directory", render)
    result = _validated("read_verdict", call_tool("read_verdict", {"directory": str(tmp_path)}))
    runs = {entry["label"]: entry for entry in result["runs"]}
    assert set(runs) == {"gene_1ABC_1", "gene_3BAD_1", "gene_4DEEP_1", "gene_5DEEPER_1"}
    good = runs["gene_1ABC_1"]
    assert "error" not in good
    assert good["verdict"]["call"] == "screened_clear"
    assert good["lines"]
    bad = runs["gene_3BAD_1"]
    assert bad["lines"] == []
    assert bad["error"]
    assert bad["verdict"]["call"] == "screened_clear", "the stored call is still what the file says"
    for label in ("gene_4DEEP_1", "gene_5DEEPER_1"):
        assert runs[label]["error"]
        assert "verdict" not in runs[label]
        assert str(tmp_path) not in runs[label]["error"]


def test_screen_directory_keeps_the_other_runs_too(tmp_path, monkeypatch):
    import rfproteina.disease.screen as screen
    from rfproteina.disease import mcp_api

    _screen_in(tmp_path / "gene_1ABC_1", _CURATED)
    _screen_in(tmp_path / "gene_3BAD_1", _PLAIN)
    rendered = screen.format_screen_lines_for_directory

    def render(directory, payload):
        if directory.name == "gene_3BAD_1":
            raise KeyError("gravy")
        return rendered(directory, payload)

    monkeypatch.setattr(screen, "format_screen_lines_for_directory", render)
    _quiet(monkeypatch)
    result = _validated("screen_directory", mcp_api.screen_directory({"directory": str(tmp_path)}))
    runs = {entry["label"]: entry for entry in result["runs"]}
    assert runs["gene_1ABC_1"]["verdict"]["curated_axes"] == ["ada"]
    assert runs["gene_3BAD_1"]["error"]


def test_two_runs_under_home_are_not_the_same_directory(tmp_path):
    """A home-like folder inside a path used to be cut out, and two runs shared one name."""
    from rfproteina.disease.mcp_api import call_tool, read_verdict

    for owner in ("alice", "bob"):
        _screen_in(tmp_path / "home" / owner / "gene_1ABC_1", _PLAIN)
    result = _validated("read_verdict", read_verdict({"directory": str(tmp_path)}))
    shown = [entry["directory"] for entry in result["runs"]]
    assert len(shown) == 2
    assert len(set(shown)) == 2
    for directory in shown:
        assert (Path(directory) / "screen.json").is_file(), "the path can be given back to the server"
    one = tmp_path / "home" / "alice" / "gene_1ABC_1"
    assert call_tool("read_results", {"directory": str(one)})["directory"] == str(one)


def test_only_a_leading_personal_prefix_is_removed_from_a_path(monkeypatch, tmp_path):
    from rfproteina.disease.publish import scrub_for_upload, scrub_path_prefix

    assert scrub_path_prefix("/home/alice/runs/gene_1ABC_1") == "<home>/runs/gene_1ABC_1"
    assert scrub_path_prefix("/mnt/c/Users/alice/runs") == "<home>/runs"
    assert scrub_path_prefix("C:\\Users\\alice\\runs") == "<home>\\runs"
    assert scrub_path_prefix("/data/home/alice/gene_1ABC_1") == "/data/home/alice/gene_1ABC_1"
    assert scrub_path_prefix("disease_runs/gene_1ABC_1") == "disease_runs/gene_1ABC_1"
    monkeypatch.setenv("HOME", str(tmp_path / "ann"))
    assert scrub_path_prefix(str(tmp_path / "ann" / "runs")) == "<home>/runs"
    assert scrub_path_prefix(str(tmp_path / "anna" / "runs")) == str(tmp_path / "anna" / "runs")
    # Free text is still scrubbed wherever the path sits in it.
    assert scrub_for_upload("see /data/home/alice/x") == "see /data<home>/x"


def test_a_relative_path_cannot_leave_the_checkout(monkeypatch):
    """Joined without resolving, ../.. named any directory on the machine."""
    from rfproteina.disease import mcp_api

    monkeypatch.delenv(_ROOTS, raising=False)
    _quiet(monkeypatch)
    assert mcp_api._path("tests") == _ROOT / "tests"
    assert mcp_api._path("tests/../tests") == _ROOT / "tests"
    with pytest.raises(ValueError, match="cannot leave this checkout"):
        mcp_api._path("../..")
    with pytest.raises(ValueError, match="cannot leave this checkout"):
        mcp_api.read_verdict({"directory": "tests/../../.."})
    with pytest.raises(ValueError, match="cannot leave this checkout"):
        mcp_api.screen_directory({"directory": "../.."})
    for tool in ("read_results", "backend_status", "read_sequence_measures", "retrieve_structures"):
        with pytest.raises(ValueError, match="cannot leave this checkout"):
            mcp_api.call_tool(tool, {"directory": "../../etc"})
    for name in ("log_file", "out_dir", "proteome"):
        with pytest.raises(ValueError, match="cannot leave this checkout"):
            mcp_api.screen_directory({"directory": "tests", name: "../outside"})
    with pytest.raises(ValueError, match="cannot leave this checkout"):
        mcp_api.argv_from_options({"disease": "Asthma", "accept_only": True, "catalog": "../catalog.json"})


def test_a_linked_folder_inside_the_checkout_keeps_its_relative_name(tmp_path, monkeypatch):
    """Outputs kept on another volume and linked in are named the way they always were."""
    from rfproteina.disease import mcp_api

    checkout = tmp_path / "checkout"
    elsewhere = tmp_path / "volume" / "outputs"
    checkout.mkdir()
    elsewhere.mkdir(parents=True)
    (checkout / "outputs").symlink_to(elsewhere, target_is_directory=True)
    monkeypatch.setattr(mcp_api, "_repo_root", lambda: checkout)
    monkeypatch.delenv(_ROOTS, raising=False)
    assert mcp_api._path("outputs/run1") == checkout / "outputs" / "run1"
    assert mcp_api._path("outputs/../outputs/run1") == checkout / "outputs" / "run1"
    with pytest.raises(ValueError, match="cannot leave this checkout"):
        mcp_api._path("outputs/../../volume")
    # With the directories named, a link is followed, and this one leads outside them.
    monkeypatch.setenv(_ROOTS, str(tmp_path / "allowed"))
    with pytest.raises(ValueError, match="cannot leave this checkout"):
        mcp_api._path("outputs/run1")
    monkeypatch.setenv(_ROOTS, str(tmp_path / "volume"))
    assert mcp_api._path("outputs/run1") == (elsewhere / "run1").resolve()


def test_paths_outside_the_allowed_roots_are_refused(tmp_path, monkeypatch):
    """log_file and out_dir are written to, so a client could name any file for either."""
    from rfproteina.disease import mcp_api

    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    outside.mkdir()
    run = _screen_in(allowed / "gene_1ABC_1", _PLAIN)
    _screen_in(outside / "gene_2DEF_1", _PLAIN)
    monkeypatch.setenv(_ROOTS, os.pathsep.join(["relative/is/ignored", str(allowed)]))
    calls = _quiet(monkeypatch)
    for name in ("log_file", "out_dir", "proteome"):
        with pytest.raises(ValueError, match="outside the directories this server may use"):
            mcp_api.screen_directory({"directory": str(run), name: str(outside / "victim")})
    with pytest.raises(ValueError, match="outside the directories"):
        mcp_api.screen_directory({"directory": str(outside / "gene_2DEF_1")})
    with pytest.raises(ValueError, match="outside the directories"):
        mcp_api.read_verdict({"directory": str(outside)})
    for tool in ("read_results", "backend_status", "read_sequence_measures", "retrieve_databases"):
        with pytest.raises(ValueError, match="outside the directories"):
            mcp_api.call_tool(tool, {"directory": str(outside)})
    with pytest.raises(ValueError, match="outside the directories"):
        mcp_api.run_program({"options": {"screen": str(run), "log_file": str(outside / "victim")}})
    with pytest.raises(ValueError, match="outside the directories"):
        mcp_api.argv_from_options({"disease": "Asthma", "accept_only": True, "catalog": str(outside / "c.json")})
    # A link inside an allowed directory does not lead out of it.
    link = allowed / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the directories"):
        mcp_api.read_verdict({"directory": str(link)})
    with pytest.raises(ValueError, match="outside the directories"):
        mcp_api.screen_directory({"directory": str(run), "log_file": str(link / "victim")})
    assert calls == [], "a refused call does not reach the program"
    assert not (outside / "victim").exists()
    accepted = mcp_api.screen_directory(
        {"directory": str(run), "log_file": str(allowed / "mcp.log"), "out_dir": str(allowed / "out")}
    )
    _validated("screen_directory", accepted)
    assert accepted["exit_code"] == 0
    argv = calls[0]
    assert argv[argv.index("--log-file") + 1] == str(allowed / "mcp.log")
    assert argv[argv.index("--out-dir") + 1] == str(allowed / "out")
    assert mcp_api.read_verdict({"directory": str(allowed)})["runs"]
    assert mcp_api._path("tests") == _ROOT / "tests", "the checkout itself stays allowed"
    monkeypatch.setenv(_ROOTS, "relative/only")
    with pytest.raises(ValueError, match="outside the directories"):
        mcp_api.read_verdict({"directory": str(allowed)})
    monkeypatch.delenv(_ROOTS)
    assert mcp_api._path(str(outside)) == outside, "unset, an absolute path is used as given"
    assert mcp_api.read_verdict({"directory": str(outside)})["runs"]


def test_the_server_says_which_paths_it_may_use():
    from rfproteina.disease.mcp_api import list_options

    server = _server()
    assert _ROOTS in server._INSTRUCTIONS
    notes = " ".join(list_options()["notes"])
    assert _ROOTS in notes
    assert "cannot leave it" in notes
    assert "axis_evidence" in notes
    assert "axis_evidence" in server._INSTRUCTIONS


def test_one_unreadable_file_does_not_hide_the_tree(tmp_path):
    """A job file cut short by an interrupted write used to end the whole call."""
    from rfproteina.disease.mcp_api import call_tool

    _screen_in(tmp_path / "gene_1ABC_1", dict(_CURATED, ppi_pass=True))
    (tmp_path / "gene_9XYZ_1.job.json").write_text("{", encoding="utf-8")
    _screen_in(tmp_path / "gene_2DEF_1", "{")
    traces = tmp_path / "gene_1ABC_1" / "research_traces"
    traces.mkdir()
    (traces / "binder_0.json").write_bytes(b"\xff\xfe")
    (traces / "binder_1.json").write_text(json.dumps({"kind": "binder", "example_id": "binder_1"}), encoding="utf-8")
    result = _validated("read_results", call_tool("read_results", {"directory": str(tmp_path)}))
    assert result["status"] == "recorded"
    assert [screen["folder"] for screen in result["screens"]] == ["gene_1ABC_1"]
    assert [trace["file"] for trace in result["traces"]] == ["binder_1.json"]
    unreadable = {(item["folder"], item["file"]) for item in result["unreadable"]}
    assert unreadable == {
        ("", "gene_9XYZ_1.job.json"),
        ("gene_2DEF_1", "screen.json"),
        ("gene_1ABC_1", "binder_0.json"),
    }
    assert str(tmp_path) not in json.dumps(result["unreadable"])
    status = _validated("backend_status", call_tool("backend_status", {"directory": str(tmp_path)}))
    assert status["status"] == "recorded"
    assert len(status["unreadable"]) == 3
    assert "unreadable" not in _validated("backend_status", call_tool("backend_status", {}))
    # A tool that reads one directory still names the file and stops.
    with pytest.raises(ValueError, match="screen.json is not readable"):
        call_tool("read_sequence_measures", {"directory": str(tmp_path / "gene_2DEF_1")})


def test_a_tree_with_nothing_unreadable_says_so(tmp_path):
    from rfproteina.disease.mcp_api import call_tool

    _screen_in(tmp_path / "gene_1ABC_1", _PLAIN)
    result = _validated("read_results", call_tool("read_results", {"directory": str(tmp_path)}))
    assert result["unreadable"] == []


def test_read_results_names_the_axes_a_person_declared(tmp_path):
    """The call is shown there without its axis records, so the declared axes ride beside it."""
    from rfproteina.disease.mcp_api import call_tool

    _screen_in(tmp_path / "gene_1ABC_1", _CURATED)
    _screen_in(tmp_path / "gene_2DEF_1", _PLAIN)
    result = _validated("read_results", call_tool("read_results", {"directory": str(tmp_path)}))
    screens = {screen["folder"]: screen for screen in result["screens"]}
    assert screens["gene_1ABC_1"]["call"] == "screened_clear"
    assert screens["gene_1ABC_1"]["curated_axes"] == ["ada"]
    assert screens["gene_2DEF_1"]["curated_axes"] == "not_recorded"


def test_a_screen_with_wrong_types_still_validates(tmp_path):
    """A hand-edited file used to make the server return what its own schema rejects."""
    from rfproteina.disease.mcp_api import call_tool

    _screen_in(tmp_path / "gene_1ABC_1", _CURATED)
    _screen_in(
        tmp_path / "gene_2ODD_1",
        {
            "call": "screened_clear",
            "bar": 3,
            "states": "all good",
            "hits": [1, 2],
            "untested": {"immunogenicity": 1},
            "screen_version": "three",
            "code": "abc",
            "replaced": [1],
            "pre_oracle_pass": True,
            "sequence_liability": {
                "immunogenicity": {"state": 7, "measurement": 0.3, "source": 5},
                "ada": "clear",
            },
            "toxicology": "none",
        },
    )
    result = _validated("read_verdict", call_tool("read_verdict", {"directory": str(tmp_path), "include_record": True}))
    runs = {entry["label"]: entry for entry in result["runs"]}
    odd = runs["gene_2ODD_1"]["verdict"]
    assert odd["call"] == "screened_clear"
    assert odd["pre_oracle_pass"] is True
    assert odd["sequence_liability"] == {"immunogenicity": {"measurement": 0.3}}
    assert set(odd["malformed"]) == {
        "bar",
        "states",
        "hits",
        "untested",
        "screen_version",
        "code",
        "replaced",
        "sequence_liability.immunogenicity.state",
        "sequence_liability.immunogenicity.source",
        "sequence_liability.ada",
        "toxicology",
    }
    for name in ("bar", "states", "hits", "untested", "screen_version", "code", "replaced", "toxicology"):
        assert name not in odd
    assert odd["record"]["bar"] == 3, "the record asked for is the file, not a repaired one"
    assert "malformed" not in runs["gene_1ABC_1"]["verdict"]
    _validated("read_results", call_tool("read_results", {"directory": str(tmp_path)}))
    _validated("backend_status", call_tool("backend_status", {"directory": str(tmp_path)}))
    _validated(
        "read_sequence_measures", call_tool("read_sequence_measures", {"directory": str(tmp_path / "gene_2ODD_1")})
    )


def test_the_verdict_names_its_version_and_what_it_replaced(tmp_path):
    """A stored verdict is of unknown age unless it says which record it is."""
    from rfproteina.disease.mcp_api import read_verdict

    record = dict(
        _CURATED,
        replaced={
            "digest": "0011223344556677",
            "call": "axis_hit",
            "changed_inputs": [{"file": "axis_evidence.json", "change": "added"}],
        },
    )
    _screen_in(tmp_path / "gene_1ABC_1", record)
    _screen_in(tmp_path / "gene_2DEF_1", dict(_CURATED, replaced={"digest": "0011223344556677", "call": "untested"}))
    _screen_in(tmp_path / "gene_3OLD_1", _PLAIN)
    result = _validated("read_verdict", read_verdict({"directory": str(tmp_path)}))
    runs = {entry["label"]: entry["verdict"] for entry in result["runs"]}
    verdict = runs["gene_1ABC_1"]
    assert verdict["screen_version"] == 3
    assert verdict["code"] == {"commit": "0123456789", "tree_state": "not_checked"}
    assert verdict["replaced"] == {
        "call": "axis_hit",
        "changed_inputs": [{"file": "axis_evidence.json", "change": "added"}],
    }
    assert runs["gene_2DEF_1"]["replaced"] == {"call": "untested"}
    for name in ("screen_version", "code", "replaced"):
        assert name not in runs["gene_3OLD_1"], "an older record does not gain a version it never had"


def test_screen_directory_can_name_the_bar(tmp_path, monkeypatch):
    """The bar decides which axes must be clear, and a client could not set it."""
    from rfproteina.disease import mcp_api

    run = _screen_in(tmp_path / "gene_1ABC_1", _PLAIN)
    calls = _quiet(monkeypatch)
    mcp_api.screen_directory({"directory": str(run), "bar": "plasma"})
    argv = calls[0]
    assert argv[argv.index("--bar") + 1] == "plasma"
    with pytest.raises(ValueError, match="bar"):
        mcp_api.screen_directory({"directory": str(run), "bar": 3})
    declared = {str(item["name"]): item for item in _server()._TOOLS}["screen_directory"]["inputSchema"]
    assert set(declared["properties"]) == set(mcp_api.SCREEN_FIELDS)
    assert declared["properties"]["bar"]["type"] == "string"


def test_written_is_what_this_call_wrote_and_changed_is_whether_the_bytes_moved(tmp_path, monkeypatch):
    """The modification time alone misses an atomic replace inside one timestamp."""
    from rfproteina.disease import mcp_api
    from rfproteina.disease.publish import write_text_atomically

    run = _screen_in(tmp_path / "gene_1ABC_1", _PLAIN)
    screen = run / "screen.json"

    def replace_within_one_tick(text):
        held = screen.stat()
        write_text_atomically(screen, text)
        os.utime(screen, ns=(held.st_atime_ns, held.st_mtime_ns))
        assert screen.stat().st_mtime_ns == held.st_mtime_ns

    _quiet(monkeypatch)
    untouched = _validated("screen_directory", mcp_api.screen_directory({"directory": str(run)}))["runs"][0]
    assert (untouched["written"], untouched["changed"]) == (False, False)

    _quiet(monkeypatch, lambda: replace_within_one_tick(screen.read_text(encoding="utf-8")))
    same = _validated("screen_directory", mcp_api.screen_directory({"directory": str(run)}))["runs"][0]
    assert (same["written"], same["changed"]) == (True, False), "rewritten, and the verdict is the same bytes"

    moved = json.dumps(dict(_PLAIN, states={"ada": "clear"}))
    _quiet(monkeypatch, lambda: replace_within_one_tick(moved))
    other = _validated("screen_directory", mcp_api.screen_directory({"directory": str(run)}))["runs"][0]
    assert (other["written"], other["changed"]) == (True, True)
    assert other["verdict"]["states"] == {"ada": "clear"}

    fresh = tmp_path / "gene_2DEF_1"
    fresh.mkdir()
    _quiet(monkeypatch, lambda: _screen_in(fresh, _PLAIN))
    first = mcp_api.screen_directory({"directory": str(fresh)})["runs"][0]
    assert (first["written"], first["changed"]) == (True, True), "a screen that was not there before"


def test_a_replace_within_one_timestamp_is_still_seen_as_written(tmp_path, monkeypatch):
    from rfproteina.disease import mcp_api
    from rfproteina.disease.publish import write_text_atomically

    run = _screen_in(tmp_path / "gene_1ABC_1", _PLAIN)
    screen = run / "screen.json"

    def rescreen():
        held = screen.stat()
        write_text_atomically(screen, json.dumps(dict(_PLAIN, hits=["ada"], call="axis_hit")))
        os.utime(screen, ns=(held.st_atime_ns, held.st_mtime_ns))

    _quiet(monkeypatch, rescreen)
    entry = mcp_api.screen_directory({"directory": str(run)})["runs"][0]
    assert entry["verdict"]["call"] == "axis_hit"
    assert entry["written"] is True


def test_an_upper_case_title_keeps_its_words(tmp_path):
    """Nine upper-case letters is most of a legacy structure title, not a peptide."""
    from rfproteina.disease.mcp_api import _redact, call_tool

    title = "CRYSTAL STRUCTURE OF HUMAN KALLIKREIN 7"
    detail = "RESIDUES 15-23 OF THE ACETYLTRANSFERASE DOMAIN"
    (tmp_path / "returned_structures.json").write_text(
        json.dumps(
            {
                "structures": [
                    {"entry_id": "2QXI", "chain": "A", "group": "gene", "title": title},
                    {"entry_id": "1UBQ", "chain": "A", "group": "gene", "title": f"UBIQUITIN {UBIQUITIN}"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "target_research.json").write_text(
        json.dumps(
            {
                "subjects": [
                    {
                        "identifier": "P49862",
                        "sources": {"uniprot": {"status": "record", "detail": detail}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    structures = _validated("retrieve_structures", call_tool("retrieve_structures", {"directory": str(tmp_path)}))
    assert structures["structures"][0]["title"] == title
    assert structures["structures"][1]["title"] == "UBIQUITIN [sequence omitted]"
    databases = _validated("retrieve_databases", call_tool("retrieve_databases", {"directory": str(tmp_path)}))
    assert databases["subjects"][0]["sources"]["uniprot"]["detail"] == detail
    assert UBIQUITIN not in json.dumps([structures, databases])
    # A value of a record is not prose, and the nine-residue rule still holds there.
    assert _redact("SIINFEKLV") == "[sequence omitted]"
    assert _redact("KALLIKREIN") == "[sequence omitted]"


def test_the_stored_measures_of_a_curated_screen_validate(tmp_path):
    from rfproteina.disease.mcp_api import call_tool

    record = dict(
        _CURATED,
        sequence_developability={"gravy": -0.4, "instability_index": 31.0, "sequence_pass": True},
        samples=[{"example_id": "binder_0", "contact_ms": 512.5, "sequence": UBIQUITIN}],
    )
    run = _screen_in(tmp_path / "gene_1ABC_1", record)
    measures = _validated("read_sequence_measures", call_tool("read_sequence_measures", {"directory": str(run)}))
    assert measures["sequence_developability"]["gravy"] == -0.4
    assert UBIQUITIN not in json.dumps(measures)
    status = _validated("backend_status", call_tool("backend_status", {"directory": str(run)}))
    assert status["unreadable"] == []
