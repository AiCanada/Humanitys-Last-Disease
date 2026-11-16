"""The program's modes and the artifacts a run shares with its neighbours.

``cli._run`` dispatches on three things: a ``screen``, a ``refold``, or the
disease pipeline. The ``run`` tool's guards knew the first and the last, so a
refold call -- which names no disease and asks no question -- was refused for
not naming a disease, and the mode was reachable only by passing a disease the
program would then ignore. Its result is one printed JSON document, which the
line capture handed back as fragments.

The other two tests here are about what a run directory does not hold alone.
The generation archive copies a finished run, screen and all, so a walk returns
the copy beside the run and nothing said which was which. The target lookup and
the structure list are written once per output directory, so the two tools that
read them answered ``not_recorded`` for the very runs they were performed for.
"""

import json
from pathlib import Path

import jsonschema
import pytest

from tests.helpers.disease import load_mcp_server as _server


def _run_directory(folder: Path, name: str = "gene_1ABC_1") -> Path:
    run = folder / name
    (run / "inference_samples" / "binder_0").mkdir(parents=True)
    return run


def _screen_in(run: Path, call: str = "axis_hit") -> None:
    record = {
        "program": "Humanitys-Last-Disease",
        "call": call,
        "bar": "sequence",
        "states": {"mhc_i": "hit"},
        "hits": ["mhc_i"],
        "untested": [],
    }
    (run / "screen.json").write_text(json.dumps(record), encoding="utf-8")


# --- the refold mode -------------------------------------------------------


def test_a_refold_call_is_not_refused_for_naming_no_disease(tmp_path: Path) -> None:
    """The mode reads a finished run and prompts for nothing, so neither guard applies."""
    from rfproteina.disease import mcp_api

    run = _run_directory(tmp_path)
    payload = mcp_api.run_program(
        {"options": {"refold": str(run), "log_file": str(tmp_path / "mcp.log")}},
    )
    assert payload["exit_code"] == 0, payload["messages"]
    report = payload["refold"]
    assert report["oracle"] == "af3", "the oracle name the folds are filed under"
    assert "status" in report, "the round trip's own state is the point of the call"
    assert "requests" in report, "an export says where it wrote the requests"


def test_the_refold_record_is_returned_as_an_object_not_as_printed_lines(tmp_path: Path) -> None:
    """It is printed as JSON because the sequel to the command is another command."""
    from rfproteina.disease import mcp_api

    run = _run_directory(tmp_path)
    payload = mcp_api.run_program(
        {"options": {"refold": str(run), "log_file": str(tmp_path / "mcp.log")}},
    )
    assert isinstance(payload["refold"], dict)
    # The lines it was read from are still there, so nothing is only structured.
    assert any(line.startswith("{") or line.startswith('"') for line in payload["messages"])
    schema = {str(tool["name"]): tool.get("outputSchema") for tool in _server()._TOOLS}["run"]
    jsonschema.validate(payload, schema)


def test_a_refold_only_option_without_refold_is_refused(tmp_path: Path) -> None:
    """Nothing but the refold mode reads any of them, so elsewhere they configure nothing."""
    from rfproteina.disease import mcp_api

    for name in ("refold_dir", "refold_import", "refold_oracle"):
        with pytest.raises(ValueError, match="need refold"):
            mcp_api.run_program({"options": {name: str(tmp_path), "disease": "Asthma", "accept_only": True}})


def test_refold_takes_a_finished_run_and_not_a_directory_above_one(tmp_path: Path) -> None:
    """A bare dot named the checkout root, and every guard let it through.

    ``_path`` resolves a relative path against the checkout, so the program was
    handed a directory that is not a run: it made ``refold_requests`` beside
    the source tree and reported nothing in it. A parent of runs is the same
    story, because the export reads designs from ``<run>/inference_samples``.
    """
    from rfproteina.disease import mcp_api

    run = _run_directory(tmp_path)
    with pytest.raises(ValueError, match="finished generation directory"):
        mcp_api.run_program({"options": {"refold": "."}})
    with pytest.raises(ValueError, match="finished generation directory"):
        mcp_api.run_program({"options": {"refold": str(tmp_path)}})
    # The run itself is taken: it holds inference_samples, which is the test
    # the screen applies.
    payload = mcp_api.run_program(
        {"options": {"refold": str(run), "log_file": str(tmp_path / "mcp.log")}},
    )
    assert payload["exit_code"] == 0


def test_the_oracle_name_cannot_reach_out_of_the_run(tmp_path: Path) -> None:
    """It is a name, not a path, and it is written into every filed filename.

    ``PathOracle.output_stem`` builds ``<example>_<oracle>`` and joins it under
    the run, so a separator in the name places the folds somewhere else. Every
    path option goes through ``_path``; this one is not a path and would not.
    """
    from rfproteina.disease import mcp_api

    run = _run_directory(tmp_path)
    for name in ("../../elsewhere", "a/b", "..", "", "with space"):
        with pytest.raises(ValueError, match="plain name"):
            mcp_api.run_program({"options": {"refold": str(run), "refold_oracle": name}})
    payload = mcp_api.run_program(
        {"options": {"refold": str(run), "refold_oracle": "af3_v2.1", "log_file": str(tmp_path / "mcp.log")}},
    )
    assert payload["refold"]["oracle"] == "af3_v2.1", "an ordinary name still works"


def test_a_screen_and_a_refold_in_one_call_are_refused(tmp_path: Path) -> None:
    """The program's modes are mutually exclusive; the parser would say so as exit code 2."""
    from rfproteina.disease import mcp_api

    with pytest.raises(ValueError, match="one mode"):
        mcp_api.run_program({"options": {"screen": str(tmp_path), "refold": str(tmp_path)}})


def test_the_pipeline_still_demands_a_disease_and_an_answer(monkeypatch) -> None:
    """Relaxing the guard for two modes must not relax it for the one that asks."""
    from rfproteina.disease import mcp_api

    def refused(_argv, **_kwargs):
        raise AssertionError("the program was called")

    monkeypatch.setattr(mcp_api, "_call_main", refused)
    with pytest.raises(ValueError, match="refold a finished run"):
        mcp_api.run_program({"options": {}})
    with pytest.raises(ValueError, match="accept_only"):
        mcp_api.run_program({"options": {"disease": "Asthma"}})


def test_a_document_the_capture_mangled_is_not_a_document_lost() -> None:
    """A record that will not parse leaves the printed lines as the answer."""
    from rfproteina.disease.mcp_api import _printed_json

    assert _printed_json(["{", '"run": "x",', '"status": {', '"requested": 1', "}", "}"]) == {
        "run": "x",
        "status": {"requested": 1},
    }
    assert _printed_json(["Prepared 3 designs.", "not json at all"]) is None
    assert _printed_json(["{", "truncated"]) is None
    assert _printed_json([]) is None


# --- the generation archive ------------------------------------------------


def test_an_archived_copy_says_so_and_carries_its_own_id(tmp_path: Path) -> None:
    """The walk returns the copy beside the run; only the copy is marked."""
    from rfproteina.disease import mcp_api
    from rfproteina.disease.archive import save_generation

    live = _run_directory(tmp_path)
    _screen_in(live)
    copy = save_generation(
        tmp_path,
        live,
        disease_name="asthma",
        disease_id="MONDO:0004979",
        target_id="IL5",
        group="gene",
        entry_id="1ABC",
        chain="1",
        binder_kind="protein",
    )
    assert (copy / "screen.json").is_file(), "the archive copies the verdict with the run"
    runs = {str(run["label"]): run for run in mcp_api.read_verdict({"directory": str(tmp_path)})["runs"]}
    assert len(runs) == 2, "one run and one copy of it"
    original = runs["gene_1ABC_1"]
    archived = next(run for label, run in runs.items() if label != "gene_1ABC_1")
    assert "archived" not in original, "the run the copy was taken from is not a copy"
    assert archived["archived"] is True
    assert isinstance(archived["archive_run_id"], str) and archived["archive_run_id"]


def test_the_marked_answer_still_validates_against_its_own_schema(tmp_path: Path) -> None:
    from rfproteina.disease import mcp_api
    from rfproteina.disease.archive import save_generation

    live = _run_directory(tmp_path)
    _screen_in(live)
    save_generation(
        tmp_path,
        live,
        disease_name="asthma",
        disease_id="MONDO:0004979",
        target_id="IL5",
        group="gene",
        entry_id="1ABC",
        chain="1",
        binder_kind="protein",
    )
    answer = mcp_api.read_verdict({"directory": str(tmp_path)})
    schema = {str(tool["name"]): tool.get("outputSchema") for tool in _server()._TOOLS}["read_verdict"]
    jsonschema.validate(answer, schema)


def test_a_live_run_that_names_its_target_in_a_record_is_not_a_copy(tmp_path: Path) -> None:
    """``generation.json`` is also an input a person is told to write.

    ``docs/reference/screen-inputs.md`` answers the ``target_not_named``
    blocker with "supply generation.json or a job file that does", and the
    screen reads one from any run. A test on the filename alone called such a
    run an archived copy -- and a client doing what this flag's description
    tells it then drops the only copy of that run from its tally.
    """
    from rfproteina.disease.archive import GENERATION_ARCHIVE
    from rfproteina.disease.mcp_api import _archive_copy

    live = _run_directory(tmp_path)
    (live / "generation.json").write_text(
        json.dumps({"entry_id": "8TLD", "chain": "A", "target_id": "IL13"}), encoding="utf-8"
    )
    assert _archive_copy(live) == {}, "a hand-written record is not an archive"

    # Under the archive's own root, with a record that cannot be read: still a
    # copy. Marking only the copies that carry a run_id would leave the double
    # count unfixed for every copy written before the archive recorded one.
    copy = tmp_path / GENERATION_ARCHIVE / "asthma" / "IL5" / "gene_1ABC_1"
    copy.mkdir(parents=True)
    (copy / "generation.json").write_text("{ not json", encoding="utf-8")
    assert _archive_copy(copy) == {"archived": True}

    # The archive root with no record beside the run, and a bare directory.
    bare = tmp_path / GENERATION_ARCHIVE / "asthma" / "IL5" / "gene_2ABC_1"
    bare.mkdir(parents=True)
    assert _archive_copy(bare) == {}
    assert _archive_copy(tmp_path) == {}


# --- artifacts written once per output directory ---------------------------


def _lookup(folder: Path) -> None:
    body = {
        "subjects": [
            {
                "identifier": "IL5",
                "sources": {"uniprot": {"status": "recorded", "detail": "P05113"}},
                "sources_with_record": 1,
            }
        ]
    }
    (folder / "target_research.json").write_text(json.dumps(body), encoding="utf-8")


def test_a_run_is_answered_from_its_output_directory_only_when_asked(tmp_path: Path) -> None:
    """The default is pinned elsewhere on purpose, so this is what the opt-in changes.

    ``test_disease_mcp_layers.py::test_a_missing_lookup_is_not_recorded`` writes a
    lookup in a parent and requires the child to answer not_recorded: a lookup
    covering an output directory is not a measurement of one run in it. The
    capability was simply unreachable, not wrong, so it is a request now.
    """
    from rfproteina.disease import mcp_layers

    run = _run_directory(tmp_path)
    _lookup(tmp_path)
    silent = mcp_layers.retrieve_databases({"directory": str(run)})
    assert silent["status"] == mcp_layers.NOT_RECORDED, "the default must not read a parent's lookup"
    assert "read_from" not in silent
    assert silent["databases"], "the names it could have asked are still listed"
    asked = mcp_layers.retrieve_databases({"directory": str(run), "from_output_dir": True})
    assert asked["status"] == "recorded"
    assert asked["subjects"][0]["identifier"] == "IL5"
    assert asked["read_from"].endswith(tmp_path.name), "where it came from has to travel with it"


def test_a_run_holding_its_own_lookup_does_not_climb(tmp_path: Path) -> None:
    """The search starts where it was told to, so nothing above it is consulted."""
    from rfproteina.disease import mcp_layers

    run = _run_directory(tmp_path)
    _lookup(run)
    answer = mcp_layers.retrieve_databases({"directory": str(run), "from_output_dir": True})
    assert answer["status"] == "recorded"
    assert "read_from" not in answer


def test_the_structure_list_is_found_beside_the_output_directory_too(tmp_path: Path) -> None:
    from rfproteina.disease import mcp_layers

    run = _run_directory(tmp_path)
    body = {"structures": [{"entry_id": "1ABC", "chain": "A", "group": "gene", "databases": 3, "predicted": False}]}
    (tmp_path / "returned_structures.json").write_text(json.dumps(body), encoding="utf-8")
    assert mcp_layers.retrieve_structures({"directory": str(run)})["status"] == mcp_layers.NOT_RECORDED
    answer = mcp_layers.retrieve_structures({"directory": str(run), "from_output_dir": True})
    assert answer["status"] == "recorded"
    assert answer["structures"][0]["entry_id"] == "1ABC"
    assert answer["read_from"].endswith(tmp_path.name)
    asked = mcp_layers.retrieve_databases({"directory": str(run), "from_output_dir": True})
    counts = asked["structure_database_counts"]
    assert counts["counts"] == [3]
    # One answer, one provenance: read_from is the field the schema declares,
    # and it must speak for whatever the answer was actually read from.
    assert asked["read_from"].endswith(tmp_path.name)
    assert "read_from" not in counts, "a provenance buried in an undeclared block is one a client will not read"


def test_the_climb_reaches_the_output_directory_from_an_archived_run(tmp_path: Path) -> None:
    """The archive nests a copy four deep, which is what the bound is for."""
    from rfproteina.disease import mcp_layers

    deep = tmp_path / "generations" / "asthma" / "IL5" / "gene_1ABC_1"
    (deep / "inference_samples").mkdir(parents=True)
    _lookup(tmp_path)
    answer = mcp_layers.retrieve_databases({"directory": str(deep), "from_output_dir": True})
    assert answer["status"] == "recorded"
    assert answer["read_from"].endswith(tmp_path.name)


def test_an_unknown_argument_is_still_refused(tmp_path: Path) -> None:
    """The opt-in widens the allowlist by exactly one name."""
    from rfproteina.disease import mcp_layers

    run = _run_directory(tmp_path)
    with pytest.raises(ValueError, match="Unknown option"):
        mcp_layers.retrieve_databases({"directory": str(run), "from_parent": True})
    with pytest.raises(ValueError, match="true or false"):
        mcp_layers.retrieve_databases({"directory": str(run), "from_output_dir": "yes"})


# --- the reason a measurement is missing ------------------------------------


def _compressed_screen(run: Path) -> None:
    """What the screen writes for a run whose designs were written compressed.

    ``_designs_blocker`` answers designs_compressed, and ``_with_reason`` puts
    that on the sequence and interface blocks, which are otherwise null.
    """
    record = {
        "program": "Humanitys-Last-Disease",
        "call": "untested",
        "bar": "sequence",
        "states": {},
        "hits": [],
        "untested": [],
        "sequence_developability": {
            "gravy": None,
            "instability_index": None,
            "sequence_pass": "absent",
            "sequence_blocker": "designs_compressed",
        },
        "designed_interface": {
            "contact_ms": None,
            "clash_fraction": None,
            "interface_blocker": "designs_compressed",
        },
        "samples": [],
    }
    (run / "screen.json").write_text(json.dumps(record), encoding="utf-8")


def test_a_missing_measurement_arrives_with_the_reason_the_screen_recorded(tmp_path: Path) -> None:
    """Three ways to reach a null gravy, and they used to be one silence."""
    from rfproteina.disease import mcp_layers

    run = _run_directory(tmp_path)
    _compressed_screen(run)
    answer = mcp_layers.read_sequence_measures({"directory": str(run)})
    block = answer["sequence_developability"]
    reason = block["sequence_blocker"]
    assert reason == "designs_compressed", "designs never written and a chain never named both read as null"
    assert block["gravy"] is None
    # And the answer no longer claims a measurement. A stored null is not a
    # number, and neither is the text absent the screen writes for a verdict it
    # could not reach, so a block of three of them is not a recorded one.
    assert answer["status"] == mcp_layers.NOT_RECORDED, "every number was null and the status said recorded"


def test_the_reason_is_not_what_reports_a_measurement(tmp_path: Path) -> None:
    """A screen whose block holds only the reason must not read as recorded.

    The stored nulls still flip the status, which is the half of this that
    changes a stored answer and is left for the owner to decide. Pinned here so
    the current behaviour is visible rather than assumed.
    """
    from rfproteina.disease import mcp_layers

    run = _run_directory(tmp_path)
    (run / "screen.json").write_text(
        json.dumps(
            {
                "program": "Humanitys-Last-Disease",
                "call": "untested",
                "bar": "sequence",
                "states": {},
                "hits": [],
                "untested": [],
                "sequence_developability": {"sequence_blocker": "designs_absent"},
                "samples": [],
            }
        ),
        encoding="utf-8",
    )
    answer = mcp_layers.read_sequence_measures({"directory": str(run)})
    assert answer["sequence_developability"]["sequence_blocker"] == "designs_absent"
    assert answer["status"] == mcp_layers.NOT_RECORDED, "the reason alone is not a measurement"


def test_a_sample_carries_its_own_three_reasons(tmp_path: Path) -> None:
    """row_blocker is about the metrics row, sequence_blocker the chain, interface_blocker the surface."""
    from rfproteina.disease import mcp_layers

    run = _run_directory(tmp_path)
    (run / "screen.json").write_text(
        json.dumps(
            {
                "program": "Humanitys-Last-Disease",
                "call": "untested",
                "bar": "sequence",
                "states": {},
                "hits": [],
                "untested": [],
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
                        "sequence_blocker": "structure_unreadable",
                        "interface_blocker": "binder_chain_absent",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sample = mcp_layers.read_sequence_measures({"directory": str(run)})["samples"][0]
    assert sample["row_blocker"] == "metrics_row_ambiguous"
    assert sample["sequence_blocker"] == "structure_unreadable"
    assert sample["interface_blocker"] == "binder_chain_absent"
    # The same allowlist feeds read_results, so the reasons travel there too.
    results = mcp_layers.read_results({"directory": str(run)})
    assert results["screens"][0]["samples"][0]["row_blocker"] == "metrics_row_ambiguous"


# --- what read_results says about a verdict ---------------------------------


def test_the_results_view_is_the_verdict_view_less_what_it_holds_back() -> None:
    """One list, derived. A third hand-written list is how the first two drifted."""
    from rfproteina.disease import mcp_api, mcp_layers

    fields = set(mcp_layers.screen_fields())
    assert fields == set(mcp_api._VERDICT_FIELDS) - mcp_layers._SCREEN_OMITTED
    stale = mcp_layers._SCREEN_OMITTED.difference(mcp_api._VERDICT_FIELDS)
    assert not stale, f"_SCREEN_OMITTED holds back {sorted(stale)}, which the verdict view does not return anyway"


def test_a_verdict_from_read_results_can_be_checked_and_dated(tmp_path: Path) -> None:
    """It returned a call with no axes, no bar axes, no cuts, no code, no version."""
    from rfproteina.disease import mcp_layers

    run = _run_directory(tmp_path)
    (run / "screen.json").write_text(
        json.dumps(
            {
                "program": "Humanitys-Last-Disease",
                "screen_version": 3,
                "call": "screened_clear",
                "bar": "sequence",
                "bar_axes": ["mhc_i", "ada"],
                "states": {"mhc_i": "clear", "ada": "clear"},
                "hits": [],
                "untested": [],
                "requested_class": "requested_clear",
                "code": {"commit": "0" * 40, "tree_state": "clean"},
                "cuts": {"gravy_max": 0.1, "gravy_max_source": "program_convention"},
                "metrics_file": "inference_metrics/per_example_metrics.csv",
                "solubility": {"solubility_blocker": "solubility_not_configured"},
                "ppi_pass": "absent",
                "ppi_blocker": "oracle_not_configured",
                "pre_oracle_pass": True,
            }
        ),
        encoding="utf-8",
    )
    screen = mcp_layers.read_results({"directory": str(run)})["screens"][0]
    assert screen["call"] == "screened_clear"
    assert screen["states"] == {"mhc_i": "clear", "ada": "clear"}, "a call with no axes behind it is unreadable"
    assert screen["bar_axes"] == ["mhc_i", "ada"], "which axes that bar required, at the moment of the verdict"
    assert screen["cuts"]["gravy_max_source"] == "program_convention"
    assert screen["screen_version"] == 3 and screen["code"]["tree_state"] == "clean"
    assert screen["hits"] == [] and screen["untested"] == []


def test_an_optional_key_the_screen_left_absent_is_not_reported_as_unrecorded(tmp_path: Path) -> None:
    """screen.py keeps its optional keys absent rather than null, and means it."""
    from rfproteina.disease import mcp_layers

    run = _run_directory(tmp_path)
    _screen_in(run)
    screen = mcp_layers.read_results({"directory": str(run)})["screens"][0]
    for name in ("job_mismatch", "design_diversity", "metrics_blocker", "chain_blocker", "target_accession"):
        assert name not in screen, f"{name} is absent-or-present; masking it loses what the absence says"
    # A required key that is genuinely missing still reports as unrecorded.
    assert screen["screen_version"] == mcp_layers.NOT_RECORDED
    assert screen["solubility"] == mcp_layers.NOT_RECORDED


def test_the_refold_summary_travels_instead_of_the_criteria_table(tmp_path: Path) -> None:
    """Four kilobytes of criteria per run is not a summary; the counts are."""
    from rfproteina.disease import mcp_layers

    run = _run_directory(tmp_path)
    (run / "screen.json").write_text(
        json.dumps(
            {
                "program": "Humanitys-Last-Disease",
                "call": "untested",
                "bar": "sequence",
                "states": {},
                "hits": [],
                "untested": [],
                "refold": {
                    "criteria": [{"name": "ppi", "state": "not_evaluated"}, {"name": "unconditional", "state": "pass"}],
                    "declared": 11,
                    "evaluated": 1,
                    "criteria_file": "configs/metrics/pass_criteria/all_pass_criteria.yaml",
                },
            }
        ),
        encoding="utf-8",
    )
    screen = mcp_layers.read_results({"directory": str(run)})["screens"][0]
    assert screen["refold"] == {
        "declared": 11,
        "evaluated": 1,
        "criteria_file": "configs/metrics/pass_criteria/all_pass_criteria.yaml",
        "criteria": 2,
    }, "the counts, and the criteria as a number rather than a table"
    assert mcp_layers._refold_summary(None) == mcp_layers.NOT_RECORDED


# --- what the climb may reach ----------------------------------------------


def test_the_climb_stops_where_the_server_may_not_read(tmp_path: Path, monkeypatch) -> None:
    """``_path`` is where containment is enforced; a lexical walk left it behind.

    With RFD4_MCP_ROOTS set -- the mode whose whole purpose is containment --
    the four-step climb read files that ``_path`` itself refuses to open.
    """
    from rfproteina.disease import mcp_layers

    allowed = tmp_path / "allowed"
    run = allowed / "gene_1ABC_1"
    (run / "inference_samples").mkdir(parents=True)
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(allowed))

    # Above the allowed root: the climb must not see it.
    _lookup(tmp_path)
    refused = mcp_layers.retrieve_databases({"directory": str(run), "from_output_dir": True})
    assert refused["status"] == mcp_layers.NOT_RECORDED, "the climb read a file _path would have refused to open"
    assert "read_from" not in refused

    # Inside the allowed root: that is what the opt-in is for.
    _lookup(allowed)
    found = mcp_layers.retrieve_databases({"directory": str(run), "from_output_dir": True})
    assert found["status"] == "recorded"
    assert found["read_from"].endswith("allowed")


def test_the_bound_follows_what_path_promises(tmp_path: Path, monkeypatch) -> None:
    """Three modes, three boundaries."""
    from rfproteina.disease import mcp_layers
    from rfproteina.disease.mcp_api import _repo_root

    checkout = _repo_root()
    monkeypatch.delenv("RFD4_MCP_ROOTS", raising=False)
    # A relative argument is promised it cannot leave the checkout, and every
    # relative path resolves inside it.
    assert mcp_layers._climb_bound(checkout / "disease_runs") == (checkout,)
    # An absolute directory elsewhere: _path allows the caller to name its
    # parent outright, so the step count is the only bound that means anything.
    assert mcp_layers._climb_bound(tmp_path / "elsewhere") is None
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    assert mcp_layers._climb_bound(tmp_path / "elsewhere") == (checkout, tmp_path)


# --- the printed record -----------------------------------------------------


def test_the_printed_record_is_parsed_before_it_is_scrubbed() -> None:
    """Reassembling it from the redacted lines returned something that was not printed.

    The free-text scrubber replaces a home segment wherever it sits, so a path
    came back in a form that names no directory, and the nine-residue rule
    turned an example id into the words "sequence omitted".
    """
    from rfproteina.disease.mcp_api import _printed_json

    lines = [
        "{",
        '"oracle": "af3",',
        '"run": "/home/ann/runs/gene_8TLD_2",',
        '"requests": "/home/ann/runs/gene_8TLD_2/refold_requests",',
        '"status": {',
        '"outstanding": ["KALLIKREIN_A_0", "TRANSFERASE_B_1"],',
        '"requested": 2',
        "}",
        "}",
    ]
    record = _printed_json(lines)
    assert record["oracle"] == "af3"
    # The personal prefix goes, as it does for every other path this server
    # returns, and the rest of the path stays a path.
    assert record["run"] == "<home>/runs/gene_8TLD_2", "a client's next command needs this to name a directory"
    assert record["requests"].endswith("gene_8TLD_2/refold_requests")
    outstanding = record["status"]["outstanding"]
    assert outstanding == ["KALLIKREIN_A_0", "TRANSFERASE_B_1"], "an example id is not a sequence"
    assert record["status"]["requested"] == 2


def test_an_integral_float_is_the_integer_the_schema_promised() -> None:
    """JSON has one number type, and the published schema says integer."""
    from rfproteina.disease.mcp_api import argv_from_options

    assert argv_from_options({"choice": 2.0}) == ["--choice", "2"]
    assert argv_from_options({"plasma_hub_max": 10.0}) == ["--plasma-hub-max", "10"]
    with pytest.raises(ValueError, match="integer"):
        argv_from_options({"choice": 2.5})
    # A bool is refused a step earlier, before the integer branch is reached.
    with pytest.raises(ValueError, match="wrong type"):
        argv_from_options({"choice": True})


# --- the described surface -------------------------------------------------


def test_the_architecture_names_every_tool_this_server_serves() -> None:
    """A client that calls this to learn the surface was told about two fewer tools."""
    from rfproteina.disease.mcp_layers import describe_architecture

    payload = describe_architecture()
    named: set[str] = set()
    for layer in payload["layers"]:
        tool = layer.get("tool")
        if isinstance(tool, str):
            named.add(tool)
        for capability in layer.get("capabilities", ()):
            if isinstance(capability.get("tool"), str):
                named.add(capability["tool"])
    served = {str(tool["name"]) for tool in _server()._TOOLS}
    assert served - named == {"describe_architecture"}, "every tool but this call itself is named here"
