"""The three MCP layers, and what backend_status says about a directory.

These tests read stored files and do not start generation. The
backend_status half was a separate file until its subject turned out to
be this one: a reason is never a measurement there, and every test of it
checks the top-level answer as well as the component, because one
component saying recorded makes the whole call say it.
"""

import json
from pathlib import Path

import pytest

from tests.helpers.disease import ONE_OF_EACH
from tests.helpers.disease import load_mcp_server as _server

_PEPTIDE = ONE_OF_EACH


def _keys(value, found):
    if isinstance(value, dict):
        for key, item in value.items():
            found.append(key)
            _keys(item, found)
    elif isinstance(value, list):
        for item in value:
            _keys(item, found)


def _component(result, name):
    return next(item for item in result["components"] if item["id"] == name)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_describe_architecture_names_the_three_layers(monkeypatch):
    from rfproteina.disease.mcp_api import call_tool
    from rfproteina.disease.research import MODEL_CHOICES

    probe_key = "sk-" + "testfixture" * 3
    monkeypatch.setenv("XAI_API_KEY", probe_key)
    result = call_tool("describe_architecture", {})
    layers = {layer["id"]: layer for layer in result["layers"]}
    assert list(layers) == ["ai_interface", "mcp_integration", "computational_backend"]
    providers = layers["ai_interface"]["providers"]
    assert [item["name"] for item in providers] == ["OpenAI", "Claude", "xAI", "Other compatible models"]
    assert providers[0]["provider"] == "openai"
    assert providers[2]["provider"] == "grok"
    assert providers[2]["name"] == "xAI"
    assert [model["id"] for model in providers[0]["models"]] == [item[0] for item in MODEL_CHOICES["openai"]]
    assert [model["id"] for model in providers[1]["models"]] == [item[0] for item in MODEL_CHOICES["claude"]]
    assert [model["id"] for model in providers[2]["models"]] == [item[0] for item in MODEL_CHOICES["grok"]]
    other = providers[3]
    assert other["providers"] == ["nvidia", "openrouter"]
    expected = [("nvidia", model_id) for model_id, _label in MODEL_CHOICES["nvidia"]]
    expected += [("openrouter", model_id) for model_id, _label in MODEL_CHOICES["openrouter"]]
    assert [(model["provider"], model["id"]) for model in other["models"]] == expected
    tools = [item["tool"] for item in layers["mcp_integration"]["capabilities"]]
    # list_options and read_verdict were missing, so a client that called this
    # to learn the surface was never told that the one tool returning a verdict
    # with its per-axis evidence exists.
    assert tools == [
        "list_options",
        "retrieve_databases",
        "retrieve_structures",
        "read_sequence_measures",
        "run",
        "screen_directory",
        "read_verdict",
        "read_results",
        "declare_corpora",
    ]
    docking = next(item for item in layers["computational_backend"]["components"] if item["id"] == "docking")
    assert docking["status"] == "not_performed"
    assert docking["note"] == (
        "Docking pose search is unavailable. AutoDock Vina can score supplied poses during screening."
    )
    encoded = json.dumps(result)
    names = []
    _keys(result, names)
    assert "sequence" not in names
    assert probe_key not in encoded
    assert "sk-" not in encoded
    assert "XAI_API_KEY" not in encoded
    assert "api.x.ai" not in encoded
    assert "not a cure" not in encoded
    assert "API keys are not accepted." in layers["ai_interface"]["notes"]
    assert "A model reply is not a measurement." in layers["ai_interface"]["notes"]
    assert result["program"] == "Humanitys-Last-Disease"


def test_retrieve_databases_reads_stored_statuses_and_drops_the_narrative(tmp_path):
    from rfproteina.disease.mcp_api import call_tool
    from rfproteina.disease.research import SOURCE_ORDER

    _write(
        tmp_path / "target_research.json",
        {
            "subjects": [
                {
                    "identifier": "P00533",
                    "sources_with_record": 4,
                    "narrative": _PEPTIDE,
                    "models": {"grok": _PEPTIDE},
                    "generation_inputs": ["nope"],
                    "sources": {
                        "uniprot": {"status": "record", "detail": f"saw {_PEPTIDE} there"},
                        "ncbi": {"status": "no record"},
                    },
                }
            ]
        },
    )
    _write(
        tmp_path / "returned_structures.json",
        {"version": 1, "structures": [{"entry_id": "1EGF", "chain": "A", "databases": 2}]},
    )
    result = call_tool("retrieve_databases", {"directory": str(tmp_path)})
    subject = result["subjects"][0]
    assert result["status"] == "recorded"
    assert subject["identifier"] == "P00533"
    assert subject["sources_with_record"] == 4
    assert subject["sources"]["uniprot"]["status"] == "record"
    assert subject["sources"]["uniprot"]["detail"] == "saw [sequence omitted] there"
    assert subject["sources"]["ncbi"]["status"] == "no record"
    assert subject["sources"]["string"]["status"] == "not_recorded"
    assert set(subject["sources"]) == set(SOURCE_ORDER)
    assert "narrative" not in subject
    assert "models" not in subject
    assert "generation_inputs" not in subject
    assert result["structure_database_counts"]["counts"] == [2]
    assert "name" not in result["structure_database_counts"]
    encoded = json.dumps(result)
    assert _PEPTIDE not in encoded
    names = []
    _keys(result, names)
    assert "sequence" not in names


def test_a_missing_lookup_is_not_recorded(tmp_path):
    from rfproteina.disease.mcp_api import call_tool

    child = tmp_path / "child"
    child.mkdir()
    _write(
        tmp_path / "target_research.json",
        {
            "subjects": [
                {"identifier": "P00533", "sources_with_record": 1, "sources": {"uniprot": {"status": "record"}}}
            ]
        },
    )
    result = call_tool("retrieve_databases", {"directory": str(child)})
    assert result["status"] == "not_recorded"
    assert result["subjects"] == []
    assert len(result["databases"]) == 8
    assert {item["status"] for item in result["databases"]} == {"not_recorded"}
    assert "sources_with_record" not in json.dumps(result)
    assert "structure_database_counts" not in result
    assert result["program"] == "Humanitys-Last-Disease"


def test_retrieve_structures_returns_rows_without_a_sequence(tmp_path):
    from rfproteina.disease.mcp_api import call_tool

    _write(
        tmp_path / "returned_structures.json",
        {
            "version": 1,
            "structures": [
                {
                    "entry_id": "1EGF",
                    "chain": "A",
                    "accession": "P00533",
                    "group": "gene",
                    "title": "EGF receptor",
                    "predicted": False,
                    "source": "rcsb",
                    "kind": "experimental",
                    "program_score": 4,
                    "databases": 2,
                    "sequence": _PEPTIDE,
                },
                {
                    "entry_id": "1EGF",
                    "chain": "A",
                    "group": "gene",
                    "title": "duplicate",
                },
                {
                    "entry_id": "1DNA",
                    "chain": "B",
                    "group": "gene",
                    "predicted": "yes",
                    "source": "alphafold",
                    "kind": "predicted",
                },
            ],
        },
    )
    _write(
        tmp_path / "generation.json",
        {
            "disease_name": "Example",
            "entry_id": "1EGF",
            "chain": "A",
            "binder_kind": "protein",
            "sequence": _PEPTIDE,
        },
    )
    result = call_tool("retrieve_structures", {"directory": str(tmp_path)})
    assert result["status"] == "recorded"
    assert [row["entry_id"] for row in result["structures"]] == ["1EGF", "1DNA"]
    assert result["structures"][0]["title"] == "EGF receptor"
    assert result["structures"][0]["predicted"] is False
    assert result["structures"][1]["predicted"] == "not_recorded"
    assert result["generations"][0]["disease_name"] == "Example"
    assert result["generations"][0]["binder_kind"] == "protein"
    encoded = json.dumps(result)
    assert _PEPTIDE not in encoded
    assert "program_score" not in encoded
    names = []
    _keys(result, names)
    assert "sequence" not in names


def test_read_sequence_measures_returns_gravy_and_drops_the_sequence(tmp_path):
    from rfproteina.disease.mcp_api import call_tool

    _write(
        tmp_path / "screen.json",
        {
            "sequence_developability": {
                "gravy": -0.0359,
                "instability_index": 30.0,
                "sequence_pass": True,
                "sequence": _PEPTIDE,
            },
            "samples": [{"example_id": "binder_0", "gravy": -0.0359, "sequence": _PEPTIDE, "contact_ms": 1.0}],
        },
    )
    result = call_tool("read_sequence_measures", {"directory": str(tmp_path)})
    assert result["sequence_developability"]["gravy"] == -0.0359
    assert result["samples"][0]["gravy"] == -0.0359
    assert result["samples"][0]["example_id"] == "binder_0"
    assert result["note"] == "The sequence text is not returned."
    bare = tmp_path / "bare"
    bare.mkdir()
    empty = call_tool("read_sequence_measures", {"directory": str(bare)})
    assert empty["status"] == "not_recorded"
    encoded = json.dumps(result)
    assert _PEPTIDE not in encoded
    names = []
    _keys(result, names)
    assert "sequence" not in names


def test_read_results_returns_the_trace_and_scrubs_the_home_path(tmp_path):
    from rfproteina.disease.mcp_api import call_tool

    home_fixture = "/".join(("", "home", tmp_path.name))
    _write(
        tmp_path / "research_traces" / "binder_0.json",
        {"kind": "binder", "example_id": "binder_0", "note": "kept", "sequence": _PEPTIDE},
    )
    _write(
        tmp_path / "generation.json",
        {
            "disease_name": "Example",
            "disease_id": "MONDO_1",
            "target_id": "P00533",
            "group": "gene",
            "entry_id": "1EGF",
            "chain": "A",
            "binder_kind": "protein",
            "source": f"{home_fixture}/disease_runs/gene_1EGF_A",
            "saved_at": "2026-01-01T00:00:00+00:00",
            "sequence": _PEPTIDE,
        },
    )
    _write(
        tmp_path / "gene_1EGF_A.job.json",
        {
            "class_name": "requested_clear",
            "binder_kind": "protein",
            "target_id": "P00533",
            "generation_started": False,
            "annotated_cif": f"{home_fixture}/gene_1EGF_A.cif",
            "design_spec": f"{home_fixture}/gene_1EGF_A.yaml",
        },
    )
    _write(
        tmp_path / "screen.json",
        {"call": "untested", "bar": "sequence", "ppi_pass": "absent", "sequence": _PEPTIDE},
    )
    result = call_tool("read_results", {"directory": str(tmp_path)})
    assert result["status"] == "recorded"
    assert result["complete"] is True
    assert result["traces"][0]["example_id"] == "binder_0"
    assert result["traces"][0]["note"] == "kept"
    assert result["generations"][0]["disease_name"] == "Example"
    assert result["generations"][0]["source"] == "<home>/disease_runs/gene_1EGF_A"
    assert result["jobs"][0]["class_name"] == "requested_clear"
    assert result["screens"][0]["call"] == "untested"
    encoded = json.dumps(result)
    assert _PEPTIDE not in encoded
    assert home_fixture not in encoded
    names = []
    _keys(result, names)
    assert "sequence" not in names
    assert "annotated_cif" not in names
    assert "design_spec" not in names


def test_backend_status_without_a_directory_does_not_invent_a_number():
    from rfproteina.disease.mcp_api import call_tool

    result = call_tool("backend_status", {})
    assert "directory" not in result
    assert result["status"] == "not_recorded"
    docking = _component(result, "docking")
    gpu = _component(result, "gpu")
    assert docking == {
        "id": "docking",
        "name": "Docking",
        "status": "not_performed",
        "note": "Docking pose search is unavailable. AutoDock Vina can score supplied poses during screening.",
    }
    assert gpu == {
        "id": "gpu",
        "name": "GPU processing",
        "status": "not_performed",
        "note": "This call does not start GPU work.",
    }
    assert _component(result, "modified_rfd4")["status"] == "not_recorded"
    assert not any(char.isdigit() for char in json.dumps(docking))
    # The three components that read the screens: with no directory there are
    # none, so each must be exactly its own four keys. Pinned by equality
    # because every disclosure added to them is conditional, and a key that
    # appeared here would mean one of those conditions fires on nothing.
    assert _component(result, "structural_prediction") == {
        "id": "structural_prediction",
        "name": "Structural prediction",
        "status": "not_performed",
        "note": "Structural prediction is not launched here.",
    }
    assert _component(result, "binding_assessment") == {
        "id": "binding_assessment",
        "name": "Binding assessment",
        "status": "not_performed",
        "note": "No stored contact or PPI measurement is in this directory.",
    }
    assert _component(result, "candidate_ranking") == {
        "id": "candidate_ranking",
        "name": "Candidate ranking",
        "status": "not_performed",
        "note": "No binder trace is stored.",
    }


def test_backend_status_ranks_stored_confidence_and_reports_the_experiment(tmp_path):
    from rfproteina.disease.mcp_api import call_tool

    def trace(example, plddt, experiment, checkpoint):
        return {
            "kind": "binder",
            "example_id": example,
            "generative_model": {"command": {"experiment": experiment, "checkpoint_name": checkpoint}},
            "model_confidence": {"status": "recorded", "metrics": {"plddt": plddt}},
        }

    original = tmp_path / "original"
    _write(original / "research_traces" / "a.json", trace("a", 10, "inference_on_user_inputs", "release.ckpt"))
    _write(original / "research_traces" / "b.json", trace("b", 30, "inference_on_user_inputs", "release.ckpt"))
    ranked = call_tool("backend_status", {"directory": str(original)})
    ranking = _component(ranked, "candidate_ranking")
    assert ranking["metric"] == "plddt"
    assert [row["example_id"] for row in ranking["order"]] == ["b", "a"]
    assert [row["value"] for row in ranking["order"]] == [30, 10]
    assert _component(ranked, "modified_rfd4")["status"] == "not_performed"
    assert _component(ranked, "modified_rfd4")["experiment"] == "inference_on_user_inputs"
    assert _component(ranked, "modified_rfd4")["checkpoint_name"] == "release.ckpt"
    assert _component(ranked, "docking")["status"] == "not_performed"
    assert "contact_ms" not in _component(ranked, "docking")

    modified = tmp_path / "modified"
    _write(modified / "research_traces" / "binder_0.json", trace("binder_0", 12, "disease_binder", "modified.ckpt"))
    reported = call_tool("backend_status", {"directory": str(modified)})
    component = _component(reported, "modified_rfd4")
    assert component["status"] == "recorded"
    assert component["experiment"] == "disease_binder"
    assert component["checkpoint_name"] == "modified.ckpt"
    assert _component(reported, "docking")["status"] == "not_performed"


def test_unknown_arguments_and_tools_are_refused(tmp_path):
    from rfproteina.disease.mcp_api import call_tool

    with pytest.raises(ValueError, match="Unknown option"):
        call_tool("describe_architecture", {"directory": str(tmp_path)})
    with pytest.raises(ValueError, match="directory"):
        call_tool("retrieve_databases", {})
    with pytest.raises(ValueError, match="Unknown option"):
        call_tool("read_sequence_measures", {"directory": str(tmp_path), "sequence": _PEPTIDE})
    with pytest.raises(ValueError, match="Unknown tool"):
        call_tool("execute_design", {})
    assert call_tool("backend_status", {})["program"] == "Humanitys-Last-Disease"
    assert call_tool("describe_architecture", {})["program"] == "Humanitys-Last-Disease"
    with pytest.raises(ValueError):
        call_tool("backend_status", {"directory": ""})
    with pytest.raises(ValueError):
        call_tool("backend_status", {"directory": 3})


def test_a_truncated_result_walk_is_incomplete(tmp_path, monkeypatch):
    from rfproteina.disease.mcp_api import call_tool
    from rfproteina.disease.screen import ScreenWalkTruncatedError

    def _boom(_directory):
        raise ScreenWalkTruncatedError([], 1)

    monkeypatch.setattr("rfproteina.disease.screen.walk_for_screens", _boom)
    _write(
        tmp_path / "generation.json",
        {"disease_name": "Example", "entry_id": "1EGF", "chain": "A", "binder_kind": "protein"},
    )
    result = call_tool("read_results", {"directory": str(tmp_path)})
    assert result["complete"] is False
    assert result["generations"][0]["disease_name"] == "Example"


def test_a_huge_or_broken_file_names_only_the_file(tmp_path):
    from rfproteina.disease.mcp_api import call_tool

    huge = tmp_path / "target_research.json"
    huge.write_bytes(b"{" + b" " * 2_000_000)
    with pytest.raises(ValueError, match="target_research.json is too large to return") as raised:
        call_tool("retrieve_databases", {"directory": str(tmp_path)})
    assert str(tmp_path) not in str(raised.value)
    broken = tmp_path / "bare"
    broken.mkdir()
    (broken / "screen.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="screen.json is not readable") as unreadable:
        call_tool("read_sequence_measures", {"directory": str(broken)})
    assert str(broken) not in str(unreadable.value)


def test_an_empty_lookup_file_is_not_a_recorded_lookup(tmp_path):
    """A file with no subject in it is not a lookup, and said so until it existed.

    ``retrieve_databases`` set ``status`` to recorded the moment
    ``target_research.json`` was a readable file, and in the same breath dropped
    the ``databases`` list and the note saying none of them was consulted. An
    empty or non-object file therefore came back as a recorded lookup with an
    empty subject list and nothing left to explain it.
    """
    from rfproteina.disease.mcp_api import call_tool

    for name, body in (("blank", {}), ("array", []), ("rows", {"subjects": []})):
        room = tmp_path / name
        _write(room / "target_research.json", body)
        result = call_tool("retrieve_databases", {"directory": str(room)})
        assert result["status"] == "not_recorded", name
        assert result["subjects"] == []
        assert len(result["databases"]) == 8
        assert "no stored lookup" in result["note"]

    held = tmp_path / "held"
    _write(
        held / "target_research.json",
        {"subjects": [{"identifier": "P00533", "sources": {"uniprot": {"status": "no record"}}}]},
    )
    kept = call_tool("retrieve_databases", {"directory": str(held)})
    # One subject that found nothing is still a lookup that was performed.
    assert kept["status"] == "recorded"
    assert "databases" not in kept
    assert "note" not in kept


def test_a_screen_with_no_sequence_measurement_is_not_recorded(tmp_path):
    """``read_sequence_measures`` reported recorded for a screen holding none.

    The status followed the existence of ``screen.json`` rather than the
    presence of a measurement, so a verdict written before the sequence was
    read came back as recorded with gravy, the instability index, the sequence
    verdict and the sample list all empty.
    """
    from rfproteina.disease.mcp_api import call_tool

    bare = tmp_path / "bare"
    _write(bare / "screen.json", {"call": "untested", "bar": "sequence"})
    empty = call_tool("read_sequence_measures", {"directory": str(bare)})
    assert empty["status"] == "not_recorded"
    assert empty["sequence_developability"]["gravy"] == "not_recorded"
    assert empty["samples"] == []

    scored = tmp_path / "scored"
    _write(scored / "screen.json", {"sequence_developability": {"gravy": -0.2}})
    assert call_tool("read_sequence_measures", {"directory": str(scored)})["status"] == "recorded"
    sampled = tmp_path / "sampled"
    _write(sampled / "screen.json", {"samples": [{"example_id": "binder_0"}]})
    assert call_tool("read_sequence_measures", {"directory": str(sampled)})["status"] == "recorded"


def test_candidate_ranking_puts_the_best_design_first_on_an_error_metric(tmp_path):
    """The order was descending whatever the metric meant.

    ``min_ipae_coi`` is a predicted aligned error: this repository's own
    ``configs/metrics/pass_criteria/all_pass_criteria.yaml`` bounds it with
    ``max`` and bounds ``ptm_coi`` with ``min``, so lower is better on the one
    and higher on the other. ``_ranking`` picked the alphabetically first
    shared metric -- which is the ipae one for any PPI trace -- and sorted it
    descending, so the worst candidate was listed first under a component
    called Candidate ranking.
    """
    from rfproteina.disease.mcp_api import call_tool

    def trace(example, ipae, ptm):
        return {
            "kind": "binder",
            "example_id": example,
            "model_confidence": {
                "status": "recorded",
                "metrics": {"oracle_ppi.min_ipae_coi": ipae, "oracle_ppi.ptm_coi": ptm},
            },
        }

    _write(tmp_path / "research_traces" / "weak.json", trace("weak", 7.5, 41.0))
    _write(tmp_path / "research_traces" / "strong.json", trace("strong", 0.8, 92.0))
    ranking = _component(call_tool("backend_status", {"directory": str(tmp_path)}), "candidate_ranking")
    assert ranking["metric"] == "oracle_ppi.min_ipae_coi"
    assert ranking["metric_direction"] == "lower_is_better"
    assert [row["example_id"] for row in ranking["order"]] == ["strong", "weak"]
    assert [row["value"] for row in ranking["order"]] == [0.8, 7.5]

    # A confidence metric still ranks the highest number first, and says so.
    higher = tmp_path / "higher"
    _write(
        higher / "research_traces" / "a.json",
        {"kind": "binder", "example_id": "a", "model_confidence": {"metrics": {"plddt": 10}}},
    )
    _write(
        higher / "research_traces" / "b.json",
        {"kind": "binder", "example_id": "b", "model_confidence": {"metrics": {"plddt": 30}}},
    )
    confident = _component(call_tool("backend_status", {"directory": str(higher)}), "candidate_ranking")
    assert confident["metric_direction"] == "higher_is_better"
    assert [row["example_id"] for row in confident["order"]] == ["b", "a"]


def test_a_cut_title_is_not_returned_as_the_whole_title(tmp_path):
    """A structure title past the cap came back as its first 300 characters.

    Nothing in the result said it had been cut, so a caller comparing the title
    against a deposit, or quoting it, was quoting a truncation as the record.
    """
    from rfproteina.disease.mcp_api import call_tool

    long = "Crystal structure of the " + ("very " * 80) + "important kinase domain"
    _write(
        tmp_path / "returned_structures.json",
        {"version": 1, "structures": [{"entry_id": "1ABC", "chain": "A", "title": long}]},
    )
    row = call_tool("retrieve_structures", {"directory": str(tmp_path)})["structures"][0]
    assert len(long) > 300
    assert row["title"].endswith(" [cut]")
    assert len(row["title"]) <= 300

    short = tmp_path / "short"
    _write(
        short / "returned_structures.json",
        {"version": 1, "structures": [{"entry_id": "1ABC", "chain": "A", "title": "EGF receptor"}]},
    )
    kept = call_tool("retrieve_structures", {"directory": str(short)})["structures"][0]
    assert kept["title"] == "EGF receptor"


def test_the_server_tool_list_matches_the_program():
    from rfproteina.disease.mcp_api import TOOL_NAMES

    server = _server()
    names = [tool["name"] for tool in server._TOOLS]
    assert names == list(TOOL_NAMES)
    assert names[:4] == ["list_options", "run", "screen_directory", "read_verdict"]
    assert "not a cure" not in json.dumps(server._TOOLS) + server._INSTRUCTIONS
    assert "Do not shell out" in server._INSTRUCTIONS


def _binder(example, **metrics):
    return {
        "kind": "binder",
        "example_id": example,
        "model_confidence": {"status": "recorded", "metrics": metrics},
    }


def _screen(**fields):
    record = {"program": "Humanitys-Last-Disease", "call": "untested", "bar": "sequence"}
    record.update(fields)
    return record


def _refold_block(criteria, **fields):
    record = {
        "declared": 11,
        "criteria_file": "configs/metrics/pass_criteria/all_pass_criteria.yaml",
        "criteria": criteria,
    }
    record.update(fields)
    return record


def _thresholds(count, **fields):
    bound = {"bound": "max", "cut": 1.5, "measured": 2.0, "met": False}
    bound.update(fields)
    return [{"metric": f"column_{index}", **bound} for index in range(count)]


# --- the ranking and what the screen says about its candidates ---------------


def test_candidate_ranking_discloses_eight_names_for_one_molecule(tmp_path: Path) -> None:
    """Eight separately ranked candidates, beside a screen recording one molecule."""
    from rfproteina.disease.diversity import diversity_phrase
    from rfproteina.disease.mcp_api import call_tool

    spread = {
        "method": "positional_identity",
        "threshold": 0.9,
        "designs": 8,
        "distinct": 1,
        "groups": 1,
        "largest_group": 8,
        "max_identity": 1.0,
        "pairwise": "measured",
    }
    for index in range(8):
        _write(tmp_path / "research_traces" / f"binder_{index}.json", _binder(f"binder_{index}", plddt=90 - index))
    _write(tmp_path / "screen.json", _screen(design_diversity=spread))
    ranking = _component(call_tool("backend_status", {"directory": str(tmp_path)}), "candidate_ranking")

    assert len(ranking["order"]) == 8, "the order is not collapsed; the disclosure rides beside it"
    assert set(ranking["order"][0]) == {"example_id", "file", "value"}
    assert ranking["metric"] == "plddt"
    assert ranking["order"][0]["example_id"] == "binder_0", "best first, as before"
    row = ranking["design_diversity"][0]
    assert (row["designs"], row["distinct"], row["pairwise"]) == (8, 1, "measured")
    assert row["phrase"] == diversity_phrase(spread), "the producer's own clause, not a second wording"
    assert "molecule more than once" in ranking["order_caveat"]
    assert "names no design" in ranking["order_caveat"], "it cannot say which rows are the copies"


def test_candidate_ranking_leaves_a_distinct_batch_uncaveated(tmp_path: Path) -> None:
    from rfproteina.disease.mcp_api import call_tool

    for index in range(3):
        _write(tmp_path / "research_traces" / f"binder_{index}.json", _binder(f"binder_{index}", plddt=90 - index))
    spread = {"designs": 3, "distinct": 3, "groups": 3, "largest_group": 1, "pairwise": "measured"}
    _write(tmp_path / "screen.json", _screen(design_diversity=spread))
    ranking = _component(call_tool("backend_status", {"directory": str(tmp_path)}), "candidate_ranking")
    assert ranking["design_diversity"][0]["distinct"] == 3
    assert "order_caveat" not in ranking, "three distinct designs are three shots"


def test_a_spread_without_a_binder_trace_does_not_rank_anything(tmp_path: Path) -> None:
    """A measured spread is not a candidate record, so it must not report one."""
    from rfproteina.disease.mcp_api import call_tool

    _write(tmp_path / "screen.json", _screen(design_diversity={"designs": 8, "distinct": 1, "pairwise": "measured"}))
    result = call_tool("backend_status", {"directory": str(tmp_path)})
    ranking = _component(result, "candidate_ranking")
    assert ranking["status"] == "not_performed"
    assert ranking["note"] == "No binder trace is stored."
    assert ranking["design_diversity"][0]["distinct"] == 1, "and it is still reported"
    assert "order_caveat" not in ranking
    assert result["status"] == "not_recorded", "a spread is not a backend record"


# --- the binding assessment and the reason there is no contact ---------------


def test_binding_names_why_there_is_no_contact(tmp_path: Path) -> None:
    """Six ways to a null contact surface, and this component called the directory empty."""
    from rfproteina.disease.mcp_api import call_tool

    _write(
        tmp_path / "screen.json",
        _screen(
            ppi_pass="absent",
            ppi_blocker="oracle_not_configured",
            chain_blocker="chain_disagreement",
            designed_interface={"contact_ms": None, "clash_fraction": None, "interface_blocker": "coordinates_absent"},
            samples=[
                {"example_id": "binder_0", "contact_ms": None, "interface_blocker": "coordinates_absent"},
                {"example_id": "binder_1", "contact_ms": None, "interface_blocker": "binder_chain_absent"},
            ],
        ),
    )
    result = call_tool("backend_status", {"directory": str(tmp_path)})
    binding = _component(result, "binding_assessment")
    assert binding["status"] == "not_performed", "a named reason is not a measurement"
    assert binding["note"] == "No stored contact or PPI measurement is in this directory."
    item = binding["measurements"][0]
    assert item["ppi_blocker"] == "oracle_not_configured", "a false and a never-asked were one answer"
    assert item["chain_blocker"] == "chain_disagreement"
    assert item["interface_blocker"] == "coordinates_absent"
    assert [row["interface_blocker"] for row in item["samples"]] == ["coordinates_absent", "binder_chain_absent"]
    assert result["status"] == "not_recorded", "a tree of reasons is not a recorded backend"


def test_binding_returns_the_clash_fraction_it_read_past(tmp_path: Path) -> None:
    """It was read twice and dropped twice, so a design measured for clash vanished."""
    from rfproteina.disease.mcp_api import call_tool

    both = tmp_path / "both"
    _write(both / "screen.json", _screen(samples=[{"example_id": "a", "contact_ms": 412.5, "clash_fraction": 0.0031}]))
    item = _component(call_tool("backend_status", {"directory": str(both)}), "binding_assessment")["measurements"][0]
    assert item["samples"][0]["contact_ms"] == 412.5
    assert item["samples"][0]["clash_fraction"] == 0.0031

    clash = tmp_path / "clash_only"
    _write(clash / "screen.json", _screen(samples=[{"example_id": "a", "contact_ms": None, "clash_fraction": 0.4}]))
    component = _component(call_tool("backend_status", {"directory": str(clash)}), "binding_assessment")
    assert component["status"] == "recorded", "a clash fraction is a number this directory holds"
    kept = component["measurements"][0]["samples"][0]
    assert kept["clash_fraction"] == 0.4
    assert "contact_ms" not in kept

    fallback = tmp_path / "fallback"
    _write(
        fallback / "screen.json", _screen(samples=[], designed_interface={"contact_ms": 673.18, "clash_fraction": 0.0})
    )
    only = _component(call_tool("backend_status", {"directory": str(fallback)}), "binding_assessment")
    assert only["measurements"][0]["samples"][0] == {"contact_ms": 673.18, "clash_fraction": 0.0}


# --- structural prediction and the refold evaluation ------------------------


def test_structural_names_which_criteria_were_evaluated_and_which_bounds_were_missed(tmp_path: Path) -> None:
    """A false ppi_pass named none of the bounds behind it. This record does."""
    from rfproteina.disease.mcp_api import call_tool

    criteria = [
        {
            "name": "ppi",
            "state": "fail",
            "thresholds": _thresholds(3, bound="min", cut=80.0, measured=31.0),
            "failed_thresholds": ["oracle_ppi.ptm_coi", "oracle_ppi.min_ipae_coi"],
        },
        {"name": "unconditional", "state": "pass", "thresholds": _thresholds(2)},
        {"name": "internal_coords", "state": "not_evaluated", "thresholds_blocker": "thresholds_not_resolved"},
    ]
    criteria += [
        {"name": f"other_{index}", "state": "not_evaluated", "thresholds": _thresholds(16)} for index in range(8)
    ]
    _write(tmp_path / "screen.json", _screen(refold=_refold_block(criteria, evaluated=2)))
    component = _component(call_tool("backend_status", {"directory": str(tmp_path)}), "structural_prediction")

    assert component["status"] == "recorded"
    row = component["refold"][0]
    assert row["passed"] == ["unconditional"]
    assert row["failed"] == [{"name": "ppi", "failed_thresholds": ["oracle_ppi.ptm_coi", "oracle_ppi.min_ipae_coi"]}]
    assert row["thresholds_not_resolved"] == ["internal_coords"]
    assert (row["declared"], row["evaluated"], row["criteria"]) == (11, 2, 11)
    # The cut: the bound records the answer was computed from do not travel.
    names: list[str] = []
    _keys(component, names)
    assert not {"thresholds", "bound", "cut", "measured", "met"} & set(names)


def test_structural_does_not_carry_a_bound_record_per_folder(tmp_path: Path) -> None:
    """Eleven criteria times sixteen thresholds, times every folder under the directory."""
    from rfproteina.disease.mcp_api import call_tool

    criteria = [{"name": f"criterion_{index}", "state": "fail", "thresholds": _thresholds(16)} for index in range(11)]
    block = _refold_block(criteria, evaluated=11)
    for index in range(8):
        _write(tmp_path / f"gene_{index}ABC_1" / "screen.json", _screen(refold=block))
    component = _component(call_tool("backend_status", {"directory": str(tmp_path)}), "structural_prediction")
    stored = len(json.dumps(block)) * 8
    assert stored > 32_000, "the stored blocks really are this big"
    assert len(json.dumps(component)) < 8_000, f"the component grew with the bound records ({stored} stored)"
    assert len(component["refold"]) == 8


def test_structural_reports_a_refold_blocker_without_claiming_a_structure(tmp_path: Path) -> None:
    from rfproteina.disease.mcp_api import call_tool

    _write(tmp_path / "missing" / "screen.json", _screen(refold={"refold_blocker": "criteria_not_found"}))
    criteria = [{"name": f"criterion_{index}", "state": "not_evaluated"} for index in range(11)]
    _write(
        tmp_path / "unscored" / "screen.json",
        _screen(refold=_refold_block(criteria, evaluated=0, refold_blocker="not_evaluated")),
    )
    result = call_tool("backend_status", {"directory": str(tmp_path)})
    component = _component(result, "structural_prediction")
    assert component["status"] == "not_performed", "no criterion was scored, which is not a prediction"
    assert component["note"] == "Structural prediction is not launched here."
    rows = {row["folder"]: row for row in component["refold"]}
    assert rows["missing"]["refold_blocker"] == "criteria_not_found"
    assert "passed" not in rows["unscored"] and "failed" not in rows["unscored"]
    assert rows["unscored"]["refold_blocker"] == "not_evaluated"
    assert result["status"] == "not_recorded"


def test_structural_names_why_no_ppi_flag_was_recorded(tmp_path: Path) -> None:
    from rfproteina.disease.mcp_api import call_tool

    _write(tmp_path / "screen.json", _screen(ppi_pass="absent", ppi_blocker="not_requested"))
    result = call_tool("backend_status", {"directory": str(tmp_path)})
    component = _component(result, "structural_prediction")
    assert component["ppi_pass"] == [{"folder": "", "ppi_blocker": "not_requested"}]
    assert component["status"] == "not_performed"
    assert result["status"] == "not_recorded"


def test_an_evaluated_count_without_criteria_does_not_claim_a_prediction(tmp_path: Path) -> None:
    """The count is read from what was classified here, not from the stored number."""
    from rfproteina.disease.mcp_api import call_tool

    _write(tmp_path / "screen.json", _screen(refold=_refold_block([], evaluated=7)))
    result = call_tool("backend_status", {"directory": str(tmp_path)})
    assert _component(result, "structural_prediction")["status"] == "not_performed"
    assert result["status"] == "not_recorded"


def test_the_whole_backend_answer_validates_with_every_disclosure_present(tmp_path: Path) -> None:
    """One directory carrying all three: a spread, a refold evaluation and a reason."""
    import jsonschema

    from rfproteina.disease.mcp_api import call_tool

    _write(tmp_path / "research_traces" / "binder_0.json", _binder("binder_0", plddt=88.0))
    _write(
        tmp_path / "screen.json",
        _screen(
            ppi_pass="absent",
            ppi_blocker="oracle_not_configured",
            design_diversity={"designs": 4, "distinct": 1, "pairwise": "measured"},
            designed_interface={"contact_ms": None, "interface_blocker": "designs_compressed"},
            refold=_refold_block(
                [{"name": "ppi", "state": "fail", "failed_thresholds": ["oracle_ppi.ptm_coi"]}], evaluated=1
            ),
            samples=[],
        ),
    )
    answer = call_tool("backend_status", {"directory": str(tmp_path)})
    schema = {str(tool["name"]): tool.get("outputSchema") for tool in _server()._TOOLS}["backend_status"]
    jsonschema.validate(answer, schema)
    json.dumps(answer, allow_nan=False)
    assert _component(answer, "candidate_ranking")["order_caveat"]
    assert _component(answer, "structural_prediction")["refold"][0]["failed"]
    assert _component(answer, "binding_assessment")["measurements"][0]["interface_blocker"] == "designs_compressed"
