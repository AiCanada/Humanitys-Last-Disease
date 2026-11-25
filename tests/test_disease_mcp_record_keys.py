"""The MCP view is pinned to the records it reads, not only to its own schema.

``screen.py`` governs its own shape: ``SCREEN_VERSION`` is raised whenever a key
is added, and ``tests/test_disease_screen_shape.py`` pins both key sets beside
that number. The MCP tests pinned ``_AXIS_FIELDS`` against ``_AXIS_SCHEMA`` and
``_VERDICT_FIELDS`` against the same schema -- MCP against MCP. Nothing held
either against the program, so the record grew through version 2 and version 3
while this view kept returning the version-1 field set, and every MCP test
passed the whole way. ``refold`` is the plainest case: the key version 3 exists
for was filtered out beside a ``screen_version: 3`` that announced it.

So the pin here runs the other way. Every key the screen writes is either
returned, shaped by a function of its own, or named in an exclusion list with a
reason. A new key in ``screen.py`` fails this file until someone decides which
of the three it is. The same pin is applied to the generation record the archive
writes, and to the one rule this package spells twice.
"""

import json
from pathlib import Path

import jsonschema

from tests.helpers.disease import load_mcp_server as _server

#: Screen keys the verdict deliberately does not carry, and why. Each is a
#: decision that survives review; an omission is not.
_NOT_IN_THE_VERDICT = {
    # Per-design measurements. ``read_sequence_measures`` returns these, and the
    # verdict is a decision rather than a copy of the table behind it.
    "samples": "read_sequence_measures returns the per-design measurements",
    "sequence_developability": "read_sequence_measures returns the first design's sequence block",
    "designed_interface": "read_sequence_measures and backend_status return the interface block",
}
#: Three exclusions were retired rather than kept, and the reasons are recorded
#: here so the decision is not made twice.
#:
#: ``inputs`` was excluded because ``replaced.changed_inputs`` already named the
#: supplied files that moved. It does, but only on a re-screen whose verdict
#: changed; a first screen writes no ``replaced``, so no client could read what
#: a fresh verdict rested on. It is now returned.
#:
#: ``chemical_liability`` and ``modality_screen`` were excluded as "no state,
#: enters no call". Six blocks above them in ``_VERDICT_FIELDS`` are no-call
#: reports returned by decision, so that reason excluded two blocks on exactly
#: the ground the same list had already set aside. Both are now returned, and
#: ``mcp_layers._SCREEN_OMITTED`` keeps them out of the tree walk for size the
#: way it keeps the other reports out.


def _screen_keys() -> set[str]:
    from rfproteina.disease.screen import SCREEN_OPTIONAL_KEYS, SCREEN_REQUIRED_KEYS

    return set(SCREEN_REQUIRED_KEYS) | set(SCREEN_OPTIONAL_KEYS)


def _handled() -> set[str]:
    """Keys the view accounts for, each by a different route."""
    from rfproteina.disease import mcp_api

    # ``replaced`` is reshaped by _replaced rather than copied, and the axis
    # groups are reshaped by _axis_summary. Neither can be a plain field.
    return set(mcp_api._VERDICT_FIELDS) | set(mcp_api._AXIS_GROUPS) | {"replaced"}


def test_every_screen_key_is_returned_or_excluded_by_name() -> None:
    """The pin. A key added to screen.py lands here before it lands nowhere."""
    missing = _screen_keys() - _handled() - set(_NOT_IN_THE_VERDICT)
    assert not missing, (
        f"screen.json now carries {sorted(missing)} and the verdict view does not. "
        "Add each to mcp_api._VERDICT_FIELDS and the matching property in the server's "
        "_VERDICT_SCHEMA, or name it in _NOT_IN_THE_VERDICT above with the reason it stays out."
    )


def test_the_verdict_returns_nothing_the_screen_does_not_write() -> None:
    """The other direction: a field kept here after the writer dropped it is dead."""
    from rfproteina.disease import mcp_api

    dead = set(mcp_api._VERDICT_FIELDS) - _screen_keys()
    assert not dead, f"_VERDICT_FIELDS keeps {sorted(dead)}, which no screen writes any more"


def test_no_exclusion_outlives_the_key_it_excludes() -> None:
    """A reason for a key that no longer exists is a note about nothing."""
    stale = set(_NOT_IN_THE_VERDICT) - _screen_keys()
    assert not stale, f"_NOT_IN_THE_VERDICT names {sorted(stale)}, which screen.py no longer writes"


def test_the_published_schema_declares_exactly_what_the_verdict_returns() -> None:
    """Equality, not containment: an undeclared field a validating client meets is dropped."""
    from rfproteina.disease import mcp_api

    declared = set(_server()._VERDICT_SCHEMA["properties"])
    derived = {"replaced", "curated_axes", "malformed", "record"}
    expected = set(mcp_api._VERDICT_FIELDS) | set(mcp_api._AXIS_GROUPS) | derived
    assert declared == expected, "the verdict view and its published shape have drifted apart"


def _screened(folder: Path, record: dict) -> Path:
    """A finished run carrying one stored screen, without running a screen."""
    run = folder / "gene_1ABC_1"
    (run / "inference_samples" / "binder_0").mkdir(parents=True)
    (run / "screen.json").write_text(json.dumps(record), encoding="utf-8")
    return run


def test_affinity_rows_keep_their_identity_through_redaction(tmp_path: Path) -> None:
    from rfproteina.disease import mcp_api

    supplied_id = "ACDEFGHIKLMNPQRSTVWY"
    report = {
        "method": "prodigy_ic_nis",
        "attempted": 2,
        "predicted": 1,
        "untested": 1,
        "samples": {
            supplied_id: {"status": "predicted", "residues_by_chain": {supplied_id: 12}, "derived_kd_mol_l": 2e-7},
            "binder_1": {"status": "untested", "blocker": "structure_unreadable"},
        },
    }
    run = _screened(tmp_path, {"screen_version": 6, "call": "untested", "prodigy_affinity": report})
    verdict = mcp_api.read_verdict({"directory": str(run), "include_record": True})["runs"][0]["verdict"]
    for item in (verdict, verdict["record"]):
        result = item["prodigy_affinity"]
        assert result["attempted"] == 2 and result["predicted"] == 1
        by_id = {row["example_id"]: row for row in result["samples"]}
        assert by_id["[sequence omitted]"]["derived_kd_mol_l"] == 2e-7
        assert by_id["[sequence omitted]"]["residues_by_chain"] == [{"chain": "[sequence omitted]", "residues": 12}]
        assert by_id["binder_1"]["blocker"] == "structure_unreadable"
        assert supplied_id not in json.dumps(result)
    assert supplied_id in (run / "screen.json").read_text(encoding="utf-8")


def test_ames_views_keep_attributed_probability_without_molecular_graphs(tmp_path: Path) -> None:
    from rfproteina.disease import ligand_ames, mcp_api, mcp_layers
    from rfproteina.disease.corpus_provenance import declare

    raw_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    compound = "smiles_sha256_" + "a1" * 32
    method = {"method": ligand_ames.METHOD, "endpoint": ligand_ames.ENDPOINT, "unit": ligand_ames.UNIT}
    entry = {"compound_id": compound, **method, "status": "predicted", "predicted_ames_probability": 0.0}
    provenance = declare(ligand_ames.METHOD, against=ligand_ames.ENDPOINT)
    report = {
        **method,
        "status": "predicted",
        "scope": "candidate_table",
        "attempted": 1,
        "unique_compounds": 1,
        "unique_predicted": 1,
        "corpus_provenance": provenance,
        "inputs": [{"candidate_id": "lead_1", "compound_id": compound, "smiles": raw_smiles}],
        "compounds": [{**entry, "unrecognized": {"input_graph": raw_smiles}}],
        "structures": {compound: raw_smiles},
        "provider": {"device": "cpu", "ensemble_members": 5, "unrecognized": [raw_smiles]},
    }
    run = _screened(tmp_path, {"call": "untested", "ligand_ames": report})
    result = mcp_api.read_verdict({"directory": str(run), "include_record": True})
    verdict = result["runs"][0]["verdict"]
    expected = {
        **{key: report[key] for key in (*method, "scope", "status", "attempted", "unique_compounds", "unique_predicted")},
        "inputs": [{"candidate_id": "lead_1", "compound_id": compound}],
        "compounds": [entry],
        "provider": {"device": "cpu", "ensemble_members": 5},
        "corpus_provenance": provenance,
    }
    assert verdict["ligand_ames"] == verdict["record"]["ligand_ames"] == expected
    assert raw_smiles not in json.dumps(result)
    assert verdict["call"] == "untested"
    assert "ligand_ames" not in mcp_layers.screen_fields()
    tools = {tool["name"]: tool for tool in _server()._TOOLS}
    jsonschema.validate(result, tools["read_verdict"]["outputSchema"])
    # The selected-method path uses the same single-field verdict projection.
    selected = {
        "program": "Humanitys-Last-Disease",
        "directory": str(run),
        "candidate_id": None,
        "results": [{"id": "ligand_ames", "status": "predicted", "measurement": verdict["ligand_ames"]}],
    }
    jsonschema.validate(selected, tools["run_selected_screens"]["outputSchema"])


def test_ames_invalid_stored_scores_do_not_show_as_experimental_passes() -> None:
    from rfproteina.disease import ligand_ames, mcp_api
    from tests.helpers.disease import screen_page

    method = {"method": ligand_ames.METHOD, "endpoint": ligand_ames.ENDPOINT, "unit": ligand_ames.UNIT}
    compound = {**method, "compound_id": "smiles_sha256_" + "b2" * 32, "status": "predicted"}
    report = {
        **method, "status": "predicted", "scope": "candidate_table", "unique_compounds": 1, "unique_predicted": 1,
        "inputs": [{"candidate_id": "lead_1", "compound_id": compound["compound_id"]}],
        "compounds": [{**compound, "predicted_ames_probability": 2.0}],
    }
    payload = {"call": "untested", "ligand_ames": report}
    view = mcp_api._verdict(payload, include_record=False)["ligand_ames"]
    assert view["compounds"] == [{**compound, "status": "untested", "blocker": "invalid_provider_result"}]
    assert view["status"] == "untested" and view["unique_predicted"] == 0
    assert "ADMET-AI Ames prediction untested." in screen_page(payload)
    report["compounds"][0]["predicted_ames_probability"] = 0.123456
    rendered = screen_page(payload)
    assert "Predicted Ames-positive probability 0.123456." in rendered
    assert "does not establish an experimental result or a pass" in rendered
    report.update(status="untested", blocker="ames_input_limit_exceeded")
    assert "Ames screen input exceeds" in screen_page(payload)


def test_ames_view_reconciles_stored_counts_and_rejects_unattributed_or_blocked_scores() -> None:
    from rfproteina.disease import ligand_ames, mcp_api

    method = {"method": ligand_ames.METHOD, "endpoint": ligand_ames.ENDPOINT, "unit": ligand_ames.UNIT}
    actual_id, foreign_id = ("smiles_sha256_" + character * 64 for character in "de")
    actual = {**method, "status": "predicted", "compound_id": actual_id, "predicted_ames_probability": 0.0}
    foreign = {**actual, "compound_id": foreign_id, "predicted_ames_probability": 1.0}
    base = {
        **method, "scope": "candidate_table", "status": "predicted",
        "attempted": 95, "unique_compounds": 40, "unique_predicted": 30,
        "inputs": [{"candidate_id": name, "compound_id": actual_id} for name in ("lead_a", "lead_b")],
        "compounds": [actual, foreign],
    }
    view = mcp_api._ames_view(base)
    assert view["status"] == "predicted" and view["compounds"] == [actual]
    assert (view["attempted"], view["unique_compounds"], view["unique_predicted"]) == (2, 1, 1)
    for fields, expected in (
        ({"compounds": [actual, actual]}, "duplicate_provider_row"),
        ({"compounds": [foreign]}, "missing_provider_row"),
        ({"blocker": "provider_timeout"}, "provider_timeout"),
    ):
        untested = mcp_api._ames_view({**base, **fields})
        assert untested["status"] == "untested" and untested["unique_predicted"] == 0
        assert untested["compounds"][0]["blocker"] == expected
        assert "predicted_ames_probability" not in untested["compounds"][0]


def test_clearance_views_keep_native_units_and_zero_without_molecular_graphs(tmp_path: Path) -> None:
    from rfproteina.disease import ligand_microsomal_clearance as clearance
    from rfproteina.disease import mcp_api, mcp_layers
    from rfproteina.disease.corpus_provenance import declare
    from tests.helpers.disease import screen_page

    compound = "smiles_sha256_" + "a1" * 32
    raw_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    identity = {"method": clearance.METHOD, "endpoint": clearance.ENDPOINT, "unit": clearance.UNIT}
    entry = {"compound_id": compound, **identity, "status": "predicted", "predicted_clint_ul_min_mg": 0.0}
    provenance = declare(clearance.METHOD, against=clearance.ENDPOINT)
    report = {
        **identity,
        "status": "predicted",
        "scope": "candidate_table",
        "inputs": [{"candidate_id": "lead_1", "compound_id": compound, "smiles": raw_smiles}],
        "compounds": [{**entry, "unrecognized": {"input_graph": raw_smiles}}],
        "structures": {compound: raw_smiles},
        "provider": {"device": "cpu", "ensemble_members": 5, "unrecognized": [raw_smiles]},
        "corpus_provenance": provenance,
    }
    payload = {"call": "untested", clearance.METHOD_ID: report}
    run = _screened(tmp_path, payload)
    answer = mcp_api.read_verdict({"directory": str(run), "include_record": True})
    verdict = answer["runs"][0]["verdict"]
    view = verdict[clearance.METHOD_ID]
    assert view == verdict["record"][clearance.METHOD_ID]
    assert view["compounds"] == [entry]
    assert view["inputs"] == [{"candidate_id": "lead_1", "compound_id": compound}]
    assert view["provider"] == {"device": "cpu", "ensemble_members": 5}
    assert view["corpus_provenance"] == provenance
    assert verdict["call"] == "untested" and raw_smiles not in json.dumps(answer)
    assert clearance.METHOD_ID not in mcp_layers.screen_fields()
    displayed = screen_page(payload)
    assert "Predicted microsomal intrinsic clearance 0 uL/min/mg microsomal protein." in displayed
    assert "does not establish systemic clearance or half-life" in displayed
    registered = {tool["name"]: tool for tool in _server()._TOOLS}
    jsonschema.validate(answer, registered["read_verdict"]["outputSchema"])
    selected = {
        "program": "Humanitys-Last-Disease",
        "directory": str(run),
        "candidate_id": None,
        "results": [{"id": clearance.METHOD_ID, "status": "predicted", "measurement": view}],
    }
    output_schema = registered["run_selected_screens"]["outputSchema"]
    validator = jsonschema.Draft202012Validator(output_schema)
    assert validator.is_valid(selected)
    selected["results"][0]["measurement"] = {**view, "unit": "mL/min/kg"}
    assert not validator.is_valid(selected)


def test_clearance_view_does_not_convert_wrong_units_or_repair_bad_predictions() -> None:
    from rfproteina.disease import ligand_microsomal_clearance as clearance
    from rfproteina.disease import mcp_api
    from tests.helpers.disease import screen_page

    compound = "smiles_sha256_" + "b2" * 32
    identity = {"method": clearance.METHOD, "endpoint": clearance.ENDPOINT, "unit": clearance.UNIT}
    key = "predicted_clint_ul_min_mg"
    entry = {**identity, "compound_id": compound, "status": "predicted", key: 42.5}
    report = {
        **identity, "status": "predicted", "scope": "candidate_table",
        "inputs": [{"candidate_id": "lead_1", "compound_id": compound}],
        "compounds": [entry],
    }
    view = mcp_api._microsomal_clearance_view(report)
    assert view["status"] == "predicted" and view["compounds"][0][key] == 42.5
    for alteration in (
        {"endpoint": "Clearance_Hepatocyte_AZ"},
        {"unit": "log(uL/min/mg)"},
        {key: -0.01},
        {key: float("inf")},
        {key: float("nan")},
        {key: True},
        {key: 10**400},
    ):
        bad_report = {**report, "compounds": [{**entry, **alteration}]}
        view = mcp_api._microsomal_clearance_view(bad_report)
        assert view["status"] == "untested" and view["unique_predicted"] == 0
        assert view["compounds"][0]["blocker"] == "invalid_provider_result" and key not in view["compounds"][0]
        rendered = screen_page({"call": "untested", clearance.METHOD_ID: bad_report})
        assert "microsomal clearance prediction untested" in rendered
        assert "Predicted microsomal intrinsic clearance" not in rendered


def _dili_report(probability: float = 0.0) -> dict:
    from rfproteina.disease import ligand_dili as dili
    from rfproteina.disease.corpus_provenance import declare

    compound = "smiles_sha256_" + "6e" * 32
    return {
        **dili.SPEC.identity,
        "scope": "candidate_table",
        "status": "predicted",
        "inputs": [{"candidate_id": "lead_1", "compound_id": compound}],
        "compounds": [{
            **dili.SPEC.identity, "compound_id": compound, "status": "predicted", dili.VALUE_FIELD: probability,
        }],
        "corpus_provenance": declare(dili.METHOD, against=dili.ENDPOINT),
    }


def test_dili_reports_advisory_model_probability_without_candidate_graphs(tmp_path: Path) -> None:
    from rfproteina.disease import ligand_dili as dili
    from rfproteina.disease import mcp_api, mcp_layers
    from tests.helpers.disease import screen_page

    raw = _dili_report()
    clean_inputs = [dict(raw["inputs"][0])]
    clean_compounds = [dict(raw["compounds"][0])]
    raw_graph = "CC(=O)Nc1ccc(O)cc1"
    raw["inputs"][0]["smiles"] = raw_graph
    raw["compounds"][0].update(smiles=raw_graph, threshold=0.5, clinical_probability=0.0, call="pass")
    raw.update(
        attempted=900, unique_predicted=900, threshold=0.5, assay_outcome="negative",
        provider={"device": "cpu", "ensemble_members": 5, "raw_graph": raw_graph},
    )
    payload = {"call": "untested", dili.METHOD_ID: raw}
    run = _screened(tmp_path, payload)
    answer = mcp_api.read_verdict({"directory": str(run), "include_record": True})
    verdict = answer["runs"][0]["verdict"]
    view = verdict[dili.METHOD_ID]
    assert verdict["call"] == "untested" and view == verdict["record"][dili.METHOD_ID]
    assert view["inputs"] == clean_inputs and view["compounds"] == clean_compounds
    assert view["provider"] == {"device": "cpu", "ensemble_members": 5}
    assert view["status"] == "predicted" and view["attempted"] == view["unique_compounds"] == view["unique_predicted"] == 1
    assert view["corpus_provenance"] == raw["corpus_provenance"]
    assert {"threshold", "assay_outcome"}.isdisjoint(view) and raw_graph not in json.dumps(answer)
    assert dili.METHOD_ID not in mcp_layers.screen_fields()
    rendered = screen_page(payload)
    assert "Predicted DILI-positive model probability 0." in rendered
    assert "does not establish patient risk, severity, or safety. No pass cutoff applies." in rendered
    assert raw_graph not in rendered

    schemas = {tool["name"]: tool["outputSchema"] for tool in _server()._TOOLS if "outputSchema" in tool}
    jsonschema.validate(answer, schemas["read_verdict"])
    validator = jsonschema.Draft202012Validator(schemas["run_selected_screens"])
    result = {"id": dili.METHOD_ID, "status": "predicted", "measurement": view}
    selected = {"program": "Humanitys-Last-Disease", "directory": str(run), "candidate_id": None, "results": [result]}
    assert validator.is_valid(selected)
    for key, value in (("endpoint", "AMES"), ("unit", "percent"), (dili.VALUE_FIELD, 1.01)):
        result["measurement"] = {**view, "compounds": [{**view["compounds"][0], key: value}]}
        assert not validator.is_valid(selected)


def test_dili_rejects_bad_attribution_invalid_scores_and_unrecognized_blocker_text() -> None:
    from rfproteina.disease import ligand_dili as dili
    from rfproteina.disease.mcp_api import _dili_view
    from tests.helpers.disease import screen_page

    score = dili.VALUE_FIELD
    for alteration in (
        {"method": "admet_ai_ames_v2"}, {"endpoint": "DILIrank"}, {"unit": "%"},
        {score: -0.01}, {score: 1.01}, {score: "0.25"}, {score: True}, {score: float("nan")},
        {score: float("inf")}, {score: 10**400}, {"blocker": "should_not_echo_user_text"},
    ):
        raw = _dili_report(1.0)
        raw["compounds"][0].update(alteration)
        view = _dili_view(raw)
        assert view["status"] == "untested" and view["unique_predicted"] == 0
        assert view["compounds"][0]["blocker"] == "invalid_provider_result" and score not in view["compounds"][0]
        rendered = screen_page({"call": "untested", dili.METHOD_ID: raw})
        assert "drug-induced liver injury prediction untested" in rendered
        assert "Predicted DILI-positive model probability" not in rendered

    valid = _dili_report(1.0)
    assert _dili_view(valid)["compounds"][0][score] == 1.0
    valid.update(status="untested", blocker="dili_inputs_over_limit")
    assert _dili_view(valid)["blocker"] == "dili_inputs_over_limit"
    assert "The DILI input exceeds the file or nomination limit." in screen_page({"call": "untested", dili.METHOD_ID: valid})


def test_dili_preserves_fixed_panel_fallback_reasons() -> None:
    from rfproteina.disease import ligand_dili as dili
    from rfproteina.disease.mcp_api import _verdict

    for reason in ("provider_or_input_unavailable", "input_unavailable"):
        report = {"blocker": reason}
        assert _verdict({dili.METHOD_ID: report}, include_record=False)[dili.METHOD_ID] == {
            "status": "untested", "blocker": reason,
        }


def _delphi_report(probability: float = 0.0) -> dict:
    from rfproteina.disease import delphi_psr as psr
    from rfproteina.disease.corpus_provenance import declare

    return {
        "method": psr.METHOD, "endpoint": psr.ENDPOINT, "unit": psr.UNIT, "scope": "candidate",
        "input_sha256": "a" * 64,
        "status": "predicted", "attempted": 1, "predicted": 1,
        "pairs": [{"candidate_id": "binder_0", "status": "predicted", psr.VALUE_FIELD: probability}],
        "provider": {
            "release": psr.RELEASE, "device": "cpu", "checkpoint_sha256": psr.CHECKPOINT_SHA256,
            "torch": "2.8.0+cu128", "anarcii": "2.0.3",
        },
        "corpus_provenance": declare(psr.METHOD, against=psr.ENDPOINT),
    }


def test_delphi_keeps_advisory_probability_without_sequence_fields(tmp_path: Path) -> None:
    from rfproteina.disease import delphi_psr as psr
    from rfproteina.disease import mcp_api, mcp_layers
    from tests.helpers.disease import screen_page

    raw = _delphi_report()
    private = "ACDFG"
    clean = _delphi_report()
    raw["pairs"][0].update(vh=private, vl=private, cdr3=private, pass_call=True, threshold=0.5)
    raw["provider"]["unexpected"] = private
    raw.update(attempted=999, predicted=900, vh=private, reported_psr_signal=17.5)
    payload = {"call": "untested", psr.METHOD_ID: raw}
    run = _screened(tmp_path, payload)
    answer = mcp_api.read_verdict({"directory": str(run), "include_record": True})
    verdict = answer["runs"][0]["verdict"]
    assert verdict["call"] == "untested" and verdict[psr.METHOD_ID] == verdict["record"][psr.METHOD_ID] == clean
    assert private not in json.dumps(answer)
    assert psr.METHOD_ID not in mcp_layers.screen_fields()
    rendered = screen_page(payload)
    assert "Predicted PSR-filter pass probability 0." in rendered
    assert "lower predicted polyreactivity" in rendered and "No experimental result or pass cutoff" in rendered
    assert private not in rendered

    schemas = {tool["name"]: tool["outputSchema"] for tool in _server()._TOOLS if "outputSchema" in tool}
    jsonschema.validate(answer, schemas["read_verdict"])
    validator = jsonschema.Draft202012Validator(schemas["run_selected_screens"])
    result = {"id": psr.METHOD_ID, "status": "predicted", "measurement": clean}
    selected = {"program": "Humanitys-Last-Disease", "directory": str(run), "candidate_id": None, "results": [result]}
    assert validator.is_valid(selected)
    for key, value in (("method", "another_method"), ("endpoint", "sec_filter"), ("unit", "%")):
        result["measurement"] = {**clean, key: value}
        assert not validator.is_valid(selected)
    result["measurement"] = {**clean, "input_sha256": "b" * 64 + "\n"}
    assert not validator.is_valid(selected)
    for bad_row in ({psr.VALUE_FIELD: 1.01}, {"blocker": "delphi_inference_failed"}, {"vh": private}):
        result["measurement"] = {**clean, "pairs": [{**clean["pairs"][0], **bad_row}]}
        assert not validator.is_valid(selected)
    result["measurement"] = {"status": "untested", "blocker": "input_unavailable"}
    assert validator.is_valid(selected)


def test_delphi_refuses_invalid_provenance_scores_and_ambiguous_nominations() -> None:
    from rfproteina.disease import delphi_psr as psr
    from rfproteina.disease.delphi_inputs import MAX_ROWS
    from rfproteina.disease.mcp_api import _delphi_view

    score = psr.VALUE_FIELD
    assert _delphi_view(_delphi_report(1.0))["pairs"][0][score] == 1.0
    for value in (-0.1, 1.1, True, "0.5", float("nan"), float("inf"), 10**400):
        result = _delphi_view(_delphi_report(value))
        assert result["status"] == "untested" and result["predicted"] == 0
        assert result["pairs"][0]["blocker"] == "delphi_prediction_record_invalid"
        assert score not in result["pairs"][0]

    for field, bad_value in (("release", "another-release"), ("device", "cuda"), ("checkpoint_sha256", "0" * 64)):
        report = _delphi_report(0.6)
        report["provider"][field] = bad_value
        result = _delphi_view(report)
        assert result["status"] == "untested" and "provider" not in result and score not in result["pairs"][0]

    report = _delphi_report(0.6)
    for digest in (None, "a" * 63, "a" * 64 + "\n", "A" * 64, "z" * 64, 48):
        assert "input_sha256" not in _delphi_view({**report, "input_sha256": digest})
    assert _delphi_view({**report, "scope": "standalone"}) == {
        "status": "untested", "blocker": "delphi_prediction_record_invalid",
    }
    first = report["pairs"][0]
    invalid = {"candidate_id": "private/path/", "status": "predicted", score: 0.8}
    duplicate = {"candidate_id": "binder_1", "status": "predicted", score: 0.1}
    report.update(scope="run", pairs=[first, invalid, duplicate, duplicate])
    result = _delphi_view(report)
    assert (result["status"], result["attempted"], result["predicted"]) == ("partial", 4, 1)
    assert result["pairs"][1] == {"candidate_id": "", "status": "untested", "blocker": "delphi_candidate_id_invalid"}
    assert [row["blocker"] for row in result["pairs"][2:]] == ["delphi_candidate_id_duplicate"] * 2
    assert _delphi_view({**report, "pairs": [first] * (MAX_ROWS + 1)})["blocker"] == "delphi_pairs_over_limit"
    assert _delphi_view({**report, "scope": "candidate"})["predicted"] == 0


def test_delphi_displays_recognized_blockers_including_panel_fallbacks() -> None:
    from rfproteina.disease import delphi_psr as psr
    from rfproteina.disease.mcp_api import _DELPHI_BLOCKERS, _delphi_view
    from rfproteina.disease.screen_board import _ACTIONS, _BLOCKERS
    from tests.helpers.disease import screen_page

    assert _DELPHI_BLOCKERS.issubset(_BLOCKERS) and _DELPHI_BLOCKERS.issubset(_ACTIONS)
    for reason in ("provider_or_input_unavailable", "input_unavailable"):
        assert _delphi_view({"blocker": reason}) == {"status": "untested", "blocker": reason}
    report = _delphi_report()
    report.update(status="untested", blocker="delphi_no_predictions")
    report["pairs"][0].update(status="untested", blocker="delphi_chain_roles_invalid")
    assert psr.VALUE_FIELD not in _delphi_view(report)["pairs"][0]
    rendered = screen_page({"call": "untested", psr.METHOD_ID: report})
    assert "did not identify the submitted VH and VL roles" in rendered
    assert "Predicted PSR-filter pass probability" not in rendered


def _peptide_rbc_report() -> dict:
    from rfproteina.disease import peptide_rbc as rbc
    from rfproteina.disease.corpus_provenance import declare

    human = {
        "candidate_id": "binder_0", "observation_id": "lab_readout_01", "formulation_id": "sample_form_01",
        "source_id": "report_01", "species": "human", "matrix": "washed_rbc", "context_id": "assay_context_01",
        "concentration": 1.0, "concentration_unit": "uM", "exposure_minutes": 10.0,
        "hemolysis_percent": 0.0, "controls_qc": "accepted", "status": "reported",
    }
    other = {
        **human, "observation_id": "lab_readout_02", "species": "rabbit", "matrix": "whole_blood",
        "concentration": 2.5, "concentration_unit": "ug/mL", "exposure_minutes": 17.0, "hemolysis_percent": 100.0,
    }
    refused = {
        "candidate_id": "binder_0", "observation_id": "lab_readout_03", "formulation_id": "sample_form_02",
        "status": "untested", "blocker": "peptide_rbc_controls_failed",
    }
    return {
        "method": rbc.METHOD, "endpoint": rbc.ENDPOINT, "unit": rbc.UNIT, "scope": "candidate",
        "input_sha256": "a" * 64, "status": "partial", "attempted": 3, "reported": 2, "human_reported": 1,
        "observations": [human, other, refused], "corpus_provenance": declare(rbc.METHOD),
    }


def test_peptide_rbc_keeps_species_and_measured_assay_context_separate(tmp_path: Path) -> None:
    from rfproteina.disease import mcp_api, mcp_layers
    from rfproteina.disease import peptide_rbc as rbc
    from tests.helpers.disease import screen_page

    private = "ACDEFGHIKLMNPQRSTVWYACDEFGHI"
    raw, clean = _peptide_rbc_report(), _peptide_rbc_report()
    raw["observations"][0].update(sequence=private, threshold=5, pass_call=True, source_path="private/path")
    raw["observations"][2].update(hemolysis_percent=20, concentration=0, controls_qc="failed")
    raw.update(attempted=999, reported=800, human_reported=200, prediction=0, safe=True)
    payload = {"call": "untested", rbc.METHOD_ID: raw}
    finished = _screened(tmp_path, payload)
    answer = mcp_api.read_verdict({"directory": str(finished), "include_record": True})
    verdict = answer["runs"][0]["verdict"]
    assert verdict["call"] == "untested" and verdict[rbc.METHOD_ID] == verdict["record"][rbc.METHOD_ID] == clean
    assert private not in json.dumps(answer)
    assert rbc.METHOD_ID not in mcp_layers.screen_fields()
    board = screen_page(payload)
    assert "Human observations 1. Non-human observations 1." in board
    assert "(human, washed red cells) reported 0%" in board
    assert "(rabbit, whole blood) reported 100%" in board
    assert "The assay report says the required controls did not pass quality review." in board
    assert "The report does not establish safety." in board and private not in board
    raw["observations"][1].update(hemolysis_percent=99.9999999, concentration=2.00000000002)
    assert "reported 99.9999999% hemolysis at 2.00000000002" in screen_page(payload)

    schemas = {tool["name"]: tool["outputSchema"] for tool in _server()._TOOLS if "outputSchema" in tool}
    jsonschema.validate(answer, schemas["read_verdict"])
    validator = jsonschema.Draft202012Validator(schemas["run_selected_screens"])
    result = {"id": rbc.METHOD_ID, "status": "partial", "measurement": clean}
    selected = {"program": "Humanitys-Last-Disease", "directory": str(finished), "candidate_id": None,
                "results": [result]}
    assert validator.is_valid(selected)
    for changes in ({"method": "model_prediction"}, {"input_sha256": "a" * 64 + "\n"}, {"context_id": private}):
        result["measurement"] = {**clean, **changes}
        assert not validator.is_valid(selected)
    for changes in ({"species": "unknown"}, {"controls_qc": "failed"}, {"source_id": "unsafe/path"},
                    {"source_id": "report_01\n"}, {"hemolysis_percent": 100.001}, {"exposure_minutes": 0},
                    {"concentration_unit": "mM"}, {"matrix": "serum"}):
        result["measurement"] = {**clean, "observations": [{**clean["observations"][0], **changes}]}
        assert not validator.is_valid(selected)
    without_digest = dict(clean)
    without_digest.pop("input_sha256")
    result["measurement"] = without_digest
    assert not validator.is_valid(selected)
    result["measurement"] = {**clean, "observations": [{**clean["observations"][2], "hemolysis_percent": 20}]}
    assert not validator.is_valid(selected)


def test_peptide_rbc_projection_rejects_bad_attribution_and_forged_values() -> None:
    from rfproteina.disease.mcp_api import _peptide_rbc_view
    from rfproteina.disease.peptide_rbc import MAX_ROWS

    report = _peptide_rbc_report()
    row = report["observations"][0]
    for properties in (
        {"hemolysis_percent": False}, {"hemolysis_percent": -1}, {"hemolysis_percent": 101},
        {"hemolysis_percent": "<1"}, {"hemolysis_percent": float("nan")}, {"exposure_minutes": float("inf")},
        {"concentration": 10**400}, {"concentration": 0}, {"matrix": []}, {"species": "humanized"},
        {"controls_qc": "failed"}, {"concentration_unit": "nM"}, {"context_id": "unsafe/path"},
        {"source_id": "ACDEFGHIKLMNPQRSTVWYACDEFGHI"}, {"blocker": "private/provider/path"},
    ):
        bad = _peptide_rbc_view({**report, "observations": [{**row, **properties}]})
        assert bad["status"] == "untested" and bad["reported"] == bad["human_reported"] == 0
        assert "hemolysis_percent" not in bad["observations"][0]

    refused = _peptide_rbc_view({**report, "input_sha256": "A" * 64})
    assert "input_sha256" not in refused and refused["status"] == "untested"
    assert _peptide_rbc_view({**report, "method": "hemopi2"}) == {
        "status": "untested", "blocker": "peptide_rbc_record_invalid",
    }
    assert _peptide_rbc_view({**report, "observations": [row] * (MAX_ROWS + 1)})["blocker"] == "peptide_rbc_over_limit"
    assert _peptide_rbc_view({"blocker": "input_unavailable"}) == {
        "status": "untested", "blocker": "input_unavailable",
    }
    duplicate = _peptide_rbc_view({**report, "observations": [row, row, report["observations"][1]]})
    assert duplicate["status"] == "partial" and duplicate["human_reported"] == 0
    assert [item["blocker"] for item in duplicate["observations"][:2]] == ["peptide_rbc_duplicate_observation"] * 2

    other = {**row, "candidate_id": "binder_1"}
    conflicting = _peptide_rbc_view({**report, "observations": [row, other]})
    assert conflicting["status"] == "untested" and conflicting["reported"] == 0
    distinct = _peptide_rbc_view({**report, "scope": "run", "observations": [row, other]})
    assert distinct["status"] == "reported" and distinct["reported"] == distinct["human_reported"] == 2
    assert [item["candidate_id"] for item in distinct["observations"]] == ["binder_0", "binder_1"]


def _httk_report() -> dict:
    from rfproteina.disease import httk
    from rfproteina.disease.corpus_provenance import declare

    return {
        "method": httk.METHOD, "endpoint": httk.ENDPOINT, "unit": httk.UNIT, "scenario": httk.SCENARIO,
        "reference_intake_mg_kg_day": 1, "scope": "run", "status": "partial", "attempted": 2, "reported": 1,
        "manifest_sha256": "b" * 64, "provider": dict(httk.PROVIDER), "corpus_provenance": declare(httk.METHOD),
        "members": [
            {"candidate_id": "ligand_0", "cas": "103-90-2", "dtxsid": "DTXSID2020006",
             "compound_id": "smiles_sha256_2908012d4d1af71b20b6ce8aa15ec902a6ec288a21be1855d631a6045052fe25",
             "reference_inchikey": "RZVAJINKPMORJF-UHFFFAOYSA-N", "human_in_vitro": {
                 "clint_ul_min_per_million_hepatocytes": 0.3598, "clint_reference": "Paini 2020",
                 "fraction_unbound_plasma": 0.9398, "fraction_unbound_plasma_reference": "Paini 2020",
             }, "status": "reported", "p05": 0, "median": 0.12, "p95": 0.2},
            {"candidate_id": "ligand_1", "cas": "58-08-2", "dtxsid": "DTXSID0020232", "status": "untested",
             "blocker": "httk_missing_ligand"},
        ],
    }


def test_httk_saved_verdict_and_board_expose_only_fixed_scenario_summary(tmp_path: Path) -> None:
    from rfproteina.disease import httk, mcp_api, mcp_layers
    from tests.helpers.disease import screen_page

    private = "PRIVATE LIGAND INFO AND R ARGUMENTS"
    raw, expected = _httk_report(), _httk_report()
    raw.update(attempted=999, reported=999, dose=private, source_path=private)
    raw["provider"]["executable"] = private
    raw["members"][0]["raw_smiles"] = private
    raw["members"][1]["median"] = 999
    payload = {
        "call": "untested", "method_selection": {"prepared_measurements": {httk.METHOD_ID: raw}},
        httk.METHOD_ID: {"untyped_copy": private},
    }
    finished = _screened(tmp_path, payload)
    answer = mcp_api.read_verdict({"directory": str(finished), "include_record": True})
    verdict = answer["runs"][0]["verdict"]
    for obj in (verdict, verdict["record"]):
        assert obj["method_selection"]["prepared_measurements"][httk.METHOD_ID] == expected
        assert httk.METHOD_ID not in obj
    assert private not in json.dumps(answer) and httk.METHOD_ID not in mcp_layers.screen_fields()
    assert verdict["call"] == "untested"

    board = screen_page(payload)
    assert "EPA httk estimated total plasma steady-state concentration for 1 of 2 nominated candidates." in board
    assert "ligand_0 / 5th percentile 0, median 0.12, 95th percentile 0.2 mg/L." in board
    assert "ligand_1 / EPA httk untested." in board
    assert "fixed oral reference intake of 1 mg/kg/day" in board
    assert "matched each reported ligand structure" in board and "EPA DSSTox CCD 2025-12-09" in board
    assert "0.3598" in board and "0.9398" in board and "Paini 2020" in board and private not in board

    schemas = {tool["name"]: tool["outputSchema"] for tool in _server()._TOOLS if "outputSchema" in tool}
    jsonschema.validate(answer, schemas["read_verdict"])
    result = {"id": httk.METHOD_ID, "status": "partial", "measurement": expected}
    selected = {"program": "Humanitys-Last-Disease", "directory": str(finished), "candidate_id": None, "results": [result]}
    validator = jsonschema.Draft202012Validator(schemas["run_selected_screens"])
    assert validator.is_valid(selected)
    for patch in ({"reference_intake_mg_kg_day": 2}, {"scenario": "custom"}, {"scope": "candidate"},
                  {"provider": {**expected["provider"], "in_vitro_uncertainty": True}},
                  {"provider": {**expected["provider"], "identity_source": "unverified"}},
                  {"status": "reported"}, {"status": "untested", "blocker": "httk_no_usable_results"}):
        result["measurement"] = {**expected, **patch}
        assert not validator.is_valid(selected)
    for field in ("reference_inchikey", "human_in_vitro"):
        reported, untested = expected["members"]
        incomplete = {k: v for k, v in reported.items() if k != field}
        result["measurement"] = {**expected, "members": [incomplete, untested]}
        assert not validator.is_valid(selected)


def test_httk_projection_rejects_unattributed_and_out_of_domain_estimates() -> None:
    from rfproteina.disease import httk
    from rfproteina.disease.mcp_api import _httk_view

    raw = _httk_report()
    row = raw["members"][0]
    for patch in ({"p05": False}, {"median": float("nan")}, {"p95": float("inf")}, {"p05": -0.1}, {"p05": 0.3},
                  {"cas": "103-90-3"}, {"dtxsid": "DTXSIDfoo"}, {"compound_id": "verbatim example ligand"},
                  {"reference_inchikey": ""}, {"reference_inchikey": "ABC"}, {"human_in_vitro": None},
                  {"human_in_vitro": {**row["human_in_vitro"], "clint_reference": "unpublished fixture"}},
                  {"human_in_vitro": {**row["human_in_vitro"], "fraction_unbound_plasma": False}},
                  {"human_in_vitro": {**row["human_in_vitro"], "path": "private"}},
                  {"candidate_id": "some/folder"}, {"candidate_id": "ACDEFGHIKLMNPQRSTVWYACDEFGHI"},
                  {"blocker": "untrusted output"}):
        clean = _httk_view({**raw, "members": [{**row, **patch}]})
        assert clean["status"] == "untested" and clean["reported"] == 0 and "provider" not in clean
        assert clean["members"][0]["blocker"] == "httk_record_invalid" and "median" not in clean["members"][0]
    for patch in ({"provider": {**httk.PROVIDER, "seed": False}}, {"manifest_sha256": "b" * 63},
                  {"scope": "candidate"}, {"reference_intake_mg_kg_day": True},
                  {"members": [row, row]}, {"members": [row] * (httk.MAX_ENTRIES + 1)}):
        clean = _httk_view({**raw, **patch})
        assert clean["status"] == "untested" and not clean.get("reported") and "provider" not in clean
    conflict = {**row, "candidate_id": "ligand_2", "compound_id": "smiles_sha256_" + "c" * 64}
    clean = _httk_view({**raw, "members": [row, conflict]})
    assert clean["status"] == "untested" and clean["reported"] == 0
    assert all(member["blocker"] == "httk_ambiguous_ligand" for member in clean["members"])


def test_httk_structure_verification_and_portable_v2_records(tmp_path, monkeypatch) -> None:
    from rfproteina.disease import httk, httk_identity, mcp_api
    from tests.helpers.disease import screen_page

    raw = _httk_report()
    row = raw["members"][0]
    index = {(row["cas"], row["dtxsid"]): row["reference_inchikey"]}
    monkeypatch.setattr(httk_identity, "load_index", lambda: index)
    wrong = {**row, "reference_inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N"}
    result = mcp_api._httk_view({**raw, "scope": "candidate", "members": [wrong]})
    withheld, = result["members"]
    assert withheld["blocker"] == "httk_structure_mismatch" and "median" not in withheld
    assert "reference_inchikey" not in withheld and "human_in_vitro" not in withheld
    untested = {k: v for k, v in row.items() if k not in {"p05", "median", "p95", "human_in_vitro"}}
    untested.update(status="untested", blocker="httk_human_inputs_unavailable")
    assert mcp_api._httk_view({**raw, "members": [untested]})["members"] == [untested]

    def no_local_copy():
        raise httk_identity.IdentityReferenceError("httk_identity_reference_unavailable")

    monkeypatch.setattr(httk_identity, "load_index", no_local_copy)
    assert mcp_api._httk_view(raw) == raw
    legacy = {**raw, "method": "epa_httk_human_css_v1"}
    payload = {"call": "untested", "method_selection": {"prepared_measurements": {httk.METHOD_ID: legacy}}}
    finished = _screened(tmp_path, payload)
    verdict = mcp_api.read_verdict({"directory": str(finished), "include_record": True})["runs"][0]["verdict"]
    for view in (verdict, verdict["record"]):
        assert view["method_selection"]["prepared_measurements"][httk.METHOD_ID] == {
            "status": "untested", "blocker": "httk_legacy_identity_unverified",
        }
    assert "predates independent structure verification" in screen_page(payload)


def _molprobity_report() -> dict:
    from rfproteina.disease import molprobity_geometry as mp
    from rfproteina.disease.corpus_provenance import declare

    backbone = {
        "status": "reported", "evaluated": 50, "outliers": 1, "favored": 47, "allowed": 2, "outlier_percent": 2.0,
    }
    rotamers = {
        "status": "reported", "evaluated": 40, "outliers": 0, "favored": 38, "allowed": 2, "outlier_percent": 0.0,
    }
    clashes = {"status": "reported", "atoms": 700, "serious_clashes": 7, "clashes_per_1000_atoms": 10.0}
    return {
        "method": mp.METHOD, "model_scope": mp.MODEL_SCOPE, "scope": "run", "status": "partial",
        "attempted": 2, "reported": 2, "complete": 1,
        "samples": [
            {"candidate_id": "binder_0", "status": "reported", "input_sha256": "a" * 64,
             "metrics": {"ramachandran": dict(backbone), "rotamers": dict(rotamers), "clashscore": dict(clashes)}},
            {"candidate_id": "binder_1", "status": "partial", "input_sha256": "b" * 64, "metrics": {
                "ramachandran": dict(backbone), "rotamers": dict(rotamers),
                "clashscore": {"status": "untested", "blocker": "molprobity_incomplete_heavy_atoms"},
            }},
        ],
        "provider": {"cctbx_base": mp.CCTBX_VERSION, "device": "cpu",
                     "clash_preparation": "reduce2_no_flips", "clash_method": "probe2"},
        "corpus_provenance": declare(mp.METHOD),
    }


def test_molprobity_preserves_separate_metrics_and_recomputes_only_rates(tmp_path: Path) -> None:
    from rfproteina.disease import mcp_api, mcp_layers
    from rfproteina.disease import molprobity_geometry as mp
    from tests.helpers.disease import screen_page

    geometry = "PRIVATE COORDINATES AND GEOMETRY"
    raw, clean = _molprobity_report(), _molprobity_report()
    raw.update(attempted=999, reported=999, complete=998, composite_score=0, pass_call=True)
    raw["provider"].update(executable=geometry)
    raw["samples"][0].update(cif=geometry)
    raw["samples"][0]["metrics"]["ramachandran"].update(outlier_percent=100, coordinates=geometry)
    raw["samples"][0]["metrics"]["clashscore"].update(clashes_per_1000_atoms=0, probes=[geometry])
    raw["samples"][1]["metrics"]["clashscore"].update(serious_clashes=0, clashes_per_1000_atoms=0)
    payload = {"call": "untested", mp.METHOD_ID: raw}
    finished = _screened(tmp_path, payload)
    answer = mcp_api.read_verdict({"directory": str(finished), "include_record": True})
    verdict = answer["runs"][0]["verdict"]
    assert verdict["call"] == "untested" and verdict[mp.METHOD_ID] == verdict["record"][mp.METHOD_ID] == clean
    assert geometry not in json.dumps(answer) and mp.METHOD_ID not in mcp_layers.screen_fields()

    board = screen_page(payload)
    assert "MolProbity whole-model geometry for 2 of 2 selected models. All three checks were available for 1." in board
    assert "binder_0 / Ramachandran outliers 1 of 50 evaluated residues (2%)." in board
    assert "binder_0 / Side-chain rotamer outliers 0 of 40 evaluated residues (0%)." in board
    assert "binder_0 / Clashscore 10 serious clashes per 1,000 atoms (7 clashes, 700 eligible atoms" in board
    assert "binder_1 / Clashscore untested." in board and "lacks heavy atoms" in board
    assert "do not establish binding affinity" in board and geometry not in board

    schemas = {tool["name"]: tool["outputSchema"] for tool in _server()._TOOLS if "outputSchema" in tool}
    jsonschema.validate(answer, schemas["read_verdict"])
    validator = jsonschema.Draft202012Validator(schemas["run_selected_screens"])
    result = {"id": mp.METHOD_ID, "status": "partial", "measurement": clean}
    selected = {"program": "Humanitys-Last-Disease", "directory": str(finished), "candidate_id": None,
                "results": [result]}
    assert validator.is_valid(selected)
    for patch in ({"model_scope": "binder_only"}, {"pass_call": True}, {"scope": "candidate"}):
        result["measurement"] = {**clean, **patch}
        assert not validator.is_valid(selected)
    for metric, patch in (
        ("clashscore", {"atoms": 0}), ("clashscore", {"serious_clashes": False}),
        ("rotamers", {"outlier_percent": 200}), ("rotamers", {"outliers": -1}),
        ("ramachandran", {"blocker": "molprobity_metric_failed"}), ("ramachandran", {"allowed": "2"}),
    ):
        sample = clean["samples"][0]
        metrics = sample["metrics"]
        result["measurement"] = {**clean, "samples": [{**sample, "metrics": {
            **metrics, metric: {**metrics[metric], **patch},
        }}]}
        assert not validator.is_valid(selected)


def test_molprobity_refuses_unattributed_or_unserializable_results() -> None:
    from rfproteina.disease.mcp_api import _molprobity_view
    from rfproteina.disease.molprobity_geometry import MAX_MODELS

    report = _molprobity_report()
    row = report["samples"][0]
    zero = _molprobity_view({**report, "samples": [{**row, "metrics": {**row["metrics"], "clashscore": {
        "status": "reported", "atoms": 30, "serious_clashes": 0,
    }}}]})
    assert zero["status"] == "reported" and zero["samples"][0]["metrics"]["clashscore"]["clashes_per_1000_atoms"] == 0
    for patch in ({"input_sha256": "A" * 64}, {"input_sha256": "a" * 64 + "\n"}, {"candidate_id": "hidden/pdb"},
                  {"candidate_id": "ACDEFGHIKLMNPQRSTVWYACDEFGHI"}, {"blocker": "arbitrary stderr"}):
        refused = _molprobity_view({**report, "samples": [{**row, **patch}]})
        assert refused["status"] == "untested" and refused["reported"] == refused["complete"] == 0
        assert "metrics" not in refused["samples"][0]
    for metric, patch in (
        ("clashscore", {"atoms": 0}), ("clashscore", {"serious_clashes": -2}),
        ("clashscore", {"serious_clashes": 2**53}), ("ramachandran", {"evaluated": 0}),
        ("ramachandran", {"favored": []}), ("ramachandran", {"allowed": False}), ("rotamers", {"allowed": 3}),
    ):
        metrics = row["metrics"]
        bad = _molprobity_view({**report, "samples": [{**row, "metrics": {
            **metrics, metric: {**metrics[metric], **patch},
        }}]})
        assert bad["status"] == "partial" and bad["reported"] == 1 and bad["complete"] == 0
        assert bad["samples"][0]["metrics"][metric] == {"status": "untested", "blocker": "molprobity_record_invalid"}
    wrong_provider = _molprobity_view({**report, "provider": {**report["provider"], "clash_method": "contact_count"}})
    assert wrong_provider["status"] == "untested" and "provider" not in wrong_provider
    duplicate = _molprobity_view({**report, "samples": [row, row, report["samples"][1]]})
    assert duplicate["reported"] == 1 and duplicate["complete"] == 0
    assert [sample["blocker"] for sample in duplicate["samples"][:2]] == ["molprobity_candidate_ambiguous"] * 2
    conflicting = _molprobity_view({**report, "scope": "candidate"})
    assert conflicting["status"] == "untested" and conflicting["samples"] == []
    assert _molprobity_view({**report, "samples": [row] * (MAX_MODELS + 1)})["blocker"] == "molprobity_too_many_models"
    assert _molprobity_view({"blocker": "input_unavailable"}) == {"status": "untested", "blocker": "input_unavailable"}


def _caco2_report() -> dict:
    from rfproteina.disease import ligand_caco2 as caco2

    compound = "smiles_sha256_" + "c3" * 32
    heads = {
        name: {"endpoint": endpoint, "scale": caco2.SCALE, "status": "predicted", "model_score": score}
        for (name, endpoint), score in zip(caco2.HEADS, (-1.25, 0.0), strict=True)
    }
    return {
        "method": caco2.METHOD, "scale": caco2.SCALE, "scope": "candidate_table", "status": "predicted",
        "inputs": [{"candidate_id": "lead_1", "compound_id": compound}],
        "compounds": [{"method": caco2.METHOD, "compound_id": compound, "status": "predicted", "heads": heads}],
    }


def test_caco2_keeps_two_native_scores_without_graphs_or_units(tmp_path: Path) -> None:
    from rfproteina.disease import ligand_caco2 as caco2
    from rfproteina.disease import mcp_api, mcp_layers
    from rfproteina.disease.corpus_provenance import declare
    from tests.helpers.disease import screen_page

    raw_graph = "CC(=O)Oc1ccccc1C(=O)O"
    report = _caco2_report()
    provider = {
        "device": "cpu", "release": caco2.RELEASE, "checkpoint_sha256": caco2.CHECKPOINT_SHA256,
        "chemprop": "2.3.1", "rdkit": "2025.9.2", "torch": "2.8.0+cpu",
    }
    clean_inputs = [dict(report["inputs"][0])]
    clean_heads = {key: dict(value) for key, value in report["compounds"][0]["heads"].items()}
    report.update(provider={**provider, "raw_graph": raw_graph}, corpus_provenance=declare(caco2.METHOD))
    report["inputs"][0]["smiles"] = raw_graph
    report["compounds"][0]["heads"]["efflux_ratio"]["features"] = [raw_graph]
    report["compounds"][0]["heads"]["mdck"] = {"endpoint": "MDCKER", "model_score": 17, "smiles": raw_graph}
    payload = {"call": "untested", caco2.METHOD_ID: report}
    run = _screened(tmp_path, payload)
    answer = mcp_api.read_verdict({"directory": str(run), "include_record": True})
    verdict = answer["runs"][0]["verdict"]
    view = verdict[caco2.METHOD_ID]
    assert view == verdict["record"][caco2.METHOD_ID]
    assert view["inputs"] == clean_inputs and view["compounds"][0]["heads"] == clean_heads
    assert view["provider"] == provider and view["corpus_provenance"] == declare(caco2.METHOD)
    assert view["unique_complete"] == view["unique_predicted"] == view["unique_compounds"] == view["attempted"] == 1
    assert view["status"] == "predicted" and verdict["call"] == "untested"
    assert raw_graph not in json.dumps(answer) and "MDCKER" not in json.dumps(answer)
    assert caco2.METHOD_ID not in mcp_layers.screen_fields()

    displayed = screen_page(payload)
    assert "Efflux ratio native model score -1.25." in displayed
    assert "Apparent permeability native model score 0." in displayed
    assert "no physical units or pass cutoff" in displayed
    assert raw_graph not in displayed
    schemas = {tool["name"]: tool["outputSchema"] for tool in _server()._TOOLS if "outputSchema" in tool}
    jsonschema.validate(answer, schemas["read_verdict"])
    validator = jsonschema.Draft202012Validator(schemas["run_selected_screens"])
    result = {"id": caco2.METHOD_ID, "status": "predicted", "measurement": view}
    selected = {"program": "Humanitys-Last-Disease", "directory": str(run), "candidate_id": None, "results": [result]}
    assert validator.is_valid(selected)
    view["compounds"][0]["heads"]["apparent_permeability"]["unit"] = "cm/s"
    assert not validator.is_valid(selected)


def test_caco2_rejects_mislabeled_and_nonfinite_heads_without_losing_the_other() -> None:
    from rfproteina.disease.mcp_api import _caco2_view

    for alteration in (
        {"endpoint": "Caco2Papp"}, {"scale": "efflux_ratio"}, {"unit": "ratio"},
        {"model_score": float("inf")}, {"model_score": float("nan")}, {"model_score": True},
        {"model_score": 10**400}, {"model_score": "0.25"},
    ):
        raw = _caco2_report()
        raw["compounds"][0]["heads"]["efflux_ratio"].update(alteration)
        view = _caco2_view(raw)
        assert view["status"] == "partial" and view["unique_predicted"] == 1 and view["unique_complete"] == 0
        compound = view["compounds"][0]
        assert compound["status"] == "partial"
        assert compound["heads"]["efflux_ratio"]["blocker"] == "invalid_provider_result"
        assert "model_score" not in compound["heads"]["efflux_ratio"]
        assert compound["heads"]["apparent_permeability"]["model_score"] == 0.0


def test_caco2_projection_does_not_trust_stored_coverage_or_provenance() -> None:
    from rfproteina.disease.mcp_api import _caco2_view
    from tests.helpers.disease import screen_page

    raw = _caco2_report()
    row = raw["compounds"][0]
    raw.update(
        attempted=255, unique_predicted=255, unique_complete=99,
        provider={"device": "cuda", "release": "foreign", "checkpoint_sha256": "e8" * 32},
        corpus_provenance={"bogus": "release"},
    )
    raw["compounds"].extend([{**row, "compound_id": "smiles_sha256_" + "f6" * 32}, row])
    view = _caco2_view(raw)
    assert view["status"] == "untested" and view["attempted"] == view["unique_compounds"] == 1
    assert view["unique_predicted"] == view["unique_complete"] == 0 and len(view["compounds"]) == 1
    assert "corpus_provenance" not in view and view["provider"] == {}
    assert all(head["blocker"] == "duplicate_provider_row" for head in view["compounds"][0]["heads"].values())
    raw.update(status="untested", blocker="caco2_checkpoint_invalid")
    html = screen_page({"call": "untested", "ligand_caco2": raw})
    assert "Caco-2 prediction untested" in html
    assert "does not match the pinned Caco-2 release" in html
    assert "native model score -1.25" not in html


def _version_three_record() -> dict:
    """A screen in the shape the current writer produces, trimmed to the keys under test."""
    return {
        "program": "Humanitys-Last-Disease",
        "screen_version": 3,
        "call": "axis_hit",
        "bar": "sequence",
        "bar_axes": ["mhc_i", "mhc_ii", "ada"],
        "states": {"mhc_i": "hit"},
        "hits": ["mhc_i"],
        "untested": [],
        "code": {"commit": "0" * 40, "tree_state": "clean"},
        "cuts": {
            "gravy_max": 0.1,
            "gravy_max_source": "program_convention",
            "mhc_i_rank_max": 2.0,
            "plasma_abundance_max": None,
        },
        "metrics_file": "inference_metrics/per_example_metrics.csv",
        "metrics_blocker": "column_repeated",
        "chain_blocker": "chain_disagreement",
        # The shape a real screen writes, taken from one: a threshold names the
        # metric column, which kind of bound it is, the cut, the value reached
        # and whether it held. A criterion the oracle never scored is
        # not_evaluated with a null measurement, which is the ordinary case on a
        # run with no refolding configured.
        "refold": {
            "criteria": [
                {
                    "name": "ppi",
                    "state": "fail",
                    "thresholds": [
                        {
                            "metric": "oracle_ppi.ptm_coi",
                            "bound": "min",
                            "cut": 80.0,
                            "measured": 31.0,
                            "met": False,
                        }
                    ],
                    "failed_thresholds": ["oracle_ppi.ptm_coi"],
                },
                {
                    "name": "unconditional",
                    "state": "not_evaluated",
                    "thresholds": [
                        {
                            "metric": "refolded_backbone_rmsd.rmsd",
                            "bound": "max",
                            "cut": 1.5,
                            "measured": None,
                            "met": None,
                        }
                    ],
                },
            ],
            "declared": 11,
            "evaluated": 1,
            "criteria_file": "configs/metrics/pass_criteria/all_pass_criteria.yaml",
        },
        "solubility": {"solubility_blocker": "solubility_not_configured"},
        "design_diversity": {"method": "pairwise_identity", "designs": 8, "distinct": 1},
        "sequence_liability": {
            "mhc_i": {
                "state": "hit",
                "measurement": 3.0,
                "blocker": None,
                "source": "measured",
                "method": "calis_2013_class_i_9mer",
                "presentation": "restricted",
                "covered_examples": ["binder_0"],
                "uncovered_examples": ["binder_1", "binder_2"],
                "mask_unpublished_alleles": ["HLA-B*08:01"],
                "displaced_method": "calis_2013_class_i_9mer",
                "displaced_table": "mhc_i",
            }
        },
        "target_liability": {
            "plasma_abundance": {
                "state": "untested",
                "measurement": 4130.0,
                "blocker": "threshold_not_set",
                "source": "untested",
                "assay": "immunoassay",
                "units": "pg/L",
            },
            "plasma_hub": {
                "state": "clear",
                "measurement": 4,
                "blocker": None,
                "source": "measured",
                "threshold": 10,
                "confidence": 700,
                "network": "physical",
                "release": "12.0",
            },
        },
        "nucleic_liability": {
            "off_target_hybridization": {
                "state": "clear",
                "measurement": 0,
                "blocker": None,
                "source": "measured",
                "reference_file": "transcriptome.fa",
                "reference_digest": "a" * 16,
                "reference_bases": 1_234_567,
            }
        },
    }


def test_a_version_three_screen_returns_the_blocks_that_qualify_its_verdict(tmp_path: Path) -> None:
    """Each of these was on disk and filtered out of the answer."""
    from rfproteina.disease import mcp_api

    _screened(tmp_path, _version_three_record())
    run = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]
    assert "error" not in run, run.get("error")
    verdict = run["verdict"]
    assert verdict["bar_axes"] == ["mhc_i", "mhc_ii", "ada"], "a bar name cannot be checked without its axes"
    assert verdict["cuts"]["gravy_max_source"] == "program_convention"
    assert verdict["metrics_file"] == "inference_metrics/per_example_metrics.csv"
    assert verdict["metrics_blocker"] == "column_repeated"
    assert verdict["chain_blocker"] == "chain_disagreement"
    failed = verdict["refold"]["criteria"][0]["failed_thresholds"]
    assert failed == ["oracle_ppi.ptm_coi"], "a false ppi_pass named none of the bounds behind it; this record does"
    missed = verdict["refold"]["criteria"][0]["thresholds"][0]
    assert missed["cut"] == 80.0, "the cut the criterion declares"
    assert missed["measured"] == 31.0, "the least favourable value the run reached against it"
    assert missed["met"] is False, "and whether it held"
    assert verdict["refold"]["declared"] == 11
    assert verdict["solubility"] == {"solubility_blocker": "solubility_not_configured"}
    assert verdict["design_diversity"]["distinct"] == 1, "eight designs and one molecule is one shot, not eight"


def test_a_version_three_screen_returns_the_qualifiers_on_each_axis(tmp_path: Path) -> None:
    """A measurement is not a verdict without what produced it and what it covers."""
    from rfproteina.disease import mcp_api

    _screened(tmp_path, _version_three_record())
    verdict = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]
    mhc = verdict["sequence_liability"]["mhc_i"]
    assert mhc["method"] == "calis_2013_class_i_9mer", "a supplied score and a computed one read alike without this"
    assert mhc["covered_examples"] == ["binder_0"]
    assert mhc["uncovered_examples"] == ["binder_1", "binder_2"], "the designs this state does not speak for"
    assert mhc["mask_unpublished_alleles"] == ["HLA-B*08:01"]
    assert mhc["presentation"] == "restricted"
    assert mhc["displaced_table"] == "mhc_i"
    abundance = verdict["target_liability"]["plasma_abundance"]
    assert abundance["assay"] == "immunoassay", "units said pg/L and nothing said how it was obtained"
    hub = verdict["target_liability"]["plasma_hub"]
    assert (hub["confidence"], hub["network"], hub["release"]) == (700, "physical", "12.0")
    off_target = verdict["nucleic_liability"]["off_target_hybridization"]
    assert off_target["reference_file"] == "transcriptome.fa"
    assert off_target["reference_bases"] == 1_234_567, "a clear against a stub is not a clear against a transcriptome"


def test_an_axis_says_what_scale_its_measurement_is_on(tmp_path: Path) -> None:
    """The producers write these beside keys the view already kept, line for line.

    ``_mhc_record`` writes not_nm_columns, rank, ic50, trigger and
    unscored_peptides one after another, and three of the five were returned.
    These axes carry no units and no threshold, so a clear arrived as a bare
    number that is a percentile rank on one table and an affinity in nM on
    another, with nothing saying which.
    """
    from rfproteina.disease import mcp_api

    record = _version_three_record()
    record["sequence_liability"]["mhc_i"].update(
        {
            "state": "clear",
            "measurement": 0.8,
            "trigger": "rank",
            "rank": 0.8,
            "ic50": None,
            "binding_peptides": 3,
            "reference_density": 1.2,
            "reference_method_undeclared": True,
            "unparsed_columns": ["ic50_nm"],
        }
    )
    record["sequence_liability"]["proteome_offtarget"] = {
        "state": "clear",
        "measurement": 0,
        "blocker": None,
        "source": "measured",
        "proteome_kmers": 20_402_131,
        "proteome_digest": "9f" * 8,
    }
    record["target_liability"]["paralog"] = {
        "state": "untested",
        "measurement": 0,
        "blocker": "cluster_singleton_not_evidence",
        "source": "untested",
        "compara_examined": 12,
        "compara_blocker": "gene_not_in_compara",
    }
    record["toxicology"] = {
        "herg": {
            "state": "clear",
            "measurement": 10000.0,
            "blocker": None,
            "source": "measured",
            "units": "nM",
            "threshold": 10000.0,
            "qualifier": ">",
            "n_readings": 3,
        }
    }
    record["nucleic_liability"]["off_target_hybridization"].update({"seed_width": 15, "seeds_searched": 86})
    _screened(tmp_path, record)
    answer = mcp_api.read_verdict({"directory": str(tmp_path)})
    run = answer["runs"][0]
    assert "error" not in run, run.get("error")
    verdict = run["verdict"]
    mhc = verdict["sequence_liability"]["mhc_i"]
    assert (mhc["trigger"], mhc["rank"], mhc["ic50"]) == ("rank", 0.8, None), "which scale the number is on"
    assert mhc["binding_peptides"] == 3, "the numerator of the density reference_density is the bar for"
    assert mhc["reference_method_undeclared"] is True, "the bar came from a tool neither table named"
    assert mhc["unparsed_columns"] == ["ic50_nm"]
    off_target = verdict["sequence_liability"]["proteome_offtarget"]
    assert off_target["proteome_kmers"] == 20_402_131, "which proteome the absence claim is about, and how large"
    assert off_target["proteome_digest"] == "9f" * 8
    paralog = verdict["target_liability"]["paralog"]
    assert (paralog["compara_examined"], paralog["compara_blocker"]) == (12, "gene_not_in_compara")
    herg = verdict["toxicology"]["herg"]
    assert herg["qualifier"] == ">", "a verdict read off a bound must not look like one read off a number"
    assert herg["n_readings"] == 3
    oligo = verdict["nucleic_liability"]["off_target_hybridization"]
    assert (oligo["seed_width"], oligo["seeds_searched"]) == (15, 86), "the width and the denominator of a zero"
    schema = {str(tool["name"]): tool.get("outputSchema") for tool in _server()._TOOLS}["read_verdict"]
    jsonschema.validate(answer, schema)


def test_the_whole_version_three_answer_validates_against_its_own_schema(tmp_path: Path) -> None:
    """A client that validates throws away the whole answer over one undeclared field."""
    from rfproteina.disease import mcp_api

    _screened(tmp_path, _version_three_record())
    answer = mcp_api.read_verdict({"directory": str(tmp_path)})
    schema = {str(tool["name"]): tool.get("outputSchema") for tool in _server()._TOOLS}["read_verdict"]
    jsonschema.validate(answer, schema)


def test_a_new_block_of_the_wrong_type_is_named_rather_than_returned(tmp_path: Path) -> None:
    """These fields are typed in the schema, so a hand-edited file must not reach a client."""
    from rfproteina.disease import mcp_api

    record = _version_three_record()
    record["cuts"] = "not an object"
    record["refold"] = 3
    _screened(tmp_path, record)
    verdict = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]
    assert "cuts" not in verdict and "refold" not in verdict
    assert {"cuts", "refold"} <= set(verdict["malformed"])


def test_the_version_four_reports_arrive_without_the_design_slice_or_the_molecule(tmp_path: Path) -> None:
    """The four reports reach a client, less two keys only screen.json should hold.

    ``repeat_sequence`` is a slice of the designed construct, and an RNA slice passed
    the sequence rule untouched; ``smiles`` is the screened molecule, which the same
    rule mangled. A tree walk holds the bulky reports back the way it holds refold.
    """
    from rfproteina.disease import mcp_api, mcp_layers

    record = _version_three_record()
    record["designed_interface_quality"] = {"interface_quality_blocker": "coordinates_absent"}
    record["protein_sol"] = {"samples": [{"example_id": "sample-1", "blocker": "protein_sol_sample_missing"}]}
    record["mhc_class_i"] = {"n_windows": 68, "predictor_available": False}
    record["synthesis"] = {"longest_repeat": 26, "repeat_sequence": "AUGGCUACGUUAGCCAUGGCUACGUU"}
    record["ligand_alerts"] = {"smiles": "CCCCCCCCCCCCCCCC(=O)O", "heavy_atoms": 18}
    _screened(tmp_path, record)
    answer = mcp_api.read_verdict({"directory": str(tmp_path)})
    verdict = answer["runs"][0]["verdict"]
    assert verdict["designed_interface_quality"] == {"interface_quality_blocker": "coordinates_absent"}
    assert verdict["protein_sol"] == record["protein_sol"]
    assert "protein_sol" not in mcp_layers.screen_fields()
    assert verdict["mhc_class_i"] == {"n_windows": 68, "predictor_available": False}
    assert verdict["synthesis"] == {"longest_repeat": 26}, "the length stays; the slice of the design does not"
    assert verdict["ligand_alerts"] == {"heavy_atoms": 18}, "the molecule stays in screen.json"
    schema = {str(tool["name"]): tool.get("outputSchema") for tool in _server()._TOOLS}["read_verdict"]
    jsonschema.validate(answer, schema)
    assert {"designed_interface_quality", "synthesis", "ligand_alerts"} <= mcp_layers._SCREEN_OMITTED
    assert "mhc_class_i" in mcp_layers.screen_fields()


def test_a_fresh_verdict_names_the_files_it_rested_on(tmp_path: Path) -> None:
    """``replaced.changed_inputs`` cannot answer this: a first screen has no ``replaced``.

    The size and the digest are the point. A swapped or rebuilt predictor table
    is otherwise invisible between two runs, and the screen it changed could
    say only that the verdict changed.
    """
    from rfproteina.disease import mcp_api

    record = _version_three_record()
    record["inputs"] = [
        {"file": "mhc_i.tsv", "bytes": 4096, "digest": "a" * 64},
        {"file": "toxicology.tsv", "bytes": 512, "digest": "b" * 64, "link": True},
    ]
    assert "replaced" not in record, "the case this field exists for"
    _screened(tmp_path, record)
    answer = mcp_api.read_verdict({"directory": str(tmp_path)})
    verdict = answer["runs"][0]["verdict"]
    assert [entry["file"] for entry in verdict["inputs"]] == ["mhc_i.tsv", "toxicology.tsv"]
    assert verdict["inputs"][1]["link"] is True
    assert "malformed" not in verdict
    schema = {str(tool["name"]): tool.get("outputSchema") for tool in _server()._TOOLS}["read_verdict"]
    jsonschema.validate(answer, schema)


def test_the_two_sentinel_input_lists_are_returned_as_they_are_stored(tmp_path: Path) -> None:
    """Withheld whole, both of them: a truncated list would read as a complete one."""
    from rfproteina.disease import mcp_api

    for stored, expected in (
        ([{"file": "inputs_not_listed"}], "inputs_not_listed"),
        ([{"file": "inputs_truncated", "files": 91}], "inputs_truncated"),
    ):
        folder = tmp_path / expected
        folder.mkdir()
        record = _version_three_record()
        record["inputs"] = stored
        _screened(folder, record)
        verdict = mcp_api.read_verdict({"directory": str(folder)})["runs"][0]["verdict"]
        assert verdict["inputs"][0]["file"] == expected
        assert "inputs" not in verdict.get("malformed", [])


def test_an_inputs_list_of_the_wrong_shape_is_named_rather_than_returned(tmp_path: Path) -> None:
    """It is the one new field that is a list, so the list check has to hold."""
    from rfproteina.disease import mcp_api

    record = _version_three_record()
    record["inputs"] = ["mhc_i.tsv", "toxicology.tsv"]
    _screened(tmp_path, record)
    verdict = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]
    assert "inputs" not in verdict
    assert "inputs" in verdict["malformed"]


def test_the_two_no_call_reports_reach_a_client_without_the_whole_record(tmp_path: Path) -> None:
    """They were excluded as no-call beside six no-call blocks that are returned.

    Returned whole, with nothing held back: neither carries a sequence, a
    molecule or a path. A residue composition is a count per letter and a site
    list is residue numbers. The tree walk drops both for size, as it does the
    other reports.
    """
    from rfproteina.disease import mcp_api, mcp_layers

    record = _version_three_record()
    record["chemical_liability"] = {
        "deamidation_count": 1,
        "deamidation_sites": [{"residue": 35, "motif": "NG"}],
        "net_charge_ph74": 1.2994,
        "isoelectric_point": 8.7386,
        "pka_set": "bjellqvist_1994",
        "free_cysteine": "untested",
    }
    record["modality_screen"] = {
        "binder_kind": "protein",
        "length": 68,
        "aliphatic_index": 81.6176,
        "molecular_weight": 8004.0886,
        "extinction_280_reduced": 8480,
        "extinction_source": "pace_1995",
        "half_life_mammalian": "30 hour",
        "residue_composition": {"A": 2, "C": 2},
        "atom_counts": {"C": 353, "H": 557},
        "furin_count": 0,
        "transmembrane_spans": [],
    }
    _screened(tmp_path, record)
    answer = mcp_api.read_verdict({"directory": str(tmp_path)})
    verdict = answer["runs"][0]["verdict"]
    assert verdict["chemical_liability"]["pka_set"] == "bjellqvist_1994", "what the charge rests on"
    assert verdict["chemical_liability"]["deamidation_sites"] == [{"residue": 35, "motif": "NG"}]
    assert verdict["modality_screen"]["extinction_source"] == "pace_1995"
    assert verdict["modality_screen"]["residue_composition"] == {"A": 2, "C": 2}
    assert "malformed" not in verdict
    jsonschema.validate(
        answer, {str(tool["name"]): tool.get("outputSchema") for tool in _server()._TOOLS}["read_verdict"]
    )
    assert {"chemical_liability", "modality_screen"} <= mcp_layers._SCREEN_OMITTED
    assert "inputs" in mcp_layers.screen_fields(), "the one new field a tree reader compares between runs"
    walked = mcp_layers.read_results({"directory": str(tmp_path)})["screens"][0]
    assert "modality_screen" not in walked and "chemical_liability" not in walked


def _with_provenance() -> dict:
    """A screen carrying the nested provenance both blocks that have one now write.

    Nested rather than top-level, which is why the pin at the head of this file
    stayed green through the change: it compares top-level names, and ``cuts``
    was already one.
    """
    record = _version_three_record()
    record["cuts"]["provenance"] = {
        "gravy_max": {
            "kind": "convention",
            "source": "program_convention",
            "caveat": "No published bar says where a developable protein stops.",
        },
        "mhc_i_rank_max": {
            "kind": "recommended",
            "source": "iedb_class_i_selection",
            "caveat": "IEDB's recommended class I selection: percentile rank <= 1, or IC50 <= 500 nM.",
        },
    }
    record["designed_interface_quality"] = {
        "interface_polar_contacts": 9,
        "provenance": {
            "interface_polar_contacts": {
                "kind": "borrowed",
                "source": "bindcraft_average_n_interface_hbonds",
                "caveat": "BindCraft's default of 3. DEFINITION DIFFERS: it counts Rosetta hydrogen bonds.",
                "bar": {"min": 3},
            },
            "shape_complementarity": {
                "kind": "reported_only",
                "source": "lawrence_colman_1993_scale_differs",
                "bar": None,
            },
        },
    }
    return record


def test_the_kind_vocabulary_is_the_one_the_program_declares() -> None:
    """The check imports the closed set rather than copying it.

    A copy is the failure this record exists to prevent. The set gained
    ``borrowed`` and ``reported_only`` after the first draft of the check was
    written, and against a literal list every borrowed BindCraft default would
    have been dropped from the verdict as malformed -- silently turning the
    disclosure that a bar is borrowed into no disclosure at all.
    """
    from rfproteina.disease import mcp_api
    from rfproteina.disease.threshold_provenance import KINDS

    assert KINDS, "the vocabulary is not empty"
    assert all(mcp_api._is_kind(kind) for kind in KINDS)
    assert not mcp_api._is_kind("published_somewhere")
    assert not mcp_api._is_kind(None)
    assert set(mcp_api._PROVENANCE_TYPES) == set(
        mcp_api._PROVENANCE_FIELDS
    ), "every provenance field is checked, for the reason _AXIS_TYPES exists"
    declared = set(_server()._PROVENANCE_SCHEMA["additionalProperties"]["properties"])
    assert declared == set(mcp_api._PROVENANCE_FIELDS), "the provenance view and its published shape have drifted apart"
    assert "required" not in _server()._PROVENANCE_SCHEMA["additionalProperties"], (
        "a field is dropped when it is malformed, so requiring one would make a "
        "validating client discard the whole answer over a hand-edited caveat"
    )


def test_a_threshold_says_what_kind_of_number_it_is(tmp_path: Path) -> None:
    """A reader cannot tell a published cut from a borrowed default without this.

    Both blocks that carry the record are checked here: the cuts the verdict was
    taken against, and the interface bars, which also carry the bar itself
    because the measurement and its threshold otherwise sit in two files.
    """
    from rfproteina.disease import mcp_api

    _screened(tmp_path, _with_provenance())
    answer = mcp_api.read_verdict({"directory": str(tmp_path)})
    verdict = answer["runs"][0]["verdict"]
    cuts = verdict["cuts"]["provenance"]
    assert cuts["gravy_max"]["kind"] == "convention"
    assert cuts["mhc_i_rank_max"]["source"] == "iedb_class_i_selection"
    assert "malformed" not in verdict
    # The cuts themselves are untouched beside their account.
    assert verdict["cuts"]["gravy_max"] == 0.1
    interface = verdict["designed_interface_quality"]["provenance"]
    assert interface["interface_polar_contacts"]["bar"] == {"min": 3}
    assert interface["shape_complementarity"]["bar"] is None, "no cut adopted is an answer, not a gap"
    schema = {str(tool["name"]): tool.get("outputSchema") for tool in _server()._TOOLS}["read_verdict"]
    jsonschema.validate(answer, schema)


def test_a_kind_this_program_does_not_declare_is_named_rather_than_returned(tmp_path: Path) -> None:
    """One bad field costs that field, not the entry, and not the other eleven accounts."""
    from rfproteina.disease import mcp_api

    record = _with_provenance()
    record["cuts"]["provenance"]["gravy_max"]["kind"] = "invented"
    record["cuts"]["provenance"]["mhc_i_rank_max"]["caveat"] = 17
    record["cuts"]["provenance"]["plasma_fold_max"] = "not an entry"
    _screened(tmp_path, record)
    answer = mcp_api.read_verdict({"directory": str(tmp_path)})
    verdict = answer["runs"][0]["verdict"]
    kept = verdict["cuts"]["provenance"]
    assert "kind" not in kept["gravy_max"], "a kind outside the closed set is not a kind"
    assert kept["gravy_max"]["source"] == "program_convention", "the slug two tests pin still travels"
    assert "caveat" not in kept["mhc_i_rank_max"]
    assert kept["mhc_i_rank_max"]["kind"] == "recommended", "the sound half of the entry stands"
    assert "plasma_fold_max" not in kept
    assert {
        "cuts.provenance.gravy_max.kind",
        "cuts.provenance.mhc_i_rank_max.caveat",
        "cuts.provenance.plasma_fold_max",
    } <= set(verdict["malformed"])
    schema = {str(tool["name"]): tool.get("outputSchema") for tool in _server()._TOOLS}["read_verdict"]
    jsonschema.validate(answer, schema)


def test_a_tree_of_runs_does_not_repeat_the_threshold_prose(tmp_path: Path) -> None:
    """The accounts are the same on every run; the numbers are not.

    Measured at 4,602 bytes on a protein screen, which is more than any block
    the walk already drops outright. ``cuts`` itself has to stay: the numbers a
    verdict was taken against are what the verdict means.
    """
    from rfproteina.disease import mcp_api, mcp_layers

    _screened(tmp_path, _with_provenance())
    result = mcp_layers.read_results({"directory": str(tmp_path)})
    cuts = result["screens"][0]["cuts"]
    assert "provenance" not in cuts
    assert cuts["gravy_max"] == 0.1, "the cut stays; only the prose about it goes"
    assert "iedb_class_i_selection" not in json.dumps(result)
    assert mcp_layers._SCREEN_TRIMMED["cuts"] == frozenset({"provenance"})
    assert "cuts" in mcp_layers.screen_fields(), "trimmed, not omitted"
    # The one run's own answer still carries it, which is where a reader goes.
    verdict = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]
    assert "provenance" in verdict["cuts"]


def test_the_stored_screen_keeps_every_account_the_view_trims(tmp_path: Path) -> None:
    """``include_record`` is the full record, so nothing here is lost to a reader who asks."""
    from rfproteina.disease import mcp_api

    _screened(tmp_path, _with_provenance())
    answer = mcp_api.read_verdict({"directory": str(tmp_path), "include_record": True})
    record = answer["runs"][0]["verdict"]["record"]
    assert set(record["cuts"]["provenance"]) == {"gravy_max", "mhc_i_rank_max"}
    assert record["designed_interface_quality"]["provenance"]["shape_complementarity"]["bar"] is None


def _alternatives(pattern: str) -> set[str]:
    """The top-level branches of an alternation. No branch here contains a bar."""
    return {part for part in pattern.split("|") if part}


def test_the_confidence_rule_is_the_same_rule_the_trace_writer_applies() -> None:
    """One rule, two spellings: the reader's copy was missing ``confidence``.

    The writer stores a metric whose name matches its rule; this module reads
    those metrics back through its own. A branch in one and not the other means
    a column kept on purpose is dropped on the way out -- and a trace carrying
    only such columns reaches ``_ranking`` with no shared metric, which orders
    candidates by filename.
    """
    from rfproteina.disease import mcp_layers, refold, trace

    assert trace.trace_metric_kind is refold.trace_metric_kind
    written = _alternatives(refold._CONFIDENCE.pattern)
    read = _alternatives(mcp_layers._HIGHER_IS_BETTER) | _alternatives(mcp_layers._LOWER_IS_BETTER)
    assert written == read, "the trace writer and the MCP reader disagree about what a confidence metric is"


def test_a_metric_named_confidence_is_ranked_as_a_score() -> None:
    """It is matched, and matched in the direction where a higher number is better."""
    from rfproteina.disease import mcp_layers

    assert mcp_layers._CONFIDENCE.search("mean_confidence") is not None
    assert mcp_layers._ERROR_METRIC.search("mean_confidence") is None


def test_the_generation_view_returns_every_key_the_archive_writes(tmp_path: Path) -> None:
    """The archive record's own identity and shape were being dropped from it."""
    from rfproteina.disease import mcp_layers
    from rfproteina.disease.archive import save_generation

    dest = save_generation(
        tmp_path,
        None,
        disease_name="asthma",
        disease_id="MONDO:0004979",
        target_id="IL5",
        group="gene",
        entry_id="8TLD",
        chain="2",
        binder_kind="protein",
    )
    written = set(json.loads((dest / "generation.json").read_text(encoding="utf-8")))
    expected = set(mcp_layers._GENERATION_FIELDS)
    assert written == expected, "the archive writes keys the generation view does not return, or it keeps dead ones"


def test_an_option_whose_value_is_null_is_not_an_option() -> None:
    """``screen: null`` read as a screen to the guards and as no screen to the program.

    It passed both, so the call went down the disease pipeline with nothing to
    answer the confirmation prompt -- the one thing the second guard exists to
    prevent -- after a live disease search.
    """
    import pytest

    from rfproteina.disease import mcp_api

    with pytest.raises(ValueError, match="accept_only or choice"):
        mcp_api.run_program({"options": {"screen": None, "disease": "asthma"}})
    with pytest.raises(ValueError, match="Set disease"):
        mcp_api.run_program({"options": {"screen": None}})
    with pytest.raises(ValueError, match="accept_only or choice"):
        mcp_api.run_program({"options": {"disease": "asthma", "choice": None}})


# --- what a reader needs to weigh one axis -----------------------------------
#
# Each of these was added by a fix whose whole point was that the reader
# could see the field, and the view kept dropping the newest ones. They were
# a separate file; the subject is the same pinning as above.


def _screened_axes(folder: Path, axes: dict) -> Path:
    """A finished run carrying a stored screen, without running one."""
    run = folder / "gene_1ABC_1"
    (run / "inference_samples" / "binder_0").mkdir(parents=True)
    record = {
        "program": "Humanitys-Last-Disease",
        "call": "axis_hit",
        "bar": "sequence",
        "states": {name: entry["state"] for name, entry in axes.items()},
        "hits": [name for name, entry in axes.items() if entry["state"] == "hit"],
        "untested": [],
        "sequence_liability": axes,
    }
    (run / "screen.json").write_text(json.dumps(record), encoding="utf-8")
    return run


def test_a_hit_from_a_partial_scan_says_so_in_the_verdict(tmp_path: Path) -> None:
    """A count over part of a chain is still a hit, and the reader must see which part."""
    from rfproteina.disease import mcp_api

    _screened_axes(
        tmp_path,
        {
            "mhc_i": {
                "state": "hit",
                "measurement": 17.6471,
                "blocker": None,
                "source": "measured",
                "scan_limit": "scan_incomplete",
            }
        },
    )
    axis = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]["sequence_liability"]["mhc_i"]
    assert axis["state"] == "hit"
    assert axis["scan_limit"] == "scan_incomplete", "the hit rests on a partial scan and the view hid it"


def test_an_axis_says_which_cut_it_was_judged_against(tmp_path: Path) -> None:
    """One number decided ten panel axes, and the record now names the cut per axis."""
    from rfproteina.disease import mcp_api

    run = tmp_path / "gene_2ABC_1"
    (run / "inference_samples" / "binder_0").mkdir(parents=True)
    (run / "screen.json").write_text(
        json.dumps(
            {
                "program": "Humanitys-Last-Disease",
                "call": "axis_hit",
                "bar": "coverage",
                "states": {"herg": "hit"},
                "hits": ["herg"],
                "untested": [],
                "toxicology": {
                    "herg": {
                        "state": "hit",
                        "measurement": 50.0,
                        "blocker": None,
                        "source": "measured",
                        "threshold": 10000.0,
                        "threshold_source": "program_convention_10uM",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    axis = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]["toxicology"]["herg"]
    assert axis["threshold"] == 10000.0
    assert axis["threshold_source"] == "program_convention_10uM", "a cut with no provenance reads as published"


def test_an_ordinary_measured_axis_gains_none_of_them(tmp_path: Path) -> None:
    """These fields say something happened. Absent, they must not appear at all."""
    from rfproteina.disease import mcp_api

    _screened_axes(tmp_path, {"mhc_i": {"state": "clear", "measurement": 1.0, "blocker": None, "source": "measured"}})
    axis = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]["sequence_liability"]["mhc_i"]
    assert axis == {"state": "clear", "measurement": 1.0, "blocker": None, "source": "measured"}


def test_the_whole_answer_still_validates_against_its_own_schema(tmp_path: Path) -> None:
    """A field the schema forbids makes a validating client throw the verdict away."""
    import jsonschema

    from rfproteina.disease import mcp_api

    _screened_axes(
        tmp_path,
        {
            "ada": {
                "state": "clear",
                "measurement": 1.0,
                "blocker": None,
                "source": "axis_evidence",
                "curated": "basis_not_stated",
                "measured_state": "hit",
                "measured_blocker": None,
            },
            "mhc_i": {
                "state": "hit",
                "measurement": 17.6471,
                "blocker": None,
                "source": "measured",
                "scan_limit": "unscored_windows",
            },
        },
    )
    answer = mcp_api.read_verdict({"directory": str(tmp_path)})
    schema = {str(tool["name"]): tool.get("outputSchema") for tool in _server()._TOOLS}["read_verdict"]
    jsonschema.validate(answer, schema)


def test_a_clear_carries_the_bar_it_was_taken_against(tmp_path: Path) -> None:
    """A density clear with no reference is a number beside no comparison.

    ``measurement`` alone says 0.8 per 100 windows. Whether that is a clear
    depends entirely on the bar, and the view used to drop it.
    """
    from rfproteina.disease import mcp_api

    _screened_axes(
        tmp_path,
        {
            "mhc_i": {
                "state": "clear",
                "measurement": 0.8,
                "blocker": None,
                "source": "measured",
                "reference_density": 1.2,
                "units": "binders per 100 windows",
            }
        },
    )
    axis = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]["sequence_liability"]["mhc_i"]
    assert axis["reference_density"] == 1.2
    assert axis["units"] == "binders per 100 windows"


def test_a_clear_says_how_much_of_the_scan_was_never_scored(tmp_path: Path) -> None:
    """Declined windows are not non-binders, and a clear rests on there being none."""
    from rfproteina.disease import mcp_api

    _screened_axes(
        tmp_path,
        {
            "mhc_i": {
                "state": "clear",
                "measurement": 0.0,
                "blocker": None,
                "source": "measured",
                "unscored_peptides": 41,
                "presented_complete": False,
            }
        },
    )
    axis = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]["sequence_liability"]["mhc_i"]
    assert axis["unscored_peptides"] == 41
    assert axis["presented_complete"] is False


def test_an_unread_affinity_column_is_named_in_the_verdict(tmp_path: Path) -> None:
    """The columns a verdict could not read are part of what it rests on."""
    from rfproteina.disease import mcp_api

    _screened_axes(
        tmp_path,
        {
            "mhc_i": {
                "state": "untested",
                "measurement": 2.0,
                "blocker": "ic50_not_nm",
                "source": "measured",
                "not_nm_columns": ["ic50", "affinity"],
            }
        },
    )
    axis = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]["sequence_liability"]["mhc_i"]
    assert axis["not_nm_columns"] == ["ic50", "affinity"]


def test_a_plasma_fold_verdict_carries_the_study_it_cites(tmp_path: Path) -> None:
    """That axis refuses to be measured without a study, so the study is the basis."""
    from rfproteina.disease import mcp_api

    run = tmp_path / "gene_1ABC_1"
    (run / "inference_samples" / "binder_0").mkdir(parents=True)
    (run / "screen.json").write_text(
        json.dumps(
            {
                "program": "Humanitys-Last-Disease",
                "call": "axis_hit",
                "bar": "target",
                "states": {"plasma_fold": "hit"},
                "hits": ["plasma_fold"],
                "untested": [],
                "target_liability": {
                    "plasma_fold": {
                        "state": "hit",
                        "measurement": 9.0,
                        "blocker": None,
                        "source": "measured",
                        "threshold": 2.0,
                        "fold_source": "Sun 2023",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    axis = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]["target_liability"]["plasma_fold"]
    assert axis["fold_source"] == "Sun 2023", "a fold verdict without its study is the rumour the axis refuses"


def test_a_clear_says_when_its_bar_may_be_a_hundredfold_out(tmp_path: Path) -> None:
    """The sharpest disclosure on the axis, and the view used to drop it.

    Every percentile rank in a binders-only reference reads the same whether the
    column is a percentage or a fraction of one. The verdict stands -- refusing
    would refuse the ordinary export -- so the disclosure is the only thing
    standing between a reader and a bar that may be a hundredfold wrong.
    """
    from rfproteina.disease import mcp_api

    _screened_axes(
        tmp_path,
        {
            "mhc_i": {
                "state": "clear",
                "measurement": 0.8,
                "blocker": None,
                "source": "measured",
                "reference_density": 1.2,
                "reference_rank_scale_ambiguous": True,
                "reference_proteins_too_short": ["P0CG47"],
            }
        },
    )
    axis = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]["sequence_liability"]["mhc_i"]
    assert axis["state"] == "clear", "the disclosure must not change the verdict"
    assert axis["reference_rank_scale_ambiguous"] is True
    assert axis["reference_proteins_too_short"] == ["P0CG47"]
