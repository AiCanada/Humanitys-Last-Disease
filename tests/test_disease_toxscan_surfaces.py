"""A saved ToxScan export follows the nominated structure through each screening entry point."""

import csv
import hashlib
import json

import jsonschema
import pytest

from rfproteina.disease import (
    mcp_api,
    screen,
    screen_board,
    screen_panel,
    screen_selection,
    toxscan,
    toxscan_validation,
    ui,
)
from scripts import humanitys_last_disease_mcp as server
from scripts import screen_extras


def write_export(directory, candidate_id, overrides=None, *, smiles="CCO"):
    # Representative platform export values exercise matching and presentation without a hosted prediction request.
    csv_file = directory / toxscan.SOURCE_FILE
    row = {
        "SMILES": "OCC", "AD Analysis (AD: Applicability Domain)": "In AD", "Similarity Score in AD analysis": "0.78",
        **dict.fromkeys(toxscan.CLASSIFICATIONS, "NO"), **dict.fromkeys(toxscan.REGRESSIONS, "2.5mg/kg"),
        "Eye irritation": "YES",
        **(overrides or {}),
    }
    with csv_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerow(row)
    (directory / "ligand_candidates.tsv").write_text(f"candidate_id\tsmiles\n{candidate_id}\t{smiles}\nother\tCCC\n")
    (directory / "generation.json").write_text('{"binder_kind":"ligand"}')
    (directory / toxscan.MANIFEST).write_text(json.dumps({
        "version": 1, "candidate_id": candidate_id, "source": toxscan.SOURCE,
        "export_sha256": hashlib.sha256(csv_file.read_bytes()).hexdigest(),
    }))


def test_export_follows_only_its_matched_structure(tmp_path, monkeypatch, capsys):
    pytest.importorskip("rdkit")
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    monkeypatch.setattr(screen_selection, "installations", lambda keys: dict.fromkeys(keys, True))
    monkeypatch.setattr(screen, "_sample_measurements", lambda *args, **kwargs: ([], [], []))
    candidate_id = "lead_1"
    write_export(tmp_path, candidate_id)
    method = toxscan.METHOD_ID
    assert screen_extras.main([method, str(tmp_path), "--candidate", candidate_id]) == 0
    original = json.loads(capsys.readouterr().out)
    assert original["scope"] == "candidate" and original["candidate_id"] == candidate_id
    assert original["identity_check_version"] == 2
    assert original["results"]["classifications"]["Eye irritation"] == "YES"
    assert original["results"]["source_displays"]["Rat oral LD50"] == 2.5
    assert original["applicability_domain"] == "in_domain" and original["similarity_score"] == 0.78
    validation = original["classification_validation"]
    assert set(validation) == set(toxscan.CLASSIFICATIONS)
    assert validation["Ames Mutagenicity"] == {
        "auroc": 0.879, "accuracy": 0.876, "auroc_sd": 0.016, "accuracy_sd": 0.012,
        "assay": "Bacterial reverse-mutation assay",
    }
    assert original["classification_validation_source"] == toxscan_validation.SOURCE

    candidates = {c.example_id: c.key for c in screen_panel.candidates_for(tmp_path)}
    args = {"directory": str(tmp_path), "methods": [method]}
    picked, = mcp_api.call_tool("run_selected_screens", {**args, "candidate_id": candidates[candidate_id]})["results"]
    assert picked["status"] == "complete"
    assert picked["measurement"]["candidate_id_sha256"] == hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
    for field in ("candidate_id", "source_sha256", "structure_sha256", "result_sha256", "results", "similarity_score"):
        assert picked["measurement"][field] == original[field]
    other, = mcp_api.call_tool("run_selected_screens", {**args, "candidate_id": candidates["other"]})["results"]
    assert other["status"] == "inapplicable" and "measurement" not in other

    record = screen.screen_generation(tmp_path, binder_kind="ligand", target_id="", screen_methods=[method])
    assert record["method_selection"]["prepared_measurements"][method] == original
    stored, = mcp_api.call_tool("read_verdict", {"directory": str(tmp_path), "include_record": True})["runs"]
    for verdict in (stored["verdict"], stored["verdict"]["record"]):
        assert verdict["method_selection"]["prepared_measurements"][method] == picked["measurement"]
    old_payload = {k: original[k] for k in (
        "candidate_id", "source_sha256", "structure_sha256", "applicability_domain", "similarity_score", "results",
    )}
    older = {k: v for k, v in original.items()
             if not k.startswith("classification_") and k != "identity_check_version"}
    old_bytes = json.dumps(old_payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    older["result_sha256"] = hashlib.sha256(old_bytes).hexdigest()
    historical = mcp_api._toxscan_view(older)
    assert historical["identity_check_version"] == 1 and historical["classification_validation"] == validation
    assert any("Re-import the original export" in line for line in screen_board._toxscan_lines(older))
    choices = screen_selection.plan(tmp_path, "ligand", example=candidate_id, available={method: True})
    historical_page = ui._scientific_screen_panel(choices, (method,), [{**picked, "measurement": historical}],
                                                  directory=tmp_path)
    assert "Previous ToxScan import (re-import" in historical_page
    board = screen_board._summary(record, "export")
    assert candidate_id in board and "1 of 26" in board and "Eye irritation" in board
    assert "inside its applicability domain" in board and "0.78" in board
    assert "Mean AUROC was 0.982" in board and "95.5%" in board
    assert "standard deviations for all 26" in board
    detail = ui._scientific_result_detail(picked)
    assert "Validation for reported ToxScan endpoints" in detail and toxscan_validation.SOURCE in detail
    validator = jsonschema.Draft202012Validator(server._PREPARED_MEASUREMENT_SCHEMAS[method])
    validator.validate(picked["measurement"])
    validator.validate(historical)
    assert not validator.is_valid({**picked["measurement"], "identity_check_version": None})
    assert not validator.is_valid({**picked["measurement"], "results": {}})
    assert not validator.is_valid({**picked["measurement"], "classification_validation": {}})
    bad = {**validation, "Eye irritation": {**validation["Eye irritation"], "auroc_sd": 1.2}}
    assert not validator.is_valid({**picked["measurement"], "classification_validation": bad})

    # Readback preserves the prior receipt, while both the browser and a candidate-scoped rerun use current inputs.
    write_export(tmp_path, candidate_id, {"Eye irritation": "NO", "AD Analysis (AD: Applicability Domain)": "In AD",
                                          "Similarity Score in AD analysis": "0.00"})
    available = {method: True}
    choices = screen_selection.plan(tmp_path, "ligand", example=candidate_id, available=available)
    page = ui._scientific_screen_panel(choices, (method,), [picked], directory=tmp_path)
    assert "Previous ToxScan import" in page
    rerun, = mcp_api.call_tool("run_selected_screens", {**args, "candidate_id": candidates[candidate_id]})["results"]
    assert rerun["status"] == "complete" and rerun["measurement"]["results"]["classifications"]["Eye irritation"] == "NO"
    assert rerun["measurement"]["source_sha256"] != original["source_sha256"]
    assert "Previous ToxScan import" not in ui._scientific_screen_panel(choices, (method,), [rerun], directory=tmp_path)
    assert "disagree" in " ".join(screen_board._toxscan_lines(toxscan.screen_run(tmp_path, candidate_id)))
    stored, = mcp_api.call_tool("read_verdict", {"directory": str(tmp_path)})["runs"]
    assert stored["verdict"]["method_selection"]["prepared_measurements"][method] == picked["measurement"]


def test_standalone_blockers_and_malformed_saved_export(tmp_path, capsys):
    method = toxscan.METHOD_ID
    assert screen_extras.main([method, str(tmp_path)]) == 2
    missing = json.loads(capsys.readouterr().out)
    assert missing["blocker"] == "toxscan_manifest_missing"
    assert any(toxscan.MANIFEST in line for line in screen_board._toxscan_lines(missing))
    forged = {**missing, "status": "complete", "source_sha256": "/private/file", "arbitrary": "private"}
    projected = mcp_api._verdict({"method_selection": {"prepared_measurements": {method: forged}}}, include_record=True)
    rejected = screen_panel._redact_fresh_assay(method, forged)
    assert rejected["blocker"] == "toxscan_invalid_report" and rejected["results"] == {} and "arbitrary" not in rejected
    assert rejected["classification_validation"] == {}
    jsonschema.Draft202012Validator(server._PREPARED_MEASUREMENT_SCHEMAS[method]).validate(rejected)
    for shown in (projected, projected["record"]):
        assert shown["method_selection"]["prepared_measurements"][method] == rejected
    assert any("failed validation" in line for line in screen_board._toxscan_lines(forged))


@pytest.mark.parametrize(("sensitive_id", "smiles"), [("ACDEFGHIKLMNP", "CCO"),
                                                       ("tetraethylenepentamine", "NCCNCCNCCNCCN")])
def test_private_candidate_name_is_withheld_without_losing_model_results(tmp_path, sensitive_id, smiles):
    pytest.importorskip("rdkit")
    write_export(tmp_path, sensitive_id, {"SMILES": smiles}, smiles=smiles)
    original = toxscan.screen_run(tmp_path, sensitive_id)
    assert original["status"] == "complete"
    public = screen_panel._redact_fresh_assay(toxscan.METHOD_ID, original)
    assert public["candidate_id_withheld"] and "candidate_id" not in public
    candidate_digest = hashlib.sha256(sensitive_id.encode("utf-8")).hexdigest()
    assert public["candidate_id_sha256"] == candidate_digest
    assert public["results"] == original["results"] and public["result_sha256"] == original["result_sha256"]
    saved = mcp_api._verdict({"method_selection": {"prepared_measurements": {toxscan.METHOD_ID: original}}},
                             include_record=True)
    for surface in (saved, saved["record"]):
        assert surface["method_selection"]["prepared_measurements"][toxscan.METHOD_ID] == public
    board = " ".join(screen_board._toxscan_lines(original))
    assert sensitive_id not in board and candidate_digest[:12] in board
    jsonschema.Draft202012Validator(server._PREPARED_MEASUREMENT_SCHEMAS[toxscan.METHOD_ID]).validate(public)


def test_board_reports_coverage_and_keeps_unknown_domain_separate_from_similarity(tmp_path):
    pytest.importorskip("rdkit")
    write_export(tmp_path, "lead", {
        **dict.fromkeys(toxscan.CLASSIFICATIONS, ""), "Eye irritation": "YES", "Eye corrosion": "NO",
        **dict.fromkeys(toxscan.REGRESSIONS, ""), "AD Analysis (AD: Applicability Domain)": "",
        "Similarity Score in AD analysis": "0.87",
    })
    report = toxscan.screen_run(tmp_path, "lead")
    assert report["status"] == "complete"
    assert set(report["classification_validation"]) == {"Eye irritation", "Eye corrosion"}
    lines = " ".join(screen_board._toxscan_lines(report))
    assert "1 of 2 reported" in lines and "24 of 26 classification labels blank" in lines
    assert "domain unreported" in lines and "ten closest training structures was 0.87" in lines
    assert "All five numerical endpoint fields were blank" in lines
    validator = jsonschema.Draft202012Validator(server._PREPARED_MEASUREMENT_SCHEMAS[toxscan.METHOD_ID])
    public = screen_panel._redact_fresh_assay(toxscan.METHOD_ID, report)
    validator.validate(public)
    uncalled = {**public["classification_validation"], "CYP2C9": public["classification_validation"]["Eye irritation"]}
    assert not validator.is_valid({**public, "classification_validation": uncalled})
    assert not validator.is_valid({**public, "similarity_score": 2.5})
    empty = {**public["results"], "classifications": dict.fromkeys(toxscan.CLASSIFICATIONS)}
    assert not validator.is_valid({**public, "results": empty})


def test_numeric_only_export_has_no_classification_validation(tmp_path):
    pytest.importorskip("rdkit")
    write_export(tmp_path, "lead", {**dict.fromkeys(toxscan.CLASSIFICATIONS, ""),
                                   **dict.fromkeys(toxscan.REGRESSIONS, ""), "Rat oral LD50": "2.5mg/kg"})
    report = toxscan.screen_run(tmp_path, "lead")
    assert report["status"] == "complete" and report["classification_validation"] == {}
    public = screen_panel._redact_fresh_assay(toxscan.METHOD_ID, report)
    jsonschema.Draft202012Validator(server._PREPARED_MEASUREMENT_SCHEMAS[toxscan.METHOD_ID]).validate(public)
    assert not screen_board.toxscan_validation_lines(public)
