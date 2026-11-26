"""A multi-organ reference result stays attributable across the picker, CLI and saved verdict."""

import json

import jsonschema
import pytest

from rfproteina.disease import (
    canagliflozin,
    mcp_api,
    screen,
    screen_assays,
    screen_board,
    screen_panel,
    screen_selection,
)
from scripts import humanitys_last_disease_mcp as server
from scripts import screen_extras


def test_reference_control_through_cli_picker_and_saved_report(tmp_path, monkeypatch, capsys):
    # These synthetic measurements exercise the interfaces independently of installing the optional solver.
    endpoints = {
        "gut_lumen_parent_cmax_0_24h": (8, "umol/L"), "liver_tissue_parent_cmax_0_24h": (7, "umol/L"),
        "kidney_plasma_parent_cmax_0_24h": (6, "umol/L"), "venous_parent_cmax_0_24h": (5, "umol/L"),
        "venous_m5_cmax_0_24h": (4, "umol/L"), "venous_m7_cmax_0_24h": (3, "umol/L"),
        "venous_parent_auc_0_24h": (30, "umol*h/L"), "venous_parent_tmax_0_24h": (2, "h"),
        "urine_m5_48h": (9, "umol"), "urine_m7_48h": (10, "umol"),
        "urine_total_168h": (20, "umol"), "feces_total_168h": (25, "umol"),
    }
    values = {key: {"value": value, "unit": unit} for key, (value, unit) in endpoints.items()}
    profile = {
        "time_unit": "h", "concentration_unit": "umol/L", "time": [0, 1, 2, 4, 8, 12, 24],
        "gut_lumen_parent": [0, 8, 4, 2, 1, 0.5, 0],
        "liver_tissue_parent": [0, 3, 7, 4, 2, 1, 0.3],
        "kidney_plasma_parent": [0, 3, 6, 3, 1, 0.5, 0.2],
        "venous_parent": [0, 3, 5, 2, 1, 0.5, 0.2],
    }
    monkeypatch.setattr(canagliflozin.runtime, "simulate", lambda: {"measurements": values, "profile": profile})
    monkeypatch.setattr(screen_selection, "installations", lambda keys: dict.fromkeys(keys, True))
    monkeypatch.setattr(screen, "_sample_measurements", lambda *args, **kwargs: ([], [], []))
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    (tmp_path / "generation.json").write_text('{"binder_kind":"ligand"}')
    (tmp_path / canagliflozin.MANIFEST).write_text('{"version":1,"scenario":"oral_reference"}')
    method = canagliflozin.METHOD_ID

    assert screen_extras.main([method, str(tmp_path)]) == 0
    reference = json.loads(capsys.readouterr().out)
    assert reference["status"] == "complete" and reference["measurements"] == values
    assert reference["scope"] == "reference_control"

    args = {"directory": str(tmp_path), "methods": [method]}
    live, = mcp_api.call_tool("run_selected_screens", args)["results"]
    assert live["status"] == "complete"
    assert live["measurement"]["measurements"] == values
    assert live["measurement"]["profile"] == profile
    comparison = live["measurement"]["recovery_168h"]
    assert comparison == reference["recovery_168h"]
    assert comparison["study_n"] == canagliflozin.RECOVERY_STUDY_N
    assert comparison["study_source"] == canagliflozin.RECOVERY_STUDY_SOURCE
    for route, expected_mean in (("urine", 32.5), ("feces", 60.4)):
        result = comparison[route]
        assert result["study_mean_percent_dose"] == expected_mean
        assert result["difference_percentage_points"] == pytest.approx(result["simulated_percent_dose"] - expected_mean)
    for field in ("model_sha256", "model_revision", "scope", "scenario", "dose_mg", "duration_h"):
        assert live["measurement"][field] == reference[field]
    (tmp_path / "ligand_candidates.tsv").write_text("candidate_id\tsmiles\ncandidate_1\tCCO\n")
    candidate, = screen_panel.candidates_for(tmp_path)
    scoped, = mcp_api.call_tool("run_selected_screens", {**args, "candidate_id": candidate.key})["results"]
    assert scoped["status"] == "inapplicable" and "measurement" not in scoped
    assert screen_assays.measure(tmp_path, method, "candidate_1")["blocker"] == "canagliflozin_requires_run_scope"

    record = screen.screen_generation(tmp_path, binder_kind="ligand", target_id="", screen_methods=[method])
    saved = json.loads((tmp_path / "screen.json").read_text())["method_selection"]["prepared_measurements"][method]
    assert saved == reference
    readback, = mcp_api.call_tool("read_verdict", {"directory": str(tmp_path), "include_record": True})["runs"]
    assert "error" not in readback
    projected = readback["verdict"]
    for result in (projected, projected["record"]):
        assert result["method_selection"]["prepared_measurements"][method] == live["measurement"]
    board = screen_board._summary(record, "reference")
    for phrase in ("188 mg", "170 hours", "8 umol/L in the gut lumen", "7 umol/L in liver tissue",
                   "6 umol/L in kidney plasma", "30 umol*h/L", "parent, M7 and M9", "Table 15", "6 participants",
                   f"{comparison['urine']['difference_percentage_points']:+.2f} percentage points for urine"):
        assert phrase in board

    schema = server._PREPARED_MEASUREMENT_SCHEMAS[method]
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(live["measurement"])
    assert not validator.is_valid({**saved, "profile": {}})
    assert not validator.is_valid({**saved, "recovery_168h": {**comparison, "study_n": 60}})
    old = {key: item for key, item in saved.items() if key != "recovery_168h"}
    before_comparison = {**record, "method_selection": {"prepared_measurements": {method: old}}}
    (tmp_path / "screen.json").write_text(json.dumps(before_comparison))
    restored, = mcp_api.call_tool("read_verdict", {"directory": str(tmp_path), "include_record": True})["runs"]
    for result in (restored["verdict"], restored["verdict"]["record"]):
        assert result["method_selection"]["prepared_measurements"][method]["recovery_168h"] == comparison
    assert not validator.is_valid({**saved, "profile": {**profile, "venous_parent": [0]}})
    assert not validator.is_valid({**saved, "measurements": {}})
    assert not validator.is_valid({**saved, "measurements": {**values, "urine_total_168h": {"value": 20, "unit": "h"}}})
    damaged = {**saved, "profile": {**profile, "time": list(reversed(profile["time"]))}}
    rejected = mcp_api._canagliflozin_view(damaged)
    assert rejected["blocker"] == "canagliflozin_invalid_report"
    assert rejected["measurements"] == rejected["profile"] == {}


def test_missing_nomination_and_invalid_saved_results_have_actionable_blockers(tmp_path, capsys):
    method = canagliflozin.METHOD_ID
    assert screen_extras.main([method, str(tmp_path)]) == 2
    missing = json.loads(capsys.readouterr().out)
    assert missing["status"] == "untested" and missing["blocker"] == "canagliflozin_manifest_missing"
    assert any(canagliflozin.MANIFEST in line for line in screen_board._canagliflozin_lines(missing))
    forged = {**missing, "status": "complete", "model_sha256": "bad", "private_path": "/private/model"}
    view = mcp_api._verdict({"method_selection": {"prepared_measurements": {method: forged}}}, include_record=True)
    fresh = screen_panel._redact_fresh_assay(method, forged)
    assert fresh["blocker"] == "canagliflozin_invalid_report" and fresh["measurements"] == fresh["profile"] == {}
    jsonschema.Draft202012Validator(server._PREPARED_MEASUREMENT_SCHEMAS[method]).validate(fresh)
    assert "private_path" not in fresh
    assert fresh["recovery_168h"] == {}
    for result in (view, view["record"]):
        assert result["method_selection"]["prepared_measurements"][method] == fresh
    board = screen_board._canagliflozin_lines(forged)
    assert any("report failed validation" in line for line in board)
    assert all("peak" not in line.lower() for line in board)


def test_only_a_nominated_ligand_control_counts_as_a_standalone_screening_run(tmp_path):
    manifest = tmp_path / canagliflozin.MANIFEST
    manifest.write_text('{"version":1,"scenario":"oral_reference"}')
    assert not screen_panel.nominated_ligand_run(tmp_path)
    metadata = tmp_path / "generation.json"
    metadata.write_text('{"binder_kind":"protein"}')
    assert not screen_panel.nominated_ligand_run(tmp_path)
    metadata.write_text('{"binder_kind":"ligand"}')
    manifest.write_text('{"version":1,"scenario":"another_compound"}')
    assert not screen_panel.nominated_ligand_run(tmp_path)


def test_directory_screen_discovers_a_nested_reference_with_an_explicit_binder_kind(tmp_path, monkeypatch):
    run = tmp_path / "control"
    run.mkdir()
    (run / canagliflozin.MANIFEST).write_text('{"version":1,"scenario":"oral_reference"}')
    other = tmp_path / "invalid"
    other.mkdir()
    (other / canagliflozin.MANIFEST).write_text('{"version":1,"scenario":"other"}')
    # This directory is a reference run by nomination, even without generated structures or generation.json.
    assert screen.generation_directories(tmp_path) == []
    assert screen.generation_directories(run, prepared_ligands=True, binder_kind="ligand") == [run]
    assert screen.generation_directories(tmp_path, prepared_ligands=True, binder_kind="protein") == []
    monkeypatch.setattr(screen_selection, "installations", lambda keys: dict.fromkeys(keys, True))
    monkeypatch.setattr(screen, "_sample_measurements", lambda *args, **kwargs: ([], [], []))
    monkeypatch.setenv("HUMANITYS_LAST_DISEASE_LOG", str(tmp_path / "screen-test.log"))
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))

    def unavailable():
        raise canagliflozin.runtime.CanagliflozinRuntimeError("canagliflozin_solver_unavailable")

    monkeypatch.setattr(canagliflozin.runtime, "simulate", unavailable)
    result = mcp_api.screen_directory({
        "directory": str(tmp_path), "binder_kind": "ligand", "screen_methods": [canagliflozin.METHOD_ID],
    })
    assert result["exit_code"] == 0 and result["complete"]
    entry, = result["runs"]
    assert entry["label"] == "control" and entry["written"]
    report = entry["verdict"]["method_selection"]["prepared_measurements"][canagliflozin.METHOD_ID]
    assert report["blocker"] == "canagliflozin_solver_unavailable" and report["scope"] == "reference_control"
    assert not (other / "screen.json").exists()


def test_hidden_reference_marks_the_sweep_partial_only_when_nominated(tmp_path):
    hidden = tmp_path / ".archive"
    hidden.mkdir()
    manifest = hidden / canagliflozin.MANIFEST
    manifest.write_text('{"version":1,"scenario":"invalid"}')
    assert screen.generation_directories(tmp_path, prepared_ligands=True, binder_kind="ligand") == []
    manifest.write_text('{"version":1,"scenario":"oral_reference"}')
    with pytest.raises(screen.ScreenWalkTruncatedError):
        screen.generation_directories(tmp_path, prepared_ligands=True, binder_kind="ligand")


def test_mixed_archive_preserves_the_limit_and_each_runs_declared_kind(tmp_path):
    reference = tmp_path / "a-reference"
    reference.mkdir()
    (reference / "generation.json").write_text('{"binder_kind":"ligand"}')
    (reference / canagliflozin.MANIFEST).write_text('{"version":1,"scenario":"oral_reference"}')
    protein = tmp_path / "b-protein"
    (protein / "inference_samples").mkdir(parents=True)
    (protein / "generation.json").write_text('{"binder_kind":"protein"}')
    assert screen.generation_directories(tmp_path) == [protein]
    assert screen.generation_directories(tmp_path, prepared_ligands=True) == [reference, protein]
    with pytest.raises(screen.ScreenWalkTruncatedError) as stopped:
        screen.generation_directories(tmp_path, prepared_ligands=True, limit=1)
    assert stopped.value.found == [reference]
