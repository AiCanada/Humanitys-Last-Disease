import json

import jsonschema
import pytest

from rfproteina.disease import ligand_regressor, mcp_api, screen_performance, screen_selection
from tests.helpers.disease import load_mcp_server


def _score(axis, backend, error=0.63):
    return {"axis": axis, "backend": backend, "rmse_log10": error, "test_rows": 72, "replicates": 2}


def _bundle(tmp_path, monkeypatch, backend, scores):
    root = tmp_path / backend
    root.mkdir()
    variable = "RFD4_LIGAND_REGRESSOR" if backend == "forest" else "RFD4_CHEMPROP_REGRESSOR"
    monkeypatch.setenv(variable, str(root))
    manifest = {
        "version": 2, "backend": backend,
        "method": ligand_regressor.METHOD_FOREST if backend == "forest" else ligand_regressor.METHOD_CHEMPROP,
        "morgan": {
            "radius": ligand_regressor.MORGAN_RADIUS, "bits": ligand_regressor.MORGAN_BITS,
            "counts": ligand_regressor.MORGAN_COUNTS,
        },
        "axes": {},
    }
    for axis, score in scores.items():
        # A deliberately unreadable estimator is enough for manifest-only inspection.
        (root / axis).write_bytes(b"invalid estimator")
        manifest["axes"][axis] = {"estimator": axis, "score": score}
    (root / "ligand_regressor.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


@pytest.mark.parametrize("backend", ["forest", "chemprop"])
def test_fresh_matching_bundle_reports_separate_scaffold_errors(tmp_path, monkeypatch, backend):
    scores = {"bcrp": _score("bcrp", backend), "bsep": {**_score("bsep", backend, 1.23), "replicates": 1}}
    root = _bundle(tmp_path, monkeypatch, backend, scores)
    key = "bindingdb_" + backend
    info = screen_performance.for_method(key, "compute", installed=True)
    assert info["status"] == "local_benchmark"
    assert "local model manifest" in info["summary"] and "log10(nM)" in info["summary"]
    assert "BCRP 0.63 (72 test rows, 2 scaffold splits)" in info["summary"]
    assert "BSEP 1.23 (72 test rows, 1 scaffold split)" in info["summary"]
    assert "No verified estimate for NaV1.5" in info["summary"] and str(root) not in str(info)
    assert "same compound" in info["interpretation"] and "tenfold" in info["interpretation"]
    # Replacing or removing a bundle at its same location must not leave a stale score.
    (root / "bcrp").unlink()
    assert screen_performance.for_method(key, "compute")["status"] == "not_verified"


@pytest.mark.parametrize("change", [
    {"axis": "bcrp"}, {"backend": "chemprop"}, {"rmse_log10": True}, {"rmse_log10": float("nan")},
    {"rmse_log10": float("inf")}, {"rmse_log10": -0.2}, {"rmse_log10": "0.6"}, {"rmse_log10": 10**500},
    {"test_rows": True}, {"test_rows": 0}, {"replicates": 1.5}, {"replicates": 73},
])
def test_invalid_axis_score_does_not_hide_valid_sibling_or_claim_coverage(tmp_path, monkeypatch, change):
    scores = {"nav1_5": {**_score("nav1_5", "forest"), **change}, "bsep": _score("bsep", "forest", 0.42)}
    _bundle(tmp_path, monkeypatch, "forest", scores)
    summary = screen_performance.for_method("bindingdb_forest", "compute")["summary"]
    assert "BSEP 0.42" in summary and "No verified estimate for NaV1.5, BCRP" in summary


def test_mismatched_backend_or_unscored_manifest_has_no_local_benchmark(tmp_path, monkeypatch):
    root = _bundle(tmp_path, monkeypatch, "forest", {"bcrp": _score("bcrp", "forest")})
    monkeypatch.setenv("RFD4_CHEMPROP_REGRESSOR", str(root))
    assert screen_performance.for_method("bindingdb_chemprop", "compute")["status"] == "not_verified"
    manifest = root / "ligand_regressor.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["axes"]["bcrp"]["score"] = None
    manifest.write_text(json.dumps(data), encoding="utf-8")
    assert screen_performance.for_method("bindingdb_forest", "compute")["status"] == "not_verified"


def test_stored_readings_and_uninstalled_producers_never_probe_local_bundle(monkeypatch):
    monkeypatch.setattr(ligand_regressor, "regressor_manifest", lambda *a: pytest.fail("unexpected bundle probe"))
    monkeypatch.setattr(ligand_regressor, "default_model_dir", lambda *a: pytest.fail("unexpected bundle lookup"))
    for operation, installed in (("recorded_evidence", True), ("compute", False), ("native_result_reader", True)):
        info = screen_performance.for_method("bindingdb_forest", operation, installed=installed)
        assert info["status"] == "not_verified"


def test_selected_results_capture_run_start_evidence_even_if_a_model_disappears(tmp_path, monkeypatch):
    root = _bundle(tmp_path, monkeypatch, "forest", {"bsep": _score("bsep", "forest", 0.42)})
    (tmp_path / "generation.json").write_text('{"binder_kind":"ligand"}')
    (tmp_path / "ligand_candidates.tsv").write_text("candidate_id\tsmiles\ncompound\tCCO\n")
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    monkeypatch.setattr(screen_selection, "installations", lambda keys: dict.fromkeys(keys, True))
    expected = screen_performance.for_method("bindingdb_forest", "compute")

    def remove_model_on_first_screen(line):
        if line.startswith("Running RDKit"):
            (root / "bsep").unlink()

    request = {
        "directory": str(tmp_path),
        "methods": ["ligand_alerts", "bindingdb_forest", "molprobity_geometry"],
        "include_performance": True,
    }
    reply = mcp_api.call_tool("run_selected_screens", request, on_line=remove_model_on_first_screen)
    rows = {row["id"]: row for row in reply["results"]}
    assert set(rows) == set(request["methods"]) and all(row["performance"]["basis"] for row in rows.values())
    assert rows["ligand_alerts"]["status"] == "reported"
    assert rows["molprobity_geometry"]["status"] == "inapplicable"
    assert rows["bindingdb_forest"]["status"] == "untested"
    assert rows["bindingdb_forest"]["performance"] == expected and "BSEP 0.42" in expected["summary"]
    assert screen_performance.for_method("bindingdb_forest", "compute")["status"] == "not_verified"
    tool = next(t for t in load_mcp_server()._TOOLS if t["name"] == "run_selected_screens")
    jsonschema.validate(request, tool["inputSchema"])
    jsonschema.validate(reply, tool["outputSchema"])
    del rows["bindingdb_forest"]["performance"]["basis"]
    with pytest.raises(jsonschema.ValidationError, match="basis"):
        jsonschema.validate(reply, tool["outputSchema"])
