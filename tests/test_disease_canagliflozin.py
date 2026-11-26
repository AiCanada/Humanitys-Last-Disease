"""The fixed SBML reference requires an explicit nomination, authenticated input and dimensioned outputs."""

import hashlib
import json
import subprocess

import pytest

from rfproteina.disease import canagliflozin as screen
from rfproteina.disease import canagliflozin_runtime as runtime
from rfproteina.disease import canagliflozin_worker as worker
from scripts import install_canagliflozin as installer


def nominate(directory):
    (directory / screen.MANIFEST).write_text(json.dumps({"version": 1, "scenario": "oral_reference"}), encoding="utf-8")


@pytest.mark.parametrize("text", [
    '{"version":true,"scenario":"oral_reference"}', '{"version":2,"scenario":"oral_reference"}',
    '{"version":1,"scenario":"oral_reference","dose_mg":50}', '{"version":1,"scenario":"other"}',
    '{"version":1,"scenario":"other","scenario":"oral_reference"}', '{}', '[]', 'invalid', ' ' * 4097,
])
def test_manifest_is_a_single_known_nomination(tmp_path, text):
    (tmp_path / screen.MANIFEST).write_text(text, encoding="utf-8")
    report = screen.screen_run(tmp_path)
    assert report["status"] == "untested"
    assert report["blocker"] == "canagliflozin_invalid_manifest"
    assert report["measurements"] == report["profile"] == {}
    assert screen.report_view(report) == report


def test_refuses_candidate_scope_or_missing_nomination(tmp_path, monkeypatch):
    def unexpected_run():
        pytest.fail("unprepared control started a solver")
    monkeypatch.setattr(runtime, "simulate", unexpected_run)
    assert screen.screen_run(tmp_path)["blocker"] == "canagliflozin_manifest_missing"
    nominate(tmp_path)
    assert screen.prepare(tmp_path) == screen.SCENARIO
    assert screen.screen_run(tmp_path, candidate="a")["blocker"] == "canagliflozin_requires_run_scope"


def test_model_corruption_missing_solver_and_oversize_install_input(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "install_directory", lambda: tmp_path / "assets")
    nominate(tmp_path)
    assert not screen.installed()
    assert screen.screen_run(tmp_path)["blocker"] == "canagliflozin_model_missing"
    asset = runtime.install_directory() / "model" / runtime.MODEL_FILENAME
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"x" * runtime.MODEL_SIZE)
    assert screen.screen_run(tmp_path)["blocker"] == "canagliflozin_model_integrity"
    monkeypatch.setattr(runtime, "MODEL_SHA256", hashlib.sha256(asset.read_bytes()).hexdigest())
    assert not screen.installed()
    assert screen.screen_run(tmp_path)["blocker"] == "canagliflozin_solver_unavailable"
    asset.write_bytes(b"x" * (runtime.MODEL_SIZE + 1))
    with pytest.raises(runtime.CanagliflozinRuntimeError, match="canagliflozin_model_integrity"):
        runtime.model_bytes(asset)


def test_subprocess_failure_is_attributed_and_has_no_results(tmp_path, monkeypatch):
    python = runtime.interpreter(tmp_path)
    python.parent.mkdir(parents=True)
    python.touch()
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("canagliflozin", 90)
    monkeypatch.setattr(runtime.subprocess, "run", timeout)
    with pytest.raises(runtime.CanagliflozinRuntimeError, match="canagliflozin_timeout"):
        runtime.invoke(tmp_path, b"model")


@pytest.fixture
def fields():
    measurements = {key: {"value": 2, "unit": unit} for key, unit in worker.UNITS.items()}
    measurements["venous_parent_auc_0_24h"]["value"] = 3
    measurements["venous_parent_tmax_0_24h"]["value"] = 1
    measurements["urine_total_168h"]["value"] = 5
    profile = {"time_unit": "h", "concentration_unit": "umol/L", "time": [0, 1, 2, 4, 8, 12, 24]}
    for compartment in worker.profiles.COMPARTMENTS:
        profile[compartment] = [0, 2, 1.5, 1, 0.8, 0.5, 0.1]
    return {"measurements": measurements, "profile": profile}


@pytest.mark.parametrize(("metric", "value"), [
    ("urine_m5_48h", float("nan")), ("urine_m7_48h", True), ("venous_parent_tmax_0_24h", 0.888),
    ("venous_parent_auc_0_24h", 500), ("urine_total_168h", 1), ("feces_total_168h", 500),
    ("venous_parent_cmax_0_24h", 10**400),
])
def test_invalid_measurement_discards_entire_reference(tmp_path, fields, monkeypatch, metric, value):
    nominate(tmp_path)
    monkeypatch.setattr(runtime, "simulate", lambda: fields)
    report = screen.screen_run(tmp_path)
    assert report["status"] == "complete"
    report["measurements"][metric]["value"] = value
    invalid = screen.report_view(report)
    assert invalid["blocker"] == "canagliflozin_invalid_report"
    assert invalid["measurements"] == invalid["profile"] == {}


def test_saved_report_requires_correct_release_and_units(tmp_path, fields, monkeypatch):
    nominate(tmp_path)
    monkeypatch.setattr(runtime, "simulate", lambda: fields)
    report = screen.screen_run(tmp_path)
    assert screen.report_view({**report, "unknown": "ignored"}) == report
    for key in ("model_revision", "model_sha256", "solver_version", "scope", "scenario"):
        assert screen.report_view({**report, key: "other"})["blocker"] == "canagliflozin_invalid_report"
    report["measurements"]["urine_total_168h"]["unit"] = "umol/L"
    assert screen.report_view(report)["blocker"] == "canagliflozin_invalid_report"


def test_recovery_comparison_is_derived_from_mass_and_upgrades_saved_reports(tmp_path, fields, monkeypatch):
    nominate(tmp_path)
    dose_umol = 188_000 / 444.518
    fields["measurements"]["urine_total_168h"]["value"] = dose_umol * 0.40
    fields["measurements"]["feces_total_168h"]["value"] = dose_umol * 0.50
    monkeypatch.setattr(runtime, "simulate", lambda: fields)
    report = screen.screen_run(tmp_path)
    legacy = report.copy()
    comparison = legacy.pop("recovery_168h")
    assert screen.report_view(legacy) == report
    assert comparison["study_n"] == 6 and "204042Orig1s000ClinPharmR.pdf" in comparison["study_source"]
    assert comparison["urine"] == pytest.approx({
        "simulated_percent_dose": 40, "study_mean_percent_dose": 32.5, "difference_percentage_points": 7.5,
    })
    assert comparison["feces"] == pytest.approx({
        "simulated_percent_dose": 50, "study_mean_percent_dose": 60.4, "difference_percentage_points": -10.4,
    })
    comparison["urine"]["simulated_percent_dose"] = 4
    refused = screen.report_view(report)
    assert refused["blocker"] == "canagliflozin_invalid_report" and refused["recovery_168h"] == {}


def test_bounded_offline_wheel_install_rejects_changed_or_missing_bytes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    package = "libroadrunner"
    expected = b"official"
    wheel = installer.Wheel("tag", len(expected), hashlib.sha256(expected).hexdigest(), "asset")
    monkeypatch.setattr(installer, "WHEELS", {installer.sys.platform: {package: wheel}})
    filename = f"{package}-{worker.SOLVER_VERSION}-cp312-cp312-tag.whl"
    for prefix, raw in (("corrupt", b"tampered"), ("large", expected * 2)):
        temp = tmp_path / prefix
        temp.mkdir()
        (source / filename).write_bytes(raw)
        with pytest.raises(installer.InstallError):
            installer._stage_wheels(temp, source)
        assert not (temp / "wheels" / filename).exists()
    temp = tmp_path / "verified"
    temp.mkdir()
    (source / filename).write_bytes(expected)
    out = installer._stage_wheels(temp, source)
    assert (out / filename).read_bytes() == expected
    assert wheel.sha256 in (temp / "requirements.txt").read_text()


def test_real_mamidi_reference_simulation(tmp_path):
    if not runtime.install_directory().exists():
        pytest.skip("optional canagliflozin model and CPU solver are not installed")
    nominate(tmp_path)
    report = screen.screen_run(tmp_path)
    assert report["status"] == "complete", report.get("blocker")
    assert screen.report_view(report) == report
    # Independent run of this release: set the oral dose in the XML before compiling,
    # retain native CVODE tolerances, and use one-minute output on the 170 h trajectory.
    expected = {
        "gut_lumen_parent_cmax_0_24h": 189.442951, "liver_tissue_parent_cmax_0_24h": 4.190459,
        "kidney_plasma_parent_cmax_0_24h": 3.175993, "venous_parent_cmax_0_24h": 3.282337,
        "venous_m5_cmax_0_24h": 1.246511, "venous_m7_cmax_0_24h": 1.860037,
        "venous_parent_auc_0_24h": 24.510279, "venous_parent_tmax_0_24h": 83 / 60,
        "urine_m5_48h": 35.314518, "urine_m7_48h": 74.118813, "urine_total_168h": 128.427371,
        "feces_total_168h": 243.521328,
    }
    assert {name: item["value"] for name, item in report["measurements"].items()} == pytest.approx(expected, rel=2e-5)
    assert report["profile"]["venous_parent"][1] == pytest.approx(3.13906, rel=2e-5)
    assert report["recovery_168h"]["urine"]["simulated_percent_dose"] == pytest.approx(30.3661, abs=0.0001)
    assert report["recovery_168h"]["feces"]["difference_percentage_points"] == pytest.approx(-2.8204, abs=0.0001)
