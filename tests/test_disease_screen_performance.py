"""Scientific evidence shown during selection and persisted with the decision."""

import csv
from importlib import metadata
from urllib.parse import urlsplit

import jsonschema
import pytest

from rfproteina.disease import screen_performance, screen_selection
from tests.helpers.disease import load_mcp_server


def test_every_registered_method_explains_its_evidence_and_experimental_use():
    assert set(screen_selection.METHOD_IDS) == set(screen_performance._METHODS)
    for method in screen_selection.METHODS:
        choice = screen_selection.Choice(method, True, True, False, "Manual selection")
        evidence = choice.as_record()["performance"]
        assert evidence == choice.performance
        assert evidence["basis"].strip() and evidence["basis"] not in {evidence["status"], "Unspecified evidence basis"}
        assert evidence["summary"] and evidence["interpretation"] and evidence["summary"] != evidence["interpretation"]
        if evidence["status"] == "published_benchmark":
            assert urlsplit(evidence["source"]).scheme == "https"
        evidence["summary"] = "client edit"
        assert choice.performance["summary"] != "client edit"


def test_fallback_for_a_new_method_and_attribution_of_saved_benchmarks():
    fresh = screen_performance.for_method("bayesherg", "compute")
    saved = screen_performance.for_method("bayesherg", "recorded_evidence")
    assert "producer version" in saved["summary"] and saved["summary"].startswith(fresh["summary"])
    assert saved["interpretation"] == fresh["interpretation"]
    assert saved["basis"] == fresh["basis"]
    new = screen_performance.for_method("new_method", "compute")
    assert new["status"] == "not_verified" and "known results" in new["interpretation"]
    legacy = screen_performance.Evidence("source_evidence", "Prior result.", "Consult the original assay.").as_record()
    assert legacy["basis"] == "Unspecified evidence basis"


def test_mcp_requires_an_interpretation_for_each_supported_kind_of_evidence():
    tool = next(t for t in load_mcp_server()._TOOLS if t["name"] == "plan_screening")
    row_schema = tool["outputSchema"]["properties"]["methods"]["items"]
    record = screen_selection.Choice(screen_selection.METHODS[0], True, True, True, "Nominated").as_record()
    for status in ("published_benchmark", "local_benchmark", "reference_dependent", "source_evidence", "not_verified"):
        record["performance"]["status"] = status
        jsonschema.validate(record, row_schema)
    del record["performance"]["interpretation"]
    with pytest.raises(jsonschema.ValidationError, match="interpretation"):
        jsonschema.validate(record, row_schema)


def test_measurement_repeatability_and_model_error_have_distinct_meanings():
    cellprofiler = screen_performance.for_method("cellprofiler", "compute")
    assert cellprofiler["status"] == "reference_dependent"
    assert "controls" in cellprofiler["summary"] and "pathologist" in cellprofiler["interpretation"]
    caco = screen_performance.for_method("ligand_caco2", "compute")
    assert all(x in caco["summary"] for x in ("0.40", "0.19", "retrained"))
    assert "bidirectional" in caco["interpretation"] and "efflux rankings weakly" in caco["interpretation"]
    pka = screen_performance.for_method("qupkake", "compute")
    assert "RMSE" in pka["summary"] and "assay pH" in pka["interpretation"]


def test_benchmarks_distinguish_study_design_and_classifier_direction():
    def info(key):
        return screen_performance.for_method(key, "compute")

    assert info("protein_sol")["basis"] == info("prodigy_affinity")["basis"] == "Development-set fit"
    assert info("spyrmsd")["basis"] == "Agreement between implementations"
    assert info("ligand_caco2")["basis"] == "External test of an earlier model"
    assert "macrocycles" in info("ligand_caco2")["summary"]
    dimorphite = info("dimorphite_dl")
    assert all(x in dimorphite["summary"] for x in ("v1.0", "58.8%", "40.2%", "0.9%", "v2.1.0"))
    assert "population fractions" in dimorphite["interpretation"]
    assert "25 of 30 blockers" in info("bayesherg")["summary"]
    assert "12 of 14 nonblockers" in info("bayesherg")["summary"]
    psr = info("delphi_psr")
    assert psr["basis"] == "Internal cross-validation"
    assert "Higher scores favor passing" in psr["interpretation"] and "lower polyreactivity" in psr["interpretation"]
    for key in ("ligand_ames", "ligand_dili"):
        assert "holdout" in info(key)["basis"] and "average precision" in info(key)["summary"]


def test_admet_display_matches_the_installed_release_when_available():
    try:
        dist = metadata.distribution("admet-ai")
    except metadata.PackageNotFoundError:
        pytest.skip("The optional ADMET-AI package is not installed")
    if dist.version != "2.0.1":
        pytest.skip("This comparison applies to the 2.0.1 release")
    with dist.locate_file("admet_ai/resources/data/admet.csv").open(encoding="utf-8", newline="") as handle:
        table = {row["id"]: row for row in csv.DictReader(handle)}
    for key, endpoint, metrics in (
        ("ligand_solubility", "Solubility_AqSolDB", (("R^2", 3), ("MAE", 3))),
        ("ligand_microsomal_clearance", "Clearance_Microsome_AZ", (("R^2", 3), ("MAE", 2))),
        ("ligand_ames", "AMES", (("AUROC", 3), ("AUPRC", 3))),
        ("ligand_dili", "DILI", (("AUROC", 3), ("AUPRC", 3))),
    ):
        evidence = screen_performance.for_method(key, "compute")
        assert evidence["status"] == "published_benchmark" and "v_2.0.1" in evidence["source"]
        for column, decimals in metrics:
            assert f"{float(table[endpoint][column]):.{decimals}f}" in evidence["summary"]


@pytest.mark.parametrize(("key", "status", "scope", "application"), [
    ("badapple2", "source_evidence", "906", "counterscreen"),
    ("metfrag", "published_benchmark", "30 of 473", "standard"),
    ("fastsolv", "published_benchmark", "before forming the four-model ensemble", "solid form"),
    ("jazzy", "published_benchmark", "9.19 kJ/mol", "gas-to-water"),
    ("crest", "reference_dependent", "reduced search", "same molecular form"),
    ("espsim", "published_benchmark", "charge", "prealigned"),
    ("propka", "published_benchmark", "protein residue", "titration"),
    ("p2rank", "published_benchmark", "within 4 Å", "receptor sites"),
    ("openantigens", "source_evidence", "5,328", "control staining"),
])
def test_new_screens_keep_benchmark_scope_and_experimental_application(key, status, scope, application):
    method = next(method for method in screen_selection.METHODS if method.key == key)
    for installed in (False, True):
        record = screen_selection.Choice(method, installed, False, False, "No nomination").as_record()
        evidence = record["performance"]
        assert evidence["status"] == status
        assert scope in f"{evidence['summary']} {evidence['basis']}"
        assert application in evidence["interpretation"]
        assert urlsplit(evidence["source"]).scheme == "https"
