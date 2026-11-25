"""Keep the three published endpoints on their respective cohorts and scales."""

import json

import jsonschema
import pytest

from rfproteina.disease import mcp_api
from rfproteina.disease.immuscope_study import published_study
from scripts import screen_extras
from tests.helpers.disease import load_mcp_server


def test_cli_and_mcp_report_the_same_study(capsys):
    assert screen_extras.main(["immuscope-study"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["evidence_type"] == "published_benchmark"
    response = mcp_api.call_tool("declare_corpora", {})
    assert record == response["reference_studies"]["immuscope"]
    tool = next(tool for tool in load_mcp_server()._TOOLS if tool["name"] == "declare_corpora")
    jsonschema.validate(response, tool["outputSchema"])
    endpoints = record["endpoints"]
    assert set(endpoints) == {"ImmuScope-EL", "ImmuScope-CD4", "ImmuScope-IM"}
    assert endpoints["ImmuScope-EL"]["metrics"] == {
        "area_under_precision_recall_curve": 0.856, "positive_predictive_value": 0.796,
    }
    cd4 = endpoints["ImmuScope-CD4"]
    assert cd4["metrics"] == {"mean_per_triplet_roc_auc": 0.825}
    assert (cd4["paired_comparisons_in_figure"], cd4["distinct_triplets_in_methods"]) == (824, 842)
    immunogenicity = endpoints["ImmuScope-IM"]
    assert immunogenicity["test_records"] == 17897
    assert immunogenicity["metrics"] == {"roc_auc": 0.909, "roc_auc_95_percent_ci": [0.901, 0.918]}


def test_reference_has_no_candidate_input_or_shared_mutable_state():
    with pytest.raises(SystemExit) as exc:
        screen_extras.main(["immuscope-study", "--candidate", "unknown"])
    assert exc.value.code == 2
    study = published_study()
    study["endpoints"].pop("ImmuScope-EL")
    assert "ImmuScope-EL" in published_study()["endpoints"]
