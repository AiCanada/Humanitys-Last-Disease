"""A local Ames prediction stays attached to the selected graph through screening and readback."""

import hashlib
import json

import jsonschema
import pytest

from rfproteina.disease import (
    corpus_provenance,
    mcp_api,
    screen,
    screen_board,
    screen_panel,
    screen_selection,
    toxscan,
    toxscan_ames,
    toxscan_ames_model,
    toxscan_inputs,
    ui,
)
from scripts import humanitys_last_disease_mcp as server
from scripts import screen_extras

METHOD = toxscan_ames.METHOD_ID
PRIVATE_ID = "private_lead_7"
SMILES = "O=C(O)c1ccccc1OC(C)=O"


@pytest.fixture
def local_fit(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    (tmp_path / "generation.json").write_text('{"binder_kind":"ligand"}')
    (tmp_path / toxscan_ames.TABLE).write_text(f"candidate_id\tsmiles\n{PRIVATE_ID}\t{SMILES}\n")
    validation = {
        # A count-consistent synthetic receipt isolates report/readback behavior from an installation.
        "n_train": 6247,
        "n_validation": 694,
        "n_test": 731,
        "test_positives": 614,
        "test_negatives": 117,
        "tp": 560,
        "tn": 75,
        "fp": 42,
        "fn": 54,
        "accuracy": 635 / 731,
        "selected_epoch": 7,
        "roc_auc": 0.812,
        "validation_roc_auc": 0.819,
        "test_fingerprint_seen_in_train": 65,
    }
    expected_graph = toxscan_inputs.canonical(SMILES)

    def predict(graph):
        assert graph == expected_graph
        return {
            "score": 0.5,
            "class_call": "YES",
            "model_sha256": "8" * 64,
            "archive_sha256": toxscan_ames.ARCHIVE,
            "validation": dict(validation),
            "graph_membership": "not_in_retained_splits",
            "fingerprint_seen_in_training": False,
        }

    monkeypatch.setattr(toxscan_ames_model, "predict", predict)
    monkeypatch.setattr(toxscan_ames_model, "benchmark", lambda: dict(validation))
    monkeypatch.setattr(toxscan_ames_model, "fingerprint_seen_in_training", lambda graph: False)
    monkeypatch.setattr(screen_selection, "installations", lambda keys=None: dict.fromkeys(keys or {METHOD}, True))
    monkeypatch.setattr(screen, "_sample_measurements", lambda *args, **kwargs: ([], [], []))
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    return validation


def test_offline_compute_plan_and_verdict_expose_only_digest_bound_evidence(tmp_path, local_fit, capsys):
    choices = screen_selection.plan(tmp_path, "ligand", example=PRIVATE_ID, available={METHOD: True})
    choice = next(item for item in choices if item.method.key == METHOD)
    assert choice.installed and choice.applicable and choice.recommended
    assert choice.method.operation == "compute" and "0.812" in json.dumps(choice.performance)
    (entry,) = screen_panel.candidates_for(tmp_path)
    selection = {"directory": str(tmp_path), "candidate_id": entry.key, "methods": [METHOD]}
    (picked,) = mcp_api.call_tool("run_selected_screens", selection)["results"]
    assert picked["status"] == "complete"
    measurement = picked["measurement"]
    assert measurement["scope"] == "candidate" and measurement["method"] == METHOD
    assert measurement["class_call"] == "YES" and measurement["score"] == 0.5
    assert measurement["validation"] == local_fit
    assert measurement["graph_membership"] == "not_in_retained_splits"
    assert measurement["candidate_id_sha256"] == hashlib.sha256(PRIVATE_ID.encode()).hexdigest()
    assert measurement["structure_sha256"] == hashlib.sha256(toxscan_inputs.canonical(SMILES).encode()).hexdigest()
    assert measurement["corpus_provenance"]["overlap"] == corpus_provenance.BENCHMARK_LABEL_CORPUS
    assert screen_extras.main(["toxscan-ames", str(tmp_path), "--candidate", PRIVATE_ID]) == 0
    assert json.loads(capsys.readouterr().out) == measurement

    record = screen.screen_generation(tmp_path, binder_kind="ligand", target_id="", screen_methods=[METHOD])
    assert record["method_selection"]["prepared_measurements"][METHOD] == measurement
    (stored,) = mcp_api.call_tool("read_verdict", {"directory": str(tmp_path), "include_record": True})["runs"]
    for verdict in (stored["verdict"], stored["verdict"]["record"]):
        assert verdict["method_selection"]["prepared_measurements"][METHOD] == measurement

    board = " ".join(screen_board.toxscan_ames_lines(measurement))
    detail = ui._scientific_result_detail(picked)
    for rendered in (board, detail):
        assert "predicted YES" in rendered and "0.5000" in rendered
        assert "560 of 614" in rendered and "75 of 117" in rendered
        assert "0.812" in rendered and "reverse-mutation" in rendered
    for value in (json.dumps(measurement), board, detail):
        assert PRIVATE_ID not in value and SMILES not in value and toxscan_inputs.canonical(SMILES) not in value
    validator = jsonschema.Draft202012Validator(server._PREPARED_MEASUREMENT_SCHEMAS[METHOD])
    validator.validate(measurement)
    assert not validator.is_valid({**measurement, "score": 1.1})
    assert not validator.is_valid({**measurement, "class_call": "unclassified"})


@pytest.mark.parametrize(
    "field,value",
    [
        ("class_call", "NO"),
        ("scope", "run"),
        ("structure_sha256", "1" * 64),
        ("candidate_id_sha256", "2" * 64),
        ("model_sha256", "3" * 64),
        ("graph_membership", "train"),
        ("result_sha256", "4" * 64),
    ],
)
def test_inconsistent_or_reattributed_result_cannot_be_projected(tmp_path, local_fit, field, value):
    report = toxscan_ames.screen_run(tmp_path)
    assert report["status"] == "complete"
    invalid = mcp_api._toxscan_ames_view({**report, field: value})
    assert invalid["status"] == "untested" and invalid["blocker"] == "toxscan_ames_invalid_report"
    assert "score" not in invalid and "validation" not in invalid


@pytest.mark.parametrize("author_call", ["NO", "YES", None])
def test_same_corpus_ames_call_is_selectable_without_defaulting_to_a_second_opinion(
    tmp_path, local_fit, monkeypatch, author_call
):
    (tmp_path / toxscan.MANIFEST).touch()
    author_report = {"results": {"classifications": {"Ames Mutagenicity": author_call}}}
    monkeypatch.setattr(toxscan, "screen_run", lambda *_: author_report)
    choices = screen_selection.plan(tmp_path, "ligand", example=PRIVATE_ID, available={METHOD: True})
    choice = next(item for item in choices if item.method.key == METHOD)
    assert choice.installed and choice.applicable
    assert choice.recommended is (author_call is None)
    if author_call is not None:
        assert "same study corpus" in choice.reason
    missing = toxscan_ames.screen_run(tmp_path, candidate="unlisted")
    assert missing["blocker"] == "toxscan_candidate_not_nominated" and "score" not in missing


def test_training_membership_is_visible_and_failed_weights_do_not_yield_a_score(tmp_path, local_fit, monkeypatch):
    monkeypatch.setattr(toxscan_ames_model, "fingerprint_seen_in_training", lambda graph: True)
    choices = screen_selection.plan(tmp_path, "ligand", example=PRIVATE_ID, available={METHOD: True})
    choice = next(item for item in choices if item.method.key == METHOD)
    assert choice.installed and choice.applicable and not choice.recommended
    assert METHOD not in screen_selection.select(choices, "recommended")
    assert METHOD in screen_selection.select(choices, "all_installed")
    assert screen_selection.select(choices, "explicit", [METHOD]) == (METHOD,)
    assert "fingerprint" in choice.reason.lower() and "train" in choice.reason.lower()
    predict = toxscan_ames_model.predict
    monkeypatch.setattr(
        toxscan_ames_model,
        "predict",
        lambda graph: {**predict(graph), "graph_membership": "train", "fingerprint_seen_in_training": True},
    )
    report = toxscan_ames.screen_run(tmp_path)
    assert report["status"] == "complete"
    board = " ".join(screen_board.toxscan_ames_lines(report)).lower()
    assert "train partition" in board and "training included this molecular fingerprint" in board

    def fail_to_load(_):
        raise toxscan_ames_model.AmesModelError("toxscan_ames_model_integrity")

    monkeypatch.setattr(toxscan_ames_model, "predict", fail_to_load)
    blocked = mcp_api._toxscan_ames_view(toxscan_ames.screen_run(tmp_path))
    assert blocked["blocker"] == "toxscan_ames_model_integrity" and "score" not in blocked
    jsonschema.Draft202012Validator(server._PREPARED_MEASUREMENT_SCHEMAS[METHOD]).validate(blocked)
