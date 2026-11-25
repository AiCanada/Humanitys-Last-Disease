"""Ames predictions stay attached to nominated molecules, never to a safety verdict."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
from importlib.metadata import PathDistribution
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from rfproteina.disease import ligand_ames as ames


def test_output_uses_the_exact_endpoint_and_input_identity() -> None:
    frame = pd.DataFrame(
        {"AMES": [1.0, 0.0, 0.2], "AMES_drugbank_approved_percentile": [18, 22, 45]},
        index=["CO", "CCO", "CC"],
    )
    assert ames._values(frame, ["CC", "CCO", "CO"]) == {"CC": 0.2, "CCO": 0.0, "CO": 1.0}
    with pytest.raises(ValueError, match="missing_or_duplicate_endpoint"):
        ames._values(frame.iloc[:, [1]], ["CO"])
    with pytest.raises(ValueError, match="missing_or_duplicate_endpoint"):
        ames._values(frame.iloc[:, [0, 0]], ["CO"])
    with pytest.raises(ValueError, match="unexpected_provider_molecule"):
        ames._values(frame, ["CO", "CCO"])


def test_bad_probabilities_are_withheld_without_dropping_valid_rows() -> None:
    frame = pd.DataFrame(
        {"AMES": [0.0, 0.8, 1.0, -0.01, 1.01, True, float("nan"), float("inf"), "0.5", 0.0]},
        index=["dup", "dup", "one", "negative", "excess", "boolean", "nan", "infinite", "text", "zero"],
        dtype=object,
    )
    assert ames._values(frame, ["missing", *dict.fromkeys(frame.index)]) == {
        "missing": "missing_provider_row",
        "dup": "duplicate_provider_row",
        "one": 1.0,
        **dict.fromkeys(("negative", "excess", "boolean", "nan", "infinite", "text"), "invalid_provider_result"),
        "zero": 0.0,
    }


def test_candidate_identity_coverage_and_private_structures(tmp_path, monkeypatch) -> None:
    pytest.importorskip("rdkit")
    calls = []

    def predict(keys):
        calls.append(keys)
        return {"values": {key: 0.0 if key == "CCO" else 1.0 for key in reversed(keys)}}

    monkeypatch.setattr(ames, "_local_predictions", predict)
    (tmp_path / "ligand_smiles.txt").write_text("C legacy ligand\n", encoding="utf-8")
    (tmp_path / "ligand_candidates.tsv").write_text(
        "candidate_id\tsmiles\n"
        "first\tCCO\nsecond\tOCC\nmapped\t[CH3:3]CO\n"
        "left\tF[C@H](Cl)Br\nright\tF[C@@H](Cl)Br\n"
        "salt\tCCO.[Na+]\ncarbonfree\t[Na+]\nradical\t[CH3]\ndummy\tCC*\n"
        "unreadable\tCCO name\ndup\tCO\ndup\tCCC\nshort\n",
        encoding="utf-8-sig",
    )
    report = ames.screen_run(tmp_path)
    assert report["scope"] == "candidate_table"
    assert report["status"] == "partial"
    assert report["attempted"] == 13
    assert report["unique_compounds"] == report["unique_predicted"] == 3
    assert len(calls) == 1 and len(calls[0]) == 3 and "C" not in calls[0]
    rows = report["inputs"]
    assert rows[0]["compound_id"] == rows[1]["compound_id"] == rows[2]["compound_id"]
    assert rows[3]["compound_id"] != rows[4]["compound_id"]
    assert [row["blocker"] for row in rows[5:9]] == ["unsupported_ames_structure"] * 4
    assert all("compound_id" not in row for row in rows[5:])
    assert [row["blocker"] for row in rows[10:]] == [
        "duplicate_candidate_id", "duplicate_candidate_id", "invalid_candidate_table_width"
    ]
    values = {row["compound_id"]: row["predicted_ames_probability"] for row in report["compounds"]}
    assert values[rows[0]["compound_id"]] == 0.0
    assert values[rows[3]["compound_id"]] == values[rows[4]["compound_id"]] == 1.0
    assert all(row["status"] == "predicted" for row in report["compounds"])
    serialized = json.dumps(report)
    assert "structures" not in report
    assert all(key not in serialized for key in calls[0])
    assert "call" not in report and "threshold" not in report

    selected = ames.screen_run(tmp_path, candidate="second")
    assert selected["status"] == "predicted" and selected["attempted"] == 1
    assert selected["inputs"][0]["compound_id"] == rows[0]["compound_id"]
    assert calls[-1] == ["CCO"]


def test_missing_or_invalid_nomination_never_falls_back(tmp_path, monkeypatch) -> None:
    def forbidden(_keys):
        pytest.fail("an invalid nomination reached the provider")

    monkeypatch.setattr(ames, "_local_predictions", forbidden)
    missing = ames.screen_run(tmp_path)
    assert missing["status"] == "untested" and missing["blocker"]
    (tmp_path / "ligand_smiles.txt").write_text("CCO legacy ligand\n", encoding="utf-8")
    blocked = ames.screen_run(tmp_path, candidate="run_ligand")
    assert blocked["status"] == "untested" and blocked["blocker"] == "candidate_requires_named_nomination"
    table = tmp_path / "ligand_candidates.tsv"
    table.write_text("id\tsmiles\nnomination\tCCC\n", encoding="utf-8")
    blocked = ames.screen_run(tmp_path)
    assert blocked["status"] == "untested" and blocked["blocker"] == "invalid_candidate_table_header"
    table.write_bytes(b"\xff")
    blocked = ames.screen_run(tmp_path)
    assert blocked["status"] == "untested" and blocked["blocker"] == "ligand_input_unreadable"


def test_provider_failure_preserves_nomination_evidence(monkeypatch) -> None:
    pytest.importorskip("rdkit")
    monkeypatch.setattr(ames, "_local_predictions", lambda _keys: {"blocker": "provider_timeout"})
    report = ames.profile([("first", "CCO"), ("second", "OCC")], scope="candidate_table")
    assert report["status"] == "untested" and report["blocker"] == "provider_timeout"
    assert report["attempted"] == 2 and report["unique_compounds"] == 1 and report["unique_predicted"] == 0
    assert report["inputs"][0]["compound_id"] == report["inputs"][1]["compound_id"]
    assert report["compounds"][0]["blocker"] == "provider_timeout"
    assert "predicted_ames_probability" not in report["compounds"][0]


def test_batch_recovers_attributable_predictions_after_one_featurization_failure() -> None:
    class Model:
        def predict(self, inputs):
            if "CC" in inputs:
                raise ValueError("provider could not featurize the molecule")
            return pd.DataFrame({ames.ENDPOINT: [0.25] * len(inputs)}, index=inputs)

    assert ames._score_batch(Model(), ["CO", "CC", "CCO"]) == {
        "CO": 0.25,
        "CC": "invalid_provider_result",
        "CCO": 0.25,
    }


def test_worker_refuses_invalid_json_nonobject_and_oversized_reports(monkeypatch) -> None:
    responses = iter(("not JSON", "[0.25]", " " * (ames._MAX_RESPONSE_BYTES + 1)))

    def worker(argv, **kwargs):
        assert argv[1:] == ["-m", ames.__name__]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == kwargs["env"]["HIP_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["timeout"] > 0
        assert json.loads(kwargs["input"]) == ["CO"]
        return SimpleNamespace(stdout=next(responses))

    monkeypatch.setattr(ames.subprocess, "run", worker)
    for _ in range(3):
        assert ames._local_predictions(["CO"]) == {"blocker": "provider_worker_failed"}


@pytest.mark.parametrize("values", [[], {"CO": 0.25, "CC": 0.5}])
def test_invalid_worker_value_map_never_publishes_a_partial_prediction(monkeypatch, values) -> None:
    pytest.importorskip("rdkit")
    monkeypatch.setattr(ames, "_local_predictions", lambda _keys: {"values": values})
    report = ames.profile([("nomination", "CO")], scope="candidate_table")
    assert report["status"] == "untested" and report["blocker"] == "provider_worker_failed"
    assert report["unique_predicted"] == 0
    assert "predicted_ames_probability" not in report["compounds"][0]


def test_size_limit_applies_to_authoritative_source_even_with_valid_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ames, "_local_predictions", lambda _keys: pytest.fail("oversized input reached inference"))
    (tmp_path / "ligand_smiles.txt").write_text("CCO legacy ligand\n", encoding="utf-8")
    table = tmp_path / "ligand_candidates.tsv"
    table.write_bytes(b"candidate_id\tsmiles\n" + b" " * ames.MAX_INPUT_BYTES)
    report = ames.screen_run(tmp_path)
    assert report["scope"] == "candidate_table"
    assert report["status"] == "untested" and report["blocker"] == "ames_input_limit_exceeded"


@pytest.mark.parametrize("candidate", [None, "candidate_0"])
def test_nomination_limit_cannot_be_bypassed_by_selecting_one_candidate(tmp_path, monkeypatch, candidate) -> None:
    monkeypatch.setattr(ames, "_local_predictions", lambda _keys: pytest.fail("oversized table reached inference"))
    table = tmp_path / "ligand_candidates.tsv"
    table.write_text(
        "candidate_id\tsmiles\n"
        + "".join(f"candidate_{i}\tCCO\n" for i in range(ames.MAX_CANDIDATES + 1)),
        encoding="utf-8",
    )
    report = ames.screen_run(tmp_path, candidate=candidate)
    assert report["status"] == "untested" and report["blocker"] == "ames_input_limit_exceeded"


def test_symlinked_table_is_rejected_without_reading_its_target_or_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ames, "_local_predictions", lambda _keys: pytest.fail("symlinked input reached inference"))
    (tmp_path / "ligand_smiles.txt").write_text("CCO legacy ligand\n", encoding="utf-8")
    linked = tmp_path / "outside.tsv"
    linked.write_text("candidate_id\tsmiles\nnomination\tCCC\n", encoding="utf-8")
    try:
        (tmp_path / "ligand_candidates.tsv").symlink_to(linked)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    report = ames.screen_run(tmp_path)
    assert report["scope"] == "candidate_table"
    assert report["status"] == "untested" and report["blocker"] == "ligand_input_unreadable"


def test_asset_verification_checks_the_ames_definition_and_all_five_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = tmp_path / "admet_ai-2.0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: admet-ai\nVersion: 2.0.1\n", encoding="utf-8")
    root = tmp_path / "admet_ai/resources"
    ensemble = root / "models/admet_classification"
    ensemble.mkdir(parents=True)
    weights = [ensemble / f"model_{i}.pt" for i in range(5)]
    for weight in weights:
        weight.write_bytes(b"checkpoint fixture")
    endpoint = root / "data/admet.csv"
    endpoint.parent.mkdir()
    header = "id,task_type,units,minimum,maximum,species\n"
    endpoint.write_text(header + "AMES,classification,-,0.0,1.0,salmonella typhimurium\n", encoding="utf-8")

    def record_files():
        with (info / "RECORD").open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target)
            for path in [*weights, endpoint]:
                data = path.read_bytes()
                checksum = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
                writer.writerow([path.relative_to(tmp_path).as_posix(), f"sha256={checksum}", len(data)])

    record_files()
    monkeypatch.setattr("importlib.metadata.distribution", lambda name: PathDistribution(info))
    ames.verified_assets()
    weights[-1].write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="installed_artifact_digest_mismatch"):
        ames.verified_assets()
    endpoint.write_text(header + "AMES,regression,-,0.0,1.0,salmonella typhimurium\n", encoding="utf-8")
    record_files()
    with pytest.raises(ValueError, match="endpoint_definition_mismatch"):
        ames.verified_assets()
