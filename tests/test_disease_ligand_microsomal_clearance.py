"""Pin the native human liver microsomal endpoint and its attribution contract."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
from importlib.metadata import PathDistribution
from pathlib import Path

import pandas as pd
import pytest

from rfproteina.disease import ligand_microsomal_clearance as clearance


def test_native_values_are_not_logged_rescaled_clamped_or_taken_from_other_heads() -> None:
    frame = pd.DataFrame(
        {
            "Clearance_Hepatocyte_AZ": [71, 72, 73],
            "Clearance_Microsome_AZ": [0.0, 125.375, 210.25],
            "Clearance_Microsome_AZ_drugbank_approved_percentile": [5, 50, 95],
        },
        index=["CO", "CCO", "CC"],
    )
    assert clearance._values(frame, ["CC", "CO", "CCO"]) == {"CC": 210.25, "CO": 0.0, "CCO": 125.375}
    with pytest.raises(clearance.ClearanceError, match="missing_or_duplicate_endpoint"):
        clearance._values(frame.iloc[:, [0, 2]], ["CO"])
    with pytest.raises(clearance.ClearanceError, match="missing_or_duplicate_endpoint"):
        clearance._values(frame.iloc[:, [1, 1]], ["CO"])
    with pytest.raises(clearance.ClearanceError, match="unexpected_provider_molecule"):
        clearance._values(frame, ["CO", "CCO"])


def test_bad_regression_values_and_missing_or_duplicate_rows_are_not_attributed() -> None:
    values = {
        "negative": -0.125,
        "nan": float("nan"),
        "infinity": float("inf"),
        "boolean": True,
        "numeric_text": "5.1",
        "zero": 0,
    }
    frame = pd.DataFrame(
        {clearance.ENDPOINT: [*values.values(), 4.125, 9.125]},
        index=[*values, "duplicate", "duplicate"],
        dtype=object,
    )
    assert clearance._values(frame, ["missing", "duplicate", *values]) == {
        "missing": "missing_provider_row",
        "duplicate": "duplicate_provider_row",
        **dict.fromkeys(("negative", "nan", "infinity", "boolean", "numeric_text"), "invalid_provider_result"),
        "zero": 0.0,
    }


def test_failed_candidates_survive_and_equivalent_graphs_share_the_right_result(tmp_path, monkeypatch) -> None:
    pytest.importorskip("rdkit")
    calls = []

    def predict(keys):
        calls.append(keys)
        # Deliberately return a different order from the input.
        return {"values": {key: 0.0 if key == "CCO" else 172.125 for key in reversed(keys)}}

    monkeypatch.setattr(clearance, "_local_predictions", predict)
    (tmp_path / "ligand_smiles.txt").write_text("CN run fallback\n", encoding="utf-8")
    (tmp_path / "ligand_candidates.tsv").write_text(
        "candidate_id\tsmiles\nfirst\tCCO\nalias\tOCC\nother\tCC(=O)N\n"
        "disconnected\tCCO.[Na+]\nduplicate\tCCN\nduplicate\tOC\nmalformed\n",
        encoding="utf-8",
    )
    report = clearance.screen_run(tmp_path)
    assert report["scope"] == "candidate_table"
    assert report["status"] == "partial" and report["attempted"] == 7
    assert report["unique_compounds"] == report["unique_predicted"] == 2
    assert (report["method"], report["endpoint"], report["unit"]) == (
        "admet_ai_microsomal_clearance_v2", "Clearance_Microsome_AZ", "uL/min/mg"
    )
    assert calls == [["CCO", "CC(N)=O"]]
    first, alias, other, *blocked = report["inputs"]
    assert first["compound_id"] == alias["compound_id"] != other["compound_id"]
    assert all("compound_id" not in row and row.get("blocker") for row in blocked)
    assert [row["blocker"] for row in blocked[1:]] == [
        "duplicate_candidate_id", "duplicate_candidate_id", "invalid_candidate_table_width",
    ]
    scores = {row["compound_id"]: row["predicted_clint_ul_min_mg"] for row in report["compounds"]}
    assert scores == {first["compound_id"]: 0.0, other["compound_id"]: 172.125}
    assert all("predicted_ames_probability" not in row for row in report["compounds"])
    rendered = json.dumps(report)
    assert "structures" not in report and not ({"call", "threshold"} & report.keys())
    assert not any(smiles in rendered for smiles in ("CCO", "OCC", "CC(=O)N", "CC(N)=O"))

    selected = clearance.screen_run(tmp_path, candidate="alias")
    assert calls[-1] == ["CCO"] and selected["status"] == "predicted" and selected["attempted"] == 1
    assert selected["inputs"][0]["compound_id"] == first["compound_id"]


@pytest.mark.parametrize("values", [{"CCO": 32.5, "foreign": 51.0}, [], {"CCO": -7.0}])
def test_worker_result_cannot_bypass_value_and_attribution_validation(monkeypatch, values) -> None:
    pytest.importorskip("rdkit")
    monkeypatch.setattr(clearance, "_local_predictions", lambda _keys: {"values": values})
    report = clearance.profile([("candidate", "CCO")], scope="candidate_table")
    assert report["status"] == "untested" and report["unique_predicted"] == 0
    row, = report["compounds"]
    assert row["status"] == "untested" and "predicted_clint_ul_min_mg" not in row


def test_invalid_authoritative_table_and_run_level_fallback_are_not_candidate_evidence(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(clearance, "_local_predictions", lambda _: pytest.fail("invalid input reached provider"))
    (tmp_path / "ligand_smiles.txt").write_text("CCO legacy ligand\n", encoding="utf-8")
    run_level = clearance.screen_run(tmp_path, candidate="run_ligand")
    assert run_level["status"] == "untested" and run_level["blocker"] == "candidate_requires_named_nomination"
    table = tmp_path / "ligand_candidates.tsv"
    table.write_text("id\tsmiles\ncandidate_0\tCCO\n", encoding="utf-8")
    report = clearance.screen_run(tmp_path)
    assert report["scope"] == "candidate_table" and report["blocker"] == "invalid_candidate_table_header"
    table.write_bytes(b"candidate_id\tsmiles\ncandidate_0\tCCO\n" + b" " * clearance.MAX_INPUT_BYTES)
    limited = clearance.screen_run(tmp_path, candidate="candidate_0")
    assert limited["status"] == "untested" and limited["blocker"]
    table.write_text(
        "candidate_id\tsmiles\n" + "".join(f"candidate_{i}\tCCO\n" for i in range(clearance.MAX_CANDIDATES + 1)),
        encoding="utf-8",
    )
    limited = clearance.screen_run(tmp_path, candidate="candidate_0")
    assert limited["status"] == "untested" and limited["blocker"]


def test_batch_failure_preserves_other_molecules_at_the_original_scale() -> None:
    class Model:
        def predict(self, inputs):
            if "CC" in inputs:
                raise ValueError("provider could not featurize a molecule")
            return pd.DataFrame({clearance.ENDPOINT: [43.125] * len(inputs)}, index=inputs)

    assert clearance._score_batch(Model(), ["CO", "CC", "CCO"]) == {
        "CO": 43.125, "CC": "invalid_provider_result", "CCO": 43.125,
    }


@pytest.fixture
def installed_wheel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    info = tmp_path / "admet_ai-2.0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: admet-ai\nVersion: 2.0.1\n", encoding="utf-8")
    root = tmp_path / "admet_ai/resources"
    ensemble = root / "models/admet_regression"
    ensemble.mkdir(parents=True)
    weights = [ensemble / f"model_{i}.pt" for i in range(5)]
    for weight in weights:
        weight.write_bytes(b"regression checkpoint fixture")
    manifest = root / "data/admet.csv"
    manifest.parent.mkdir()

    def record(rows):
        manifest.write_text("id,task_type,units,minimum,maximum,species\n" + rows, encoding="utf-8")
        with (info / "RECORD").open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target)
            for path in [*weights, manifest]:
                data = path.read_bytes()
                checksum = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
                writer.writerow([path.relative_to(tmp_path).as_posix(), f"sha256={checksum}", len(data)])

    monkeypatch.setattr("importlib.metadata.distribution", lambda name: PathDistribution(info))
    return record, weights


_HUMAN_MICROSOMAL = "Clearance_Microsome_AZ,regression,uL/min/mg,0.0,inf,human\n"


def test_exact_manifest_and_all_regression_members_are_verified(installed_wheel) -> None:
    record, weights = installed_wheel
    record(_HUMAN_MICROSOMAL + "Clearance_Hepatocyte_AZ,regression,uL/min/10^6 cells,0.0,inf,human & rat\n")
    clearance.verified_assets()
    # Deliberately keep size constant: validation must reach the digest check.
    original = weights[-1].read_bytes()
    weights[-1].write_bytes(b"X" + original[1:])
    with pytest.raises(clearance.ClearanceError, match="installed_artifact_digest_mismatch"):
        clearance.verified_assets()


@pytest.mark.parametrize(
    "wrong_row",
    [
        _HUMAN_MICROSOMAL.replace("uL/min/mg", "log(uL/min/mg)"),
        _HUMAN_MICROSOMAL.replace("regression", "classification"),
        _HUMAN_MICROSOMAL.replace("human", "rat"),
        _HUMAN_MICROSOMAL.replace("0.0", "-5"),
        _HUMAN_MICROSOMAL.replace("inf", "150"),
        _HUMAN_MICROSOMAL * 2,
        _HUMAN_MICROSOMAL.replace("Microsome", "Hepatocyte"),
    ],
)
def test_altering_release_semantics_is_rejected_even_with_consistent_digests(installed_wheel, wrong_row) -> None:
    record, _ = installed_wheel
    record(wrong_row)
    with pytest.raises(clearance.ClearanceError, match="endpoint_definition_mismatch"):
        clearance.verified_assets()
