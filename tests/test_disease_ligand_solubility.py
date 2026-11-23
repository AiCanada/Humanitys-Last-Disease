"""Regression contracts for the signed, compound-level AqSolDB measurement."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rfproteina.disease import ligand_solubility as sol
from rfproteina.disease.corpus_provenance import BENCHMARK_LABEL_CORPUS


def test_signed_values_follow_the_exact_rows_and_head() -> None:
    frame = pd.DataFrame(
        {sol.ENDPOINT: [1.5, -2.6, 0.0], sol.ENDPOINT + "_drugbank_approved_percentile": [18, 22, 45]},
        index=["CC", "CO", "CCO"],
    )
    assert sol._values(frame, ["CO", "CCO", "CC"]) == {"CO": -2.6, "CCO": 0.0, "CC": 1.5}
    with pytest.raises(sol.SolubilityError, match="missing_or_duplicate_endpoint"):
        sol._values(frame.iloc[:, [1]], ["CC"])
    with pytest.raises(sol.SolubilityError, match="missing_or_duplicate_endpoint"):
        sol._values(frame.iloc[:, [0, 0]], ["CC"])
    with pytest.raises(sol.SolubilityError, match="unexpected_provider_molecule"):
        sol._values(frame, ["CC"])


def test_missing_duplicate_and_invalid_values_are_local_to_one_compound() -> None:
    frame = pd.DataFrame(
        {sol.ENDPOINT: [0.0, 1.1, True, float("nan"), float("inf"), "-1", -3.0]},
        index=["a", "a", "b", "c", "d", "e", "f"],
        dtype=object,
    )
    result = sol._values(frame, list("abcdefg"))
    assert result == {
        "a": "duplicate_provider_row",
        **dict.fromkeys("bcde", "nonfinite_or_nonnumeric_prediction"),
        "f": -3.0,
        "g": "missing_provider_row",
    }


def test_a_provider_featurization_failure_preserves_its_batch_mates() -> None:
    class Model:
        def predict(self, inputs):
            if "C" in inputs:
                raise ValueError("one molecule rejected")
            return pd.DataFrame({sol.ENDPOINT: [-2.5] * len(inputs)}, index=inputs)

    assert sol._score_batch(Model(), ["CO", "C", "CCO"]) == {"CO": -2.5, "C": "provider_failed_ValueError", "CCO": -2.5}


def test_table_identity_coverage_and_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("rdkit")
    calls = []

    def predict(keys):
        calls.append(keys)
        return {"values": dict.fromkeys(keys, -1.5)}

    monkeypatch.setattr(sol, "_local_predictions", predict)
    (tmp_path / "ligand_smiles.txt").write_text("C legacy ligand\n", encoding="utf-8")
    (tmp_path / "ligand_candidates.tsv").write_text(
        "candidate_id\tsmiles\na\tCCO\nb\tOCC\nmapped\t[CH3:3]CO\nsalt\tCCO.[Na+]\n"
        "left\tF[C@H](Cl)Br\nright\tF[C@@H](Cl)Br\nisotope\t[13CH3]CO\n"
        "bad\tCCO name\ndummy\t*\ndup\tCO\ndup\tCCC\nshort\nwide\tC\tC\n"
        "nul\tCC\x00O\nextended\tCCO |$x$|\n",
        encoding="utf-8-sig",
    )
    block = sol.screen_run(tmp_path)
    assert block["scope"] == "candidate_table"
    assert (block["attempted"], block["unique_compounds"], block["unique_predicted"]) == (15, 5, 5)
    assert block["status"] == "partial" and len(calls) == 1 and len(calls[0]) == 5
    rows = block["inputs"]
    assert rows[0]["compound_id"] == rows[1]["compound_id"] == rows[2]["compound_id"]
    reordered = sol.profile([("different_first", "CO"), ("ethanol", "OCC")], scope="standalone")
    assert reordered["inputs"][1]["compound_id"] == rows[0]["compound_id"]
    assert [row["blocker"] for row in rows[7:]] == [
        "invalid_smiles",
        "invalid_smiles",
        "duplicate_candidate_id",
        "duplicate_candidate_id",
        "invalid_candidate_table_width",
        "invalid_candidate_table_width",
        "invalid_smiles",
        "invalid_smiles",
    ]
    assert block["corpus_provenance"]["overlap"] == BENCHMARK_LABEL_CORPUS
    assert all(
        c["method"] == sol.METHOD and c["unit"] == sol.UNIT and c["endpoint"] == sol.ENDPOINT
        for c in block["compounds"]
    )


def test_bad_manifest_cannot_fall_back_to_fixed_ligand(tmp_path: Path) -> None:
    assert sol.screen_run(tmp_path) is None
    single = tmp_path / "ligand_smiles.txt"
    single.write_text("CCO\nCC\n", encoding="utf-8")
    assert sol.screen_run(tmp_path)["blocker"] == "expected_one_run_ligand"
    single.write_text("CCO |$x$|\n", encoding="utf-8")
    assert sol.screen_run(tmp_path)["blocker"] == "unsupported_extended_smiles"
    single.write_text("CCO ethanol\n", encoding="utf-8")
    table = tmp_path / "ligand_candidates.tsv"
    table.write_text("id\tsmiles\nid\tCCC\n", encoding="utf-8")
    assert sol.screen_run(tmp_path)["blocker"] == "invalid_candidate_table_header"
    table.write_bytes(b"\xff")
    assert sol.screen_run(tmp_path)["blocker"] == "ligand_input_unreadable"
    table.write_text("candidate_id\tsmiles\n", encoding="utf-8")
    assert sol.screen_run(tmp_path)["blocker"] == "no_candidates"


def test_absent_provider_preserves_all_nominations(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("rdkit")
    monkeypatch.setattr(sol, "_local_predictions", lambda keys: {"blocker": "provider_dependency_unavailable"})
    block = sol.profile([("one", "CCO"), ("two", "OCC"), ("invalid", "*")], scope="standalone")
    assert block["status"] == "untested" and block["attempted"] == 3 and block["unique_compounds"] == 1
    assert block["blocker"] == block["compounds"][0]["blocker"] == "provider_dependency_unavailable"
    assert "predicted_log_s" not in block["compounds"][0] and block["unique_predicted"] == 0


def test_partial_checkpoint_install_is_rejected_before_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Installation:
        version = "2.0.1"

        def locate_file(self, path):
            return tmp_path / path

    monkeypatch.setattr(sol.metadata, "distribution", lambda name: Installation())
    directory = tmp_path / "admet_ai/resources/models/admet_regression"
    directory.mkdir(parents=True)
    for i in range(4):
        (directory / f"model_{i}.pt").write_bytes(b"")
    with pytest.raises(sol.SolubilityError, match="incomplete_or_unrecognized_regression_ensemble"):
        sol._model()


def test_picker_verifies_checkpoint_bytes_and_endpoint_units_without_loading_models(tmp_path, monkeypatch):
    import base64
    import csv
    import hashlib
    from importlib.metadata import PathDistribution

    from rfproteina.disease.screen_selection import installations

    info = tmp_path / "admet_ai-2.0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: admet-ai\nVersion: 2.0.1\n")
    root = tmp_path / "admet_ai/resources"
    members = root / "models/admet_regression"
    members.mkdir(parents=True)
    paths = [members / f"model_{i}.pt" for i in range(5)]
    for path in paths:
        path.write_bytes(b"weight fixture")
    endpoint = root / "data/admet.csv"
    endpoint.parent.mkdir()
    endpoint.write_text("id,task_type,units\nSolubility_AqSolDB,regression,log(mol/L)\n")

    def record():
        with (info / "RECORD").open("w", newline="") as target:
            writer = csv.writer(target)
            for path in [*paths, endpoint]:
                data = path.read_bytes()
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
                writer.writerow([path.relative_to(tmp_path).as_posix(), f"sha256={digest}", len(data)])

    record()
    monkeypatch.setattr(sol.metadata, "distribution", lambda name: PathDistribution(info))
    monkeypatch.setattr(sol, "_model", lambda: pytest.fail("picker loaded a model"))
    sol.verified_assets()
    paths[0].write_bytes(b"broken fixture")
    assert not installations({"ligand_solubility"})["ligand_solubility"]
    endpoint.write_text("id,task_type,units\nSolubility_AqSolDB,regression,mg/L\n")
    record()
    with pytest.raises(sol.SolubilityError, match="endpoint_definition_mismatch"):
        sol.verified_assets()


def test_a_run_level_molecule_cannot_impersonate_a_candidate_named_run_ligand(tmp_path, monkeypatch):
    (tmp_path / "ligand_smiles.txt").write_text("CCO ethanol\n")
    monkeypatch.setattr(sol, "_local_predictions", lambda *args: pytest.fail("unattributed molecule reached inference"))
    blocked = sol.screen_run(tmp_path, candidate="run_ligand")
    assert blocked["status"] == "untested" and blocked["blocker"] == "candidate_requires_named_nomination"
