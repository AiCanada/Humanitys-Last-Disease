"""Keep the two Caco-2 predictions attributable on the release's native scale."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rfproteina.disease import ligand_caco2 as caco2


def test_public_method_and_native_matrix_contract() -> None:
    assert (caco2.METHOD_ID, caco2.METHOD, caco2.SCALE) == (
        "ligand_caco2", "az_gnn_mtl_caco2_v1", "native_model_score",
    )
    assert caco2.RELEASE == "zenodo:16948542"
    assert (caco2.BATCH_SIZE, caco2.MAX_CANDIDATES) == (64, 256)
    rows = iter(((np.float32(-0.75), 0, 111, 222), [1.125, -2.5, 333, 444]))
    assert caco2._native_values(rows, ["CCO", "CCN"]) == {"CCO": [-0.75, 0.0], "CCN": [1.125, -2.5]}


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [[1, 2, 3]],
        [[1, 2, 3, 4, 5]],
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[1, 2], [3, 4]],
        ["1234"],
        [{0: 1, 1: 2, 2: 3, 3: 4}],
        [1],
    ],
)
def test_native_matrix_rejects_wrong_head_or_row_count(rows) -> None:
    with pytest.raises(caco2.CacoError, match="caco2_output_contract_mismatch"):
        caco2._native_values(iter(rows), ["CCO"])


@pytest.mark.parametrize("invalid", [float("inf"), float("-inf"), float("nan"), True, "2.3", None])
def test_invalid_native_values_do_not_erase_the_other_head(invalid) -> None:
    result = caco2._native_values([[invalid, -0.5, 8, 7], [0, invalid, 6, 5]], ["CCO", "CCN"])
    assert result == {"CCO": ["invalid_provider_result", -0.5], "CCN": [0.0, "invalid_provider_result"]}


def test_screen_preserves_aliases_and_independent_head_outcomes(tmp_path, monkeypatch) -> None:
    pytest.importorskip("rdkit")
    calls = []

    def score(keys):
        calls.append(keys)
        return {"values": {
            key: [-0.875, 0.0] if key == "CCO" else ["invalid_provider_result", 1.25]
            for key in reversed(keys)
        }}

    monkeypatch.setattr(caco2, "_local_predictions", score)
    (tmp_path / "ligand_smiles.txt").write_text("CN run fallback\n", encoding="utf-8")
    (tmp_path / "ligand_candidates.tsv").write_text(
        "candidate_id\tsmiles\nfirst\tCCO\nalias\tOCC\nother\tCCN\n"
        "salt\tCCO.[Na+]\nduplicate\tCO\nduplicate\tOC\nmalformed\n",
        encoding="utf-8",
    )

    report = caco2.screen_run(tmp_path)
    assert calls == [["CCO", "CCN"]]
    assert (report["scope"], report["method"], report["scale"]) == (
        "candidate_table", caco2.METHOD, caco2.SCALE,
    )
    assert (report["status"], report["attempted"], report["unique_compounds"], report["unique_complete"]) == (
        "partial", 7, 2, 1,
    )
    first, alias, other, *blocked = report["inputs"]
    assert first["compound_id"] == alias["compound_id"] != other["compound_id"]
    assert all("compound_id" not in row and row.get("blocker") for row in blocked)
    assert [row["blocker"] for row in blocked[1:]] == [
        "duplicate_candidate_id", "duplicate_candidate_id", "invalid_candidate_table_width",
    ]
    assert report["unique_predicted"] == 2
    compounds = {row["compound_id"]: row for row in report["compounds"]}
    for compound in compounds.values():
        for head, endpoint in (("efflux_ratio", "Caco2ER"), ("apparent_permeability", "Caco2Papp")):
            evidence = compound["heads"][head]
            assert (evidence["endpoint"], evidence["scale"]) == (endpoint, caco2.SCALE)
    predicted = compounds[first["compound_id"]]["heads"]
    assert (predicted["efflux_ratio"]["status"], predicted["efflux_ratio"]["model_score"]) == (
        "predicted", -0.875,
    )
    assert (predicted["apparent_permeability"]["status"], predicted["apparent_permeability"]["model_score"]) == (
        "predicted", 0.0,
    )
    partial = compounds[other["compound_id"]]["heads"]
    assert partial["efflux_ratio"]["status"] == "untested"
    assert partial["efflux_ratio"]["blocker"] == "invalid_provider_result"
    assert "model_score" not in partial["efflux_ratio"]
    assert partial["apparent_permeability"]["model_score"] == 1.25
    rendered = json.dumps(report, allow_nan=False)
    assert not any(smiles in rendered for smiles in ("CCO", "OCC", "CCN", "[Na+]"))
    assert not ({"threshold", "call", "physical_unit", "structures"} & report.keys())

    selected = caco2.screen_run(tmp_path, candidate="alias")
    assert calls[-1] == ["CCO"]
    assert (selected["status"], selected["attempted"], selected["unique_complete"]) == ("predicted", 1, 1)
    assert selected["inputs"][0]["compound_id"] == first["compound_id"]


@pytest.mark.parametrize(
    "values",
    [[], {}, {"CCO": [0, 1], "foreign": [2, 3]}, {"CCO": [1]}, {"CCO": [0, 1, 2, 3]}],
)
def test_worker_result_cannot_bypass_head_shape_and_molecule_validation(monkeypatch, values) -> None:
    pytest.importorskip("rdkit")
    monkeypatch.setattr(caco2, "_local_predictions", lambda _keys: {"values": values})
    report = caco2.profile([("candidate", "CCO")], scope="candidate_table")
    assert report["status"] == "untested" and report["unique_complete"] == 0
    compound, = report["compounds"]
    for head in compound["heads"].values():
        assert head["status"] == "untested" and head["blocker"]
        assert "model_score" not in head


def test_nominations_are_bounded_before_filtering_or_running_the_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(caco2, "_local_predictions", lambda _keys: pytest.fail("invalid nomination reached model"))
    (tmp_path / "ligand_smiles.txt").write_text("CCO fallback\n", encoding="utf-8")
    legacy = caco2.screen_run(tmp_path, candidate="run_ligand")
    assert legacy["status"] == "untested" and legacy["blocker"] == "candidate_requires_named_nomination"
    table = tmp_path / "ligand_candidates.tsv"
    table.write_text("id\tsmiles\ncandidate_0\tCCO\n", encoding="utf-8")
    malformed = caco2.screen_run(tmp_path)
    assert malformed["scope"] == "candidate_table" and malformed["blocker"] == "invalid_candidate_table_header"
    table.write_text(
        "candidate_id\tsmiles\n" + "".join(f"candidate_{i}\tCCO\n" for i in range(caco2.MAX_CANDIDATES + 1)),
        encoding="utf-8",
    )
    for candidate in (None, "candidate_0"):
        report = caco2.screen_run(tmp_path, candidate=candidate)
        assert report["status"] == "untested" and report["blocker"]
        assert report["scope"] == "candidate_table" and not report["compounds"]


def test_local_inference_loads_once_and_keeps_batches_bounded(monkeypatch) -> None:
    model = object()
    calls = []

    def load():
        calls.append("load")
        return model, {"device": "cpu"}

    def score(loaded_model, keys):
        assert loaded_model is model
        calls.append(keys)
        return {key: [0.0, -1.0] for key in keys}

    monkeypatch.setattr(caco2, "_load_model", load)
    monkeypatch.setattr(caco2, "_score_batch", score)
    keys = [f"key_{index}" for index in range(2 * caco2.BATCH_SIZE + 1)]
    result = caco2._infer(keys)
    assert calls == ["load", keys[:64], keys[64:128], keys[128:]]
    assert result == {"provider": {"device": "cpu"}, "values": dict.fromkeys(keys, [0.0, -1.0])}


@pytest.fixture
def checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, bytes]:
    monkeypatch.delenv("RFD4_CACO2_CHECKPOINT", raising=False)
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    contents = b"pinned caco2 release fixture"
    monkeypatch.setattr(caco2, "CHECKPOINT_BYTES", len(contents))
    monkeypatch.setattr(caco2, "CHECKPOINT_SHA256", hashlib.sha256(contents).hexdigest())
    monkeypatch.setattr(caco2, "_runtime_version", lambda: "2.3.1")
    monkeypatch.setattr(caco2.util, "find_spec", lambda _name: object())
    path = tmp_path / "caco2" / "az_gnn_mtl_v1.pt"
    path.parent.mkdir()
    path.write_bytes(contents)
    return path, contents


def test_verified_checkpoint_never_reuses_stale_bytes(checkpoint) -> None:
    path, original = checkpoint
    verified = caco2.verified_checkpoint()
    assert isinstance(verified, bytes) and verified == original
    assert caco2.asset_available()
    path.write_bytes(b"!" + original[1:])
    with pytest.raises(caco2.CacoError):
        caco2.verified_checkpoint()
    assert not caco2.asset_available()
    path.write_bytes(original + b"!")
    with pytest.raises(caco2.CacoError):
        caco2.verified_checkpoint()
    assert not caco2.asset_available()
    path.write_bytes(original)
    assert caco2.verified_checkpoint() == verified
    assert caco2.asset_available()


def test_unqualified_chemprop_version_is_not_reported_available(monkeypatch) -> None:
    monkeypatch.setattr(caco2.metadata, "version", lambda _name: "2.1.0")
    monkeypatch.setattr(caco2, "verified_checkpoint", lambda: pytest.fail("unqualified runtime checked asset bytes"))
    assert not caco2.asset_available()


def test_explicit_checkpoint_never_falls_back_when_missing_or_corrupt(tmp_path, monkeypatch, checkpoint) -> None:
    default, original = checkpoint
    override = tmp_path / "explicit.pt"
    monkeypatch.setenv("RFD4_CACO2_CHECKPOINT", str(override))
    for contents in (None, b"!" + original[1:]):
        if contents is not None:
            override.write_bytes(contents)
        with pytest.raises(caco2.CacoError):
            caco2.verified_checkpoint()
        assert not caco2.asset_available()
    override.write_bytes(original)
    default.unlink()
    assert caco2.verified_checkpoint() == original
