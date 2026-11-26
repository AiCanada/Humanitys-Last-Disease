"""Fingerprint representation, split hygiene, safe tensor format and held-out Ames readback."""

import csv
import hashlib
import json

import numpy as np
import pytest
from safetensors.numpy import save

from rfproteina.disease import toxscan_ames_model as ames
from rfproteina.disease import toxscan_inputs
from scripts import train_toxscan_ames as trainer


def toy_bundle(path):
    validation = {
        "n_train": 2,
        "n_validation": 2,
        "n_test": 2,
        "test_positives": 1,
        "test_negatives": 1,
        "tp": 1,
        "tn": 0,
        "fp": 1,
        "fn": 0,
        "selected_epoch": 1,
        "test_fingerprint_seen_in_train": 0,
        "roc_auc": 0.5,
        "accuracy": 0.5,
        "validation_roc_auc": 0.5,
    }
    evidence = {
        "version": 1,
        "method": ames.METHOD_ID,
        "archive_sha256": ames.ARCHIVE_SHA256,
        "source_revision": ames.SOURCE_REVISION,
        "tables_sha256": ames.SOURCE_SHA256,
        "fit_settings": ames.FIT_SETTINGS,
        "fingerprint": ames.FINGERPRINT,
        "validation": validation,
        "split_audit": {"test": {"graphs_with_fingerprint_seen_in_train": 0}},
    }
    tensors = {}
    for i, (n_in, n_out) in enumerate(
        zip(ames.FIT_SETTINGS["sizes"][:-1], ames.FIT_SETTINGS["sizes"][1:], strict=True)
    ):
        tensors[f"w{i}"] = np.zeros((n_out, n_in), dtype=np.float32)
        tensors[f"b{i}"] = np.zeros(n_out, dtype=np.float32)
    smiles = ("CCO", "CCC", "CCCl", "CCBr", "CCN", "CCF")
    for i, split in enumerate(ames.SPLITS):
        tensors[split] = trainer._digest_matrix(
            [hashlib.sha256(s.encode()).digest() for s in smiles[i * 2 : i * 2 + 2]]
        )
    tensors["training_fingerprints"] = trainer._digest_matrix(
        [ames.fingerprint_digest(ames.fingerprint(s)) for s in smiles[:2]]
    )
    tensors["test_scores"] = np.array([0.5, 0.5], dtype=np.float64)
    tensors["test_labels"] = np.array([0, 1], dtype=np.uint8)
    tensors["evidence_digest"] = np.frombuffer(ames.evidence_digest(evidence), dtype=np.uint8)
    blob = save(tensors)
    evidence["model_sha256"] = hashlib.sha256(blob).hexdigest()
    (path / ames.WEIGHTS).write_bytes(blob)
    (path / ames.RECEIPT).write_text(json.dumps(evidence))
    return evidence


def test_native_fingerprint_matches_archive_and_keeps_chirality_at_the_identity_level():
    assert ames.verify_representation()
    a, b = "C[C@H](Cl)O", "C[C@@H](Cl)O"
    assert toxscan_inputs.canonical(a) != toxscan_inputs.canonical(b)
    assert np.array_equal(ames.fingerprint(a), ames.fingerprint(b))


def test_local_fit_scores_new_molecules_and_detects_changed_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(ames, "release_directory", lambda: tmp_path)
    assert not ames.installed() and ames.benchmark() is None and ames.installed_sha256() is None
    evidence = toy_bundle(tmp_path)
    assert ames.installed() and ames.benchmark() == evidence["validation"]
    assert ames.installed_sha256() == evidence["model_sha256"]
    with monkeypatch.context() as no_forward:
        no_forward.setattr(ames, "score_vector", lambda *args: pytest.fail("unexpected model forward"))
        ames._cached_model.cache_clear()
        assert ames.fingerprint_seen_in_training("C(C)O")
        assert not ames.fingerprint_seen_in_training("CCF")
    with pytest.raises(ames.AmesModelError, match="toxscan_ames_invalid_metrics"):
        ames.validate_validation({**evidence["validation"], "test_fingerprint_seen_in_train": 3})
    for smiles, relation, exposed in (
        ("C(C)O", "train", True),
        ("CCN", "test", False),
        ("COC", "not_in_retained_splits", False),
    ):
        result = ames.predict(smiles)
        assert result["graph_membership"] == relation and result["fingerprint_seen_in_training"] is exposed
        assert result["score"] == 0.5 and result["class_call"] == "YES"
    with pytest.raises(ames.AmesModelError, match="toxscan_ames_invalid_structure"):
        ames.predict("C~C")
    evidence["validation"]["roc_auc"] = 1
    (tmp_path / ames.RECEIPT).write_text(json.dumps(evidence))
    assert not ames.installed()
    with pytest.raises(ames.AmesModelError, match="toxscan_ames_model_integrity"):
        ames.predict("CCO")


def test_source_graph_duplicates_conflicts_and_overlaps_do_not_cross_splits(tmp_path, monkeypatch):
    values = {
        "train": [("CCO", 1), ("OCC", 1), ("CO", 0), ("CCC", 0), ("CC", 1), ("CC", 0)],
        "valid": [("CCCl", 1), ("CCBr", 0), ("CCC", 1)],
        "test": [("CCC", 0), ("CCF", 1), ("CCN", 0), ("C~C", 1)],
    }
    digests = {}
    for split, rows in values.items():
        path = tmp_path / f"{split}.csv"
        with path.open("w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["smiles", "ames"])
            writer.writerows(rows)
        digests[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(ames, "SOURCE_SHA256", digests)
    out, audit = trainer.read_splits(tmp_path)
    assert set(out["train"]) == {("CCO", 1), ("CO", 0)}
    assert audit["test"]["invalid_structure_rows"] == audit["train"]["conflict_graphs_removed"] == 1
    assert [audit[k]["overlap_graphs_removed"] for k in ames.SPLITS] == [1, 1, 1]
    (tmp_path / "train.csv").write_text("smiles,ames\nC,1\n")
    with pytest.raises(ValueError, match="unverified"):
        trainer.read_splits(tmp_path)


def test_runtime_rechecks_held_out_rank_statistic_and_boundary():
    metrics = {"n_test": 4, "test_positives": 2, "test_negatives": 2, "tp": 2, "tn": 1, "roc_auc": 0.875}
    scores = np.array([0.5, 0.9, 0.2, 0.5], dtype=np.float64)
    labels = np.array([1, 1, 0, 0], dtype=np.uint8)
    ames._verify_test_scores(scores, labels, metrics)
    with pytest.raises(ValueError):
        ames._verify_test_scores(scores, labels, {**metrics, "roc_auc": 1.0})


def test_invalid_or_oversized_tensor_file_cannot_enable_model(tmp_path, monkeypatch):
    monkeypatch.setattr(ames, "release_directory", lambda: tmp_path)
    evidence = toy_bundle(tmp_path)
    invalid = b"not a tensor file"
    evidence["model_sha256"] = hashlib.sha256(invalid).hexdigest()
    (tmp_path / ames.RECEIPT).write_text(json.dumps(evidence))
    (tmp_path / ames.WEIGHTS).write_bytes(invalid)
    assert ames.benchmark() is None
    with (tmp_path / ames.WEIGHTS).open("wb") as out:
        out.truncate(ames._MAX_MODEL_BYTES + 1)
    with pytest.raises(ames.AmesModelError, match="toxscan_ames_model_integrity"):
        ames.predict("CCO")


def test_genuine_installed_ames_fit_scores_unseen_molecule_and_marks_training_control():
    if not ames.release_directory().exists():
        pytest.skip("optional trained Ames model is not installed")
    assert ames.installed()
    novel = "CCOC(=O)N1CCN(C(=O)c2cc(F)cc(O)c2)CC1"
    report = ames.predict(novel)
    assert report["validation"] == ames.validate_validation(ames.benchmark())
    assert report["graph_membership"] == "not_in_retained_splits"
    assert not report["fingerprint_seen_in_training"]
    assert report["class_call"] == ("YES" if report["score"] >= 0.5 else "NO")
    assert ames.fingerprint_seen_in_training("NCCNCCN")
    assert ames.predict("NCCNCCN")["graph_membership"] == "train"
    if report["model_sha256"] == "b22f07cb13783af53a50df9879f5c860fd94baf6ffb376e40f267ce6cf7a50b2":
        assert report["score"] == pytest.approx(0.22609453, abs=2e-6)
        assert report["validation"]["roc_auc"] == pytest.approx(0.8062167655)
        assert report["validation"]["test_fingerprint_seen_in_train"] == 67
