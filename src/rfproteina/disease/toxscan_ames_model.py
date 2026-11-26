"""Locally trained Ames adaptation of the ToxScan molecular fingerprint branch."""

from __future__ import annotations

import hashlib
import json
import math
import stat
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rfproteina.disease import toxscan_inputs
from rfproteina.disease.ligand_inputs import read_regular

if TYPE_CHECKING:
    from rdkit.Chem.rdFingerprintGenerator import FingerprintGenerator64

METHOD_ID = "toxscan_fingerprint_ames"
ARCHIVE_SHA256 = "e199ef53ee13ed2fe598ac96dd1ca4340b2b9aa602677c61e489cbc2dc4dd71f"
SOURCE_REVISION = "5eecd78a645366fbd2564ccb1c16032abf05a581"
WEIGHTS = "model.safetensors"
RECEIPT = "evidence.json"
SPLITS = ("train", "valid", "test")
SOURCE_SHA256 = {
    "train": "a120fa647216fdb53450133194a80970f2be0c831cf75a5923b136c73910ea9d",
    "valid": "546e8311cc2dc3d98e3d3ceeccca7a7778fed8cfae1fa4c9494c15569d47c9e5",
    "test": "e657ffaa79324cb324b8026ef2a5ebd1e120234501da0f0cde3d8f1cce024155",
}
FIT_SETTINGS = {
    "seed": 42,
    "batch_size": 64,
    "max_epochs": 50,
    "patience": 8,
    "learning_rate": 0.0003,
    "optimizer": "Adam",
    "dropout": 0.1,
    "selection": "max_validation_roc_auc_first_tie",
    "decision_threshold": 0.5,
    "sizes": [1024, 1024, 512, 512, 1],
}
FINGERPRINT = {"family": "Morgan", "radius": 2, "bits": 1024, "explicit_hydrogens": True, "chirality": False}
_COUNT_KEYS = (
    "n_train",
    "n_validation",
    "n_test",
    "test_positives",
    "test_negatives",
    "tp",
    "tn",
    "fp",
    "fn",
    "selected_epoch",
    "test_fingerprint_seen_in_train",
)
_METRIC_KEYS = ("roc_auc", "accuracy", "validation_roc_auc")
_MAX_MODEL_BYTES = 12 * 1024 * 1024


class AmesModelError(ValueError):
    """A named blocker for this local fit or its applicability."""


def release_directory() -> Path:
    return Path(sys.prefix) / "share" / METHOD_ID


def validate_validation(value: object) -> dict:
    if (
        not isinstance(value, dict)
        or set(value) != {*_COUNT_KEYS, *_METRIC_KEYS}
        or any(type(value[key]) is not int or not 0 <= value[key] <= 100_000 for key in _COUNT_KEYS)
        or any(type(value[key]) not in (float, int) or not 0 <= value[key] <= 1 for key in _METRIC_KEYS)
    ):
        raise AmesModelError("toxscan_ames_invalid_metrics")
    if (
        min(value[k] for k in ("n_train", "n_validation", "n_test", "test_positives", "test_negatives")) <= 0
        or not 1 <= value["selected_epoch"] <= FIT_SETTINGS["max_epochs"]
        or value["test_positives"] + value["test_negatives"] != value["n_test"]
        or value["test_fingerprint_seen_in_train"] > value["n_test"]
        or value["tp"] + value["fn"] != value["test_positives"]
        or value["tn"] + value["fp"] != value["test_negatives"]
        or not math.isclose(value["accuracy"], (value["tp"] + value["tn"]) / value["n_test"], rel_tol=0, abs_tol=1e-12)
    ):
        raise AmesModelError("toxscan_ames_invalid_metrics")
    return {**{k: value[k] for k in _COUNT_KEYS}, **{k: float(value[k]) for k in _METRIC_KEYS}}


@lru_cache(maxsize=1)
def _generator() -> FingerprintGenerator64:
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(
        radius=FINGERPRINT["radius"],
        fpSize=FINGERPRINT["bits"],
        includeChirality=FINGERPRINT["chirality"],
    )


def fingerprint(identity: str) -> np.ndarray:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(identity)
    if molecule is None:
        raise AmesModelError("toxscan_ames_invalid_structure")
    if FINGERPRINT["explicit_hydrogens"]:
        molecule = Chem.AddHs(molecule)
    return _generator().GetFingerprintAsNumPy(molecule)


def fingerprint_digest(vector: np.ndarray) -> bytes:
    return hashlib.sha256(np.packbits(vector, bitorder="big").tobytes()).digest()


@lru_cache(maxsize=1)
def verify_representation() -> bool:
    # These are bit-vector checksums from the released LMDB, including a chiral control.
    controls = {
        "ClCCOCCCl": "990ca582b90729fd6fccfc0b594f86aff92c21c215d02a6877fcc820d9f31785",
        "C[C@@H](Oc1ccc(Cl)cc1Cl)C(=O)O": "7d37577a548ca02bbeb45a2bd4e5e616136556709c818f9ad412ca531d691eab",
    }
    for smiles, expected in controls.items():
        bits = fingerprint(smiles)
        if hashlib.sha256((bits + ord("0")).tobytes()).hexdigest() != expected:
            raise AmesModelError("toxscan_ames_runtime_unavailable")
    return True


def evidence_digest(evidence: dict) -> bytes:
    settings = {key: val for key, val in evidence.items() if key != "model_sha256"}
    data = json.dumps(settings, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return hashlib.sha256(data).digest()


def _unique_meta(items: list[tuple]) -> dict:
    out = dict(items)
    if len(out) != len(items):
        raise ValueError("duplicate field in the local validation receipt")
    return out


def _verify_test_scores(scores: np.ndarray, labels: np.ndarray, validation: dict) -> None:
    count = validation["n_test"]
    if (
        scores.dtype != np.float64
        or labels.dtype != np.uint8
        or scores.shape != (count,)
        or labels.shape != (count,)
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
        or np.any(labels > 1)
    ):
        raise ValueError("invalid held-out observations")
    positives = scores[labels == 1]
    negatives = np.sort(scores[labels == 0])
    if len(positives) != validation["test_positives"] or len(negatives) != validation["test_negatives"]:
        raise ValueError("held-out class denominators disagree")
    # Count wins and half-credit ties over positive/negative pairs without a square matrix.
    wins = np.searchsorted(negatives, positives, side="left") + np.searchsorted(negatives, positives, side="right")
    auc = float(np.sum(wins)) / (2 * len(positives) * len(negatives))
    if (
        int(np.count_nonzero(positives >= 0.5)) != validation["tp"]
        or int(np.count_nonzero(negatives < 0.5)) != validation["tn"]
        or not math.isclose(auc, validation["roc_auc"], rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError("held-out scores disagree with the recorded performance")


@dataclass(frozen=True)
class Fit:
    tensors: dict
    membership: dict
    fingerprints: frozenset
    validation: dict
    sha256: str


def verify_bundle(directory: Path) -> Fit:
    try:
        from safetensors import SafetensorError
        from safetensors.numpy import load
        from scipy.special import expit, ndtr  # noqa: F401

        verify_representation()
    except (ImportError, OSError) as error:
        raise AmesModelError("toxscan_ames_runtime_unavailable") from error
    try:
        meta = json.loads(read_regular(directory / RECEIPT, 32_768, "fit", ValueError), object_pairs_hook=_unique_meta)
        raw = read_regular(directory / WEIGHTS, _MAX_MODEL_BYTES, "fit", ValueError)
        digest = hashlib.sha256(raw).hexdigest()
        if (
            not isinstance(meta, dict)
            or type(meta.get("version")) is not int
            or meta["version"] != 1
            or meta.get("method") != METHOD_ID
            or meta.get("model_sha256") != digest
            or meta.get("archive_sha256") != ARCHIVE_SHA256
            or meta.get("source_revision") != SOURCE_REVISION
            or meta.get("tables_sha256") != SOURCE_SHA256
            or meta.get("fit_settings") != FIT_SETTINGS
            or meta.get("fingerprint") != FINGERPRINT
        ):
            raise ValueError("fit attribution mismatch")
        tensors = load(raw)
        widths = FIT_SETTINGS["sizes"]
        shapes = {f"w{i}": (widths[i + 1], widths[i]) for i in range(4)}
        shapes.update({f"b{i}": (widths[i + 1],) for i in range(4)})
        if set(tensors) != {*shapes, *SPLITS, "training_fingerprints", "evidence_digest", "test_labels", "test_scores"}:
            raise ValueError("unrecognized tensor layout")
        attestation = tensors.pop("evidence_digest")
        if (
            attestation.dtype != np.uint8
            or attestation.shape != (32,)
            or attestation.tobytes() != evidence_digest(meta)
        ):
            raise ValueError("evaluation receipt does not belong to these weights")
        # The initial fit recorded representation overlap in its digest-bound split audit.
        # Project that count into all reports, including reports from the original fit.
        seen = meta["split_audit"]["test"]["graphs_with_fingerprint_seen_in_train"]
        validation = validate_validation({"test_fingerprint_seen_in_train": seen, **meta["validation"]})
        if type(seen) is not int or seen != validation["test_fingerprint_seen_in_train"]:
            raise ValueError("test fingerprint overlap disagrees with the split audit")
        _verify_test_scores(tensors.pop("test_scores"), tensors.pop("test_labels"), validation)
        for key, shape in shapes.items():
            array = tensors[key]
            if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
                raise ValueError("invalid weights")
        membership = {}
        known = {}
        counts = {"train": validation["n_train"], "valid": validation["n_validation"], "test": validation["n_test"]}
        counts["training_fingerprints"] = validation["n_train"]
        for key, count in counts.items():
            array = tensors.pop(key)
            if array.dtype != np.uint8 or array.shape != (count, 32):
                raise ValueError("invalid membership array")
            known[key] = frozenset(row.tobytes() for row in array)
            if key in SPLITS:
                if len(known[key]) != count or membership.keys() & known[key]:
                    raise ValueError("evaluation partitions overlap")
                membership.update(dict.fromkeys(known[key], "validation" if key == "valid" else key))
        return Fit(tensors, membership, known["training_fingerprints"], validation, digest)
    except (ValueError, KeyError, TypeError, RecursionError, OSError, SafetensorError) as error:
        raise AmesModelError("toxscan_ames_model_integrity") from error


@lru_cache(maxsize=2)
def _cached_model(directory: Path, signature: tuple) -> Fit:
    return verify_bundle(directory)


def _model() -> Fit:
    directory = release_directory()
    try:
        paths = (directory / RECEIPT, directory / WEIGHTS)
        infos = [path.lstat() for path in paths]
        if not all(stat.S_ISREG(info.st_mode) for info in infos):
            raise AmesModelError("toxscan_ames_model_integrity")
        signature = tuple((s.st_dev, s.st_ino, s.st_size, s.st_ctime_ns, s.st_mtime_ns) for s in infos)
        return _cached_model(directory, signature)
    except FileNotFoundError as error:
        raise AmesModelError("toxscan_ames_model_missing") from error
    except OSError as error:
        raise AmesModelError("toxscan_ames_model_integrity") from error


def benchmark() -> dict | None:
    try:
        return dict(_model().validation)
    except AmesModelError:
        return None


def installed() -> bool:
    return benchmark() is not None


def installed_sha256() -> str | None:
    """Identify the verified current local fit for readback of an earlier report."""
    try:
        return _model().sha256
    except AmesModelError:
        return None


def score_vector(vector: np.ndarray, tensors: dict) -> float:
    from scipy.special import expit, ndtr

    first = tensors["w0"] @ vector.astype(np.float32) + tensors["b0"]
    projected = tensors["w1"] @ (first * ndtr(first)) + tensors["b1"]
    hidden = np.tanh(tensors["w2"] @ projected + tensors["b2"])
    score = float(expit((tensors["w3"] @ hidden + tensors["b3"]).item()))
    if not math.isfinite(score):
        raise AmesModelError("toxscan_ames_model_integrity")
    return score


def _candidate_features(smiles: str) -> tuple[str, np.ndarray]:
    try:
        identity = toxscan_inputs.canonical(smiles)
        return identity, fingerprint(identity)
    except (ImportError, OSError) as error:
        raise AmesModelError("toxscan_ames_runtime_unavailable") from error
    except toxscan_inputs.ToxScanError as error:
        blocker = (
            "toxscan_ames_runtime_unavailable"
            if str(error) == "toxscan_rdkit_unavailable"
            else "toxscan_ames_invalid_structure"
        )
        raise AmesModelError(blocker) from error


def fingerprint_seen_in_training(smiles: str) -> bool:
    """Check representation exposure without scoring the candidate."""
    _, vector = _candidate_features(smiles)
    return fingerprint_digest(vector) in _model().fingerprints


def predict(smiles: str) -> dict:
    identity, vector = _candidate_features(smiles)
    model = _model()
    score = score_vector(vector, model.tensors)
    graph = hashlib.sha256(identity.encode("ascii")).digest()
    return {
        "score": score,
        "class_call": "YES" if score >= 0.5 else "NO",
        "model_sha256": model.sha256,
        "archive_sha256": ARCHIVE_SHA256,
        "validation": dict(model.validation),
        "graph_membership": model.membership.get(graph, "not_in_retained_splits"),
        "fingerprint_seen_in_training": fingerprint_digest(vector) in model.fingerprints,
    }
