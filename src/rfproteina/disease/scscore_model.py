"""CPU inference for Coley et al.'s released full-Reaxys 1024-bit Boolean SCScore model."""

from __future__ import annotations

import gzip
import hashlib
import json
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rfproteina.disease import ligand_inputs
from rfproteina.disease.screening_tools import data_dir

if TYPE_CHECKING:
    from rdkit import Chem

MODEL = "full_reaxys_model_1024bool"
REVISION = "37090a6aa8220408f572de20224c33c221312d16"
FILENAME = "model.ckpt-10654.as_numpy.json.gz"
URL = f"https://raw.githubusercontent.com/connorcoley/scscore/{REVISION}/models/{MODEL}/{FILENAME}"
SHA256 = "fa90bd552307b2efd05eb1e53033bc9eadadfd5bdca682e541040982af99536b"
WEIGHT_BYTES = 6_178_760
_WIDTHS = (1024, 300, 300, 300, 300, 300, 1)


class SCScoreError(ValueError):
    """An attributable input or provider blocker."""


def weights_path() -> Path:
    return data_dir("scscore", FILENAME)


def read_regular(path: Path, limit: int, role: str) -> bytes:
    return ligand_inputs.read_regular(path, limit, role, SCScoreError)


def verify_weights(data: bytes) -> bytes:
    if len(data) != WEIGHT_BYTES or hashlib.sha256(data).hexdigest() != SHA256:
        raise SCScoreError("scscore_weights_digest_mismatch")
    return data


def _weight_bytes() -> bytes:
    return verify_weights(read_regular(weights_path(), WEIGHT_BYTES, "scscore_weights"))


def installed() -> bool:
    try:
        from rdkit.Chem import rdFingerprintGenerator

        _weight_bytes()
    except (ImportError, OSError, SCScoreError):
        return False
    return callable(getattr(rdFingerprintGenerator, "GetMorganGenerator", None))


@lru_cache(maxsize=1)
def _layers(verified_bytes: bytes) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    # The caller has authenticated these exact compressed bytes before decompression.
    arrays = json.loads(gzip.decompress(verified_bytes))
    layers = []
    for (n_in, n_out), weights, bias in zip(pairwise(_WIDTHS), arrays[::2], arrays[1::2], strict=True):
        matrix, offset = np.asarray(weights, dtype=np.float64), np.asarray(bias, dtype=np.float64)
        if matrix.shape != (n_in, n_out) or offset.shape != (n_out,):
            raise SCScoreError("invalid_scscore_layer_shape")
        matrix.setflags(write=False)
        offset.setflags(write=False)
        layers.append((matrix, offset))
    return tuple(layers)


def score(molecules: dict[str, Chem.Mol]) -> tuple[dict[str, float | str], dict]:
    """Fingerprint each distinct nominated form once, then run the released network in batch."""
    from rdkit import rdBase
    from rdkit.Chem import rdFingerprintGenerator

    layers = _layers(_weight_bytes())
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=1024, includeChirality=True, countSimulation=False
    )
    inputs = {}
    values = {}
    for identity, molecule in molecules.items():
        try:
            bits = generator.GetFingerprintAsNumPy(molecule)
            if bits.shape != (1024,) or not bits.any():
                raise ValueError("empty_fingerprint")
        except (RuntimeError, ValueError, OverflowError):
            values[identity] = "scscore_fingerprint_unavailable"
        else:
            inputs[identity] = bits
    if inputs:
        x = np.stack(list(inputs.values())).astype(np.float64)
        for matrix, offset in layers[:-1]:
            x = np.maximum(x @ matrix + offset, 0)
        matrix, offset = layers[-1]
        logits = (x @ matrix + offset)[:, 0]
        # Stable form of the published 1 + 4 * sigmoid(logit).
        predictions = 1.0 + 4.0 * np.exp(-np.logaddexp(0.0, -logits))
        for identity, logit, value in zip(inputs, logits, predictions, strict=True):
            values[identity] = float(value) if np.isfinite(logit) and 1 <= value <= 5 else "invalid_scscore_result"
    return values, {
        "model": MODEL,
        "revision": REVISION,
        "weights_sha256": SHA256,
        "device": "cpu",
        "rdkit": rdBase.rdkitVersion,
        "numpy": np.__version__,
        "fingerprint": {"type": "Morgan bit vector", "radius": 2, "bits": 1024, "chirality": True},
    }
