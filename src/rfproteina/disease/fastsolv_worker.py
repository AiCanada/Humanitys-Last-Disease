"""Compute the published FASTSOLV ensemble on CPU from verified checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from contextlib import redirect_stdout
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch

MAX_REQUESTS = 128
MAX_INPUT_BYTES = 128 * 1024
_MAX_SMILES = 512
_MAX_HEAVY_ATOMS = 80
_MORDRED_RELEASE = "2.0.7"
_FEATURES = 1_613
_FEATURE_SHA256 = "66b45a427709098addaf6e451e249f0a05eb986421a415bef5bcacec609873ec"
_HIDDEN = 3_000
_INPUTS = 2 * _FEATURES + 1
_ARCHITECTURE = {
    "num_layers": 2,
    "hidden_size": _HIDDEN,
    "num_features": _FEATURES,
    "activation_fxn": "leakyrelu",
    "input_activation": "clamp3",
}
_STATE_SHAPES = {
    "target_means": (1,),
    "target_vars": (1,),
    "solute_means": (_FEATURES,),
    "solute_vars": (_FEATURES,),
    "solvent_means": (_FEATURES,),
    "solvent_vars": (_FEATURES,),
    "temperature_means": (1,),
    "temperature_vars": (1,),
    "fnn.2.weight": (_HIDDEN, _INPUTS),
    "fnn.2.bias": (_HIDDEN,),
    "fnn.4.weight": (_HIDDEN, _HIDDEN),
    "fnn.4.bias": (_HIDDEN,),
    "fnn.6.weight": (1, _HIDDEN),
    "fnn.6.bias": (1,),
}


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate request field")
    return result


def _smiles(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= _MAX_SMILES
        and value.isascii()
        and value.isprintable()
        and not any(character.isspace() for character in value)
    )


def _requests(raw: bytes) -> list[tuple[str, str, float]]:
    if not 0 < len(raw) <= MAX_INPUT_BYTES:
        raise ValueError("invalid request size")
    entries = json.loads(raw.decode("ascii"), object_pairs_hook=_unique)
    if not isinstance(entries, list) or not 0 < len(entries) <= MAX_REQUESTS:
        raise ValueError("invalid request count")
    requests = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.keys() != {"solute_smiles", "solvent_smiles", "temperature_k"}:
            raise ValueError("invalid request fields")
        solute, solvent, temperature = (entry[field] for field in ("solute_smiles", "solvent_smiles", "temperature_k"))
        if (
            not _smiles(solute)
            or not _smiles(solvent)
            or type(temperature) not in {int, float}
            or not 200 <= temperature <= 500
        ):
            raise ValueError("invalid request values")
        requests.append((solute, solvent, float(temperature)))
    return requests


def _features(molecules: list[str]) -> dict[str, list[float]]:
    if version("mordredcommunity") != _MORDRED_RELEASE:
        raise ValueError("unsupported Mordred distribution")
    from mordred import Calculator, descriptors
    from rdkit import Chem, rdBase

    calculator = Calculator(descriptors, ignore_3D=True)
    names = tuple(map(str, calculator.descriptors))
    if len(names) != _FEATURES or hashlib.sha256("\n".join(names).encode()).hexdigest() != _FEATURE_SHA256:
        raise ValueError("unexpected Mordred feature order")
    data = {}
    with rdBase.BlockLogs():
        for smiles in molecules:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None or not 0 < molecule.GetNumHeavyAtoms() <= _MAX_HEAVY_ATOMS:
                raise ValueError("invalid molecular input")
            data[smiles] = [float(value) for value in calculator(molecule)]
    return data


def _load_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    import torch

    raw = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(raw, dict) or not isinstance(raw.get("hyper_parameters"), dict):
        raise ValueError("invalid model metadata")
    if any(raw["hyper_parameters"].get(name) != value for name, value in _ARCHITECTURE.items()):
        raise ValueError("unexpected model architecture")
    state = raw.get("state_dict")
    if not isinstance(state, dict) or state.keys() != _STATE_SHAPES.keys():
        raise ValueError("invalid model tensor names")
    for name, tensor in state.items():
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or tuple(tensor.shape) != _STATE_SHAPES[name]
            or not tensor.is_contiguous()
        ):
            raise ValueError("invalid model tensor")
    return state


def _scaled(values: torch.Tensor, means: torch.Tensor, variances: torch.Tensor) -> torch.Tensor:
    import torch

    means = torch.nan_to_num(means)
    values = torch.where(values.isnan(), means, values)
    return torch.nan_to_num((values - means) / variances.sqrt(), posinf=0.0, neginf=0.0)


def _predict_checkpoint(checkpoint: Path, inputs: tuple[torch.Tensor, ...]) -> np.ndarray:
    import torch
    from torch.nn import functional as nn

    state = _load_state(checkpoint)
    scaled = [
        _scaled(values, state[f"{role}_means"], state[f"{role}_vars"])
        for role, values in zip(("solute", "solvent", "temperature"), inputs, strict=True)
    ]
    values = torch.cat(scaled, dim=1).clamp(-3.0, 3.0)
    for layer in ("2", "4"):
        values = nn.leaky_relu(nn.linear(values, state[f"fnn.{layer}.weight"], state[f"fnn.{layer}.bias"]), 0.01)
    output = nn.linear(values, state["fnn.6.weight"], state["fnn.6.bias"])
    log_s = output * state["target_vars"].sqrt() + state["target_means"]
    return log_s.squeeze(1).numpy()


def _predict(requests: list[tuple[str, str, float]], checkpoints: tuple[Path, ...]) -> list[dict[str, float]]:
    import numpy as np
    import torch

    if len(checkpoints) != 4:
        raise ValueError("incomplete model ensemble")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    molecules = list(dict.fromkeys(smiles for request in requests for smiles in request[:2]))
    features = _features(molecules)
    solutes, solvents, temperatures = zip(*requests, strict=True)
    inputs = (
        torch.tensor([features[smiles] for smiles in solutes], dtype=torch.float32, device="cpu"),
        torch.tensor([features[smiles] for smiles in solvents], dtype=torch.float32, device="cpu"),
        torch.tensor(temperatures, dtype=torch.float32, device="cpu").unsqueeze(1),
    )
    with torch.inference_mode():
        predictions = np.stack([_predict_checkpoint(path, inputs) for path in checkpoints], axis=1)
    if predictions.shape != (len(requests), 4) or not np.isfinite(predictions).all():
        raise ValueError("invalid ensemble prediction")
    rows = [
        {"log_s": float(mean), "ensemble_sd": float(sd)}
        for mean, sd in zip(predictions.mean(axis=1), predictions.std(axis=1, ddof=0), strict=True)
    ]
    if any(not math.isfinite(value) for row in rows for value in row.values()):
        raise ValueError("invalid ensemble summary")
    return rows


def main() -> int:
    """Serve one bounded stdin request as a numeric JSON response."""
    from rfproteina.disease.fastsolv_runtime import verified_assets

    with redirect_stdout(sys.stderr):
        try:
            requests = _requests(sys.stdin.buffer.read(MAX_INPUT_BYTES + 1))
            rows = _predict(requests, verified_assets())
        except (ArithmeticError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            return 1
    json.dump(rows, sys.stdout, allow_nan=False, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
