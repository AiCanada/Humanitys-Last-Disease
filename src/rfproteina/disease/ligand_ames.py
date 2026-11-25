"""Advisory bacterial mutagenicity predictions for nominated small molecules."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Sequence
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from rfproteina.disease import ligand_admet as admet

if TYPE_CHECKING:
    from admet_ai import ADMETModel
    from pandas import DataFrame
    from rdkit import Chem

METHOD_ID = "ligand_ames"
METHOD = "admet_ai_ames_v2"
ENDPOINT = "AMES"
UNIT = "probability"
BATCH_SIZE = admet.BATCH_SIZE
MAX_INPUT_BYTES = admet.MAX_INPUT_BYTES
MAX_CANDIDATES = admet.MAX_CANDIDATES
_MAX_RESPONSE_BYTES = admet.MAX_RESPONSE_BYTES

AmesError = admet.AdmetError
valid_candidate_id = admet.valid_candidate_id
SPEC = admet.Endpoint(
    METHOD, ENDPOINT, UNIT, "predicted_ames_probability", "classification", "-", 1.0,
    structure_blocker="unsupported_ames_structure",
    input_blocker="ames_input_limit_exceeded",
    resource_blocker="ames_resource_limit_exceeded",
)


def verified_assets(ensemble_dir: Path | None = None) -> tuple[metadata.Distribution, list[Path], dict[str, str]]:
    """Check the local endpoint definition and the five released ensemble members."""
    return admet.verified_assets(SPEC, ensemble_dir)


def _values(frame: DataFrame, smiles: list[str]) -> dict[str, float | str]:
    return admet.values(SPEC, frame, smiles)


def _score_batch(model: ADMETModel, batch: list[str]) -> dict[str, float | str]:
    return admet.score_batch(SPEC, model, batch)


def _local_predictions(smiles: list[str]) -> dict:
    return admet.local_predictions(__name__, smiles, run_process=subprocess.run)


def molecular_identity(smiles: str, chem: ModuleType) -> tuple[Chem.Mol, str]:
    """Preserve the submitted charge, connectivity and stereochemistry."""
    return admet.molecular_identity(SPEC, smiles, chem)


def profile(candidates: Iterable[Sequence[str]], *, scope: str) -> dict:
    """Return an advisory, compound-level prediction and nomination coverage."""
    return admet.profile(SPEC, candidates, scope=scope, predict=_local_predictions)


def screen_inputs(directory: Path, candidate: str | None) -> tuple[str, list[Sequence[str]], str | None]:
    """Read the bounded nomination source used by both plan and execution."""
    return admet.screen_inputs(SPEC, directory, candidate)


def screen_run(directory: Path, *, candidate: str | None = None) -> dict:
    """Run the endpoint for a finished run or one explicitly named candidate."""
    return admet.screen_run(SPEC, directory, candidate=candidate, predict=_local_predictions)


if __name__ == "__main__":
    admet.run_worker(SPEC)
