"""Advisory ADMET-AI DILI label predictions for nominated small molecules."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Sequence
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from rfproteina.disease import ligand_admet as admet

if TYPE_CHECKING:
    from rdkit import Chem

METHOD_ID = "ligand_dili"
METHOD = "admet_ai_dili_v2"
ENDPOINT = "DILI"
UNIT = "probability"
VALUE_FIELD = "predicted_dili_probability"
BATCH_SIZE = admet.BATCH_SIZE

DiliError = admet.AdmetError
valid_candidate_id = admet.valid_candidate_id
SPEC = admet.Endpoint(
    method=METHOD,
    endpoint=ENDPOINT,
    unit=UNIT,
    value_field=VALUE_FIELD,
    task_type="classification",
    manifest_unit="-",
    maximum=1.0,
    structure_blocker="dili_structure_unsupported",
    input_blocker="dili_inputs_over_limit",
    resource_blocker="dili_resources_unavailable",
    species="human",
)


def verified_assets() -> tuple[metadata.Distribution, list[Path], dict[str, str]]:
    """Check the installed human DILI head and its classification ensemble."""
    return admet.verified_assets(SPEC)


def _local_predictions(smiles: list[str]) -> dict:
    return admet.local_predictions(__name__, smiles, run_process=subprocess.run)


def molecular_identity(smiles: str, chem: ModuleType) -> tuple[Chem.Mol, str]:
    """Preserve the submitted structure, charge and stereochemistry."""
    return admet.molecular_identity(SPEC, smiles, chem)


def profile(candidates: Iterable[Sequence[str]], *, scope: str) -> dict:
    """Return model label scores and coverage, without a clinical safety call."""
    return admet.profile(SPEC, candidates, scope=scope, predict=_local_predictions)


def screen_inputs(directory: Path, candidate: str | None) -> tuple[str, list[Sequence[str]], str | None]:
    """Read the bounded nomination source used by both planning and execution."""
    return admet.screen_inputs(SPEC, directory, candidate)


def screen_run(directory: Path, *, candidate: str | None = None) -> dict:
    """Screen a finished run's nominations or one explicitly named candidate."""
    return admet.screen_run(SPEC, directory, candidate=candidate, predict=_local_predictions)


if __name__ == "__main__":
    admet.run_worker(SPEC)
