"""Developability filters: GRAVY, instability index, SAP, and exposed hydrophobics.

Sequence metrics (GRAVY, Guruprasad instability index) are computed in-process from
the designed-chain sequence. Structure metrics (Rosetta SAP score and
ExposedHydrophobics) run via a nested PyRosetta apptainer, following the same pattern
as HBDesigner PyRosetta scoring.
"""

import json
import os
import shutil
import subprocess
import tempfile
from functools import cached_property
from pathlib import Path
from typing import Any

from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.selection import get_residue_starts
from atomworks.io.utils.sequence import aa_chem_comp_3to1
from biotite.structure import AtomArray

from rfproteina.metrics.metric import Metric
from rfproteina.utils.chain import get_chain_of_interest
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.sequence_properties import (
    _DIWV as _DIWV,
)
from rfproteina.utils.sequence_properties import (
    _KYTE_DOOLITTLE as _KYTE_DOOLITTLE,
)
from rfproteina.utils.sequence_properties import (
    compute_gravy,
    compute_instability_index,
)
from rfproteina.utils.structure_export import write_input_structure_cif

logger = RankedLogger(__name__, rank_zero_only=True)

_AA3_TO_1 = aa_chem_comp_3to1(standard_only=True)

def _chain_attr(atom_array: AtomArray) -> str:
    return "chain_iid" if "chain_iid" in atom_array.get_annotation_categories() else "chain_id"


def _resolve_chain_of_interest(atom_array: AtomArray, chain_of_interest: str | None) -> str | None:
    if chain_of_interest is not None:
        return chain_of_interest
    coi, _ = get_chain_of_interest(atom_array)
    return coi


def subset_to_designed_protein(
    atom_array: AtomArray,
    chain_of_interest: str | None = None,
) -> AtomArray:
    """Return protein atoms on the designed chain (chain of interest).

    Args:
        atom_array: Full generated structure.
        chain_of_interest: Optional chain id/iid; inferred if omitted.

    Returns:
        Protein-only AtomArray for the designed chain. Empty if none found.
    """
    if atom_array.array_length() == 0:
        return atom_array
    ensure_annotations(atom_array, "is_protein")
    protein = atom_array[atom_array.is_protein]
    if protein.array_length() == 0:
        return protein
    coi = _resolve_chain_of_interest(protein, chain_of_interest)
    if coi is None:
        return protein
    attr = _chain_attr(protein)
    return protein[protein.get_annotation(attr) == coi]


def protein_one_letter_sequence(atom_array: AtomArray) -> str:
    """Extract the one-letter sequence, skipping non-standard residues."""
    if atom_array.array_length() == 0:
        return ""
    res_names = atom_array.res_name[get_residue_starts(atom_array)]
    return "".join(_AA3_TO_1[name] for name in res_names if name in _AA3_TO_1)


class SequenceDevelopabilityMetric(Metric):
    """Sequence-only developability scores for the designed chain.

    Reports:
        - ``gravy``: Kyte-Doolittle grand average of hydropathicity
        - ``instability_index``: Guruprasad instability index (ProtParam; <40 ~ stable)
    """

    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted_atom_array": "predicted_atom_array",
            "chain_of_interest": ("extra_info", "chain_of_interest"),
        }

    @property
    def optional_kwargs(self) -> frozenset[str]:
        return frozenset(["chain_of_interest"])

    def compute(
        self,
        predicted_atom_array: AtomArray,
        chain_of_interest: str | None = None,
    ) -> dict[str, float]:
        designed = subset_to_designed_protein(predicted_atom_array, chain_of_interest)
        sequence = protein_one_letter_sequence(designed)
        if not sequence:
            return {"gravy": float("nan"), "instability_index": float("nan")}
        return {
            "gravy": compute_gravy(sequence),
            "instability_index": compute_instability_index(sequence),
        }


class RosettaDevelopabilityMetric(Metric):
    """Rosetta SAP score and ExposedHydrophobics via a nested PyRosetta apptainer.

    Reports:
        - ``sap_score``: whole-pose SAP (``calculate_sap``)
        - ``hydrophobic_exposure_sasa``: Rosetta ExposedHydrophobics filter score
          (sum of ``max(0, SASA - sasa_cutoff)`` over hydrophobic residues)

    Both are computed on the designed protein chain only.
    """

    def __init__(
        self,
        container: str | None = None,
        score_script: str | None = None,
        apptainer_bin: str | None = None,
        sasa_cutoff: float = 20.0,
        timeout_s: float = 180.0,
        binds: list[str] | None = None,
        scratch_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        container = container or os.environ.get("PYROSETTA_CONTAINER")
        self.container = Path(container) if container else None
        default_script = Path(__file__).with_name("score_pose_rosetta.py")
        self.score_script = Path(score_script or default_script).resolve()
        self.apptainer_bin = apptainer_bin or shutil.which("apptainer") or "apptainer"
        self.sasa_cutoff = float(sasa_cutoff)
        self.timeout_s = float(timeout_s)
        # Extra binds come from the project-wide APPTAINER_BIND_PATHS (set per-cluster in
        # .ipd/.env.<cluster>) -- no cluster paths are hardcoded here.
        if binds is None:
            raw = os.environ.get("APPTAINER_BIND_PATHS", "")
            binds = [spec.strip() for spec in raw.split(";") if spec.strip()]
        self.binds = list(binds)
        self.scratch_dir = Path(
            scratch_dir
            or os.environ.get(
                "DEVELOPABILITY_METRIC_SCRATCH",
                os.environ.get("TMPDIR", "/tmp"),
            )
        )
        # Bind the installed helper explicitly; site-packages may be outside default container mounts.
        self.binds.append(f"{self.score_script.parent}:{self.score_script.parent}:ro")
        # The CIF/JSON handoff lives under scratch_dir, so it must be visible inside the container.
        if str(self.scratch_dir) not in self.binds:
            self.binds.append(str(self.scratch_dir))
        super().__init__(**kwargs)

    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted_atom_array": "predicted_atom_array",
            "chain_of_interest": ("extra_info", "chain_of_interest"),
        }

    @property
    def optional_kwargs(self) -> frozenset[str]:
        return frozenset(["chain_of_interest"])

    def compute(
        self,
        predicted_atom_array: AtomArray,
        chain_of_interest: str | None = None,
    ) -> dict[str, float]:
        if self.container is None:
            raise OSError("PyRosetta container not set: pass container=... or set $PYROSETTA_CONTAINER.")
        if not self.container.is_file():
            raise FileNotFoundError(f"PyRosetta container missing: {self.container}")
        if not self.score_script.is_file():
            raise FileNotFoundError(f"Rosetta score script missing: {self.score_script}")

        designed = subset_to_designed_protein(predicted_atom_array, chain_of_interest)
        if designed.array_length() == 0:
            return {"sap_score": float("nan"), "hydrophobic_exposure_sasa": float("nan")}

        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="rfd4_developability_", dir=self.scratch_dir) as tmp:
            cif_path = Path(tmp) / "design.cif"
            out_json = Path(tmp) / "score.json"
            write_input_structure_cif(designed, cif_path)
            command = [
                self.apptainer_bin,
                "exec",
                *[arg for bind in self.binds for arg in ("--bind", bind)],
                str(self.container),
                "python",
                str(self.score_script),
                "--cif",
                str(cif_path),
                "--sasa_cutoff",
                str(self.sasa_cutoff),
                "--output_json",
                str(out_json),
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
            except subprocess.TimeoutExpired:
                logger.warning(f"Rosetta developability score timed out after {self.timeout_s}s; returning nan.")
                return {"sap_score": float("nan"), "hydrophobic_exposure_sasa": float("nan")}
            if out_json.is_file():
                payload = json.loads(out_json.read_text(encoding="utf-8"))
            else:
                try:
                    payload = json.loads(completed.stdout.strip().splitlines()[-1])
                except Exception as exc:
                    raise RuntimeError(
                        "Rosetta developability subprocess failed without JSON output: "
                        f"returncode={completed.returncode} stderr={completed.stderr[-2000:]}"
                    ) from exc
            if completed.returncode != 0 or "error" in payload:
                raise RuntimeError(
                    f"Rosetta developability score failed (rc={completed.returncode}): "
                    f"{payload.get('error') or completed.stderr[-2000:]}"
                )
            return {
                "sap_score": float(payload["sap_score"]),
                "hydrophobic_exposure_sasa": float(payload["hydrophobic_exposure_sasa"]),
            }
