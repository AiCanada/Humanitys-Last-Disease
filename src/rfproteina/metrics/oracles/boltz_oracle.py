"""Boltz-2 refolding oracle: a free, open structure + affinity predictor.

Boltz (https://github.com/jwohlwend/boltz) is released under the MIT License,
verified on 2026-11-16 from the repository's own ``LICENSE``. Boltz-2 adds a
binding-affinity head on top of AlphaFold3-class structure prediction.

This adapter uses Boltz for *structure refolding* through the existing
:class:`~rfproteina.metrics.oracles.oracle_metric.OracleMetric` contract (select
it via ``refolding_oracles/boltz2.yaml``). It deliberately does **not** fold the
affinity head into the refold confidences: Boltz's ``affinity_pred_value`` is
``log10(IC50)`` with IC50 in micromolar (verified from ``docs/prediction.md``),
which is a different quantity and a different unit from anything the screening
layer consumes in nanomolar. Mixing it into an nM panel is exactly the
units-error the project's screening handoff warns about, so :func:`parse_affinity`
is provided as a clearly-labelled, separate reader and is not wired into the
refold metrics.

Field provenance (verified from Boltz ``main`` / ``docs/prediction.md`` on 2026-11-16)
-------------------------------------------------------------------------------------
* CLI: ``boltz predict <input.yaml> --out_dir <dir> --output_format mmcif``.
* Per-sample outputs under ``<out>/predictions/<stem>/``:
  ``<stem>_model_0.cif`` (best sample) and
  ``confidence_<stem>_model_0.json`` with ``confidence_score``, ``ptm``,
  ``iptm``, ``complex_plddt``, ``chains_ptm`` and ``pair_chains_iptm``. All in
  0..1 (pde is in angstroms). :func:`extract_confidence_metrics` rescales.
* Optional ``affinity_<stem>.json`` with ``affinity_pred_value`` (log10 IC50 in
  micromolar) and ``affinity_probability_binary`` (0..1).

Nothing here downloads weights at import; the ``boltz`` CLI fetches them on first
run. The model is large and GPU-bound in practice; tests mock the fold.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

from rfproteina.metrics.oracles.free_oracle_common import (
    PredictOutput,
    SequenceRefoldOracle,
)
from rfproteina.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def _chain_key(mapping: dict, chain_id: str, index: int) -> object | None:
    """Boltz keys per-chain dicts by the YAML ``id`` or by positional index.

    Try the chain id first, then its string index, so the reader works whichever
    convention the installed Boltz version uses.
    """
    if chain_id in mapping:
        return mapping[chain_id]
    return mapping.get(str(index))


class Boltz2Oracle(SequenceRefoldOracle):
    """Refold a design with Boltz-2 (MIT, open weights).

    Args:
        oracle_name: Output-filename label (default ``"boltz2"``).
        executable: Name/path of the Boltz CLI (default ``"boltz"``).
        diffusion_samples: Number of diffusion samples (``--diffusion_samples``).
        use_msa_server: Pass ``--use_msa_server`` so Boltz builds MSAs itself.
            Off by default: the server is an external dependency the caller opts
            into, not a silent network call.
        extra_args: Extra CLI tokens appended verbatim.
        **kwargs: Forwarded to :class:`OracleMetric`.
    """

    def __init__(
        self,
        oracle_name: str = "boltz2",
        *,
        executable: str = "boltz",
        diffusion_samples: int = 1,
        use_msa_server: bool = False,
        extra_args: tuple[str, ...] = (),
        **kwargs: object,
    ) -> None:
        super().__init__(oracle_name=oracle_name, **kwargs)  # type: ignore[arg-type]
        self.executable = executable
        self.diffusion_samples = diffusion_samples
        self.use_msa_server = use_msa_server
        self.extra_args = tuple(extra_args)

    def _backend_available(self) -> bool:
        return importlib.util.find_spec("boltz") is not None or shutil.which(self.executable) is not None

    @staticmethod
    def to_yaml(sequences: dict[str, str]) -> str:
        """Render per-chain protein sequences as a Boltz v1 input YAML.

        Written by hand (no PyYAML dependency added) in Boltz's documented schema.
        """
        lines = ["version: 1", "sequences:"]
        for chain, seq in sequences.items():
            lines += ["  - protein:", f"      id: {chain}", f'      sequence: "{seq}"']
        return "\n".join(lines) + "\n"

    def _predict(self, sequences: dict[str, str], work_dir: Path) -> PredictOutput:
        import subprocess

        stem = "design"
        input_yaml = work_dir / f"{stem}.yaml"
        input_yaml.write_text(self.to_yaml(sequences))
        out_dir = work_dir / "boltz_out"

        cmd = [
            self.executable,
            "predict",
            str(input_yaml),
            "--out_dir",
            str(out_dir),
            "--output_format",
            "mmcif",
            "--diffusion_samples",
            str(self.diffusion_samples),
        ]
        if self.use_msa_server:
            cmd.append("--use_msa_server")
        cmd.extend(self.extra_args)
        logger.info(f"{self.oracle_name}: running {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        predictions_dir = out_dir / "predictions" / stem
        return self.build_output(predictions_dir, stem, list(sequences.keys()))

    @staticmethod
    def build_output(predictions_dir: Path, stem: str, chain_ids: list[str]) -> PredictOutput:
        """Parse Boltz's per-sample outputs into a :class:`PredictOutput`.

        Reads ``<stem>_model_0.cif`` (the top sample) and
        ``confidence_<stem>_model_0.json``, mapping Boltz's per-chain dicts onto
        AF3-shaped ordered lists/matrix.
        """
        predictions_dir = Path(predictions_dir)
        cif_path = predictions_dir / f"{stem}_model_0.cif"
        if not cif_path.exists():
            raise FileNotFoundError(f"Boltz produced no top-model CIF at {cif_path}")
        confidence_path = predictions_dir / f"confidence_{stem}_model_0.json"
        if not confidence_path.exists():
            raise FileNotFoundError(f"Boltz produced no confidence JSON at {confidence_path}")
        conf = json.loads(confidence_path.read_text())

        chains_ptm = conf.get("chains_ptm", {}) or {}
        pair_iptm = conf.get("pair_chains_iptm", {}) or {}
        chain_ptm_list = [
            float(_chain_key(chains_ptm, chain, i) or 0.0) for i, chain in enumerate(chain_ids)
        ]
        matrix: list[list[float | None]] = []
        for i, chain_i in enumerate(chain_ids):
            row_src = _chain_key(pair_iptm, chain_i, i) or {}
            row: list[float | None] = []
            for j, chain_j in enumerate(chain_ids):
                row.append(None if i == j else float(_chain_key(row_src, chain_j, j) or 0.0))
            matrix.append(row)

        summary: dict[str, object] = {
            "overall_plddt": float(conf.get("complex_plddt", 0.0)),
            "ptm": float(conf.get("ptm", 0.0)),
            "iptm": float(conf.get("iptm", 0.0)),
            "ranking_score": float(conf.get("confidence_score", 0.0)),
            "chain_ptm": chain_ptm_list,
            "chain_pair_iptm": matrix,
            "chain_ids": chain_ids,
        }
        return PredictOutput(cif_path=cif_path, summary_confidences=summary)


def parse_affinity(predictions_dir: Path, stem: str) -> dict[str, float] | None:
    """Read Boltz's optional affinity output, if present. NOT a nanomolar value.

    Returns ``{"affinity_log10_ic50_um", "affinity_binder_probability"}`` or
    ``None`` when no affinity file exists. ``affinity_log10_ic50_um`` is
    ``log10(IC50)`` with IC50 in micromolar (Boltz convention, lower = stronger);
    it must never be dropped into a nanomolar panel. Boltz also documents a
    conversion to pIC50 in kcal/mol: ``(6 - y) * 1.364``.
    """
    affinity_path = Path(predictions_dir) / f"affinity_{stem}.json"
    if not affinity_path.exists():
        return None
    data = json.loads(affinity_path.read_text())
    return {
        "affinity_log10_ic50_um": float(data["affinity_pred_value"]),
        "affinity_binder_probability": float(data["affinity_probability_binary"]),
    }
