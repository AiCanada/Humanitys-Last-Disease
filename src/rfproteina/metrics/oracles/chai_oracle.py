"""Chai-1 refolding oracle: a free, open-weights AlphaFold3-class structure predictor.

Chai-1 (https://github.com/chaidiscovery/chai-lab) is released under the Apache
2.0 License for **both code and model weights**, commercial use included. Verified
on 2026-11-16 from the repository's own ``LICENSE`` (Apache 2.0) and its README,
which states: "Chai-1 is released under an Apache 2.0 License (both code and model
weights), which means it can be used for both academic and commercial purposes,
including for drug discovery." This makes it the least licence-encumbered
AF3-class oracle available here, in contrast to AlphaFold 3's gated weights.

This adapter wires Chai-1 into the existing
:class:`~rfproteina.metrics.oracles.oracle_metric.OracleMetric` contract via
:class:`~rfproteina.metrics.oracles.free_oracle_common.SequenceRefoldOracle`, so a
user selects it exactly like the AF3 oracle (``refolding_oracles/chai1.yaml``).

Field provenance (verified from chai-lab ``main`` on 2026-11-16, not assumed)
----------------------------------------------------------------------------
* Inference entry point: ``chai_lab.chai1.run_inference(fasta_file, output_dir, ...)``
  returning ``StructureCandidates`` with ``cif_paths``, ``ranking_data`` and a
  per-token ``plddt`` tensor (``chai_lab/chai1.py``).
* Per-sample confidences (``chai_lab/ranking/rank.py::SampleRanking``):
  ``ptm_scores.complex_ptm`` -> ptm, ``ptm_scores.interface_ptm`` -> iptm,
  ``ptm_scores.per_chain_ptm``, ``ptm_scores.per_chain_pair_iptm``;
  ``plddt_scores.complex_plddt`` and ``plddt_scores.per_chain_plddt``
  (``chai_lab/ranking/plddt.py``). All are 0..1, which
  :func:`~rfproteina.metrics.oracles.oracle_metric.extract_confidence_metrics`
  rescales.
* FASTA format: ``>protein|name=<unique>`` headers (``examples/predict_structure.py``).

Nothing here downloads weights at import; ``run_inference`` fetches them on first
call into Chai's own cache (overridable with ``CHAI_DOWNLOADS_DIR``). The model
is several GB and requires a GPU in practice; this adapter neither hides that nor
runs it during tests (the fold is mocked).
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement

from rfproteina.metrics.oracles.free_oracle_common import (
    PredictOutput,
    SequenceRefoldOracle,
)
from rfproteina.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def _scalar(value: object) -> float:
    """Coerce a possibly-tensor, possibly-[1]-shaped confidence to a python float."""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return float(value.item())  # type: ignore[union-attr]
        except (ValueError, RuntimeError):
            pass
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        squeezed = value
        while hasattr(squeezed, "__len__") and len(squeezed) == 1:  # type: ignore[arg-type]
            squeezed = squeezed[0]  # type: ignore[index]
        return float(squeezed)  # type: ignore[arg-type]


def _vector(value: object) -> list[float]:
    """Coerce a 1-D (optionally [1, n]) confidence tensor/array to a list of floats."""
    tolist = getattr(value, "tolist", None)
    data = tolist() if callable(tolist) else list(value)  # type: ignore[arg-type]
    while isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
        data = data[0]
    return [float(x) for x in data]


def _matrix(value: object) -> list[list[float]]:
    """Coerce an n x n (optionally [1, n, n]) confidence tensor/array to a nested list."""
    tolist = getattr(value, "tolist", None)
    data = tolist() if callable(tolist) else list(value)  # type: ignore[arg-type]
    while (
        isinstance(data, list)
        and len(data) == 1
        and isinstance(data[0], list)
        and data
        and isinstance(data[0][0], list)
    ):
        data = data[0]
    return [[float(x) for x in row] for row in data]


class Chai1Oracle(SequenceRefoldOracle):
    """Refold a design with Chai-1 (Apache-2.0, open weights).

    Args:
        oracle_name: Output-filename label (default ``"chai1"``).
        num_trunk_recycles: Chai trunk recycles; default follows Chai's own default.
        num_diffn_timesteps: Chai diffusion steps; default follows Chai's own default.
        seed: Optional inference seed for reproducibility.
        use_esm_embeddings: Whether to use Chai's ESM embeddings (its default is True).
        device: Torch device string (e.g. ``"cuda:0"``); ``None`` lets Chai choose.
        **kwargs: Forwarded to :class:`OracleMetric` (``pre_oracle_transform``,
            ``template_query``, ``frac_under_thresholds``, ...).
    """

    def __init__(
        self,
        oracle_name: str = "chai1",
        *,
        num_trunk_recycles: int = 3,
        num_diffn_timesteps: int = 200,
        seed: int | None = None,
        use_esm_embeddings: bool = True,
        device: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(oracle_name=oracle_name, **kwargs)  # type: ignore[arg-type]
        self.num_trunk_recycles = num_trunk_recycles
        self.num_diffn_timesteps = num_diffn_timesteps
        self.seed = seed
        self.use_esm_embeddings = use_esm_embeddings
        self.device = device

    def _backend_available(self) -> bool:
        return self._backend_unavailable_reason() is None

    def _backend_unavailable_reason(self) -> str | None:
        try:
            release = metadata.distribution("chai-lab")
        except metadata.PackageNotFoundError:
            return "chai-lab is not installed"
        try:
            torch_version = metadata.version("torch")
        except metadata.PackageNotFoundError:
            return "PyTorch is not installed"
        for dependency in release.requires or ():
            req = Requirement(dependency)
            if req.name.lower() != "torch" or (req.marker and not req.marker.evaluate()):
                continue
            if torch_version not in req.specifier:
                return f"chai-lab {release.version} requires {req}, installed torch is {torch_version}"
        try:
            from chai_lab.chai1 import run_inference
        except (ImportError, OSError, RuntimeError) as error:
            return f"chai-lab {release.version} inference import failed ({type(error).__name__})"
        if not callable(run_inference):
            return f"chai-lab {release.version} has no callable inference entry point"
        return None

    @staticmethod
    def to_fasta(sequences: dict[str, str]) -> str:
        """Render per-chain protein sequences as a Chai-1 input FASTA."""
        return "\n".join(f">protein|name={chain}\n{seq}" for chain, seq in sequences.items()) + "\n"

    def _predict(self, sequences: dict[str, str], work_dir: Path) -> PredictOutput:
        from chai_lab.chai1 import run_inference  # lazy: heavy import, GPU weights

        fasta_path = work_dir / "input.fasta"
        fasta_path.write_text(self.to_fasta(sequences))
        output_dir = work_dir / "chai_out"

        run_kwargs: dict[str, object] = {
            "fasta_file": fasta_path,
            "output_dir": output_dir,
            "num_trunk_recycles": self.num_trunk_recycles,
            "num_diffn_timesteps": self.num_diffn_timesteps,
            "use_esm_embeddings": self.use_esm_embeddings,
        }
        if self.seed is not None:
            run_kwargs["seed"] = self.seed
        if self.device is not None:
            import torch

            run_kwargs["device"] = torch.device(self.device)

        candidates = run_inference(**run_kwargs)
        return self.build_output(candidates, list(sequences.keys()))

    @staticmethod
    def build_output(candidates: object, chain_ids: list[str]) -> PredictOutput:
        """Turn a Chai ``StructureCandidates`` into a :class:`PredictOutput`.

        Picks the top-ranked sample (Chai ranks by aggregate_score; ``.sorted()``
        orders descending) and maps its confidences onto AF3-shaped keys.
        """
        ranked = candidates.sorted() if hasattr(candidates, "sorted") else candidates  # type: ignore[attr-defined]
        cif_path = Path(ranked.cif_paths[0])  # type: ignore[attr-defined]
        rd = ranked.ranking_data[0]  # type: ignore[attr-defined]
        ptm_scores = rd.ptm_scores
        plddt_scores = rd.plddt_scores

        summary: dict[str, object] = {
            "overall_plddt": _scalar(plddt_scores.complex_plddt),
            "ptm": _scalar(ptm_scores.complex_ptm),
            "iptm": _scalar(ptm_scores.interface_ptm),
            "ranking_score": _scalar(rd.aggregate_score),
            "chain_plddt": _vector(plddt_scores.per_chain_plddt),
            "chain_ptm": _vector(ptm_scores.per_chain_ptm),
            "chain_pair_iptm": _matrix(ptm_scores.per_chain_pair_iptm),
            "chain_ids": chain_ids,
        }
        return PredictOutput(cif_path=cif_path, summary_confidences=summary)
