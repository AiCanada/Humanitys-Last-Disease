"""Shared machinery for the free, open-weights refolding oracles (Chai-1, Boltz-2).

Both Chai-1 (Apache-2.0, code and weights, verified from the repo LICENSE on
2026-11-16) and Boltz-2 (MIT, verified likewise) are sequence-in / structure-out
predictors with AlphaFold3-class confidence outputs. They slot into the same
:class:`~rfproteina.metrics.oracles.oracle_metric.OracleMetric` contract as the
AF3 oracle: fold a design, write a refolded CIF and a confidence JSON to the
canonical paths, return the flat metrics dict.

This base class carries the parts both share -- extracting per-chain protein
sequences from the generated AtomArray, driving the fold/cache loop, mapping the
predictor's AF3-shaped confidence summary through
:func:`~rfproteina.metrics.oracles.oracle_metric.extract_confidence_metrics`, and
persisting results -- and leaves one abstract method, :meth:`_predict`, to each
backend.

Honest scope
------------
These adapters save the *predictor's own* output structure as the refolded CIF.
They do **not** reproduce AF3Oracle's atom-index remapping of predicted
coordinates back onto the input atom ordering (``_normalize_af3_output`` /
``_build_atom_index_mapping``, ~300 lines). A downstream
StructureComparisonMetric that assumes the refold shares the design's exact atom
order will therefore need an alignment step for these oracles. That remap is a
named follow-up, not silently pretended here. Ligands and nucleic acids in the
design are not yet forwarded to the predictor; only protein chains are. Both
limits are surfaced, never worked around with fabricated data.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

from biotite.structure import AtomArray

from rfproteina.metrics.oracles.oracle_metric import (
    OracleJobInput,
    OracleMetric,
    extract_confidence_metrics,
)
from rfproteina.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

# Minimal 3-letter -> 1-letter map for the 20 standard amino acids. Anything else
# (modified residue, ligand, nucleotide) becomes "X" so the chain length is
# preserved rather than silently shortened.
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


class OracleBackendUnavailableError(RuntimeError):
    """Raised when a free oracle's backend package is not installed."""


@dataclass(frozen=True, slots=True)
class PredictOutput:
    """One backend prediction: a structure file and an AF3-shaped confidence dict.

    Args:
        cif_path: Path to the predicted structure the backend wrote (CIF or PDB).
        summary_confidences: AF3-shaped dict (keys like ``overall_plddt``,
            ``ptm``, ``iptm``, ``chain_plddt``, ``chain_pair_pae_min``,
            ``chain_pair_iptm``, ``chain_ids``), values in 0..1 where AF3 uses
            0..1 -- :func:`extract_confidence_metrics` rescales them.
    """

    cif_path: Path
    summary_confidences: dict


def extract_protein_sequences(atom_array: AtomArray) -> dict[str, str]:
    """Per-chain one-letter protein sequences, in first-appearance chain order.

    Chains with no standard amino acid (pure ligand/nucleic) are dropped. Within
    a kept chain, non-standard residues become ``X`` so the sequence length still
    matches the residue count. Uses ``chain_iid`` when present (it disambiguates
    repeated ``chain_id`` across assemblies), else ``chain_id``.
    """
    category = "chain_iid" if "chain_iid" in atom_array.get_annotation_categories() else "chain_id"
    chain_ids = atom_array.get_annotation(category)
    res_ids = atom_array.res_id
    res_names = atom_array.res_name

    sequences: dict[str, str] = {}
    for chain in dict.fromkeys(chain_ids.tolist()):
        chain_mask = chain_ids == chain
        seen: set[int] = set()
        letters: list[str] = []
        has_standard = False
        for res_id, res_name in zip(res_ids[chain_mask], res_names[chain_mask], strict=True):
            if res_id in seen:
                continue
            seen.add(int(res_id))
            one = _THREE_TO_ONE.get(str(res_name).upper())
            if one is not None:
                has_standard = True
            letters.append(one or "X")
        if has_standard:
            sequences[str(chain)] = "".join(letters)
    return sequences


class SequenceRefoldOracle(OracleMetric):
    """Base for sequence-in refolding oracles with AF3-shaped confidences.

    Subclasses implement :meth:`_backend_available` and :meth:`_predict`.
    Everything else -- the cache loop, sequence extraction, the confidence
    translation and persistence -- lives here.
    """

    @abstractmethod
    def _backend_available(self) -> bool:
        """True iff this oracle's prediction package is importable."""
        raise NotImplementedError

    @abstractmethod
    def _predict(self, sequences: dict[str, str], work_dir: Path) -> PredictOutput:
        """Run the backend on ``sequences`` and return its structure + confidences.

        ``work_dir`` is a scratch directory the backend may write into freely.
        """
        raise NotImplementedError

    def fold_one(self, job: OracleJobInput) -> dict:
        """Fold one design; see :meth:`fold_batch`."""
        n_ok = self.fold_batch([job])
        if n_ok < 1:
            raise RuntimeError(f"{self.oracle_name}.fold_one produced nothing for {job.example_id!r}")
        return self._load_confidence_json(self.confidence_path(job.out_dir, job.example_id, job.save_id))

    def fold_batch(self, jobs: list[OracleJobInput]) -> int:
        """Fold every not-yet-cached design, one backend call each.

        Returns the number of jobs with a refold on disk afterwards (cached hits
        counted). A design whose fold raises is logged and skipped, matching
        AF3Oracle's policy of never letting one design strand a whole rank.
        """
        pending = [j for j in jobs if not self.already_refolded(j.out_dir, j.example_id, j.save_id)]
        n_cached = len(jobs) - len(pending)
        if not pending:
            return n_cached
        if not self._backend_available():
            raise OracleBackendUnavailableError(
                f"{self.oracle_name}: backend package is not installed, so nothing can be folded and "
                f"{len(pending)} design(s) have no refold on disk. Install the backend (see the oracle's "
                "module docstring for the package and licence) or import existing refolds. No structure was invented."
            )

        import shutil
        import tempfile

        n_ok = 0
        for job in pending:
            sequences = extract_protein_sequences(job.atom_array)
            if not sequences:
                logger.warning(f"{self.oracle_name}: {job.example_id!r} has no protein chain to fold; skipping")
                continue
            work_dir = Path(tempfile.mkdtemp(prefix=f"{self.oracle_name}_"))
            try:
                output = self._predict(sequences, work_dir)
                self._persist(job, sequences, output)
                n_ok += 1
            except Exception as error:  # isolate one design's failure from the rest of the batch
                logger.warning(f"{self.oracle_name}: fold failed for {job.example_id!r}; skipping: {error}")
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)
        return n_cached + n_ok

    def _persist(self, job: OracleJobInput, sequences: dict[str, str], output: PredictOutput) -> None:
        """Copy the predicted structure to the canonical path and save metrics."""
        import shutil

        cif_path = self.refolded_cif_path(job.out_dir, job.example_id, job.save_id)
        cif_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep the predictor's own encoding; the canonical suffix is .cif, and the
        # backend is expected to write CIF. A PDB output is copied verbatim and the
        # reader's find_structure_file resolves it.
        shutil.copyfile(output.cif_path, cif_path)

        chain_of_interest = job.chain_of_interest
        chains_in_order = list(sequences.keys())
        summary = dict(output.summary_confidences)
        summary.setdefault("chain_ids", chains_in_order)
        metrics = extract_confidence_metrics(
            summary_confidences=summary,
            chain_of_interest=chain_of_interest,
            chains_in_order=summary["chain_ids"],
        )
        metrics["chain_of_interest"] = chain_of_interest
        # A backend may carry extra named scalars its native scoring produces but
        # that the AF3 confidence schema has no slot for (e.g. a per-residue error
        # estimate). They are merged verbatim under their own keys, never dressed
        # up as an AF3 metric. Non-scalar values are dropped to keep the JSON flat.
        extra = output.summary_confidences.get("_extra")
        if isinstance(extra, dict):
            for key, value in extra.items():
                if isinstance(value, int | float | str | bool) or value is None:
                    metrics[str(key)] = value
        self._save_confidence_json(metrics, self.confidence_path(job.out_dir, job.example_id, job.save_id))

