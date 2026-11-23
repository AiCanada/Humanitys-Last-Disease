"""Pass/fail criterion metric for design evaluation.

Evaluates a single threshold criterion against an already-computed metrics dict
(the merged oracle + refolding output). Used as entries in the pass_criteria
MetricManager, which runs last in the validation flow.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from biotite.structure import AtomArray

from rfproteina.metrics.metric import Metric
from rfproteina.metrics.utils.metrics_utils import _write_chain_only_cif_if_unambiguous
from rfproteina.utils.logging import RankedLogger
from rfproteina.utils.metric_thresholds import COMPARISONS, MetricThreshold

logger = RankedLogger(__name__, rank_zero_only=False)


class PassCriterionMetric(Metric):
    """Evaluate a pass criterion against an already-computed metrics dict.

    Intended as entries in a ``pass_criteria`` :py:class:`~rfproteina.metrics.metric.MetricManager`
    that runs after the oracle and refolding MetricManagers. Receives the merged
    oracle + refolding output via ``extra_info["metrics"]`` and returns a binary
    pass/fail flag.

    Args:
        metric_thresholds: Dict mapping metric keys to threshold specs (dicts with
            optional inclusive ``"min"`` / ``"max"`` or strict ``"min_exclusive"`` /
            ``"max_exclusive"`` bounds). All thresholds must pass. A spec may
            instead carry ``"min_of"`` / ``"max_of"`` / ``"mean_of"``, a list of
            metric keys reduced to a single value before thresholding; the outer
            key is then just a label. Every listed key must be present, or the
            result is indeterminate.
        archive_dir: Optional shared directory for atomically publishing passing designed chains.
            Use one directory per target and pass definition; requires the evaluated AtomArray.

    Example:
        >>> PassCriterionMetric(metric_thresholds={"standard.plddt_coi": {"min": 70.0}}).compute(
        ...     {"standard.plddt_coi": 85.0}
        ... )
        {"pass": 1.0, "all_metrics_present": 1.0}

        >>> PassCriterionMetric(
        ...     metric_thresholds={
        ...         "ipae": {"min_of": ["oracle.ipae_C", "oracle.ipae_D"], "max": 2.0}
        ...     }
        ... ).compute({"oracle.ipae_C": 3.1, "oracle.ipae_D": 1.7})
        {"pass": 1.0, "all_metrics_present": 1.0}
    """

    def __init__(
        self,
        metric_thresholds: dict[str, dict[str, Any]],
        archive_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        if archive_dir is not None and not archive_dir:
            raise ValueError("archive_dir must be a nonempty directory path.")
        self.archive_dir = Path(archive_dir) if archive_dir is not None else None
        super().__init__(**kwargs)
        self.metric_thresholds = metric_thresholds
        # Resolved here so a malformed spec fails at config load rather than mid-validation.
        self._thresholds = {label: MetricThreshold.parse(label, spec) for label, spec in metric_thresholds.items()}
        if not self._thresholds:
            raise ValueError("A pass criterion requires at least one threshold.")

    @property
    def required_metric_keys(self) -> set[str]:
        """Metric keys required by direct thresholds and reductions."""
        return {key for spec in self._thresholds.values() for key in spec.keys}

    @property
    def kwargs_to_compute_args(self) -> dict:
        mapping = {"metrics": ("extra_info", "metrics")}
        if self.archive_dir is not None:
            mapping["atom_array"] = "predicted_atom_array"
        return mapping

    def compute(self, metrics: dict, atom_array: AtomArray | None = None) -> dict[str, float]:
        """Evaluate the criteria and return a pass/fail flag.

        Args:
            metrics: Merged oracle + refolding metrics dict, keyed by
                ``{manager_name}.{metric_name}`` dot notation.
            atom_array: Completed design to publish when archive_dir is configured and all criteria pass.

        Returns:
            Dict with ``"pass"`` (``1.0``/``0.0``) and ``"all_metrics_present"``
            (``0.0`` if any required key was missing, in which case ``"pass"``
            is always ``0.0`` but should be treated as indeterminate).
        """
        all_present = True
        passed = True
        for label, spec in self._thresholds.items():
            value = spec.measurement(metrics)
            if value is None:
                logger.warning(f"Missing or non-finite input for criterion metric {label!r}.")
                all_present = False
                passed = False
            elif not all(COMPARISONS[bound](value, cut) for bound, cut in spec.bounds):
                passed = False

        if passed and self.archive_dir is not None:
            if atom_array is None:
                raise ValueError("Passer publication requires the completed design's AtomArray.")
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=self.archive_dir, prefix=".pending_") as directory:
                path = _write_chain_only_cif_if_unambiguous(atom_array, "passer", Path(directory))
                if path is None:
                    raise ValueError("Passer publication requires an unambiguous chain of interest.")
                path.replace(self.archive_dir / f"{uuid4().hex}.cif")
        return {"pass": float(passed), "all_metrics_present": float(all_present)}
