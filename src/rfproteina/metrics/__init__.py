"""Metrics for model evaluation.

Re-exports the base metric framework so ``from rfproteina.metrics import
MetricManager`` keeps working; the concrete metrics live in submodules.

The re-export is **lazy**, and deliberately so. Importing :py:mod:`metric` here
eagerly meant that importing *anything* under ``rfproteina.metrics`` pulled in
the whole training stack, because the package's ``__init__`` runs first:

    metric -> rfproteina.utils.ddp -> lightning_fabric.utilities

and ``lightning_fabric.utilities.imports`` version-checks ``torchmetrics`` by
importing it, whose ``torchmetrics.functional.text.bert`` imports
``transformers``, whose ``utils.import_utils`` calls
``importlib.metadata.packages_distributions()`` -- which stats every file of
every installed distribution. On a checkout mounted from Windows that walk cost
168 s, paid in full by whichever test or command first touched the package.

:py:mod:`~rfproteina.metrics.oracles.imported_refolds` keeps ``OracleMetric``
behind ``TYPE_CHECKING`` precisely to stay free of that stack -- it reads
refolds off disk and needs no model -- and an eager re-export here undid the
effort, since the module cannot be reached without importing this package.

Nothing changes for a caller that wants a metric: the first attribute access
imports :py:mod:`metric` and caches the result in ``globals()``, so the cost is
paid once, by whoever actually needs it.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rfproteina.metrics.metric import Metric, MetricInputError, MetricManager

__all__ = [
    "Metric",
    "MetricInputError",
    "MetricManager",
]


def __getattr__(name: str) -> object:
    """Import :py:mod:`metric` on first use of one of the three re-exported names.

    Anything else raises :py:class:`AttributeError`, which is what keeps
    ``from rfproteina.metrics import <submodule>`` working: the import machinery
    tries this hook first and falls back to importing the submodule when it
    fails. Returning a placeholder, or raising anything else, would break that.
    """
    if name in __all__:
        from rfproteina.metrics import metric

        value = getattr(metric, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """The re-exported names, whether or not they have been imported yet."""
    return sorted(set(globals()) | set(__all__))
