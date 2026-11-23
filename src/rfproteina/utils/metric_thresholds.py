"""Threshold definitions shared by the evaluator and the CSV report reader."""

import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

COMPARISONS = {"min": operator.ge, "max": operator.le, "min_exclusive": operator.gt, "max_exclusive": operator.lt}
_REDUCERS = {"min_of": min, "max_of": max, "mean_of": fmean}


def finite_number(raw: object) -> float | None:
    if isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (ValueError, TypeError, OverflowError):
        return None
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class MetricThreshold:
    keys: tuple[str, ...]
    reduction: str | None
    bounds: tuple[tuple[str, float], ...]

    @classmethod
    def parse(cls, label: str, spec: object) -> "MetricThreshold":
        if not isinstance(spec, Mapping) or not isinstance(label, str) or not label:
            raise ValueError(f"Invalid threshold definition for {label!r}.")
        if set(spec).difference(COMPARISONS, _REDUCERS):
            raise ValueError(f"Unrecognized threshold option for {label!r}.")
        reductions = [key for key in _REDUCERS if key in spec]
        if len(reductions) > 1:
            raise ValueError(f"More than one reduction for {label!r}.")
        reduction = next(iter(reductions), None)
        keys = spec[reduction] if reduction else [label]
        if (
            not isinstance(keys, Sequence) or isinstance(keys, str | bytes) or not keys
            or not all(isinstance(key, str) and key.strip() for key in keys)
        ):
            raise ValueError(f"Expected a nonempty list of metric keys for {label!r}.")
        bounds = []
        for name, raw in spec.items():
            if name in COMPARISONS and raw is not None:
                cut = finite_number(raw)
                if cut is None:
                    raise ValueError(f"Nonfinite or invalid {name} bound for {label!r}.")
                bounds.append((name, cut))
        if not bounds:
            raise ValueError(f"No bounds declared for {label!r}.")
        return cls(tuple(keys), reduction, tuple(bounds))

    def measurement(self, metrics: Mapping) -> float | None:
        values = []
        for key in self.keys:
            value = finite_number(metrics.get(key))
            if value is None:
                return None
            values.append(value)
        if self.reduction is None:
            return values[0]
        try:
            return finite_number(_REDUCERS[self.reduction](values))
        except OverflowError:
            return None
