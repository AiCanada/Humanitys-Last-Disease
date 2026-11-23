"""Thermodynamic integration on archived, one-dimensional alchemlyb derivative data."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from importlib.metadata import version
from itertools import pairwise
from numbers import Real
from pathlib import Path


def _finite_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def analyze_ti_stage(
    path: str | Path, states: Sequence[float], *, discard_before: float | None = None,
) -> dict:
    """Fit alchemlyb TI to one stage of a completed study in pandas Parquet format.

    Supply the increasing lambda schedule from the study and a combined dH/dλ table
    with the original temperature and energy-unit metadata. Time is in the source
    table's units. Each window is subsampled for autocorrelation after any time cut.
    """
    import numpy as np
    import pandas as pd

    try:
        from alchemlyb import concat
        from alchemlyb.estimators import TI
        from alchemlyb.postprocessors.units import to_kcalmol
        from alchemlyb.preprocessing.subsampling import decorrelate_dhdl
        from pymbar.utils import ParameterError
    except ImportError as exc:
        raise ValueError("alchemlyb and its numerical dependencies are required") from exc

    schedule = list(states)
    if (len(schedule) < 2 or not all(_finite_number(state) for state in schedule)
            or any(a >= b for a, b in pairwise(schedule))):
        raise ValueError("the declared lambda schedule must contain at least two finite, increasing states")
    if discard_before is not None and not _finite_number(discard_before):
        raise ValueError("the time cut must be finite")
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        table = pd.read_parquet(stream, engine="pyarrow")
    temperature = table.attrs.get("temperature")
    if not _finite_number(temperature) or temperature <= 0 or table.attrs.get("energy_unit") != "kT":
        raise ValueError("expected recorded temperature in kelvin and reduced derivatives in kT")
    if table.index.nlevels != 2 or table.index.names[0] != "time":
        raise ValueError("expected a time index and one lambda coordinate")
    coordinate = table.index.names[1]
    if (not isinstance(coordinate, str) or not coordinate.endswith("-lambda")
            or list(table.columns) != [coordinate.removesuffix("-lambda")]):
        raise ValueError("the derivative column must match the simulated lambda coordinate")
    times = table.index.get_level_values("time")
    lambdas = table.index.get_level_values(coordinate)
    if any(values.dtype.kind not in "ifu" for values in (times, lambdas, table.iloc[:, 0])):
        raise ValueError("time, lambda and energy derivatives must be numeric")
    if not all(not values.isna().any() and np.isfinite(values).all() for values in (times, lambdas, table.iloc[:, 0])):
        raise ValueError("nonfinite time, lambda or energy derivative")
    if not table.index.is_unique or set(lambdas) != set(schedule):
        raise ValueError("duplicate samples or a mismatch with the declared lambda schedule")
    samples, windows = [], []
    for state, window in table.groupby(level=coordinate, sort=True):
        window = window.sort_index(level="time")
        original = len(window)
        if discard_before is not None:
            window = window.loc[window.index.get_level_values("time") >= discard_before]
        spacing = np.diff(window.index.get_level_values("time").to_numpy(dtype=float))
        if not len(spacing) or not np.allclose(spacing, spacing[0], rtol=1e-6, atol=0):
            raise ValueError(f"lambda {state} requires regularly spaced samples after the time cut")
        try:
            used = decorrelate_dhdl(window, drop_duplicates=False, sort=False)
        except ParameterError as exc:
            raise ValueError(f"lambda {state} has no estimable sampling inefficiency") from exc
        if len(used) < 2:
            raise ValueError(f"lambda {state} has fewer than two decorrelated samples")
        samples.append(used)
        windows.append({"lambda": float(state), "recorded": original, "after_time_cut": len(window), "used": len(used)})
    fitted = TI().fit(concat(samples))
    start, end = schedule[0], schedule[-1]
    delta = float(to_kcalmol(fitted.delta_f_).at[start, end])
    error = float(to_kcalmol(fitted.d_delta_f_).at[start, end])
    if not math.isfinite(delta) or not math.isfinite(error) or error < 0:
        raise ValueError("the estimator returned an invalid result")
    return {
        "method": "alchemlyb_thermodynamic_integration", "version": version("alchemlyb"),
        "scope": "one supplied alchemical stage", "source_sha256": digest, "coordinate": coordinate,
        "initial_lambda": float(start), "final_lambda": float(end), "temperature_kelvin": float(temperature),
        "delta_g_final_minus_initial": delta, "standard_error": error, "units": "kcal/mol",
        "uncertainty_basis": "propagated standard errors of decorrelated window means",
        "discard_before": discard_before, "windows": windows,
    }
