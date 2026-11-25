"""Read individual PK-Sim concentration outputs in the native wide CSV format."""

from __future__ import annotations

import csv
import io
import math
import re
from itertools import pairwise
from typing import TypedDict


class CurveError(ValueError):
    """An export cannot provide the requested attributed concentration curves."""


class ExposureCurve(TypedDict):
    output_path: str
    individual_id: int
    time_unit: str
    concentration_unit: str
    time: list[float]
    concentration: list[float]
    point_count: int
    grid_cmax: float
    grid_tmax: float
    auc_start: float
    auc_end: float
    auc_trapezoidal: float
    auc_unit: str


MAX_EXPORT_BYTES = 16 * 1024 * 1024
MAX_OUTPUTS = 32
MAX_POINTS = 20_000
_MAX_HEADER_COLUMNS = 1024 + 2
_MAX_RETURNED_VALUES = 128_000
_TIME_UNITS = {"s", "min", "h", "d", "day"}
_CONCENTRATION_UNIT = re.compile(r"(?:[munpµμ]?mol|[munpµμ]?g)/(?:[munµμ]?[lL]|d[lL])")
_NUMBER = re.compile(r"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))(?:[eE][+-]?[0-9]+)?")


def _number(field: str) -> float:
    # Float accepts Python syntax (including underscores) that the exporter never writes.
    match = _NUMBER.fullmatch(field) if len(field) <= 64 else None
    if match is None:
        raise CurveError("pksim_invalid_export")
    value = float(field)
    if not math.isfinite(value) or value < 0 or (value == 0 and re.search("[1-9]", match[1])):
        raise CurveError("pksim_invalid_export")
    return value


def _columns(header: list[str], paths: tuple[str, ...]) -> tuple[str, list[tuple[int, str]]]:
    if not 3 <= len(header) <= _MAX_HEADER_COLUMNS or header[0] != "IndividualId":
        raise CurveError("pksim_invalid_export")
    time_label = re.fullmatch(r"Time \[([^\[\]]+)\]", header[1])
    if time_label is None or time_label[1] not in _TIME_UNITS:
        raise CurveError("pksim_unsupported_unit")
    by_path: dict[str, tuple[int, str]] = {}
    for index, cell in enumerate(header[2:], 2):
        if len(cell) > 2048 or not cell.isprintable():
            raise CurveError("pksim_invalid_export")
        # OSPSuite omits the unit suffix for dimensionless quantities. Such columns
        # can coexist with selected concentrations and must remain distinguishable.
        name, sep, suffix = cell.rpartition(" [")
        if sep and suffix.endswith("]"):
            unit = suffix[:-1]
        else:
            name, unit = cell, ""
        if "|" not in name or not all(name.split("|")) or name in by_path:
            raise CurveError("pksim_invalid_export")
        by_path[name] = index, unit
    if any(path not in by_path for path in paths):
        raise CurveError("pksim_output_missing")
    selected = [by_path[path] for path in paths]
    if any(not _CONCENTRATION_UNIT.fullmatch(unit) for _, unit in selected):
        raise CurveError("pksim_unsupported_unit")
    return time_label[1], selected


def _summary(
    path: str, unit: str, individual: int, time_unit: str, time: list[float], values: list[float]
) -> ExposureCurve:
    peak = max(values)
    # Midpoints avoid overflow from adding two large concentrations. The result
    # covers exactly the sampled interval, with no extrapolation or fitted Tmax.
    try:
        auc = math.fsum(
            (end - start) * (low + (high - low) * 0.5)
            for (start, end), (low, high) in zip(pairwise(time), pairwise(values), strict=True)
        )
    except OverflowError as exc:
        raise CurveError("pksim_invalid_export") from exc
    if not math.isfinite(auc):
        raise CurveError("pksim_invalid_export")
    return ExposureCurve(
        output_path=path,
        individual_id=individual,
        time_unit=time_unit,
        concentration_unit=unit,
        time=time,
        concentration=values,
        point_count=len(time),
        grid_cmax=peak,
        grid_tmax=time[values.index(peak)],
        auc_start=time[0],
        auc_end=time[-1],
        auc_trapezoidal=auc,
        auc_unit=f"({unit})*{time_unit}",
    )


def extract(raw: bytes, paths: tuple[str, ...]) -> list[ExposureCurve]:
    """Read one individual and preserve each nominated quantity's native path, time grid and scale."""
    if not isinstance(raw, bytes) or len(raw) > MAX_EXPORT_BYTES:
        raise CurveError("pksim_invalid_export")
    if (
        not 1 <= len(paths) <= MAX_OUTPUTS
        or any(not isinstance(p, str) for p in paths)
        or len(set(paths)) != len(paths)
    ):
        raise CurveError("pksim_invalid_export")
    try:
        stream = io.StringIO(raw.decode("utf-8-sig"), newline="")
        rows = csv.reader(stream, strict=True)
        header = next(rows, [])
        time_unit, columns = _columns(header, paths)
        time: list[float] = []
        concentrations: list[list[float]] = [[] for _ in columns]
        individual_id = None
        limit = min(MAX_POINTS, _MAX_RETURNED_VALUES // len(columns))
        for row in rows:
            if len(row) != len(header) or len(time) >= limit:
                raise CurveError("pksim_invalid_export")
            if individual_id is None:
                if not re.fullmatch(r"0|[1-9][0-9]{0,9}", row[0]):
                    raise CurveError("pksim_invalid_export")
                individual_id = row[0]
            if row[0] != individual_id:
                raise CurveError("pksim_invalid_export")
            t = _number(row[1])
            if time and t <= time[-1]:
                raise CurveError("pksim_invalid_export")
            time.append(t)
            for (index, _), values in zip(columns, concentrations, strict=True):
                values.append(_number(row[index]))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise CurveError("pksim_invalid_export") from exc
    if individual_id is None or len(time) < 2:
        raise CurveError("pksim_invalid_export")
    return [
        _summary(path, unit, int(individual_id), time_unit, time, values)
        for path, (_, unit), values in zip(paths, columns, concentrations, strict=True)
    ]
