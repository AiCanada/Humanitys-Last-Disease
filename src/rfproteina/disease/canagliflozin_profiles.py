"""Matched concentrations for the oral canagliflozin control, sampled on the worker's minute grid."""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray

HOURS = (0, 1, 2, 4, 8, 12, 24)
COMPARTMENTS = {
    "gut_lumen_parent": "[Cgu_can]",
    "liver_tissue_parent": "[LI__can]",
    "kidney_plasma_parent": "[Cki_plasma_can]",
    "venous_parent": "[Cve_can]",
}


def profile_from_trajectory(data: ndarray, columns: Sequence[str]) -> dict:
    """Extract concentrations from a validated minute-grid trajectory and convert mmol/L to umol/L."""
    profile = {"time_unit": "h", "concentration_unit": "umol/L", "time": list(HOURS)}
    for compartment, symbol in COMPARTMENTS.items():
        column = columns.index(symbol)
        profile[compartment] = [float(data[hour * 60, column]) * 1000 for hour in HOURS]
    return profile


def validated_profile(raw: object, measurements: dict) -> dict:
    """Require each named compartment at every reference time with the matching first-day peak."""
    if not isinstance(raw, dict) or raw.keys() != {"time", "time_unit", "concentration_unit", *COMPARTMENTS}:
        raise ValueError("unexpected reference profile fields")
    times = raw["time"]
    if (raw["time_unit"] != "h" or raw["concentration_unit"] != "umol/L"
            or not isinstance(times, list) or times != list(HOURS)
            or any(type(hour) not in (int, float) for hour in times)):
        raise ValueError("invalid reference profile schedule or units")
    result = {"time_unit": "h", "concentration_unit": "umol/L", "time": list(HOURS)}
    for compartment in COMPARTMENTS:
        values = raw[compartment]
        peak = measurements[f"{compartment}_cmax_0_24h"]["value"]
        if not isinstance(values, list) or len(values) != len(HOURS):
            raise ValueError("incomplete reference profile")
        cleaned = []
        for value in values:
            if type(value) not in (int, float) or not 0 <= value <= sys.float_info.max:
                raise ValueError("invalid reference concentration")
            if value > peak and not math.isclose(value, peak, rel_tol=1e-10, abs_tol=0):
                raise ValueError("sampled concentration exceeds the first-day peak")
            cleaned.append(float(value))
        if cleaned[0] != 0:
            raise ValueError("reference profile has a nonzero predose concentration")
        result[compartment] = cleaned
    _validate_venous_exposure(result["venous_parent"], measurements)
    return result


def _validate_venous_exposure(concentrations: list[float], measurements: dict) -> None:
    """Bound minute-grid trapezoidal exposure using every known plasma sample and the peak."""
    peak = measurements["venous_parent_cmax_0_24h"]["value"]
    peak_minute = round(measurements["venous_parent_tmax_0_24h"]["value"] * 60)
    known = dict(zip((hour * 60 for hour in HOURS), concentrations, strict=True))
    if peak_minute in known and not math.isclose(known[peak_minute], peak, rel_tol=1e-10):
        raise ValueError("venous concentration at its peak time differs from its first-day peak")
    known[peak_minute] = peak

    # Unknown minute-grid concentrations range from zero to Cmax. Trapezoid
    # endpoints have half weight. Normalizing by Cmax keeps large finite values safe.
    end = HOURS[-1] * 60
    weights = {minute: 0.5 if minute in (0, end) else 1.0 for minute in known}
    known_area = math.fsum(weights[minute] * (value / peak) for minute, value in known.items())
    lower = known_area / end
    upper = (end - sum(weights.values()) + known_area) / end
    mean_to_peak = (measurements["venous_parent_auc_0_24h"]["value"] / HOURS[-1]) / peak
    if (
        (mean_to_peak < lower and not math.isclose(mean_to_peak, lower, rel_tol=1e-10))
        or (mean_to_peak > upper and not math.isclose(mean_to_peak, upper, rel_tol=1e-10))
    ):
        raise ValueError("venous AUC is inconsistent with its sampled concentrations and peak")
