"""Isolated CPU worker for the released Mamidi2014 canagliflozin reference experiment."""

from __future__ import annotations

import json
import math
import sys
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray

if __package__:
    from . import canagliflozin_profiles as profiles
else:
    # -I excludes the script's directory. Add only our installed sibling helpers.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import canagliflozin_profiles as profiles

SOLVER_VERSION = "2.11.0"
VERSIONS = {"libroadrunner": SOLVER_VERSION, "numpy": "2.5.3"}
SOLVER_UNAVAILABLE = 42
DOSE_MG = 188
DURATION_H = 170
BODY_WEIGHT_KG = 75
FASTING_GLUCOSE_MMOL_L = 5
PARENT_MOLAR_MASS = 444.518  # SBML Mr_can, g/mol
CVODE_ATOL = 1e-12
CVODE_RTOL = 1e-8

# Brackets denote SBML concentrations (mmol/L). Unbracketed excreta are amounts (mmol).
CONCENTRATIONS = {
    "gut_lumen_parent_cmax_0_24h": "[Cgu_can]",
    "liver_tissue_parent_cmax_0_24h": "[LI__can]",
    "kidney_plasma_parent_cmax_0_24h": "[Cki_plasma_can]",
    "venous_parent_cmax_0_24h": "[Cve_can]",
    "venous_m5_cmax_0_24h": "[Cve_m5]",
    "venous_m7_cmax_0_24h": "[Cve_m7]",
}
EXCRETA = {
    "urine_m5_48h": ("Aurine_m5", 48),
    "urine_m7_48h": ("Aurine_m7", 48),
    "urine_total_168h": ("Aurine_cantot", 168),
    "feces_total_168h": ("Afeces_cantot", 168),
}
UNITS = {
    **dict.fromkeys(CONCENTRATIONS, "umol/L"),
    "venous_parent_auc_0_24h": "umol*h/L",
    "venous_parent_tmax_0_24h": "h",
    **dict.fromkeys(EXCRETA, "umol"),
}


def validated_measurements(raw: object) -> dict:
    """Check dimensions and physical consistency of this positive-dose reference output."""
    if not isinstance(raw, dict) or raw.keys() != UNITS.keys():
        raise ValueError("unexpected reference measurements")
    result = {}
    for name, unit in UNITS.items():
        measured = raw[name]
        if not isinstance(measured, dict) or measured.keys() != {"value", "unit"} or measured["unit"] != unit:
            raise ValueError("invalid reference unit")
        value = measured["value"]
        if type(value) not in (int, float) or not 0 < value <= sys.float_info.max:
            raise ValueError("invalid reference value")
        result[name] = {"value": float(value), "unit": unit}
    values = {key: item["value"] for key, item in result.items()}
    peak_min = values["venous_parent_tmax_0_24h"] * 60
    if not (0 < peak_min <= 1440 and math.isclose(peak_min, round(peak_min), rel_tol=0, abs_tol=1e-9)):
        raise ValueError("peak time outside the sampling grid")
    if values["venous_parent_auc_0_24h"] / 24 > values["venous_parent_cmax_0_24h"]:
        raise ValueError("mean exceeds the sampled peak")
    if values["urine_m5_48h"] + values["urine_m7_48h"] > values["urine_total_168h"]:
        raise ValueError("cumulative urine recovery is inconsistent")
    if values["urine_total_168h"] + values["feces_total_168h"] > DOSE_MG * 1000 / PARENT_MOLAR_MASS * (1 + 1e-6):
        raise ValueError("excreta exceed the administered molar dose")
    return result


def validated_result(raw: object) -> dict:
    if not isinstance(raw, dict) or raw.keys() != {"measurements", "profile"}:
        raise ValueError("incomplete reference result")
    measurements = validated_measurements(raw["measurements"])
    return {"measurements": measurements, "profile": profiles.validated_profile(raw["profile"], measurements)}


def _physical_trajectory(data: ndarray, columns: list[str]) -> ndarray:
    """Check the full trajectory before retaining selected hours and summary measurements."""
    import numpy as np

    trace = np.asarray(data)
    # All selected species start at exactly zero in the SBML, before integration.
    # Apply the absolute numerical floor in each reported scale (mmol/L or mmol).
    if np.any(trace[0, 1:] != 0) or np.any(trace[:, 1:] < -CVODE_ATOL):
        raise ValueError("invalid predose state or negative concentration or amount")
    trace = np.maximum(trace, 0)
    # Explicit species order keeps the mass-balance comparisons independent of column order.
    amount_species = ("Aurine_m5", "Aurine_m7", "Aurine_cantot", "Afeces_cantot")
    amounts = trace[:, [columns.index(selection) for selection in amount_species]]
    previous_peak = np.maximum.accumulate(amounts, axis=0)
    if np.any(previous_peak - amounts > CVODE_ATOL + CVODE_RTOL * previous_peak):
        raise ValueError("cumulative excretion decreased")
    urine_m5, urine_m7, urine_total, feces_total = amounts.T
    molar_dose = DOSE_MG / PARENT_MOLAR_MASS  # mmol
    if (
        np.any(urine_m5 + urine_m7 > urine_total * (1 + CVODE_RTOL) + 3 * CVODE_ATOL)
        or np.any(urine_total + feces_total > molar_dose * (1 + CVODE_RTOL) + 2 * CVODE_ATOL)
    ):
        raise ValueError("inconsistent excretion mass balance")
    return trace


def summarize(data: ndarray) -> dict:
    """Reduce minute-grid concentrations and cumulative amounts in their respective windows."""
    import numpy as np

    columns = ["time", *CONCENTRATIONS.values(), *(selection for selection, _ in EXCRETA.values())]
    if (
        data.shape != (DURATION_H * 60 + 1, len(columns)) or list(data.colnames) != columns
        or not np.isfinite(data).all() or not np.array_equal(data[:, 0], np.arange(DURATION_H * 60 + 1))
    ):
        raise ValueError("incomplete trajectory")
    data = _physical_trajectory(data, columns)
    first_day = data[:24 * 60 + 1]
    # Convert mmol/L to umol/L, and integrate using time in hours.
    values = {name: float(first_day[:, idx].max()) * 1000 for idx, name in enumerate(CONCENTRATIONS, start=1)}
    plasma = np.asarray(first_day[:, columns.index("[Cve_can]")]) * 1000
    values["venous_parent_tmax_0_24h"] = int(plasma.argmax()) / 60
    values["venous_parent_auc_0_24h"] = float(np.trapezoid(plasma, dx=1 / 60))
    for name, (selection, hour) in EXCRETA.items():
        values[name] = float(data[hour * 60, columns.index(selection)]) * 1000
    return validated_result({
        "measurements": {name: {"value": value, "unit": UNITS[name]} for name, value in values.items()},
        "profile": profiles.profile_from_trajectory(data, columns),
    })


def simulate(roadrunner: ModuleType, xml: bytes) -> dict:
    model = roadrunner.RoadRunner(xml.decode("utf-8"))
    # All other initial values and physiological parameters come from the pinned SBML.
    model["init(BW)"] = BODY_WEIGHT_KG
    model["init([KI__fpg])"] = FASTING_GLUCOSE_MMOL_L
    model["init(PODOSE_can)"] = DOSE_MG
    model.resetAll()
    model.setIntegrator("cvode")
    model.integrator.relative_tolerance = CVODE_RTOL
    model.integrator.absolute_tolerance = CVODE_ATOL
    columns = ["time", *CONCENTRATIONS.values(), *(selection for selection, _ in EXCRETA.values())]
    minutes = DURATION_H * 60
    return summarize(model.simulate(0, minutes, minutes + 1, selections=columns))


def main() -> int:
    try:
        import roadrunner

        if any(metadata.version(name) != version for name, version in VERSIONS.items()):
            return SOLVER_UNAVAILABLE
    except (ImportError, OSError):
        return SOLVER_UNAVAILABLE
    if sys.argv[1:] == ["--check-installation"]:
        result = {"libroadrunner": SOLVER_VERSION}
    elif len(sys.argv) == 1:
        result = simulate(roadrunner, sys.stdin.buffer.read(253_000))
    else:
        return 2
    print(json.dumps(result, allow_nan=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
