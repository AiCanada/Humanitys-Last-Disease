"""Time-matched parent concentrations preserve tissue identity, timing and units."""

import numpy as np
import pytest

from rfproteina.disease import canagliflozin_profiles as profiles
from rfproteina.disease import canagliflozin_worker as worker


@pytest.fixture
def concentration_profile():
    # Deliberately reverse the SBML columns. The extractor must select the named compartment.
    columns = ["time", "[Cve_can]", "[Cki_plasma_can]", "[LI__can]", "[Cgu_can]"]
    data = np.zeros((24 * 60 + 1, len(columns)))
    minutes = np.arange(len(data))
    for idx in range(1, len(columns)):
        data[:, idx] = idx * minutes / 60_000  # mmol/L
    measured = profiles.profile_from_trajectory(data, columns)
    peaks = {f"{name}_cmax_0_24h": {"value": measured[name][-1], "unit": "umol/L"}
             for name in profiles.COMPARTMENTS}
    peaks["venous_parent_tmax_0_24h"] = {"value": 24, "unit": "h"}
    peaks["venous_parent_auc_0_24h"] = {"value": 288, "unit": "umol*h/L"}
    return measured, peaks


def test_reference_profile_extracts_and_converts_each_scheduled_sample(concentration_profile):
    profile, peaks = concentration_profile
    times = [0, 1, 2, 4, 8, 12, 24]
    assert profile["time"] == times and profile["time_unit"] == "h" and profile["concentration_unit"] == "umol/L"
    for idx, name in enumerate(("venous_parent", "kidney_plasma_parent", "liver_tissue_parent", "gut_lumen_parent"), 1):
        assert profile[name] == pytest.approx([hour * idx for hour in times])
    assert profiles.validated_profile(profile, peaks) == profile


@pytest.mark.parametrize(("field", "value"), [
    ("time", [0, 1, 2, 4, 8, 12, 23]), ("time", [False, 1, 2, 4, 8, 12, 24]),
    ("concentration_unit", "mmol/L"), ("time_unit", "min"), ("unrecognized", []),
    ("venous_parent", [0, 1]), ("venous_parent", [1, 1, 2, 4, 8, 12, 24]),
    ("venous_parent", [0, True, 2, 4, 8, 12, 24]), ("venous_parent", [0, -1, 2, 4, 8, 12, 24]),
    ("venous_parent", [0, 25, 2, 4, 8, 12, 24]), ("venous_parent", [0, float("nan"), 2, 4, 8, 12, 24]),
    ("venous_parent", [0, 1, 2, 4, 8, 12, 23]),
    ("venous_parent", [0, float("inf"), 2, 4, 8, 12, 24]),
])
def test_reference_profile_rejects_inconsistent_units_times_and_observations(concentration_profile, field, value):
    profile, peaks = concentration_profile
    with pytest.raises(ValueError):
        profiles.validated_profile({**profile, field: value}, peaks)


@pytest.mark.parametrize("peak_minute", [60, 83, 1440])
def test_sparse_plasma_profile_constrains_both_extremes_of_minute_grid_auc(concentration_profile, peak_minute):
    profile, measurements = concentration_profile
    concentrations = np.array([0, 2, 1.5, 1, 0.8, 0.5, 0.1])
    peak = 3.0
    sample_indices = np.array(profiles.HOURS) * 60
    concentrations[sample_indices == peak_minute] = peak
    profile["venous_parent"] = concentrations.tolist()
    measurements["venous_parent_cmax_0_24h"]["value"] = peak
    measurements["venous_parent_tmax_0_24h"]["value"] = peak_minute / 60
    # Construct the two extreme full curves independently of the validator.
    extrema = []
    for unknown_concentration in (0.0, peak):
        curve = np.full(1441, unknown_concentration)
        curve[sample_indices] = concentrations
        curve[peak_minute] = peak
        auc = float(np.trapezoid(curve, dx=1 / 60))
        extrema.append(auc)
        measurements["venous_parent_auc_0_24h"]["value"] = auc
        assert profiles.validated_profile(profile, measurements) == profile
    for auc in (extrema[0] - 0.001, extrema[1] + 0.001):
        measurements["venous_parent_auc_0_24h"]["value"] = auc
        with pytest.raises(ValueError, match="venous AUC"):
            profiles.validated_profile(profile, measurements)


class _NamedTrajectory(np.ndarray):
    """RoadRunner's time-course result exposes column names alongside its ndarray."""

    colnames: list[str]


@pytest.fixture
def trajectory():
    minutes = np.arange(worker.DURATION_H * 60 + 1)
    concentrations = np.tile(np.minimum(minutes, 61)[:, None] / 61_000, (1, len(worker.CONCENTRATIONS)))
    excreta = np.outer(minutes / minutes[-1], [0.03, 0.05, 0.1, 0.2])
    trace = np.column_stack([minutes, concentrations, excreta]).view(_NamedTrajectory)
    trace.colnames = ["time", *worker.CONCENTRATIONS.values(), *(selection for selection, _ in worker.EXCRETA.values())]
    return trace


@pytest.mark.parametrize(("minute", "symbol", "new_value", "cause"), [
    (61, "[Cve_m5]", -1e-6, "negative"),
    (0, "Afeces_cantot", 1e-8, "predose"),
    (3301, "Aurine_m5", 1e-6, "decreased"),
    (10200, "Aurine_m5", 0.06, "mass balance"),
    (10200, "Afeces_cantot", 0.4, "mass balance"),
])
def test_entire_trajectory_is_checked_before_reducing_to_selected_hours(trajectory, minute, symbol, new_value, cause):
    trajectory[minute, trajectory.colnames.index(symbol)] = new_value
    with pytest.raises(ValueError, match=cause):
        worker.summarize(trajectory)


def test_numerical_roundoff_in_solver_units_is_tolerated(trajectory):
    trajectory[60, trajectory.colnames.index("[Cgu_can]")] = -worker.CVODE_ATOL / 2
    column = trajectory.colnames.index("Aurine_m5")
    trajectory[-1, column] = trajectory[-2, column] - worker.CVODE_ATOL / 2
    assert worker.summarize(trajectory)["profile"]["gut_lumen_parent"][1] == 0
