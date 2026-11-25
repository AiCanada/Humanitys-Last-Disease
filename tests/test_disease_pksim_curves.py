"""A native PK-Sim result must retain analyte, location, individual, time grid and units."""

from __future__ import annotations

import csv
import io

import pytest

from rfproteina.disease import pksim_curves as curves

PLASMA = "Organism|PeripheralVenousBlood|Aciclovir|Plasma (Peripheral Venous Blood)"
LIVER = "Organism|Liver|Aciclovir|Tissue"


def export(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(columns)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def native(rows):
    return export(["IndividualId", "Time [min]", f"{PLASMA} [µmol/l]"], rows)


def test_upstream_output_and_observed_grid_summary():
    # First three rows of OSPSuite-R/inst/extdata/SimResults.csv, blob
    # d37f2d3170a7f5b36eef7dabf73c369a320383af.
    raw = native([(0, 0, 0), (0, 1, 3.254674), (0, 2, 9.100459)])
    (result,) = curves.extract(raw, (PLASMA,))
    assert result == {
        "output_path": PLASMA,
        "individual_id": 0,
        "time_unit": "min",
        "concentration_unit": "µmol/l",
        "time": [0, 1, 2],
        "concentration": [0, 3.254674, 9.100459],
        "point_count": 3,
        "grid_tmax": 2,
        "grid_cmax": 9.100459,
        "auc_start": 0,
        "auc_end": 2,
        "auc_unit": "(µmol/l)*min",
        "auc_trapezoidal": pytest.approx(7.8049035),
    }


def test_nonuniform_nonzero_start_exact_output_order_native_units_and_first_tied_peak():
    raw = export(
        ["IndividualId", "Time [h]", f"{PLASMA} [µmol/l]", "Organism|Lumen|Fraction", f"{LIVER} [ng/mL]"],
        [(0, 2, 10, 0, 1), (0, 4, 10, 0.25, 3), (0, 5, 0, 1, 9)],
    )
    liver, plasma = curves.extract(raw, (LIVER, PLASMA))
    assert (liver["output_path"], liver["grid_tmax"], liver["auc_start"], liver["auc_end"]) == (LIVER, 5, 2, 5)
    assert (liver["auc_trapezoidal"], liver["auc_unit"]) == (10, "(ng/mL)*h")
    assert (plasma["grid_tmax"], plasma["grid_cmax"], plasma["auc_trapezoidal"]) == (2, 10, 25)


@pytest.mark.parametrize(
    "rows",
    [
        [(0, 0, 0)],
        [(0, 0, 0), (1, 1, 3)],
        [("1e2", 0, 0), ("1e2", 1, 3)],
        [(0, 1, 0), (0, 1, 3)],
        [(0, 1, 1), (0, 0, 3)],
        [(0, -1, 0), (0, 1, 3)],
        [(0, 0, 0), (0, "nan", 3)],
        [(0, 0, 0), (0, 1, -0.01)],
        [(0, 0, 0), (0, 1, "nan")],
        [(0, 0, 0), (0, 1, "1_000")],
        [(0, 0, 0), (0, 1, "1e400")],
        [(0, 0, 0), (0, 1, "-1e-900")],
        [(0, 0, 0), (0, 1, "")],
        [(0, 0, 0), (0, 1, 1, 2)],
        [(0, 0, "1e308"), (0, 10, "1e308")],  # valid values, unrepresentable AUC
    ],
)
def test_invalid_individual_time_grid_or_numeric_data(rows):
    with pytest.raises(curves.CurveError, match="^pksim_invalid_export$"):
        curves.extract(native(rows), (PLASMA,))


@pytest.mark.parametrize(
    "columns,expected",
    [
        (["Id", "Time [min]", f"{PLASMA} [µmol/l]"], "pksim_invalid_export"),
        (["IndividualId", "Time", PLASMA], "pksim_unsupported_unit"),
        (["IndividualId", "Time [s]", PLASMA], "pksim_unsupported_unit"),
        (["IndividualId", "Time [min]", f"{PLASMA} [µmol]"], "pksim_unsupported_unit"),
        (["IndividualId", "Time [min]", f"{PLASMA} [ml/min]"], "pksim_unsupported_unit"),
        (["IndividualId", "Time [min]", f"{PLASMA} [µmol/l]", f"{PLASMA} [ng/ml]"], "pksim_invalid_export"),
        (["IndividualId", "Time [min]", f"{LIVER} [µmol/l]"], "pksim_output_missing"),
    ],
)
def test_curve_identity_and_dimensions(columns, expected):
    with pytest.raises(curves.CurveError, match=f"^{expected}$"):
        curves.extract(export(columns, [(0, 0, 1), (0, 1, 3)]), (PLASMA,))


def test_export_limits_are_rejections_not_truncation(monkeypatch):
    raw = native([(0, 0, 1), (0, 1, 2), (0, 2, 3)])
    monkeypatch.setattr(curves, "MAX_POINTS", 2)
    with pytest.raises(curves.CurveError, match="^pksim_invalid_export$"):
        curves.extract(raw, (PLASMA,))
    monkeypatch.setattr(curves, "MAX_EXPORT_BYTES", len(raw) - 1)
    with pytest.raises(curves.CurveError, match="^pksim_invalid_export$"):
        curves.extract(raw, (PLASMA,))


def test_utf8_and_unique_selections():
    raw = native([(0, 0, 1), (0, 1, 2)])
    for data, paths in [(b"\xff", (PLASMA,)), (raw, ()), (raw, (PLASMA, PLASMA))]:
        with pytest.raises(curves.CurveError, match="^pksim_invalid_export$"):
            curves.extract(data, paths)
