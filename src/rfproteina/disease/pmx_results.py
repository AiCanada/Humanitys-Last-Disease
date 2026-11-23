"""Verify the BAR estimate in a completed pmx nonequilibrium work report."""

from __future__ import annotations

import hashlib
import math
import re
import shlex
from decimal import Decimal
from importlib.metadata import version
from itertools import takewhile
from pathlib import Path

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
# pmx.estimators.kb, in kJ/(mol K).
_PMX_R = 0.00831447215


def _field(lines: list[str], label: str, pattern: str, *, optional: bool = False) -> str | None:
    rows = [line for line in lines if line.startswith(label)]
    if not rows and optional:
        return None
    if len(rows) == 1:
        match = re.fullmatch(pattern, rows[0][len(label):].strip())
        if match:
            return match[1]
    raise ValueError(f"missing, ambiguous or invalid pmx field {label!r}")


def _work(path: Path) -> tuple[dict[str, float], str]:
    data = path.read_bytes()
    values = {}
    for line in data.decode("utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        pieces = line.rsplit(maxsplit=1)
        if len(pieces) != 2:
            raise ValueError("expected a trajectory identifier and an integrated work value")
        identifier, printed = pieces
        identifier = identifier.strip()
        value = float(printed)
        if not identifier or identifier in values or not math.isfinite(value):
            raise ValueError("empty or duplicate trajectory identifier, or nonfinite work")
        values[identifier] = value
    if len(values) < 2:
        raise ValueError("at least two work samples per direction are required to verify uncertainty")
    return values, hashlib.sha256(data).hexdigest()


def _matches_printed(value: float, printed: str) -> bool:
    ref = Decimal(printed)
    rounding = Decimal(1).scaleb(ref.as_tuple().exponent) / 2
    return math.isfinite(value) and abs(Decimal(str(value)) - ref) <= rounding


def read_pmx_bar(report: str | Path, forward: str | Path, reverse: str | Path) -> dict:
    """Audit one pmx BAR report against both original integrated-work files.

    The supported report records the expanded -fA and -fB trajectory lists.
    pmx stores both sets of integrated values in the forward sign convention.
    Only the analytical standard error is reproducible from these data without
    repeating the author's random bootstrap.
    """
    import numpy as np
    from scipy.special import expit

    try:
        from pymbar.other_estimators import bar
        from pymbar.utils import BoundsError, ConvergenceError, ParameterError
    except ImportError as exc:
        raise ValueError("PyMBAR is required for independent work verification") from exc

    raw = Path(report).read_bytes()
    lines = [line.strip() for line in raw.decode("utf-8-sig").splitlines()]
    producer = _field(lines, "# analyze_dhdl.py, pmx version", r"=\s*([\w.+-]+)")
    command = _field(lines, "# command", r"=\s*(.+)")
    tokens = shlex.split(command)
    temp = float(_field(lines, "Temperature", rf":\s*({_NUMBER})\s+K"))
    if not math.isfinite(temp) or temp <= 0:
        raise ValueError("the recorded temperature must be positive and finite")
    forward_count = int(_field(lines, "Number of forward (0->1) trajectories", r":\s*(\d+)"))
    reverse_count = int(_field(lines, "Number of reverse (1->0) trajectories", r":\s*(\d+)"))
    energy_pattern = rf"=\s*({_NUMBER})\s+kJ/mol"
    delta = _field(lines, "BAR: dG", energy_pattern)
    error = _field(lines, "BAR: Std Err (analytical)", energy_pattern)
    bootstrap = _field(lines, "BAR: Std Err (bootstrap)", energy_pattern, optional=True)
    if any(not math.isfinite(float(v)) for v in (delta, error, bootstrap) if v is not None):
        raise ValueError("nonfinite value in the published BAR report")
    if float(error) < 0 or (bootstrap is not None and float(bootstrap) < 0):
        raise ValueError("negative reported BAR uncertainty")

    works_a, sha_a = _work(Path(forward))
    works_b, sha_b = _work(Path(reverse))
    if works_a.keys() & works_b.keys():
        raise ValueError("a trajectory is assigned to both directions")
    for flag, work, count in (("-fA", works_a, forward_count), ("-fB", works_b, reverse_count)):
        if tokens.count(flag) != 1:
            raise ValueError(f"the report must identify the original trajectories for {flag}")
        ids = list(takewhile(lambda arg: not arg.startswith("-"), tokens[tokens.index(flag) + 1:]))
        if len(ids) != count or len(work) != count or set(ids) != work.keys():
            raise ValueError(f"reported and supplied trajectories differ for {flag}")

    thermal_energy = _PMX_R * temp
    wf = np.asarray(list(works_a.values())) / thermal_energy
    wr = -np.asarray(list(works_b.values())) / thermal_energy
    try:
        fitted = bar(wf, wr, compute_uncertainty=False)
    except (BoundsError, ConvergenceError, ParameterError, FloatingPointError) as exc:
        raise ValueError("BAR could not verify the supplied work distributions") from exc
    verified_delta = float(fitted["Delta_f"]) * thermal_energy
    # pmx BAR.calc_err uses the overlap of both work distributions at the fitted
    # free energy. The PyMBAR point estimate uses the same balance equation,
    # while the packages use different analytical variance estimators.
    centered = math.log(forward_count / reverse_count) + np.concatenate((wf, -wr)) - fitted["Delta_f"]
    overlap_sum = math.fsum(expit(centered) * expit(-centered))
    if not math.isfinite(overlap_sum) or overlap_sum <= 0:
        raise ValueError("insufficient numerical overlap to verify the pmx uncertainty")
    variance = 1 / overlap_sum - (1 / forward_count + 1 / reverse_count)
    if variance < -math.ulp(1 / overlap_sum) * 2:
        raise ValueError("invalid pmx analytical variance")
    verified_error = math.sqrt(max(0.0, variance)) * thermal_energy
    if not _matches_printed(verified_delta, delta) or not _matches_printed(verified_error, error):
        raise ValueError("work files disagree with the reported BAR estimate or analytical uncertainty")
    return {
        "method": "pmx_completed_bar", "pmx_version": producer, "verification_engine": f"PyMBAR {version('pymbar')}",
        "scope": "one supplied transformation leg", "temperature_kelvin": temp, "units": "kJ/mol",
        "delta_g_0_to_1": float(delta), "analytical_standard_error": float(error),
        "reported_bootstrap_standard_error": None if bootstrap is None else float(bootstrap),
        "analytical_uncertainty_basis": "pmx BAR.calc_err work-overlap variance",
        "verified_delta_g_0_to_1": verified_delta, "verified_analytical_standard_error": verified_error,
        "forward_count": forward_count, "reverse_count": reverse_count,
        "report_sha256": hashlib.sha256(raw).hexdigest(), "forward_sha256": sha_a, "reverse_sha256": sha_b,
    }
