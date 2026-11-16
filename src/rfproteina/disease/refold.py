"""The refolding pass criteria this repository declares, read off the metrics table.

``configs/metrics/pass_criteria/all_pass_criteria.yaml`` declares eleven named
criteria -- ``unconditional``, ``ppi``, ``antibody``, ``dna``,
``motif_scaffold``, ``internal_coords``, ``sm_binder``, ``ame``,
``ame_no_ligand``, ``disco`` and ``developability`` -- each a set of metric
columns with a bound on every one. ``rfd4-evaluate`` evaluates them and writes a
``<name>.pass`` column for each into ``per_example_metrics.csv``.

The disease screen read two of them. ``ppi.pass`` became ``ppi_pass`` and
``pre_oracle_pass`` was read directly; the other nine were computed, written
into the file the screen opens, and thrown away. ``developability`` is the one
that stings: it bounds Rosetta SAP and hydrophobic exposed surface area
alongside the same GRAVY and instability index this screen recomputes from the
sequence, so the screen was recomputing half of a criterion while discarding
the whole of it.

Worse, a bare ``ppi_pass`` of ``false`` said nothing about *which* bound was
missed. The three numbers behind it -- ``ptm_coi`` at 80, ``min_ipae_coi`` at
1.5, the aligned binder RMSD at 2.5 -- decide very different things about a
design, and a reader who wants to know whether to re-fold or redesign has to
know which one it was.

The criteria, their columns and their cuts are read from that config file at
screen time rather than copied into this module. A cut written here would
become a second opinion about the same bar the moment someone edited the
config, which is the whole failure this screen keeps trying to avoid.

This reports and judges nothing. ``ppi_pass`` has always been recorded without
moving the call, and the other ten are recorded on the same terms: they are the
evaluation's verdict, not this screen's, and a screen that silently adopted
nine new ways to fail would change every stored call that had none of them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from rfproteina.disease.publish import _repository_root

#: Where the criteria live, relative to the repository root.
CRITERIA_FILE = Path("configs") / "metrics" / "pass_criteria" / "all_pass_criteria.yaml"
#: Criteria one config may declare. The file holds eleven; a file past this is
#: not that file.
CRITERIA_LIMIT = 64
#: Thresholds one criterion may bound.
THRESHOLD_LIMIT = 16
#: Bounds this reads. ``min_of`` and the other reductions are declared but
#: commented out in the config, and a reduction read as a plain bound would
#: compare the wrong number, so an unknown form is named instead of guessed.
_BOUNDS = {
    "max": ("max", False),
    "max_exclusive": ("max", True),
    "min": ("min", False),
    "min_exclusive": ("min", True),
}
#: A same-file Hydra interpolation, which ``internal_coords`` uses to borrow
#: ``motif_scaffold``'s thresholds. Resolved by name; any other interpolation
#: is reported unresolved rather than parsed.
_INTERPOLATION = re.compile(r"^\$\{metrics\.pass_criteria\.([A-Za-z0-9_]+)\.metric_thresholds\}$")
_ABSENT_CELL = frozenset({"", "nan", "none", "null", "na", "n/a", "."})
_PASS_CELLS = {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False}
_TAGS_COLUMN = "metrics_tags"


def refold_report(rows: Sequence[Mapping[str, str]], *, root: Path | None = None) -> dict[str, object]:
    """Every declared criterion, and what this run's metrics table says about it.

    ``rows`` are the metrics rows the screen already read. A criterion whose
    ``<name>.pass`` column is absent was not evaluated, which on a run with no
    oracle configured is every one of them; that is reported once at the top
    rather than eleven times.
    """
    declared = read_criteria(root)
    if declared is None:
        return {"refold_blocker": "criteria_not_found"}
    if not declared:
        return {"refold_blocker": "criteria_not_declared"}
    criteria: list[dict[str, object]] = []
    evaluated = 0
    for name, thresholds in declared.items():
        found = _criterion(name, thresholds, rows)
        if found["state"] in {"pass", "fail"}:
            evaluated += 1
        criteria.append(found)
    record: dict[str, object] = {
        "criteria": criteria,
        "declared": len(declared),
        "evaluated": evaluated,
        "criteria_file": CRITERIA_FILE.as_posix(),
    }
    if not evaluated:
        # One reason for the run, not one per criterion. The refolding oracle
        # writes every one of these columns, so none of them present is a
        # statement about the oracle rather than about eleven criteria.
        record["refold_blocker"] = "not_evaluated"
    return record


def read_criteria(root: Path | None = None) -> dict[str, object] | None:
    """Each criterion's thresholds, by name, or None when the config is unreadable.

    Read from the repository rather than copied here: a cut written into this
    module would become a second opinion about the same bar the moment the
    config changed.
    """
    base = root if root is not None else _repository_root()
    if base is None:
        return None
    try:
        payload = yaml.safe_load((base / CRITERIA_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError, RecursionError):
        return None
    if not isinstance(payload, dict):
        return None
    raw = {
        name: entry.get("metric_thresholds")
        for name, entry in payload.items()
        if isinstance(name, str) and name and isinstance(entry, dict)
    }
    if len(raw) > CRITERIA_LIMIT:
        return None
    return {name: _resolved(value, raw) for name, value in list(raw.items())[:CRITERIA_LIMIT]}


def _resolved(value: object, raw: Mapping[str, object]) -> object:
    """One criterion's thresholds, following a same-file interpolation once."""
    if not isinstance(value, str):
        return value
    match = _INTERPOLATION.fullmatch(value.strip())
    if match is None:
        return value
    borrowed = raw.get(match.group(1))
    # One hop only. A chain of interpolations is not something this config has
    # and following one blindly is how a cycle becomes a hang.
    return borrowed if isinstance(borrowed, dict) else value


def _criterion(
    name: str,
    thresholds: object,
    rows: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    """One criterion: its stored verdict, its bounds, and which bounds were missed."""
    found: dict[str, object] = {"name": name, "state": _pass_state(f"{name}.pass", rows)}
    if not isinstance(thresholds, dict):
        # An interpolation this module does not resolve, or a form the config
        # never had. The verdict column above still stands on its own.
        found["thresholds_blocker"] = "thresholds_not_resolved"
        return found
    bounds: list[dict[str, object]] = []
    failed: list[str] = []
    for column, declared in list(thresholds.items())[:THRESHOLD_LIMIT]:
        if not isinstance(column, str) or not column:
            continue
        bound = _bound(column, declared, rows)
        bounds.append(bound)
        if bound.get("met") is False:
            failed.append(column)
    if bounds:
        found["thresholds"] = bounds
    if failed:
        # Which bound was missed. A bare false said only that something was,
        # and ptm_coi and an aligned RMSD ask for different work.
        found["failed_thresholds"] = failed
    return found


def _bound(column: str, declared: object, rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    """One metric column's cut, the worst value the run reached, and whether it held."""
    found: dict[str, object] = {"metric": column}
    if not isinstance(declared, dict):
        found["bound"] = "unsupported"
        return found
    shape = [(key, value) for key, value in declared.items() if key in _BOUNDS]
    if len(shape) != 1:
        # A reduction such as min_of, or two bounds on one column. Reading
        # either as a plain cut would compare the wrong number.
        found["bound"] = "unsupported"
        return found
    key, raw_cut = shape[0]
    side, exclusive = _BOUNDS[key]
    cut = _number(raw_cut)
    if cut is None:
        found["bound"] = "unsupported"
        return found
    found["bound"] = key
    found["cut"] = cut
    measured = _worst(column, rows, side)
    found["measured"] = measured
    if measured is None:
        found["met"] = None
        return found
    if side == "max":
        found["met"] = measured < cut if exclusive else measured <= cut
    else:
        found["met"] = measured > cut if exclusive else measured >= cut
    return found


def _worst(column: str, rows: Sequence[Mapping[str, str]], side: str) -> float | None:
    """The least favourable value any design reached, or None when none carried it.

    The worst rather than the first or the mean, so a report cannot read milder
    than the designs it summarizes. The same rule ``rollup_axis`` applies to an
    axis across a run.
    """
    values = [value for value in (_cell(row.get(column)) for row in rows) if value is not None]
    if not values:
        return None
    return max(values) if side == "max" else min(values)


def _pass_state(column: str, rows: Sequence[Mapping[str, str]]) -> str:
    """``pass``, ``fail`` or ``not_evaluated`` for one criterion's own column.

    A false anywhere is a fail for the run: the criterion is a per-design
    verdict and a run is not a passer because one of its designs was.
    """
    seen = False
    for row in rows:
        raw = row.get(column)
        if raw is None:
            continue
        text = str(raw).strip().casefold()
        if text in _ABSENT_CELL:
            continue
        value = _PASS_CELLS.get(text)
        if value is None:
            number = _number(text)
            value = None if number is None else bool(number)
        if value is False:
            return "fail"
        if value is True:
            seen = True
    return "pass" if seen else "not_evaluated"


def applicable_tags(rows: Sequence[Mapping[str, str]]) -> list[str]:
    """The metrics tags this run recorded, which decide which criteria applied.

    Empty on every run that has not been evaluated, which is why a criterion
    with no column is reported as not evaluated rather than as not applicable:
    without tags there is nothing to tell the two apart.
    """
    found: list[str] = []
    for row in rows:
        raw = row.get(_TAGS_COLUMN)
        if not isinstance(raw, str):
            continue
        for piece in re.split(r"[,;\s\[\]'\"]+", raw):
            token = piece.strip()
            if token and token not in found:
                found.append(token)
    return found


def _cell(raw: object) -> float | None:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.casefold() in _ABSENT_CELL:
        return None
    return _number(text)


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value) if _finite(float(value)) else None
    if not isinstance(value, str):
        return None
    try:
        found = float(value.strip())
    except ValueError:
        return None
    return found if _finite(found) else None


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
