"""Read stored refold flags and compare applicable rows with the checkout's declared bounds.

Each criterion reports its coverage and keeps the evaluation verdict separate from
comparisons against the repository defaults. A run may have used override thresholds.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from rfproteina.disease.publish import _repository_root
from rfproteina.utils.metric_tags import recorded_metric_tags
from rfproteina.utils.metric_thresholds import COMPARISONS, MetricThreshold, finite_number

#: Where the criteria live, relative to the repository root.
CRITERIA_FILE = Path("configs") / "metrics" / "pass_criteria" / "all_pass_criteria.yaml"
#: Criteria one config may declare. The file holds eleven; a file past this is
#: not that file.
CRITERIA_LIMIT = 64
#: Thresholds one criterion may bound.
THRESHOLD_LIMIT = 16
#: A same-file Hydra interpolation, which ``internal_coords`` uses to borrow
#: ``motif_scaffold``'s thresholds. Resolved by name; any other interpolation
#: is reported unresolved rather than parsed.
_INTERPOLATION = re.compile(r"^\$\{metrics\.pass_criteria\.([A-Za-z0-9_]+)\.metric_thresholds\}$")
_ABSENT_CELL = frozenset({"", "nan", "none", "null", "na", "n/a", "."})
_PASS_CELLS = {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False}
_TAGS_COLUMN = "metrics_tags"


_CONFIDENCE = re.compile(
    r"plddt|iptm|(?<![A-Za-z])ptm(?![A-Za-z])|ipae|(?<![A-Za-z])pae(?![A-Za-z])|confidence", re.IGNORECASE,
)
_TRACE_EXCLUDED = re.compile(r"sequence|path|cif|structure", re.IGNORECASE)


def trace_metric_kind(key: str) -> str | None:
    """Classify columns copied into a per-design trace so the reader checks their headers."""
    if not key or _TRACE_EXCLUDED.search(key):
        return None
    if key == "seed" or key.startswith("generation."):
        return "generation"
    return "confidence" if _CONFIDENCE.search(key) else None


def refold_report(rows: Sequence[Mapping[str, str]], *, root: Path | None = None) -> dict[str, object]:
    """Every declared criterion, and what this run's metrics table says about it.

    ``rows`` are the metrics rows the screen already read. A criterion whose
    ``<name>.pass`` column is absent was not evaluated, which on a run with no
    oracle configured is every one of them; that is reported once at the top
    rather than eleven times.
    """
    definitions = _definitions(root)
    if definitions is None:
        return {"refold_blocker": "criteria_not_found"}
    if not definitions:
        return {"refold_blocker": "criteria_not_declared"}
    declared = _thresholds(definitions)
    criteria: list[dict[str, object]] = []
    evaluated = 0
    for name, thresholds in declared.items():
        found = _criterion(name, thresholds, _criterion_rows(name, definitions[name], rows))
        if found["state"] in {"pass", "fail"}:
            evaluated += 1
        criteria.append(found)
    record: dict[str, object] = {
        "criteria": criteria,
        "declared": len(declared),
        "evaluated": evaluated,
        "criteria_file": CRITERIA_FILE.as_posix(),
        "threshold_source": "repository_defaults",
    }
    if not evaluated:
        # One reason for the run, not one per criterion. The refolding oracle
        # writes every one of these columns, so none of them present is a
        # statement about the oracle rather than about eleven criteria.
        record["refold_blocker"] = (
            "metrics_incomplete" if any(item["state"] == "incomplete" for item in criteria) else "not_evaluated"
        )
    return record


def read_criteria(root: Path | None = None) -> dict[str, object] | None:
    """Each criterion's thresholds, by name, or None when the config is unreadable.

    Read from the repository rather than copied here: a cut written into this
    module would become a second opinion about the same bar the moment the
    config changed.
    """
    definitions = _definitions(root)
    return None if definitions is None else _thresholds(definitions)


def _definitions(root: Path | None) -> dict[str, dict] | None:
    base = root if root is not None else _repository_root()
    if base is None:
        return None
    try:
        payload = yaml.safe_load((base / CRITERIA_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError, RecursionError):
        return None
    if not isinstance(payload, dict):
        return None
    entries = {
        name: entry
        for name, entry in payload.items()
        if isinstance(name, str) and name and isinstance(entry, dict)
    }
    return entries if len(entries) <= CRITERIA_LIMIT else None


def _thresholds(definitions: Mapping[str, dict]) -> dict[str, object]:
    raw = {name: entry.get("metric_thresholds") for name, entry in definitions.items()}
    return {name: _resolved(value, raw) for name, value in raw.items()}


def _criterion_rows(name: str, definition: Mapping, rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    """Use explicit evaluator output first, and the configured tags for unscored rows."""
    column, presence = f"{name}.pass", f"{name}.all_metrics_present"

    def reported(row: Mapping[str, str]) -> bool:
        return any(str(row.get(key, "")).strip().casefold() not in _ABSENT_CELL for key in (column, presence))

    any_reported = any(reported(row) for row in rows)
    required_any = set(definition.get("required_tags_any", ()))
    required_all = set(definition.get("required_tags_all", ()))
    prohibited = set(definition.get("prohibited_tags", ()))
    selected = []
    for row in rows:
        if reported(row):
            selected.append(row)
            continue
        if not str(row.get(_TAGS_COLUMN) or "").strip():
            # Legacy tables lack applicability metadata. If this criterion ran,
            # a blank result in a peer row cannot establish a whole-run pass.
            if any_reported:
                selected.append(row)
            continue
        tags = set(applicable_tags([row]))
        if (not required_any or tags & required_any) and required_all <= tags and not tags & prohibited:
            selected.append(row)
    return selected


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
    column = f"{name}.pass"
    flags = [row_pass_flag(row, column) for row in rows]
    measured = sum(value is not None for value in flags)
    if False in flags:
        state = "fail"
    elif flags and measured == len(flags):
        state = "pass"
    elif any(str(row.get(key) or "").strip() for row in rows for key in (column, f"{name}.all_metrics_present")):
        state = "incomplete"
    else:
        state = "not_evaluated"
    found: dict[str, object] = {"name": name, "state": state, "rows_measured": measured, "rows_expected": len(rows)}
    if not isinstance(thresholds, dict):
        # An interpolation this module does not resolve, or a form the config
        # never had. The verdict column above still stands on its own.
        found["thresholds_blocker"] = "thresholds_not_resolved"
        return found
    bounds: list[dict[str, object]] = []
    failed: list[str] = []
    if len(thresholds) > THRESHOLD_LIMIT:
        found["thresholds_blocker"] = "thresholds_not_resolved"
        return found
    for column, declared in thresholds.items():
        if not isinstance(column, str) or not column:
            continue
        comparisons = _bounds(column, declared, rows)
        bounds.extend(comparisons)
        if any(bound.get("met") is False for bound in comparisons):
            failed.append(column)
    if bounds:
        found["thresholds"] = bounds
    if failed:
        # Which bound was missed. A bare false said only that something was,
        # and ptm_coi and an aligned RMSD ask for different work.
        found["failed_thresholds"] = failed
    return found


def _bounds(column: str, declared: object, rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    """Evaluate each bound against the least favorable measured design."""
    try:
        spec = MetricThreshold.parse(column, declared)
    except ValueError:
        return [{"metric": column, "bound": "unsupported"}]
    values = [value for row in rows if (value := spec.measurement(row)) is not None]
    results = []
    for bound, cut in spec.bounds:
        worst = min if bound.startswith("min") else max
        measured = worst(values) if values else None
        met = COMPARISONS[bound](measured, cut) if measured is not None else None
        if met and len(values) < len(rows):
            met = None
        record = {"metric": column, "bound": bound, "cut": cut, "measured": measured, "met": met}
        if spec.reduction is not None:
            record.update(reduction=spec.reduction, input_metrics=list(spec.keys))
        results.append(record)
    return results


def binary_flag(raw: object) -> bool | None:
    """Read a CSV boolean or a finite numeric spelling of exactly zero or one."""
    if not isinstance(raw, str):
        return None
    text = raw.strip().casefold()
    if text in _PASS_CELLS:
        return _PASS_CELLS[text]
    number = finite_number(text)
    return bool(number) if number in (0, 1) else None


def row_pass_flag(row: Mapping[str, str], column: str) -> bool | None:
    """A stored zero with missing criterion inputs is an indeterminate evaluation."""
    if column.endswith(".pass"):
        presence = column.removesuffix(".pass") + ".all_metrics_present"
        if presence in row and binary_flag(row[presence]) is not True:
            return None
    return binary_flag(row.get(column))


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
        for token in recorded_metric_tags(raw):
            if token not in found:
                found.append(token)
    return found
