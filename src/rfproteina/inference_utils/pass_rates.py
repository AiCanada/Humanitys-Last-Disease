"""Pass-rate denominators for post-generation metrics tables."""

import pandas as pd

from rfproteina.utils.metric_tags import recorded_metric_tags


def _binary(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(series):
        series = series.astype("string").str.strip().str.lower().replace({"true": "1", "false": "0"})
    numbers = pd.to_numeric(series, errors="coerce")
    return numbers.where(numbers.isin([0, 1])).astype(float)


def pass_attempts(frame: pd.DataFrame, column: str) -> pd.Series:
    """One for a confirmed pass and zero for other attempts in an evaluated cohort.

    A cohort shares dataset, task and recorded metric tags. A cohort with no
    evidence this criterion ran has an undefined rate. Within an evaluated
    cohort, designs with missing results remain in the denominator.
    """
    flags = _binary(frame[column])
    attempted = flags.notna()
    passed = flags.eq(1)
    presence = column.removesuffix(".pass") + ".all_metrics_present"
    if presence in frame:
        present = _binary(frame[presence])
        attempted |= present.notna()
        passed &= present.eq(1)
    groups = [frame[key] for key in ("dataset_name", "task") if key in frame]
    if "metrics_tags" in frame:
        tag_sets = frame["metrics_tags"].fillna("").astype(str).map(recorded_metric_tags).map(frozenset)
        groups.append(tag_sets)
    if groups:
        covered = attempted.groupby(groups, sort=False, dropna=False).transform("any")
    else:
        covered = pd.Series(attempted.any(), index=frame.index)
    return passed.astype(float).where(covered)
