#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../scripts/rfproteina_exec.sh" training "$0" "$@"'
"""Concatenate and aggregate per-example metrics from multi-job evaluation runs."""

import multiprocessing as mp
import os
import re
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import rootutils

from rfproteina.utils.env import load_dotenv
from rfproteina.utils.logging import RankedLogger, print_df_as_table
from rfproteina.utils.metrics_files import PRIMARY, Shard, job_tables, metrics_snapshot, verified_sources

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

dotenv_path = os.path.join(os.environ["PROJECT_ROOT"], ".env")
load_dotenv(dotenv_path)

from atomworks.io.parser import parse  # noqa: E402  (load_dotenv must run first)
from atomworks.io.utils.selection import get_residue_starts  # noqa: E402

from rfproteina.conditions import SEQUENCE_PROBABILITY_CONDITIONS  # noqa: E402  (load_dotenv must run first)
from rfproteina.inference_utils.foldseek import (  # noqa: E402  (load_dotenv must run first)
    add_diversity_summary_metrics,
    add_foldseek_diversity_to_per_example_df,
    chain_cif_dir,
    cluster_summary_for_group,
)
from rfproteina.inference_utils.metrics_tables import (  # noqa: E402
    archive_job_file,
    locked_metrics_directory,
    read_metrics_csv,
    write_metrics_tables,
)
from rfproteina.inference_utils.pass_rates import pass_attempts  # noqa: E402
from rfproteina.inference_utils.solve_rate_metrics import (  # noqa: E402  (load_dotenv must run first)
    compute_solve_rate_metrics,
)
from rfproteina.metrics.sequence_adherence_metric import (  # noqa: E402  (load_dotenv must run first)
    sequence_probability_values_masks_and_outcomes,
)
from rfproteina.utils.parse_presets import (  # noqa: E402  (load_dotenv must run first)
    ANNOTATED_CIF,
)

logger = RankedLogger(__name__, rank_zero_only=False)

_PROBABILITY_COLUMNS = [
    "dataset_name", "task", "n_designs", "n_paths", "n_readable", "n_with_soft_conditions", "n",
    "absolute_mass_error", "mean_absolute_error", "median_absolute_error",
]


def _parse_sequence_probability_observations(
    task: tuple[str, str, str],
) -> tuple[list[tuple[str, str, str, int, str, str, float, int]], str | None]:
    """Read soft conditioning and outcomes keyed by task, output position, and identity."""
    dataset_name, task_name, designed_path_raw = task
    designed_path = Path(designed_path_raw)
    if not designed_path.is_file():
        return [], f"{designed_path}: file not found"

    try:
        predicted = parse(designed_path, config=ANNOTATED_CIF)["assemblies"]["1"][0]
        values, masks, outcomes = sequence_probability_values_masks_and_outcomes(predicted)
        residues = predicted[get_residue_starts(predicted)]
        positions = list(zip(residues.chain_id, residues.res_id, residues.ins_code, strict=True))
        if len(set(positions)) != len(positions):
            raise ValueError("Output chain ID, residue ID, and insertion code must uniquely identify each residue.")
    except Exception as error:
        return [], f"{designed_path}: {error}"

    soft_residue_indices, soft_token_indices = np.where(masks & (values > 0.0) & (values < 1.0))
    token_names = tuple(condition.token_name for condition in SEQUENCE_PROBABILITY_CONDITIONS)
    return [
        (
            dataset_name,
            task_name,
            str(residues.chain_id[residue_index]),
            int(residues.res_id[residue_index]),
            str(residues.ins_code[residue_index]),
            token_names[token_index],
            float(values[residue_index, token_index]),
            int(outcomes[residue_index, token_index]),
        )
        for residue_index, token_index in zip(soft_residue_indices, soft_token_indices, strict=True)
    ], None


def aggregate_sequence_probability_errors(
    per_example_df: pd.DataFrame,
    workers: int | None = None,
) -> pd.DataFrame:
    """Pool soft probability observations across designs before computing per-task errors.

    Returns:
        One row per dataset/task with n (number of pairs), absolute_mass_error
        (abs(sum(q - p))), mean_absolute_error, and median_absolute_error.
        Coverage counts distinguish missing paths, unreadable structures and
        structures carrying soft conditions. Errors are undefined for zero pairs.
    """
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be positive, got {workers}.")
    if "designed_structure_path" not in per_example_df:
        return pd.DataFrame(columns=_PROBABILITY_COLUMNS)

    tasks = []
    summary = {}
    for _, row in per_example_df.iterrows():
        path = row["designed_structure_path"]
        task_name = row.get("task")
        has_path = isinstance(path, str) and bool(path)
        if not isinstance(task_name, str) or not task_name:
            if not has_path:
                continue
            raise ValueError(f"Sequence-probability aggregation requires a nonempty task label for {path}.")
        dataset_name = row.get("dataset_name", "unknown")
        key = (str(dataset_name) if pd.notna(dataset_name) else "unknown", task_name)
        counts = summary.setdefault(key, {
            "dataset_name": key[0], "task": key[1], "n_designs": 0, "n_paths": 0, "n_readable": 0,
            "n_with_soft_conditions": 0, "n": 0,
        })
        counts["n_designs"] += 1
        if has_path:
            counts["n_paths"] += 1
            tasks.append((*key, path))
    if not tasks:
        return pd.DataFrame(summary.values(), columns=_PROBABILITY_COLUMNS)

    if workers is None:
        allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
        if not allocated_cpus:
            try:
                allocated_cpus = len(os.sched_getaffinity(0))
            except AttributeError:
                allocated_cpus = os.cpu_count() or 1
        workers = min(8, allocated_cpus)
    workers = min(workers, len(tasks))
    if workers == 1:
        parsed = map(_parse_sequence_probability_observations, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
        chunksize = max(1, len(tasks) // (workers * 4))
        parsed = executor.map(_parse_sequence_probability_observations, tasks, chunksize=chunksize)

    observations: list[tuple[str, str, str, int, str, str, float, int]] = []
    errors: list[str] = []
    try:
        for (dataset, task_name, _), (records, error) in zip(tasks, parsed, strict=True):
            observations.extend(records)
            if error is not None:
                errors.append(error)
            else:
                summary[dataset, task_name]["n_readable"] += 1
                summary[dataset, task_name]["n_with_soft_conditions"] += bool(records)
    finally:
        if workers > 1:
            executor.shutdown()
    if errors:
        logger.warning(
            f"Failed to reconstruct sequence-probability observations for {len(errors)} output structure(s). "
            f"First failure: {errors[0]}"
        )

    if not observations:
        return pd.DataFrame(summary.values(), columns=_PROBABILITY_COLUMNS)

    observed = pd.DataFrame(
        observations,
        columns=["dataset_name", "task", "chain_id", "res_id", "ins_code", "token", "probability", "outcome"],
    )
    pairs = observed.groupby(["dataset_name", "task", "chain_id", "res_id", "ins_code", "token"], sort=True)[
        ["probability", "outcome"]
    ].mean()

    for (dataset_name, task_name), group in pairs.groupby(level=["dataset_name", "task"], sort=True):
        deviation = group["outcome"] - group["probability"]
        summary[dataset_name, task_name].update(
            n=len(group),
            absolute_mass_error=float(abs(deviation.sum())),
            mean_absolute_error=float(deviation.abs().mean()),
            median_absolute_error=float(deviation.abs().median()),
        )

    return pd.DataFrame(summary.values(), columns=_PROBABILITY_COLUMNS)


def aggregate_metrics_from_per_example_df(
    per_example_df: pd.DataFrame,
    title: str = "Aggregated Metrics",
    group_by: str = "dataset_name",
) -> pd.DataFrame:
    """Aggregate per-example metrics by a grouping column.

    Args:
        per_example_df: DataFrame with columns including 'example_id', 'dataset_name',
            and metric columns.
        title: Title for the printed table.
        group_by: Column to group by.

    Returns:
        DataFrame with one row per group value, columns: <group_by>, n, and one column per metric.
    """
    # Ensure dataset_name column exists when grouping by it
    if group_by == "dataset_name" and "dataset_name" not in per_example_df.columns:
        per_example_df = per_example_df.copy()
        per_example_df["dataset_name"] = "unknown"
    if group_by == "task" and "task" not in per_example_df:
        per_example_df = per_example_df.assign(task=pd.NA)

    non_metric_cols = {
        "example_id",
        "dataset_name",
        "task",
        "designed_structure_path",
        "sequence_redesigned_structure_path",
        "chain_of_interest",
        "foldseek_cluster_id",
        "seed",
        "ckpt_name",
        "ckpt_path",
        "input_cif_path",
        "input_json_path",
        "output_base_dir",
        "designed_chain_path",
    }

    agg_rows = []
    group_cols = ["dataset_name", "task"] if group_by == "task" and "dataset_name" in per_example_df else [group_by]
    for group_key, group_df in per_example_df.groupby(group_cols, dropna=False):
        metric_cols = [
            c for c in group_df.select_dtypes(include="number")
            if c not in non_metric_cols and not c.startswith("generation.")
        ]

        row: dict = dict(zip(group_cols, group_key, strict=True))
        row["n"] = len(group_df)

        # Carry over identifier columns (non-metric, non-group_by) when they
        # have a unique value within the group (e.g. dataset_name per task).
        identifier_cols = non_metric_cols - {
            group_by,
            "example_id",
            "designed_structure_path",
            "sequence_redesigned_structure_path",
            "foldseek_cluster_id",
            "designed_chain_path",
        }
        for col in identifier_cols:
            if col in group_df.columns:
                unique_vals = group_df[col].dropna().unique()
                if len(unique_vals) == 1:
                    row[col] = unique_vals[0]

        pass_cols = [col for col in group_df if col.endswith(".pass")]
        cluster_summary = cluster_summary_for_group(group_df)
        row.update(cluster_summary)
        for metric_name in pass_cols:
            row[metric_name] = pass_attempts(group_df, metric_name).mean()
        for metric_name in metric_cols:
            if metric_name in pass_cols or metric_name in cluster_summary:
                continue
            values = group_df[metric_name].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) > 0:
                row[metric_name] = float(np.nanmean(values))

        # Dataset-level solve-rate statistics (e.g., percentage of targets with at least n% passing designs)
        if isinstance(group_by, str) and group_by == "dataset_name":
            row.update(compute_solve_rate_metrics(group_df))

        agg_rows.append(row)

    if not agg_rows:
        raise ValueError("No valid metrics were found after aggregation.")

    agg_df = pd.DataFrame(agg_rows)
    print_df_as_table(agg_df, title=title)

    return agg_df


def _source_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    primary = source / PRIMARY
    return [*job_tables(source), *([primary] if primary.is_file() else [])]


def _export_snapshot(source: Path) -> dict:
    if source.is_file():
        return metrics_snapshot(source.parent, [source.name], include_jobs=source.name == PRIMARY)
    return metrics_snapshot(source, [PRIMARY])


def _read_published(path: Path, *, check_jobs: bool = True) -> tuple[pd.DataFrame, list[Shard]]:
    """Read the exact primary named by the receipt, including any standalone rows."""
    snapshot: list[Shard] = []
    frame = read_metrics_csv(path, sources=snapshot)
    carried, blocker = verified_sources(path.parent, snapshot[0][2], check_jobs=check_jobs)
    if blocker:
        raise ValueError(f"Cannot use published metrics ({blocker}). Check the source directory and its receipt.")
    return frame, carried


def _job_file_order(path: Path, live: Path) -> tuple:
    name, _, revision = path.stem.rpartition("_revision")
    numbered = bool(re.fullmatch(r"[0-9]+", revision))
    return (path.stat().st_mtime_ns, name if numbered else path.stem,
            path.parent == live, int(revision) if numbered else 0)


def _records(frame: pd.DataFrame) -> Iterator[tuple]:
    return frame.astype(object).where(frame.notna(), None).itertuples(index=False, name=None)


def _previous_rows(previous: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    """Keep standalone observations, including the multiplicity of anonymous attempts."""
    shared = previous.columns.intersection(jobs.columns).difference(_cluster_columns(previous))
    if shared.empty:
        return previous
    named_jobs = jobs.get("example_id", pd.Series(index=jobs.index, dtype=object)).notna()
    anonymous = jobs.loc[~named_jobs, shared]
    counts = Counter(_records(anonymous))
    keep = previous.get("example_id", pd.Series(index=previous.index, dtype=object)).notna()
    anonymous = previous.loc[~keep, shared]
    for index, values in zip(anonymous.index, _records(anonymous), strict=True):
        if counts[values]:
            counts[values] -= 1
        else:
            keep.at[index] = True
    return previous.loc[keep]


def _cluster_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame if c in {"foldseek_cluster_id", "cluster_coverage", "cluster_diversity"}
            or c.endswith(".clustered_pass_rate")]


def _retain_measured_clusters(merged: pd.DataFrame, sources: list[pd.DataFrame]) -> pd.DataFrame:
    """Reuse assignments only from a single table that measured the entire current cohort."""
    fields = _cluster_columns(merged)
    raw = merged.columns.difference(fields)
    scope = [c for c in ("output_base_dir", "dataset_name", "task") if c in raw]

    def cohorts(frame: pd.DataFrame) -> Iterable[pd.DataFrame]:
        if scope:
            return (group for _, group in frame.groupby(scope, dropna=False, sort=False))
        return (frame,)

    # A cohort signature includes repeated anonymous rows. Cluster 0 in one
    # job and cluster 0 in another have no shared structural interpretation.
    pending = {}
    for group in cohorts(merged):
        rows = list(_records(group[raw]))
        pending[frozenset(Counter(rows).items())] = (group.index, rows)
    empty = tuple(0.0 if field == "cluster_coverage" else np.nan for field in fields)
    restored = [empty] * len(merged)
    for source in sources:
        if "foldseek_cluster_id" not in source or not pending:
            continue
        for group in cohorts(source.reindex(columns=merged.columns)):
            rows = list(_records(group[raw]))
            destination = pending.pop(frozenset(Counter(rows).items()), None)
            if destination is None:
                continue
            by_row = defaultdict(deque)
            for row, evidence in zip(rows, _records(group[fields]), strict=True):
                by_row[row].append(evidence)
            indices, current_rows = destination
            for index, row in zip(indices, current_rows, strict=True):
                restored[index] = by_row[row].popleft()
    merged[fields] = pd.DataFrame(restored, columns=fields, index=merged.index)
    return merged


def _lost_cluster_evidence(previous: pd.DataFrame, current: pd.DataFrame) -> bool:
    if "foldseek_cluster_id" not in previous or not previous["foldseek_cluster_id"].notna().any():
        return False
    if "foldseek_cluster_id" not in current:
        return True
    columns = previous.columns.intersection(current.columns)
    old = previous.loc[previous["foldseek_cluster_id"].notna(), columns]
    new = current.loc[current["foldseek_cluster_id"].notna(), columns]
    return bool(Counter(_records(old)) - Counter(_records(new)))


def _remove_clustered_chains(frame: pd.DataFrame) -> None:
    columns = ["output_base_dir", "dataset_name", "task", "example_id", "designed_chain_path"]
    if not {*columns, "foldseek_cluster_id"} <= set(frame):
        return
    complete = frame["foldseek_cluster_id"].notna().groupby(frame["output_base_dir"]).transform("all")
    for base, dataset, task, example, filename in frame.loc[complete.eq(True), columns].dropna().itertuples(
        index=False, name=None
    ):
        root = (Path(base) / "foldseek_input_chains").resolve()
        expected_dir = chain_cif_dir(Path(base), str(dataset), str(task)).resolve()
        path = Path(filename)
        names = {f"{example}_chain{suffix}" for suffix in (".cif", ".cif.gz", ".cif.zst")}
        if (expected_dir.is_relative_to(root) and path.parent.resolve() == expected_dir
                and path.name in names and path.is_file()):
            path.unlink()


def concatenate_results(
    input: str,
    cluster_with_foldseek: bool = False,
    cleanup: bool = True,
    cleanup_chains: bool = True,
    output_dir: str | None = None,
    foldseek_threads: int | None = None,
    sequence_probability_workers: int | None = None,
) -> None:
    """Concatenate per-example and aggregated metrics from multiple jobs.

    Can also be used to run foldseek post-hoc by calling on the output dataframe.

    Args:
        input: Path to either the directory containing the individual job result files,
            or the already-concatenated per_example_metrics.csv file.
        cluster_with_foldseek: Whether to cluster results using Foldseek.
        cleanup: Archive individual job files after a successful in-place concatenation.
        cleanup_chains: Remove fully assigned chain exports after a successful in-place clustering.
        output_dir: Directory where results should be saved. Defaults to the input directory
            (or parent directory when input is a file). Exporting elsewhere preserves source files.
        foldseek_threads: Number of CPU threads to pass to foldseek. Defaults to all
            available CPUs. Set explicitly when running multiple clustering jobs in parallel
            on the same node to avoid resource contention.
        sequence_probability_workers: Number of processes used to reconstruct probability
            observations from output CIFs. Defaults to up to eight CPUs from the SLURM allocation
            or process affinity; set explicitly to use more.
    """
    source = Path(input)
    if not source.exists():
        raise FileNotFoundError(f"Metrics input does not exist: {source}")
    source_dir = source.parent if source.is_file() else source
    output_path = Path(output_dir) if output_dir is not None else source_dir
    in_place = source_dir.resolve() == output_path.resolve()
    with locked_metrics_directory(output_path):
        _concatenate_results(
            source, output_path, in_place=in_place, cluster_with_foldseek=cluster_with_foldseek,
            cleanup=cleanup, cleanup_chains=cleanup_chains, foldseek_threads=foldseek_threads,
            sequence_probability_workers=sequence_probability_workers,
        )


def _concatenate_results(
    source: Path, output_path: Path, *, in_place: bool, cluster_with_foldseek: bool, cleanup: bool,
    cleanup_chains: bool, foldseek_threads: int | None, sequence_probability_workers: int | None,
) -> None:
    snapshot = None if in_place else _export_snapshot(source)
    single_source = None
    published = None
    per_example_files = []
    sources: list[Shard] = []
    if source.is_file():
        if source.name == PRIMARY:
            concatenated_per_example, sources = _read_published(source)
        else:
            concatenated_per_example = read_metrics_csv(source, sources=sources)
        single_source = source
    else:
        per_example_files = [f for f in _source_files(source) if f.name != "per_example_metrics.csv"]
        # Order oldest→newest so that, when deduplicating by example_id below, the most recent
        # metrics for each design win (a rerun's fresh rows override stale ones).
        per_example_files.sort(key=lambda path: _job_file_order(path, source))

        existing = source / "per_example_metrics.csv"
        if not per_example_files and not existing.is_file():
            raise FileNotFoundError(f"No per-example metric files found in {source}")

        logger.info(f"Found {len(per_example_files)} per-example metric files to concatenate")

        # 1. Concatenate per-example metrics
        if not per_example_files:
            single_source = existing
            concatenated_per_example, sources = _read_published(existing)
        else:
            all_per_example_dfs = [read_metrics_csv(file, sources=sources) for file in per_example_files]
            concatenated_per_example = pd.concat(all_per_example_dfs, ignore_index=True)

        # Keep a retry as a complete observation. Filling its missing fields
        # from an older run can combine a stale pass and a new failing score.
        if per_example_files and existing.is_file():
            published, _ = _read_published(existing, check_jobs=False)
            previous = _previous_rows(published, concatenated_per_example)
            concatenated_per_example = pd.concat([previous, concatenated_per_example], ignore_index=True)
        if per_example_files and "example_id" in concatenated_per_example.columns:
            n_before = len(concatenated_per_example)
            identity = [c for c in ("dataset_name", "task", "example_id", "output_base_dir", "designed_structure_path")
                        if c in concatenated_per_example]
            named = concatenated_per_example["example_id"].notna()
            duplicate = concatenated_per_example.duplicated(identity, keep="last")
            concatenated_per_example = concatenated_per_example.loc[~(named & duplicate)].reset_index(drop=True)
            n_dropped = n_before - len(concatenated_per_example)
            if n_dropped:
                logger.info(f"Replaced {n_dropped} earlier observations with their latest per-design rows.")
        if per_example_files and not cluster_with_foldseek and "foldseek_cluster_id" in concatenated_per_example:
            candidates = list(zip(per_example_files, all_per_example_dfs, strict=True))
            if published is not None:
                candidates.append((existing, published))
            candidates.sort(key=lambda item: _job_file_order(item[0], source), reverse=True)
            concatenated_per_example = _retain_measured_clusters(concatenated_per_example, [df for _, df in candidates])

    # 2. (Optional) Cluster results using Foldseek
    if cluster_with_foldseek:
        prior = published if published is not None else concatenated_per_example
        concatenated_per_example = concatenated_per_example.drop(columns=_cluster_columns(concatenated_per_example))
        # Chains dirs are discovered from output_base_dir in the DataFrame (no hardcoded path).
        concatenated_per_example = add_foldseek_diversity_to_per_example_df(
            concatenated_per_example, threads=foldseek_threads
        )
        if ("foldseek_cluster_id" in prior and prior["foldseek_cluster_id"].notna().any()
                and not concatenated_per_example["foldseek_cluster_id"].notna().any()):
            raise ValueError(
                "Foldseek returned no assignments for a previously clustered run. Check the chain exports."
            )
        concatenated_per_example = add_diversity_summary_metrics(concatenated_per_example)

    probability_errors = aggregate_sequence_probability_errors(
        concatenated_per_example,
        workers=sequence_probability_workers,
    )
    agg_df = aggregate_metrics_from_per_example_df(concatenated_per_example, title="Final Aggregated Metrics")
    per_task_df = aggregate_metrics_from_per_example_df(
        concatenated_per_example, title="Per-Task Metrics", group_by="task"
    )
    tables = {"aggregated_metrics.csv": agg_df, "per_task_metrics.csv": per_task_df}
    if not probability_errors.empty or (output_path / "sequence_probability_errors.csv").exists():
        tables["sequence_probability_errors.csv"] = probability_errors
    per_example_output = output_path / "per_example_metrics.csv"
    if cluster_with_foldseek or single_source is None or single_source.resolve() != per_example_output.resolve():
        tables[per_example_output.name] = concatenated_per_example
    if not in_place and snapshot != _export_snapshot(source):
        raise RuntimeError("Source metrics changed during export. Retry with the completed input.")
    archive = None
    if per_example_output.name in tables and per_example_output.exists():
        old_output = published if published is not None and in_place else read_metrics_csv(per_example_output)
        if _lost_cluster_evidence(old_output, concatenated_per_example):
            archive = "previous_reports"
    write_metrics_tables(output_path, tables, archive_replaced=archive, sources=sources)
    logger.info(f"Saved metrics for {len(concatenated_per_example)} designs in {output_path}")

    if in_place and cluster_with_foldseek and cleanup_chains:
        _remove_clustered_chains(concatenated_per_example)

    if in_place and cleanup and per_example_files:
        job_files_dir = output_path / "individual_job_results"
        logger.info(f"Moving individual job files to {job_files_dir}")
        for file in per_example_files:
            if file.parent == source:
                archive_job_file(file, job_files_dir)
            companion = source / file.name.replace("per_example_metrics", "aggregated_metrics", 1)
            if companion.is_file():
                archive_job_file(companion, job_files_dir)


def main() -> None:
    fire.Fire(concatenate_results)


if __name__ == "__main__":
    main()
