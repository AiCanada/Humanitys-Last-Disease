"""Foldseek diversity clustering: subprocess primitive, DataFrame wrappers, DDP entry.

Bulk clustering
---------------
:py:func:`cluster_structures` accepts the ``foldseek_input_chains/`` subdirectory produced by
:py:func:`~rfproteina.inference_utils.foldseek.chain_cif_dir` and returns ``{example_id: cluster_id}``.
"""

import os
import subprocess
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch.distributed as dist
from atomworks.io.utils.compression import transfer_with_compression
from atomworks.io.utils.io_utils import infer_pdb_file_type
from atomworks.ml.utils.misc import hash_sequence

from rfproteina.inference_utils.pass_rates import pass_attempts
from rfproteina.utils.logging import RankedLogger
from rfproteina.utils.structure_files import iter_structure_files, structure_file_stem

logger = RankedLogger(__name__, rank_zero_only=False)


def diversity_foldseek(
    list_of_cif_paths: list[str],
    tm_threshold: float = 0.5,
    alignment_type: int = 1,
    min_seq_id: float = 0.0,
    timeout: float | None = None,
    threads: int | None = None,
) -> list[int] | None:
    """Cluster single-chain structures with Foldseek; return per-structure cluster IDs.

    Caller must guarantee each input file is already a single-chain structure.
    Files are decoded into a temporary staging dir and ``easy-cluster``
    is invoked directly. The staging dir is created on tmpfs (``$TMPDIR`` /
    ``/tmp``) and removed on exit.

    Args:
        list_of_cif_paths: paths to single-chain structures to cluster.
        tm_threshold: TM-score threshold for clustering.
        alignment_type: 1 (structure) or 2 (structure+sequence).
        min_seq_id: required iff ``alignment_type == 2``.
        timeout: total wall budget in seconds.
        threads: ``--threads`` for foldseek; defaults to foldseek's own
            default (which itself defaults to all cores).
    """
    assert alignment_type in (1, 2), f"alignment_type {alignment_type} not valid"
    if alignment_type == 1:
        assert min_seq_id == 0.0, f"alignment_type == 1 only admits min_seq_id = 0, got {min_seq_id}"
    if alignment_type == 2:
        assert min_seq_id > 0.0, f"alignment_type == 2 requires min_seq_id > 0, got {min_seq_id}"
    if not list_of_cif_paths:
        return None

    foldseek_exec = os.getenv("FOLDSEEK_EXEC")
    if foldseek_exec is None:
        raise ValueError("FOLDSEEK_EXEC environment variable not set")

    with tempfile.TemporaryDirectory(prefix="foldseek_diversity_") as path_tmp:
        path_designable = os.path.join(path_tmp, "samples")
        os.makedirs(path_designable)

        # staged_stems[i] is the basename Foldseek will emit for input i; defining
        # the convention once here means a future rename only touches one site.
        staged_stems: list[str] = []
        for i, fcif in enumerate(list_of_cif_paths):
            stem = f"{i + 1}"
            file_type = infer_pdb_file_type(fcif)
            transfer_with_compression(fcif, Path(path_designable) / f"{stem}.{file_type}")
            staged_stems.append(stem)

        command = [
            foldseek_exec,
            "easy-cluster",
            path_designable,
            os.path.join(path_tmp, "res"),
            path_tmp,
            "--alignment-type",
            str(alignment_type),
            "--cov-mode",
            "0",
            "--min-seq-id",
            f"{min_seq_id}",
            "--tmscore-threshold",
            f"{tm_threshold}",
        ]
        if threads is not None:
            command += ["--threads", str(threads)]

        try:
            subprocess.run(command, check=True, text=True, capture_output=True, timeout=timeout)
        except Exception as e:
            logger.info(f"Foldseek diversity failed: {e} \n Returning None")
            return None

        res_file = os.path.join(path_tmp, "res_cluster.tsv")
        with open(res_file) as f:
            reps = []
            els = []
            for line in f:
                line = line.strip()
                if line:
                    words = line.split()
                    if len(words) == 2:
                        word1, word2 = words
                        reps.append(word1)
                        els.append(word2)
                    else:
                        logger.error(f"Strange line parsing Foldseek clustering results: {line}")

    cluster_ids = {rep: idx for idx, rep in enumerate(np.unique(reps))}
    el_to_cluster = {el: cluster_ids[rep] for rep, el in zip(reps, els, strict=True)}

    # Loud failure if Foldseek's emitted stems don't exactly match what we staged
    # (drops a member, adds an unexpected one, or staging convention drifted).
    expected = set(staged_stems)
    got = set(el_to_cluster)
    if expected != got:
        raise RuntimeError(f"Foldseek output mismatch: missing={sorted(expected - got)} extra={sorted(got - expected)}")

    # Reorder by input position so callers can zip this list with list_of_cif_paths.
    return [el_to_cluster[stem] for stem in staged_stems]


def cluster_structures(
    chains_dir: Path | str,
    tm_threshold: float = 0.5,
    alignment_type: int = 1,
    min_seq_id: float = 0.0,
    timeout: float | None = None,
    threads: int | None = None,
) -> dict[str, int] | None:
    """Cluster all chain-of-interest CIF files and return ``{example_id: cluster_id}``.

    Pass the ``foldseek_input_chains/`` subdirectory of the output tree (produced by
    :py:func:`~rfproteina.inference_utils.foldseek.chain_cif_dir`). The ``_chain`` suffix is
    stripped from each file stem to recover the example ID.

    Args:
        chains_dir: The ``foldseek_input_chains/`` subdirectory containing ``*_chain.cif`` files.
        tm_threshold: TM-score threshold for clustering.
        alignment_type: 1 (structure) or 2 (structure+sequence).
        min_seq_id: Required when ``alignment_type == 2``.
        timeout: Total wall-clock budget in seconds.
        threads: ``--threads`` for foldseek.

    Returns:
        ``{example_id: cluster_id}`` or ``None`` on foldseek failure.
    """
    chains_dir = Path(chains_dir)
    cif_files = iter_structure_files(chains_dir, stem_suffix="_chain")
    if not cif_files:
        logger.warning(f"cluster_structures: no *_chain.cif files found under {chains_dir}")
        return None
    paths = [str(p) for p in cif_files]
    example_ids = [structure_file_stem(p).removesuffix("_chain") for p in cif_files]
    cluster_ids = diversity_foldseek(
        paths,
        tm_threshold=tm_threshold,
        alignment_type=alignment_type,
        min_seq_id=min_seq_id,
        timeout=timeout,
        threads=threads,
    )
    if cluster_ids is None:
        return None
    return dict(zip(example_ids, cluster_ids, strict=True))


def _base_pass_cols(df: pd.DataFrame, pass_cols: list[str] | None = None) -> list[str]:
    """Return the solve-rate pass columns in ``df`` (explicit ``pass_cols``, else ``.pass`` fallback)."""
    if pass_cols is not None:
        return [c for c in pass_cols if c in df.columns]
    return [c for c in df.columns if c.endswith(".pass")]


def add_foldseek_diversity_to_per_example_df(
    per_example_df: pd.DataFrame,
    group_by: list[str] | None = None,
    timeout: float | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Add Foldseek clustering information to per-example DataFrame output by validation or external eval.

    Two modes:

    - **Output-tree mode** (df has ``output_base_dir`` column): auto-discovers
      ``foldseek_input_chains/`` group dirs from the df; cluster IDs are joined by
      the output tree, dataset, task and example ID.

    - **Per-group copy mode** (fallback): groups by ``group_by``, copies each group's chain
      files from ``designed_chain_path`` to a flat temp directory, and runs ``easy-cluster``
      per group in parallel threads.

    Args:
        per_example_df: DataFrame with an ``example_id`` column and (for per-group mode)
            a ``designed_chain_path`` column pointing to single-chain PDB files.
        group_by: Columns to group by when clustering (per-group mode only).
            Defaults to dataset and task when both are available.
        timeout: Wall-clock budget in seconds. In per-group mode the remaining budget is
            passed to each per-group subprocess; in output-tree mode it is the total budget.
        **kwargs: Additional arguments forwarded to :py:func:`diversity_foldseek` /
            :py:func:`cluster_structures` (``tm_threshold``, ``alignment_type``,
            ``min_seq_id``, ``threads``).

    Returns:
        DataFrame with additional column: ``foldseek_cluster_id``.
    """
    cs_kwargs = {k: kwargs[k] for k in ("tm_threshold", "alignment_type", "min_seq_id", "threads") if k in kwargs}

    if "output_base_dir" in per_example_df.columns:
        records = per_example_df.to_dict("records")
        groups = _partition_groups(records)
        combined_map = {}
        for g in groups:
            gmap = cluster_structures(g["chains_dir"], timeout=timeout, **cs_kwargs)
            if gmap:
                combined_map.update({(g["group"], eid): cid for eid, cid in gmap.items()})
            else:
                logger.warning(f"cluster_structures returned None for {g['chains_dir']}; skipping group.")
        out = per_example_df.copy()
        out["foldseek_cluster_id"] = [combined_map.get(_cluster_key(row)) for row in records]
        return out

    if group_by is None:
        group_by = [key for key in ("dataset_name", "task") if key in per_example_df] or ["task"]

    if "designed_chain_path" not in per_example_df.columns:
        raise ValueError("designed_chain_path column not found in per_example_df")
    for col in group_by:
        if col not in per_example_df.columns:
            raise ValueError(f"{col} column not found in per_example_df")

    groups = list(per_example_df.groupby(group_by, dropna=False))
    n_groups = len(groups)
    if not n_groups:
        return per_example_df.assign(foldseek_cluster_id=None)

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if "threads" in kwargs and kwargs["threads"] is not None:
        available_cpus = kwargs.pop("threads")
    elif slurm_cpus is not None:
        available_cpus = int(slurm_cpus)
    else:
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            available_cpus = os.cpu_count() or 1

    # Run all groups in parallel; divide CPUs evenly so each foldseek subprocess
    # gets threads_per_worker threads rather than competing for all CPUs.
    n_parallel = min(n_groups, available_cpus)
    threads_per_worker = max(1, available_cpus // n_parallel)
    logger.info(
        f"Foldseek: {n_groups} groups, {n_parallel} parallel workers, "
        f"{threads_per_worker} threads/worker ({available_cpus} CPUs total)"
    )

    start_time = time.monotonic()

    def _run_group(key: Any, group_df: pd.DataFrame) -> pd.DataFrame:
        group_vals = key if isinstance(key, tuple) else (key,)
        group_label = "/".join(str(v) for v in group_vals)

        # NaN designed_chain_path indicates an upstream failure; exclude those rows
        # from foldseek but don't let them block the rest of the group.
        nan_mask = group_df["designed_chain_path"].isna()
        if nan_mask.any():
            logger.warning(
                f"Group {group_label} has {nan_mask.sum()} row(s) with missing designed_chain_path "
                "(upstream failure); these will be treated as failed designs."
            )
        valid_df = group_df[~nan_mask]

        if valid_df.empty:
            logger.warning(
                f"designed_chain_path is missing for all rows in group {group_label}."
                " This group will not get cluster ids."
            )
            out = group_df.copy()
            out["foldseek_cluster_id"] = None
            return out

        # Compute remaining budget at submission time; all groups start near-simultaneously
        # so this is a fair approximation of the per-group wall-clock limit.
        remaining = (timeout - (time.monotonic() - start_time)) if timeout is not None else None

        unique_paths = valid_df["designed_chain_path"].unique().tolist()
        cluster_assignments = diversity_foldseek(
            list_of_cif_paths=unique_paths,
            timeout=remaining,
            threads=threads_per_worker,
            **kwargs,
        )
        out = group_df.copy()
        out["foldseek_cluster_id"] = None  # NaN-path rows stay None (failed designs)
        if cluster_assignments is None:
            logger.warning(f"Foldseek clustering failed for group {group_label}. This group will not get cluster IDs")
        else:
            path_to_cluster = dict(zip(unique_paths, cluster_assignments, strict=False))
            out.loc[~nan_mask, "foldseek_cluster_id"] = out.loc[~nan_mask, "designed_chain_path"].map(path_to_cluster)
        return out

    # ThreadPoolExecutor is safe inside DDP rank-0: threads do not fork the Python
    # process, so no CUDA context is duplicated. Each thread spawns an independent
    # Foldseek subprocess that has no knowledge of CUDA.
    processed: list[pd.DataFrame] = [None] * n_groups  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        future_to_idx = {executor.submit(_run_group, key, gdf): i for i, (key, gdf) in enumerate(groups)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                processed[idx] = future.result()
            except Exception as e:
                key, gdf = groups[idx]
                group_vals = key if isinstance(key, tuple) else (key,)
                group_label = "/".join(str(v) for v in group_vals)
                logger.warning(f"Foldseek group {group_label} raised an exception: {e}; skipping cluster ids")
                out = gdf.copy()
                out["foldseek_cluster_id"] = None
                processed[idx] = out

    return pd.concat(processed, ignore_index=True)


def add_diversity_summary_metrics(
    per_example_df: pd.DataFrame,
    group_by: list[str] | None = None,
    pass_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Broadcast per-target diversity metrics onto per-example rows.

    Adds assignment coverage, diversity, and distinct passing clusters per attempted
    design. An unassigned passer makes its clustered rate indeterminate.

    Args:
        per_example_df: DataFrame with a ``foldseek_cluster_id`` column (added by
            :py:func:`add_foldseek_diversity_to_per_example_df`).
        group_by: Columns that define the clustering groups. Must match the ``group_by``
            used in :py:func:`add_foldseek_diversity_to_per_example_df`. Defaults to
            the output tree, dataset and task when available.
        pass_cols: Explicit solve-rate pass columns (see :py:func:`_base_pass_cols`).
    """
    if group_by is None:
        group_by = [key for key in ("output_base_dir", "dataset_name", "task") if key in per_example_df] or ["task"]

    missing_cols = [c for c in group_by if c not in per_example_df.columns]
    if missing_cols:
        warnings.warn(
            f"Grouping columns {missing_cols} not found in DataFrame. "
            f"Treating all samples as having the same {missing_cols} for diversity clustering.",
            stacklevel=2,
        )
        per_example_df = per_example_df.copy()
        for c in missing_cols:
            per_example_df[c] = "all"

    if "foldseek_cluster_id" not in per_example_df.columns:
        raise ValueError(
            "foldseek_cluster_id column not found in per_example_df. "
            "Run add_foldseek_diversity_to_per_example_df first."
        )

    pass_cols = _base_pass_cols(per_example_df, pass_cols)
    per_example_df = per_example_df.copy()

    cluster_rows = []
    for group_key, group_df in per_example_df.groupby(group_by, dropna=False):
        group_vals = group_key if isinstance(group_key, tuple) else (group_key,)
        group_dict = dict(zip(group_by, group_vals, strict=False))

        n = len(group_df)
        clusters = group_df["foldseek_cluster_id"]
        assigned = clusters.count()
        row: dict = {
            **group_dict,
            "cluster_coverage": assigned / n,
            "cluster_diversity": clusters.nunique() / n if assigned == n else float("nan"),
        }
        for pass_col in pass_cols:
            prefix = pass_col[: -len("pass")]  # "{crit}." from "{crit}.pass"
            cpr_col = f"{prefix}clustered_pass_rate"
            attempts = pass_attempts(group_df, pass_col)
            denominator = attempts.count()
            passing = clusters.loc[attempts.eq(1)]
            if not assigned or not denominator or passing.isna().any():
                cpr_val: float = float("nan")
            else:
                cpr_val = passing.nunique() / denominator
            row[cpr_col] = cpr_val
        cluster_rows.append(row)

    if not cluster_rows:
        return per_example_df.assign(cluster_coverage=np.nan, cluster_diversity=np.nan, **{
            col.removesuffix("pass") + "clustered_pass_rate": np.nan for col in pass_cols
        })
    cluster_metrics_df = pd.DataFrame(cluster_rows)
    per_example_df = per_example_df.merge(cluster_metrics_df, on=group_by, how="left")

    return per_example_df


def cluster_summary_for_group(frame: pd.DataFrame) -> dict[str, float]:
    """Assignment coverage can span clustering jobs. Cluster IDs and their rates are local to one cohort."""
    if "foldseek_cluster_id" not in frame or frame.empty:
        return {}
    scope = [key for key in ("output_base_dir", "dataset_name", "task") if key in frame]
    one_cohort = not scope or len(frame[scope].drop_duplicates()) == 1
    summary = {"cluster_coverage": float(frame["foldseek_cluster_id"].notna().mean())}
    for column in frame:
        if column == "cluster_diversity" or column.endswith(".clustered_pass_rate"):
            values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            summary[column] = (
                float(values.iloc[0]) if one_cohort and values.notna().all() and values.nunique() == 1 else np.nan
            )
    return summary


def chain_cif_dir(
    out_dir: Path,
    dataset_name: str | None = None,
    task: str | None = None,
) -> Path:
    """Return the directory of chain-only CIFs that foldseek clusters together.

    Grouped by dataset and task because those define a clustering job; each design's own
    id is in the filename, so no further nesting is needed. ``dataset_name`` is hashed
    because it is typically an input path, which cannot be a single path segment.

    Args:
        out_dir: Top-level output base directory.
        dataset_name: Dataset name (groups foldseek clustering jobs by dataset).
        task: Task label (e.g. ``"unconditional"``).
    """
    subdir = out_dir / "foldseek_input_chains"
    if dataset_name is None and task is None:
        return subdir
    ds_seg = hash_sequence(dataset_name, 8) if dataset_name is not None else "unknown_dataset"
    return subdir / ds_seg / (task if task is not None else "unknown_task")


def _cluster_key(row: dict) -> tuple[tuple[str, str, str], str]:
    parts = []
    for key, fallback in (("output_base_dir", ""), ("dataset_name", "unknown_dataset"), ("task", "unknown_task")):
        value = row.get(key)
        parts.append(str(value) if pd.notna(value) else fallback)
    return (parts[0], parts[1], parts[2]), str(row.get("example_id", ""))


def _partition_groups(all_design_metrics: list[dict]) -> list[dict]:
    """Locate each output tree's dataset and task chain directory.

    Derives the group chains directory from ``output_base_dir``, ``dataset_name``,
    and ``task`` columns.  Missing ``dataset_name`` or ``task`` values are replaced
    with ``"unknown_dataset"`` / ``"unknown_task"`` so that partially-annotated rows
    still contribute a valid chains directory.  Only includes groups whose chains
    directory exists on disk (i.e. at least one chain PDB was written).  Returns
    ``[]`` when ``output_base_dir`` is missing or no metrics are provided.
    """
    if not all_design_metrics:
        return []
    df = pd.DataFrame(all_design_metrics)
    if "output_base_dir" not in df.columns:
        return []
    valid = df.dropna(subset=["output_base_dir"]).copy()
    if valid.empty:
        return []
    if "dataset_name" in valid.columns:
        valid["dataset_name"] = valid["dataset_name"].fillna("unknown_dataset")
    else:
        valid["dataset_name"] = "unknown_dataset"
    if "task" in valid.columns:
        valid["task"] = valid["task"].fillna("unknown_task")
    else:
        valid["task"] = "unknown_task"

    groups: list[dict] = []
    seen: set[Path] = set()
    for (output_base, ds, task), _ in valid.groupby(["output_base_dir", "dataset_name", "task"]):
        chains_dir = chain_cif_dir(Path(output_base), dataset_name=str(ds), task=str(task))
        if chains_dir in seen or not chains_dir.is_dir():
            continue
        seen.add(chains_dir)
        groups.append({
            "key": f"{chains_dir.parent.name}/{task}",
            "chains_dir": chains_dir,
            "dataset_name": str(ds),
            "group": (str(output_base), str(ds), str(task)),
        })
    return groups


def _cluster_groups_locally(groups: list[dict], rank_label: str = "0") -> dict[tuple[tuple[str, str, str], str], int]:
    """Run :py:func:`cluster_structures` on each group directory serially.

    Each group's ``chains_dir`` is passed directly to foldseek — no file copying. Results
    are keyed by output tree, dataset, task and example because IDs repeat across groups.
    """
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        available_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or os.cpu_count() or 4

    results = {}
    for g in groups:
        t0 = time.perf_counter()
        try:
            chain_map = cluster_structures(g["chains_dir"], threads=available_cpus)
            if chain_map:
                results.update({(g["group"], eid): cid for eid, cid in chain_map.items()})
            logger.info(
                f"[foldseek] rank{rank_label} group={g['key']} "
                f"n={len(chain_map or {})} threads={available_cpus} "
                f"wall={time.perf_counter() - t0:.2f}s"
            )
        except Exception as e:
            logger.warning(f"[foldseek] rank{rank_label} group {g['key']} failed: {e}")
    return results


def run_foldseek_distributed(
    all_design_metrics: list[dict] | None,
    *,
    world_size: int,
    global_rank: int,
    is_inference: bool,
) -> dict[tuple[tuple[str, str, str], str], int] | None:
    """Cluster ``(dataset_name, task)`` groups, optionally across DDP ranks.

    Each rank receives a slice of the ``foldseek_input_chains/{dataset}/{task}/`` directories and
    runs :py:func:`cluster_structures` on each one independently — no file copying.
    Rank 0 returns cluster assignments keyed by group and example ID. Other ranks
    return ``None``. Returns ``None`` (no-op) for inference or when ``FOLDSEEK_EXEC``
    is unset.
    """
    if os.getenv("FOLDSEEK_EXEC") is None or is_inference:
        return None  # all ranks must take the same exit; collective would deadlock

    if world_size == 1:
        return _cluster_groups_locally(_partition_groups(all_design_metrics or []))

    # Rank 0 builds per-rank work lists and scatters them.
    per_rank_groups: list[list[dict]] | None = None
    if global_rank == 0:
        per_rank_groups = [[] for _ in range(world_size)]
        for i, group in enumerate(_partition_groups(all_design_metrics or [])):
            per_rank_groups[i % world_size].append(group)

    output: list = [None]
    dist.scatter_object_list(output, per_rank_groups, src=0)
    my_groups: list[dict] = output[0] or []

    my_results = _cluster_groups_locally(my_groups, rank_label=str(global_rank))

    gathered: list | None = [None] * world_size if global_rank == 0 else None
    dist.gather_object(my_results, gathered, dst=0)
    if global_rank != 0:
        return None
    merged = {}
    for d in gathered or []:
        if d:
            merged.update(d)
    return merged


def apply_cluster_map_to_metrics(
    metrics: list[dict],
    cluster_map: dict[tuple[tuple[str, str, str], str], int] | None,
) -> None:
    """Write each matched group's cluster assignment to the corresponding design.

    Mutates ``metrics`` in place. No-op when ``cluster_map`` is ``None`` or empty.
    """
    if not cluster_map:
        return
    for m in metrics:
        cluster_id = cluster_map.get(_cluster_key(m))
        if cluster_id is not None:
            m["foldseek_cluster_id"] = cluster_id
