"""Collect published post-generation tables by configuration and report their coverage."""

import argparse
import csv
import json
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from rfproteina.inference_utils.metrics_tables import locked_metrics_directory, open_metrics_csv, staged_metrics_tables
from rfproteina.utils.metrics_files import (
    PENDING,
    PRIMARY,
    RECEIPT,
    csv_records,
    file_sha256,
    job_tables,
    metrics_snapshot,
    verified_receipt,
)

_REPORTS = ("aggregated_metrics.csv", "per_task_metrics.csv", PRIMARY)
_STAGES = ("inference_metrics", "evaluation_metrics")
_COVERAGE = "sweep_coverage.csv"


@dataclass
class _Table:
    path: Path
    status: str
    columns: tuple[str, ...] = ()
    digest: str = ""
    rows: int | None = None

    @property
    def config(self) -> str:
        return self.path.parent.parent.name


def _directories(root: Path, expected: Sequence[str] | None, stage: str | None) -> list[Path]:
    stages = (stage,) if stage else _STAGES
    if expected is None:
        configs = sorted(child for child in root.iterdir() if child.is_dir() and any(
            (child / name).is_dir() or (child / name.replace("_metrics", "_samples")).is_dir() for name in stages
        ))
    else:
        if isinstance(expected, str) or len(set(expected)) != len(expected) or any(
            not name or name in {".", ".."} or "/" in name or "\\" in name or Path(name).drive for name in expected
        ):
            raise ValueError("Expected sweep configurations must be unique directory names.")
        configs = [root / name for name in sorted(expected)]
    directories = []
    for config in configs:
        existing = [config / name for name in stages if (config / name).is_dir()]
        if len(existing) > 1:
            publications = [path for path in existing if any(
                (path / name).exists() for name in (PRIMARY, RECEIPT, PENDING)
            ) or job_tables(path)]
            if publications:
                directories.append(publications[0])
                continue
        samples = [config / name for name in stages if (config / name.replace("_metrics", "_samples")).is_dir()]
        directories.append(next(iter(existing or samples), config / stages[0]))
    return directories


def _config_tables(directory: Path, snapshot: dict[Path, tuple[int, ...]]) -> list[_Table]:
    primary = directory / PRIMARY
    receipt = None
    primary_digest = ""
    if primary in snapshot:
        primary_digest = file_sha256(primary)
        receipt, blocker = verified_receipt(directory, primary_digest)
        if blocker:
            raise ValueError(f"{directory}: {blocker}. Check the per-configuration publication.")
    elif directory / RECEIPT in snapshot or job_tables(directory):
        raise ValueError(f"{directory}: published primary table missing. Aggregate the configuration first.")

    tables = []
    for name in _REPORTS:
        path = directory / name
        if path not in snapshot:
            tables.append(_Table(path, "missing"))
        elif not primary_digest:
            tables.append(_Table(path, "primary_missing"))
        elif receipt and receipt.version >= 2 and name not in receipt.reports:
            tables.append(_Table(path, "not_published"))
        else:
            expected = receipt.reports.get(name, "") if receipt else ""
            try:
                with path.open(encoding="utf-8-sig", newline="") as stream, closing(csv_records(stream)) as rows:
                    columns = tuple(next(rows, ()))
                if not columns:
                    raise ValueError("Missing CSV header")
            except (ValueError, OSError) as error:
                raise ValueError(f"Cannot read the header of {path}: {error}") from error
            if any(column.strip() == "sweep_config" for column in columns):
                raise ValueError(f"Reserved sweep_config column in {path}. Supply a per-configuration table.")
            tables.append(_Table(path, "verified" if expected else "legacy", columns,
                                 expected or (primary_digest if name == PRIMARY else "")))
    return tables


def _append_table(table: _Table, columns: list[str], output: TextIO) -> None:
    # Copy source text one record at a time, in the union of the source schemas.
    positions = {name: index for index, name in enumerate(table.columns)}
    order = [positions.get(name) for name in columns[1:]]
    writer = csv.writer(output)
    signature = []
    count = 0
    with open_metrics_csv(table.path, sources=signature) as stream, closing(csv_records(stream)) as rows:
        if tuple(next(rows, ())) != table.columns:
            raise RuntimeError(f"Header changed during sweep export: {table.path}")
        for row in rows:
            writer.writerow([table.config, *(row[i] if i is not None else "" for i in order)])
            count += 1
    digest = signature[0][2]
    if table.digest and table.digest != digest:
        raise ValueError(f"Published digest mismatch for {table.path}. Check the original report.")
    table.rows, table.digest = count, digest


def concatenate_sweep_results(
    sweep_dir: str, *, expected_configs: Sequence[str] | None = None, metrics_dir: str | None = None,
) -> None:
    """Publish a consistent sweep. An explicit config list requires complete results for every member."""
    root = Path(sweep_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {root}")
    if metrics_dir is not None and metrics_dir not in _STAGES:
        raise ValueError(f"Unknown metrics directory: {metrics_dir}")
    with locked_metrics_directory(root):
        directories = _directories(root, expected_configs, metrics_dir)
        snapshots = {directory: metrics_snapshot(directory, _REPORTS) for directory in directories}
        tables = [table for directory in directories for table in _config_tables(directory, snapshots[directory])]
        missing = [f"{table.config}/{table.path.name} ({table.status})" for table in tables if not table.columns]
        if expected_configs is not None and missing:
            raise ValueError("Incomplete planned sweep: " + ", ".join(missing))
        if not any(table.columns and table.path.name == PRIMARY for table in tables):
            raise FileNotFoundError(f"No published per-example CSV was found under {root}")

        names = [*("sweep_" + name for name in _REPORTS[:-1]), _COVERAGE, "sweep_" + PRIMARY]
        with staged_metrics_tables(root, names) as staged:
            for name in _REPORTS:
                available = [table for table in tables if table.path.name == name and table.columns]
                columns = ["sweep_config", *dict.fromkeys(col for table in available for col in table.columns)]
                with (staged / ("sweep_" + name)).open("w", encoding="utf-8", newline="") as handle:
                    csv.writer(handle).writerow(columns)
                    for table in available:
                        _append_table(table, columns, handle)
            if expected_configs is not None:
                empty = [f"{t.config}/{t.path.name}" for t in tables if t.rows == 0]
                if empty:
                    raise ValueError("Empty tables in planned configurations: " + ", ".join(empty))
            with (staged / _COVERAGE).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["sweep_config", "metrics_directory", "table", "status", "rows", "sha256",
                                 "source_columns"])
                writer.writerows(
                    (t.config, t.path.parent.name, t.path.name, t.status, t.rows, t.digest,
                     json.dumps(t.columns, ensure_ascii=False) if t.columns else "") for t in tables
                )
            if directories != _directories(root, expected_configs, metrics_dir) or any(
                snapshot != metrics_snapshot(directory, _REPORTS) for directory, snapshot in snapshots.items()
            ):
                raise RuntimeError("Source metrics changed during sweep export. Retry with the completed publications.")
    if missing:
        print("Unavailable tables: " + ", ".join(missing))
    print(f"Published sweep reports for {len(directories)} configuration(s). Coverage is in {_COVERAGE}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sweep-dir",
        type=str,
        required=True,
        help="root containing per-configuration run directories",
    )
    parser.add_argument("--expected-config", action="append",
                        help="repeat for each configuration required in a complete sweep")
    parser.add_argument("--metrics-dir", choices=_STAGES,
                        help="select a stage explicitly. Inference takes precedence when both contain results or jobs.")
    args = parser.parse_args()
    try:
        concatenate_sweep_results(args.sweep_dir, expected_configs=args.expected_config, metrics_dir=args.metrics_dir)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Sweep export failed: {error}\n")


if __name__ == "__main__":
    main()
