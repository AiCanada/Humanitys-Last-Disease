"""Read metric identifiers verbatim and publish recoverable batches of CSV reports."""

import io
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, TextIO

import pandas as pd
from filelock import FileLock

from rfproteina.utils.metrics_files import (
    PENDING,
    PRIMARY,
    RECEIPT,
    Shard,
    file_version,
    shard_record,
    validate_layout,
    write_receipt,
)

_ARCHIVES = ("individual_job_results", "previous_reports")
_IDENTIFIERS = ("dataset_name", "task", "example_id", "output_base_dir", "designed_structure_path",
                "designed_chain_path", "chain_of_interest", "metrics_tags")


@contextmanager
def open_metrics_csv(path: Path | str, *, sources: list[Shard] | None = None) -> Iterator[TextIO]:
    """Validate the logical records and optionally identify the exact file consumed."""
    path = Path(path)
    with path.open("rb") as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as stream:
        before = os.fstat(raw.fileno())
        try:
            validate_layout(stream)
            yield stream
        except ValueError as error:
            raise ValueError(f"Invalid metrics CSV {path.name}: {error}") from error
        if sources is not None:
            stream.seek(0)
            source = shard_record(path, raw)
        after = os.fstat(raw.fileno())
        if file_version(before) != file_version(after):
            raise ValueError(f"Metrics CSV changed while being read: {path.name}")
        if sources is not None:
            sources.append(source)


def read_metrics_csv(path: Path | str, *, sources: list[Shard] | None = None) -> pd.DataFrame:
    # A chain, target or example can legitimately be named "NA", "null", or "0001".
    with open_metrics_csv(path, sources=sources) as stream:
        return pd.read_csv(stream, converters=dict.fromkeys(_IDENTIFIERS, _identifier), float_precision="round_trip")


def _identifier(cell: str) -> str | None:
    return cell or None


def metrics_write_pending(directory: Path) -> bool:
    return (directory / PENDING).exists()


def archive_job_file(path: Path, archive: Path) -> None:
    """Move a shard under the metrics-directory lock, retaining every revision."""
    archive.mkdir(exist_ok=True)
    destination = archive / path.name
    revision = 0
    while destination.exists():
        revision += 1
        destination = archive / f"{path.stem}_revision{revision}{path.suffix}"
    path.rename(destination)


def _recover(directory: Path) -> None:
    pending = directory / PENDING
    if not pending.exists():
        return
    owner = pending / "format-v1"
    if not owner.is_file() or pending.is_symlink():
        if pending.is_dir() and not pending.is_symlink() and not any(pending.iterdir()):
            pending.rmdir()
            return
        raise RuntimeError(f"Unrecognized metrics staging directory: {pending.name}")
    publishing = pending / "publishing"
    committed = pending / "committed"
    if publishing.exists():
        # A missing staged file was already installed. Moving each backup back
        # also marks it restored if recovery itself is interrupted.
        for backup in (pending / "previous").iterdir():
            if not (pending / "new" / backup.name).exists():
                os.replace(backup, directory / backup.name)
        for marker in (pending / "absent").iterdir():
            if not (pending / "new" / marker.name).exists():
                (directory / marker.name).unlink(missing_ok=True)
        publishing.unlink()
    elif committed.exists():
        for name in _ARCHIVES:
            if (pending / name).is_file():
                for backup in (pending / "previous").iterdir():
                    archive_job_file(backup, directory / name)
        committed.unlink()
    for name in ("new", "previous", "absent"):
        folder = pending / name
        if folder.exists():
            shutil.rmtree(folder)
    for name in _ARCHIVES:
        (pending / name).unlink(missing_ok=True)
    owner.unlink()
    pending.rmdir()


@contextmanager
def locked_metrics_directory(directory: Path) -> Iterator[None]:
    """Serialize cooperating writers and recover a writer interrupted during replacement."""
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".rfproteina-metrics.lock")):
        _recover(directory)
        yield


@contextmanager
def staged_metrics_tables(
    directory: Path, names: list[str],
    *, archive_replaced: Literal["individual_job_results", "previous_reports"] | None = None,
) -> Iterator[Path]:
    """Stage a batch under the directory lock, then publish in the supplied order."""
    if archive_replaced is not None and archive_replaced not in _ARCHIVES:
        raise ValueError("Unrecognized metrics archive.")
    if len(set(names)) != len(names) or any(Path(name).name != name or not name.endswith(".csv") for name in names):
        raise ValueError("Metric table names must be unique CSV basenames.")
    pending = directory / PENDING
    pending.mkdir()
    try:
        (pending / "format-v1").touch()
        staged, previous, absent = (pending / name for name in ("new", "previous", "absent"))
        for folder in (staged, previous, absent):
            folder.mkdir()
        yield staged
        if any(not (staged / name).is_file() for name in names):
            raise ValueError("A staged metrics table is missing.")
        for name in names:
            current = directory / name
            if current.exists():
                shutil.copymode(current, staged / name)
                try:
                    os.link(current, previous / name)
                except OSError:
                    shutil.copy2(current, previous / name)
            else:
                (absent / name).touch()
        if archive_replaced:
            (pending / archive_replaced).touch()
        publishing = pending / "publishing"
        publishing.touch()
        for name in names:
            os.replace(staged / name, directory / name)
        os.replace(publishing, pending / "committed")
    finally:
        _recover(directory)


def write_metrics_tables(
    directory: Path, tables: dict[str, pd.DataFrame],
    *, archive_replaced: Literal["individual_job_results", "previous_reports"] | None = None,
    sources: list[Shard] | None = None,
) -> None:
    """Publish metric frames and bind this batch's companion tables to its primary."""
    names = [name for name in tables if name != PRIMARY]
    publish_receipt = PRIMARY in tables or sources is not None
    if publish_receipt:
        names.append(RECEIPT)
    if PRIMARY in tables:
        names.append(PRIMARY)
    with staged_metrics_tables(directory, names, archive_replaced=archive_replaced) as staged:
        for name, frame in tables.items():
            frame.to_csv(staged / name, index=False, lineterminator="\r\n")
        if publish_receipt:
            primary = staged / PRIMARY if PRIMARY in tables else directory / PRIMARY
            companions = (staged / name for name in tables if name != PRIMARY)
            write_receipt(staged / RECEIPT, primary, sources or [], companions)
