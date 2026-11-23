"""CSV layout and provenance shared by metric writers and lightweight screen readers."""

import csv
import fnmatch
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, TextIO

PRIMARY = "per_example_metrics.csv"
ARCHIVE = "individual_job_results"
PENDING = ".rfproteina-metrics-pending"
RECEIPT = "aggregation_sources.csv"
Shard = tuple[str, str, str]


class MetricsBlocker(StrEnum):
    """Public reasons a post-generation report cannot attribute a measurement."""

    AGGREGATION_REQUIRED = "metrics_aggregation_required"
    PUBLICATION_PENDING = "metrics_publication_pending"
    PROVENANCE_MISMATCH = "metrics_provenance_mismatch"
    PROVENANCE_UNREADABLE = "metrics_provenance_unreadable"
    ROW_SHARED = "metrics_row_shared"
    TABLE_ONLY = "evaluation_structure_table_only"


@dataclass
class MetricsReceipt:
    version: int
    reports: dict[str, str]
    sources: list[Shard]


def _check_layout(stream: TextIO) -> None:
    reader = csv.reader(stream, strict=True)
    header = next((row for row in reader if row), None)
    if header is None:
        raise ValueError("missing CSV header")
    names = [name.strip() for name in header]
    if not all(names) or len(names) != len(set(names)) or any("\x00" in name for name in names):
        raise ValueError("empty, repeated or invalid column name")
    for row in reader:
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError(f"line {reader.line_num} has {len(row)} fields for {len(header)} columns")
        if any("\x00" in cell for cell in row):
            raise ValueError(f"line {reader.line_num} contains a null character")


def validate_layout(stream: TextIO) -> None:
    """Validate full logical records on the same file descriptor pandas will read."""
    try:
        _check_layout(stream)
    except csv.Error as error:
        if "field larger than field limit" not in str(error):
            raise ValueError(f"invalid CSV quoting: {error}") from error
        # The stdlib field limit is process-wide. Validate exceptionally wide
        # legitimate columns in a private interpreter without changing other readers.
        stream.seek(0)
        result = subprocess.run([sys.executable, "-I", __file__], stdin=stream, capture_output=True, text=True)
        if result.returncode:
            raise ValueError(result.stderr.strip() or "wide CSV validation failed") from error
    stream.seek(0)


def wide_csv_rows(payload: bytes) -> list[list[str]]:
    """Decode a size-bounded screen table without changing other readers' field limits."""
    result = subprocess.run([sys.executable, "-I", __file__, "--rows"], input=payload, capture_output=True)
    if result.returncode:
        raise csv.Error(result.stderr.decode("utf-8", errors="replace").strip() or "wide CSV decoding failed")
    return json.loads(result.stdout)


def csv_records(stream: TextIO) -> Iterator[list[str]]:
    """Stream nonblank records, including fields wider than the local interpreter limit."""
    consumed = 0
    try:
        for row in csv.reader(stream, strict=True):
            if row:
                consumed += 1
                yield row
        return
    except csv.Error as error:
        if "field larger than field limit" not in str(error):
            raise ValueError(f"Invalid CSV quoting: {error}") from error
    # An isolated interpreter can lift its own field limit. Resume at the same
    # logical record without buffering the entire file or changing other readers.
    stream.seek(0)
    with subprocess.Popen(
        [sys.executable, "-I", __file__, "--stream-rows"], stdin=stream, stdout=subprocess.PIPE,
    ) as process:
        try:
            for line in process.stdout:
                if consumed:
                    consumed -= 1
                else:
                    yield json.loads(line)
            if process.wait():
                raise ValueError("Wide CSV decoding failed")
        finally:
            if process.poll() is None:
                process.terminate()


def job_tables(directory: Path) -> list[Path]:
    found = []
    for root in (directory, directory / ARCHIVE):
        try:
            with os.scandir(root) as entries:
                found.extend(root / entry.name for entry in entries if entry.name != PRIMARY
                             and fnmatch.fnmatch(entry.name, "*per_example_metrics*.csv"))
        except FileNotFoundError:
            continue
    return found


def file_version(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def metrics_snapshot(
    directory: Path, names: Iterable[str], *, include_jobs: bool = True,
) -> dict[Path, tuple[int, ...]]:
    """Capture published files, the receipt and the set of jobs for an optimistic read."""
    if (directory / PENDING).exists():
        raise RuntimeError(f"A metrics publication is pending in {directory}. Retry after the writer finishes.")
    paths = {directory / name for name in (*names, RECEIPT)}
    if include_jobs:
        paths.update(job_tables(directory))
    versions = {}
    for path in paths:
        try:
            versions[path] = file_version(path.stat())
        except FileNotFoundError:
            if path.is_symlink():
                raise
    return versions


def file_sha256(path: Path) -> str:
    with path.open("rb") as raw:
        return hashlib.file_digest(raw, "sha256").hexdigest()


def shard_record(path: Path, raw: BinaryIO) -> Shard:
    """Identify the bytes read, retaining the ordering timestamp through archive renames."""
    name = re.sub(r"(?:_revision[0-9]+)+$", "", path.stem) + path.suffix
    return name, str(os.fstat(raw.fileno()).st_mtime_ns), hashlib.file_digest(raw, "sha256").hexdigest()


def write_receipt(path: Path, primary: Path, sources: list[Shard], reports: Iterable[Path]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["kind", "source", "mtime_ns", "sha256"])
        for report in (primary, *sorted(reports)):
            writer.writerow(["report", report.name, "", file_sha256(report)])
        writer.writerows(("shard", *source) for source in sorted(sources))


def _read_receipt(path: Path) -> MetricsReceipt:
    reports = {}
    sources = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, None)
        if header == ["source", "mtime_ns", "sha256"]:
            version = 1
        elif header == ["kind", "source", "mtime_ns", "sha256"]:
            version = 2
        else:
            raise ValueError("Unrecognized metrics receipt header")
        for index, row in enumerate(reader):
            if version == 1:
                row = ["report" if index == 0 else "shard", *row]
            if len(row) != 4:
                raise ValueError("Invalid metrics receipt row")
            kind, name, stamp, digest = row
            if (not name or "/" in name or "\\" in name or "\x00" in name
                    or not re.fullmatch("[a-f0-9]{64}", digest)):
                raise ValueError("Invalid metrics receipt filename or digest")
            if kind == "report" and not stamp and name not in reports:
                reports[name] = digest
            elif kind == "shard" and re.fullmatch(r"-?[0-9]+", stamp):
                sources.append((name, stamp, digest))
            else:
                raise ValueError("Invalid or duplicate metrics receipt entry")
    if PRIMARY not in reports:
        raise ValueError("Metrics receipt has no primary table")
    return MetricsReceipt(version, reports, sources)


def verified_receipt(
    directory: Path, primary_digest: str, *, check_jobs: bool = True,
) -> tuple[MetricsReceipt | None, str | None]:
    """Verify a published snapshot against its surviving local shards, including multiplicity."""
    try:
        before = metrics_snapshot(directory, [PRIMARY], include_jobs=check_jobs)
        if directory / PRIMARY not in before:
            return None, MetricsBlocker.PROVENANCE_UNREADABLE.value
        path = directory / RECEIPT
        shards = before.keys() - {directory / PRIMARY, path}
        reevaluation = directory / "per_example_metrics_reeval.csv"
        receipt = _read_receipt(path) if path in before else None
        if receipt is None:
            # Older in-place aggregations archived their inputs. The historical
            # standalone re-evaluation can coexist with a preferred primary CSV.
            if any(p.parent == directory and p != reevaluation for p in shards):
                return None, MetricsBlocker.AGGREGATION_REQUIRED.value
        else:
            if receipt.reports[PRIMARY] != primary_digest:
                return None, MetricsBlocker.PROVENANCE_MISMATCH.value
            recorded = Counter(receipt.sources)
            for shard in shards:
                if receipt.version == 2 and not receipt.sources and shard == reevaluation:
                    continue
                with shard.open("rb") as raw:
                    key = shard_record(shard, raw)
                if not recorded[key]:
                    return None, MetricsBlocker.AGGREGATION_REQUIRED.value
                recorded[key] -= 1
        if before != metrics_snapshot(directory, [PRIMARY], include_jobs=check_jobs):
            return None, MetricsBlocker.AGGREGATION_REQUIRED.value
    except RuntimeError:
        return None, MetricsBlocker.PUBLICATION_PENDING.value
    except (OSError, ValueError, csv.Error):
        return None, MetricsBlocker.PROVENANCE_UNREADABLE.value
    return receipt, None


def verified_sources(
    directory: Path, primary_digest: str, *, check_jobs: bool = True,
) -> tuple[list[Shard], str | None]:
    receipt, blocker = verified_receipt(directory, primary_digest, check_jobs=check_jobs)
    return (receipt.sources if receipt else []), blocker


if __name__ == "__main__":
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:  # The C long used by some platforms is 32 bits.
        csv.field_size_limit(2**31 - 1)
    try:
        stream = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8-sig", newline="")
        if sys.argv[1:] == ["--rows"]:
            json.dump(list(csv.reader(stream, strict=True)), sys.stdout)
        elif sys.argv[1:] == ["--stream-rows"]:
            for cells in csv.reader(stream, strict=True):
                if cells:
                    print(json.dumps(cells))
        else:
            _check_layout(stream)
    except (csv.Error, UnicodeError, ValueError) as exc:
        sys.exit(str(exc))
