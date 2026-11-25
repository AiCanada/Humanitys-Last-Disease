"""Run native CellProfiler H-DAB image analysis on a local, digest-pinned CPU image."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from rfproteina.disease.local_docker import local_engines

if TYPE_CHECKING:
    import numpy as np

VERSION = "4.2.8"
DIGEST = "sha256:fec440caa2b44edf80f9bd440a3d8c6b214d72437332ad2586a32d79e2eae2a4"
IMAGE = f"cellprofiler/cellprofiler@{DIGEST}"
MAX_REGIONS = 4
MAX_PIXELS = 4_194_304
_BOOT = "import sys; s=sys.stdin.buffer; exec(compile(s.read(int(s.readline())), 'cellprofiler_worker.py', 'exec'))"


class CellProfilerError(ValueError):
    """A named runtime, measurement or attribution failure."""


def _engine() -> list[str] | None:
    for engine in local_engines():
        try:
            status = subprocess.run(
                [*engine, "image", "inspect", "--format", "{{.Os}}/{{.Architecture}} {{json .RepoDigests}}", IMAGE],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            platform, _, raw = status.stdout.partition(b" ")
            if status.returncode == 0 and platform == b"linux/amd64":
                digests = json.loads(raw)
                if isinstance(digests, list) and (IMAGE in digests or f"docker.io/{IMAGE}" in digests):
                    return engine
        except (OSError, ValueError, subprocess.TimeoutExpired):
            continue
    return None


def installed() -> bool:
    return _engine() is not None


def _pack(regions: Iterable[tuple[np.ndarray, np.ndarray]]) -> tuple[bytearray, list[tuple[str, int]]]:
    import numpy as np

    message = bytearray(4)  # The count precedes the fixed-size field headers and pixel buffers.
    expected = []
    for rgb, mask in regions:
        if len(expected) >= MAX_REGIONS or not isinstance(rgb, np.ndarray) or not isinstance(mask, np.ndarray):
            raise CellProfilerError("cellprofiler_measurement_failed")
        if (
            rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[-1] != 3
            or mask.dtype != np.bool_
            or mask.shape != rgb.shape[:2]
            or not 1 <= mask.size <= MAX_PIXELS
        ):
            raise CellProfilerError("cellprofiler_measurement_failed")
        count = int(np.count_nonzero(mask))
        if not count:
            raise CellProfilerError("cellprofiler_measurement_failed")
        header = struct.pack(">II", *mask.shape)
        start = len(message)
        message.extend(header)
        message.extend(rgb.tobytes(order="C"))
        # Conversion normalizes the byte representation even of boolean array views.
        message.extend(mask.astype(np.uint8).tobytes(order="C"))
        digest = hashlib.sha256(memoryview(message)[start:]).hexdigest()
        expected.append((digest, count))
    if not expected:
        raise CellProfilerError("cellprofiler_measurement_failed")
    struct.pack_into(">I", message, 0, len(expected))
    return message, expected


def _decode(raw: bytes, expected: list[tuple[str, int]]) -> list[dict]:
    if len(raw) > 8192:
        raise CellProfilerError("cellprofiler_measurement_failed")
    try:
        record = json.loads(raw)
    except (ValueError, RecursionError) as error:
        raise CellProfilerError("cellprofiler_measurement_failed") from error
    if not isinstance(record, dict) or record.get("version") != VERSION:
        raise CellProfilerError("cellprofiler_measurement_failed")
    rows = record.get("regions")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise CellProfilerError("cellprofiler_measurement_failed")
    results = []
    for row, (digest, count) in zip(rows, expected, strict=True):
        if (
            not isinstance(row, dict)
            or row.get("region_sha256") != digest
            or type(row.get("roi_pixels")) is not int
            or row["roi_pixels"] != count
        ):
            raise CellProfilerError("cellprofiler_measurement_failed")
        for key in ("mean_dab", "median_dab"):
            value = row.get(key)
            if type(value) not in (float, int) or not 0 <= value <= 1 or not math.isfinite(value):
                raise CellProfilerError("cellprofiler_measurement_failed")
        results.append(
            {"roi_pixels": count, "mean_dab": float(row["mean_dab"]), "median_dab": float(row["median_dab"])}
        )
    return results


def measure_regions(regions: Iterable[tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    """Measure one to four 8-bit RGB fields and their aligned, explicit boolean masks."""
    payload, expected = _pack(regions)
    engine = _engine()
    if engine is None:
        raise CellProfilerError("cellprofiler_runtime_unavailable")
    name = f"rfd4-cellprofiler-{uuid4().hex}"
    command = [
        *engine,
        "run",
        "--name",
        name,
        "--rm",
        "--pull=never",
        "--platform=linux/amd64",
        "--network=none",
        "--read-only",
        "--tmpfs=/tmp:rw,nosuid,nodev,size=128m",
        "--memory=3g",
        "--cpus=2",
        "--pids-limit=128",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--log-driver=none",
        "-e",
        "HOME=/tmp",
        "-e",
        "OPENBLAS_NUM_THREADS=1",
        "-e",
        "OMP_NUM_THREADS=1",
        "-i",
        "--entrypoint=python",
        IMAGE,
        "-I",
        "-c",
        _BOOT,
    ]
    try:
        source = Path(__file__).with_name("cellprofiler_worker.py").read_bytes()
        payload[:0] = str(len(source)).encode("ascii") + b"\n" + source
        result = subprocess.run(
            command, input=payload, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=180, check=False
        )
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        with suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run([*engine, "rm", "--force", name], capture_output=True, timeout=10, check=False)
        if isinstance(error, KeyboardInterrupt):
            raise
        raise CellProfilerError("cellprofiler_execution_timeout") from error
    except OSError as error:
        raise CellProfilerError("cellprofiler_measurement_failed") from error
    if result.returncode:
        raise CellProfilerError("cellprofiler_measurement_failed")
    return _decode(result.stdout, expected)


def measure_region(rgb: np.ndarray, mask: np.ndarray) -> dict:
    """Measure DAB intensity inside one submitted region."""
    return measure_regions([(rgb, mask)])[0]
