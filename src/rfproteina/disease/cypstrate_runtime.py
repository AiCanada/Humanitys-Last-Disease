"""Run the authors' immutable linux/amd64 CYPstrate image on a local Docker engine."""

from __future__ import annotations

import json
import subprocess
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from rfproteina.disease.local_docker import local_engines as _local_engines

VERSION = "0.0.25"
REVISION = "c153960de7fba9b9b07047b7f20e55277a88fb41"
DIGEST = "sha256:587227598961627192f5c007757df46369d2a2960e004b726a95eb1bf3774d5b"
IMAGE = f"ghcr.io/molinfo-vienna/cypstrate@{DIGEST}"
_CONFIG = "sha256:dfb4ddd5ba2beb5b8383f886e0f2779a30319f3780de03fec0e4c2f646561f00"
_BOOT = "import sys; s=sys.stdin.buffer; exec(compile(s.read(int(s.readline())), 'cypstrate_worker.py', 'exec'))"


class CypstrateError(ValueError):
    """A named input, runtime or attribution problem."""


def _engine() -> list[str] | None:
    for engine in _local_engines():
        try:
            result = subprocess.run(
                [*engine, "image", "inspect", "--format", "{{.Id}}", IMAGE],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        # The classic image store reports the config ID, the containerd store the manifest ID.
        if result.returncode == 0 and result.stdout.strip() in (DIGEST.encode(), _CONFIG.encode()):
            return engine
    return None


def installed() -> bool:
    from importlib.util import find_spec

    return find_spec("rdkit") is not None and _engine() is not None


def predict(inputs: list[dict], mode: str) -> dict:
    engine = _engine()
    if not engine:
        raise CypstrateError("cypstrate_image_unavailable")
    try:
        source = Path(__file__).with_name("cypstrate_worker.py").read_bytes()
    except OSError as error:
        raise CypstrateError("cypstrate_runtime_failed") from error
    payload = json.dumps({"inputs": inputs, "mode": mode}, ensure_ascii=True, allow_nan=False).encode("ascii")
    name = f"rfd4-cypstrate-{uuid4().hex}"
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
        "--shm-size=128m",
        "--memory=3g",
        "--cpus=4",
        "--pids-limit=128",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--log-driver=none",
        "-e",
        "OPENBLAS_NUM_THREADS=1",
        "-e",
        "OMP_NUM_THREADS=1",
        "-i",
        "--entrypoint=/env/bin/python",
        IMAGE,
        "-I",
        "-c",
        _BOOT,
    ]
    # Only the fixed project worker is code. Its following JSON input stays on stdin.
    message = str(len(source)).encode("ascii") + b"\n" + source + payload
    try:
        result = subprocess.run(
            command,
            input=message,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=120 + 45 * ((len(inputs) + 15) // 16),
            check=False,
        )
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        with suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run([*engine, "rm", "--force", name], capture_output=True, timeout=10, check=False)
        if isinstance(error, KeyboardInterrupt):
            raise
        raise CypstrateError("cypstrate_execution_timeout") from error
    except OSError as error:
        raise CypstrateError("cypstrate_runtime_failed") from error
    if result.returncode or len(result.stdout) > 2 * 1024 * 1024:
        raise CypstrateError("cypstrate_runtime_failed")
    try:
        record = json.loads(result.stdout)
    except (ValueError, RecursionError) as error:
        raise CypstrateError("cypstrate_invalid_output") from error
    if not isinstance(record, dict) or record.get("version") != VERSION or record.get("mode") != mode:
        raise CypstrateError("cypstrate_invalid_output")
    return record
