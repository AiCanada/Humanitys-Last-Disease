"""Run planned generation commands on this machine and on remote machines.

Nothing here caps the number of *machines*. How many requests run at once is
capped on each side separately and only when asked: ``local_workers`` bounds the
processes started here, ``remote_workers`` bounds the requests in flight across
all remote machines at once. Unset, either side starts every request of its kind
at once. ``local`` and ``localhost`` run on this machine. Every other host is
reached with ``ssh``.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from rfproteina.disease.shards import WorkerShard, generation_commands

Runner = Callable[[list[str]], int]
Transfer = Callable[[str, Path, str], None]
Retrieve = Callable[[str, str, Path], None]
LOCAL_HOSTS = frozenset({"local", "localhost"})
#: Generation children started by this process and not yet finished, so that a
#: caller asked to stop has something to act on. Held here rather than returned
#: because ``Runner`` is one injected call returning an exit code, and every
#: caller of it would otherwise have to thread a handle back out.
_running: set[subprocess.Popen[bytes]] = set()
_running_lock = threading.Lock()


@dataclass(frozen=True)
class GenerationRequest:
    """One ``rfd4-generate`` invocation and the files it needs.

    ``companions`` are files that must sit beside a remote condition spec.
    The spec's ``input:`` path is relative to the spec file.
    """

    host: str
    argv: list[str]
    local_input: Path
    remote_input: str
    remote_out_dir: str
    companions: tuple[tuple[Path, str], ...] = ()


@dataclass(frozen=True)
class RunResult:
    """The command that was launched and the code it returned."""

    host: str
    argv: list[str]
    returncode: int


def is_local_host(host: str) -> bool:
    """Whether ``host`` means this machine."""
    return host.strip().casefold() in LOCAL_HOSTS


def generation_requests(
    shards: list[WorkerShard],
    out_dir: str,
    modified_ckpt: Path | None = None,
    remote_dir: str = "disease_runs",
    binder_kind: str | None = None,
) -> list[GenerationRequest]:
    """Pair each planned command with its local input and remote paths.

    ``binder_kind`` is forwarded into the command only when ``modified_ckpt``
    is an existing file, matching ``generation_commands``. A remote directory
    is required only when a request leaves this machine.
    """
    commands = generation_commands(shards, out_dir, modified_ckpt=modified_ckpt, binder_kind=binder_kind)
    assigned = [(shard.host, structure) for shard in shards for structure in shard.structures]
    if len(commands) != len(assigned):
        raise RuntimeError("Each assigned structure needs one generation command.")
    if any(not is_local_host(host) for host, _structure in assigned) and not remote_dir.strip():
        raise ValueError("A remote directory is required for generation on other machines.")
    requests: list[GenerationRequest] = []
    root = remote_dir.rstrip("/")
    for command, (host, structure) in zip(commands, assigned, strict=True):
        filename = _command_input_name(command)
        local_input = Path(out_dir) / filename
        companions = _companions(local_input, root)
        requests.append(
            GenerationRequest(
                host=host,
                argv=command,
                local_input=local_input,
                remote_input=f"{root}/{filename}",
                remote_out_dir=f"{root}/{structure.group}_{structure.entry_id}_{structure.chain}",
                companions=companions,
            )
        )
    return requests


def execution_argv(request: GenerationRequest) -> list[str]:
    """Argv for this machine, or an ``ssh`` argv that carries the remote paths."""
    if is_local_host(request.host):
        return list(request.argv)
    remote = [_rewrite(part, request) for part in request.argv]
    return ["ssh", request.host, shlex.join(remote)]


def execute_generation(
    requests: list[GenerationRequest],
    *,
    runner: Runner,
    transfer: Transfer,
    retrieve: Retrieve | None = None,
    local_workers: int | None = None,
    remote_workers: int | None = None,
) -> list[RunResult]:
    """Start every request. Local and remote requests run at the same time.

    ``local_workers`` limits only this machine. ``remote_workers`` limits how
    many requests are in flight on remote machines **at once, across all of
    them**. ``None`` on either side starts every request of that kind at once,
    which is what both did before the remote limit existed, so an unset limit
    changes nothing.

    The two counts are separate because they bound different things. Local
    workers bound processes competing for one machine's CPU and GPU. Remote
    workers bound the fan-out itself -- the concurrent ssh sessions and file
    transfers this machine opens -- which is why the remote slot is held for the
    whole leg, transfers included, and not just for the run.

    A per-machine remote limit is a different question and is not this: the host
    list already answers it, because a name repeated in that list is a second
    worker on that machine.
    """
    if local_workers is not None and local_workers < 1:
        raise ValueError("Local workers must be at least 1. Omit the limit to start every local request.")
    if remote_workers is not None and remote_workers < 1:
        raise ValueError("Remote workers must be at least 1. Omit the limit to start every remote request.")
    local_count = sum(1 for request in requests if is_local_host(request.host))
    remote_count = len(requests) - local_count
    slots = threading.Semaphore(local_workers if local_workers is not None else max(local_count, 1))
    remote_slots = threading.Semaphore(remote_workers if remote_workers is not None else max(remote_count, 1))

    def launch(request: GenerationRequest) -> RunResult:
        argv = execution_argv(request)
        if is_local_host(request.host):
            with slots:
                code = runner(argv)
        else:
            with remote_slots:
                transfer(request.host, request.local_input, request.remote_input)
                for local_path, remote_path in request.companions:
                    transfer(request.host, local_path, remote_path)
                code = runner(argv)
                if code == 0 and retrieve is not None:
                    retrieve(request.host, request.remote_out_dir, request.local_input.parent)
        return RunResult(host=request.host, argv=argv, returncode=code)

    if not requests:
        return []
    with ThreadPoolExecutor(max_workers=len(requests)) as pool:
        return list(pool.map(launch, requests))


def subprocess_runner(argv: list[str]) -> int:
    """Run ``argv`` on this machine and return its exit code.

    The child is held in :data:`_running` while it runs, so :func:`stop_running`
    can end it. Generation is the long step -- hours -- and with no handle on the
    child a caller that has been asked to stop can only wait for it. Remote
    requests come through here too: their ``argv`` is the ``ssh`` command.

    An interrupt kills the child and is re-raised, which is what
    ``subprocess.run`` did and what Ctrl-C at the panel still expects.
    """
    if not argv:
        raise ValueError("A generation command needs a program to run.")
    process = subprocess.Popen([local_program(argv[0]), *argv[1:]])
    with _running_lock:
        _running.add(process)
    try:
        return process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        with _running_lock:
            _running.discard(process)


def stop_running() -> int:
    """End every generation child this process started, and say how many there were.

    ``terminate`` rather than ``kill``, so the child can close what it was
    writing. A terminated child makes ``wait`` return a negative code, which
    reaches the caller as an ordinary failed request rather than an exception.

    This is best effort by nature: for a remote host the signal reaches the local
    ``ssh``, not the job on the other machine, and a file copy already in flight
    is not held here at all.
    """
    with _running_lock:
        processes = list(_running)
    for process in processes:
        try:
            process.terminate()
        except OSError:
            continue
    return len(processes)


def local_program(name: str) -> str:
    """``name`` as something this machine can start.

    ``rfd4-generate`` is a console script of the environment that installed this
    package, so it is on ``PATH`` only while that environment is activated. The
    panel is normally started by its own path instead -- ``.venv/bin/human``, or
    ``.venv/bin/python -m`` -- and neither puts ``.venv/bin`` on ``PATH``. The
    script is then sitting beside the running interpreter and is still not found,
    which is the reported "No such file or directory: 'rfd4-generate'".

    A name carrying a directory is already a path and is left alone, as is one
    ``PATH`` resolves. A name found nowhere is left alone too, so the error still
    names what is missing rather than a path it guessed at. Only this machine's
    own commands come through here: a command for another machine is carried by
    ``ssh``, and that machine's ``PATH`` is the one that has to find it.
    """
    if os.sep in name or (os.altsep is not None and os.altsep in name):
        return name
    found = shutil.which(name)
    if found is not None:
        return found
    beside = Path(sys.executable).parent / name
    for candidate in (beside, beside.with_name(f"{beside.name}.exe")):
        if candidate.is_file():
            return str(candidate)
    return name


def scp_transfer(host: str, local_path: Path, remote_path: str) -> None:
    """Copy one generation file to ``host`` before generation."""
    parent = str(Path(remote_path).parent).replace("\\", "/")
    if parent in {"", "."}:
        parent = "."
    subprocess.run(["ssh", host, f"mkdir -p {shlex.quote(parent)}"], check=True)
    subprocess.run(["scp", "-q", str(local_path), f"{host}:{remote_path}"], check=True)


def scp_retrieve(host: str, remote_out_dir: str, local_out_dir: Path) -> None:
    """Copy a finished remote output directory back to this machine."""
    local_out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["scp", "-q", "-r", f"{host}:{remote_out_dir}", str(local_out_dir)], check=True)


def _command_input_name(command: list[str]) -> str:
    for part in command:
        if part.startswith("inputs="):
            return Path(part.split("=", 1)[1]).name
    raise RuntimeError("Each generation command needs an inputs path.")


def _companions(local_input: Path, root: str) -> tuple[tuple[Path, str], ...]:
    if not local_input.name.endswith(".yaml"):
        return ()
    companion = _condition_spec_input(local_input)
    return ((companion, f"{root}/{companion.name}"),)


def _condition_spec_input(spec_path: Path) -> Path:
    for line in spec_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("input:"):
            continue
        raw = stripped.split(":", 1)[1].strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
            raw = raw[1:-1]
        if not raw:
            break
        path = Path(raw) if Path(raw).is_absolute() else spec_path.parent / raw
        if not path.is_file():
            raise ValueError(f"Condition spec {spec_path.name} names missing input {raw}.")
        return path
    raise ValueError(f"Condition spec {spec_path.name} has no input file.")


def _rewrite(part: str, request: GenerationRequest) -> str:
    if part.startswith("inputs="):
        return f"inputs={request.remote_input}"
    if part.startswith("out_dir="):
        return f"out_dir={request.remote_out_dir}"
    return part
