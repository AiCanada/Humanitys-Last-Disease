"""Check one screening program's installed footprint and qualify a local workload."""

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory

DISK_LIMIT = 25_000_000_000
VRAM_LIMIT = 96_000_000_000


def _scan(
    roots: list[Path], *, seen: set | None = None, exclude: set | None = None, transient: bool = False
) -> tuple[int, OSError | ValueError | None]:
    """Keep a proven lower bound even if a different root cannot be inspected."""
    pending = list(roots)
    if seen is None:
        seen = set()
    total = 0
    first_error = None
    while pending:
        path = pending.pop()
        try:
            info = path.stat()
            identity = (info.st_dev, info.st_ino) if info.st_ino else path.resolve()
            if identity in seen or (exclude is not None and identity in exclude):
                continue
            seen.add(identity)
            if stat.S_ISDIR(info.st_mode):
                with os.scandir(path) as entries:
                    pending.extend(Path(entry.path) for entry in entries)
            elif stat.S_ISREG(info.st_mode):
                total += info.st_size
            else:
                raise ValueError(f"Unsupported file type in resource root: {path.name}")
        except (OSError, ValueError) as error:
            if transient and isinstance(error, FileNotFoundError):
                # A completed temporary file may disappear between scandir and stat.
                continue
            if first_error is None:
                first_error = error
    return total, first_error


def footprint(roots: list[Path]) -> int:
    """Sum logical file lengths once, including linked targets and overlapping roots."""
    total, error = _scan(roots)
    if error is not None:
        raise error
    return total


def verdict(disk: int | None, vram: int | None) -> str:
    if (disk is not None and disk >= DISK_LIMIT) or (vram is not None and vram > VRAM_LIMIT):
        return "over_limit"
    return "unverified" if disk is None or vram is None else "within_limits"


def _disk_evidence(roots: list[Path], seen: set | None = None) -> dict:
    size, error = _scan(roots, seen=seen)
    if error is None:
        return {"installed_bytes": size}
    return {"installed_bytes": None, "installed_bytes_lower_bound": size, "disk_error": type(error).__name__}


def _cuda_capacity() -> int:
    # The probe exits before the workload starts, releasing its CUDA context.
    # Honor the caller's CUDA visibility and sum every locally visible device.
    code = (
        "import torch; print(sum(torch.cuda.get_device_properties(i).total_memory "
        "for i in range(torch.cuda.device_count())))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    capacity = int(result.stdout.strip())
    if capacity <= 0:
        raise ValueError("No CUDA device capacity was available.")
    return capacity


def _qualification(record: dict) -> str:
    disk = record["installed_bytes"]
    vram = record.get("vram_upper_bound_bytes", record["vram_bytes"])
    disk_limit = record["disk_limit_exclusive_bytes"]
    disk_evidence = max(
        disk or 0, record.get("installed_bytes_lower_bound", 0), record.get("disk_peak_observed_bytes", 0)
    )
    if disk_evidence >= disk_limit:
        return "over_limit"
    if verdict(disk, record["vram_bytes"]) == "over_limit":
        return "over_limit"
    if record.get("blocker"):
        return "unverified"
    return verdict(disk, vram)


def _run_workload(
    args: argparse.Namespace, executable: str | None, roots: list[Path], installed_ids: set, record: dict
) -> None:
    record["workload_exit_code"] = None
    if _qualification(record) == "over_limit":
        record["blocker"] = "disk_limit"
        return
    if record["installed_bytes"] is None:
        record["blocker"] = "installed_footprint_unverified"
        return
    if executable is None:
        record["blocker"] = "executable_unavailable"
        return
    env = os.environ.copy()
    if args.cuda:
        try:
            capacity = _cuda_capacity()
        except (OSError, ValueError, subprocess.SubprocessError):
            record["blocker"] = "cuda_capacity_unavailable"
            return
        record.update(vram_upper_bound_bytes=capacity, vram_evidence="visible_cuda_capacity")
        if capacity > VRAM_LIMIT:
            record["blocker"] = "cuda_capacity_above_limit"
            return
    else:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        record["vram_evidence"] = "cpu_cuda_hidden"
    record["installed_bytes_before_run"] = record["installed_bytes"]
    try:
        with TemporaryDirectory(prefix="screening-resources-") as directory:
            scratch = Path(directory).resolve()
            env.update(TMPDIR=str(scratch), TEMP=str(scratch), TMP=str(scratch))
            working_dir = scratch if args.scratch_cwd else None
            with subprocess.Popen([executable, *args.run[1:]], env=env, cwd=working_dir, stdout=2) as process:
                peak = samples = 0
                while True:
                    size, error = _scan([scratch], exclude=installed_ids, transient=True)
                    peak = max(peak, size)
                    samples += 1
                    if error is not None:
                        record.update(blocker="scratch_size_unverified", scratch_error=type(error).__name__)
                    if process.returncode is not None:
                        break
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=0.05)
                record.update(workload_exit_code=process.returncode, scratch_peak_observed_bytes=peak,
                              scratch_samples=samples, scratch_includes_cwd=args.scratch_cwd)
    except OSError:
        record["blocker"] = "workload_start_failed"
        return
    if record["workload_exit_code"]:
        record["blocker"] = "workload_failed"
    record.update(_disk_evidence(roots))
    record["disk_peak_observed_bytes"] = max(
        record["installed_bytes_before_run"] + peak,
        record["installed_bytes"] or record.get("installed_bytes_lower_bound", 0),
    )
    if record["disk_peak_observed_bytes"] >= record["disk_limit_exclusive_bytes"]:
        record["blocker"] = "disk_limit"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", help="name of the program and configuration measured")
    parser.add_argument("--root", type=Path, action="append", required=True,
                        help="repeat for the runtime, executable, code, weights, and databases used")
    parser.add_argument("--disk-limit-bytes", type=int, default=DISK_LIMIT,
                        help="exclusive disk budget for this run, up to the project limit")
    parser.add_argument("--scratch-cwd", action="store_true",
                        help="run in the monitored temporary directory; pass absolute input and program paths")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cpu", action="store_true", help="this configuration executes entirely on CPU")
    group.add_argument("--peak-vram-bytes", type=int, help="measured device memory for the tested workload")
    group.add_argument("--cuda", action="store_true",
                       help="qualify a local CUDA workload against visible device capacity")
    parser.add_argument("--run", nargs=argparse.REMAINDER, metavar="COMMAND",
                        help="run this command and its arguments after checking resources; put this option last")
    args = parser.parse_args(argv)
    if not 0 < args.disk_limit_bytes <= DISK_LIMIT:
        parser.error("--disk-limit-bytes must be positive and no greater than the project limit")
    if args.scratch_cwd and not args.run:
        parser.error("--scratch-cwd requires --run")
    if args.peak_vram_bytes is not None and args.peak_vram_bytes < 0:
        parser.error("--peak-vram-bytes must be nonnegative")
    if args.run == [] or (args.run and not (args.cpu or args.cuda)) or (args.cuda and not args.run):
        parser.error("Workload qualification requires --cpu or --cuda and a command after --run.")
    vram = 0 if args.cpu else args.peak_vram_bytes
    evidence = "supplied_peak" if args.peak_vram_bytes is not None else "unknown"
    record = {"program": args.program, "vram_bytes": vram,
              "vram_evidence": "cpu_declared" if args.cpu else evidence,
              "disk_limit_exclusive_bytes": args.disk_limit_bytes, "vram_limit_inclusive_bytes": VRAM_LIMIT}
    roots = list(args.root)
    executable = shutil.which(args.run[0]) if args.run else None
    if executable:
        roots.append(Path(executable))
    installed_ids: set = set()
    record.update(_disk_evidence(roots, installed_ids if args.run else None))
    if args.run:
        # Preserve an executable resolved relative to the caller when changing the working directory.
        executable = os.path.abspath(executable) if executable else None
        _run_workload(args, executable, roots, installed_ids, record)
    state = _qualification(record)
    record["status"] = state
    print(json.dumps(record, sort_keys=True))
    return 0 if state == "within_limits" else 1


if __name__ == "__main__":
    raise SystemExit(main())
