"""Run a registered individual PK-Sim snapshot with the pinned portable Windows solver."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path, PureWindowsPath

from rfproteina.disease.screening_tools import data_dir

VERSION = "12.3.173"
ARCHIVE_SHA256 = "94a704b67c05042131cd5e51b4237b1633466a08cd6ec77506322b4d009c8567"
ARCHIVE_SIZE = 109_150_975
ARCHIVE_URL = (
    f"https://github.com/Open-Systems-Pharmacology/PK-Sim/releases/download/v{VERSION}/PKSim-Portable.{VERSION}.zip"
)
FILES_SHA256 = "cfea979f91b326d61083104f4cbb9d8d9ea505ebd4697cb1a5f04781a2c5a57b"
MAX_SNAPSHOT = 16 * 1024 * 1024
MAX_EXPORT = 8 * 1024 * 1024
TIMEOUT = 180
_SHA = re.compile(r"[0-9a-f]{64}")
STANDARD_INCHIKEY = re.compile(r"[A-Z]{14}-[A-Z]{8}SA-[A-Z]")
_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.,+()=\-]{0,119}")


class PKSimError(ValueError):
    """Named PK-Sim input, execution or output diagnostic."""


def cache_path() -> Path:
    path = data_dir(".cache", "pksim")
    # A shared .env.local may name the mounted Windows drive using WSL syntax.
    if os.name == "nt" and (mounted := re.fullmatch(r"/mnt/([A-Za-z])/(.+)", path.as_posix())):
        path = Path(f"{mounted[1]}:/{mounted[2]}")
    return path.resolve()


def engine_path() -> Path:
    return cache_path() / VERSION


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _read(path: Path, size: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > size:
        raise ValueError("invalid regular file")
    with os.fdopen(os.open(path, flags), "rb") as inp:
        opened = os.fstat(inp.fileno())
        if not os.path.samestat(before, opened) or not stat.S_ISREG(opened.st_mode) or opened.st_size > size:
            raise ValueError("file changed")
        data = inp.read(size + 1)
        after = os.fstat(inp.fileno())
    if (
        len(data) != opened.st_size
        or (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
        != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        or len(data) > size
    ):
        raise ValueError("file changed")
    return data


@lru_cache(maxsize=256)
def _file_hash(path: Path, signature: tuple[int, ...]) -> str:
    # The largest constituent of this pinned release is the 30.7 MB SQLite database.
    return hashlib.sha256(_read(path, min(signature[2], 32 * 1024 * 1024))).hexdigest()


def installed() -> bool:
    try:
        root = engine_path()
        if root.is_symlink() or root.is_junction():
            return False
        entries = _read(root / "release-manifest.json", 64 * 1024)
        if hashlib.sha256(entries).hexdigest() != FILES_SHA256:
            return False
        _winpath(root)
        expected = json.loads(entries)
        expected_dirs = {p.as_posix() for name in expected for p in Path(name).parents if p != Path(".")}
        actual = set()
        for dirname, directories, files in os.walk(root):
            parent = Path(dirname)
            for name in directories:
                d = parent / name
                if d.is_symlink() or d.is_junction() or d.relative_to(root).as_posix() not in expected_dirs:
                    return False
            actual.update((parent / name).relative_to(root).as_posix() for name in files)
        if actual != {*expected, "release-manifest.json"}:
            return False
        for filename, digest in expected.items():
            path = root / filename
            info = path.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or _file_hash(path, (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
                != digest
            ):
                return False
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    return True


def _json_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = dict(pairs)
    if len({k.casefold() for k in result}) != len(pairs) or any(k.startswith("$") for k in result):
        raise ValueError("ambiguous or unsupported snapshot metadata")
    return result


def parse_snapshot(data: bytes) -> dict:
    try:
        if len(data) > MAX_SNAPSHOT:
            raise ValueError
        model = json.loads(data.decode("utf-8-sig"), object_pairs_hook=_json_pairs)
        if not isinstance(model, dict) or model.get("ApplicationName") != "PK-Sim":
            raise ValueError
        if not isinstance(model.get("Simulations"), list) or not 1 <= len(model["Simulations"]) <= 256:
            raise ValueError
        canonical_json(model)  # also reject NaN, Infinity and numbers outside finite binary64
        return model
    except (ValueError, RecursionError, UnicodeError) as exc:
        raise PKSimError("pksim_unsupported_snapshot") from exc


def _registered(digest: str) -> tuple[bytes, dict]:
    if not isinstance(digest, str) or not _SHA.fullmatch(digest):
        raise PKSimError("pksim_model_unregistered")
    directory = cache_path() / "models" / digest
    try:
        if directory.is_symlink() or directory.is_junction():
            raise ValueError
        data = _read(directory / "snapshot.json", MAX_SNAPSHOT)
        binding = json.loads(_read(directory / "identity.json", 4096), object_pairs_hook=_json_pairs)
        if hashlib.sha256(data).hexdigest() != digest or not isinstance(binding, dict):
            raise ValueError
        if set(binding) != {"model_sha256", "compound", "inchikey"} or binding["model_sha256"] != digest:
            raise ValueError
        if not isinstance(binding["inchikey"], str) or not STANDARD_INCHIKEY.fullmatch(binding["inchikey"]):
            raise ValueError
        return data, binding
    except (OSError, ValueError, RecursionError) as exc:
        raise PKSimError("pksim_model_unregistered") from exc


def load_model(digest: str, compound: str) -> tuple[bytes, str]:
    data, binding = _registered(digest)
    if binding["compound"] != compound:
        raise PKSimError("pksim_compound_mismatch")
    return data, binding["inchikey"]


def valid_simulation_name(name: object) -> bool:
    """Whether the selected simulation can map to one native result filename."""
    return (
        isinstance(name, str)
        and bool(_FILENAME.fullmatch(name))
        and not PureWindowsPath(name).is_reserved()
        and not name.endswith((".", " "))
    )


def _individual_snapshot(data: bytes, simulation: str, compound: str) -> bytes:
    return canonical_json(prepare_individual_project(parse_snapshot(data), simulation, compound))


def prepare_individual_project(model: dict, simulation: str, compound: str) -> dict:
    """Validate and project a freshly parsed project in place for a single solver run."""
    if not valid_simulation_name(simulation):
        raise PKSimError("pksim_unsupported_snapshot")
    matches = [s for s in model["Simulations"] if isinstance(s, dict) and s.get("Name") == simulation]
    if len(matches) != 1 or not matches[0].get("Individual") or matches[0].get("Population"):
        raise PKSimError("pksim_unsupported_snapshot")
    selected = matches[0]
    # Simulation-local copies can change the individual, compound or protocol cited
    # in the project. We cannot attribute that override to the registered context.
    if any(k.casefold() == "alteredbuildingblocks" and value for k, value in selected.items()):
        raise PKSimError("pksim_unsupported_snapshot")
    compounds = selected.get("Compounds")
    if not isinstance(compounds, list) or not any(isinstance(c, dict) and c.get("Name") == compound for c in compounds):
        raise PKSimError("pksim_compound_mismatch")
    # Keep every scientific building block and effective solver setting. Study analysis and
    # experimental overlays can refer to files or other simulations, and are not solver inputs.
    scientific_fields = {
        "ApplicationName",
        "Version",
        "Description",
        "Name",
        "ExpressionProfiles",
        "Individuals",
        "Compounds",
        "Formulations",
        "Protocols",
        "ObserverSets",
        "Events",
        "Simulations",
    }
    study_fields = {
        "ObservedData",
        "ParameterIdentifications",
        "SimulationComparisons",
        "SensitivityAnalyses",
        "QualificationPlans",
        "Populations",
    }
    for field in tuple(model):
        if field in study_fields or field.endswith("Classifications"):
            del model[field]
        elif field not in scientific_fields:
            raise PKSimError("pksim_unsupported_snapshot")
    for field in tuple(selected):
        if field.casefold() in {
            "observeddata",
            "individualanalyses",
            "populationanalyses",
            "outputmappings",
            "hasresults",
        }:
            del selected[field]
    model["Name"] = "model"
    model["Simulations"] = [selected]
    _require_self_contained(model)
    return model


def _require_self_contained(model: dict) -> None:
    """A solver input may reference PK-Sim quantities, never an external file or command."""
    # Newtonsoft's DTO binding accepts differently cased names. The provenance
    # reader uses the exact serialized field names, so both must see the same case.
    provenance_fields = {
        name.casefold(): name
        for name in (
            "Name",
            "Individual",
            "Population",
            "OriginData",
            "Species",
            "Compound",
            "Compounds",
            "IsSmallMolecule",
            "Protocol",
            "ApplicationType",
            "Parameters",
            "Unit",
            "Value",
            "OutputSchema",
            "OutputSelections",
        )
    }
    pending = [model]
    while pending:
        current = pending.pop()
        if isinstance(current, list):
            pending.extend(current)
        elif isinstance(current, dict):
            for key, value in current.items():
                field = key.casefold()
                if field in provenance_fields and key != provenance_fields[field]:
                    raise PKSimError("pksim_unsupported_snapshot")
                if field in {
                    "file",
                    "filename",
                    "filepath",
                    "inputfile",
                    "inputfiles",
                    "url",
                    "uri",
                    "script",
                    "scriptfile",
                    "command",
                    "executable",
                    "assembly",
                    "directory",
                }:
                    raise PKSimError("pksim_unsupported_snapshot")
                if (
                    field == "path"
                    and isinstance(value, str)
                    and (
                        value.startswith(("/", "\\", "./", ".\\", "../", "..\\"))
                        or re.match(r"^[a-z][a-z0-9+.-]*:", value, re.I)
                    )
                ):
                    raise PKSimError("pksim_unsupported_snapshot")
                pending.append(value)


def _winpath(path: Path) -> str:
    if os.name == "nt":
        return str(path)
    if not Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists() and not os.getenv("WSL_INTEROP"):
        raise PKSimError("pksim_runtime_unavailable")
    # A Windows process must be able to open the private input/output directory from WSL.
    wslpath = Path("/usr/bin/wslpath")
    if wslpath.exists():
        try:
            proc = subprocess.run([wslpath, "-w", str(path)], capture_output=True, check=True, timeout=5)
            value = proc.stdout.decode("utf-8").strip()
        except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
            raise PKSimError("pksim_runtime_unavailable") from exc
        if re.match(r"^[A-Za-z]:\\", value):
            return value
    raise PKSimError("pksim_runtime_unavailable")


def run(snapshot: bytes, simulation: str) -> bytes:
    """Return one native individual result CSV. CLI success alone does not indicate solver success."""
    if not isinstance(snapshot, bytes) or len(snapshot) > MAX_SNAPSHOT:
        raise PKSimError("pksim_unsupported_snapshot")
    _, binding = _registered(hashlib.sha256(snapshot).hexdigest())
    single = _individual_snapshot(snapshot, simulation, binding["compound"])
    if not installed():
        raise PKSimError("pksim_runtime_unavailable")
    try:
        with tempfile.TemporaryDirectory(prefix="run-", dir=cache_path()) as scratch:
            workspace = Path(scratch)
            inputs, outputs = workspace / "source", workspace / "results"
            inputs.mkdir()
            outputs.mkdir()
            (inputs / "model.json").write_bytes(single)
            subprocess.run(
                [
                    str(engine_path() / "PKSim.CLI.exe"),
                    "run",
                    "--input",
                    _winpath(inputs),
                    "--output",
                    _winpath(outputs),
                    "--csv",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=workspace,
                check=True,
                timeout=TIMEOUT,
            )
            try:
                result = outputs / f"model-{simulation}-Results.csv"
                summary = outputs / "results.csv"
                if set(outputs.iterdir()) != {result, summary}:
                    raise ValueError
                # PK-Sim 12.3 writes the simulation name unquoted, including any literal commas.
                if _read(summary, 1024).decode("utf-8-sig").splitlines() != ['"Sim_id"', simulation]:
                    raise ValueError
                raw = _read(result, MAX_EXPORT)
                if not raw:
                    raise ValueError
                return raw
            except (OSError, ValueError) as exc:
                raise PKSimError("pksim_invalid_export") from exc
    except subprocess.TimeoutExpired as exc:
        raise PKSimError("pksim_execution_timeout") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise PKSimError("pksim_execution_failed") from exc
