#!/usr/bin/env python3
"""Install ThermoMPNN's weights and virtual environment under ``DATA_PATH``.

The source is not cloned. It is vendored in this repository, at
``screening subpackage/thermompnn/ThermoMPNN``, so the code that runs is code a
reviewer can read in the same diff as everything else. What this script installs
is the part that is deliberately *not* in the repository: the two weight files
the model loads, and the virtual environment that can run them.

Both weight files are downloaded from the upstream commit
``screening subpackage/thermompnn/manifest.toml`` pins, and each is checked
against the sha256 recorded there before it is kept. A file that does not match
is deleted rather than used, because a stability number from unidentified
weights is worse than no number at all.

The environment is built from the vendored ``pyproject.toml`` and ``uv.lock``.
That lockfile selects Python 3.10 and the CPU build of torch 2.5.1; a newer
torch defaults ``torch.load`` to ``weights_only=True`` and refuses these
checkpoints.

    python scripts/install_thermompnn.py
    python scripts/install_thermompnn.py --have

``stability.py`` reads the source from the checkout and the weights from
``DATA_PATH/thermompnn/ThermoMPNN``; ``thermompnn_runner.py`` is what joins them.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

import tomllib

_RAW = "https://raw.githubusercontent.com/Kuhlman-Lab/ThermoMPNN"
_CHECKPOINT = "models/thermoMPNN_default.pt"
_CHUNK = 1 << 20
_TIMEOUT_S = 300.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _manifest_path() -> Path:
    return _repo_root() / "screening subpackage" / "thermompnn" / "manifest.toml"


def _manifest() -> dict:
    path = _manifest_path()
    if not path.is_file():
        raise SystemExit(f"{path} is missing, so the upstream commit and checksums are unknown.")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _code_root() -> Path:
    return _repo_root() / "screening subpackage" / "thermompnn" / "ThermoMPNN"


def _unquote(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _ensure_data_path() -> None:
    if os.environ.get("DATA_PATH", "").strip():
        return
    env_file = _repo_root() / ".env.local"
    if not env_file.is_file():
        raise SystemExit("DATA_PATH is unset and .env.local does not name one.")
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "DATA_PATH":
            os.environ["DATA_PATH"] = _unquote(value)
            return
    raise SystemExit("DATA_PATH is unset and .env.local does not name one.")


def _root() -> Path:
    return Path(os.environ["DATA_PATH"]) / "thermompnn"


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            sha.update(chunk)
    return sha.hexdigest()


def _have(weights: Path, entries: list[dict]) -> int:
    """Report each weight file and the environment, then stop."""
    ok = True
    for entry in entries:
        target = weights / entry["path"]
        present = target.is_file()
        size = target.stat().st_size if present else 0
        matches = present and size == entry["bytes"] and _digest(target) == entry["sha256"]
        print(f"weight      {target}")
        print(f"status      {'verified' if matches else 'present but unverified' if present else 'absent'}")
        if present:
            print(f"bytes       {size} (expected {entry['bytes']})")
        ok = ok and matches
    code = _code_root()
    print(f"source      {code}")
    print(f"status      {'present' if (code / 'analysis' / 'custom_inference.py').is_file() else 'absent'}")
    python = _venv_python(_root() / "venv")
    print(f"runner      {python if python else 'absent'}")
    return 0 if ok and python is not None else 1


def _venv_python(venv: Path) -> Path | None:
    for relative in ("bin/python", "Scripts/python.exe"):
        candidate = venv / relative
        if candidate.is_file():
            return candidate
    return None


def _venv_version(venv: Path) -> str:
    """The interpreter version recorded in ``pyvenv.cfg``, or an empty string."""
    cfg = venv / "pyvenv.cfg"
    if not cfg.is_file():
        return ""
    found = ""
    for line in cfg.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "version_info":
            return value.strip()
        if sep and key.strip() == "version":
            found = value.strip()
    return found


def _fetch(commit: str, entry: dict, weights: Path) -> Path:
    """Download one weight file and keep it only if it matches the manifest.

    Written to a neighbouring ``.part`` and renamed on success, so an
    interrupted download cannot be mistaken for an installed file by the next
    run, or by ``stability.predictor_available``.
    """
    target = weights / entry["path"]
    if target.is_file() and target.stat().st_size == entry["bytes"] and _digest(target) == entry["sha256"]:
        print(f"verified    {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    url = f"{_RAW}/{commit}/{entry['path']}"
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, _CHUNK)
    size = partial.stat().st_size
    digest = _digest(partial)
    if size != entry["bytes"] or digest != entry["sha256"]:
        partial.unlink()
        raise SystemExit(
            f"{entry['path']} does not match the manifest and was deleted.\n"
            f"  expected {entry['bytes']} bytes, sha256 {entry['sha256']}\n"
            f"  received {size} bytes, sha256 {digest}"
        )
    partial.replace(target)
    print(f"verified    {target}  ({size} bytes)")
    return target


def _note_legacy_clone(weights: Path) -> None:
    """Say so when the pre-vendoring clone is still there, and change nothing."""
    if not (weights / ".git").exists():
        return
    print(
        f"note        {weights} still holds the upstream clone this install replaced.\n"
        "            Its code and .git are no longer used; the weight directories under it\n"
        "            are. Nothing here was deleted -- removing the rest is your call."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install ThermoMPNN's weights and environment.")
    parser.add_argument("--have", action="store_true", help="report the weights and runner, then stop")
    args = parser.parse_args(argv)
    _ensure_data_path()
    manifest = _manifest()
    entries = manifest.get("weights", [])
    if not entries:
        raise SystemExit(f"{_manifest_path()} lists no [[weights]], so there is nothing to verify.")
    root = _root()
    weights = root / "ThermoMPNN"
    if args.have:
        return _have(weights, entries)

    code = _code_root()
    for relative in ("analysis/custom_inference.py", "pyproject.toml", "uv.lock", "local.yaml"):
        if not (code / relative).is_file():
            raise SystemExit(
                f"The vendored source is incomplete: {code / relative} is missing. "
                "It is tracked in this repository, so restore it with git rather than cloning."
            )
    commit = manifest["upstream"]["commit"]
    weights.mkdir(parents=True, exist_ok=True)
    _note_legacy_clone(weights)
    for entry in entries:
        _fetch(commit, entry, weights)

    venv = root / "venv"
    recorded = _venv_version(venv)
    if venv.exists() and not recorded.startswith("3.10"):
        print(f"replacing virtual environment (found Python {recorded or 'unknown'})")
        shutil.rmtree(venv)
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(venv)
    env["UV_LINK_MODE"] = "copy"
    print("installing the locked CPU stack (Python 3.10, torch 2.5.1)")
    subprocess.run(
        ["uv", "sync", "--frozen", "--extra", "cpu", "--python", "3.10"],
        cwd=code,
        env=env,
        check=True,
    )
    python = _venv_python(venv)
    if python is None:
        raise SystemExit(f"uv finished but no interpreter is present under {venv}.")

    # Proves the split layout end to end: the vendored code builds the model,
    # and both weight files load from DATA_PATH. A wrong weights root fails
    # here rather than on the first design that asks for a ddG.
    runner = _repo_root() / "src" / "rfproteina" / "disease" / "thermompnn_runner.py"
    print("loading the checkpoint")
    subprocess.run(
        [
            str(python),
            "-I",
            str(runner),
            "--code",
            str(code),
            "--weights",
            str(weights),
            "--checkpoint",
            str(weights / _CHECKPOINT),
            "--check-only",
        ],
        check=True,
    )
    print(f"ready       source  {code}")
    print(f"ready       weights {weights}")
    print(f"ready       runner  {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
