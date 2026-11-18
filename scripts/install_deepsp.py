#!/usr/bin/env python3
"""Download the MIT DeepSP models into ``DATA_PATH/deepsp/models``.

DeepSP predicts spatial aggregation propensity and the spatial charge maps from
sequence alone, which is the surrogate standing in for the Rosetta SAP this
project does not run. The runner is ``rfproteina.disease.deepsp`` and it needs
no TensorFlow: the three Conv1D models are evaluated in torch, checked against
DeepSP's own published descriptors.

About 17 MB over six files, and the file names are taken from
``deepsp.DEEPSP_WEIGHTS`` rather than repeated here, so a rename upstream is a
one-place change.

It also fetches DeepSP's own 16 example antibodies and the 480 descriptors it
published for them, about 9 KB. They are not needed to run the models, but
without them ``tests/test_disease_deepsp.py`` can only skip the one test that
proves the torch port reproduces the original.

    python scripts/install_deepsp.py
    python scripts/install_deepsp.py --have
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfproteina.disease.deepsp import DEEPSP_FAMILIES, DEEPSP_SUBDIR, DEEPSP_WEIGHTS

#: The repository the models are published from, pinned to a branch rather than
#: a tag because DeepSP publishes no tags.
_BASE = "https://raw.githubusercontent.com/Lailabcode/DeepSP/main/DeepSP_models"
#: The repository root, which is where the reference set below lives.
_ROOT = "https://raw.githubusercontent.com/Lailabcode/DeepSP/main"

#: DeepSP's own example antibodies and the descriptors it published for them:
#: 16 sequences and 480 values, about 9 KB. Not needed to run the models, which
#: is why ``deepsp_available`` does not ask for them -- but
#: ``tests/test_disease_deepsp.py`` checks the torch port against these, and
#: without them that test could only skip. A port nobody checked against the
#: original is a guess with a confident interface.
_REFERENCE = ("DeepSP_input.csv", "DeepSP_descriptors_anarci2.csv")
#: Nothing here is bigger than the largest weight file by a wide margin. A
#: redirect to an HTML error page is a few KB; a model is megabytes.
_SMALLEST_MODEL = 4096


def _unquote(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _ensure_data_path() -> None:
    if os.environ.get("DATA_PATH", "").strip():
        return
    env_file = Path(__file__).resolve().parent.parent / ".env.local"
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


def _destination() -> Path:
    return Path(os.environ["DATA_PATH"]) / DEEPSP_SUBDIR


def _wanted() -> list[str]:
    """Every file the runner opens, in a stable order."""
    return [name for family in DEEPSP_FAMILIES for name in DEEPSP_WEIGHTS[family]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--have", action="store_true", help="report the models, then stop")
    parser.add_argument("--force", action="store_true", help="fetch even where a file is already present")
    args = parser.parse_args(argv)
    _ensure_data_path()
    destination = _destination()
    names = _wanted()

    if args.have:
        present = [name for name in names if (destination / name).is_file()]
        print(f"models    {destination}")
        print(f"present   {len(present)} of {len(names)}")
        for name in names:
            path = destination / name
            mark = f"{path.stat().st_size:>9,} B" if path.is_file() else "    absent"
            print(f"  {mark}  {name}")
        for name in _REFERENCE:
            print(f"reference {name}: {'present' if (destination / name).is_file() else 'absent'}")
        return 0 if len(present) == len(names) else 1

    destination.mkdir(parents=True, exist_ok=True)
    written = 0
    for name in names:
        path = destination / name
        if path.is_file() and not args.force:
            print(f"have     {name}")
            continue
        url = f"{_BASE}/{name}"
        print(f"fetching {url}")
        # The URL is built from _BASE above, not from anything a caller passes.
        with urllib.request.urlopen(url, timeout=300) as response:
            payload = response.read()
        if name.endswith(".h5") and len(payload) < _SMALLEST_MODEL:
            # A short body here is an error page, not a model. Writing it would
            # leave a file that looks installed and fails at load.
            raise SystemExit(f"{name} came back as {len(payload)} bytes, which is not a model.")
        path.write_bytes(payload)
        written += 1
        print(f"  {len(payload):>9,} B  sha256 {hashlib.sha256(payload).hexdigest()[:16]}")
    for name in _REFERENCE:
        path = destination / name
        if path.is_file() and not args.force:
            print(f"have     {name}")
            continue
        url = f"{_ROOT}/{name}"
        print(f"fetching {url}")
        with urllib.request.urlopen(url, timeout=300) as response:
            payload = response.read()
        # A CSV, so the check is that it parses as one and carries a header,
        # not a size: an HTML error page is bigger than this file.
        text = payload.decode("utf-8-sig", errors="strict")
        header = text.splitlines()[0] if text.splitlines() else ""
        if "Name" not in header.split(","):
            raise SystemExit(f"{name} has no Name column; its first line is {header[:80]!r}.")
        path.write_bytes(payload)
        written += 1
        print(f"  {len(payload):>9,} B  {name}  ({len(text.splitlines()) - 1} rows)")

    total = sum((destination / name).stat().st_size for name in names if (destination / name).is_file())
    print(f"ready  {destination}  ({written} fetched, {total / 1024 / 1024:.1f} MB total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
