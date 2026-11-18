#!/usr/bin/env python3
"""Download the Octant CYP release once and store CYP3A4 as nanomolar.

The table is the CC BY 4.0 OpenADMET release, 2.27 MB, five TSVs. Only
``inhibition.tsv`` carries ``CYP3A4_pIC50``. That column is −log10(IC50 in
molar); the subset written beside the download has already been converted
with ``10 ** (9 - pIC50)``. CYP2J2 stays in ``reactivity.tsv`` and is not a
panel row.

    python scripts/install_octant.py
    python scripts/install_octant.py --have
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rfproteina.disease.octant import INHIBITION_FILE, OCTANT_FILES, SUBSET_FILE, ensure_release, octant_cyp3a4_subset


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


def _root() -> Path:
    return Path(os.environ["DATA_PATH"]) / "octant"


def _have(root: Path) -> int:
    present = [name for name in OCTANT_FILES if (root / name).is_file() and (root / name).stat().st_size > 0]
    subset = root / SUBSET_FILE
    print(f"release    {root}")
    print(f"files      {len(present)}/{len(OCTANT_FILES)}")
    print(f"subset     {subset}")
    print(f"status     {'present' if subset.is_file() else 'absent'}")
    return 0 if len(present) == len(OCTANT_FILES) and subset.is_file() else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--have", action="store_true", help="report the release and the subset, then stop")
    args = parser.parse_args(argv)
    _ensure_data_path()
    root = _root()
    if args.have:
        return _have(root)
    print(f"loading the Octant CYP release into {root}")
    sizes = ensure_release(root)
    for name, size in sizes.items():
        print(f"ready  {root / name}  ({size} bytes)")
    result = octant_cyp3a4_subset(root / INHIBITION_FILE, root)
    if result.blocker:
        raise SystemExit(result.blocker)
    print(f"ready  {result.path}  ({result.rows} CYP3A4 rows, nanomolar)")
    if result.skipped:
        print(f"skipped  {dict(result.skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
