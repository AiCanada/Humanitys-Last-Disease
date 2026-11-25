#!/usr/bin/env python3
"""Install the single pinned DELPHI PSR checkpoint from a local download."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from rfproteina.disease import delphi_psr


def install(source: Path, target: Path) -> bool:
    """Publish verified bytes atomically without replacing any existing file."""
    raw = delphi_psr.verified_checkpoint(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".delphi-install-", dir=target.parent) as temporary:
        staged = Path(temporary) / target.name
        with staged.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staged, target)
        except FileExistsError:
            delphi_psr.verified_checkpoint(target)
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="path to the published DELPHI v4 paired one-hot PSR .pt file")
    args = parser.parse_args()
    try:
        changed = install(args.checkpoint, delphi_psr.checkpoint_path())
    except (delphi_psr.DelPhiError, OSError) as error:
        parser.exit(1, f"DELPHI PSR installation failed ({type(error).__name__}): {error}\n")
    if changed:
        print("Installed the verified DELPHI v4 PSR checkpoint.")
    else:
        print("Verified the existing DELPHI PSR checkpoint.")


if __name__ == "__main__":
    main()
