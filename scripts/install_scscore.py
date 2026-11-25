#!/usr/bin/env python
"""Install the pinned SCScore Boolean model in DATA_PATH for offline NumPy inference."""

import argparse
import os
from http.client import HTTPException
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from rfproteina.disease import scscore_model as model


def install(source: Path | None = None) -> Path:
    target = model.weights_path()
    if source is None:
        if target.exists() or target.is_symlink():
            try:
                model.verify_weights(model.read_regular(target, model.WEIGHT_BYTES, "scscore_weights"))
            except model.SCScoreError as exc:
                if str(exc) not in {"scscore_weights_size_limit", "scscore_weights_digest_mismatch"}:
                    raise
            else:
                return target
        with urlopen(model.URL, timeout=60) as response:
            data = response.read(model.WEIGHT_BYTES + 1)
    else:
        data = model.read_regular(source, model.WEIGHT_BYTES, "source_weights")
    model.verify_weights(data)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as staging:
        tmp = Path(staging) / target.name
        with tmp.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-file", type=Path, help="install from a local copy of the released .json.gz checkpoint")
    args = parser.parse_args()
    try:
        install(args.from_file)
    except (model.SCScoreError, OSError, HTTPException) as exc:
        parser.exit(1, f"SCScore installation failed: {exc}\n")
    print(f"Installed {model.MODEL} (SHA-256 {model.SHA256})")


if __name__ == "__main__":
    main()
