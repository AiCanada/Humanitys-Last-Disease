#!/usr/bin/env python
"""Install the compact, release-verified EPA structure crosswalk used by httk."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

from rfproteina.disease import httk, httk_identity

_MEMBER = "DSSTox_CCD_dump_12092025/DSSToxCCDdump.csv"
_PACKAGE_IDS = """
stopifnot(as.character(utils::packageVersion("httk")) == "2.7.4")
d <- httk::chem.physical_and_invitro.data
utils::write.table(unique(d[c("CAS", "DTXSID")]), file=stdout(), sep="\t",
                   quote=FALSE, row.names=FALSE, col.names=FALSE)
"""


def _reference_pairs() -> set[tuple[str, str]]:
    rscript = httk.runtime_path()
    if rscript is None:
        raise ValueError("Install and configure R with the pinned httk 2.7.4 release first.")
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, **dict.fromkeys(("TMPDIR", "TEMP", "TMP"), ".")}
        raw = subprocess.run(
            [str(rscript), "--vanilla", "-e", _PACKAGE_IDS],
            cwd=tmp,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
            timeout=60,
        ).stdout
    if len(raw) > 1024**2:
        raise ValueError("Unexpected httk chemical reference table size.")
    pairs = set()
    for line in raw.decode("utf-8").splitlines():
        cas, sid = line.split("\t")
        if httk.valid_cas(cas) and httk.valid_dtxsid(sid):
            pairs.add((cas, sid))
    return pairs


def build_index(archive: Path, pairs: set[tuple[str, str]]) -> bytes:
    with archive.open("rb") as stream:
        if os.fstat(stream.fileno()).st_size != httk_identity.ARCHIVE_BYTES or (
            hashlib.file_digest(stream, "sha256").hexdigest() != httk_identity.ARCHIVE_SHA256
        ):
            raise ValueError("EPA DSSTox archive integrity check failed.")
        with zipfile.ZipFile(stream) as bundle, bundle.open(_MEMBER) as data:
            # Other columns include free-text chemical names. Only the three identifier columns are used.
            reader = csv.DictReader(io.TextIOWrapper(data, encoding="utf-8-sig", errors="surrogateescape", newline=""))
            if not reader.fieldnames or not {"CASRN", "DTXSID", "INCHIKEY"}.issubset(reader.fieldnames):
                raise ValueError("Unexpected EPA DSSTox table columns.")
            selected = {}
            for row in reader:
                pair = (row["CASRN"], row["DTXSID"])
                if pair in pairs:
                    key = row["INCHIKEY"]
                    # Duplicate assignments, including a structural and a nonstructural row,
                    # do not identify a unique test substance.
                    selected[pair] = key if pair not in selected else None
    return "".join(
        f"{cas}\t{sid}\t{key}\n"
        for (cas, sid), key in sorted(selected.items())
        if key and httk_identity.STANDARD_KEY.fullmatch(key)
    ).encode("ascii")


def install(archive: Path | None) -> Path:
    target = httk_identity.index_path()
    if target.exists() or target.is_symlink():
        httk_identity.load_index()
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as staging:
        work = Path(staging)
        if archive is None:
            archive = work / "dsstox.zip"
            with urlopen(httk_identity.ARCHIVE_URL, timeout=90) as source, archive.open("xb") as dest:
                left = httk_identity.ARCHIVE_BYTES
                while left and (chunk := source.read(min(left, 1024**2))):
                    dest.write(chunk)
                    left -= len(chunk)
                if left or source.read(1):
                    raise ValueError("EPA DSSTox download length mismatch.")
        rows = build_index(archive, _reference_pairs())
        if hashlib.sha256(rows).hexdigest() != httk_identity.INDEX_SHA256:
            raise ValueError("The derived index differs from the pinned intersection of the EPA releases.")
        part = work / "index.tsv"
        part.write_bytes(rows)
        os.replace(part, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-file", type=Path, help="use the original DSSTox December 2025 CSV archive on disk")
    args = parser.parse_args()
    try:
        install(args.from_file)
    except (OSError, ValueError, csv.Error, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"EPA httk identity setup failed. {exc}\n")
    print(f"Verified {httk_identity.SOURCE} for httk {httk.VERSION}. Index SHA-256 {httk_identity.INDEX_SHA256}")


if __name__ == "__main__":
    main()
