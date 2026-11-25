#!/usr/bin/env python
"""Install the verified portable Windows PK-Sim solver or register an operator-reviewed local model."""

import argparse
import hashlib
import os
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

from rfproteina.disease import pksim_runtime as runtime


def install(archive: Path | None = None) -> Path:
    target = runtime.engine_path()
    if target.exists():
        if runtime.installed():
            return target
        raise ValueError("The existing PK-Sim installation failed verification. Inspect it before reinstalling.")
    target.parent.mkdir(parents=True, exist_ok=True)
    runtime._winpath(target.parent)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        base = Path(tmp)
        if archive is None:
            archive = base / "release.zip"
            with urlopen(runtime.ARCHIVE_URL, timeout=90) as response, archive.open("xb") as dest:
                left = runtime.ARCHIVE_SIZE
                while left and (part := response.read(min(left, 1024 * 1024))):
                    dest.write(part)
                    left -= len(part)
                if left or response.read(1):
                    raise ValueError("PK-Sim release length mismatch")
        with archive.open("rb") as inp:
            if os.fstat(inp.fileno()).st_size != runtime.ARCHIVE_SIZE:
                raise ValueError("PK-Sim release length mismatch")
            if hashlib.file_digest(inp, "sha256").hexdigest() != runtime.ARCHIVE_SHA256:
                raise ValueError("PK-Sim release checksum mismatch")
            # Only extract the exact archive published by OSP. An interrupted install
            # remains private until every constituent file matches the release manifest.
            dest_dir = base / "engine"
            with zipfile.ZipFile(inp) as bundle:
                bundle.extractall(dest_dir)
                manifest = {}
                for entry in bundle.infolist():
                    if not entry.is_dir():
                        with (dest_dir / entry.filename).open("rb") as stream:
                            manifest[entry.filename] = hashlib.file_digest(stream, "sha256").hexdigest()
            packed = runtime.canonical_json(manifest)
            if hashlib.sha256(packed).hexdigest() != runtime.FILES_SHA256:
                raise ValueError("Extracted PK-Sim release failed verification")
            (dest_dir / "release-manifest.json").write_bytes(packed)
            try:
                dest_dir.rename(target)
            except OSError:
                if not runtime.installed():
                    raise
    return target


def register(path: Path, compound: str, inchikey: str) -> str:
    if not runtime.STANDARD_INCHIKEY.fullmatch(inchikey):
        raise ValueError("A full standard InChIKey must identify the registered compound")
    if not compound or len(compound) > 128 or not compound.isprintable() or compound.strip() != compound:
        raise ValueError("Invalid primary compound name")
    data = runtime._read(path, runtime.MAX_SNAPSHOT)
    model = runtime.parse_snapshot(data)
    compounds = model.get("Compounds")
    if (
        not isinstance(compounds, list)
        or len([c for c in compounds if isinstance(c, dict) and c.get("Name") == compound]) != 1
    ):
        raise ValueError("Exactly one project compound must match the registered name")
    digest = hashlib.sha256(data).hexdigest()
    target = runtime.cache_path() / "models" / digest
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        staging = Path(tmp) / "registration"
        staging.mkdir()
        (staging / "snapshot.json").write_bytes(data)
        (staging / "identity.json").write_bytes(
            runtime.canonical_json(
                {
                    "model_sha256": digest,
                    "compound": compound,
                    "inchikey": inchikey,
                }
            )
        )
        try:
            staging.rename(target)
        except OSError:
            if not target.is_dir():
                raise
            _, registered_key = runtime.load_model(digest, compound)
            if registered_key != inchikey:
                raise ValueError("Registered model has a different chemical identity") from None
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--from-file", type=Path, help="install from the official pinned portable ZIP on disk")
    modes.add_argument("--model", type=Path, help="register a trusted, self-contained PK-Sim project snapshot")
    parser.add_argument("--compound", help="exact primary compound name in the reviewed snapshot")
    parser.add_argument("--inchikey", help="full standard InChIKey independently checked for that compound")
    args = parser.parse_args()
    if args.model is not None and not (args.compound and args.inchikey):
        parser.error("--model requires --compound and --inchikey")
    if args.model is None and (args.compound or args.inchikey):
        parser.error("chemical identity is used only with --model")
    try:
        if args.model:
            print("Registered PK-Sim model SHA-256 " + register(args.model, args.compound, args.inchikey))
        else:
            install(args.from_file)
            print(f"Installed PK-Sim {runtime.VERSION} (archive SHA-256 {runtime.ARCHIVE_SHA256})")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"PK-Sim installation or registration failed: {exc}\n")


if __name__ == "__main__":
    main()
