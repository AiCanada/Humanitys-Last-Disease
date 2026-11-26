"""Extract Ames labels from the pinned public ToxScan release with python-lmdb.

Usage: python scripts/extract_toxscan_ames.py --archive dataset.tar.gz --output ames-labels
Only this offline preparation command requires lmdb (tested with 1.7.5).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickletools
import shutil
import struct
import tarfile
import tempfile
import uuid
from collections import Counter
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any

RELEASE_SHA256 = "e199ef53ee13ed2fe598ac96dd1ca4340b2b9aa602677c61e489cbc2dc4dd71f"
RELEASE_SIZE = 658_927_149
DATABASE_SIZE = 2_517_606_400
DATABASE_RECORDS = 39_757
SPLITS = ("train", "valid", "test")
REFERENCE = "https://github.com/FanmengWang/ToxScan/tree/5eecd78a645366fbd2564ccb1c16032abf05a581"
ENDPOINT = "Ames Mutagenicity"
MISSING = -10_000
HEADER_LIMIT = 16_384
Tokens = list[tuple[str, Any]]


def _field(tokens: Tokens, name: str, following: str) -> Tokens:
    start = tokens.index(("SHORT_BINUNICODE", name))
    end = tokens.index(("SHORT_BINUNICODE", following), start + 1)
    return tokens[start + 1 : end]


def _array_item(tokens: Tokens, *, dtype: str, length: int, index: int, format_code: str) -> int | float:
    buffers = [arg for op, arg in tokens if op == "BYTEARRAY8"]
    if (
        ("SHORT_BINUNICODE", dtype) not in tokens
        or ("BININT1", length) not in tokens
        or len(buffers) != 1
        or len(buffers[0]) != struct.calcsize(format_code) * length
    ):
        raise ValueError("Unexpected ToxScan target array encoding")
    return struct.unpack_from(format_code, buffers[0], index * struct.calcsize(format_code))[0]


def read_ames_record(prefix: bytes) -> tuple[str, int | None]:
    """Inspect bounded protocol-5 data as tokens. No pickle operations are executed."""
    tokens = []
    strings_by_memo = {}
    memo_index = 0
    previous = None
    for opcode, arg, _ in pickletools.genops(prefix[:HEADER_LIMIT]):
        name = opcode.name
        if name == "MEMOIZE":
            if previous and previous[0] in ("SHORT_BINUNICODE", "BINUNICODE"):
                strings_by_memo[memo_index] = previous[1]
            memo_index += 1
        elif name != "FRAME":
            tokens.append((name, arg))
        if name == "SHORT_BINUNICODE" and arg == "regression_target":
            break
        previous = (name, arg)
    else:
        raise ValueError("Incomplete ToxScan label header")

    if tokens[:4] != [("PROTO", 5), ("EMPTY_DICT", None), ("MARK", None), ("SHORT_BINUNICODE", "data")]:
        raise ValueError("Unexpected ToxScan record header")
    value = _field(tokens, "data", "title")
    if len(value) != 1 or value[0][0] not in ("SHORT_BINUNICODE", "BINUNICODE"):
        raise ValueError("Missing ToxScan SMILES")
    smiles = value[0][1]
    if not smiles or not smiles.isascii() or not smiles.isprintable():
        raise ValueError("Invalid ToxScan SMILES text")
    titles = [arg for op, arg in _field(tokens, "title", "classification_title") if op == "SHORT_BINUNICODE"]
    class_titles = [
        strings_by_memo.get(arg)
        for op, arg in _field(tokens, "classification_title", "regression_title")
        if op == "BINGET"
    ]
    if len(titles) != 31 or len(class_titles) != 26 or titles[:26] != class_titles or class_titles.count(ENDPOINT) != 1:
        raise ValueError("Inconsistent ToxScan endpoint ordering")
    idx = titles.index(ENDPOINT)
    target = _field(tokens, "target", "classification_target")
    if ("SHORT_BINUNICODE", "<") not in target:
        raise ValueError("Unexpected ToxScan target byte order")
    overall = _array_item(target, dtype="f4", length=31, index=idx, format_code="<f")
    classes = _field(tokens, "classification_target", "regression_target")
    label = _array_item(classes, dtype="i8", length=26, index=idx, format_code="<q")
    if label not in (MISSING, 0, 1) or overall != label:
        raise ValueError("Ames label disagrees with the released classification vector")
    return smiles, None if label == MISSING else label


def _extract_database(db_path: Path, csv_path: Path) -> tuple[dict[str, int | str], set[str]]:
    import lmdb

    counts = Counter()
    labels = {}
    with (
        lmdb.open(str(db_path), subdir=False, readonly=True, lock=False, create=False) as env,
        env.begin(buffers=True) as txn,
        csv_path.open("x", encoding="utf-8", newline="") as out,
    ):
        if txn.stat()["entries"] != DATABASE_RECORDS:
            raise ValueError("Unexpected number of LMDB records")
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(["smiles", "ames"])
        for _, record in txn.cursor():
            smiles, label = read_ames_record(bytes(record[:HEADER_LIMIT]))
            if label is None:
                counts["missing"] += 1
                continue
            if smiles in labels:
                raise ValueError("Duplicate labeled SMILES in one Ames split")
            labels[smiles] = label
            counts["positive" if label else "negative"] += 1
            writer.writerow([smiles, label])
    with csv_path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"records": DATABASE_RECORDS, "labeled": len(labels), **counts, "csv_sha256": digest}, set(labels)


def extract(archive: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Output destination already exists: {output}")
    with archive.open("rb") as source:
        if (
            archive.stat().st_size != RELEASE_SIZE
            or hashlib.file_digest(source, "sha256").hexdigest() != RELEASE_SHA256
        ):
            raise ValueError("ToxScan dataset archive does not match the pinned release")
        source.seek(0)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as tmp, ExitStack() as cleanup:
            work = Path(tmp)
            # Public CSVs inherit their destination's ACL, including across Windows and WSL.
            # Python's temporary directory intentionally restricts its children's Windows ACLs.
            ready = output.with_name(f".{output.name}.{uuid.uuid4().hex}")
            ready.mkdir()
            cleanup.callback(shutil.rmtree, ready, ignore_errors=True)
            splits = {}
            identities = {}
            with tarfile.open(fileobj=source, mode="r|gz") as package:
                for member in package:
                    if member.isdir():
                        continue
                    filename = PurePosixPath(member.name).name
                    if (
                        not member.isfile()
                        or member.size != DATABASE_SIZE
                        or filename not in {s + ".lmdb" for s in SPLITS}
                    ):
                        raise ValueError("Unexpected member in ToxScan dataset release")
                    split = filename.removesuffix(".lmdb")
                    if split in splits:
                        raise ValueError("Repeated split database")
                    # Fixed local path, one database staged at a time. Archive paths are never extracted.
                    db_path = work / "data.lmdb"
                    with package.extractfile(member) as db_input, db_path.open("xb") as db_output:
                        shutil.copyfileobj(db_input, db_output, length=4 << 20)
                    splits[split], identities[split] = _extract_database(db_path, ready / (split + ".csv"))
                    db_path.unlink()
            if set(splits) != set(SPLITS):
                raise ValueError("Missing ToxScan release split")
            overlaps = {
                a + "/" + b: len(identities[a] & identities[b]) for i, a in enumerate(SPLITS) for b in SPLITS[i + 1 :]
            }
            manifest = {
                "source": REFERENCE,
                "archive_sha256": RELEASE_SHA256,
                "endpoint": ENDPOINT,
                "missing_label": MISSING,
                "columns": ["smiles", "ames"],
                "splits": splits,
                "exact_labeled_smiles_overlap": overlaps,
            }
            (ready / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            ready.rename(output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.archive, args.output), indent=2, sort_keys=True))
