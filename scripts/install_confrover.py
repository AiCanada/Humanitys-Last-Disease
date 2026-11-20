#!/usr/bin/env python3
"""Install what the Conformational Dynamics Analyst needs, once, deliberately.

Everything fetchable lives here. The generator this serves will, left to
itself, clone a CUDA library and query a public MSA server the first time it is
asked for an ensemble -- and neither belongs behind a screen:
``screening_tools`` opens by promising its family installs nothing and
downloads nothing, and a design's amino-acid sequence must not leave the
machine because somebody pressed a button on a results page.

So this script does the downloading and the screen does the checking. Each
piece lands under ``DATA_PATH/confrover`` and the screen declines by name until
it is there:

    weights/          the two checkpoints, each checked against manifest.toml
    cutlass/          the CUDA library the generator builds kernels from
    openfold_params/  the folding parameters it embeds a sequence with
    msa/              the alignment cache, which is the network step
    folding_repr/     representations derived from the two above
    igso3/            the SO(3) lookup tables the diffuser writes on first use
    venv/             an interpreter that can import the vendored package

Nothing is written into the checkout. The generator derives every cache path it
uses from one root, so pointing that root here is what keeps a run from
littering the vendored tree -- which matters more than it sounds, because the
tree carries a digest its manifest records.

Run with --check to report what is present and exit without fetching.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import tomllib

#: The pieces, in the order the screen reports them missing.
PIECES = ("weights", "cutlass", "openfold_params", "msa")
#: Pinned rather than tracking a branch. The generator's own code clones this
#: with ``-b <branch> --single-branch``, which is whatever that branch holds on
#: the day it runs; a screen whose kernels change under it is not reproducible.
CUTLASS_REPO = "https://github.com/NVIDIA/cutlass"
CUTLASS_TAG = "v3.5.1"
#: The three packages this project's interpreter is short of. Measured, not
#: guessed: everything else the generator imports is already here.
MISSING_PACKAGES = ("ml-collections", "dm-tree", "mdtraj")


def checkout_root() -> Path:
    """The repository root, by searching upward for the marker file."""
    here = Path(__file__).resolve()
    for folder in (here, *here.parents):
        if (folder / ".project-root").is_file() or (folder / "pyproject.toml").is_file():
            return folder
    return here.parent.parent


def vendored() -> Path:
    """Where the generator's source was vendored."""
    return checkout_root() / "screening subpackage" / "confrover"


def data_root() -> Path | None:
    """``DATA_PATH/confrover``, or None when DATA_PATH is unset."""
    named = os.environ.get("DATA_PATH", "").strip()
    return Path(named) / "confrover" if named else None


def manifest() -> dict:
    """The vendored tree's manifest, which records the weights and their digests."""
    path = vendored() / "manifest.toml"
    with path.open("rb") as handle:
        return tomllib.load(handle)


def digest_of(path: Path) -> str:
    """sha256 of a file, read in blocks so a 75 MB checkpoint is not held whole."""
    found = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            found.update(block)
    return found.hexdigest()


def installed(path: Path) -> bool:
    """Whether a piece is really there, not just the directory it writes into.

    An empty directory counts as absent, matching what the screen checks. This
    script creating ``weights/`` before it had anything to put in it was enough
    to make a wholly uninstalled machine report the weights as present.
    """
    if path.is_file():
        return True
    return path.is_dir() and any(path.iterdir())


def report(root: Path) -> int:
    """Say what is present, and name the first thing the screen would decline on."""
    print(f"DATA_PATH/confrover: {root}")
    missing = []
    for name in PIECES:
        there = installed(root / name)
        print(f"  {name:<18} {'present' if there else 'ABSENT'}")
        if not there:
            missing.append(name)
    for name in ("folding_repr", "igso3", "venv"):
        state = "present" if installed(root / name) else "absent (written on use)"
        print(f"  {name:<18} {state}")
    weights = manifest().get("weights", [])
    for entry in weights:
        path = root / "weights" / str(entry["path"])
        if not path.is_file():
            print(f"  checkpoint ABSENT: {entry['path']}")
            continue
        size = path.stat().st_size
        ok = size == int(entry["bytes"]) and digest_of(path) == str(entry["sha256"])
        print(f"  checkpoint {'verified' if ok else 'DIGEST MISMATCH'}: {entry['path']}")
    if missing:
        print(f"\nThe screen will decline until these are installed: {', '.join(missing)}")
        return 1
    print("\nEverything the screen checks for is present.")
    return 0


def fetch_weights(root: Path, *, offline: bool) -> int:
    """Both checkpoints, each verified against the digest the manifest records.

    Written through a temporary name and moved into place, so an interrupted
    download never leaves a half file that looks installed.
    """
    # Not created until there is something to put in it, so an --offline run
    # or an interrupted one does not leave a directory that reads as installed.
    target = root / "weights"
    for entry in manifest().get("weights", []):
        name = str(entry["path"])
        destination = target / name
        if destination.is_file() and digest_of(destination) == str(entry["sha256"]):
            print(f"  {name}: already verified")
            continue
        source = str(entry.get("source", ""))
        where = str(entry.get("source_path", name))
        if offline:
            print(f"  {name}: ABSENT and --offline was asked for ({source}/{where})")
            continue
        print(f"  {name}: fetching from {source}")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("    huggingface_hub is not installed; cannot fetch. Copy the file by hand.")
            return 1
        repo = source.rsplit("/datasets/", 1)[-1] if "/datasets/" in source else source
        try:
            got = hf_hub_download(repo_id=repo, filename=where, repo_type="dataset")
        except Exception as error:  # the operator wants the reason, not a traceback
            print(f"    could not fetch: {type(error).__name__}: {error}")
            return 1
        target.mkdir(parents=True, exist_ok=True)
        part = destination.with_suffix(destination.suffix + ".part")
        shutil.copyfile(got, part)
        found = digest_of(part)
        if found != str(entry["sha256"]):
            part.unlink(missing_ok=True)
            print(f"    DIGEST MISMATCH: expected {entry['sha256']}, got {found}")
            return 1
        part.replace(destination)
        print(f"    verified {found[:16]}... and installed")
    return 0


def fetch_cutlass(root: Path, *, offline: bool) -> int:
    """The CUDA library, at a pinned tag rather than a moving branch.

    This is the clone the generator would otherwise do on first use, inside
    what a reader thinks is a compute call. Doing it here, once, at a tag, is
    the whole reason this script exists.
    """
    target = root / "cutlass"
    if target.exists():
        print("  cutlass: already present")
        return 0
    if offline:
        print(f"  cutlass: ABSENT and --offline was asked for ({CUTLASS_REPO} {CUTLASS_TAG})")
        return 0
    if shutil.which("git") is None:
        print("  cutlass: git is not on PATH; cannot clone")
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  cutlass: cloning {CUTLASS_REPO} at {CUTLASS_TAG}")
    found = subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            CUTLASS_TAG,
            "--single-branch",
            CUTLASS_REPO,
            str(target),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if found.returncode != 0:
        print(f"    clone failed: {found.stderr.strip()[:200]}")
        return 1
    print("    cloned")
    return 0


def build_venv(root: Path, *, offline: bool) -> int:
    """An interpreter that can import the vendored package.

    Built with ``--system-site-packages`` so it inherits this project's torch
    rather than installing a second copy of it, then given the three packages
    the project is short of and the vendored tree itself. The tree has to be
    *installed* rather than merely importable: its ``env.py`` reads its own
    distribution version at import, which is not available from a path entry.
    """
    target = root / "venv"
    python = target / "bin" / "python"
    if python.is_file():
        print("  venv: already built")
        return 0
    if offline:
        print("  venv: ABSENT and --offline was asked for")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    print("  venv: creating, inheriting this project's site-packages")
    made = subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if made.returncode != 0:
        print(f"    could not create: {made.stderr.strip()[:200]}")
        return 1
    print(f"  venv: installing {', '.join(MISSING_PACKAGES)}")
    step = subprocess.run(
        [str(python), "-m", "pip", "install", "-q", *MISSING_PACKAGES],
        capture_output=True,
        text=True,
        check=False,
    )
    if step.returncode != 0:
        print(f"    pip failed: {step.stderr.strip()[:300]}")
        return 1
    print("    installed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="report what is present and exit")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="do everything that needs no network, and name what is left",
    )
    parser.add_argument("--data-path", type=Path, help="override DATA_PATH/confrover")
    args = parser.parse_args(argv)

    root = Path(args.data_path) if args.data_path else data_root()
    if root is None:
        print("DATA_PATH is not set, so there is nowhere to install to.", file=sys.stderr)
        return 2
    if not (vendored() / "rbase").is_dir():
        print(f"no vendored package at {vendored() / 'rbase'}", file=sys.stderr)
        return 2
    if args.check:
        return report(root)

    root.mkdir(parents=True, exist_ok=True)
    print(f"installing into {root}")
    failed = 0
    failed |= fetch_weights(root, offline=args.offline)
    failed |= fetch_cutlass(root, offline=args.offline)
    failed |= build_venv(root, offline=args.offline)

    # The two that are the operator's own call, named rather than fetched. One
    # needs a licence decision and the other sends a sequence to a third party,
    # and neither is a thing a script should do quietly.
    print()
    print("Left for you, deliberately:")
    print("  openfold_params/  the folding parameters. Check their licence against your use")
    print("                    before copying them in; this script will not fetch weights")
    print("                    whose terms it cannot read.")
    print("  msa/              the alignment cache. Building one queries a public MSA server,")
    print("                    which sends the design's sequence off this machine. Use the")
    print("                    generator's own `rbase query_msa` when you have decided to.")
    print()
    if failed:
        return failed
    return report(root)


if __name__ == "__main__":
    raise SystemExit(main())
