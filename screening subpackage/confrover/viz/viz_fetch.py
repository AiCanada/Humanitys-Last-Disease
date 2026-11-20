# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Accept a source that is not on this machine yet.

The viewer's whole point is that someone else can run it, and the ensembles it
draws are published on the Hub rather than sitting in the checkout. So a source
may be given three ways, and they are tried in that order:

    /a/local/path                  used as-is
    hf:CASP17 E2460 Submit/x.tgz   that path inside the project's own dataset
    hf://datasets/OWNER/NAME/x     any file in any dataset

A local path always wins, and is never checked against the Hub -- someone
working offline with their own output should not need a network round trip to
be told what they already know.

``huggingface_hub`` is imported inside the function that needs it, not at module
scope, so the local-file path -- which is the common one -- keeps this tool's
only hard dependency at numpy. The environment variable that points at a local
copy of the dataset is the same one the rest of the repo uses, ``RBASE_DATA``,
so configuring it once serves both.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Kept in step with ``scripts/data_paths.py`` by a test, because the two are
#: deliberately not allowed to import each other: that module is free to grow
#: dependencies this one is not.
DATASET_REPO = "AICanada/H.U.M.A.N"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_REPO}"
ENV_ROOT = "RBASE_DATA"

HF_PREFIX = "hf://"
SHORT_PREFIX = "hf:"


def parse(source: str) -> tuple[str, str] | None:
    """``(repo_id, path_in_repo)`` for a Hub source, or None for a local one.

    ``hf://datasets/o/n/p`` and the bare ``hf://o/n/p`` both name a dataset;
    ``hf:p`` is shorthand for this project's dataset. Nothing else is a Hub
    source -- in particular a Windows path is never mistaken for one, since
    ``C:\\x`` has no ``hf:`` prefix.
    """
    s = str(source)
    if s.startswith(HF_PREFIX):
        rest = s[len(HF_PREFIX):]
        for lead in ("datasets/", "dataset/"):
            if rest.startswith(lead):
                rest = rest[len(lead):]
                break
        parts = rest.split("/")
        if len(parts) < 3:
            raise SystemExit(
                f"{source!r}: an hf:// source needs owner/name/path, "
                f"e.g. hf://datasets/{DATASET_REPO}/catalog.json")
        return "/".join(parts[:2]), "/".join(parts[2:])
    if s.startswith(SHORT_PREFIX):
        rel = s[len(SHORT_PREFIX):].lstrip("/")
        if not rel:
            raise SystemExit(f"{source!r}: hf: needs a path inside {DATASET_REPO}")
        return DATASET_REPO, rel
    return None


def _local_first(repo_id: str, rel: str) -> Path | None:
    """A configured local copy beats the download, for the project's own data."""
    root = os.environ.get(ENV_ROOT, "").strip()
    if not root or repo_id != DATASET_REPO:
        return None
    candidate = Path(root).expanduser() / rel
    return candidate if candidate.exists() else None


def fetch(source, *, verbose: bool = True) -> Path:
    """Resolve ``source`` to a path on disk, downloading it only if it must."""
    spec = parse(source)
    if spec is None:
        return Path(source)

    repo_id, rel = spec
    local = _local_first(repo_id, rel)
    if local is not None:
        if verbose:
            print(f"  {source} -> {local} (from ${ENV_ROOT})")
        return local

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            f"{source!r} needs huggingface_hub to download.\n"
            f"  pip install huggingface_hub, or fetch the file yourself from\n"
            f"  https://huggingface.co/datasets/{repo_id} and pass the local path."
        ) from None

    if verbose:
        print(f"  fetching {rel} from {repo_id} ...")
    try:
        got = Path(hf_hub_download(repo_id, rel, repo_type="dataset"))
    except Exception as exc:
        raise SystemExit(
            f"could not fetch {rel!r} from {repo_id}:\n"
            f"  {type(exc).__name__}: {str(exc).splitlines()[0][:200]}\n"
            f"  a private dataset needs `hf auth login` first; a missing file may\n"
            f"  be documented but not uploaded -- check "
            f"https://huggingface.co/datasets/{repo_id}/tree/main"
        ) from None
    if verbose:
        print(f"  -> {got}")
    return got
