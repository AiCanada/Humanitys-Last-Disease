#!/usr/bin/env python3
"""Decide whether CrossAbSense may be downloaded.

The five Apache-2.0 oracle heads are small. They load ESM-C 6B, ESM-C 300M
and ProtT5. Measured on 2026-11-17, that set is 39.38 GB, past the 25 GB cap,
so a normal run fetches nothing. ``--allow-over-cap`` is the explicit exception
and is the only way this script downloads.

    python scripts/install_crossabsense.py
    python scripts/install_crossabsense.py --allow-over-cap
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rfproteina.disease.crossabsense import (
    CAP_BYTES,
    ESMC_6B_BYTES,
    ESMC_300M_BYTES,
    ORACLE_FINAL_BYTES,
    PROTT5_BYTES,
    download_permitted,
    oracle_bytes,
    required_bytes,
)


def _gib(size: int) -> str:
    return f"{size / 1_000_000_000:.2f} GB"


def _root() -> Path:
    data = os.environ.get("DATA_PATH", "").strip()
    if not data:
        env_file = Path(__file__).resolve().parent.parent / ".env.local"
        for raw in env_file.read_text(encoding="utf-8").splitlines() if env_file.is_file() else []:
            line = raw.strip()
            if line.startswith("DATA_PATH="):
                data = line.split("=", 1)[1].strip().strip("'\"")
                break
    if not data:
        raise SystemExit("DATA_PATH is unset and .env.local does not name one.")
    os.environ["DATA_PATH"] = data
    return Path(data) / "crossabsense"


def _fetch(root: Path) -> None:
    """Download the final heads and the encoders the five axes name.

    Fold checkpoints are left on the Hub. ESM-C 6B is the safetensors set.
    ProtT5 is one ``pytorch_model.bin``, not the two duplicate copies.
    """
    from huggingface_hub import snapshot_download

    cache = root / "hf"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    models = root / "models"
    print(f"downloading final checkpoints into {models}")
    snapshot_download(
        repo_id="SimonCrouzet/CrossAbSense",
        local_dir=str(models),
        allow_patterns=["*_*/final.ckpt", "*_*/*.yaml", "*_*/*.txt", "*_*/*.json"],
    )
    print("downloading ESM-C 300M")
    snapshot_download(repo_id="EvolutionaryScale/esmc-300m-2024-12")
    print("downloading ProtT5 XL (one weight file)")
    snapshot_download(
        repo_id="Rostlab/prot_t5_xl_uniref50",
        allow_patterns=[
            "pytorch_model.bin",
            "config.json",
            "spiece.model",
            "special_tokens_map.json",
            "tokenizer_config.json",
        ],
    )
    print("downloading ESM-C 6B safetensors")
    snapshot_download(repo_id="biohub/esmc-6b-2024-12")
    print(f"ready  {root}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-over-cap",
        action="store_true",
        help="download even though the set is over 25 GB",
    )
    args = parser.parse_args(argv)
    print("CrossAbSense download check")
    print(f"cap        {_gib(CAP_BYTES)}")
    print(f"esmc_6b    {_gib(ESMC_6B_BYTES)}")
    print(f"esmc_300m  {_gib(ESMC_300M_BYTES)}")
    print(f"prott5     {_gib(PROTT5_BYTES)}")
    print(f"oracles    {_gib(oracle_bytes())}  ({len(ORACLE_FINAL_BYTES)} final checkpoints)")
    print(f"required   {_gib(required_bytes())}")
    if download_permitted():
        print("permitted  yes")
    elif not args.allow_over_cap:
        print("permitted  no")
        print("stopped    the full download reaches 25 GB, so no weights were fetched")
        return 2
    else:
        print("permitted  exception --allow-over-cap")
    _fetch(_root())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
