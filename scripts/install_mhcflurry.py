#!/usr/bin/env python3
"""Install MHCflurry beside this program and download the class I weights.

MHCflurry 2.3 needs pandas 3. This project's environment pins pandas 2, so the
package is installed into its own virtual environment under ``DATA_PATH`` and
never into the project environment. The presentation weights go to
``DATA_PATH/mhcflurry/models``. That is the directory the class I screen reads.

    python scripts/install_mhcflurry.py
    python scripts/install_mhcflurry.py --have

``DATA_PATH`` is taken from the environment, then from ``.env.local``. The
download is MHCflurry's own ``models_class1_presentation`` bundle at the
release named in ``rfproteina.disease.mhc_tables``.

Three short tolerated reference sequences are fetched from UniProt alongside
the weights, into ``DATA_PATH/mhcflurry/reference``. The class I and class II
axes measure a design's peptide density against those, and without them both
axes report ``reference_not_configured`` even on a machine that has the
predictor and the weights.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rfproteina.disease.limits import USER_AGENT
from rfproteina.disease.mhc_tables import (
    MHCFLURRY_DOWNLOADS_ENV,
    MODELS_BUNDLE,
    MODELS_RELEASE,
    REFERENCE_ACCESSIONS,
    ensure_allele_file,
    install_root,
    reference_sequences,
    weights_present,
)

#: UniProt's own FASTA endpoint for one accession.
REFERENCE_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"


def _unquote(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _data_path_from_file() -> str:
    """The ``DATA_PATH`` assignment in ``.env.local``, or an empty string."""
    env_file = Path(__file__).resolve().parent.parent / ".env.local"
    if not env_file.is_file():
        return ""
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "DATA_PATH":
            return _unquote(value)
    return ""


def _ensure_data_path() -> None:
    if os.environ.get("DATA_PATH", "").strip():
        return
    found = _data_path_from_file()
    if not found:
        raise SystemExit("DATA_PATH is unset and .env.local does not name one.")
    os.environ["DATA_PATH"] = found


def _venv_python(venv: Path) -> Path:
    windows = venv / "Scripts" / "python.exe"
    if windows.is_file():
        return windows
    posix = venv / "bin" / "python"
    if posix.is_file():
        return posix
    raise SystemExit(f"No Python interpreter in {venv}.")


def _downloads_command(python: Path) -> Path:
    directory = python.parent
    for name in ("mhcflurry-downloads.exe", "mhcflurry-downloads"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise SystemExit(f"mhcflurry-downloads is not installed beside {python}.")


def fetch_reference(root: Path) -> Path:
    """Fetch the three tolerated sequences the MHC axes measure against.

    Without them the class I and class II axes report
    ``reference_not_configured``: the predictor runs and yields a peptide
    density, but there is nothing tolerated to compare that density with, so
    the axis cannot be called either way. A screen on a machine holding the
    predictor and the weights still reported both axes untested for want of
    these three short files.

    The accessions are serum albumin, the IgG1 constant region and
    ubiquitin-B: abundant human proteins a healthy immune system meets
    constantly, which is what makes their peptide density the tolerated
    baseline.

    A file already on disk is kept, so this is safe to re-run. Each reply is
    checked for a FASTA header rather than for a size, because an HTML error
    page is larger than a short sequence and would otherwise be written and
    then silently parsed as nothing.
    """
    folder = root / "reference"
    folder.mkdir(parents=True, exist_ok=True)
    for accession in REFERENCE_ACCESSIONS:
        target = folder / f"{accession}.fasta"
        if target.is_file() and target.stat().st_size > 0:
            print(f"keeping       {target}")
            continue
        url = REFERENCE_URL.format(accession=accession)
        print(f"fetching      {url}")
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=60) as response:
            payload = response.read()
        text = payload.decode("utf-8", errors="replace").lstrip()
        if not text.startswith(">") or "\n" not in text.strip():
            raise SystemExit(f"{accession}: the reply is not a FASTA record, so nothing was written.")
        target.write_bytes(payload)
        print(f"wrote         {target}  ({len(payload)} bytes)")
    return folder


def _have(root: Path) -> int:
    python = root / "venv" / "bin" / "python"
    windows = root / "venv" / "Scripts" / "python.exe"
    present = weights_present(root / "models")
    status, found = reference_sequences()
    print(f"install root  {root}")
    print(f"interpreter   {python if python.is_file() else windows}")
    print(f"weights       {'present' if present else 'absent'}  ({root / 'models'})")
    print(f"reference     {status}  {len(found)}/{len(REFERENCE_ACCESSIONS)}  ({root / 'reference'})")
    if status != "ready":
        missing = [a for a in REFERENCE_ACCESSIONS if a not in found]
        print(f"              missing {', '.join(missing)} -- the MHC axes stay untested without these")
    ready = present and (python.is_file() or windows.is_file()) and status == "ready"
    return 0 if ready else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--have", action="store_true", help="report the install and download, then stop")
    args = parser.parse_args(argv)
    _ensure_data_path()
    root = install_root()
    if args.have:
        return _have(root)

    venv = root / "venv"
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    print(f"install root  {root}")
    if not (venv / "pyvenv.cfg").is_file():
        print(f"creating virtual environment  {venv}")
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    python = _venv_python(venv)
    print(f"installing mhcflurry>=2.3,<2.4 into {python}")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--upgrade", "mhcflurry>=2.3,<2.4"],
        check=True,
    )
    command = _downloads_command(python)
    environment = os.environ.copy()
    environment[MHCFLURRY_DOWNLOADS_ENV] = str(models)
    print(f"fetching {MODELS_BUNDLE} release {MODELS_RELEASE} into {models}")
    subprocess.run(
        [str(command), "fetch", MODELS_BUNDLE, "--release", MODELS_RELEASE],
        check=True,
        env=environment,
    )
    if not weights_present(models):
        raise SystemExit(f"Fetch finished but {MODELS_BUNDLE} is not under {models}.")
    alleles = ensure_allele_file()
    reference = fetch_reference(root)
    status, found = reference_sequences()
    print(f"ready  {python}")
    print(f"ready  {models}")
    print(f"ready  {alleles}")
    print(f"{'ready' if status == 'ready' else status}  {reference}  ({len(found)}/{len(REFERENCE_ACCESSIONS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
