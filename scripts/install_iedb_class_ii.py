#!/usr/bin/env python3
"""Fetch IEDB's class II binding data for the DR panel into ``DATA_PATH/iedb``.

The class II axis has no predictor because every option is licence-blocked
(``immunogenicity.CLASS_II_OPTIONS``). IEDB's measurements are CC BY, so the
corpus to train one is available even where the trained tools are not. This
fetches it and ``rfproteina.disease.class_ii`` trains on it.

**Not the 10 GB export.** ``tables.SOURCES`` carries
``mhc_ligand_full_tsv.zip`` and says what it is: a 10.26 GB census of every
organism IEDB holds, under a two-row header, sent with no Content-Length. None
of that is needed here. The query API answers the same question filtered --
class II, these seven alleles, a quantitative measurement -- and the answer is
about 20,000 rows over a few MB.

    python scripts/install_iedb_class_ii.py
    python scripts/install_iedb_class_ii.py --have

IEDB asks that its data be cited. The licence and the per-row assay and
reference ids are written into the table so a derived model can say what it
learned from.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfproteina.disease.immunogenicity import DR_PANEL

#: The filtered endpoint. ``mhc_export`` carries the assay detail that
#: ``mhc_search`` summarises away: the measurement, its units, its inequality
#: and the method behind it.
_ENDPOINT = "https://query-api.iedb.org/mhc_export"
#: IEDB writes the panel alleles with an ``HLA-`` prefix; ``DR_PANEL`` does not.
_PREFIX = "HLA-"
#: PostgREST pages. 5,000 is well inside its limit and keeps the request count low.
_PAGE = 5000
#: An offset without an order is the silent paging bug: rows shift between
#: requests, so a page repeats some records and never shows others. IEDB
#: refuses the combination outright -- "offset parameter without an order
#: parameter" -- which is the right call. ``assay_id`` is unique, so ordering
#: on it makes the paging total.
_ORDER = "assay_id.asc"
_TIMEOUT = 180
#: What is read from each record, and what it is called in the written table.
_FIELDS = {
    "mhc_restriction__name": "allele",
    "epitope__name": "peptide",
    "assay__quantitative_measurement": "measurement",
    "assay__units": "units",
    "assay__measurement_inequality": "inequality",
    "assay__method": "assay_method",
    "assay__qualitative_measurement": "qualitative",
    "assay_id": "assay_id",
    "reference__pmid": "pubmed_id",
}
_COLUMNS = (
    "allele",
    "peptide",
    "measurement",
    "units",
    "inequality",
    "assay_method",
    "qualitative",
    "assay_id",
    "pubmed_id",
)
_TABLE = "class_ii_dr_panel.tsv"
#: Verified at the source rather than assumed. IEDB states this in the JSON-LD
#: metadata of its own export page; the prose page at /terms returns 404, so
#: the machine-readable declaration is the statement that exists.
LICENCE = (
    "IEDB, CC BY 4.0 -- declared as dcterms:license in the JSON-LD metadata of "
    "https://www.iedb.org/database_export_v3.php, read 2026-11-17. Attribute IEDB and the per-row reference."
)


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


def _destination() -> Path:
    return Path(os.environ["DATA_PATH"]) / "iedb"


def _get(url: str) -> list[dict[str, object]]:
    request = urllib.request.Request(url, headers={"User-Agent": "rfproteina-screening/1.0"})
    # The URL is built from _ENDPOINT and the panel constant, not from input.
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_allele(allele: str) -> list[dict[str, object]]:
    """Every class II record for one allele that carries a quantitative value."""
    name = urllib.parse.quote(f"{_PREFIX}{allele}", safe="")
    select = urllib.parse.quote(",".join(_FIELDS), safe=",")
    out: list[dict[str, object]] = []
    offset = 0
    while True:
        url = (
            f"{_ENDPOINT}?mhc_restriction__name=eq.{name}"
            f"&assay__quantitative_measurement=not.is.null"
            f"&select={select}&order={_ORDER}&limit={_PAGE}&offset={offset}"
        )
        page = _get(url)
        out.extend(page)
        if len(page) < _PAGE:
            return out
        offset += _PAGE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--have", action="store_true", help="report the table, then stop")
    args = parser.parse_args(argv)
    _ensure_data_path()
    destination = _destination()
    path = destination / _TABLE

    if args.have:
        print(f"table   {path}")
        if not path.is_file():
            print("rows    absent")
            return 1
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        print(f"rows    {len(rows):,}")
        per_allele: dict[str, int] = {}
        for row in rows:
            per_allele[row["allele"]] = per_allele.get(row["allele"], 0) + 1
        for allele in DR_PANEL:
            print(f"  {per_allele.get(allele, 0):>6}  {allele}")
        return 0 if all(per_allele.get(allele) for allele in DR_PANEL) else 1

    destination.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, str]] = []
    for allele in DR_PANEL:
        print(f"fetching {_PREFIX}{allele}")
        records = _fetch_allele(allele)
        kept = 0
        for record in records:
            row = {name: record.get(field) for field, name in _FIELDS.items()}
            peptide = str(row.get("peptide") or "").strip().upper()
            if not peptide:
                continue
            # The allele is written without the HLA- prefix, which is how
            # DR_PANEL and the screen's own tables spell it.
            written.append(
                {
                    "allele": allele,
                    "peptide": peptide,
                    "measurement": "" if row.get("measurement") is None else str(row["measurement"]),
                    "units": str(row.get("units") or "").strip(),
                    "inequality": str(row.get("inequality") or "").strip(),
                    "assay_method": str(row.get("assay_method") or "").strip(),
                    "qualitative": str(row.get("qualitative") or "").strip(),
                    "assay_id": str(row.get("assay_id") or "").strip(),
                    "pubmed_id": str(row.get("pubmed_id") or "").strip(),
                }
            )
            kept += 1
        print(f"  {kept:>6} records")

    if not written:
        raise SystemExit("IEDB returned no class II records for the DR panel.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_COLUMNS), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(written)
    (destination / "class_ii_dr_panel.LICENCE.txt").write_text(LICENCE + "\n", encoding="utf-8")
    print(f"ready  {path}  ({len(written):,} rows)")
    print(f"       {LICENCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
