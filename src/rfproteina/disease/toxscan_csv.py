"""Read the author service's wide-format ToxScan CSV, with its original endpoint names."""

from __future__ import annotations

import csv
import math
import re
import sys
from decimal import Decimal, InvalidOperation
from io import StringIO

from rfproteina.disease.toxscan_inputs import MAX_EXPORT_ROWS, ToxScanError, canonical
from rfproteina.disease.toxscan_validation import SOURCE_DISPLAY_ENDPOINTS

AD_HEADER = "AD Analysis (AD: Applicability Domain)"
SIMILARITY_HEADER = "Similarity Score in AD analysis"
CLASSIFICATIONS = (
    "Carcinogenicity", "Ames Mutagenicity", "Respiratory toxicity", "Eye irritation", "Eye corrosion",
    "Cardiotoxicity1", "Cardiotoxicity10", "Cardiotoxicity30", "Cardiotoxicity5",
    "CYP1A2", "CYP2C19", "CYP2C9", "CYP2D6", "CYP3A4",
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
)
REGRESSIONS = SOURCE_DISPLAY_ENDPOINTS
HEADERS = ("SMILES", AD_HEADER, SIMILARITY_HEADER, *REGRESSIONS, *CLASSIFICATIONS)
REGRESSION_SCALE = "unverified_source_display"
_NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
_DISPLAY = re.compile(rf"({_NUMBER})[ \t]*mg/kg")
_DOMAINS = {
    "In AD": "in_domain", "Out of AD": "out_of_domain", "Out AD": "out_of_domain",
    "Out-of-AD": "out_of_domain", "out-of-AD": "out_of_domain",
    "": "unreported",
}


def _numeric(value: str, *, similarity: bool = False) -> float:
    if len(value) > 60 or not re.fullmatch(_NUMBER, value):
        raise ToxScanError("toxscan_invalid_export_value")
    try:
        exact = Decimal(value)
    except InvalidOperation as error:
        raise ToxScanError("toxscan_invalid_export_value") from error
    number = float(exact)
    # Enforce the source-scale bounds before conversion can round an out-of-range value onto 0 or 1.
    if not math.isfinite(number) or (number == 0 and not exact.is_zero()) or (similarity and not 0 <= exact <= 1):
        raise ToxScanError("toxscan_invalid_export_value")
    return number


def _finite(value: object) -> bool:
    return type(value) in (int, float) and -sys.float_info.max <= value <= sys.float_info.max


def validate_fields(raw: dict) -> dict:
    domain = raw.get("applicability_domain")
    similarity = raw.get("similarity_score")
    results = raw.get("results")
    if (
        domain not in tuple(_DOMAINS.values())
        # The paper defines the mean of ten Avalon-fingerprint Tanimoto similarities.
        or (similarity is not None and (not _finite(similarity) or not 0 <= similarity <= 1))
        or not isinstance(results, dict) or results.keys() != {"classifications", "source_displays", "regression_scale"}
        or results["regression_scale"] != REGRESSION_SCALE
    ):
        raise ToxScanError("toxscan_invalid_export_value")
    calls, displays = results["classifications"], results["source_displays"]
    if (
        not isinstance(calls, dict) or calls.keys() != set(CLASSIFICATIONS)
        or any(value not in ("YES", "NO", None) for value in calls.values())
        or not isinstance(displays, dict) or displays.keys() != set(REGRESSIONS)
        or any(value is not None and not _finite(value) for value in displays.values())
        or not any(value is not None for value in (*calls.values(), *displays.values()))
    ):
        raise ToxScanError("toxscan_invalid_export_value")
    return {
        "applicability_domain": domain,
        "similarity_score": None if similarity is None else float(similarity),
        "results": {
            "classifications": {key: calls[key] for key in CLASSIFICATIONS},
            "source_displays": {key: None if displays[key] is None else float(displays[key]) for key in REGRESSIONS},
            "regression_scale": REGRESSION_SCALE,
        },
    }


def _fields(row: dict[str, str]) -> dict:
    try:
        domain = _DOMAINS[row[AD_HEADER]]
    except KeyError as error:
        raise ToxScanError("toxscan_invalid_domain") from error
    similarity = row[SIMILARITY_HEADER]
    displays = {}
    for endpoint in REGRESSIONS:
        source = row[endpoint]
        if source == "":
            displays[endpoint] = None
            continue
        value = _DISPLAY.fullmatch(source) if len(source) <= 70 else None
        if value is None:
            raise ToxScanError("toxscan_invalid_export_value")
        displays[endpoint] = _numeric(value.group(1))
    return validate_fields({
        "applicability_domain": domain,
        "similarity_score": _numeric(similarity, similarity=True) if similarity else None,
        "results": {
            "classifications": {key: row[key] or None for key in CLASSIFICATIONS},
            "source_displays": displays, "regression_scale": REGRESSION_SCALE,
        },
    })


def read_export(raw: bytes, structure: str) -> dict:
    """Match exactly one graph, independent of export order, and reject benchmark or mixed schemas."""
    try:
        reader = csv.reader(StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
        headers = next(reader, [])
        if len(headers) != len(HEADERS) or set(headers) != set(HEADERS):
            raise ToxScanError("toxscan_invalid_export_schema")
        selected = None
        for n, values in enumerate(reader, start=1):
            if n > MAX_EXPORT_ROWS or len(values) != len(headers):
                raise ToxScanError("toxscan_invalid_export")
            row = dict(zip(headers, values, strict=True))
            if canonical(row["SMILES"]) == structure:
                if selected is not None:
                    raise ToxScanError("toxscan_ambiguous_export")
                selected = row
    except (csv.Error, UnicodeError) as error:
        raise ToxScanError("toxscan_invalid_export") from error
    if selected is None:
        raise ToxScanError("toxscan_structure_not_found")
    return _fields(selected)
