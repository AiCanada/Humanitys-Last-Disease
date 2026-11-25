"""Quantify paired H-DAB fields from a nominated frozen tissue or chip imaging experiment."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import batched
from pathlib import Path

from rfproteina.disease import cellprofiler_images as images
from rfproteina.disease import cellprofiler_runtime as runtime
from rfproteina.disease import ligand_inputs
from rfproteina.disease.corpus_provenance import declare

METHOD_ID = "cellprofiler"
MANIFEST = "cellprofiler.json"
ENDPOINT = "regional_dab_stain_intensity"
MAX_FIELDS = 1024
MAX_COMPARISONS = 512
MAX_PANEL_BYTES = 4 * 1024**3
MAX_PANEL_PIXELS = 1024**3
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
_LABEL = re.compile(r"[\w][\w .(),+-]{0,95}")
_CONTEXT = ("material", "specimen_id", "tissue", "region", "assay_batch", "acquisition_id")
_FIELD_KEYS = {*_CONTEXT, "role", "reagent_id", "image", "mask"}
_NEGATIVE = {"isotype_control", "no_primary_control"}
_ROLES = {"candidate", "positive_control", *_NEGATIVE}


class StainError(ValueError):
    """A nomination cannot support the requested regional comparison."""


def _id(value: object) -> bool:
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def _unique(pairs: list) -> dict:
    values = dict(pairs)
    if len(values) != len(pairs):
        raise ValueError("duplicate JSON key")
    return values


def read_manifest(directory: Path) -> tuple[bytes | None, str | None]:
    """Read a stable manifest for both the panel and the finished-run input inventory."""
    try:
        return ligand_inputs.read_regular(directory / MANIFEST, 1024 * 1024, "cellprofiler_manifest", ValueError), None
    except ValueError as exc:
        if str(exc) == "cellprofiler_manifest_missing":
            return None, "cellprofiler_manifest_missing"
        if str(exc) == "cellprofiler_manifest_size_limit":
            return None, "cellprofiler_budget_exceeded"
        return None, "cellprofiler_invalid_manifest"


@dataclass(frozen=True)
class Panel:
    digest: str
    fields: dict
    comparisons: list[dict]

    @classmethod
    def read(cls, directory: Path, candidate: str | None) -> Panel:
        raw, blocker = read_manifest(directory)
        if raw is None:
            raise StainError(blocker)
        try:
            doc = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique)
        except (ValueError, RecursionError) as exc:
            raise StainError("cellprofiler_invalid_manifest") from exc
        if not isinstance(doc, dict) or doc.keys() != {"stains", "fields", "comparisons"}:
            raise StainError("cellprofiler_invalid_manifest")
        fields, rows = doc["fields"], doc["comparisons"]
        if doc["stains"] != "hematoxylin_dab" or not isinstance(fields, dict) or not isinstance(rows, list):
            raise StainError("cellprofiler_invalid_manifest")
        if len(fields) > MAX_FIELDS or len(rows) > MAX_COMPARISONS:
            raise StainError("cellprofiler_budget_exceeded")
        if not rows or not fields or not all(_id(key) for key in fields):
            raise StainError("cellprofiler_invalid_manifest")
        if any(
            not isinstance(row, dict) or not _id(row.get("candidate_id")) or not _id(row.get("comparison_id"))
            for row in rows
        ):
            raise StainError("cellprofiler_invalid_manifest")
        if candidate is not None and not any(row["candidate_id"] == candidate for row in rows):
            raise StainError("cellprofiler_candidate_not_nominated")
        return cls(hashlib.sha256(raw).hexdigest(), fields, rows)


def _field(raw: object) -> dict:
    if not isinstance(raw, dict) or not isinstance(raw.get("role"), str) or raw["role"] not in _ROLES:
        raise StainError("cellprofiler_invalid_pair")
    keys = _FIELD_KEYS | {"candidate_id"} if raw["role"] == "candidate" else _FIELD_KEYS
    if raw.keys() != keys or not all(
        _id(raw[k]) for k in ("specimen_id", "assay_batch", "acquisition_id", "reagent_id")
    ):
        raise StainError("cellprofiler_invalid_pair")
    if "candidate_id" in raw and not _id(raw["candidate_id"]):
        raise StainError("cellprofiler_invalid_pair")
    if not all(
        isinstance(raw[k], str) and _LABEL.fullmatch(raw[k]) and raw[k] == raw[k].strip() for k in ("tissue", "region")
    ):
        raise StainError("cellprofiler_invalid_pair")
    if not all(isinstance(raw[k], str) and raw[k] for k in ("image", "mask")):
        raise StainError("cellprofiler_invalid_pair")
    if raw["material"] not in ("human_frozen_tissue", "human_chip") and not (
        raw["role"] == "positive_control" and raw["material"] == "assay_control"
    ):
        raise StainError("cellprofiler_invalid_pair")
    return raw


def _matched_fields(panel: Panel, row: dict) -> dict[str, dict]:
    required = {"comparison_id", "candidate_id", "test", "negative"}
    if row.keys() not in (required, required | {"positive"}):
        raise StainError("cellprofiler_invalid_pair")
    roles = ["test", "negative"] + (["positive"] if "positive" in row else [])
    if not all(_id(row[k]) and row[k] in panel.fields for k in roles):
        raise StainError("cellprofiler_invalid_pair")
    fields = {k: _field(panel.fields[row[k]]) for k in roles}
    test, negative = fields["test"], fields["negative"]
    if test["role"] != "candidate" or test["candidate_id"] != row["candidate_id"] or negative["role"] not in _NEGATIVE:
        raise StainError("cellprofiler_control_mismatch")
    if any(test[k] != negative[k] for k in _CONTEXT) or test["reagent_id"] == negative["reagent_id"]:
        raise StainError("cellprofiler_control_mismatch")
    positive = fields.get("positive")
    if positive and (
        positive["role"] != "positive_control" or any(test[k] != positive[k] for k in ("assay_batch", "acquisition_id"))
    ):
        raise StainError("cellprofiler_control_mismatch")
    return fields


def _plan(directory: Path, panel: Panel) -> tuple[list[dict], dict[str, dict]]:
    names = Counter((r["candidate_id"], r["comparison_id"]) for r in panel.comparisons)
    pairs = Counter(
        (r["candidate_id"], r["test"], r["negative"])
        for r in panel.comparisons
        if _id(r.get("test")) and _id(r.get("negative"))
    )
    planned, inputs = [], {}
    checked: dict[str, str | None] = {}
    source_sizes = {}
    for row in panel.comparisons:
        key = row["candidate_id"], row["comparison_id"]
        result = {"method": METHOD_ID, "candidate_id": key[0], "comparison_id": key[1], "status": "untested"}
        try:
            fields = _matched_fields(panel, row)
            if names[key] > 1 or pairs[row["candidate_id"], row["test"], row["negative"]] > 1:
                raise StainError("cellprofiler_duplicate_pair")
            if len({v["image"] for v in fields.values()}) != len(fields):
                raise StainError("cellprofiler_image_conflict")
            for role, field in fields.items():
                field_id = row[role]
                if field_id not in checked:
                    try:
                        source_sizes[field_id] = images.check_region_files(directory, field["image"], field["mask"])
                        checked[field_id] = None
                    except images.ImageError as exc:
                        checked[field_id] = str(exc)
                if checked[field_id]:
                    raise StainError(checked[field_id])
            result["fields"] = {role: row[role] for role in fields}
            inputs.update({row[role]: v for role, v in fields.items()})
        except StainError as exc:
            result["blocker"] = str(exc)
        planned.append(result)
    if sum(source_sizes[name] for name in inputs) > MAX_PANEL_BYTES:
        raise StainError("cellprofiler_budget_exceeded")
    return planned, inputs


def readiness(directory: Path, candidate: str | None = None) -> tuple[bool, bool, str]:
    """Check nominated roles, matching metadata and local files. Full decoding is part of execution."""
    try:
        rows, _ = _plan(directory, Panel.read(directory, candidate))
        if candidate is not None:
            rows = [row for row in rows if row["candidate_id"] == candidate]
    except StainError as exc:
        return False, False, f"No eligible CellProfiler imaging panel ({exc})."
    ready = sum("fields" in row for row in rows)
    reasons = ", ".join(sorted({row["blocker"] for row in rows if "blocker" in row}))
    suffix = f" ({reasons})" if reasons else ""
    return bool(ready), ready == len(rows), f"{ready} of {len(rows)} matched regional comparisons prepared{suffix}."


def _measure_fields(directory: Path, inputs: dict[str, dict], selected: set[str]) -> tuple[dict, dict]:
    """Retain scalar results and digests only. Defer use of a field until all identities have been checked."""
    measured, errors = {}, {}
    identities: dict[str, dict[tuple, set[str]]] = defaultdict(lambda: defaultdict(set))
    regions_by_pixels: dict[tuple, set[str]] = defaultdict(set)
    total_bytes = total_pixels = 0
    for group in batched(inputs, runtime.MAX_REGIONS):
        regions = {}
        for name in group:
            field = inputs[name]
            try:
                region = images.read_region(directory, field["image"], field["mask"])
            except images.ImageError as exc:
                errors[name] = str(exc)
                continue
            total_bytes += region.source_byte_count
            total_pixels += region.width * region.height
            if total_bytes > MAX_PANEL_BYTES or total_pixels > MAX_PANEL_PIXELS:
                return {}, dict.fromkeys(selected, "cellprofiler_budget_exceeded")
            if name in selected:
                regions[name] = region
            # One digital field may have multiple annotated regions. Its specimen, role
            # and reagent attribution must agree across the entire nominated panel.
            identity = tuple(field.get(k) for k in (*_CONTEXT, "role", "reagent_id", "candidate_id") if k != "region")
            identities[region.pixel_sha256][identity].add(name)
            mask_pixels = hashlib.sha256(memoryview(region.mask).cast("B")).hexdigest()
            regions_by_pixels[region.pixel_sha256, mask_pixels].add(name)
        if not regions:
            continue
        try:
            stats = runtime.measure_regions([(r.rgb, r.mask) for r in regions.values()])
            if len(stats) != len(regions):
                raise StainError("cellprofiler_measurement_failed")
        except (runtime.CellProfilerError, StainError) as exc:
            errors.update(dict.fromkeys(regions, str(exc)))
            continue
        for (name, region), values in zip(regions.items(), stats, strict=True):
            if (
                not isinstance(values, dict)
                or values.get("roi_pixels") != region.roi_pixels
                or not all(
                    type(values.get(k)) in (float, int) and 0 <= values[k] <= 1 and math.isfinite(values[k])
                    for k in ("mean_dab", "median_dab")
                )
            ):
                errors[name] = "cellprofiler_measurement_failed"
                continue
            field = inputs[name]
            measured[name] = {
                "method": METHOD_ID,
                "field_id": name,
                **{k: field[k] for k in (*_CONTEXT, "role", "reagent_id")},
                "width": region.width,
                "height": region.height,
                "roi_pixels": region.roi_pixels,
                "image_sha256": region.image_sha256,
                "mask_sha256": region.mask_sha256,
                "mean_dab": values["mean_dab"],
                "median_dab": values["median_dab"],
            }
    for assignments in identities.values():
        if len(assignments) > 1:
            for names in assignments.values():
                errors.update(dict.fromkeys(names, "cellprofiler_image_conflict"))
    for names in regions_by_pixels.values():
        if len(names) > 1:
            errors.update(dict.fromkeys(names, "cellprofiler_image_conflict"))
    return measured, errors


def measure(directory: Path, candidate: str | None = None) -> dict:
    report = {
        "method": METHOD_ID,
        "endpoint": ENDPOINT,
        "provider_version": runtime.VERSION,
        "provider_image_digest": runtime.DIGEST,
        "scope": "candidate" if candidate is not None else "run",
        "status": "untested",
        "attempted": 0,
        "reported": 0,
        "comparisons": [],
        "stains": "hematoxylin_dab",
        "intensity_unit": "CellProfiler normalized DAB intensity (arbitrary units)",
        "interpretation": "Regional stain contrast for pathology review. "
        "Specific binding and organ injury require separate evidence.",
        "corpus_provenance": declare(METHOD_ID, against=ENDPOINT),
    }
    try:
        panel = Panel.read(directory, candidate)
        report["manifest_sha256"] = panel.digest
        rows, inputs = _plan(directory, panel)
        if candidate is not None:
            rows = [row for row in rows if row["candidate_id"] == candidate]
    except StainError as exc:
        return {**report, "blocker": str(exc)}
    selected = {name for row in rows for name in row.get("fields", {}).values()}
    measured, errors = _measure_fields(directory, inputs, selected) if selected else ({}, {})
    for row in rows:
        if "fields" not in row:
            continue
        fields = row.pop("fields")
        bad = {role: errors[name] for role, name in fields.items() if name in errors}
        if bad:
            row.update(blocker=next(iter(bad.values())), field_ids=fields, field_blockers=bad)
            continue
        row.update({role: measured[name] for role, name in fields.items()})
        row.update(
            status="reported",
            positive_control_status="measured" if "positive" in row else "not_supplied",
            test_minus_negative_mean_dab=row["test"]["mean_dab"] - row["negative"]["mean_dab"],
        )
    count = sum(row["status"] == "reported" for row in rows)
    report.update(attempted=len(rows), reported=count, comparisons=rows)
    if count:
        report["status"] = "reported" if count == len(rows) else "partial"
    elif len(blockers := {row["blocker"] for row in rows}) == 1:
        report["blocker"] = blockers.pop()
    return report
