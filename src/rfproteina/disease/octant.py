"""Octant CYP3A4 inhibition, loaded once and stored in nanomolar.

The OpenADMET release
``openadmet/Octant_CYP_inhibition_reactivity_blog_release`` is CC BY 4.0
and 2,269,722 bytes. ``inhibition.tsv`` is one row per compound and its
potency column is ``CYP3A4_pIC50``, defined by the dataset card as
-log10(IC50 in molar). :mod:`rfproteina.disease.toxicology` reads only
nanomolar concentrations, so the conversion ``10 ** (9 - pIC50)`` happens
here, before a row is allowed into ``toxicology.tsv``.

CYP2J2 is in ``reactivity.tsv`` of the same release. It is not one of
:data:`rfproteina.disease.combinations.LIGAND_AXES`, and a percent remaining
is not an IC50, so those rows are never written as a panel reading.

The download is the whole release, kept under ``DATA_PATH/octant``. What the
panel lookup reads is the subset this module writes, whose affinity column is
already nanomolar. The raw table is refused at that lookup: a pIC50 of 6 left
in an nM column is a 6 nM hit, and the number was 1000 nM.
"""

from __future__ import annotations

import csv
import math
import os
import statistics
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from urllib.request import Request, urlopen

from rfproteina.disease.combinations import LIGAND_AXES, panel_cut, panel_cut_source
from rfproteina.disease.ligand_safety import (
    BINDINGDB_PANEL_ACCESSIONS,
    AxisMeasurement,
    LigandUnreadableError,
    PanelMeasurements,
    SafetyBackendUnavailableError,
    _inchikey,
    to_toxicology_rows,
)
from rfproteina.disease.limits import USER_AGENT
from rfproteina.disease.screening_tools import data_dir
from rfproteina.disease.tables import PANEL_SUBSET_COLUMNS, SubsetResult

#: Dataset-card licence tag ``license: cc-by-4.0``. The release is 2,269,722 bytes.
OCTANT_LICENCE = "CC BY 4.0"
OCTANT_RELEASE = "https://huggingface.co/datasets/openadmet/Octant_CYP_inhibition_reactivity_blog_release"
OCTANT_RELEASE_BYTES = 2_269_722
_DOWNLOAD = f"{OCTANT_RELEASE}/resolve/main/"
#: Every file in the release. Loaded once; only ``inhibition.tsv`` feeds the panel.
OCTANT_FILES: tuple[str, ...] = (
    "inhibition.tsv",
    "inhibition_wells.tsv",
    "reactivity.tsv",
    "reactivity_wells.tsv",
    "will_it_fly_in_mass_spec.tsv",
)
INHIBITION_FILE = "inhibition.tsv"
SUBSET_FILE = "octant_cyp3a4_subset.tsv"
SUBSET_REQUIRED_COLUMNS = ("inchikey", "panel_axis", "affinity_nanomolar", "accession", "assay", "relation", "curation")

OCTANT_CURATION = "OpenADMET Octant CYP inhibition (CC BY 4.0)"
OCTANT_METHOD = "octant_cyp3a4_pic50_to_nm_v1"
OCTANT_AXIS = "cyp3a4"
#: UniProt accession of human CYP3A4, the same one the other panel sources use.
OCTANT_ACCESSION = BINDINGDB_PANEL_ACCESSIONS[OCTANT_AXIS]

_PIC50 = "CYP3A4_pIC50"
_SMILES = "standardized_smiles"
_BATCH = "ocnt_batch"
_ACTIVITY = "activity_status"
_DIRECTION = "direction"
#: Present on ``reactivity.tsv``. Not a panel axis, and not an IC50.
_ENZYME = "enzyme"
_NOT_PANEL = "CYP2J2"
_NEEDED = (_BATCH, _PIC50, _SMILES)

_TOXICOLOGY_COLUMNS = ("axis", "protein", "nm", "unit", "relation", "source", "method", "inchikey",
                       "standard_type", "assay_id")


def pic50_to_nm(pic50: float) -> float:
    """IC50 in nM from a pIC50 defined as -log10(IC50 in molar).

    pIC50 6 is 10**(-6) mol/L, which is 1 micromol/L, which is 1000 nM.
    pIC50 9 is 1 nM. The toxicology cut is 10 micromol/L, which is pIC50 5.
    """
    if not math.isfinite(pic50):
        raise ValueError("pIC50 is not finite")
    try:
        value = 10.0 ** (9.0 - pic50)
    except OverflowError as error:
        raise ValueError("pIC50 outside representable concentration range") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError("pIC50 outside representable concentration range")
    return value


def octant_directory() -> Path:
    """``DATA_PATH/octant``, or ``<checkout>/data/octant`` when DATA_PATH is unset.

    The checkout is located by searching upward for ``.project-root`` rather
    than by a fixed parent count, which would resolve silently to the wrong
    directory if this module changed depth.
    """
    return data_dir("octant")


def ensure_release(destination: Path, *, timeout_s: int = 120) -> dict[str, int]:
    """Download each release file once. A non-empty file already there is kept.

    Returns the byte size of each file after the call. A body that is not a
    TSV (an HTML error page, for one) is discarded rather than stored.
    """
    destination.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    for name in OCTANT_FILES:
        path = destination / name
        if path.is_file() and path.stat().st_size > 0 and _looks_like_tsv(path):
            sizes[name] = path.stat().st_size
            continue
        payload = _download(f"{_DOWNLOAD}{name}", timeout_s)
        if not payload or payload[:1] == b"<" or b"\t" not in payload.splitlines()[0]:
            raise OSError(f"{name} did not come back as a TSV")
        partial = path.with_suffix(path.suffix + ".part")
        partial.write_bytes(payload)
        partial.replace(path)
        sizes[name] = len(payload)
    return sizes


def octant_cyp3a4_subset(inhibition: Path, destination: Path) -> SubsetResult:
    """Write the CYP3A4 rows of ``inhibition.tsv`` with the affinity already in nM.

    A row with no finite ``CYP3A4_pIC50`` is counted and skipped. The release
    leaves that cell empty precisely when the dose-response fit did not pass,
    so there is no concentration to invent. ``CYP2J2`` on the same row, or an
    ``enzyme`` cell naming it, is counted under ``cyp2j2_not_a_panel_axis`` and
    does not become a row: it is not a panel protein, and the reactivity table
    does not carry an IC50.
    """
    if not inhibition.is_file():
        return SubsetResult(None, 0, 0, (), 0, "inhibition_not_found")
    try:
        kept, scanned, skipped, ligands = _convert(inhibition)
    except SafetyBackendUnavailableError as error:
        return SubsetResult(None, 0, 0, (), 0, str(error))
    except (OSError, UnicodeError, csv.Error) as error:
        return SubsetResult(None, 0, 0, (), 0, f"inhibition_unreadable:{type(error).__name__}")
    if not kept:
        return SubsetResult(None, 0, 0, (), scanned, "no_cyp3a4_pic50", skipped=dict(skipped))
    destination.mkdir(parents=True, exist_ok=True)
    out = destination / SUBSET_FILE
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(PANEL_SUBSET_COLUMNS)
        writer.writerows(kept)
    return SubsetResult(
        out,
        len(kept),
        len(ligands),
        (OCTANT_AXIS,),
        scanned,
        None,
        missing=tuple(axis for axis in LIGAND_AXES if axis != OCTANT_AXIS),
        skipped=dict(skipped),
        curation={OCTANT_CURATION: len(kept)},
    )


def measured_nm_from_octant(smiles: str, subset: Path) -> PanelMeasurements:
    """This ligand's CYP3A4 reading from a subset whose affinity is already nM.

    Raises :class:`SafetyBackendUnavailableError` with
    ``octant_pic50_not_converted`` when handed the raw inhibition table. The
    other nine panel axes stay untested: this release does not measure them.
    """
    inchikey = _inchikey(smiles)
    values, batch_notes = _matching_nanomolar(subset, inchikey)
    value = statistics.median(values) if values else None
    note = f"median_of_{len(values)}_nM_readings" if values else "ligand_not_in_octant_cyp3a4"
    if batch_notes:
        note = f"{note}; {batch_notes[0]}" if len(batch_notes) == 1 else f"{note}; {len(batch_notes)}_octant_batches"
    reading = AxisMeasurement(
        axis=OCTANT_AXIS,
        chembl_target="",
        value_nm=value,
        n_values=len(values),
        unit="nM",
        relation="=" if value is not None else "",
        panel_cut_nm=panel_cut(OCTANT_AXIS),
        threshold_source=panel_cut_source(OCTANT_AXIS),
        source=OCTANT_CURATION if value is not None else "",
        activity_ids=(),
        note=note,
        target=OCTANT_ACCESSION if value is not None else "",
    ).as_record()
    reading["standard_type"] = "IC50"
    notes = [
        "CYP3A4_pIC50 was converted to nM with 10**(9-pIC50) before this reading was stored",
        "CYP2J2 is in the same Octant release and is not a panel axis",
        f"matched on the full InChIKey {inchikey}",
        OCTANT_LICENCE,
    ]
    return PanelMeasurements(
        smiles=smiles.strip(),
        inchikey=inchikey,
        molecule_chembl_id=None,
        axes={OCTANT_AXIS: reading},
        method=OCTANT_METHOD,
        notes=tuple(notes),
    )


def installed_subset() -> Path | None:
    """The converted CYP3A4 subset under ``DATA_PATH/octant``, if it has been built."""
    path = octant_directory() / SUBSET_FILE
    return path if path.is_file() else None


def prepare_toxicology_table(
    directory: Path,
    smiles: str,
    *,
    subset: Path | None = None,
) -> Path | None:
    """Fill a missing ``toxicology.tsv`` from the Octant CYP3A4 subset.

    Called by the ligand screen before it reads the table. The subset affinity
    is already nanomolar. An existing table is left alone, so an operator file
    or a ChEMBL export is not overwritten with one CYP3A4 row. A ligand the
    subset does not contain, a missing subset, or a SMILES RDKit cannot read
    leaves the file absent and the axis untested.
    """
    table = directory / "toxicology.tsv"
    if table.is_file() or (directory / "toxicology.csv").is_file():
        return None
    source = subset if subset is not None else installed_subset()
    if source is None or not Path(source).is_file():
        return None
    text = smiles.strip()
    if not text:
        return None
    try:
        measured = measured_nm_from_octant(text, Path(source))
    except LigandUnreadableError:
        return None
    except SafetyBackendUnavailableError:
        return None
    return write_toxicology_tsv(table, measured)


def write_toxicology_tsv(path: Path, measured: PanelMeasurements) -> Path | None:
    """Write ``toxicology.tsv`` for one ligand, or leave an existing table alone.

    Rows come from :func:`to_toxicology_rows`, so the only concentration column
    is ``nm`` and its unit cell is ``nM``. A ligand the subset does not contain
    writes nothing: an empty table would say the panel had been reported.
    Returns the path when a file was written.
    """
    if path.is_file():
        return None
    rows = to_toxicology_rows(measured)
    if not rows:
        return None
    for row in rows:
        if row.get("unit") != "nM":
            raise ValueError("a toxicology row reached the writer without a nM unit")
        float(row["nm"])
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete table only if the destination is still absent.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".octant-", delete=False) as staged:
        temporary = Path(staged.name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_TOXICOLOGY_COLUMNS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        try:
            os.link(temporary, path)
        except FileExistsError:
            return None
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _convert(inhibition: Path) -> tuple[list[tuple[str, ...]], int, Counter[str], set[str]]:
    kept: list[tuple[str, ...]] = []
    skipped: Counter[str] = Counter()
    ligands: set[str] = set()
    scanned = 0
    with inhibition.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        names = reader.fieldnames or []
        if _PIC50 not in names:
            raise SafetyBackendUnavailableError(f"column_absent:{_PIC50}")
        for column in _NEEDED:
            if column not in names:
                raise SafetyBackendUnavailableError(f"column_absent:{column}")
        saw_cyp2j2_column = any(_NOT_PANEL in (name or "") for name in names)
        for row in reader:
            scanned += 1
            enzyme = (row.get(_ENZYME) or "").strip()
            if enzyme == _NOT_PANEL:
                skipped["cyp2j2_not_a_panel_axis"] += 1
                continue
            if enzyme and enzyme != "CYP3A4":
                skipped["enzyme_not_cyp3a4"] += 1
                continue
            # A CYP2J2 potency sitting beside a real CYP3A4 pIC50 is recorded
            # and then ignored. The CYP3A4 reading is the one the panel can take.
            if saw_cyp2j2_column and _cyp2j2_cell(row):
                skipped["cyp2j2_not_a_panel_axis"] += 1
            raw = (row.get(_PIC50) or "").strip()
            if not raw:
                skipped["no_pic50"] += 1
                continue
            try:
                pic50 = float(raw)
            except ValueError:
                skipped["pic50_unparsed"] += 1
                continue
            if not math.isfinite(pic50):
                skipped["pic50_not_finite"] += 1
                continue
            smiles = (row.get(_SMILES) or "").strip()
            if not smiles:
                skipped["smiles_absent"] += 1
                continue
            try:
                inchikey = _inchikey(smiles)
            except LigandUnreadableError:
                skipped["smiles_unparsed"] += 1
                continue
            try:
                nanomolar = pic50_to_nm(pic50)
            except ValueError:
                skipped["pic50_outside_numeric_range"] += 1
                continue
            batch = (row.get(_BATCH) or "").strip()
            activity = (row.get(_ACTIVITY) or "").strip()
            direction = (row.get(_DIRECTION) or "").strip()
            kept.append(
                (
                    inchikey,
                    "",
                    smiles,
                    OCTANT_AXIS,
                    OCTANT_ACCESSION,
                    format(nanomolar, ".6g"),
                    "=",
                    "IC50",
                    "1",
                    OCTANT_CURATION,
                    _source(batch, pic50, activity, direction),
                )
            )
            ligands.add(inchikey)
    return kept, scanned, skipped, ligands


def _cyp2j2_cell(row: Mapping[str, str]) -> bool:
    return any(_NOT_PANEL in name and (value or "").strip() for name, value in row.items())


def _source(batch: str, pic50: float, activity: str, direction: str) -> str:
    parts = ["Octant", OCTANT_LICENCE]
    if batch:
        parts.append(batch)
    parts.append(f"CYP3A4_pIC50={pic50} converted with 10**(9-pIC50)")
    if activity:
        parts.append(f"activity={activity}")
    if direction:
        parts.append(f"direction={direction}")
    return "; ".join(parts)


def _matching_nanomolar(subset: Path, inchikey: str) -> tuple[list[float], list[str]]:
    if not subset.is_file():
        raise SafetyBackendUnavailableError(f"octant_subset_not_found:{subset}")
    values: list[float] = []
    notes: list[str] = []
    with subset.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        names = reader.fieldnames or []
        if _PIC50 in names:
            raise SafetyBackendUnavailableError("octant_pic50_not_converted")
        if len(names) != len(set(names)):
            raise SafetyBackendUnavailableError("octant_subset_duplicate_columns")
        for column in SUBSET_REQUIRED_COLUMNS:
            if column not in names:
                raise SafetyBackendUnavailableError(f"octant_subset_column_absent:{column}")
        for row in reader:
            if (row.get("panel_axis") or "").strip() != OCTANT_AXIS:
                continue
            if (row.get("inchikey") or "").strip() != inchikey:
                continue
            if (None in row or row.get("accession") != OCTANT_ACCESSION or row.get("assay") != "IC50"
                    or row.get("relation") != "=" or row.get("curation") != OCTANT_CURATION):
                raise SafetyBackendUnavailableError("octant_subset_measurement_mismatch")
            try:
                value = float((row.get("affinity_nanomolar") or "").strip())
            except ValueError:
                continue
            if not math.isfinite(value) or value <= 0:
                raise SafetyBackendUnavailableError("octant_subset_concentration_invalid")
            values.append(value)
            source = (row.get("source") or "").strip()
            if source:
                notes.append(source)
    return values, notes


def _download(url: str, timeout_s: int) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    # The URL is never caller-supplied: it is the release constant in
    # OCTANT_RELEASE joined with one of the file names listed in this module,
    # so no argument can redirect the scheme or the host.
    with urlopen(request, timeout=timeout_s) as response:
        return response.read()


def _looks_like_tsv(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            first = handle.readline()
    except OSError:
        return False
    return "\t" in first and not first.lstrip().startswith("<")
