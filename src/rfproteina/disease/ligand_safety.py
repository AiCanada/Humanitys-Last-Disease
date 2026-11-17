"""Small-molecule safety readings, keeping measured concentrations and predicted
probabilities in separate columns that can never be confused.

:mod:`rfproteina.disease.toxicology` reads the ligand safety panel in nanomolar
and refuses a number without a unit, for the right reason: a classifier
probability is not a concentration, and 0.9 in a column the screen reads as nM
is a 0.9 nM hit that was never measured. The earlier survey named this exact
units problem. This module respects it on both sides:

* **The honest route.** :func:`fetch_measured_nm` pulls *measured* activities
  from ChEMBL (free REST API, no key) for the panel targets and returns them in
  nM with their provenance, ready to write as ``toxicology.tsv`` rows the
  existing screen already reads. Every value carries ``unit="nM"`` and the
  ChEMBL activity ids it came from.

* **The predicted route, fenced off.** :func:`bayesherg_probability` and
  :func:`admet_ai_predictions` wrap BayeshERG (Kim and Nam, MIT) and ADMET-AI
  (Swanson et al., MIT) behind import guards. Their outputs are *probabilities*,
  stored only under ``*_prob`` keys with ``unit="probability"``. They are never
  written into an ``*_nm`` column, and :func:`assert_no_probability_in_nm`
  enforces that -- the test suite calls it, and so can a caller.

The 10 micromolar panel cut these feed is a program convention with no
publication behind it; that is recorded in
:data:`rfproteina.disease.combinations.PANEL_NM_SOURCE` and not restated as a
threshold here.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from rfproteina.disease.combinations import LIGAND_AXES, panel_cut, panel_cut_source
from rfproteina.disease.limits import USER_AGENT

#: ChEMBL target ids for each panel axis, verified against the ChEMBL target API
#: on 2026-11-16: each is the Homo sapiens SINGLE PROTEIN entry for that gene.
#: NOT guessed; a target whose id could not be verified would be omitted, not
#: filled with a plausible-looking id.
CHEMBL_TARGETS: dict[str, dict[str, str]] = {
    "herg": {"chembl_id": "CHEMBL240", "pref_name": "KCNH2 (hERG)"},
    "nav1_5": {"chembl_id": "CHEMBL1980", "pref_name": "SCN5A (Nav1.5)"},
    "cyp3a4": {"chembl_id": "CHEMBL340", "pref_name": "Cytochrome P450 3A4"},
    "cyp2d6": {"chembl_id": "CHEMBL289", "pref_name": "Cytochrome P450 2D6"},
    "cyp2c9": {"chembl_id": "CHEMBL3397", "pref_name": "Cytochrome P450 2C9"},
    "cyp1a2": {"chembl_id": "CHEMBL3356", "pref_name": "Cytochrome P450 1A2"},
    "cyp2c19": {"chembl_id": "CHEMBL3622", "pref_name": "Cytochrome P450 2C19"},
    "pgp": {"chembl_id": "CHEMBL4302", "pref_name": "ABCB1 (P-glycoprotein)"},
    "bcrp": {"chembl_id": "CHEMBL5393", "pref_name": "ABCG2 (BCRP)"},
    "bsep": {"chembl_id": "CHEMBL6020", "pref_name": "ABCB11 (BSEP)"},
}

#: The base of the ChEMBL REST API. One place to change it, one place a test
#: points its fake server at.
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
#: Activity types taken as an affinity in nM. Order is the preference order.
_AFFINITY_TYPES = ("IC50", "Ki", "Kd")
#: Relations that make a reading a determinate concentration. ``>`` and ``<``
#: qualify the number and are recorded as skipped rather than averaged in.
_DETERMINATE_RELATIONS = frozenset({"=", None, ""})
_TIMEOUT = 60

#: Anything urlopen-shaped. Injected so a test drives a server it wrote, exactly
#: as :mod:`rfproteina.disease.tables` does.
Opener = Callable[..., Any]


class SafetyBackendUnavailableError(Exception):
    """An optional predictor is not installed. Its absence is reported, never faked."""


class LigandUnreadableError(Exception):
    """RDKit is missing, or the SMILES does not parse. Never guessed around."""


@dataclass(frozen=True)
class AxisMeasurement:
    """Measured activity for one panel axis, in nM, with where it came from."""

    axis: str
    chembl_target: str
    value_nm: float | None
    n_values: int
    unit: str
    relation: str
    panel_cut_nm: float
    threshold_source: str
    source: str
    activity_ids: tuple[int, ...]
    note: str = ""

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["activity_ids"] = list(self.activity_ids)
        return record


@dataclass(frozen=True)
class PanelMeasurements:
    """Measured nM for the panel. Nothing here is a probability."""

    smiles: str
    inchikey: str | None
    molecule_chembl_id: str | None
    axes: dict[str, dict[str, object]]
    unit: str = "nM"
    method: str = "chembl_measured_nm_v1"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["axes"] = {axis: dict(reading) for axis, reading in self.axes.items()}
        record["notes"] = list(self.notes)
        return record


@dataclass(frozen=True)
class PanelProbabilities:
    """Predicted probabilities. Every key here ends in ``_prob`` and is unitless on 0-1."""

    smiles: str
    model: str
    probabilities: dict[str, float]
    unit: str = "probability"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["probabilities"] = dict(self.probabilities)
        record["notes"] = list(self.notes)
        return record


def _inchikey(smiles: str) -> str:
    text = (smiles or "").strip()
    if not text:
        raise LigandUnreadableError("empty_smiles")
    try:
        from rdkit import Chem
    except ImportError as error:  # pragma: no cover - rdkit is pinned in pyproject
        raise LigandUnreadableError(f"rdkit_unavailable:{error}") from error
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise LigandUnreadableError("smiles_unparsable")
    return str(Chem.MolToInchiKey(mol))


def _get(url: str, opener: Opener) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with opener(request, timeout=_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def _molecule_id(inchikey: str, opener: Opener) -> str | None:
    url = f"{CHEMBL_BASE}/molecule?molecule_structures__standard_inchi_key={quote(inchikey)}&format=json&limit=1"
    payload = _get(url, opener)
    molecules = payload.get("molecules") or []
    if not molecules:
        return None
    return str(molecules[0].get("molecule_chembl_id")) if molecules[0].get("molecule_chembl_id") else None


def _axis_activities(molecule_id: str, target_id: str, opener: Opener, limit: int) -> list[dict[str, Any]]:
    types = ",".join(_AFFINITY_TYPES)
    url = (
        f"{CHEMBL_BASE}/activity?molecule_chembl_id={quote(molecule_id)}"
        f"&target_chembl_id={quote(target_id)}&standard_type__in={quote(types)}"
        f"&format=json&limit={int(limit)}"
    )
    payload = _get(url, opener)
    return list(payload.get("activities") or [])


def _nanomolar(activities: Sequence[dict[str, Any]]) -> tuple[list[float], list[int], int]:
    """Determinate nM readings, their activity ids, and how many were skipped for a bound."""
    values: list[float] = []
    ids: list[int] = []
    skipped = 0
    for row in activities:
        unit = row.get("standard_units")
        if unit != "nM":
            continue
        relation = row.get("standard_relation")
        if relation not in _DETERMINATE_RELATIONS:
            skipped += 1
            continue
        raw = row.get("standard_value")
        if raw is None:
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number) or number < 0:
            continue
        values.append(number)
        if row.get("activity_id") is not None:
            ids.append(int(row["activity_id"]))
    return values, ids, skipped


def fetch_measured_nm(
    smiles: str,
    axes: Sequence[str] = LIGAND_AXES,
    opener: Opener = urlopen,
    limit: int = 1000,
) -> PanelMeasurements:
    """Measured nM for each panel axis from ChEMBL, aggregated as the median.

    Only axes in :data:`CHEMBL_TARGETS` can be fetched. A molecule ChEMBL has
    never seen yields every axis ``untested`` with a reason, never a zero or a
    guessed value. Nothing here converts a probability into a concentration,
    because nothing here produces a probability.
    """
    inchikey = _inchikey(smiles)
    molecule_id = _molecule_id(inchikey, opener)
    readings: dict[str, dict[str, object]] = {}
    for axis in axes:
        target = CHEMBL_TARGETS.get(axis)
        if target is None:
            readings[axis] = AxisMeasurement(
                axis=axis,
                chembl_target="",
                value_nm=None,
                n_values=0,
                unit="nM",
                relation="",
                panel_cut_nm=panel_cut(axis),
                threshold_source=panel_cut_source(axis),
                source="",
                activity_ids=(),
                note="axis_not_in_chembl_target_map",
            ).as_record()
            continue
        if molecule_id is None:
            readings[axis] = AxisMeasurement(
                axis=axis,
                chembl_target=target["chembl_id"],
                value_nm=None,
                n_values=0,
                unit="nM",
                relation="",
                panel_cut_nm=panel_cut(axis),
                threshold_source=panel_cut_source(axis),
                source=f"ChEMBL {target['chembl_id']} ({target['pref_name']})",
                activity_ids=(),
                note="molecule_not_in_chembl",
            ).as_record()
            continue
        activities = _axis_activities(molecule_id, target["chembl_id"], opener, limit)
        values, ids, skipped = _nanomolar(activities)
        value = round(float(statistics.median(values)), 3) if values else None
        note = f"median_of_{len(values)}_nM_readings" if values else "no_determinate_nM_activity"
        if skipped:
            note = f"{note}; {skipped}_bounded_readings_excluded"
        readings[axis] = AxisMeasurement(
            axis=axis,
            chembl_target=target["chembl_id"],
            value_nm=value,
            n_values=len(values),
            unit="nM",
            relation="=",
            panel_cut_nm=panel_cut(axis),
            threshold_source=panel_cut_source(axis),
            source=f"ChEMBL {target['chembl_id']} ({target['pref_name']})",
            activity_ids=tuple(ids),
            note=note,
        ).as_record()
    return PanelMeasurements(
        smiles=smiles.strip(),
        inchikey=inchikey,
        molecule_chembl_id=molecule_id,
        axes=readings,
        notes=(
            "every value is a measured ChEMBL activity in nM, aggregated as the median",
            "no probability is produced or stored here; nothing is converted into a concentration",
            "bounded readings (> or <) are excluded from the median and counted in the note",
        ),
    )


def to_toxicology_rows(measured: PanelMeasurements) -> list[dict[str, str]]:
    """Long-form rows for ``toxicology.tsv``: axis, protein, nm, unit, relation, source.

    Only axes with a determinate value become rows; an untested axis is left out
    so the screen reports it ``untested`` rather than reading a blank as a number.
    """
    rows: list[dict[str, str]] = []
    for axis, reading in measured.axes.items():
        value = reading.get("value_nm")
        if value is None:
            continue
        rows.append(
            {
                "axis": axis,
                "protein": str(reading.get("chembl_target", "")),
                "nm": str(value),
                "unit": "nM",
                "relation": str(reading.get("relation", "=")),
                "source": str(reading.get("source", "")),
            }
        )
    return rows


def bayesherg_probability(smiles: str) -> PanelProbabilities:
    """BayeshERG hERG probability with its uncertainty. Raises if BayeshERG is absent.

    BayeshERG (Kim and Nam, MIT) returns a calibrated probability, not a
    concentration. It is stored under ``herg_prob`` with ``unit="probability"``
    and never written into an ``*_nm`` column.
    """
    text = (smiles or "").strip()
    if not text:
        raise LigandUnreadableError("empty_smiles")
    try:
        import bayesherg  # type: ignore[import-not-found]
    except ImportError as error:
        raise SafetyBackendUnavailableError(f"bayesherg_unavailable:{error}") from error
    result = bayesherg.predict(text)  # pragma: no cover - exercised only where installed
    probabilities = {
        "herg_prob": float(result["probability"]),
        "herg_prob_std": float(result.get("uncertainty", float("nan"))),
    }
    return PanelProbabilities(
        smiles=text,
        model="BayeshERG",
        probabilities=probabilities,
        notes=("a probability is not a concentration and is never written to an *_nm column",),
    )


def admet_ai_predictions(smiles: str) -> PanelProbabilities:
    """ADMET-AI classification probabilities. Raises if ADMET-AI is absent.

    ADMET-AI (Swanson et al., MIT) returns probabilities for hERG, the CYPs and
    the transporters among many endpoints. Only its probability columns are kept
    here, each under a ``*_prob`` key with ``unit="probability"``. Its regression
    endpoints are not turned into nM, because they are not the same quantity.
    """
    text = (smiles or "").strip()
    if not text:
        raise LigandUnreadableError("empty_smiles")
    try:
        from admet_ai import ADMETModel  # type: ignore[import-not-found]
    except ImportError as error:
        raise SafetyBackendUnavailableError(f"admet_ai_unavailable:{error}") from error
    model = ADMETModel()  # pragma: no cover - exercised only where installed
    predictions = model.predict(smiles=text)  # pragma: no cover
    probabilities = {  # pragma: no cover
        f"{key}_prob": float(value)
        for key, value in predictions.items()
        if isinstance(value, int | float) and 0.0 <= float(value) <= 1.0
    }
    return PanelProbabilities(  # pragma: no cover
        smiles=text,
        model="ADMET-AI",
        probabilities=probabilities,
        notes=("only probability-valued endpoints are kept; nothing is converted into nM",),
    )


def assert_no_probability_in_nm(record: dict[str, object]) -> None:
    """Raise if any ``*_nm`` field carries a probability, or any ``*_prob`` a concentration.

    The contract is structural, not statistical: an ``*_nm`` value must declare
    nM and an ``*_prob`` value must be a unitless 0-1 number. A probability that
    wandered into an ``*_nm`` column is the one error this whole module exists to
    prevent, so it is checked rather than trusted.
    """

    def walk(node: object, unit: str | None) -> None:
        if isinstance(node, dict):
            local_unit = node.get("unit", unit) if isinstance(node.get("unit"), str) else unit
            for key, value in node.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    number = float(value)
                    if key.endswith("_nm") and 0.0 <= number <= 1.0 and local_unit == "probability":
                        raise ValueError(f"probability stored under nM key: {key}={number}")
                    if key.endswith("_prob") and not (0.0 <= number <= 1.0):
                        raise ValueError(f"non-probability stored under _prob key: {key}={number}")
                    if key.endswith("_prob") and local_unit not in {None, "probability"}:
                        raise ValueError(f"_prob key declares a non-probability unit: {key} unit={local_unit}")
                walk(value, local_unit)
        elif isinstance(node, list):
            for item in node:
                walk(item, unit)

    walk(record, None)
