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
  :func:`admet_ai_predictions` wrap BayeshERG (Kim et al. 2022) and ADMET-AI
  (Swanson et al. 2024, MIT, verified from the installed distribution's
  ``License-Expression``) behind import guards. BayeshERG is not installed here
  and its licence has not been read at source, so it must be before this route
  is used. Their outputs are *probabilities*,
  stored only under ``*_prob`` keys with ``unit="probability"``. They are never
  written into an ``*_nm`` column, and :func:`assert_no_probability_in_nm`
  enforces that -- the test suite calls it, and so can a caller.

  ADMET-AI is read through an explicit endpoint allowlist
  (:data:`ADMET_AI_ENDPOINTS`) rather than by keeping whatever it returns that
  happens to sit inside 0-1. It also predicts regressions that live on 0-1, so a
  range test would file a druglikeness score, an alert count or a log
  concentration as a probability -- the same units error from the other side.

The 10 micromolar panel cut these feed is a program convention with no
publication behind it; that is recorded in
:data:`rfproteina.disease.combinations.PANEL_NM_SOURCE` and not restated as a
threshold here.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
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

#: Panel axis -> UniProt accession, for the sources that key their targets by
#: accession rather than by a ChEMBL id. Verified on 2026-11-18 against the
#: "Human SwissProt" column of the IUPHAR/BPS target table, which cross-
#: references UniProt per HGNC symbol: KCNH2, SCN5A, the five CYPs, ABCB1,
#: ABCG2, ABCB11. Not recalled, and not taken off a search result.
#:
#: Lives here beside :data:`CHEMBL_TARGETS` because target identity is this
#: module's business; :mod:`rfproteina.disease.tables` imports it rather than
#: keeping a second copy of ten accessions that could drift from this one.
BINDINGDB_PANEL_ACCESSIONS: dict[str, str] = {
    "herg": "Q12809",
    "nav1_5": "Q14524",
    "cyp3a4": "P08684",
    "cyp2d6": "P10635",
    "cyp2c9": "P11712",
    "cyp1a2": "P05177",
    "cyp2c19": "P33261",
    "pgp": "P08183",
    "bcrp": "Q9UNQ0",
    "bsep": "O95342",
}

#: Method id for the offline route, so a stored record says which source and
#: which rule produced it. The live ChEMBL route is ``chembl_measured_nm_v1``.
BINDINGDB_METHOD = "bindingdb_measured_nm_v1"

#: Same route, matched on the InChIKey skeleton instead. A separate method id
#: because the readings it pools are not guaranteed to be the same molecule:
#: same connectivity, possibly a different stereoisomer, salt or isotope.
BINDINGDB_CONNECTIVITY_METHOD = "bindingdb_measured_nm_connectivity_v1"

#: The resolutions :func:`measured_nm_from_bindingdb` will match at.
_BDB_MATCH_MODES = frozenset({"exact", "connectivity"})

#: One parse per subset file, shared by the exact lookup and the read-across:
#: both want the same rows and only one of them needs fingerprints on top.
_BDB_ROWS: dict[tuple[str, int, int], Any] = {}
_BDB_ROWS_LOCK = threading.Lock()

#: Columns :func:`measured_nm_from_bindingdb` needs from the subset. A subset
#: missing one is refused by name rather than read as empty.
#: ``test_disease_ligand_safety`` asserts these stay a subset of what
#: :func:`rfproteina.disease.tables.bindingdb_panel_subset` writes.
_BINDINGDB_NEEDED = ("inchikey", "panel_axis", "affinity_nanomolar", "relation", "assay", "chains", "curation")

#: Method id for the read-across. Distinct from the two measured methods because
#: it is not a measurement of the design at all.
READ_ACROSS_METHOD = "bindingdb_read_across_v1"

#: Morgan fingerprints with Tanimoto similarity, the ordinary choice for "how
#: alike are these two molecules". Radius 2 over 2048 bits is the common default;
#: it is recorded in the record so a stored read-across says how its similarity
#: was arrived at rather than leaving a bare number to be trusted.
_MORGAN_RADIUS = 2
_MORGAN_BITS = 2048
_MORGAN_METHOD = "morgan_r2_2048_tanimoto"

#: Columns the read-across needs from the subset. ``smiles`` is the one the
#: exact-key lookup has no use for and this cannot work without.
_READ_ACROSS_NEEDED = (
    "inchikey",
    "smiles",
    "panel_axis",
    "affinity_nanomolar",
    "relation",
    "assay",
    "curation",
)

#: Method id for the GtoPdb secondary-pharmacology read-across. Distinct from
#: the panel read-across because it answers a wider question over a different
#: file: every target GtoPdb curated, not the ten tox axes.
OFFTARGET_METHOD = "gtopdb_offtarget_read_across_v1"

#: Columns :func:`offtarget_profile` needs from the off-target subset.
#: ``test_disease_tables`` asserts these stay a subset of what
#: :func:`rfproteina.disease.tables.gtopdb_offtarget_subset` writes.
_OFFTARGET_NEEDED = (
    "inchikey",
    "smiles",
    "ligand_name",
    "target_name",
    "gene_symbol",
    "family_class",
    "affinity_nanomolar",
    "action",
    "primary_target",
    "approved",
    "withdrawn",
)

#: One fingerprint index per off-target subset, built once.
_OFFTARGET_INDEX: dict[tuple[str, int, int], Any] = {}
_OFFTARGET_LOCK = threading.Lock()

#: One fingerprint index per subset file, built once and keyed on the file's
#: identity: parsing 29,393 distinct structures is the cost, and a query against
#: them is milliseconds.
_BDB_INDEX: dict[tuple[str, int, int], Any] = {}
_BDB_INDEX_LOCK = threading.Lock()

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

#: Panel axis -> ADMET-AI endpoint id, verified against
#: ``admet_ai/resources/data/admet.csv`` at admet-ai 2.0.1 on 2026-11-17. Every id
#: below is a row whose ``task_type`` is ``classification`` with units ``-`` on
#: 0-1, and each one measures inhibition or blocking of that panel protein.
#:
#: This is an allowlist rather than a range filter, and the difference is the
#: whole point. ADMET-AI returns its regression tasks in the same frame, and
#: several of them live inside 0-1: ``QED`` is bounded on 0-1 by definition,
#: ``Lipinski`` runs 0-4, the ``*_alert`` columns are counts that are usually 0 or
#: 1, and ``LD50_Zhu``, ``Caco2_Wang`` and ``Solubility_AqSolDB`` are logs that
#: often land there. Keeping "every value between 0 and 1" would file a count or a
#: log concentration under a ``*_prob`` key, which is exactly the confusion this
#: module exists to prevent.
#:
#: The ``*_Substrate_CarbonMangels`` classifiers are deliberately absent. They
#: predict whether a molecule is metabolised *by* a CYP, not whether it inhibits
#: one, and the panel axes are inhibition -- mapping them onto ``cyp3a4`` and its
#: siblings would answer a different question under the panel's name.
ADMET_AI_ENDPOINTS: dict[str, str] = {
    "herg": "hERG",
    "cyp1a2": "CYP1A2_Veith",
    "cyp2c19": "CYP2C19_Veith",
    "cyp2c9": "CYP2C9_Veith",
    "cyp2d6": "CYP2D6_Veith",
    "cyp3a4": "CYP3A4_Veith",
    "pgp": "Pgp_Broccatelli",
}

#: The panel axes ADMET-AI has no endpoint for, derived from the allowlist so it
#: cannot drift out of step with it. Nav1.5 (SCN5A), BCRP (ABCG2) and BSEP
#: (ABCB11) appear nowhere in its task list, so it answers seven of the ten axes.
#: Named rather than silently omitted: a reader of a stored record should be able
#: to tell "ADMET-AI cannot answer this" from "ADMET-AI said nothing today".
ADMET_AI_UNCOVERED_AXES: tuple[str, ...] = tuple(axis for axis in LIGAND_AXES if axis not in ADMET_AI_ENDPOINTS)

#: Bumped when the allowlist or the reading rule changes, so a stored record
#: states which rule produced it.
ADMET_AI_METHOD = "admet_ai_classification_v1"

#: The loaded model, built once per process. Loading assembles two Chemprop
#: ensembles from disk; doing that per molecule would dominate a screen.
_ADMET_MODEL: Any | None = None
_ADMET_MODEL_LOCK = threading.Lock()


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
    #: The target's identifier in the database that produced this reading, where
    #: that is not a ChEMBL id -- a UniProt accession on the BindingDB route.
    #: ``chembl_target`` stays empty there rather than carrying an accession
    #: under a field name that says ChEMBL.
    target: str = ""

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
    """Predicted probabilities. Every key here ends in ``_prob`` and is unitless on 0-1.

    ``endpoints`` records which upstream endpoint produced each axis, and
    ``uncovered_axes`` records the panel axes the model cannot answer at all, so a
    stored reading says what was asked as well as what came back.
    """

    smiles: str
    model: str
    probabilities: dict[str, float]
    unit: str = "probability"
    notes: tuple[str, ...] = field(default_factory=tuple)
    method: str = ""
    endpoints: dict[str, str] = field(default_factory=dict)
    uncovered_axes: tuple[str, ...] = field(default_factory=tuple)

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["probabilities"] = dict(self.probabilities)
        record["notes"] = list(self.notes)
        record["endpoints"] = dict(self.endpoints)
        record["uncovered_axes"] = list(self.uncovered_axes)
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
                # Whichever database named the target. The ChEMBL route fills
                # chembl_target, the BindingDB route fills target with a UniProt
                # accession, and the column means "the protein this was measured
                # against" either way.
                "protein": str(reading.get("target") or reading.get("chembl_target") or ""),
                "nm": str(value),
                "unit": "nM",
                "relation": str(reading.get("relation", "=")),
                "source": str(reading.get("source", "")),
            }
        )
    return rows


@dataclass(frozen=True)
class _SubsetRows:
    """A BindingDB subset parsed once: the rows, and which columns it had."""

    rows: tuple[tuple[str, str, str, float, str, str, int, str], ...]
    columns: frozenset[str]


def _bdb_rows(subset: Path) -> _SubsetRows:
    """Parse a subset once and hold it, keyed on the file's identity.

    Both readers want the same 60,095 rows: the exact lookup takes a median of
    one compound's, and the read-across fingerprints all of them. Parsing twice
    was pure waste, so the parse is shared and each applies its own rule to the
    result -- bounds and complexes are kept here rather than filtered, because
    the two disagree about what to do with them.

    Keyed on path, size and modification time, so a rebuilt subset is never
    answered out of a stale cache.
    """
    stat = subset.stat()
    key = (str(subset.resolve()), stat.st_size, stat.st_mtime_ns)
    with _BDB_ROWS_LOCK:
        cached = _BDB_ROWS.get(key)
    if cached is not None:
        return cached

    rows: list[tuple[str, str, str, float, str, str, int, str]] = []
    try:
        with subset.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            names = frozenset(reader.fieldnames or ())
            for column in _BINDINGDB_NEEDED:
                if column not in names:
                    raise SafetyBackendUnavailableError(f"bindingdb_subset_column_absent:{column}")
            for row in reader:
                axis = (row.get("panel_axis") or "").strip()
                inchikey = (row.get("inchikey") or "").strip()
                if not axis or not inchikey:
                    continue
                try:
                    value = float((row.get("affinity_nanomolar") or "").strip())
                except ValueError:
                    continue
                if not math.isfinite(value) or value < 0:
                    continue
                try:
                    chains = int((row.get("chains") or "1").strip() or "1")
                except ValueError:
                    chains = 1
                rows.append(
                    (
                        inchikey,
                        (row.get("smiles") or "").strip(),
                        axis,
                        value,
                        (row.get("relation") or "=").strip() or "=",
                        (row.get("assay") or "").strip(),
                        chains,
                        (row.get("curation") or "").strip(),
                    )
                )
    except (OSError, UnicodeError, csv.Error) as error:
        raise SafetyBackendUnavailableError(f"bindingdb_subset_unreadable:{type(error).__name__}") from error

    parsed = _SubsetRows(tuple(rows), names)
    with _BDB_ROWS_LOCK:
        _BDB_ROWS[key] = parsed
    return parsed


def measured_nm_from_bindingdb(
    smiles: str,
    subset: Path,
    axes: Sequence[str] = LIGAND_AXES,
    *,
    single_chain_only: bool = False,
    match: str = "exact",
) -> PanelMeasurements:
    """Measured nM for each panel axis from a local BindingDB subset, by InChIKey.

    The offline counterpart of :func:`fetch_measured_nm`, and deliberately the
    same shape: the same median of determinate nM readings, the same
    :class:`PanelMeasurements` out, so a caller can swap the live ChEMBL fetch
    for a file without changing how it reads the answer. ``method`` says which
    one produced a stored record, and at which matching resolution.

    ``subset`` is what
    :func:`~rfproteina.disease.tables.bindingdb_panel_subset` writes. It holds
    every compound BindingDB has measured against the panel; this reads only the
    rows belonging to **this** ligand. A molecule BindingDB has never seen leaves
    every axis untested with a reason, which for a novel design is the honest
    answer -- and the reason :func:`nearest_measured_neighbours` exists.

    ``match`` picks the resolution, and the choice is real rather than cosmetic:

    * ``"exact"`` (default) compares the full 27-character InChIKey, so a
      reading belongs to this stereoisomer, this salt form, these isotopes.
    * ``"connectivity"`` compares only the first block, the 14-character
      skeleton, which is what to reach for when a file spells one drug several
      ways. BindingDB does: on ketoconazole it pools forms the exact key misses
      and moves CYP2C19 from 14,550 to 19,000 nM and P-gp from 5,600 to 5,435.
      It will also match an enantiomer, which can carry a different hERG
      liability, so a record made this way says so in its ``method`` and in
      every note and reports how many distinct structures it pooled.

      It is **not** the fix for a sparse-looking answer. Ketoconazole came back
      empty on CYP3A4 under either resolution, and the cause was
      ``single_chain_only``, not the key -- read the note before widening the
      match, because it names the real exclusion.

    A bound (``>``, ``<``) is counted and excluded rather than averaged in,
    which is :func:`_nanomolar`'s rule.

    ``single_chain_only`` defaults to **off**, and the reason is worth stating
    because the opposite looked obviously safer. A Ki against a two-protein
    complex is not a Ki for one chain of it, so excluding multi-chain rows reads
    like the conservative choice. On this data it is not: 99.0% of the subset's
    hERG rows name more than one chain, 99.5% of its CYP2D6 rows and 97.8% of
    its CYP3A4 rows, because these targets are assayed with their accessory
    subunit or redox partner -- the rows still say ``Target Name: Cytochrome
    P450 3A4``, and the second chain is the reductase, not a second target.
    Excluding them discarded 99% of the file and answered "unmeasured" for
    ketoconazole on CYP3A4, where including them gives 48.0 nM over 72 readings,
    which is the textbook figure. Set it ``True`` to be strict, and the note will
    say how many readings that cost.

    The licences are per row and travel with the readings. Whichever
    ``Curation/DataSource`` markings the kept rows carried are named in
    :attr:`PanelMeasurements.notes`, because CC BY-SA 3.0 on the ChEMBL-derived
    rows governs any redistribution of a table built from them.
    """
    if match not in _BDB_MATCH_MODES:
        raise ValueError(f"match must be one of {sorted(_BDB_MATCH_MODES)}, got {match!r}")
    inchikey = _inchikey(smiles)
    if not subset.is_file():
        raise SafetyBackendUnavailableError(f"bindingdb_subset_not_found:{subset}")

    wanted = inchikey if match == "exact" else inchikey.split("-")[0]
    parsed = _bdb_rows(subset)

    per_axis: dict[str, list[float]] = {axis: [] for axis in axes}
    bounded: dict[str, int] = dict.fromkeys(axes, 0)
    complexes: dict[str, int] = dict.fromkeys(axes, 0)
    assays: dict[str, set[str]] = {axis: set() for axis in axes}
    structures: dict[str, set[str]] = {axis: set() for axis in axes}
    markings: dict[str, None] = {}
    for key, _smiles, axis, value, relation, assay, chains, mark in parsed.rows:
        if axis not in per_axis:
            continue
        if (key if match == "exact" else key.split("-")[0]) != wanted:
            continue
        if single_chain_only and chains > 1:
            complexes[axis] += 1
            continue
        if relation not in _DETERMINATE_RELATIONS:
            bounded[axis] += 1
            continue
        per_axis[axis].append(value)
        structures[axis].add(key)
        if assay:
            assays[axis].add(assay)
        if mark:
            markings.setdefault(mark, None)

    readings: dict[str, dict[str, object]] = {}
    for axis in axes:
        accession = BINDINGDB_PANEL_ACCESSIONS.get(axis)
        if accession is None:
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
                note="axis_not_in_bindingdb_accession_map",
                target="",
            ).as_record()
            continue
        values = per_axis[axis]
        value = round(float(statistics.median(values)), 3) if values else None
        if values:
            note = f"median_of_{len(values)}_nM_readings"
            if assays[axis]:
                note = f"{note}; {'/'.join(sorted(assays[axis]))}"
            if match == "connectivity":
                note = f"{note}; connectivity match over {len(structures[axis])}_distinct_structures"
        else:
            note = "ligand_not_measured_against_this_target_in_bindingdb"
        if bounded[axis]:
            note = f"{note}; {bounded[axis]}_bounded_readings_excluded"
        if complexes[axis]:
            note = f"{note}; {complexes[axis]}_multi_chain_readings_excluded"
        readings[axis] = AxisMeasurement(
            axis=axis,
            chembl_target="",
            value_nm=value,
            n_values=len(values),
            unit="nM",
            relation="=",
            panel_cut_nm=panel_cut(axis),
            threshold_source=panel_cut_source(axis),
            source=f"BindingDB UniProt {accession}",
            activity_ids=(),
            note=note,
            target=accession,
        ).as_record()

    notes = [
        "measured values only; nothing here is a prediction and nothing is converted into a probability",
    ]
    if match == "exact":
        notes.append(f"matched on the full InChIKey {inchikey}")
    else:
        notes.append(
            f"matched on the InChIKey skeleton {wanted}, so stereochemistry, salt form and isotopes may differ"
        )
    if markings:
        notes.append("row licences present: " + "; ".join(sorted(markings)))
    else:
        notes.append("no reading matched this ligand, so no row licence applies")
    return PanelMeasurements(
        smiles=smiles.strip(),
        inchikey=inchikey,
        molecule_chembl_id=None,
        axes=readings,
        method=BINDINGDB_METHOD if match == "exact" else BINDINGDB_CONNECTIVITY_METHOD,
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class MeasuredNeighbour:
    """One measured compound near a design, and what was measured on **it**.

    ``neighbour_nanomolar`` is the most potent determinate reading that compound
    has on that axis. The most potent rather than the median because this is a
    liability warning and understating one is the dangerous direction -- it is
    also the rule :mod:`rfproteina.disease.toxicology` applies to its own table.
    """

    inchikey: str
    smiles: str
    similarity: float
    neighbour_nanomolar: float
    n_readings: int
    relation: str
    assay: str
    curation: str


@dataclass(frozen=True)
class PanelReadAcross:
    """Nearest measured analogues per panel axis. **Not** a reading of this design.

    Every number in here was measured on a different molecule. That is the whole
    point and also the whole danger, so the field names refuse the confusion by
    construction: the value is ``neighbour_nanomolar``, never ``*_nm``, and it
    never travels without the ``similarity`` that qualifies it and the
    ``inchikey`` of the compound it belongs to. Nothing here is accepted by
    :func:`~rfproteina.disease.toxicology.measure_toxicology` as a panel value,
    and :func:`to_toxicology_rows` will not take it.

    A read-across is an argument from analogy, not a measurement. It earns its
    place because the alternative for a novel design is nothing at all: an
    exact-structure lookup answers only for molecules already measured, and a
    designed ligand never is.
    """

    smiles: str
    inchikey: str
    neighbours: dict[str, list[dict[str, object]]]
    method: str = READ_ACROSS_METHOD
    fingerprint: str = _MORGAN_METHOD
    min_similarity: float = 0.0
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["neighbours"] = {axis: [dict(entry) for entry in entries] for axis, entries in self.neighbours.items()}
        record["notes"] = list(self.notes)
        return record

    @property
    def closest(self) -> dict[str, float]:
        """Best similarity found per axis, for a caller that wants one number."""
        return {
            axis: max((float(entry["similarity"]) for entry in entries), default=0.0)
            for axis, entries in self.neighbours.items()
        }


def _morgan_index(subset: Path) -> dict[str, list[tuple[Any, str, str, float, int, str, str, str]]]:
    """Fingerprints and per-compound readings from a subset, built once.

    Takes its rows from :func:`_bdb_rows`, so the file is parsed once however
    many readers want it, and adds the part only this needs: a Morgan
    fingerprint per distinct structure. That is the expensive half -- 29,393
    structures to parse -- and the reason it is held rather than repeated per
    query.

    Only determinate readings are indexed. A bound says the compound is *not*
    potent, which is worth knowing but is not a number to rank analogues by.
    """
    stat = subset.stat()
    key = (str(subset.resolve()), stat.st_size, stat.st_mtime_ns)
    with _BDB_INDEX_LOCK:
        cached = _BDB_INDEX.get(key)
    if cached is not None:
        return cached

    parsed = _bdb_rows(subset)
    if "smiles" not in parsed.columns:
        raise SafetyBackendUnavailableError("bindingdb_subset_column_absent:smiles")
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as error:
        raise SafetyBackendUnavailableError(f"rdkit_unavailable:{error}") from error

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=_MORGAN_RADIUS, fpSize=_MORGAN_BITS)
    prints: dict[str, Any] = {}
    # axis -> inchikey -> [most potent value, how many readings, relation, assay, curation, smiles]
    best: dict[str, dict[str, list[Any]]] = {}
    for inchikey, smiles, axis, value, relation, assay, _chains, curation in parsed.rows:
        if relation not in _DETERMINATE_RELATIONS or not smiles:
            continue
        if smiles not in prints:
            mol = Chem.MolFromSmiles(smiles)
            prints[smiles] = generator.GetFingerprint(mol) if mol is not None else None
        if prints[smiles] is None:
            continue
        slot = best.setdefault(axis, {})
        previous = slot.get(inchikey)
        if previous is None:
            slot[inchikey] = [value, 1, relation, assay, curation, smiles]
        else:
            previous[1] += 1
            if value < previous[0]:
                previous[0], previous[2], previous[3], previous[4] = value, relation, assay, curation

    index: dict[str, list[tuple[Any, str, str, float, int, str, str, str]]] = {
        axis: [
            (prints[entry[5]], inchikey, entry[5], entry[0], entry[1], entry[2], entry[3], entry[4])
            for inchikey, entry in compounds.items()
        ]
        for axis, compounds in best.items()
    }
    with _BDB_INDEX_LOCK:
        _BDB_INDEX[key] = index
    return index


def reset_bindingdb_index() -> None:
    """Drop the cached subset rows and fingerprints. For tests, and to free them."""
    with _BDB_ROWS_LOCK:
        _BDB_ROWS.clear()
    with _BDB_INDEX_LOCK:
        _BDB_INDEX.clear()


def nearest_measured_neighbours(
    smiles: str,
    subset: Path,
    axes: Sequence[str] = LIGAND_AXES,
    *,
    neighbours: int = 5,
    min_similarity: float = 0.35,
) -> PanelReadAcross:
    """The measured compounds most like this design, per panel axis.

    This is what makes a measured database useful to a molecule nobody has
    measured. :func:`measured_nm_from_bindingdb` answers only for a ligand
    already in BindingDB, which a designed one is not; this instead asks the
    question a chemist would ask -- *what does the closest thing anyone has
    measured look like?* -- and answers it with that compound's own number, its
    structure, and the Tanimoto similarity that qualifies it.

    Morgan fingerprints, radius 2 over 2048 bits, Tanimoto similarity, which is
    the ordinary choice for this comparison. ``min_similarity`` defaults to 0.35
    because below roughly that a Morgan/Tanimoto pair is not an analogue in any
    useful sense and reporting it would dress noise as evidence; raise it to 0.7
    for a strict read-across, or lower it deliberately to see what the nearest
    thing is at all.

    What comes back is **not** a reading of this design, and nothing in the
    record can be mistaken for one: see :class:`PanelReadAcross`. A number here
    supports a decision to look harder at an axis. It does not fill it in.
    """
    inchikey = _inchikey(smiles)
    if not subset.is_file():
        raise SafetyBackendUnavailableError(f"bindingdb_subset_not_found:{subset}")
    if neighbours < 1:
        raise ValueError("neighbours must be at least 1")
    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError(f"min_similarity must be on 0-1, got {min_similarity}")

    try:
        from rdkit import Chem
        from rdkit.Chem import DataStructs, rdFingerprintGenerator
    except ImportError as error:
        raise SafetyBackendUnavailableError(f"rdkit_unavailable:{error}") from error

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise LigandUnreadableError("smiles_unparsable")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=_MORGAN_RADIUS, fpSize=_MORGAN_BITS)
    query = generator.GetFingerprint(mol)

    index = _morgan_index(subset)
    found: dict[str, list[dict[str, object]]] = {}
    for axis in axes:
        entries = index.get(axis) or []
        if not entries:
            found[axis] = []
            continue
        scores = DataStructs.BulkTanimotoSimilarity(query, [entry[0] for entry in entries])
        ranked = sorted(
            ((score, entry) for score, entry in zip(scores, entries, strict=True) if score >= min_similarity),
            key=lambda pair: (-pair[0], pair[1][3]),
        )[:neighbours]
        found[axis] = [
            asdict(
                MeasuredNeighbour(
                    inchikey=entry[1],
                    smiles=entry[2],
                    similarity=round(float(score), 4),
                    neighbour_nanomolar=entry[3],
                    n_readings=entry[4],
                    relation=entry[5],
                    assay=entry[6],
                    curation=entry[7],
                )
            )
            for score, entry in ranked
        ]

    total = sum(len(entries) for entries in found.values())
    notes = [
        "every number here was measured on a DIFFERENT molecule; it is an analogy, not a reading of this design",
        "nothing here may be written into an *_nm column for this design",
        f"matched by {_MORGAN_METHOD} at or above Tanimoto {min_similarity}",
    ]
    if not total:
        notes.append("no measured compound reached that similarity on any axis")
    licences = sorted({str(entry["curation"]) for entries in found.values() for entry in entries if entry["curation"]})
    if licences:
        notes.append("row licences present: " + "; ".join(licences))
    return PanelReadAcross(
        smiles=smiles.strip(),
        inchikey=inchikey,
        neighbours=found,
        min_similarity=min_similarity,
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class OffTargetProfile:
    """What a design's nearest measured analogues hit, besides their target.

    Secondary pharmacology as a screen reads it: not one number per axis but a
    short list of compounds somebody measured properly, what each of them hits,
    and -- first, because it is the loudest thing here -- whether any of them is
    a withdrawn drug.

    Every number belongs to a **different molecule**, so the same discipline as
    :class:`PanelReadAcross` applies by construction: the value is
    ``neighbour_nanomolar``, never ``*_nm``, and it never travels without the
    ``similarity`` that qualifies it and the compound it belongs to.
    """

    smiles: str
    inchikey: str
    #: The analogues used, closest first.
    analogues: tuple[dict[str, object], ...]
    #: Target family class -> its hits, most potent first. ``gpcr``, ``vgic``,
    #: ``enzyme``, ``transporter``, ``nhr`` and the rest, which is how a
    #: secondary-pharmacology panel is read.
    families: dict[str, list[dict[str, object]]]
    #: Analogues that are withdrawn drugs, by name. Empty is the usual answer
    #: and the interesting one is not.
    withdrawn_analogues: tuple[str, ...] = ()
    approved_analogues: tuple[str, ...] = ()
    method: str = OFFTARGET_METHOD
    fingerprint: str = _MORGAN_METHOD
    min_similarity: float = 0.0
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["analogues"] = [dict(item) for item in self.analogues]
        record["families"] = {name: [dict(hit) for hit in hits] for name, hits in self.families.items()}
        record["withdrawn_analogues"] = list(self.withdrawn_analogues)
        record["approved_analogues"] = list(self.approved_analogues)
        record["notes"] = list(self.notes)
        return record

    @property
    def strongest(self) -> dict[str, float]:
        """Most potent nanomolar seen per family class, for a one-line read."""
        return {
            name: min(float(hit["neighbour_nanomolar"]) for hit in hits) for name, hits in self.families.items() if hits
        }


def _offtarget_index(subset: Path) -> tuple[dict[str, Any], dict[str, list[tuple[str, ...]]]]:
    """Fingerprints per compound and that compound's interactions, built once.

    Keyed on the file's identity so a rebuilt subset is never answered from a
    stale index.
    """
    stat = subset.stat()
    key = (str(subset.resolve()), stat.st_size, stat.st_mtime_ns)
    with _OFFTARGET_LOCK:
        cached = _OFFTARGET_INDEX.get(key)
    if cached is not None:
        return cached
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as error:
        raise SafetyBackendUnavailableError(f"rdkit_unavailable:{error}") from error

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=_MORGAN_RADIUS, fpSize=_MORGAN_BITS)
    prints: dict[str, Any] = {}
    interactions: dict[str, list[tuple[str, ...]]] = {}
    try:
        with subset.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            names = frozenset(reader.fieldnames or ())
            for column in _OFFTARGET_NEEDED:
                if column not in names:
                    raise SafetyBackendUnavailableError(f"gtopdb_offtarget_column_absent:{column}")
            for row in reader:
                inchikey = (row.get("inchikey") or "").strip()
                smiles = (row.get("smiles") or "").strip()
                if not inchikey or not smiles:
                    continue
                try:
                    value = float((row.get("affinity_nanomolar") or "").strip())
                except ValueError:
                    continue
                if not math.isfinite(value) or value < 0:
                    continue
                if inchikey not in prints:
                    mol = Chem.MolFromSmiles(smiles)
                    prints[inchikey] = generator.GetFingerprint(mol) if mol is not None else None
                if prints[inchikey] is None:
                    continue
                interactions.setdefault(inchikey, []).append(
                    (
                        row.get("ligand_name", "").strip(),
                        row.get("target_name", "").strip(),
                        row.get("gene_symbol", "").strip(),
                        row.get("family_class", "").strip() or "unclassified",
                        row.get("family_name", "").strip(),
                        f"{value}",
                        row.get("relation", "").strip(),
                        row.get("assay", "").strip(),
                        row.get("action", "").strip(),
                        row.get("selectivity", "").strip(),
                        row.get("primary_target", "").strip(),
                        row.get("approved", "").strip(),
                        row.get("withdrawn", "").strip(),
                        smiles,
                    )
                )
    except (OSError, UnicodeError, csv.Error) as error:
        raise SafetyBackendUnavailableError(f"gtopdb_offtarget_unreadable:{type(error).__name__}") from error

    built = ({k: v for k, v in prints.items() if v is not None}, interactions)
    with _OFFTARGET_LOCK:
        _OFFTARGET_INDEX[key] = built
    return built


def reset_offtarget_index() -> None:
    """Drop the cached off-target fingerprints. For tests, and to free them."""
    with _OFFTARGET_LOCK:
        _OFFTARGET_INDEX.clear()


def offtarget_profile(
    smiles: str,
    subset: Path,
    *,
    analogues: int = 5,
    min_similarity: float = 0.4,
    per_family: int = 6,
) -> OffTargetProfile:
    """Secondary pharmacology for a design, by analogy to what GtoPdb measured.

    The ten-axis tox panel asks whether a ligand hits hERG, the CYPs and three
    transporters. This asks the question that actually ends programmes: *what
    else does it hit?* GtoPdb's curated interactions cover 1,484 targets across
    95 channel and GPCR families, so a design's close analogues can be looked up
    across all of them at once -- Na_v, TRP, 5-HT, dopamine, muscarinic,
    opioid, the kinases, the nuclear receptors.

    Read it as: these compounds resemble the design, somebody measured them
    properly, and this is what they hit. It is an argument from analogy and the
    record says so in every note. For a designed ligand it is the only measured
    evidence available, because nobody has measured the design.

    ``withdrawn_analogues`` comes first in the record for a reason. A close
    analogue that was taken off the market is a stronger signal than any single
    affinity, and GtoPdb is one of the few sources that marks it.

    ``analogues`` caps how many compounds are used, ``per_family`` how many hits
    are reported for each family class, and ``min_similarity`` defaults to 0.4
    because below roughly there a Morgan/Tanimoto pair is not an analogue and
    reporting it dresses noise as evidence.
    """
    inchikey = _inchikey(smiles)
    if not subset.is_file():
        raise SafetyBackendUnavailableError(f"gtopdb_offtarget_subset_not_found:{subset}")
    if analogues < 1 or per_family < 1:
        raise ValueError("analogues and per_family must be at least 1")
    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError(f"min_similarity must be on 0-1, got {min_similarity}")
    try:
        from rdkit import Chem
        from rdkit.Chem import DataStructs, rdFingerprintGenerator
    except ImportError as error:
        raise SafetyBackendUnavailableError(f"rdkit_unavailable:{error}") from error

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise LigandUnreadableError("smiles_unparsable")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=_MORGAN_RADIUS, fpSize=_MORGAN_BITS)
    query = generator.GetFingerprint(mol)

    prints, interactions = _offtarget_index(subset)
    keys = list(prints)
    scores = DataStructs.BulkTanimotoSimilarity(query, [prints[key] for key in keys]) if keys else []
    ranked = sorted(
        ((score, key) for score, key in zip(scores, keys, strict=True) if score >= min_similarity),
        key=lambda pair: (-pair[0], pair[1]),
    )[:analogues]

    used: list[dict[str, object]] = []
    families: dict[str, list[dict[str, object]]] = {}
    withdrawn: list[str] = []
    approved: list[str] = []
    for score, key in ranked:
        rows = interactions.get(key) or []
        name = rows[0][0] if rows else key
        is_withdrawn = any(row[12] for row in rows)
        is_approved = any(row[11] for row in rows)
        if is_withdrawn and name not in withdrawn:
            withdrawn.append(name)
        if is_approved and name not in approved:
            approved.append(name)
        used.append(
            {
                "inchikey": key,
                "name": name,
                "smiles": rows[0][13] if rows else "",
                "similarity": round(float(score), 4),
                "measured_targets": len({row[1] for row in rows}),
                "approved": bool(is_approved),
                "withdrawn": bool(is_withdrawn),
            }
        )
        for row in rows:
            families.setdefault(row[3], []).append(
                {
                    "target_name": row[1],
                    "gene_symbol": row[2],
                    "family_name": row[4],
                    "neighbour_nanomolar": float(row[5]),
                    "relation": row[6],
                    "assay": row[7],
                    "action": row[8],
                    "selectivity": row[9],
                    "primary_target": row[10] == "true",
                    "similarity": round(float(score), 4),
                    "neighbour_inchikey": key,
                    "neighbour_name": name,
                    "neighbour_approved": bool(is_approved),
                    "neighbour_withdrawn": bool(is_withdrawn),
                }
            )
    for name, hits in families.items():
        hits.sort(key=lambda hit: float(hit["neighbour_nanomolar"]))
        families[name] = hits[:per_family]

    notes = [
        "every number here was measured on a DIFFERENT molecule; it is an analogy, not a reading of this design",
        "nothing here may be written into an *_nm column for this design",
        f"matched by {_MORGAN_METHOD} at or above Tanimoto {min_similarity}",
        "GtoPdb is ODbL; share-alike follows any redistributed derivative",
    ]
    if withdrawn:
        notes.insert(0, "a close analogue is a WITHDRAWN drug: " + "; ".join(withdrawn))
    if not used:
        notes.append("no measured compound reached that similarity")
    return OffTargetProfile(
        smiles=smiles.strip(),
        inchikey=inchikey,
        analogues=tuple(used),
        families=families,
        withdrawn_analogues=tuple(withdrawn),
        approved_analogues=tuple(approved),
        min_similarity=min_similarity,
        notes=tuple(notes),
    )


def bayesherg_probability(smiles: str) -> PanelProbabilities:
    """BayeshERG hERG probability with its uncertainty. Raises if BayeshERG is absent.

    BayeshERG (Kim et al., Brief Bioinform 23(4) 2022,
    doi:10.1093/bib/bbac211) returns a calibrated probability, not a
    concentration. It is stored under ``herg_prob`` with ``unit="probability"``
    and never written into an ``*_nm`` column. Its licence has not been read at
    source; read it before installing this backend.
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


def _admet_model() -> Any:
    """The ADMET-AI model, built once and reused. Raises if ADMET-AI is absent.

    ``include_physchem=False`` and ``drugbank_path=None`` narrow the output to the
    model's own task columns. That is a correctness measure before it is a speed
    one: the physicochemical block carries ``QED`` and the alert counts, and the
    DrugBank block carries percentiles on 0-100. None of those are probabilities,
    and all of them would otherwise arrive in the same frame as the ones that are.
    """
    global _ADMET_MODEL
    with _ADMET_MODEL_LOCK:
        if _ADMET_MODEL is None:
            try:
                from admet_ai import ADMETModel  # type: ignore[import-not-found]
            except ImportError as error:
                raise SafetyBackendUnavailableError(f"admet_ai_unavailable:{error}") from error
            _ADMET_MODEL = ADMETModel(include_physchem=False, drugbank_path=None)
        return _ADMET_MODEL


def reset_admet_model() -> None:
    """Drop the cached ADMET-AI model, releasing its ensembles. Safe to call at any time."""
    global _ADMET_MODEL
    with _ADMET_MODEL_LOCK:
        _ADMET_MODEL = None


def _as_probability(endpoint: str, raw: object) -> float:
    """``raw`` as a probability, or a refusal. A value off 0-1 is reported, never clamped."""
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{endpoint} returned a non-number: {raw!r}") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{endpoint} returned {value}, which is not a probability on 0-1")
    return value


def _rows_by_smiles(frame: Any) -> dict[str, dict[str, Any]]:
    """ADMET-AI's prediction frame as ``{smiles: {endpoint: value}}``.

    A list of SMILES gives back a DataFrame indexed by the SMILES it could parse,
    so the index is the join key. Position is not: ADMET-AI silently drops
    molecules RDKit cannot read, and lining rows up by position after a drop would
    attribute one molecule's predictions to another.
    """
    to_dict = getattr(frame, "to_dict", None)
    if to_dict is None:
        raise SafetyBackendUnavailableError(f"admet_ai_returned_unreadable_predictions:{type(frame).__name__}")
    return {str(key): dict(row) for key, row in to_dict(orient="index").items()}


def _parsable(texts: Sequence[str]) -> list[str]:
    """The SMILES RDKit can read, in the order given.

    ADMET-AI filters unreadable molecules itself, but a batch with *nothing* left
    after that filter does not come back empty: its dataloader has zero length,
    ``trainer.predict`` returns ``None`` and torch raises ``TypeError: cat():
    argument 'tensors' ... must be tuple of Tensors, not NoneType`` from inside
    the ensemble. Observed against admet-ai 2.0.1 on 2026-11-17. So the filtering
    happens here, before the model is handed anything, where the reason has a name.

    Called only after :func:`_admet_model` has succeeded, which is what makes the
    import safe: admet-ai requires ``rdkit>=2025.9.5``, so a loaded model implies a
    usable RDKit.
    """
    try:
        from rdkit import Chem
    except ImportError as error:  # pragma: no cover - admet-ai depends on rdkit
        raise LigandUnreadableError(f"rdkit_unavailable:{error}") from error
    return [text for text in texts if Chem.MolFromSmiles(text) is not None]


def admet_ai_predictions_many(smiles: Sequence[str]) -> dict[str, PanelProbabilities]:
    """ADMET-AI readings for several molecules, keyed by SMILES. Raises if ADMET-AI is absent.

    Batched because loading the model dominates the cost and the per-molecule work
    beside it is small. Duplicate SMILES are collapsed.

    A SMILES RDKit cannot parse is **absent from the result** rather than present
    with empty values, the same choice :func:`to_toxicology_rows` makes: a caller
    that needs to know compares the returned keys with what it passed. Those are
    filtered out before the model is called (see :func:`_parsable`), so a batch
    with nothing readable in it returns ``{}`` instead of reaching the ensemble.
    An empty or whitespace-only SMILES is a caller error and raises instead.
    """
    texts: list[str] = []
    seen: set[str] = set()
    for item in smiles:
        text = (item or "").strip()
        if not text:
            raise LigandUnreadableError("empty_smiles")
        if text not in seen:
            seen.add(text)
            texts.append(text)
    if not texts:
        return {}

    model = _admet_model()
    readable = _parsable(texts)
    if not readable:
        return {}
    rows = _rows_by_smiles(model.predict(smiles=readable))
    columns = set(next(iter(rows.values()), {}))
    absent = sorted(name for name in ADMET_AI_ENDPOINTS.values() if name not in columns)
    if rows and len(absent) == len(ADMET_AI_ENDPOINTS):
        raise SafetyBackendUnavailableError(
            "admet_ai_returned_no_panel_endpoint: none of "
            f"{sorted(ADMET_AI_ENDPOINTS.values())} is in its output, "
            "so its model ensembles are not loaded"
        )

    notes = (
        "only the ADMET-AI classification endpoints on the allowlist are read; its "
        "regressions and physicochemical columns are never read as a probability",
        "a probability is not a concentration and is never written to an *_nm column",
    )
    if absent:
        notes += (f"endpoint(s) missing from this ADMET-AI build: {', '.join(absent)}",)

    readings: dict[str, PanelProbabilities] = {}
    for text in texts:
        row = rows.get(text)
        if row is None:
            continue
        probabilities: dict[str, float] = {}
        endpoints: dict[str, str] = {}
        for axis, endpoint in ADMET_AI_ENDPOINTS.items():
            if endpoint not in row:
                continue
            probabilities[f"{axis}_prob"] = _as_probability(endpoint, row[endpoint])
            endpoints[axis] = endpoint
        readings[text] = PanelProbabilities(
            smiles=text,
            model="ADMET-AI",
            probabilities=probabilities,
            notes=notes,
            method=ADMET_AI_METHOD,
            endpoints=endpoints,
            uncovered_axes=ADMET_AI_UNCOVERED_AXES,
        )
    return readings


def admet_ai_predictions(smiles: str) -> PanelProbabilities:
    """ADMET-AI classification probabilities for one molecule. Raises if ADMET-AI is absent.

    ADMET-AI (Swanson et al., MIT) predicts 41 endpoints. Only the seven on
    :data:`ADMET_AI_ENDPOINTS` are read, each under an ``<axis>_prob`` key with
    ``unit="probability"``. Nothing is converted into nM -- its regression
    endpoints are different quantities in different units -- and
    :data:`ADMET_AI_UNCOVERED_AXES` records the three panel axes it cannot answer.

    For more than one molecule call :func:`admet_ai_predictions_many`, which loads
    the model once for the whole batch.
    """
    text = (smiles or "").strip()
    if not text:
        raise LigandUnreadableError("empty_smiles")
    reading = admet_ai_predictions_many([text]).get(text)
    if reading is None:
        raise LigandUnreadableError(f"unparsable_smiles:{text}")
    return reading


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
