"""Structure alerts and drug-likeness for a small-molecule design, from RDKit alone.

``toxicology.py`` reads measured or predicted affinities for the safety panel
and will not accept a number without a unit. This module flags structural
patterns for review using RDKit's filter catalogues, which
ship with the package already pinned in ``pyproject.toml``. No new dependency,
no network, no model weights.

What is deliberately NOT here:

* No predicted hERG probability. ``combinations.py`` judges the ligand panel in
  nanomolar and a classifier probability is not a concentration. Converting one
  into the other would be inventing a number. A probability belongs in its own
  column with its own bar, which nothing in this repository yet carries.
* No synthetic-accessibility score unless RDKit's contributed ``sascorer`` is
  importable in the environment; when it is missing the field is ``None`` with
  a reason, never a guess.

Alert catalogues and what they are:

* PAINS A, B and C: Baell and Holloway 2010, as encoded in RDKit's
  ``FilterCatalogParams``. A hit identifies a pattern associated with assay
  interference. Actual interference requires experimental evidence.
* BRENK: Brenk et al. 2008, unwanted fragments for lead-like libraries.
* NIH: the NIH annotated set of reactive and undesirable groups bundled with
  RDKit.

The project flags any catalogue match. Each property window identifies the
source of its cutoff and the descriptor used to calculate the value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from math import isfinite
from typing import TYPE_CHECKING

from rfproteina.disease.ligand_inputs import parse_nominated_smiles

if TYPE_CHECKING:
    from rdkit import Chem

METHOD = "rdkit_filtercatalog_v2"

#: Catalogue names requested from RDKit, in the order they are reported.
CATALOGUES = ("PAINS_A", "PAINS_B", "PAINS_C", "BRENK", "NIH")

#: Property windows. Each one says where it came from.
PROPERTY_BARS: dict[str, dict[str, object]] = {
    "mol_weight": {
        "max": 500.0,
        "source": "Lipinski et al. 1997 rule of five, molecular weight <= 500 Da",
    },
    "clogp": {
        "max": 5.0,
        "source": (
            "Lipinski et al. 1997 rule of five, ClogP <= 5. METHOD DIFFERS: the number in this "
            "record is RDKit's Wildman-Crippen atom-based logP (JCICS 39:868-873, 1999), not the "
            "CLOGP the rule was stated on; Lipinski's own cut for a different estimator is "
            "MLogP > 4.15, not 5, so which calculator produced the number matters here"
        ),
    },
    # Lipinski states both of these as atom counts, not as a SMARTS perception:
    # donors are "the sum of OHs and NHs" and acceptors "the sum of Ns and Os".
    # RDKit's NumHDonors/NumHAcceptors are a different scale, and reading the
    # published 5 and 10 against them let ritonavir clear the acceptor bar its
    # own count fails -- 11 Ns and Os, which RDKit perceives as 9 acceptors.
    # NHOHCount and NOCount are the counts the rule was stated on.
    #
    # THE TWO SCALES CROSS, so this is not a tightening. RDKit's donor and
    # acceptor SMARTS include sulfur; Lipinski's counts see only N and O, and
    # the rule of five is blind to a thiol or a thioether however well it
    # hydrogen-bonds. Measured on this build: polythiophene-11 goes from 11
    # acceptors to 0 and hexathiophenol from 6 donors to 0, both hit to clear,
    # while captopril moves in both directions at once -- acceptors 3 to 4,
    # donors 2 to 1. A sulfur-rich ligand therefore passes these two bars on
    # the published scale and must not be read as having been checked for
    # hydrogen bonding; nothing in this module screens sulfur donors.
    # This records a limit of the cited rule, not a cut of ours to move.
    "h_bond_donors": {
        "max": 5,
        "source": "Lipinski et al. 1997 rule of five, sum of OHs and NHs <= 5 (RDKit NHOHCount)",
    },
    "h_bond_acceptors": {
        "max": 10,
        "source": "Lipinski et al. 1997 rule of five, sum of Ns and Os <= 10 (RDKit NOCount)",
    },
    "qed": {
        "min": 0.5,
        "source": "suggested default pending calibration; Bickerton et al. 2012 define QED but publish no pass mark",
    },
    "sa_score": {
        "max": 6.0,
        "source": "suggested default pending calibration; Ertl and Schuffenhauer 2009 scale runs 1 (easy) to 10 (hard)",
    },
}


@dataclass(frozen=True)
class LigandAlerts:
    """One molecule, screened. ``alerts`` lists every catalogue hit by name."""

    smiles: str
    alerts: tuple[dict[str, str], ...]
    mol_weight: float | None
    clogp: float | None
    tpsa: float | None
    h_bond_donors: int | None
    h_bond_acceptors: int | None
    rotatable_bonds: int | None
    heavy_atoms: int | None
    qed: float | None
    sa_score: float | None
    sa_score_reason: str | None
    method: str = METHOD
    notes: tuple[str, ...] = field(default_factory=tuple)
    missing_catalogues: tuple[str, ...] = ()

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["alerts"] = [dict(item) for item in self.alerts]
        record["notes"] = list(self.notes)
        record["property_bars"] = PROPERTY_BARS
        record["alert_count"] = len(self.alerts) if not self.missing_catalogues or self.alerts else None
        return record


class LigandUnreadableError(Exception):
    """RDKit is missing, or the SMILES does not parse. Never guessed around."""


@lru_cache(maxsize=4)
def _catalog(names: tuple[str, ...]) -> tuple[object, tuple[str, ...]]:
    from rdkit.Chem import FilterCatalog

    params = FilterCatalog.FilterCatalogParams()
    missing = []
    for name in names:
        entry = getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, name, None)
        if entry is None or not params.AddCatalog(entry):
            missing.append(name)
    return FilterCatalog.FilterCatalog(params), tuple(missing)


def missing_catalogues() -> tuple[str, ...]:
    """Check descriptor imports and catalogue registration without scoring a molecule."""
    from rdkit.Chem import QED, Crippen, Descriptors

    functions = (
        Crippen.MolLogP,
        Descriptors.MolWt,
        Descriptors.TPSA,
        Descriptors.NHOHCount,
        Descriptors.NOCount,
        Descriptors.NumRotatableBonds,
        QED.qed,
    )
    if not all(map(callable, functions)):
        raise ImportError("rdkit_descriptors_unavailable")
    return _catalog(CATALOGUES)[1]


def _sa_score(mol: object) -> tuple[float | None, str | None]:
    """RDKit's contributed synthetic accessibility score, or a reason it is absent."""
    try:
        from rdkit.Contrib.SA_Score import sascorer
    except Exception as error:  # any import failure is the same answer
        return None, f"sascorer_unavailable:{type(error).__name__}"
    try:
        value = float(sascorer.calculateScore(mol))
        return (value, None) if isfinite(value) else (None, "sascorer_nonfinite")
    except Exception as error:
        return None, f"sascorer_failed:{type(error).__name__}"


def parse_molecule(smiles: str, *, nominated: bool) -> Chem.Mol:
    """Read a complete molecular graph on the same hydrogen convention as the descriptors."""
    text = (smiles or "").strip()
    if not text:
        raise LigandUnreadableError("empty_smiles")
    try:
        from rdkit import Chem
    except ImportError as error:  # pragma: no cover - rdkit is pinned in pyproject
        raise LigandUnreadableError(f"rdkit_unavailable:{error}") from error
    try:
        mol = parse_nominated_smiles(text, Chem, remove_hs=True) if nominated else Chem.MolFromSmiles(text)
    except (ValueError, RuntimeError, OverflowError) as error:
        raise LigandUnreadableError("smiles_unparsable") from error
    if mol is None:
        raise LigandUnreadableError("smiles_unparsable")
    if not mol.GetNumAtoms() or any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        raise LigandUnreadableError("smiles_incomplete_molecule")
    return mol


def screen_smiles(smiles: str, *, nominated: bool = False) -> LigandAlerts:
    """Run the alert catalogues and the property set over one SMILES string."""
    mol = parse_molecule(smiles, nominated=nominated)
    try:
        from rdkit.Chem import QED, Crippen, Descriptors

        catalog, missing = _catalog(CATALOGUES)
    except (ImportError, OSError) as error:
        raise LigandUnreadableError("rdkit_alert_provider_unavailable") from error
    hits: list[dict[str, str]] = []
    for match in catalog.GetMatches(mol):
        # FilterCatalogEntry has no HasProp, so the property list is asked for
        # first rather than guessed at.
        listed = set(match.GetPropList()) if hasattr(match, "GetPropList") else set()
        hits.append(
            {
                "catalogue": str(match.GetProp("FilterSet")) if "FilterSet" in listed else "unknown",
                "name": str(match.GetDescription()),
            }
        )
    hits.sort(key=lambda item: (item["catalogue"], item["name"]))

    sa_value, sa_reason = _sa_score(mol)
    return LigandAlerts(
        smiles=smiles.strip(),
        alerts=tuple(hits),
        mol_weight=float(Descriptors.MolWt(mol)),
        clogp=float(Crippen.MolLogP(mol)),
        tpsa=float(Descriptors.TPSA(mol)),
        # NHOHCount / NOCount, not NumHDonors / NumHAcceptors: see the cuts above.
        h_bond_donors=int(Descriptors.NHOHCount(mol)),
        h_bond_acceptors=int(Descriptors.NOCount(mol)),
        rotatable_bonds=int(Descriptors.NumRotatableBonds(mol)),
        heavy_atoms=int(mol.GetNumHeavyAtoms()),
        qed=float(QED.qed(mol)),
        sa_score=sa_value,
        sa_score_reason=sa_reason,
        missing_catalogues=missing,
        notes=(
            "a catalogue hit is a flag for review, not a toxicity call",
            "no predicted probability is converted into a concentration anywhere in this module",
        ),
    )


def verdicts(screened: LigandAlerts) -> dict[str, dict[str, object]]:
    """Property bars applied, plus one line for the alert catalogues.

    The alert line is ``hit`` when any catalogue matched. That is this module's
    own convention and is marked as such.
    """
    record = screened.as_record()
    out: dict[str, dict[str, object]] = {
        "structure_alerts": {
            "value": record["alert_count"],
            "state": "hit" if screened.alerts else ("untested" if screened.missing_catalogues else "clear"),
            "threshold": 0,
            "direction": "max",
            "threshold_source": "program convention: any PAINS, BRENK or NIH catalogue match is reported as a hit",
        }
    }
    if screened.missing_catalogues:
        out["structure_alerts"]["missing_catalogues"] = list(screened.missing_catalogues)
    for name, rule in PROPERTY_BARS.items():
        value = record.get(name)
        minimum = rule.get("min")
        maximum = rule.get("max")
        if value is None or not isfinite(value):
            state = "untested"
            value = None
        elif minimum is not None:
            state = "clear" if value >= minimum else "hit"
        else:
            state = "clear" if value <= maximum else "hit"
        out[name] = {
            "value": value,
            "state": state,
            "threshold": minimum if minimum is not None else maximum,
            "direction": "min" if minimum is not None else "max",
            "threshold_source": rule["source"],
        }
    return out
