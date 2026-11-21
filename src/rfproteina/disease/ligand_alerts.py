"""Structure alerts and drug-likeness for a small-molecule design, from RDKit alone.

``toxicology.py`` reads measured or predicted affinities for the safety panel
and will not accept a number without a unit. That leaves a gap: a molecule can
clear the panel and still carry a substructure every medicinal chemist would
refuse. This module fills that gap with RDKit's own filter catalogues, which
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
  ``FilterCatalogParams``. A hit means the scaffold is a frequent hitter in
  assay panels, not that the molecule is toxic.
* BRENK: Brenk et al. 2008, unwanted fragments for lead-like libraries.
* NIH: the NIH annotated set of reactive and undesirable groups bundled with
  RDKit.

Every bar below is marked for what it is. The alert bars are conventions of the
cited papers; the property windows are suggested defaults pending calibration
against whatever chemistry this program actually works on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

METHOD = "rdkit_filtercatalog_v1"

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
        "source": "Lipinski et al. 1997 rule of five, ClogP <= 5",
    },
    # Lipinski states both of these as atom counts, not as a SMARTS perception:
    # donors are "the sum of OHs and NHs" and acceptors "the sum of Ns and Os".
    # RDKit's NumHDonors/NumHAcceptors are a different scale, and reading the
    # published 5 and 10 against them let ritonavir clear the acceptor bar its
    # own count fails -- 11 Ns and Os, which RDKit perceives as 9 acceptors.
    # NHOHCount and NOCount are the counts the rule was stated on.
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

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["alerts"] = [dict(item) for item in self.alerts]
        record["notes"] = list(self.notes)
        record["property_bars"] = PROPERTY_BARS
        record["alert_count"] = len(self.alerts)
        return record


class LigandUnreadableError(Exception):
    """RDKit is missing, or the SMILES does not parse. Never guessed around."""


def _catalog() -> object:
    from rdkit.Chem import FilterCatalog

    params = FilterCatalog.FilterCatalogParams()
    for name in CATALOGUES:
        entry = getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, name, None)
        if entry is None:  # pragma: no cover - depends on the RDKit build
            continue
        params.AddCatalog(entry)
    return FilterCatalog.FilterCatalog(params)


def _sa_score(mol: object) -> tuple[float | None, str | None]:
    """RDKit's contributed synthetic accessibility score, or a reason it is absent."""
    try:
        import os
        import sys

        from rdkit.Chem import RDConfig

        contrib = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if contrib not in sys.path:
            sys.path.append(contrib)
        import sascorer  # type: ignore[import-not-found]
    except Exception as error:  # any import failure is the same answer
        return None, f"sascorer_unavailable:{type(error).__name__}"
    try:
        return round(float(sascorer.calculateScore(mol)), 3), None
    except Exception as error:
        return None, f"sascorer_failed:{type(error).__name__}"


def screen_smiles(smiles: str) -> LigandAlerts:
    """Run the alert catalogues and the property set over one SMILES string."""
    text = (smiles or "").strip()
    if not text:
        raise LigandUnreadableError("empty_smiles")
    try:
        from rdkit import Chem
        from rdkit.Chem import QED, Crippen, Descriptors
    except ImportError as error:  # pragma: no cover - rdkit is pinned in pyproject
        raise LigandUnreadableError(f"rdkit_unavailable:{error}") from error
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise LigandUnreadableError("smiles_unparsable")

    catalog = _catalog()
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
        smiles=text,
        alerts=tuple(hits),
        mol_weight=round(float(Descriptors.MolWt(mol)), 3),
        clogp=round(float(Crippen.MolLogP(mol)), 3),
        tpsa=round(float(Descriptors.TPSA(mol)), 3),
        # NHOHCount / NOCount, not NumHDonors / NumHAcceptors: see the cuts above.
        h_bond_donors=int(Descriptors.NHOHCount(mol)),
        h_bond_acceptors=int(Descriptors.NOCount(mol)),
        rotatable_bonds=int(Descriptors.NumRotatableBonds(mol)),
        heavy_atoms=int(mol.GetNumHeavyAtoms()),
        qed=round(float(QED.qed(mol)), 4),
        sa_score=sa_value,
        sa_score_reason=sa_reason,
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
    record = asdict(screened)
    out: dict[str, dict[str, object]] = {
        "structure_alerts": {
            "value": len(screened.alerts),
            "state": "hit" if screened.alerts else "clear",
            "threshold": 0,
            "direction": "max",
            "threshold_source": "program convention: any PAINS, BRENK or NIH catalogue match is reported as a hit",
        }
    }
    for name, rule in PROPERTY_BARS.items():
        value = record.get(name)
        minimum = rule.get("min")
        maximum = rule.get("max")
        if value is None:
            state = "untested"
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
