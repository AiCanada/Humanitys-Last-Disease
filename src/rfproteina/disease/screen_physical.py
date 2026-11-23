"""Independent analytical measurements on finished candidate files."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING

from rfproteina.disease import chemical, modality, screen
from rfproteina.disease.validation_checks import ValidationCandidate, _candidates, _inventory

if TYPE_CHECKING:
    from biotite.structure import AtomArray

SEQUENCE = {"gravy", "instability", "protein_composition", "protein_charge", "aggrescan"}
COORDINATES = {"psea", "clashes", "bond_length"}
METHODS = SEQUENCE | COORDINATES | {"ligand_composition", "sequence_diversity"}
CHAIN_PLANNING_LIMIT = 16 * 1024 * 1024
POLYPEPTIDE_WARNINGS = {
    "protein_composition": "modified_form_requires_mass_confirmation",
    "protein_charge": "modified_form_requires_charge_confirmation",
    "gravy": "hydropathy_of_unmodified_residues",
    "instability": "dipeptide_index_of_unmodified_residues",
    "aggrescan": "aggregation_scale_of_unmodified_residues",
}


@dataclass(frozen=True)
class Polypeptide:
    sequence: str
    modified: bool


def _polypeptide_input(text: str, chain: str) -> Polypeptide | None:
    """Resolve all entity records for the selected strand before interpreting one-letter residue codes."""
    ragged: list[list[str]] = []
    rows = screen._nonloop_category(text, "_entity_poly.")
    for columns, records in screen._cif_loops(text, ragged=ragged, prefixes=("_entity_poly.",)):
        if screen._STRAND in columns:
            rows.extend(records)
    if any(screen._STRAND in columns for columns in ragged):
        return None
    strands = {}
    for row in rows:
        for strand in screen._strands(row.get(screen._STRAND, "")):
            strands.setdefault(strand, []).append(row)
    name = chain if chain in strands else chain.split("_", 1)[0]
    sequences = set()
    modified = False
    for row in strands.get(name, []):
        kind = row.get("_entity_poly.type", "").casefold()
        if kind not in {"", ".", "?", "polypeptide(l)", "polypeptide(d)"}:
            return None
        raw = "".join(row.get(screen._ONE_LETTER, "").split()).upper()
        sequence = "".join(row.get(screen._CANONICAL, raw).split()).upper()
        if not sequence or set(sequence) - modality._STANDARD:
            return None
        sequences.add(sequence)
        modified |= raw not in {"", ".", "?", sequence}
    if len(sequences) != 1:
        return None
    return Polypeptide(sequences.pop(), modified or bool(screen._disulfide_partners(text, name)))


def _file(candidate: ValidationCandidate) -> Path:
    return max(candidate.files, key=lambda p: screen._DESIGN_SUFFIXES.index(screen._design_suffix(p)))


class Structures:
    def __init__(self, directory: Path, candidate: ValidationCandidate | None = None):
        self.directory = directory
        self.candidate = candidate

    @cached_property
    def candidates(self) -> tuple[ValidationCandidate, ...]:
        return _candidates(_inventory(self.directory))

    @cached_property
    def paths(self) -> dict[str, Path]:
        paths = {}
        for item in (self.candidate,) if self.candidate else self.candidates:
            if not item.example_id:
                raise ValueError("The structure files could not be assigned unambiguous candidate IDs.")
            path = _file(item)
            if path.resolve() != path.absolute() or not path.is_file():
                raise ValueError("A candidate structure must be a regular file within the run.")
            paths[item.example_id] = path
        return paths

    @cached_property
    def chain_ids(self) -> dict[str, str | None]:
        from rfproteina.disease.screen_inventory import Evidence

        path = screen._metrics_path(self.directory)
        rows = screen._metric_rows(self.directory) if path is not None and Evidence.safe_table(path) else []
        bindings, _ = screen._design_rows(rows, [_file(c) for c in self.candidates])
        output = {}
        for name, path in self.paths.items():
            row, reason = bindings[path]
            chain, _ = screen._interest_for_row(row, rows)
            output[name] = None if reason else chain
        return output

    @cached_property
    def chains(self) -> list[tuple[str, str | None]]:
        return [(name, screen._designed_sequence(self.paths[name], chain)) for name, chain in self.chain_ids.items()]

    def polypeptides(self, max_bytes: int | None = None) -> dict[str, Polypeptide | None]:
        result = {}
        for name, chain in self.chain_ids.items():
            text = screen._design_text(self.paths[name], max_bytes=max_bytes) if chain else None
            result[name] = _polypeptide_input(text, chain) if text and chain else None
            if text and max_bytes is not None:
                max_bytes -= len(text.encode("utf-8"))
        return result

    @cached_property
    def planned_polypeptides(self) -> dict[str, Polypeptide | None]:
        return self.polypeptides(CHAIN_PLANNING_LIMIT)

    @cached_property
    def coordinate_arrays(self) -> dict[str, AtomArray | None]:
        budget = CHAIN_PLANNING_LIMIT
        result = {}
        for name, path in self.paths.items():
            text = screen._design_text(path, max_bytes=budget)
            result[name] = _coordinate_atoms(path, text) if text else None
            if text:
                budget -= len(text.encode("utf-8"))
        return result

    def measure(self, key: str) -> dict:
        try:
            return self._measure(key)
        except (ImportError, OSError, KeyError, ValueError, RuntimeError, IndexError, OverflowError):
            return {"method": key, "status": "untested", "blocker": "required_structure_or_measurement_unavailable"}

    def _measure(self, key: str) -> dict:
        if key == "sequence_diversity":
            from rfproteina.disease.diversity import design_diversity

            polymers = self.polypeptides()
            pairs = [(name, polymer.sequence if polymer else None) for name, polymer in polymers.items()]
            diversity = design_diversity(pairs)
            if diversity is None:
                return {"method": key, "status": "untested", "blocker": "requires_two_readable_designed_chains"}
            status = "partial" if diversity.get("unreadable_designs") else "reported"
            return {"method": key, "status": status, "cohort": diversity}
        measurements = []
        if key in POLYPEPTIDE_WARNINGS:
            for name, polymer in self.polypeptides().items():
                values = _sequence(key, polymer.sequence if polymer else None)
                if values and polymer.modified:
                    values["input_warning"] = POLYPEPTIDE_WARNINGS[key]
                measurements.append({"example_id": name, **values})
        elif key in SEQUENCE:
            for name, chain in self.chains:
                values = _sequence(key, chain)
                measurements.append({"example_id": name, **values})
        else:
            for name, path in self.paths.items():
                try:
                    values = _structure(key, path, self.chain_ids[name] if key == "psea" else None)
                except (ImportError, OSError, KeyError, ValueError, RuntimeError, IndexError):
                    values = {}
                measurements.append({"example_id": name, **values})
        measured = sum(len(row) > 1 for row in measurements)
        for row in measurements:
            if len(row) == 1:
                row["blocker"] = "required_structure_or_measurement_unavailable"
        status = "untested"
        if measured:
            status = "reported" if measured == len(measurements) else "partial"
        return {
            "method": key,
            "status": status,
            "measured": measured,
            "attempted": len(measurements),
            "samples": measurements,
        }


def _sequence(key: str, chain: str | None) -> dict:
    if not chain or set(chain) - set("ACDEFGHIKLMNPQRSTVWY"):
        return {}
    if key == "protein_composition":
        return {
            "average_molecular_mass_Da": modality._molecular_weight(chain),
            "residue_counts": modality._residue_composition(chain),
            "mass_basis": "unmodified_linear_chain",
        }
    if key == "protein_charge":
        return {
            "net_charge_at_pH_7_4": chemical.net_charge(chain, chemical.PH74),
            "pka_set": chemical.PKA_SET,
            "charge_basis": "unmodified_linear_chain",
            **chemical._isoelectric_block(chain),
        }
    if key == "aggrescan":
        result = modality._aggrescan(chain)
        if result is None:
            return {}
        return {
            "normalized_aggregation_score": result["na4vss"],
            "aggregation_method": result["method"],
            "window_residues": result["window"],
        }
    from rfproteina.utils.sequence_properties import compute_gravy, compute_instability_index

    value = compute_gravy(chain) if key == "gravy" else compute_instability_index(chain)
    return {key: float(value)} if math.isfinite(value) else {}


def _coordinate_atoms(path: Path, text: str) -> AtomArray | None:
    lower = text.lower()
    if not all("_atom_site.cartn_" + axis in lower for axis in "xyz"):
        return None
    from biotite.structure import BadStructureError

    try:
        return screen._load_generated_structure(path, cif_text=text)
    except (BadStructureError, TypeError, RuntimeError, OverflowError):
        return None


def psea_chain(atoms: AtomArray, requested: str | None) -> tuple[AtomArray | None, int]:
    """Identify the protein chain and its longest observed stretch of consecutive C-alpha residues."""
    import numpy as np
    from biotite.structure import filter_amino_acids

    ca = filter_amino_acids(atoms) & (atoms.atom_name == "CA") & np.isfinite(atoms.coord).all(axis=1)
    if not ca.any():
        return None, 0
    labels = atoms.chain_iid if "chain_iid" in atoms.get_annotation_categories() else atoms.chain_id
    available = set(labels[ca])
    if requested:
        chain = requested if requested in available else requested.split("_", 1)[0]
    else:
        chain = next(iter(available)) if len(available) == 1 else None
    if chain not in available:
        return None, 0
    selected = labels == chain
    residues = atoms.res_id[ca & selected]
    # Biotite returns coil by convention for at most five residues.
    breaks = np.flatnonzero(np.diff(residues) != 1) + 1
    longest = int(np.diff(np.r_[0, breaks, len(residues)]).max())
    return atoms[selected], longest


def has_clash_pair(atoms: AtomArray) -> bool:
    """Require at least one pair that the native clash measurement could count."""
    import numpy as np
    from atomworks.ml.conditions import C_CRD

    n = atoms.array_length()
    if n < 2 or not np.isfinite(atoms.coord).all():
        return False
    fixed = C_CRD.mask(atoms, default="generate")
    n_fixed = int(np.count_nonzero(fixed))
    pairs = n * (n - 1) // 2 - n_fixed * (n_fixed - 1) // 2
    if pairs and atoms.bonds is not None:
        bonds = atoms.bonds.as_array()
        pairs -= int(np.count_nonzero(~(fixed[bonds[:, 0]] & fixed[bonds[:, 1]])))
    return pairs > 0


def has_measurable_bonds(atoms: AtomArray) -> bool:
    import numpy as np

    if atoms.bonds is None or atoms.bonds.get_bond_count() == 0:
        return False
    endpoints = atoms.bonds.as_array()[:, :2]
    return bool(np.isfinite(atoms.coord[endpoints]).all())


def _structure(key: str, path: Path, chain: str | None = None) -> dict:
    if key == "ligand_composition":
        text = screen._design_text(path)
        return modality.identified_component_weight(screen._chem_components(text)) if text else {}
    text = screen._design_text(path)
    atoms = _coordinate_atoms(path, text) if text else None
    if atoms is None:
        return {}
    if key == "psea":
        from rfproteina.metrics.sse_metrics import compute_sse_distribution

        atoms, span = psea_chain(atoms, chain)
        if atoms is None:
            return {}
        values = compute_sse_distribution(atoms)
        values = {k: v for k, v in values.items() if k in ("helix_fraction", "sheet_fraction", "loop_fraction")}
        values = {k: float(v) for k, v in values.items() if math.isfinite(v)}
        if values:
            values["longest_consecutive_ca"] = span
            if span <= 5:
                values["input_warning"] = "short_or_discontinuous_ca_trace"
        return values
    elif key == "bond_length":
        from rfproteina.metrics.design_metrics import compute_bond_length_stats

        if not has_measurable_bonds(atoms):
            return {}
        values = compute_bond_length_stats(atoms)
    else:
        from rfproteina.metrics.design_metrics import compute_clash_stats

        if not has_clash_pair(atoms):
            return {}
        values = compute_clash_stats(atoms, threshold=screen._CLASH_THRESHOLD)
    return {k: float(v) for k, v in values.items() if isinstance(v, Real) and math.isfinite(v)}
