"""Enumerate and rank phase-ordered metabolite hypotheses with the released SyGMa rules."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdkit import Chem
    from rdkit.Chem.rdChemReactions import ChemicalReaction

METHOD = "sygma_1_1_0_empirical_rules"
ENDPOINT = "human_metabolite_hypotheses"
ARCHIVE_NAME = "SyGMa-1.1.0.tar.gz"
ARCHIVE_URL = "https://files.pythonhosted.org/packages/01/0d/7960ca6076d00ff33bf2aa0ac925074a07820900b8b8ae1edae83d12e488/SyGMa-1.1.0.tar.gz"
ARCHIVE_HASH = "7daf933862c25851176086ee15ef9210bc1b73f7c0d9dc26e68be73c9c47b244"
RULE_HASHES = {
    "phase1": "251635c5d7ef13a636561b0a0a07f289d46aa094e71cd8ec39edc8e069527ff6",
    "phase2": "f4a4a9be5f55bdd6f85ff611c417ecf88880011477af8d724bf0e93b6cacbbb3",
}
MAX_ROUNDS = 4
MAX_STATES = 2048
MAX_MATCHES = 1024
MAX_OUTCOMES = 25000


class ScreenError(ValueError):
    """A named condition that prevents a complete per-parent ranking."""


def archive_path() -> Path:
    return Path(
        os.environ.get("SYGMA_RULE_ARCHIVE", Path.home() / ".local/share/rfproteina/sygma" / ARCHIVE_NAME)
    ).expanduser()


def read_archive(data: bytes) -> dict[str, bytes]:
    if hashlib.sha256(data).hexdigest() != ARCHIVE_HASH:
        raise ScreenError("sygma_archive_digest_mismatch")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        return {phase: archive.extractfile(f"SyGMa-1.1.0/sygma/rules/{phase}.txt").read() for phase in RULE_HASHES}


def scenario_valid(phase1: object, phase2: object) -> bool:
    return (
        type(phase1) is int and type(phase2) is int and min(phase1, phase2) >= 0 and 1 <= phase1 + phase2 <= MAX_ROUNDS
    )


def structure_id(smiles: str) -> str:
    return "smiles_sha256_" + hashlib.sha256(smiles.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class Rule:
    phase: str
    line: int
    name: str
    weight: Decimal
    reaction: ChemicalReaction


def load_rules() -> dict[str, list[Rule]]:
    from rdkit.Chem import rdChemReactions

    path = archive_path()
    if path.exists() or path.is_symlink() or "SYGMA_RULE_ARCHIVE" in os.environ:
        try:
            with path.open("rb") as file:
                tables = read_archive(file.read(1024 * 1024))
        except OSError as error:
            raise ScreenError("sygma_rule_archive_unavailable") from error
    else:
        try:
            release = metadata.distribution("SyGMa")
            if release.version != "1.1.0":
                raise ScreenError("unsupported_sygma_release")
            tables = {
                phase: Path(release.locate_file(f"sygma/rules/{phase}.txt")).read_bytes() for phase in RULE_HASHES
            }
        except (metadata.PackageNotFoundError, OSError) as error:
            raise ScreenError("sygma_rules_unavailable") from error
    rules = {}
    for phase, data in tables.items():
        if hashlib.sha256(data).hexdigest() != RULE_HASHES[phase]:
            raise ScreenError("sygma_rules_digest_mismatch")
        rules[phase] = []
        for number, line in enumerate(data.decode("ascii").splitlines(), 1):
            if not line.strip() or line.startswith("#"):
                continue
            smarts, weight, name, _ = line.split("\t")
            reaction = rdChemReactions.ReactionFromSmarts(smarts)
            if reaction is None or reaction.GetNumReactantTemplates() != 1 or reaction.GetNumProductTemplates() != 1:
                raise ScreenError("unsupported_sygma_rule")
            reaction.Initialize()
            rules[phase].append(Rule(phase, number, name.strip(), Decimal(weight), reaction))
    return rules


@dataclass(frozen=True)
class Step:
    rule: Rule
    product_id: str

    def record(self) -> dict:
        return {
            "phase": self.rule.phase,
            "rule_line": self.rule.line,
            "rule": self.rule.name,
            "empirical_rule_score": float(self.rule.weight),
            "product_id": self.product_id,
        }


@dataclass
class Route:
    mol: Chem.Mol
    smiles: str
    origin: tuple[bool, ...]
    score: Decimal
    steps: tuple[Step, ...]

    def better_than(self, old: Route | None) -> bool:
        if old is None or self.score != old.score:
            return old is None or self.score > old.score

        def order(steps: tuple[Step, ...]) -> tuple:
            return len(steps), tuple((s.rule.phase, s.rule.line, s.product_id) for s in steps)

        return order(self.steps) < order(old.steps)


def _lineage_key(route: Route) -> str:
    """Distinguish identical graphs reached with different surviving parent atoms."""
    from rdkit import Chem

    marked = Chem.Mol(route.mol)
    for atom, original in zip(marked.GetAtoms(), route.origin, strict=True):
        atom.SetAtomMapNum(int(original))
    return Chem.MolToSmiles(marked)


def _products(parent: Route, rule: Rule, counts: dict) -> Iterator[Route]:
    from rdkit import Chem

    matches = rule.reaction.RunReactants((parent.mol,), maxProducts=MAX_MATCHES + 1)
    if len(matches) > MAX_MATCHES:
        raise ScreenError("sygma_rule_match_limit")
    counts["reaction_matches"] += len(matches)
    if counts["reaction_matches"] > MAX_OUTCOMES:
        raise ScreenError("sygma_reaction_match_budget")
    for outcome in matches:
        # All 175 released rules use one product template. Some contain
        # multiple disconnected fragments within that template.
        for fragment in Chem.GetMolFrags(outcome[0], asMols=True, sanitizeFrags=False):
            try:
                Chem.SanitizeMol(fragment)
                Chem.AssignStereochemistry(fragment, cleanIt=True, force=True)
            except ValueError:
                counts["unsanitizable_fragments"] += 1
                continue
            origin = []
            for atom in fragment.GetAtoms():
                inherited = False
                if atom.HasProp("react_atom_idx"):
                    previous = atom.GetIntProp("react_atom_idx")
                    if not 0 <= previous < len(parent.origin):
                        raise ScreenError("sygma_atom_lineage_invalid")
                    inherited = parent.origin[previous] and atom.GetAtomicNum() > 1
                origin.append(inherited)
                for prop in ("react_atom_idx", "old_mapno", "react_idx"):
                    if atom.HasProp(prop):
                        atom.ClearProp(prop)
            smiles = Chem.MolToSmiles(fragment)
            yield Route(
                fragment,
                smiles,
                tuple(origin),
                parent.score * rule.weight,
                (*parent.steps, Step(rule, structure_id(smiles))),
            )


def _describe(mol: Chem.Mol) -> dict:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    key = Chem.MolToInchiKey(mol)
    if not key:
        raise ScreenError("sygma_product_identity_unavailable")
    return {
        "inchikey": key,
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol, separateIsotopes=True),
        "exact_mass_da": rdMolDescriptors.CalcExactMolWt(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
        "unspecified_stereoelements": sum(
            s.specified != Chem.StereoSpecified.Specified for s in Chem.FindPotentialStereo(mol)
        ),
    }


def _enumerate(smiles: str, phase1: int, phase2: int, rules: dict[str, list[Rule]]) -> dict:
    from rdkit import Chem

    if not scenario_valid(phase1, phase2):
        raise ScreenError("invalid_sygma_scenario")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or Chem.MolToSmiles(mol) != smiles:
        raise ScreenError("sygma_parent_identity_mismatch")
    root = Route(mol, smiles, tuple(a.GetAtomicNum() > 1 for a in mol.GetAtoms()), Decimal(1), ())
    total = sum(root.origin)
    states = {_lineage_key(root): root}
    counts = {"reaction_matches": 0, "unsanitizable_fragments": 0}
    for phase, cycles in (("phase1", phase1), ("phase2", phase2)):
        # States can skip a cycle or a whole phase. Only previous-round paths
        # can supply a reactant, and every rule used must belong to this phase.
        frontier = list(states.values())
        for _ in range(cycles):
            following = {}
            for parent in frontier:
                for rule in rules[phase]:
                    for route in _products(parent, rule, counts):
                        key = _lineage_key(route)
                        if route.better_than(states.get(key)):
                            states[key] = following[key] = route
                            if len(states) > MAX_STATES:
                                raise ScreenError("sygma_state_budget")
            frontier = list(following.values())
            if not frontier:
                break
    best = {}
    distinct = {r.smiles for r in states.values()} - {smiles}
    for route in states.values():
        if (
            route.smiles != smiles
            and 100 * sum(route.origin) > 15 * total
            and route.better_than(best.get(route.smiles))
        ):
            best[route.smiles] = route
    ordered = sorted(best.values(), key=lambda r: (-r.score, r.smiles))
    structures = {structure_id(smiles): smiles}
    products = []
    last_score = None
    for index, route in enumerate(ordered, 1):
        if route.score != last_score:
            rank = index
            last_score = route.score
        key = structure_id(route.smiles)
        products.append(
            {
                "method": METHOD,
                "product_id": key,
                "rank": rank,
                "path_score": float(route.score),
                "parent_heavy_atoms_retained": sum(route.origin),
                "pathway": [s.record() for s in route.steps],
                **_describe(route.mol),
            }
        )
        structures[key] = route.smiles
    # Include structures used as intermediates in the reported pathways.
    needed = {step["product_id"] for row in products for step in row["pathway"]} - structures.keys()
    if needed:
        structures.update((structure_id(s), s) for s in sorted(distinct) if structure_id(s) in needed)
    return {
        "status": "ranked" if products else "no_products",
        "products": products,
        "structures": structures,
        "parent_heavy_atoms": total,
        "size_filtered_structures": len(distinct) - len(best),
        "counts": counts,
    }


def enumerate_products(smiles: str, phase1: int, phase2: int) -> dict:
    """Compute a complete ranking using the two verified release tables."""
    from rdkit import rdBase

    rules = load_rules()
    result = _enumerate(smiles, phase1, phase2, rules)
    result["provider"] = {
        "rule_release": "1.1.0",
        "rdkit": rdBase.rdkitVersion,
        "rule_sha256": RULE_HASHES,
        "rules_per_phase": {k: len(v) for k, v in rules.items()},
        "device": "cpu",
    }
    return result


if __name__ == "__main__":
    request = json.load(sys.stdin)
    try:
        report = enumerate_products(request["smiles"], request["phase1"], request["phase2"])
    except ScreenError as error:
        report = {"status": "untested", "blocker": str(error), "products": []}
    except (ImportError, OSError):
        report = {"status": "untested", "blocker": "sygma_dependency_unavailable", "products": []}
    except Exception as error:
        report = {"status": "untested", "blocker": "sygma_provider_failed_" + type(error).__name__, "products": []}
    print(json.dumps({"compound_id": structure_id(request["smiles"]), **report}, allow_nan=False))
