"""Private CPU adapter for bounded Dimorphite-DL v2.1.0 enumeration."""

from __future__ import annotations

import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from math import isfinite, prod
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from dimorphite_dl.mol import MoleculeRecord
    from dimorphite_dl.protonate.site import ProtonationSite
    from loguru import Message
    from rdkit.Chem import Mol

VERSION = "2.1.0"
PRECISION = 1.0
MAX_MOLECULES = 32
MAX_VARIANTS = 32
MAX_SMILES = 512
MAX_REQUEST_BYTES = 64 * 1024

_PROVIDER_FAILURE = "dimorphite_provider_failure"
_VARIANT_REJECTED = "dimorphite_variant_rejected"
_VARIANT_LIMIT = "variant_budget_exceeded"
_NO_SITES = "no_recognized_sites"


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("duplicate request member")
    return value


def _valid_ph(value: object) -> bool:
    return type(value) in {int, float} and 0 <= value <= 14 and isfinite(value)


def _valid_smiles(value: object) -> bool:
    return (
        isinstance(value, str) and 0 < len(value) <= MAX_SMILES
        and all(33 <= ord(char) <= 126 for char in value)
        and not any(char in value for char in ".|")
    )


def _request(data: bytes) -> tuple[float, float, list[str]]:
    if not 0 < len(data) <= MAX_REQUEST_BYTES:
        raise ValueError("request size out of range")
    value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(value, dict) or value.keys() != {"ph_min", "ph_max", "smiles"}:
        raise ValueError("invalid request members")
    minimum, maximum, molecules = value["ph_min"], value["ph_max"], value["smiles"]
    if not (_valid_ph(minimum) and _valid_ph(maximum) and minimum <= maximum):
        raise ValueError("pH range out of bounds")
    if not (
        isinstance(molecules, list) and 0 < len(molecules) <= MAX_MOLECULES
        and all(_valid_smiles(smiles) for smiles in molecules) and len(set(molecules)) == len(molecules)
    ):
        raise ValueError("invalid ligand batch")
    return float(minimum), float(maximum), molecules


def _run(ph_min: float, ph_max: float, molecules: list[str]) -> dict[str, dict[str, str | list[str]]]:
    from rfproteina.disease.dimorphite import rules_verified

    # The provider can configure stdout/stderr and open a file when imported.
    for key in ("DIMORPHITE_DL_LOG", "DIMORPHITE_DL_LOG_LEVEL", "DIMORPHITE_DL_STDOUT", "DIMORPHITE_DL_LOG_FILE_PATH"):
        os.environ.pop(key, None)
    if not rules_verified():
        raise ValueError("unexpected Dimorphite-DL distribution or rule table")

    import dimorphite_dl
    from dimorphite_dl.protonate.run import Protonate
    from loguru import logger
    from rdkit import Chem, rdBase

    if dimorphite_dl.__version__ != VERSION:
        raise ValueError("unexpected Dimorphite-DL module")

    def branch_count(site: ProtonationSite) -> int:
        return prod(len(state.get_charges()) for _, state in site.get_unique_states(ph_min, ph_max, PRECISION))

    class BoundedProtonate(Protonate):
        """Keep pinned v2.1.0 chemistry and fail closed at its silent recovery points."""

        blocker: str | None = None

        def refuse(self, blocker: str) -> NoReturn:
            self.blocker = blocker
            raise ValueError(blocker)

        def _generate_protonated_variants(self, mol_record: MoleculeRecord, sites: list[ProtonationSite]) -> list[Mol]:
            combinations = 1
            for site in sites:
                branches = branch_count(site)
                if branches < 1:
                    self.refuse(_VARIANT_REJECTED)
                combinations *= branches
                if combinations > MAX_VARIANTS:
                    self.refuse(_VARIANT_LIMIT)
            # Upper-bound the unmerged search space so no state is removed by
            # upstream mid-enumeration pruning, even before final deduplication.
            return super()._generate_protonated_variants(mol_record, sites)

        def _protonate_single_site(
            self, molecules: list[Mol], site: ProtonationSite, original_smiles: str, site_index: int,
        ) -> list[Mol]:
            children = super()._protonate_single_site(molecules, site, original_smiles, site_index)
            if len(children) != len(molecules) * branch_count(site):
                self.refuse(_VARIANT_REJECTED)
            return children

        def _convert_molecules_to_smiles(self, molecules: list[Mol], original_smiles: str) -> list[str]:
            states = super()._convert_molecules_to_smiles(molecules, original_smiles)
            # The provider already canonicalizes and merges duplicates during
            # enumeration. A later count loss can be a failed conversion.
            if len(states) != len(molecules):
                self.refuse(_VARIANT_REJECTED)
            return states

    parser = Chem.SmilesParserParams()
    parser.parseName = False
    parser.allowCXSMILES = False

    def is_canonical(smiles: str) -> bool:
        if not _valid_smiles(smiles):
            return False
        mol = Chem.MolFromSmiles(smiles, parser)
        return mol is not None and Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True) == smiles

    noticed_problem = False

    def notice(message: Message) -> None:
        nonlocal noticed_problem
        record = message.record
        if not (
            record["function"] == "_handle_molecule_without_sites"
            and record["level"].name == "WARNING"
            and record["message"].startswith("No protonation sites found for ")
        ):
            noticed_problem = True

    # This is a dedicated subprocess. A private sink observes provider warnings
    # without writing molecule text or diagnostics into the parent process.
    logger.remove()
    sink = logger.add(notice, level="WARNING", filter="dimorphite_dl", catch=False)
    logger.enable("dimorphite_dl")
    report: dict[str, dict[str, str | list[str]]] = {}
    try:
        with rdBase.BlockLogs():
            if not all(is_canonical(smiles) for smiles in molecules):
                raise ValueError("noncanonical input")
            for smiles in molecules:
                noticed_problem = False
                try:
                    engine = BoundedProtonate(
                        smiles_input=[smiles], ph_min=ph_min, ph_max=ph_max, precision=PRECISION,
                        max_variants=MAX_VARIANTS, validate_output=True, skip_invalid=False,
                    )
                    states = engine.to_list()
                    stats = engine.get_stats()["protonation"]
                    if engine.blocker is not None:
                        report[smiles] = {"blocker": engine.blocker}
                    elif stats["variants_rejected"]:
                        report[smiles] = {"blocker": _VARIANT_REJECTED}
                    elif noticed_problem or stats["fallback_used"] or stats["molecules_processed"] != 1:
                        report[smiles] = {"blocker": _PROVIDER_FAILURE}
                    elif stats["molecules_without_sites"] == 1 and stats["molecules_with_sites"] == 0:
                        report[smiles] = {"blocker": _NO_SITES}
                    elif not (
                        stats["molecules_with_sites"] == 1 and stats["molecules_without_sites"] == 0
                        and 0 < len(states) <= MAX_VARIANTS
                        and stats["total_variants_generated"] == stats["variants_validated"] == len(states)
                        and all(is_canonical(state) for state in states) and len(set(states)) == len(states)
                    ):
                        report[smiles] = {"blocker": _VARIANT_REJECTED}
                    else:
                        report[smiles] = {"states": sorted(states)}
                except Exception:
                    report[smiles] = {"blocker": _PROVIDER_FAILURE}
    finally:
        logger.remove(sink)
        logger.disable("dimorphite_dl")
    return report


def main() -> int:
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink), redirect_stderr(sink):
            response = _run(*_request(sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)))
        print(json.dumps(response, ensure_ascii=True, allow_nan=False, separators=(",", ":")))
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
