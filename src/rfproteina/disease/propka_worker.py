"""Private, finite-output adapter for the pinned PROPKA 3.5.1 model."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import version
from importlib.resources import files
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from propka.group import Group

VERSION = "3.5.1"
PARAMETER_SHA256 = "354298a0fb5d1e0d0c7cf3ea9dea117875cd6acbcf72e2dda955578a0417d2e0"
PROTEIN_BONDS_SHA256 = "f0b03022a3840fc6135b2e366904eb0b1518aee976d6da581fcdd4bb9202338d"
MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_GROUPS = 2048


def _verify_release() -> None:
    if version("propka") != VERSION:
        raise ValueError("unexpected PROPKA distribution")
    resources = files("propka")
    checks = {"propka.cfg": PARAMETER_SHA256, "protein_bonds.json": PROTEIN_BONDS_SHA256}
    for name, expected in checks.items():
        with resources.joinpath(name).open("rb") as handle:
            data = handle.read(64 * 1024 + 1)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("unverified PROPKA chemistry resource")


class _StructuralWarning(logging.Handler):
    """Keep the fact of a provider warning without retaining its details."""

    occurred = False
    failed = False

    def emit(self, record: logging.LogRecord) -> None:
        self.occurred = True
        self.failed = self.failed or record.levelno >= logging.ERROR


def _prediction(group: Group) -> tuple[float | None, float | None]:
    if (
        not group.titratable or not group.model_pka_set or group.atom.cysteine_bridge
        or group.coupled_titrating_group or group.pka_value == 99.99
    ):
        # PROPKA uses 99.99 for a disulfide and labels a penalized covalently
        # coupled group as discarded. Neither is an ionization prediction.
        return None, None
    pka, model_pka = group.pka_value, group.model_pka
    if not all(type(value) in {int, float} and math.isfinite(value) for value in (pka, model_pka)):
        return None, None
    return float(pka), float(model_pka)


def _row(group: Group) -> dict[str, str | int | float | bool | None]:
    atom = group.atom
    if atom.type not in {"atom", "hetatm"} or type(atom.res_num) is not int:
        raise ValueError("unattributed PROPKA group")
    pka, model_pka = _prediction(group)
    return {
        "kind": "protein" if atom.type == "atom" else "ligand",
        "chain": "" if atom.chain_id == "_" else atom.chain_id.strip(),
        "residue_number": atom.res_num,
        "insertion_code": atom.icode.strip(),
        "residue_name": atom.res_name.strip(),
        "atom_name": atom.name.strip(),
        "group_type": group.type.strip(),
        "pka": pka,
        "model_pka": model_pka,
        "coupled": bool(group.non_covalently_coupled_groups or group.covalently_coupled_groups
                        or group.coupled_titrating_group),
    }


def _predict(data: bytes) -> dict[str, object]:
    if not 0 < len(data) <= MAX_INPUT_BYTES:
        raise ValueError("invalid PDB input size")
    _verify_release()
    from propka.run import single

    log = logging.getLogger("propka")
    alert = _StructuralWarning(level=logging.WARNING)
    previous_level, previous_propagate = log.level, log.propagate
    log.setLevel(logging.WARNING)
    log.propagate = False
    log.addHandler(alert)
    try:
        mol = single("receptor.pdb", stream=io.StringIO(data.decode("ascii")), write_pka=False)
    finally:
        log.removeHandler(alert)
        log.setLevel(previous_level)
        log.propagate = previous_propagate
    if alert.failed:
        raise ValueError("PROPKA reported a calculation error")
    if len(mol.conformation_names) != 1:
        raise ValueError("expected one structural conformation")
    conformation = mol.conformations[mol.conformation_names[0]]
    summary_types = set(mol.version.parameters.write_out_order)
    groups = [group for group in conformation.get_groups_for_calculations() if group.residue_type in summary_types]
    if len(groups) > MAX_GROUPS:
        raise ValueError("PROPKA group limit exceeded")
    # The original conformation preserves chain, insertion code and native
    # coupling/availability fields. The text summary rounds and can hide them.
    return {"groups": [_row(group) for group in groups], "incomplete_input": alert.occurred}


def main() -> int:
    """Read bounded private PDB bytes and emit only the attributed group report."""
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink), redirect_stderr(sink):
            response = _predict(sys.stdin.buffer.read(MAX_INPUT_BYTES + 1))
        print(json.dumps(response, allow_nan=False, separators=(",", ":")))
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
