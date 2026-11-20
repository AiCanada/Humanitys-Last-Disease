"""ImmuneBuilder (ABodyBuilder2) antibody-structure oracle: free, in place of IgFold.

ImmuneBuilder (https://github.com/oxpig/ImmuneBuilder) is released under the
BSD 3-Clause License, verified on 2026-11-16 from the repository's own
``LICENSE``. It is the free replacement for IgFold, whose licence is academic-use
only. ABodyBuilder2 predicts an antibody Fv structure from its heavy- and
light-chain sequences, with a per-residue error estimate derived from an
ensemble (README, 2026-11-16).

How it fits the oracle slot, honestly
-------------------------------------
Unlike AF3 / Chai / Boltz, ImmuneBuilder is **not** a confidence oracle: it emits
no pTM or ipTM, only a predicted structure and a per-residue coordinate-error
estimate (in angstroms). So this adapter produces the refolded structure and
carries ImmuneBuilder's own mean predicted error as a clearly-named scalar
(``immunebuilder_mean_predicted_error_a``) via the base class's ``_extra``
passthrough. It never fabricates a pTM/ipTM to look like the other oracles. A
reviewer deciding whether to gate on this must treat it as a structure source
plus an error estimate, not as an interface-confidence oracle.

Field provenance (verified from ImmuneBuilder ``main`` README on 2026-11-16)
---------------------------------------------------------------------------
* ``from ImmuneBuilder import ABodyBuilder2``; ``predictor.predict({'H': ..., 'L': ...})``
  returns an antibody object with ``.save(path)`` writing a PDB.
* ImmuneBuilder uses ANARCI for numbering; it must be installed alongside.

Nothing here downloads weights at import. The per-residue error attribute name is
read defensively and omitted (not guessed) if the installed version exposes it
differently.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

from rfproteina.metrics.oracles.free_oracle_common import (
    PredictOutput,
    SequenceRefoldOracle,
)
from rfproteina.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class ImmuneBuilderOracle(SequenceRefoldOracle):
    """Predict an antibody Fv structure with ABodyBuilder2 (BSD-3).

    Args:
        oracle_name: Output-filename label (default ``"immunebuilder"``).
        heavy_chain: Chain id of the heavy chain in the design. ``None`` falls
            back to the first protein chain, with a warning.
        light_chain: Chain id of the light chain. ``None`` falls back to the
            second protein chain, with a warning.
        **kwargs: Forwarded to :class:`OracleMetric`.
    """

    def __init__(
        self,
        oracle_name: str = "immunebuilder",
        *,
        heavy_chain: str | None = None,
        light_chain: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(oracle_name=oracle_name, **kwargs)  # type: ignore[arg-type]
        self.heavy_chain = heavy_chain
        self.light_chain = light_chain
        # ImmuneBuilder writes PDB, so the canonical refold file is a .pdb and
        # find_structure_file resolves it. Set here, not pretended to be CIF.
        self.output_cif_suffix = ".pdb"
        # ...and declared fixed, because both drivers assign the run's own
        # suffix to every oracle they hold. Without this the PDB above was
        # saved as ``..._refolded.cif.zst``: a PDB wearing a compressed-CIF
        # name, which every reader downstream refuses.
        self.output_suffix_is_fixed = True

    def _backend_available(self) -> bool:
        """Whether a fold could actually run, not merely whether the name imports.

        ``ImmuneBuilder`` pulls ANARCI at import, which in turn shells out to
        HMMER's ``hmmscan``. Probing the one package alone reported the backend
        available on a machine missing either, and the failure then surfaced
        per design inside ``fold_batch``'s catch-all as "fold failed ...
        skipping" -- a warning per design instead of one clear refusal for the
        batch, which is the distinction ``OracleBackendUnavailableError``
        exists to draw.
        """
        for module in ("ImmuneBuilder", "anarci", "openmm", "pdbfixer"):
            if importlib.util.find_spec(module) is None:
                return False
        return shutil.which("hmmscan") is not None

    def pick_chains(self, sequences: dict[str, str]) -> tuple[str, str]:
        """Resolve the heavy/light chain ids to fold.

        Raises:
            ValueError: If fewer than two protein chains are present and the ids
                were not given explicitly.
        """
        heavy = self.heavy_chain
        light = self.light_chain
        if heavy is None or light is None:
            available = list(sequences)
            if len(available) < 2:
                raise ValueError(
                    "ImmuneBuilder needs a heavy and a light chain; the design has "
                    f"{len(available)} protein chain(s) and heavy/light ids were not set."
                )
            if heavy is None:
                heavy = available[0]
                logger.warning(f"{self.oracle_name}: heavy_chain not set; assuming {heavy!r}")
            if light is None:
                light = available[1]
                logger.warning(f"{self.oracle_name}: light_chain not set; assuming {light!r}")
        for role, chain in (("heavy", heavy), ("light", light)):
            if chain not in sequences:
                raise ValueError(
                    f"{self.oracle_name}: {role} chain {chain!r} not among design chains {list(sequences)}"
                )
        return heavy, light

    def _predict(self, sequences: dict[str, str], work_dir: Path) -> PredictOutput:
        from ImmuneBuilder import ABodyBuilder2  # lazy: heavy import + ANARCI

        heavy, light = self.pick_chains(sequences)
        predictor = ABodyBuilder2()
        antibody = predictor.predict({"H": sequences[heavy], "L": sequences[light]})
        return self.build_output(antibody, work_dir, [heavy, light])

    @staticmethod
    def build_output(antibody: object, work_dir: Path, chain_ids: list[str]) -> PredictOutput:
        """Save the predicted structure and read its error estimate defensively.

        Args:
            antibody: ImmuneBuilder's prediction object, with a ``.save(path)``.
            work_dir: Scratch directory the PDB is written into.
            chain_ids: The [heavy, light] design chain ids, for the summary.
        """
        pdb_path = Path(work_dir) / "antibody.pdb"
        antibody.save(str(pdb_path))  # type: ignore[attr-defined]

        extra: dict[str, object] = {}
        for attr in ("error_estimates", "error_estimate", "errors"):
            value = getattr(antibody, attr, None)
            if value is None:
                continue
            try:
                flat = [float(x) for x in _flatten(value)]
            except (TypeError, ValueError):
                continue
            if flat:
                extra["immunebuilder_mean_predicted_error_a"] = round(sum(flat) / len(flat), 4)
                extra["immunebuilder_max_predicted_error_a"] = round(max(flat), 4)
                break

        summary: dict[str, object] = {"chain_ids": chain_ids, "_extra": extra}
        return PredictOutput(cif_path=pdb_path, summary_confidences=summary)


def _flatten(value: object) -> list[object]:
    """Flatten a nested list / array / tensor one or two levels for averaging."""
    tolist = getattr(value, "tolist", None)
    data = tolist() if callable(tolist) else value
    out: list[object] = []
    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, list | tuple):
            stack.extend(item)
        else:
            out.append(item)
    return out
