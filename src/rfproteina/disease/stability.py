"""Point-mutation stability (ddG) screen for a finished design, via ThermoMPNN.

Why this exists
---------------
The repository scores a design's *fold confidence* (AF3/AF2 pLDDT, pTM, RMSD)
but carries no term for its *thermodynamic stability*. Rosetta ``ddG`` is the
classic answer and is licence-encumbered (PyRosetta). ThermoMPNN
(https://github.com/Kuhlman-Lab/ThermoMPNN, MIT License, verified from the
repository's own ``LICENSE`` on 2026-11-16) is a free, weights-included graph
network that predicts the change in folding free energy for every point mutation
of a structure. It is the direct free replacement for a Rosetta ddG scan.

What it measures
----------------
ThermoMPNN's site-saturation mutagenesis (``analysis/SSM.py``) returns, for each
residue, the predicted ddG of mutating it to each of the other 19 amino acids.
Its output columns are ``position``, ``wildtype``, ``mutation`` and ``ddG_pred``
(kcal/mol). The *designed* sequence is the wild-type reference (its own ddG is
0), so what the scan tells us about a finished design is how much stability head-
room it has left: a design riddled with residues a single mutation would greatly
stabilise is, by that measure, far from a stability optimum.

Sign convention (verified, not assumed)
---------------------------------------
``analysis/SSM.py::retrieve_best_mutants`` selects the best mutant at a position
with ``idxmin()`` of ``ddG_pred``. So in ThermoMPNN's output **a more negative
ddG_pred is more stabilising**. That is the only interpretation this module
relies on. If a future ThermoMPNN release flips the sign, every threshold below
inverts; a reviewer must check the vendored version's convention before gating.

Where the pieces live
---------------------
ThermoMPNN's source is vendored, unmodified, in
``screening subpackage/thermompnn/ThermoMPNN`` at the upstream commit its
``manifest.toml`` records. Its weights stay under ``DATA_PATH/thermompnn/ThermoMPNN``
in upstream's layout, and its virtual environment is ``DATA_PATH/thermompnn/venv``.
``scripts/install_thermompnn.py`` fetches the weights and builds the environment.

Honesty about what is NOT done here
-----------------------------------
* No weights are vendored and nothing is downloaded by importing this module.
  The weights and the environment must already be installed (see
  :func:`predictor_available`); otherwise the screen raises a clear, actionable
  error rather than inventing numbers.
* Every threshold in :data:`STABILITY_BARS` is a *suggested default pending
  calibration*. ThermoMPNN publishes no pass mark for "is this design stable
  enough", and none is invented here. Do not gate existing pass criteria on
  these until someone calibrates them on this program's own designs; the
  integration note in the handoff flags this as opt-in.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from rfproteina.disease.screening_tools import checkout_root, data_dir, screening_code_root

METHOD = "thermompnn_point_stability_v1"
#: A checkpoint path. It wins over the copy installed under ``DATA_PATH``.
THERMOMPNN_CHECKPOINT_ENV = "THERMOMPNN_CHECKPOINT"
#: The vendored source tree. It wins over both locations :func:`code_root` tries.
THERMOMPNN_CODE_ENV = "THERMOMPNN_CODE_ROOT"
CHECKPOINT_NAME = "thermoMPNN_default.pt"
#: The ProteinMPNN weights the model is built around before the ddG checkpoint's
#: state is loaded. A second file, in a second directory, under the weights root.
PROTEINMPNN_WEIGHTS = ("vanilla_model_weights", "v_48_020.pt")

#: ddG (kcal/mol) at or below which a single mutation counts as "strongly
#: stabilising", i.e. evidence the designed residue is a stability liability.
#: Suggested default; ThermoMPNN publishes no such cut.
STRONGLY_STABILIZING_DDG = -1.0

#: A position is "improvable" when its single best available mutation is at or
#: below this ddG. Suggested default pending calibration.
IMPROVABLE_POSITION_DDG = -1.0

#: Thresholds applied by :func:`verdicts`. Each carries the fact that it is only
#: a suggested starting point, never a ThermoMPNN-published value.
STABILITY_BARS: dict[str, dict[str, object]] = {
    "fraction_positions_improvable": {
        "max": 0.25,
        "source": "suggested default pending calibration; ThermoMPNN publishes no design-level pass mark",
    },
    "n_strongly_stabilizing": {
        "max": None,
        "source": "informational; count of mutations with ddG_pred <= STRONGLY_STABILIZING_DDG, no gate",
    },
    "most_stabilizing_ddg": {
        "min": None,
        "source": "informational; most negative ddG_pred available (largest stability headroom), no gate",
    },
}

_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")


class ThermoMPNNUnavailableError(RuntimeError):
    """Raised when ThermoMPNN (or its checkpoint) is not installed.

    Carried as its own type so a caller can distinguish "the screen could not
    run" from "the screen ran and found a liability".
    """


@dataclass(frozen=True, slots=True)
class MutationDdg:
    """One predicted point mutation.

    Args:
        position: 1-based (or model-native) residue index as ThermoMPNN reports it.
        wildtype: One-letter code of the designed (reference) residue.
        mutation: One-letter code of the substituted residue.
        ddg_pred: Predicted ddG in kcal/mol. More negative = more stabilising.
    """

    position: int
    wildtype: str
    mutation: str
    ddg_pred: float


@dataclass(frozen=True, slots=True)
class StabilityReport:
    """Design-level roll-up of a site-saturation ddG scan."""

    n_positions: int
    n_mutations: int
    most_stabilizing_ddg: float | None
    most_destabilizing_ddg: float | None
    mean_ddg: float | None
    n_strongly_stabilizing: int
    fraction_positions_improvable: float | None
    mean_best_per_position: float | None
    notes: tuple[str, ...]

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def code_root() -> Path | None:
    """The ThermoMPNN source tree, or None when no candidate holds its code.

    Code and weights are resolved separately, from two roots, because they live
    in two places: the source is vendored in the repository, where it is
    readable and reviewable, and the weights stay under ``DATA_PATH``, which is
    where large binaries belong and where the installer puts them.

    ``THERMOMPNN_CODE_ROOT`` wins, then ``screening subpackage/thermompnn/
    ThermoMPNN``, then ``DATA_PATH/thermompnn/ThermoMPNN`` -- the layout before
    the source was vendored, kept so a machine whose checkout predates it still
    runs. A candidate counts only when it actually holds the entry point, so an
    empty leftover directory does not shadow a working tree.
    """
    named = os.environ.get(THERMOMPNN_CODE_ENV, "").strip()
    candidates: list[Path] = []
    if named:
        candidates.append(Path(named))
    else:
        code = screening_code_root()
        if code is None and (checkout := checkout_root()) is not None:
            # screening_code_root() returns None when the directory is absent;
            # name it anyway so the error says where the source was looked for.
            code = checkout / "screening subpackage"
        if code is not None:
            candidates.append(code / "thermompnn" / "ThermoMPNN")
        candidates.append(data_dir("thermompnn", "ThermoMPNN"))
    for candidate in candidates:
        if (candidate / "analysis" / "custom_inference.py").is_file():
            return candidate
    return None


def weights_root() -> Path:
    """``DATA_PATH/thermompnn/ThermoMPNN``: the directory the two weight files sit under.

    Upstream's own layout, kept exactly, because upstream's loader resolves
    ``vanilla_model_weights/`` relative to a single configured root.
    """
    return data_dir("thermompnn", "ThermoMPNN")


def checkpoint_path() -> Path | None:
    """The ThermoMPNN ddG weight file, or None when it is not on disk."""
    named = os.environ.get(THERMOMPNN_CHECKPOINT_ENV, "").strip()
    if named and Path(named).is_file():
        return Path(named)
    candidate = weights_root() / "models" / CHECKPOINT_NAME
    return candidate if candidate.is_file() else None


def proteinmpnn_weights_path() -> Path | None:
    """The ProteinMPNN weight file the model is built around, or None.

    Checked separately from the ddG checkpoint because it is loaded by a
    different code path, from a different directory, and its absence fails
    *inside* the model constructor rather than at the checkpoint load.
    """
    candidate = weights_root().joinpath(*PROTEINMPNN_WEIGHTS)
    return candidate if candidate.is_file() else None


def _runner_script() -> Path:
    """This package's bridge between the vendored source and the weights root."""
    return Path(__file__).resolve().with_name("thermompnn_runner.py")


def _inference_python() -> Path | None:
    root = weights_root().parent
    for relative in ("venv/bin/python", "venv/Scripts/python.exe"):
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def predictor_available() -> bool:
    """True when both weight files are on disk and a runner can load them.

    The runner is the virtual environment ``scripts/install_thermompnn.py``
    creates, driving the vendored source through :mod:`thermompnn_runner`, or
    an importable ThermoMPNN package. The check does not load the weights.
    """
    if checkpoint_path() is None or proteinmpnn_weights_path() is None:
        return False
    if code_root() is not None and _inference_python() is not None and _runner_script().is_file():
        return True
    return any(importlib.util.find_spec(name) is not None for name in ("thermompnn", "ThermoMPNN"))


def _normalize(record: Mapping[str, object]) -> MutationDdg:
    """Coerce one ThermoMPNN SSM row into a :class:`MutationDdg`.

    Accepts the column names ThermoMPNN's ``analysis/SSM.py`` writes
    (``position``, ``wildtype``, ``mutation``, ``ddG_pred``) and the lower-case
    spelling, so a reader of either a DataFrame row or a parsed CSV works.
    """
    ddg_key = next((k for k in ("ddG_pred", "ddg_pred", "ddG", "ddg") if k in record), None)
    if ddg_key is None:
        raise ValueError(f"ThermoMPNN row has no ddG column; got keys {sorted(record)!r}")
    wildtype = str(record.get("wildtype", record.get("wt", ""))).strip().upper()
    mutation = str(record.get("mutation", record.get("mut", ""))).strip().upper()
    return MutationDdg(
        position=int(record["position"]),
        wildtype=wildtype,
        mutation=mutation,
        ddg_pred=float(record[ddg_key]),  # type: ignore[arg-type]
    )


def summarize(mutations: Iterable[Mapping[str, object] | MutationDdg]) -> StabilityReport:
    """Reduce a site-saturation scan into design-level stability metrics.

    Rows whose mutation equals the wild-type (the no-op diagonal ThermoMPNN
    emits as a reference) are dropped, since a self-to-self ddG of ~0 is not a
    mutation and would dilute the counts.
    """
    rows = [m if isinstance(m, MutationDdg) else _normalize(m) for m in mutations]
    rows = [m for m in rows if m.mutation and m.mutation != m.wildtype]
    if not rows:
        return StabilityReport(
            n_positions=0,
            n_mutations=0,
            most_stabilizing_ddg=None,
            most_destabilizing_ddg=None,
            mean_ddg=None,
            n_strongly_stabilizing=0,
            fraction_positions_improvable=None,
            mean_best_per_position=None,
            notes=("no mutations scored",),
        )
    ddgs = [m.ddg_pred for m in rows]
    best_by_position: dict[int, float] = {}
    for m in rows:
        if m.position not in best_by_position or m.ddg_pred < best_by_position[m.position]:
            best_by_position[m.position] = m.ddg_pred
    improvable = sum(1 for v in best_by_position.values() if v <= IMPROVABLE_POSITION_DDG)
    return StabilityReport(
        n_positions=len(best_by_position),
        n_mutations=len(rows),
        most_stabilizing_ddg=round(min(ddgs), 4),
        most_destabilizing_ddg=round(max(ddgs), 4),
        mean_ddg=round(sum(ddgs) / len(ddgs), 4),
        n_strongly_stabilizing=sum(1 for v in ddgs if v <= STRONGLY_STABILIZING_DDG),
        fraction_positions_improvable=round(improvable / len(best_by_position), 4),
        mean_best_per_position=round(sum(best_by_position.values()) / len(best_by_position), 4),
        notes=(
            "ddG in kcal/mol; more negative = more stabilising (ThermoMPNN SSM convention)",
            "the designed sequence is the wild-type reference; these are improvements a mutation could make",
            "every threshold is a suggested default pending calibration, not a ThermoMPNN-published value",
        ),
    )


def run_site_saturation(
    structure: str | Path,
    *,
    chain: str | None = None,
    predict_fn: Callable[[Path, str | None], Sequence[Mapping[str, object]]] | None = None,
) -> list[MutationDdg]:
    """Run a ThermoMPNN site-saturation scan over ``structure``.

    Args:
        structure: Path to a PDB (or mmCIF) file of the finished design.
        chain: Chain id to scan, or ``None`` for the model's default.
        predict_fn: Injection seam for testing and for alternative back-ends. If
            given, it is called with ``(path, chain)`` and must return an
            iterable of rows carrying ``position``, ``wildtype``, ``mutation``
            and a ddG column. If ``None``, ThermoMPNN is imported and run.

    Raises:
        FileNotFoundError: If ``structure`` does not exist.
        ThermoMPNNUnavailableError: If ``predict_fn`` is ``None`` and ThermoMPNN
            is not importable.
    """
    path = Path(structure)
    if not path.exists():
        raise FileNotFoundError(f"structure not found: {path}")
    if predict_fn is not None:
        return [_normalize(row) for row in predict_fn(path, chain)]
    if not predictor_available():
        raise ThermoMPNNUnavailableError(
            "ThermoMPNN is not installed, so the stability scan cannot run. Install it "
            "(MIT, https://github.com/Kuhlman-Lab/ThermoMPNN) and fetch its checkpoint, "
            "or pass predict_fn=... to supply predictions. No ddG was invented."
        )
    return [_normalize(row) for row in _run_thermompnn(path, chain)]


def _run_thermompnn(path: Path, chain: str | None, *, timeout_s: float = 3600.0) -> Sequence[Mapping[str, object]]:
    """Run the vendored ``analysis/custom_inference.py`` against the installed weights.

    Driven through :mod:`thermompnn_runner`, which points upstream's loader at
    :func:`weights_root` before calling it: the source tree holds no weights, so
    upstream's own "weights sit beside the code" assumption has to be redirected
    once, in one place. The inference is still upstream's, and it writes the
    same CSV, with ``ddG_pred``, ``position``, ``wildtype`` and ``mutation``.

    ``-I`` keeps this package's directory off the subprocess's ``sys.path``,
    where ``datasets``, ``cache`` and ``model_utils`` would otherwise be
    importable names belonging to two different projects.
    """
    checkpoint = checkpoint_path()
    code = code_root()
    python = _inference_python()
    runner = _runner_script()
    if checkpoint is None or code is None or python is None or not runner.is_file():
        raise ThermoMPNNUnavailableError(
            "ThermoMPNN's weights, vendored source or virtual environment are not installed. "
            "Run scripts/install_thermompnn.py. No ddG was invented."
        )
    with tempfile.TemporaryDirectory(prefix="thermompnn_") as scratch:
        command = [
            str(python),
            "-I",
            str(runner),
            "--code",
            str(code),
            "--weights",
            str(weights_root()),
            "--checkpoint",
            str(checkpoint),
            "--pdb",
            str(path),
            "--chain",
            chain or "",
            "--out-dir",
            scratch,
        ]
        try:
            finished = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            # A site-saturation scan of a long chain is minutes of CPU, so an
            # unbounded wait was a hang rather than a slow answer.
            raise ThermoMPNNUnavailableError(f"thermompnn_failed:{type(error).__name__}") from error
        if finished.returncode != 0:
            detail = (finished.stderr or finished.stdout).strip().splitlines()
            tail = detail[-1] if detail else "no output"
            raise ThermoMPNNUnavailableError(f"thermompnn_failed:{tail}")
        csv_files = list(Path(scratch).glob("ThermoMPNN_inference_*.csv"))
        if len(csv_files) != 1:
            raise ThermoMPNNUnavailableError("thermompnn_failed:inference CSV was not written")
        with csv_files[0].open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


def screen_structure(
    structure: str | Path,
    *,
    chain: str | None = None,
    predict_fn: Callable[[Path, str | None], Sequence[Mapping[str, object]]] | None = None,
) -> StabilityReport:
    """Run the scan and summarise it in one call. See :func:`run_site_saturation`."""
    return summarize(run_site_saturation(structure, chain=chain, predict_fn=predict_fn))


def verdicts(report: StabilityReport) -> dict[str, dict[str, object]]:
    """Apply the suggested bars, each carrying that it is only suggested."""
    record = report.as_record()
    out: dict[str, dict[str, object]] = {}
    for name, rule in STABILITY_BARS.items():
        value = record.get(name)
        minimum = rule.get("min")
        maximum = rule.get("max")
        if value is None or (minimum is None and maximum is None):
            state = "untested" if value is None else "clear"
        else:
            low_ok = minimum is None or value >= minimum
            high_ok = maximum is None or value <= maximum
            state = "clear" if (low_ok and high_ok) else "hit"
        out[name] = {
            "value": value,
            "state": state,
            "min": minimum,
            "max": maximum,
            "threshold_source": rule["source"],
        }
    return out
