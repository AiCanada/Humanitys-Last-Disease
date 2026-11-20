"""Conformational dynamics of a finished design, read off an ensemble.

This is the analysis half of the Conformational Dynamics Analyst. It takes an
ensemble -- several conformers of one chain -- and reports how far that chain
moves: per-residue fluctuation, how far the ensemble strays from the design it
started as, and how much the conformers differ from each other. It does not
produce the ensemble; :py:mod:`rfproteina.disease.confrover_runner` does that,
and it is injected here so this module can be read, tested and trusted without
a model, a checkpoint or a GPU.

What this answers, and what it does not
---------------------------------------

It answers a question nothing else in this program asks. Every structural
number here is computed on **one** conformer: ``interface_quality`` measures
buried area and shape complementarity on a single frame, and the refold
criteria compare one predicted structure against one design. A design that
folds as drawn and then moves is invisible to all of it.

It does **not** answer whether an interface survives. RBase takes a single
polypeptide chain -- that is the model's own stated limitation, recorded under
``[scope]`` in its manifest -- so a design-plus-target complex is not an input
it has. :func:`ensemble_record` declines a multi-chain structure by name rather
than feeding it one chain of a complex and reporting the number that comes
back. The honest version of the interface question needs a model that takes two
chains, and this is not one.

Two properties of the generator decide how this module behaves
--------------------------------------------------------------

*It is out of distribution on the thing being screened.* RBase is trained on
natural proteins and their MD trajectories, over 100 ns, and a de novo design
is exactly the kind of input that is not in that set. No published number says
how far wrong it is there. So every record carries
:data:`OUT_OF_DISTRIBUTION`, and no verdict is reached without a cut a caller
names: there is no published fluctuation above which a design is too mobile,
and one invented here would fail designs on this program's authority. That is
the same stance ``target_liability.TargetLimits`` takes for its three axes and
``human_protocol.ProtocolLimits`` takes for both of its own.

*Generating an ensemble wants the network, and this module never reaches it.*
The generator's own ``generate()`` will clone a CUDA library and query a public
MSA server on first use. Neither belongs behind a screen: ``screening_tools``
opens by promising that its family installs nothing and downloads nothing.
Everything fetchable is therefore an install step the operator runs once, and
this module's job at run time is to check that each one is already present and
to decline by name when it is not -- including ``msa_not_cached``, which is the
one that would otherwise send a design's sequence to a third party.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rfproteina.disease.provenance import code_version
from rfproteina.disease.results_compare import alpha_carbons, chain_labels, kabsch_transform
from rfproteina.disease.screening_tools import CLEAR, HIT, UNTESTED

#: The reader's own name for this screen, as it appears on the tab.
ANALYST_LABEL = "Conformational Dynamics Analyst"

#: The generator, and the one sentence about it that has to travel with every
#: number it produces.
GENERATOR = "rbase_confrover_20m_v1_0"
OUT_OF_DISTRIBUTION = (
    "RBase is trained on natural proteins and their MD trajectories at 100 ns, so a de novo "
    "design is out of distribution and no published number says by how much. These are the "
    "model's conformers, not measurements of this molecule."
)
#: The second caveat, which is about the clock rather than the chemistry.
TIMESCALE_CAVEAT = (
    "The training trajectories are 100 ns, so nothing here speaks to slow conformational "
    "change. A fold that holds across this ensemble has not been shown to hold for longer."
)

#: The three things the generator can be asked for. Closed: a mode outside this
#: set is a mode whose output nothing here knows how to read.
FORWARD = "forward"
IID = "iid"
INTERPOLATE = "interp"
TASK_MODES = (FORWARD, IID, INTERPOLATE)

#: What each mode is for, in the words the tab shows.
TASK_PURPOSE = {
    FORWARD: (
        "A trajectory from the design onwards: does this fold stay where it was drawn, and "
        "which parts of it move first."
    ),
    IID: (
        "Independent conformers of the same sequence, with no trajectory between them: the "
        "spread of shapes the model thinks this sequence takes."
    ),
    INTERPOLATE: (
        "A path between two conformations you supply, for a design meant to move between two " "named states."
    ),
}

#: Frames below which a fluctuation is arithmetic rather than a measurement. Two
#: conformers have a mean they both sit a fixed distance from, so every residue
#: reads as equally mobile; the shape of the curve only means something once
#: there are several.
MIN_FRAMES = 5
#: Frames one analysis will read. An ensemble past this is a trajectory to
#: subsample before it is handed over, not one to read whole in a panel.
MAX_FRAMES = 2000
#: Residues below which the superposition says more about the arithmetic than
#: the structures, the same floor ``results_compare`` keeps for an RMSD.
MIN_RESIDUES = 8

#: Where the operator-installed payload lives. Everything the generator would
#: otherwise fetch is rooted here, so a run writes nothing into the checkout.
DATA_SUBDIR = "confrover"
#: Set to point the runner at an interpreter that can import ``rbase``.
PYTHON_ENV = "CONFROVER_PYTHON"


def refusal(blocker: str | None) -> str:
    """The blocker a refusal carries, passed through.

    A plumbing function with one job, and the job is visibility. Blockers here
    reach a reader in two shapes the vocabulary scan cannot search for: the
    second element of a tuple, ``(None, "ensemble_too_small")``, and a dict
    *value* under some other key. Names in either shape arrived with no
    sentence behind them and nothing could demand one. Passed through a helper
    that declares an optional ``blocker`` parameter, every one of them is found.

    Defined above the constants that use it, which is why a pure pass-through
    sits in the middle of a block of module-level data.
    """
    return blocker or ""


#: The three blockers a caller on another surface needs to name, exported so
#: they are named through this module and checked against this module's words.
#: The results page reaches all three: a catalog with no readable design, a
#: vendored viewer that is not there, and a draw asked for before a run.
NO_DESIGN = refusal("design_not_supplied")
NO_VIEWER = refusal("viewer_failed")
NO_ENSEMBLE = refusal("ensemble_not_supplied")

#: Seconds one generation may take. Generous next to the alignment screens --
#: this is a diffusion model producing frame after frame, and 200 denoising
#: steps per frame is minutes on a GPU and longer without one -- and bounded
#: all the same, because it runs behind a button and a wedged generator must
#: not hold the request open indefinitely. The tab says so before it starts.
GENERATE_TIMEOUT_S = 3600.0
#: Seconds one drawing may take. The page is seconds; the movie is the reason
#: this is not minutes -- the viewer's own documentation puts it at minutes to
#: an hour depending on the conformer count, which is why a movie is opt-in.
VIEW_TIMEOUT_S = 3600.0

#: Each piece of the install, and the blocker naming it when it is absent. The
#: generator would fetch or build every one of these on first use; a screen
#: must not, so each is checked instead.
#: Each name goes through :func:`refusal`, which does nothing to the string and
#: everything for whether anyone notices it. A blocker sitting as a dict
#: *value* under some other key is invisible to the vocabulary scan, so these
#: three reached a reader with no sentence behind them and nothing could demand
#: one. ``weights_not_installed`` was found only because ``payload_blocker``
#: also returns it bare from a function the scan recognises by name.
REQUIRED_PAYLOAD = {
    "weights": refusal("weights_not_installed"),
    "cutlass": refusal("cutlass_not_installed"),
    "openfold_params": refusal("openfold_params_not_installed"),
    "msa": refusal("msa_not_cached"),
}


@dataclass(frozen=True)
class DynamicsLimits:
    """The cuts a caller names. Each is None until then, and an unnamed cut withholds the call.

    The same choice ``target_liability.TargetLimits`` makes, for the same
    reason: these are real measurements with no published number that turns one
    into a liability. A design is not unusable because a 20M-parameter model
    moved its loops 3 A.
    """

    #: Angstrom of per-residue fluctuation at or above which a residue counts
    #: as mobile. None until a caller names it.
    rmsf_max_a: float | None = None
    #: Share of residues allowed to be mobile before the design counts as one
    #: that does not hold its shape. Only read when ``rmsf_max_a`` is named,
    #: because without a per-residue cut there is nothing to take a share of.
    mobile_fraction_max: float | None = None
    #: Angstrom the ensemble mean may sit from the design it started as.
    drift_max_a: float | None = None


@dataclass(frozen=True)
class Frame:
    """One conformer: a chain label, its sequence, and its alpha carbons."""

    chain: str
    sequence: str
    coords: np.ndarray

    def __len__(self) -> int:
        return len(self.sequence)


@dataclass(frozen=True)
class Ensemble:
    """Conformers of one chain, and where they came from.

    ``blocker`` is empty on an ensemble that was produced. Everything else is
    only meaningful when it is: an ensemble that was never generated holds no
    frames, and an empty frame list from a run that did not happen would read
    as a design that does not move, which is the one answer it must not give.
    """

    frames: tuple[Frame, ...] = ()
    task_mode: str = FORWARD
    design: str = ""
    replicates: int = 0
    stride_in_10ps: int | None = None
    blocker: str = ""
    detail: str = ""
    #: What produced it, so a record can say whether a number came from the
    #: model or from a caller's own trajectory.
    produced_by: str = GENERATOR

    @property
    def ran(self) -> bool:
        return not self.blocker and len(self.frames) > 0


#: A generator, injected. Supplied by the caller, this module needs no model,
#: no checkpoint and no GPU; the real one is
#: ``confrover_runner.generate_ensemble``.
Generate = Callable[[str, str, str], Ensemble]


#: One sentence per blocker, in this surface's own voice. Routed to this table
#: rather than the screening board by ``tests/test_disease_board_vocabulary.py``:
#: the analyst is read on its own tab and in a terminal, and the board never
#: draws these names.
_BLOCKER_TEXT = {
    "ensemble_not_supplied": (
        "No ensemble was produced, so there is nothing to measure. Run the generator, or hand "
        "in a trajectory of your own."
    ),
    "ensemble_too_small": (
        f"Fewer than {MIN_FRAMES} conformers. A fluctuation taken over two or three frames is "
        "arithmetic rather than a measurement: every residue reads as equally mobile."
    ),
    "ensemble_too_large": (
        f"More than {MAX_FRAMES} conformers. Subsample the trajectory before handing it over; "
        "reading one this long in a panel is not what this view is for."
    ),
    "frames_disagree_on_length": (
        "The conformers do not all have the same number of residues, so they are not an "
        "ensemble of one chain and no fluctuation can be taken across them."
    ),
    "chain_too_short": (
        f"Fewer than {MIN_RESIDUES} residues. A superposition over a handful of points says "
        "more about the arithmetic than about the structures."
    ),
    "design_chain_not_single": (
        "This structure holds more than one chain. The generator takes a single polypeptide "
        "chain, so a complex is not an input it has -- name the chain to analyse, and read the "
        "interface question somewhere that can answer it."
    ),
    "design_not_supplied": "No design structure was given, so there was nothing to generate from.",
    "design_unreadable": "The design structure could not be read as a structure.",
    "task_mode_unknown": (
        "That is not one of the three things the generator can be asked for: a forward "
        "trajectory, independent conformers, or a path between two states."
    ),
    "conditions_required": (
        "A forward trajectory starts from a conformation and an interpolation needs two, so "
        "the starting structure cannot be left out for either."
    ),
    "weights_not_installed": (
        "The RBase checkpoint is not under DATA_PATH. scripts/install_confrover.py fetches it "
        "and checks it against the digest its manifest records."
    ),
    "confrover_not_installed": (
        "No interpreter on this machine can import the vendored rbase package. "
        "scripts/install_confrover.py builds one; CONFROVER_PYTHON names another."
    ),
    "cutlass_not_installed": (
        "The CUDA library the generator builds its kernels from is not present. It would clone "
        "it from the internet on first use, which a screen must not do, so the installer "
        "fetches it once instead."
    ),
    "openfold_params_not_installed": (
        "The folding parameters the generator needs to embed a sequence are not under "
        "DATA_PATH. The installer fetches them."
    ),
    "msa_not_cached": (
        "No alignment is cached for this sequence. Building one queries a public MSA server, "
        "which would send this design's sequence off this machine -- so the screen will not do "
        "it. Cache the alignment deliberately first."
    ),
    "generator_failed": (
        "The generator ran and did not finish cleanly, so no ensemble was produced. Its own " "reason is on the record."
    ),
    "rmsf_threshold_not_set": (
        "The fluctuation was measured and not judged: no published number says how mobile a "
        "residue has to be before a design is too mobile, so this program will not pick one. "
        "Name a cut to get a verdict."
    ),
    "mobile_fraction_not_set": (
        "Which residues are mobile is named above and not judged. Whether that matters depends "
        "on how much of the chain they are, so a share of the chain is needed as well as a "
        "per-residue cut."
    ),
    "drift_threshold_not_set": (
        "The drift from the design was measured and not judged, for the same reason: no "
        "published number says how far an ensemble may sit from the structure it started as."
    ),
    "movie_tool_not_installed": (
        "The mp4 writer is not installed, so no movie was rendered. The page was still "
        "written and holds the whole ensemble; imageio-ffmpeg adds the movie."
    ),
    "viewer_failed": "The viewer ran and did not produce a page. Its own reason is on the record.",
}


def sentence(blocker: str) -> str:
    """The words for one blocker, or the name spaced out when it has none."""
    if not blocker:
        return ""
    return _BLOCKER_TEXT.get(blocker, blocker.replace("_", " "))


def blocked_record(blocker: str) -> dict[str, object]:
    """A record for an analysis that could not start at all.

    Here rather than at the caller, for the reason
    ``human_protocol.blocked_record`` is: a blocker name is checked against the
    vocabulary of the module that *writes* it, so the same literal written from
    the results page is checked against the results page's own table -- which
    speaks in fragments, for a different surface. One sentence for each name,
    in the table that renders it.
    """
    return ensemble_record(Ensemble(blocker=blocker))


def data_root() -> Path | None:
    """``DATA_PATH/confrover``, or None when DATA_PATH is unset.

    Everything the generator would otherwise fetch is rooted here -- the
    checkpoints, the CUDA clone, the folding parameters, the alignment cache
    and the SO(3) tables -- because its own cache paths all derive from one
    directory. Pointing that at DATA_PATH is what keeps a run from writing into
    the checkout.
    """
    named = os.environ.get("DATA_PATH", "").strip()
    return Path(named) / DATA_SUBDIR if named else None


def payload_blocker(root: Path | None = None) -> str | None:
    """The first piece of the install that is missing, or None when all are there.

    Checked rather than attempted. Every one of these the generator would fetch
    or build by itself on first call, and two of them reach the network: the
    whole point of asking here is that the screen never becomes the thing that
    downloads.
    """
    where = data_root() if root is None else root
    if where is None or not where.is_dir():
        return "weights_not_installed"
    for name, blocker in REQUIRED_PAYLOAD.items():
        if not _installed(where / name):
            return blocker
    return None


def _installed(path: Path) -> bool:
    """Whether a piece of the install is really there, not just its directory.

    Emptiness counts as absent, which is the point. An interrupted install
    leaves the directory it was writing into, and a bare ``exists()`` read that
    as done: the screen waved a half-install through and the failure surfaced
    deeper, inside the generator, as something that looked like a model problem.
    """
    if path.is_file():
        return True
    return path.is_dir() and any(path.iterdir())


def code_root() -> Path | None:
    """The directory the vendored ``rbase`` package sits *inside*.

    That parent, not the package itself, because it is what goes on
    ``sys.path``. Resolved through ``screening_tools.checkout_root`` so it is
    found by searching upward for the marker file rather than by counting
    parents, which is silently wrong the moment this module's depth changes.
    """
    from rfproteina.disease.screening_tools import SCREENING_CODE_DIRNAME, checkout_root

    root = checkout_root()
    if root is None:
        return None
    found = root / SCREENING_CODE_DIRNAME / "confrover"
    return found if (found / "rbase").is_dir() else None


def viewer_root() -> Path | None:
    """The vendored viewer, which reads this generator's output and nothing else.

    A flat directory rather than a package: its modules import each other by
    bare name, so this goes on ``sys.path`` as it is. The generation and
    screening pages keep their own viewers; this one draws ensembles.
    """
    code = code_root()
    if code is None:
        return None
    found = code / "viz"
    return found if (found / "visualize_ensemble.py").is_file() else None


def runner_script() -> Path:
    """The script the generator runs under, beside this module."""
    return Path(__file__).with_name("confrover_runner.py")


def ensembles_root() -> Path | None:
    """Where a generated ensemble is kept so it can be drawn afterwards.

    Under the weights directory rather than in a temporary one, because the
    run and the draw are two button presses: an ensemble that vanished with
    the process left the second button with nothing, which read as a design
    that does not move. Under DATA_PATH rather than the checkout because
    conformers are run output, and ``AGENTS.md`` is explicit that those never
    follow the code.
    """
    root = data_root()
    return root / "ensembles" if root is not None else None


def viewer_python() -> str:
    """The interpreter the viewer runs under: this project's own.

    Deliberately not :func:`python_executable`. The generator needs three
    packages this environment lacks, and the viewer needs numpy, which it has
    -- so the page can be drawn on a machine where nothing is installed, from
    an ensemble somebody else produced. Tying the two together would have made
    a viewer that works wait for a generator that is not there.
    """
    return sys.executable


def viewer_script() -> Path | None:
    """The vendored viewer's entry point, or None when it is not vendored."""
    viewer = viewer_root()
    return (viewer / "visualize_ensemble.py") if viewer is not None else None


def python_executable() -> str | None:
    """An interpreter that can run the generator, or None when there is none.

    ``CONFROVER_PYTHON`` wins, then the venv the installer builds under
    DATA_PATH. This project's own interpreter is deliberately *not* a
    fallback: it is three packages short of importing the generator, and
    falling back to it would turn a missing install into a confusing
    ``ModuleNotFoundError`` deep inside somebody else's source instead of the
    named decline the screen is supposed to give.
    """
    named = os.environ.get(PYTHON_ENV, "").strip()
    if named:
        candidate = Path(named)
        return str(candidate) if candidate.is_file() else None
    root = data_root()
    if root is None:
        return None
    for relative in ("venv/bin/python", "venv/Scripts/python.exe"):
        candidate = root / relative
        if candidate.is_file():
            return str(candidate)
    return None


def frames_from_structures(
    texts: Sequence[str],
    *,
    chain: str = "",
) -> tuple[tuple[Frame, ...], str]:
    """``(frames, blocker)`` from structure texts, one conformer each.

    The chain is named rather than guessed. A design-plus-target complex has
    two, and picking one would analyse whichever happened to come first --
    which on most of this program's outputs is the target.
    """
    found: list[Frame] = []
    for text in texts:
        if not text.strip():
            return (), refusal("design_unreadable")
        labels = chain_labels(text)
        if not labels:
            return (), refusal("design_unreadable")
        wanted = chain or (labels[0] if len(labels) == 1 else "")
        if not wanted or wanted not in labels:
            return (), refusal("design_chain_not_single")
        residues = alpha_carbons(text, wanted)
        if residues is None or len(residues.sequence) == 0:
            return (), refusal("design_unreadable")
        found.append(Frame(wanted, residues.sequence, np.asarray(residues.coords, dtype=float)))
    if not found:
        return (), refusal("ensemble_not_supplied")
    if len(found[0]) < MIN_RESIDUES:
        return (), refusal("chain_too_short")
    if len({len(frame) for frame in found}) != 1:
        return (), refusal("frames_disagree_on_length")
    return tuple(found), ""


def _superposed(frames: Sequence[Frame]) -> np.ndarray:
    """Every frame's coordinates, each rotated onto the first.

    Fluctuation is about shape, so the rigid-body part has to come out first:
    an ensemble that merely drifted across the box would otherwise read as one
    whose every residue moves. The first frame is the reference because for a
    forward trajectory it is the design itself, which is the thing the drift is
    measured from.
    """
    reference = frames[0].coords
    stack = [reference]
    for frame in frames[1:]:
        rotation, translation = kabsch_transform(frame.coords, reference)
        # ``rotation.T``, which is the convention ``kabsch_transform`` states
        # and ``results_compare._moved`` applies. Written without it, every
        # frame was rotated by the transpose of the intended rotation: an
        # ensemble that had only moved through space read as one whose every
        # residue fluctuates, and the error hid behind any planted signal large
        # enough to swamp it.
        stack.append(frame.coords @ rotation.T + translation)
    return np.stack(stack)


def rmsf(frames: Sequence[Frame]) -> np.ndarray:
    """Per-residue root-mean-square fluctuation about the ensemble mean, in angstrom."""
    stack = _superposed(frames)
    mean = stack.mean(axis=0)
    return np.sqrt(((stack - mean) ** 2).sum(axis=2).mean(axis=0))


def drift_a(frames: Sequence[Frame]) -> float:
    """How far the ensemble mean sits from the first frame, in angstrom.

    The first frame is the design on a forward trajectory, so this is the
    question "did it stay where it was drawn" rather than "how wide is the
    ensemble", which is what :func:`spread_a` answers.
    """
    stack = _superposed(frames)
    mean = stack.mean(axis=0)
    return float(np.sqrt(((mean - stack[0]) ** 2).sum(axis=1).mean()))


def spread_a(frames: Sequence[Frame]) -> float:
    """Mean pairwise RMSD across the ensemble, in angstrom.

    Over the superposed stack, so it is the spread of shapes rather than of
    positions. Taken against the mean rather than over every pair: the pairwise
    mean square is twice the variance about the centroid, so this is the same
    quantity at a fraction of the arithmetic for a thousand frames.
    """
    stack = _superposed(frames)
    mean = stack.mean(axis=0)
    per_frame = np.sqrt(((stack - mean) ** 2).sum(axis=2).mean(axis=1))
    return float(per_frame.mean())


def radius_of_gyration(frames: Sequence[Frame]) -> tuple[float, float]:
    """``(mean, spread)`` radius of gyration over the ensemble, in angstrom.

    Reported because it is the one term here that distinguishes a design whose
    loops move from one that is coming apart: fluctuation rises for both, and
    only the second expands.
    """
    values = []
    for frame in frames:
        centred = frame.coords - frame.coords.mean(axis=0)
        values.append(float(np.sqrt((centred**2).sum(axis=1).mean())))
        del centred
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std())


@dataclass(frozen=True)
class Measured:
    """What one ensemble came to, before any cut is applied."""

    frames: int
    residues: int
    chain: str
    rmsf_a: tuple[float, ...] = field(default_factory=tuple)
    drift_a: float | None = None
    spread_a: float | None = None
    radius_of_gyration_a: float | None = None
    radius_of_gyration_spread_a: float | None = None

    @property
    def rmsf_max_a(self) -> float | None:
        return max(self.rmsf_a) if self.rmsf_a else None

    @property
    def rmsf_mean_a(self) -> float | None:
        return sum(self.rmsf_a) / len(self.rmsf_a) if self.rmsf_a else None

    def mobile_residues(self, cut: float) -> tuple[int, ...]:
        """Which residue positions sit at or above ``cut``, numbered from one."""
        return tuple(index + 1 for index, value in enumerate(self.rmsf_a) if value >= cut)

    def as_record(self) -> dict[str, object]:
        return {
            "frames": self.frames,
            "residues": self.residues,
            "chain": self.chain,
            "rmsf_a": [round(value, 3) for value in self.rmsf_a],
            "rmsf_max_a": None if self.rmsf_max_a is None else round(self.rmsf_max_a, 3),
            "rmsf_mean_a": None if self.rmsf_mean_a is None else round(self.rmsf_mean_a, 3),
            "drift_a": None if self.drift_a is None else round(self.drift_a, 3),
            "spread_a": None if self.spread_a is None else round(self.spread_a, 3),
            "radius_of_gyration_a": (
                None if self.radius_of_gyration_a is None else round(self.radius_of_gyration_a, 3)
            ),
            "radius_of_gyration_spread_a": (
                None if self.radius_of_gyration_spread_a is None else round(self.radius_of_gyration_spread_a, 3)
            ),
        }


def measure(ensemble: Ensemble) -> tuple[Measured | None, str]:
    """``(measured, blocker)`` for one ensemble. Never raises.

    Every refusal is a named blocker rather than an exception, because this runs
    behind a button on a page the run has already finished: a view must not be
    able to end a run.
    """
    if not ensemble.ran:
        return None, ensemble.blocker or refusal("ensemble_not_supplied")
    frames = ensemble.frames
    if len(frames) < MIN_FRAMES:
        return None, refusal("ensemble_too_small")
    if len(frames) > MAX_FRAMES:
        return None, refusal("ensemble_too_large")
    if len({len(frame) for frame in frames}) != 1:
        return None, refusal("frames_disagree_on_length")
    if len(frames[0]) < MIN_RESIDUES:
        return None, refusal("chain_too_short")
    try:
        fluctuation = tuple(float(value) for value in rmsf(frames))
        drift = drift_a(frames)
        spread = spread_a(frames)
        gyration, gyration_spread = radius_of_gyration(frames)
    except Exception as error:
        # Broad for the reason ``human_protocol.measure_one`` is broad: a
        # degenerate ensemble can defeat the arithmetic in ways that are not an
        # OSError or a ValueError, and the alternative is a traceback out of a
        # page.
        return None, f"{refusal('generator_failed')}:{type(error).__name__}"
    if not all(math.isfinite(value) for value in (*fluctuation, drift, spread, gyration)):
        return None, refusal("generator_failed")
    return (
        Measured(
            frames=len(frames),
            residues=len(frames[0]),
            chain=frames[0].chain,
            rmsf_a=fluctuation,
            drift_a=drift,
            spread_a=spread,
            radius_of_gyration_a=gyration,
            radius_of_gyration_spread_a=gyration_spread,
        ),
        "",
    )


def _call(measured: Measured, bar: DynamicsLimits) -> tuple[str, str]:
    """``(state, blocker)`` for one measured ensemble.

    Undecided unless a cut was named, which is most of the time and is the
    correct answer rather than a gap: nothing published says how mobile is too
    mobile for a design, and a number chosen here would fail designs on this
    program's authority.
    """
    if bar.rmsf_max_a is None:
        return UNTESTED, refusal("rmsf_threshold_not_set")
    if bar.mobile_fraction_max is None:
        # A fluctuation cut on its own names which residues are mobile and
        # cannot decide anything about them: whether five mobile loops matter
        # depends on how much of the chain they are. Without the share, this
        # returned CLEAR while listing the mobile residues underneath it --
        # a verdict contradicted by the rows beside it.
        return UNTESTED, refusal("mobile_fraction_not_set")
    mobile = measured.mobile_residues(bar.rmsf_max_a)
    share = len(mobile) / measured.residues if measured.residues else 0.0
    if share > bar.mobile_fraction_max:
        return HIT, ""
    if bar.drift_max_a is None:
        # The fluctuation cleared, and the design may still have walked away
        # from where it was drawn. Clearing on half the question would be the
        # screen answering a narrower one than it appears to.
        return UNTESTED, refusal("drift_threshold_not_set")
    if measured.drift_a is not None and measured.drift_a > bar.drift_max_a:
        return HIT, ""
    return CLEAR, ""


def ensemble_record(
    ensemble: Ensemble,
    *,
    limits: DynamicsLimits | None = None,
) -> dict[str, object]:
    """One ensemble's record: what was measured, the call, and both caveats.

    The caveats ride on the record rather than on the page, so a number cannot
    be quoted without the sentence saying what produced it and what that
    producer is known not to cover.
    """
    bar = DynamicsLimits() if limits is None else limits
    record: dict[str, object] = {
        "analyst": ANALYST_LABEL,
        "generator": ensemble.produced_by,
        "task_mode": ensemble.task_mode,
        "purpose": TASK_PURPOSE.get(ensemble.task_mode, ""),
        "design": ensemble.design,
        "replicates": ensemble.replicates,
        "stride_in_10ps": ensemble.stride_in_10ps,
        "out_of_distribution": OUT_OF_DISTRIBUTION,
        "timescale": TIMESCALE_CAVEAT,
        "single_chain_only": True,
        "thresholds": {
            "rmsf_max_a": bar.rmsf_max_a,
            "mobile_fraction_max": bar.mobile_fraction_max,
            "drift_max_a": bar.drift_max_a,
        },
        "measured": None,
        "mobile_residues": [],
        "state": UNTESTED,
        "blocker": None,
        "reason": "",
        "code": code_version(),
    }
    measured, blocker = measure(ensemble)
    if measured is None:
        record["blocker"] = blocker
        record["reason"] = sentence(blocker)
        if ensemble.detail:
            record["detail"] = ensemble.detail
        return record
    record["measured"] = measured.as_record()
    state, held = _call(measured, bar)
    record["state"] = state
    if bar.rmsf_max_a is not None:
        record["mobile_residues"] = list(measured.mobile_residues(bar.rmsf_max_a))
    if held:
        record["blocker"] = held
        record["reason"] = sentence(held)
    return record


def analyse(
    design_text: str,
    *,
    chain: str = "",
    task_mode: str = FORWARD,
    limits: DynamicsLimits | None = None,
    generate: Generate | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """The whole screen: check the install, generate, measure, report.

    The order is deliberate. The design is read first, because a complex is
    refused before anything is generated from one chain of it; then the install
    is checked, because every missing piece of it is a thing the generator
    would otherwise fetch; and only then is the generator asked.
    """
    if not design_text.strip():
        return ensemble_record(Ensemble(blocker="design_not_supplied"), limits=limits)
    if task_mode not in TASK_MODES:
        return ensemble_record(Ensemble(blocker="task_mode_unknown"), limits=limits)
    labels = chain_labels(design_text)
    if not labels:
        return ensemble_record(Ensemble(blocker="design_unreadable"), limits=limits)
    wanted = chain or (labels[0] if len(labels) == 1 else "")
    if not wanted or wanted not in labels:
        return ensemble_record(Ensemble(blocker="design_chain_not_single", task_mode=task_mode), limits=limits)
    absent = payload_blocker(root)
    if absent is not None:
        return ensemble_record(Ensemble(blocker=absent, task_mode=task_mode), limits=limits)
    if generate is None:
        return ensemble_record(Ensemble(blocker="confrover_not_installed", task_mode=task_mode), limits=limits)
    produced = generate(design_text, wanted, task_mode)
    return ensemble_record(produced, limits=limits)


def generate_ensemble(
    design_text: str,
    chain: str,
    task_mode: str,
    *,
    work_dir: Path | None = None,
    frames: int = 10,
    replicates: int = 1,
    stride_in_10ps: int = 256,
    timeout_s: float = GENERATE_TIMEOUT_S,
) -> Ensemble:
    """Ask the generator for an ensemble, through its own interpreter.

    The real :data:`Generate`. Never raises: every failure -- no interpreter,
    no install, a tool that exited badly, output that cannot be read -- comes
    back as an :class:`Ensemble` carrying the blocker, because an ensemble that
    was not produced must not be able to read as a design that does not move.

    The design is written into ``work_dir`` and the conformers are read back
    out of it. Nothing is written beside the vendored source: the runner roots
    every cache the generator keeps at the weights directory.
    """
    import tempfile

    from rfproteina.disease.screening_tools import run_tool

    code = code_root()
    if code is None:
        return Ensemble(task_mode=task_mode, blocker="confrover_not_installed")
    weights = data_root()
    absent = payload_blocker()
    if weights is None or absent is not None:
        return Ensemble(task_mode=task_mode, blocker=absent or "weights_not_installed")
    python = python_executable()
    if python is None:
        return Ensemble(task_mode=task_mode, blocker="confrover_not_installed")
    residues = alpha_carbons(design_text, chain)
    if residues is None or not residues.sequence:
        return Ensemble(task_mode=task_mode, blocker="design_unreadable")

    with tempfile.TemporaryDirectory(prefix="confrover-") as temporary:
        where = Path(work_dir) if work_dir is not None else Path(temporary)
        where.mkdir(parents=True, exist_ok=True)
        design = where / "design.pdb"
        design.write_text(design_text, encoding="utf-8")
        out = where / "ensemble"
        command = [
            python,
            "-I",
            str(runner_script()),
            "--code",
            str(code),
            "--weights",
            str(weights),
            "--design",
            str(design),
            "--seqres",
            residues.sequence,
            "--task",
            task_mode,
            "--out",
            str(out),
            "--frames",
            str(frames),
            "--replicates",
            str(replicates),
            "--stride-in-10ps",
            str(stride_in_10ps),
        ]
        outcome = run_tool(command, timeout_s=timeout_s)
        said = _result_line(outcome.stdout)
        if said is None:
            return Ensemble(
                task_mode=task_mode,
                blocker="generator_failed",
                detail=_short(outcome.blocker or outcome.stderr),
            )
        if "blocker" in said:
            return Ensemble(
                task_mode=task_mode,
                blocker=str(said["blocker"]),
                detail=_short(str(said.get("detail", ""))),
            )
        written = [str(name) for name in said.get("frames", []) if isinstance(name, str)]
        texts: list[str] = []
        for name in written[:MAX_FRAMES]:
            try:
                texts.append(Path(name).read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
        if not texts:
            return Ensemble(task_mode=task_mode, blocker="ensemble_not_supplied")
        built, blocker = frames_from_structures(texts, chain="")
        if blocker:
            return Ensemble(task_mode=task_mode, blocker=blocker)
        return Ensemble(
            frames=built,
            task_mode=task_mode,
            design=chain,
            replicates=replicates,
            stride_in_10ps=stride_in_10ps if task_mode != IID else None,
        )


def _result_line(stdout: str) -> dict[str, object] | None:
    """The one line the runner prints for the caller, parsed.

    The generator logs freely, so the result is marked rather than assumed to
    be the last line. A run whose marker is absent did not get far enough to
    print one.
    """
    import json

    from rfproteina.disease.confrover_runner import RESULT_PREFIX

    for line in reversed((stdout or "").splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                found = json.loads(line[len(RESULT_PREFIX) :])
            except ValueError:
                return None
            return found if isinstance(found, dict) else None
    return None


def _short(text: str) -> str:
    """A detail cut to a line, with no path in it.

    The same rule ``human_protocol._detail`` keeps: this is stored on a record
    a reader can save, and a filesystem path belongs in a log rather than in
    something that leaves the machine.
    """
    cleaned = " ".join((text or "").split())
    cleaned = " ".join(part for part in cleaned.split(" ") if "/" not in part and "\\" not in part)
    return cleaned[:120]


def draw_ensemble(
    ensemble_dir: Path | str,
    out_dir: Path | str,
    *,
    chain: str = "",
    title: str = "",
    movie: bool = False,
    timeout_s: float = VIEW_TIMEOUT_S,
) -> dict[str, object]:
    """Draw a generated ensemble with the vendored viewer. Never raises.

    The page always, the movie only when asked and only when the writer for
    one is installed: the page takes seconds and holds the whole ensemble, and
    the movie takes minutes to an hour, so a missing mp4 writer is reported
    beside a page that was written rather than instead of one.

    The source handed over is always a local directory. The viewer accepts an
    ``hf:`` source and will fetch it, which is a route for an operator and not
    for a screen; a local path wins and is never checked against the Hub.
    """
    from rfproteina.disease.screening_tools import run_tool

    record: dict[str, object] = {
        "page": None,
        "movie": None,
        "conformers_drawn": 0,
        "blocker": None,
        "reason": "",
    }
    script = viewer_script()
    source = Path(ensemble_dir)
    if script is None:
        return _drawn(record, NO_VIEWER)
    if not source.is_dir() or not any(source.rglob("*.pdb")):
        return _drawn(record, NO_ENSEMBLE)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    command = [viewer_python(), "-I", str(script), str(source), "--out", str(out)]
    if chain:
        command += ["--chain", chain]
    if title:
        command += ["--title", title]
    if movie:
        if not _movie_writer_available():
            # The page is still worth drawing, so the movie is dropped rather
            # than the whole request refused, and the record says which.
            record["blocker"] = refusal("movie_tool_not_installed")
            record["reason"] = sentence("movie_tool_not_installed")
        else:
            command.append("--movie")

    outcome = run_tool(command, timeout_s=timeout_s)
    if not outcome.ok:
        return _drawn(record, NO_VIEWER, detail=_short(outcome.blocker or outcome.stderr))
    pages = sorted(out.glob("*.html"))
    movies = sorted(out.glob("*.mp4"))
    if not pages:
        return _drawn(record, NO_VIEWER, detail="the viewer wrote no page")
    record["page"] = str(pages[0])
    record["movie"] = str(movies[0]) if movies else None
    record["conformers_drawn"] = len(list(source.rglob("*.pdb")))
    return record


def _drawn(record: dict[str, object], blocker: str, *, detail: str = "") -> dict[str, object]:
    """One failed draw, named."""
    record["blocker"] = blocker
    record["reason"] = sentence(blocker)
    if detail:
        record["detail"] = detail
    return record


def _movie_writer_available() -> bool:
    """Whether an mp4 can be written at all.

    Asked rather than attempted. The viewer imports ``imageio_ffmpeg`` inside
    the function that renders, so without this the request would run for
    however long the page takes and then fail at the last step.
    """
    from importlib.util import find_spec

    try:
        return find_spec("imageio_ffmpeg") is not None
    except (ImportError, ValueError):
        return False


def readout_lines(record: Mapping[str, object]) -> list[str]:
    """The record as plain lines, for a terminal and for the saved copy.

    The same numbers the tab draws, written here so a headless run reports
    exactly what a reader sees in the browser rather than the two being kept in
    step by hand.
    """
    lines = [summary_line(record)]
    lines.append(f"  generator: {record.get('generator')} ({record.get('task_mode')})")
    purpose = str(record.get("purpose") or "")
    if purpose:
        lines.append(f"  asked for: {purpose}")
    lines.append(f"  out of distribution: {record.get('out_of_distribution')}")
    lines.append(f"  timescale: {record.get('timescale')}")
    blocker = record.get("blocker")
    if blocker:
        lines.append(f"  {blocker}: {record.get('reason')}")
    measured = record.get("measured")
    if isinstance(measured, Mapping):
        lines.append(
            f"  {measured.get('frames')} conformers of {measured.get('residues')} residues"
            f" (chain {measured.get('chain')})"
        )
        lines.append(f"  fluctuation: mean {measured.get('rmsf_mean_a')} A, max {measured.get('rmsf_max_a')} A")
        lines.append(f"  drift from the design: {measured.get('drift_a')} A")
        lines.append(f"  ensemble spread: {measured.get('spread_a')} A")
        lines.append(
            f"  radius of gyration: {measured.get('radius_of_gyration_a')} A"
            f" (spread {measured.get('radius_of_gyration_spread_a')} A)"
        )
    mobile = record.get("mobile_residues")
    if isinstance(mobile, list) and mobile:
        shown = ", ".join(str(position) for position in mobile[:24])
        more = "" if len(mobile) <= 24 else f" and {len(mobile) - 24} more"
        lines.append(f"  mobile residues: {shown}{more}")
    return lines


def summary_line(record: Mapping[str, object]) -> str:
    """One sentence: the call, what it was over, and the reason when there is one."""
    state = str(record.get("state", UNTESTED))
    word = {HIT: "a hit", CLEAR: "clear", UNTESTED: "undecided"}.get(state, state)
    measured = record.get("measured")
    over = ""
    if isinstance(measured, Mapping):
        frames = measured.get("frames")
        over = f" over {frames} conformer{'' if frames == 1 else 's'}"
    said = f"{ANALYST_LABEL}: {word}{over}."
    reason = str(record.get("reason") or "")
    return f"{said} {reason}".strip()
